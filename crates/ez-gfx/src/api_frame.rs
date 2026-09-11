include!("api_frame/buffers.rs");

#[derive(Clone)]
enum DraftBinding {
    Buffer(Rc<BufferInner>),
    Counter(Rc<BufferInner>),
}

mod bindable_buffer {
    pub trait Sealed {}
}

/// Buffer resource accepted by [`Frame::bind_buffer`].
pub trait BindableBuffer: bindable_buffer::Sealed {
    #[doc(hidden)]
    fn bind_to_frame(&self, frame: &mut Frame, name: CompactString) -> Result<()>;
}

impl<T: bytemuck::Pod> bindable_buffer::Sealed for Buffer<T> {}
impl<T: bytemuck::Pod> BindableBuffer for Buffer<T> {
    fn bind_to_frame(&self, frame: &mut Frame, name: CompactString) -> Result<()> {
        frame.bind_draft(name, DraftBinding::Buffer(Rc::clone(&self.inner)))
    }
}

impl<T: bytemuck::Pod> bindable_buffer::Sealed for ValueBuffer<T> {}
impl<T: bytemuck::Pod> BindableBuffer for ValueBuffer<T> {
    fn bind_to_frame(&self, frame: &mut Frame, name: CompactString) -> Result<()> {
        frame.ensure_context(&self.context)?;
        frame.bind_draft(name, DraftBinding::Buffer(Rc::clone(&self.inner)))
    }
}

impl<T: bytemuck::Pod> bindable_buffer::Sealed for CounterBuffer<T> {}
impl<T: bytemuck::Pod> BindableBuffer for CounterBuffer<T> {
    fn bind_to_frame(&self, frame: &mut Frame, name: CompactString) -> Result<()> {
        frame.bind_draft(name, DraftBinding::Counter(Rc::clone(&self.inner)))
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum FrameTarget {
    Unconfigured,
    Surface,
    RenderTarget,
}
enum PendingReadback {
    RequestedGraph(Rc<ReadbackInner>),
    AnonymousGraph,
    RequestedPresentation(Rc<ReadbackInner>),
}

const fn should_fetch_frame_readbacks(pending: usize, implicit: bool) -> bool {
    pending != 0 || implicit
}

type FrameTransients = SmallVec<[TransientInner; 4]>;
type FrameRetained = SmallVec<[Rc<dyn Any>; 4]>;
type FrameBindings = SmallVec<[(CompactString, DraftBinding); 4]>;

/// Safe-facade frame storage is reusable, but one pathological frame must not
/// pin arbitrary capacity for the context lifetime.
const MAX_FACADE_FRAME_SCRATCH_BYTES: usize = 1024 * 1024;

#[derive(Default)]
struct FacadeFrameScratch {
    retained: FrameRetained,
    transients: FrameTransients,
    readbacks: Vec<PendingReadback>,
    bindings: FrameBindings,
    raw_bindings: Vec<RawBinding>,
}

impl FacadeFrameScratch {
    fn retained_bytes(&self) -> usize {
        self.retained
            .capacity()
            .saturating_mul(core::mem::size_of::<Rc<dyn Any>>())
            .saturating_add(
                self.transients
                    .capacity()
                    .saturating_mul(core::mem::size_of::<TransientInner>()),
            )
            .saturating_add(
                self.readbacks
                    .capacity()
                    .saturating_mul(core::mem::size_of::<PendingReadback>()),
            )
            .saturating_add(
                self.bindings
                    .capacity()
                    .saturating_mul(core::mem::size_of::<(CompactString, DraftBinding)>()),
            )
            .saturating_add(
                self.raw_bindings
                    .capacity()
                    .saturating_mul(core::mem::size_of::<RawBinding>()),
            )
            .saturating_add(self.raw_bindings.iter().fold(0_usize, |bytes, binding| {
                bytes.saturating_add(binding.name.capacity())
            }))
    }

    fn clear_owned_values(&mut self) {
        self.retained.clear();
        self.transients.clear();
        self.readbacks.clear();
        self.bindings.clear();
        for binding in &mut self.raw_bindings {
            binding.name.clear();
        }
    }

    fn can_retain(&self) -> bool {
        self.retained_bytes() <= MAX_FACADE_FRAME_SCRATCH_BYTES
    }
}

/// One explicit recording transaction.
///
/// [`finish`](Self::finish) consumes and submits exactly once. Dropping an
/// unfinished frame aborts and rolls back because `Drop` cannot return errors.
pub struct Frame {
    context: Rc<ContextInner>,
    target: FrameTarget,
    surface: Option<Rc<SurfaceInner>>,
    retained: FrameRetained,
    transients: FrameTransients,
    poison: Option<Error>,
    terminal: bool,
    readbacks: Vec<PendingReadback>,
    bindings: FrameBindings,
    raw_bindings: Vec<RawBinding>,
}

impl Frame {
    fn ensure_context(&mut self, context: &Rc<ContextInner>) -> Result<()> {
        if Rc::ptr_eq(&self.context, context) {
            Ok(())
        } else {
            self.fail(Error::InvalidContext)
        }
    }

    fn fail<T>(&mut self, error: Error) -> Result<T> {
        self.poison.get_or_insert(error);
        Err(error)
    }

    fn record(&mut self, operation: impl FnOnce(ContextHandle) -> Result<()>) -> Result<()> {
        if let Some(error) = self.poison {
            return Err(error);
        }
        match operation(self.context.handle) {
            Ok(()) => Ok(()),
            Err(error) => self.fail(error),
        }
    }

    fn retain<T: Any>(&mut self, resource: &Rc<T>) {
        let retained: Rc<dyn Any> = resource.clone();
        self.retained.push(retained);
    }

    fn consume_transients(&self) {
        for transient in &self.transients {
            transient.state.set(TransientState::Consumed);
            transient.buffer.usage.set(BufferUse::Consumed);
        }
    }

    fn release_aborted_transients(&self) {
        for transient in &self.transients {
            if transient.state.replace(TransientState::Consumed) == TransientState::Live {
                match transient.handle {
                    TransientHandle::Buffer(handle) => {
                        state::release_buffer(self.context.handle, handle);
                    }
                    TransientHandle::Counter(handle) => {
                        state::release_counter(self.context.handle, handle);
                    }
                }
            }
            transient.buffer.usage.set(BufferUse::Consumed);
        }
    }

    fn materialize_buffer_state(
        context: &Rc<ContextInner>,
        transients: &mut FrameTransients,
        inner: &Rc<BufferInner>,
    ) -> Result<BufferHandle> {
        let Some(owner) = inner.context.upgrade() else {
            return Err(Error::InvalidContext);
        };
        if !Rc::ptr_eq(context, &owner) {
            return Err(Error::InvalidContext);
        }
        if let Some(handle) = transients
            .iter()
            .find_map(|transient| Rc::ptr_eq(&transient.buffer, inner).then_some(&transient.handle))
        {
            return match handle {
                TransientHandle::Buffer(handle) => Ok(*handle),
                TransientHandle::Counter(_) => Err(Error::InvalidContext),
            };
        }
        if inner.usage.get() != BufferUse::Available {
            return Err(Error::NotReady);
        }
        let handle =
            state::acquire_buffer_sized(context.handle, inner.element_size, inner.element_count)?;
        if let Err(error) = state::write_buffer_bytes(
            context.handle,
            handle,
            inner.element_size,
            &inner.bytes.borrow(),
        ) {
            state::release_buffer(context.handle, handle);
            return Err(error);
        }
        inner.usage.set(BufferUse::Claimed);
        transients.push(TransientInner {
            context: Rc::clone(context),
            buffer: Rc::clone(inner),
            handle: TransientHandle::Buffer(handle),
            state: Cell::new(TransientState::Live),
        });
        Ok(handle)
    }

    fn materialize_counter_state(
        context: &Rc<ContextInner>,
        transients: &mut FrameTransients,
        inner: &Rc<BufferInner>,
    ) -> Result<CounterBufferHandle> {
        let Some(owner) = inner.context.upgrade() else {
            return Err(Error::InvalidContext);
        };
        if !Rc::ptr_eq(context, &owner) {
            return Err(Error::InvalidContext);
        }
        if inner.element_size as usize
            != core::mem::size_of::<ez_gfx_runtime::indirect::DrawIndexedCommand>()
        {
            return Err(Error::InvalidArgument);
        }
        if let Some(handle) = transients
            .iter()
            .find_map(|transient| Rc::ptr_eq(&transient.buffer, inner).then_some(&transient.handle))
        {
            return match handle {
                TransientHandle::Counter(handle) => Ok(*handle),
                TransientHandle::Buffer(_) => Err(Error::InvalidContext),
            };
        }
        if inner.usage.get() != BufferUse::Available {
            return Err(Error::NotReady);
        }
        let handle = state::acquire_counter(context.handle, inner.element_count)?;
        if let Err(error) = state::write_counter_bytes(
            context.handle,
            handle,
            &inner.bytes.borrow(),
            inner.published_count.get(),
        ) {
            state::release_counter(context.handle, handle);
            return Err(error);
        }
        inner.usage.set(BufferUse::Claimed);
        transients.push(TransientInner {
            context: Rc::clone(context),
            buffer: Rc::clone(inner),
            handle: TransientHandle::Counter(handle),
            state: Cell::new(TransientState::Live),
        });
        Ok(handle)
    }

    fn materialize_counter(&mut self, inner: &Rc<BufferInner>) -> Result<CounterBufferHandle> {
        match Self::materialize_counter_state(&self.context, &mut self.transients, inner) {
            Ok(handle) => Ok(handle),
            Err(error) => self.fail(error),
        }
    }

    fn prepare_raw_bindings(&mut self) -> Result<()> {
        let result = {
            let context = &self.context;
            let bindings = &self.bindings;
            let raw_bindings = &mut self.raw_bindings;
            let transients = &mut self.transients;

            (|| {
                for (index, (name, binding)) in bindings.iter().enumerate() {
                    let resource = match binding {
                        DraftBinding::Buffer(inner) => {
                            Self::materialize_buffer_state(context, transients, inner)
                                .map(ResourceIdentity::Buffer)?
                        }
                        DraftBinding::Counter(inner) => {
                            Self::materialize_counter_state(context, transients, inner)
                                .map(ResourceIdentity::Counter)?
                        }
                    };
                    if let Some(raw) = raw_bindings.get_mut(index) {
                        raw.name.clear();
                        raw.name.push_str(name.as_str());
                        raw.resource = resource;
                    } else {
                        raw_bindings.push(RawBinding {
                            name: name.to_string(),
                            resource,
                        });
                    }
                }
                raw_bindings.truncate(bindings.len());
                Ok(())
            })()
        };

        match result {
            Ok(()) => Ok(()),
            Err(error) => self.fail(error),
        }
    }

    /// Adds or replaces one named buffer in the frame binding set.
    ///
    /// # Errors
    /// Returns [`Error`] when the name, frame, resource state, or ownership is invalid.
    pub fn bind_buffer<B: BindableBuffer + ?Sized>(
        &mut self,
        name: impl Into<CompactString>,
        buffer: &B,
    ) -> Result<()> {
        buffer.bind_to_frame(self, name.into())
    }

    fn bind_draft(&mut self, name: CompactString, binding: DraftBinding) -> Result<()> {
        if name.is_empty() || name.len() > 255 || name.as_bytes().contains(&0) {
            return self.fail(Error::InvalidArgument);
        }
        let inner = match &binding {
            DraftBinding::Buffer(inner) | DraftBinding::Counter(inner) => inner,
        };
        let Some(owner) = inner.context.upgrade() else {
            return self.fail(Error::InvalidContext);
        };
        self.ensure_context(&owner)?;
        let valid_state = match inner.usage.get() {
            BufferUse::Available => true,
            BufferUse::Claimed => self.transients.iter().any(|transient| {
                transient.state.get() == TransientState::Live
                    && Rc::ptr_eq(&transient.buffer, inner)
            }),
            BufferUse::Consumed => false,
        };
        if !valid_state {
            return self.fail(Error::NotReady);
        }
        if let Some((_, current)) = self
            .bindings
            .iter_mut()
            .find(|(key, _)| key.as_str() == name.as_str())
        {
            *current = binding;
        } else {
            self.bindings.push((name, binding));
        }
        Ok(())
    }

    /// Executes an indexed graphics operation with exact vertex and fragment entry points.
    ///
    /// # Errors
    /// Returns [`Error`] when resources, stage ownership, bindings, or recording state are invalid.
    pub fn execute_graphics(
        &mut self,
        vertex_shader: &VertexShader,
        fragment_shader: &FragmentShader,
        counter: &CounterBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand>,
        state_desc: ez_gfx_hal::DynamicPipelineState,
    ) -> Result<()> {
        self.ensure_context(&vertex_shader.inner.context)?;
        self.ensure_context(&fragment_shader.inner.context)?;
        let counter_handle = self.materialize_counter(&counter.inner)?;
        self.prepare_raw_bindings()?;
        self.retain(&vertex_shader.inner);
        self.retain(&fragment_shader.inner);
        if let Some(error) = self.poison {
            return Err(error);
        }
        match state::execute_graphics(
            self.context.handle,
            vertex_shader.inner.handle,
            fragment_shader.inner.handle,
            counter_handle,
            &self.raw_bindings,
            state_desc,
        ) {
            Ok(()) => Ok(()),
            Err(error) => self.fail(error),
        }
    }

    /// Executes a compute dispatch with one exact compute entry point.
    ///
    /// # Errors
    /// Returns [`Error`] when resources, bindings, dispatch, stage ownership, or recording state are invalid.
    pub fn execute_compute(&mut self, shader: &ComputeShader, groups: [u32; 3]) -> Result<()> {
        self.ensure_context(&shader.inner.context)?;
        self.prepare_raw_bindings()?;
        self.retain(&shader.inner);
        if let Some(error) = self.poison {
            return Err(error);
        }
        match state::execute_compute(
            self.context.handle,
            shader.inner.handle,
            groups,
            &self.raw_bindings,
        ) {
            Ok(()) => Ok(()),
            Err(error) => self.fail(error),
        }
    }

    /// Enqueues a texture readback and retains the texture until completion.
    ///
    /// Texture bytes are delivered as [`Event::Snapshot`] because textures do
    /// not expose a stable logical extent through the safe facade.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame or texture is stale, foreign, or not ready.
    pub fn enqueue_texture_readback(&mut self, texture: &Texture) -> Result<()> {
        self.ensure_context(&texture.inner.context)?;
        self.retain(&texture.inner);
        let result =
            self.record(|context| state::frame_enqueue_readback(context, texture.inner.handle));
        if result.is_ok() {
            self.readbacks.push(PendingReadback::AnonymousGraph);
        }
        result
    }

    fn prepare_target_readback(&mut self, target: &RenderTarget) -> Result<Readback> {
        self.ensure_context(&target.inner.context)?;
        let (width, height) = match target.extent() {
            Ok(extent) => extent,
            Err(error) => return self.fail(error),
        };
        let generation = self.context.next_readback.get();
        let Some(next) = generation.checked_add(1) else {
            return self.fail(Error::InvalidArgument);
        };
        let presented = match &target.inner.backing {
            RenderTargetBacking::Managed(_) => {
                let handle = match target.inner.managed_handle() {
                    Ok(handle) => handle,
                    Err(error) => return self.fail(error),
                };
                self.record(|context| {
                    state::frame_enqueue_render_target_readback(context, handle)
                })?;
                false
            }
            // Presented images have no stable raw target handle; request capture for this frame
            // without enabling the surface's persistent snapshot cache.
            RenderTargetBacking::Surface { .. } => {
                let Some(surface) = target.surface_lease.as_ref() else {
                    return self.fail(Error::InvalidContext);
                };
                self.record(|context| {
                    state::frame_request_presented_readback(context, surface.handle)
                })?;
                true
            }
        };
        self.context.next_readback.set(next);
        let request = Rc::new(ReadbackInner {
            _context: Rc::clone(&self.context),
            _target: Rc::clone(&target.inner),
            id: ReadbackId {
                owner: self.context.handle.into_raw(),
                generation,
            },
            width,
            height,
            state: Cell::new(ReadbackState::Queued),
        });
        self.readbacks.push(if presented {
            PendingReadback::RequestedPresentation(Rc::clone(&request))
        } else {
            PendingReadback::RequestedGraph(Rc::clone(&request))
        });
        Ok(Readback { inner: request })
    }

    /// Configures a cached named render target for this frame.
    ///
    /// A format or extent change atomically replaces the cached native image;
    /// unchanged configurations reuse it.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame is already configured or target creation fails.
    pub fn configure_render_target(
        &mut self,
        name: impl Into<String>,
        size: [u32; 2],
        format: ez_gfx_runtime::target::Format,
    ) -> Result<RenderTarget> {
        // A surface-backed frame and a second target attachment are never interchangeable.
        if self.target != FrameTarget::Unconfigured || self.surface.is_some() {
            return self.fail(Error::NotReady);
        }
        let name = name.into();
        let [width, height] = size;
        // Zero cannot name a physical image and must not evict an existing cached target.
        if width == 0 || height == 0 {
            return self.fail(Error::InvalidArgument);
        }
        // Cache identity is the stable name; size or format changes replace only its image.
        let cached = self
            .context
            .render_targets
            .borrow()
            .get(&name)
            .copied()
            .filter(|target| target.extent == (width, height) && target.format == format);
        let handle = if let Some(target) = cached {
            target.handle
        } else {
            let Ok(declaration) = ez_gfx_runtime::target::TargetDeclaration::new(
                name.clone(),
                ez_gfx_runtime::target::TargetUsage::Color,
                1.0,
                1,
                vec![format],
                ez_gfx_runtime::target::ClearValue::Color([0.1, 0.1, 0.1, 1.0]),
                true,
            ) else {
                return self.fail(Error::InvalidArgument);
            };
            let handle =
                match state::create_render_target(self.context.handle, &declaration, width, height)
                {
                    Ok(handle) => handle,
                    Err(error) => return self.fail(error),
                };
            // Publish the replacement before retiring the old native image.
            let previous = self.context.render_targets.borrow_mut().insert(
                name.clone(),
                CachedRenderTarget {
                    handle,
                    format,
                    extent: (width, height),
                },
            );
            if let Some(previous) = previous {
                state::destroy_render_target(self.context.handle, previous.handle);
            }
            handle
        };
        if let Err(error) = state::configure_render_target(self.context.handle, handle) {
            return self.fail(error);
        }
        self.target = FrameTarget::RenderTarget;
        let inner = Rc::new(RenderTargetInner {
            context: Rc::clone(&self.context),
            backing: RenderTargetBacking::Managed(name),
        });
        self.retain(&inner);
        Ok(RenderTarget {
            inner,
            surface_lease: None,
        })
    }

    /// Attaches the frame's surface swapchain and returns its logical render target.
    ///
    /// The returned target retains the surface and exposes the configured extent
    /// and format without exposing a backend swapchain image.
    /// Unsupported valid presentation modes use the deterministic fallback exposed by
    /// [`Surface::resolve_presentation_mode`].
    ///
    /// # Errors
    /// Returns [`Error`] when the frame is already configured, the extent is zero,
    /// or surface resize/acquisition fails.
    pub fn configure_swapchain(
        &mut self,
        size: [u32; 2],
        format: ez_gfx_runtime::target::Format,
        presentation_mode: PresentationMode,
    ) -> Result<RenderTarget> {
        // A second attachment would make submission/presentation ownership ambiguous.
        if self.target != FrameTarget::Unconfigured {
            return self.fail(Error::NotReady);
        }
        let Some(surface) = self.surface.as_ref().map(Rc::clone) else {
            return self.fail(Error::InvalidContext);
        };
        let [width, height] = size;
        // Minimized windows never create or resize a swapchain image.
        if width == 0 || height == 0 {
            return self.fail(Error::InvalidArgument);
        }
        let current = match state::surface_extent(self.context.handle, surface.handle) {
            Ok(extent) => extent,
            Err(error) => return self.fail(error),
        };
        if current != (width, height)
            && let Err(error) =
                state::resize_surface(self.context.handle, surface.handle, width, height)
        {
            return self.fail(error);
        }
        if let Err(error) =
            state::configure_surface(self.context.handle, surface.handle, presentation_mode)
        {
            return self.fail(error);
        }
        self.target = FrameTarget::Surface;
        let cached = {
            let target = surface.target.borrow();
            target
                .as_ref()
                .filter(|target| {
                    matches!(
                        target.backing,
                        RenderTargetBacking::Surface {
                            format: cached_format,
                            extent,
                        } if cached_format == format && extent == (width, height)
                    )
                })
                .map(Rc::clone)
        };
        let inner = cached.unwrap_or_else(|| {
            let target = Rc::new(RenderTargetInner {
                context: Rc::clone(&self.context),
                backing: RenderTargetBacking::Surface {
                    format,
                    extent: (width, height),
                },
            });
            *surface.target.borrow_mut() = Some(Rc::clone(&target));
            target
        });
        Ok(RenderTarget {
            inner,
            surface_lease: Some(surface),
        })
    }

    /// Submits and, for surface frames, presents this transaction.
    ///
    /// A prior recording error poisons the transaction: this method aborts and
    /// returns that exact error without attempting submission.
    ///
    /// # Errors
    /// Returns the exact first recording, submission, presentation, readback, or callback error.
    pub fn finish(mut self) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.context),
            owner: false,
        };
        let result = if let Some(error) = self.poison {
            let _ = state::frame_abort(self.context.handle);
            self.release_aborted_transients();
            Err(error)
        } else if self.target == FrameTarget::Unconfigured {
            let _ = state::frame_abort(self.context.handle);
            self.release_aborted_transients();
            Err(Error::NotReady)
        } else {
            match state::frame_submit(self.context.handle) {
                Ok(()) => {
                    self.consume_transients();
                    match self.target {
                        FrameTarget::Surface => state::present(self.context.handle),
                        FrameTarget::RenderTarget => Ok(()),
                        FrameTarget::Unconfigured => Err(Error::NotReady),
                    }
                }
                Err(error) => {
                    self.release_aborted_transients();
                    Err(error)
                }
            }
        };
        let implicit_readback = self
            .surface
            .as_ref()
            .is_some_and(|surface| surface.snapshot_cache.get());
        let outputs = if result.is_ok()
            && should_fetch_frame_readbacks(self.readbacks.len(), implicit_readback)
        {
            match state::frame_readbacks(self.context.handle) {
                Ok(outputs) => Some(Ok(outputs)),
                Err(Error::NotReady) if !self.readbacks.is_empty() => {
                    Some(Err(Error::NativeFailure))
                }
                Err(Error::NotReady) => None,
                Err(error) => Some(Err(error)),
            }
        } else {
            None
        };
        if result.is_err() {
            for readback in &self.readbacks {
                match readback {
                    PendingReadback::RequestedGraph(request)
                    | PendingReadback::RequestedPresentation(request) => {
                        request.state.set(ReadbackState::Aborted);
                    }
                    PendingReadback::AnonymousGraph => {}
                }
            }
        }
        let readbacks = std::mem::take(&mut self.readbacks);
        self.terminal = true;
        drop(self);
        let result = context.complete(result);
        let mut outputs = match (result, outputs) {
            (Err(error), _) | (Ok(()), Some(Err(error))) => return Err(error),
            (Ok(()), Some(Ok(outputs))) => outputs,
            (Ok(()), None) => return Ok(()),
        };
        let graph_count = readbacks
            .iter()
            .filter(|readback| {
                matches!(
                    readback,
                    PendingReadback::RequestedGraph(_) | PendingReadback::AnonymousGraph
                )
            })
            .count();
        let has_presented = readbacks
            .iter()
            .any(|readback| matches!(readback, PendingReadback::RequestedPresentation(_)));
        let minimum_outputs = graph_count + usize::from(has_presented);
        if outputs.len() < minimum_outputs {
            for request in readbacks.iter().filter_map(|readback| match readback {
                PendingReadback::RequestedGraph(request)
                | PendingReadback::RequestedPresentation(request) => Some(request),
                PendingReadback::AnonymousGraph => None,
            }) {
                request.state.set(ReadbackState::Aborted);
            }
            return Err(Error::NativeFailure);
        }
        let mut trailing = outputs.split_off(graph_count);
        let presented_output = has_presented
            .then(|| trailing.pop().ok_or(Error::NativeFailure))
            .transpose()?;
        let mut graph_outputs = outputs.into_iter();
        for readback in &readbacks {
            let dispatched = match readback {
                PendingReadback::RequestedGraph(request) => {
                    let bytes = graph_outputs.next().ok_or(Error::NativeFailure)?;
                    request.state.set(ReadbackState::Complete);
                    context.dispatch_readback(request.id, request.width, request.height, &bytes)
                }
                PendingReadback::AnonymousGraph => {
                    let bytes = graph_outputs.next().ok_or(Error::NativeFailure)?;
                    context.dispatch_snapshot(&bytes)
                }
                PendingReadback::RequestedPresentation(request) => {
                    let bytes = presented_output.as_deref().ok_or(Error::NativeFailure)?;
                    request.state.set(ReadbackState::Complete);
                    context.dispatch_readback(request.id, request.width, request.height, bytes)
                }
            };
            if let Err(error) = dispatched {
                for request in readbacks.iter().filter_map(|pending| match pending {
                    PendingReadback::RequestedGraph(request)
                    | PendingReadback::RequestedPresentation(request)
                        if request.state.get() == ReadbackState::Queued =>
                    {
                        Some(request)
                    }
                    _ => None,
                }) {
                    request.state.set(ReadbackState::Aborted);
                }
                return Err(error);
            }
        }
        debug_assert!(graph_outputs.next().is_none());
        for bytes in trailing {
            context.dispatch_snapshot(&bytes)?;
        }
        Ok(())
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if !self.terminal {
            // Rollback precedes release so Interned(frame_serial) becomes releasable.
            let _ = state::frame_abort(self.context.handle);
            self.release_aborted_transients();
            for readback in &self.readbacks {
                match readback {
                    PendingReadback::RequestedGraph(request)
                    | PendingReadback::RequestedPresentation(request) => {
                        request.state.set(ReadbackState::Aborted);
                    }
                    PendingReadback::AnonymousGraph => {}
                }
            }
        }
        let mut recycled = FacadeFrameScratch {
            retained: std::mem::take(&mut self.retained),
            transients: std::mem::take(&mut self.transients),
            readbacks: std::mem::take(&mut self.readbacks),
            bindings: std::mem::take(&mut self.bindings),
            raw_bindings: std::mem::take(&mut self.raw_bindings),
        };
        recycled.clear_owned_values();
        if !recycled.can_retain() {
            recycled = FacadeFrameScratch::default();
        }
        *self.context.frame_scratch.borrow_mut() = recycled;
    }
}

impl Surface {
    /// Begins one explicit surface recording transaction.
    ///
    /// # Errors
    /// Returns [`Error`] when readiness or concurrent recording validation fails.
    pub fn begin_frame(&self) -> Result<Frame> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
        };
        context.check_entry()?;
        context.dispatch_events()?;
        state::frame_begin(context.raw())?;
        let target_lease: Rc<dyn Any> = self.inner.clone();
        let mut scratch = std::mem::take(&mut *context.inner.frame_scratch.borrow_mut());
        scratch.retained.push(target_lease);
        Ok(Frame {
            context: Rc::clone(&context.inner),
            target: FrameTarget::Unconfigured,
            surface: Some(Rc::clone(&self.inner)),
            retained: scratch.retained,
            transients: scratch.transients,
            poison: None,
            terminal: false,
            readbacks: scratch.readbacks,
            bindings: scratch.bindings,
            raw_bindings: scratch.raw_bindings,
        })
    }
}

impl Context {
    /// Begins one target-less frame for later named render-target configuration.
    ///
    /// # Errors
    /// Returns [`Error`] when readiness or concurrent recording validation fails.
    pub fn begin_frame(&self) -> Result<Frame> {
        self.check_entry()?;
        self.dispatch_events()?;
        state::frame_begin(self.raw())?;
        let scratch = std::mem::take(&mut *self.inner.frame_scratch.borrow_mut());
        Ok(Frame {
            context: Rc::clone(&self.inner),
            target: FrameTarget::Unconfigured,
            surface: None,
            retained: scratch.retained,
            transients: scratch.transients,
            poison: None,
            terminal: false,
            readbacks: scratch.readbacks,
            bindings: scratch.bindings,
            raw_bindings: scratch.raw_bindings,
        })
    }
}
impl Surface {
    /// Returns the presentation modes available for this initialized surface.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale, its device is uninitialized, or the backend query fails.
    pub fn presentation_modes(&self) -> Result<PresentationModes> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
        };
        context.check_entry()?;
        context.complete(state::presentation_modes(
            self.inner.context.handle,
            self.inner.handle,
        ))
    }

    /// Resolves a requested presentation mode using the public deterministic fallback order.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface query fails or no supported fallback is available.
    pub fn resolve_presentation_mode(
        &self,
        requested: PresentationMode,
    ) -> Result<PresentationMode> {
        self.presentation_modes()?
            .resolve(requested)
            .ok_or(Error::Unsupported)
    }
}

#[cfg(test)]
mod inline_storage_tests {
    use super::{
        FacadeFrameScratch, FrameTransients, MAX_FACADE_BUFFER_POOL_BYTES,
        MAX_FACADE_BUFFER_POOL_ENTRIES, MAX_FACADE_FRAME_SCRATCH_BYTES, facade_pool_can_retain,
        should_fetch_frame_readbacks,
    };
    use std::{any::Any, rc::Rc};

    #[test]
    fn common_transient_storage_remains_bounded() {
        assert!(
            core::mem::size_of::<FrameTransients>() <= 256,
            "four inline transient leases must keep Frame reasonably compact",
        );
    }

    #[test]
    fn readback_fetch_is_skipped_only_for_uncached_frames_without_requests() {
        assert!(!should_fetch_frame_readbacks(0, false));
        assert!(should_fetch_frame_readbacks(1, false));
        assert!(should_fetch_frame_readbacks(0, true));
        assert!(should_fetch_frame_readbacks(1, true));
    }

    #[test]
    fn facade_pool_rejects_entry_and_byte_overflow() {
        for (entries, retained, candidate, expected) in [
            (0, 0, 1, true),
            (MAX_FACADE_BUFFER_POOL_ENTRIES, 0, 1, false),
            (0, MAX_FACADE_BUFFER_POOL_BYTES, 1, false),
            (0, usize::MAX, usize::MAX, false),
        ] {
            assert_eq!(
                facade_pool_can_retain(entries, retained, candidate),
                expected
            );
        }
    }

    #[test]
    fn warmed_spill_cardinality_reuses_backing_and_releases_owners() {
        let owners = (0..8).map(Rc::new).collect::<Vec<_>>();
        let mut scratch = FacadeFrameScratch::default();
        scratch
            .retained
            .extend(owners.iter().cloned().map(|owner| owner as Rc<dyn Any>));
        let warmed_capacity = scratch.retained.capacity();
        let warmed_backing = scratch.retained.as_ptr();

        scratch.clear_owned_values();
        assert!(owners.iter().all(|owner| Rc::strong_count(owner) == 1));
        scratch
            .retained
            .extend(owners.iter().cloned().map(|owner| owner as Rc<dyn Any>));

        assert!(warmed_capacity > 4);
        assert_eq!(scratch.retained.capacity(), warmed_capacity);
        assert_eq!(scratch.retained.as_ptr(), warmed_backing);
    }

    #[test]
    fn facade_frame_scratch_has_a_hard_byte_bound() {
        let mut scratch = FacadeFrameScratch::default();
        let entries =
            MAX_FACADE_FRAME_SCRATCH_BYTES / core::mem::size_of::<super::PendingReadback>() + 1;
        scratch.readbacks.reserve_exact(entries);

        assert!(!scratch.can_retain());
    }
}
