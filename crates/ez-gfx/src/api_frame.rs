#[derive(Clone, Copy, Eq, PartialEq)]
enum BufferUse {
    Available,
    Claimed,
    Consumed,
}

mod buffer_data {
    pub trait Sealed {}

    impl<T: bytemuck::Pod> Sealed for super::BufferSource<'_, T> {}
    impl<T: bytemuck::Pod> Sealed for &[T] {}
    impl<T: bytemuck::Pod> Sealed for &Vec<T> {}
}

/// Explicit source for one POD buffer element.
///
/// Slices and vectors can be passed directly to context acquisition helpers.
/// Arrays use `.as_slice()` so they cannot be mistaken for one array-valued element.
pub struct BufferSource<'a, T: bytemuck::Pod> {
    value: &'a T,
}

impl<'a, T: bytemuck::Pod> BufferSource<'a, T> {
    /// Borrows one element without allocating an intermediate collection.
    pub const fn one(value: &'a T) -> Self {
        Self { value }
    }
}

/// POD input accepted by context buffer acquisition helpers.
pub trait BufferData: buffer_data::Sealed {
    /// Element stored by the acquired buffer.
    type Element: bytemuck::Pod;

    #[doc(hidden)]
    fn as_buffer_slice(&self) -> &[Self::Element];
}

impl<T: bytemuck::Pod> BufferData for BufferSource<'_, T> {
    type Element = T;

    fn as_buffer_slice(&self) -> &[T] {
        core::slice::from_ref(self.value)
    }
}

impl<T: bytemuck::Pod> BufferData for &[T] {
    type Element = T;

    fn as_buffer_slice(&self) -> &[T] {
        self
    }
}

impl<T: bytemuck::Pod> BufferData for &Vec<T> {
    type Element = T;

    fn as_buffer_slice(&self) -> &[T] {
        self
    }
}

fn buffer_data_slice<D: BufferData + ?Sized>(data: &D) -> &[D::Element] {
    data.as_buffer_slice()
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum TransientState {
    Live,
    Consumed,
}

enum TransientHandle {
    Structured(StructuredBufferHandle),
    Indirect(IndirectBufferHandle),
}

struct TransientInner {
    context: Rc<ContextInner>,
    buffer: Rc<BufferInner>,
    handle: TransientHandle,
    state: Cell<TransientState>,
}

impl Drop for TransientInner {
    fn drop(&mut self) {
        // The frame owns the sole native lease after successful materialization.
        if self.state.get() != TransientState::Live {
            return;
        }
        // Successful completion consumes raw transients. Only live, abandoned
        // transactions reach this release path after frame rollback.
        match self.handle {
            TransientHandle::Structured(handle) => {
                state::release_structured(self.context.handle, handle);
            }
            TransientHandle::Indirect(handle) => {
                state::release_indirect(self.context.handle, handle);
            }
        }
    }
}

struct BufferInner {
    context: Rc<ContextInner>,
    element_size: u32,
    element_count: u32,
    bytes: RefCell<Vec<u8>>,
    published_count: Cell<u32>,
    usage: Cell<BufferUse>,
}

impl BufferInner {
    fn write<T: bytemuck::Pod>(&self, start_index: usize, values: &[T]) -> Result<()> {
        // Claimed and consumed one-frame values cannot be rewritten.
        if self.usage.get() != BufferUse::Available {
            return Err(Error::NotReady);
        }
        if core::mem::size_of::<T>() != self.element_size as usize {
            return Err(Error::InvalidArgument);
        }
        let end = start_index
            .checked_add(values.len())
            .filter(|end| *end <= self.element_count as usize)
            .ok_or(Error::InvalidArgument)?;
        let start_byte = start_index
            .checked_mul(self.element_size as usize)
            .ok_or(Error::InvalidArgument)?;
        let end_byte = end
            .checked_mul(self.element_size as usize)
            .ok_or(Error::InvalidArgument)?;
        self.bytes.borrow_mut()[start_byte..end_byte].copy_from_slice(bytemuck::cast_slice(values));
        Ok(())
    }
}

/// Context-owned typed buffer uploaded when a frame first binds it.
pub struct Buffer<T: bytemuck::Pod> {
    inner: Rc<BufferInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    /// Replaces a typed element range.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is dispatching a callback or the range is invalid.
    pub fn write(&self, start_index: usize, values: &[T]) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        self.inner.write(start_index, values)
    }
}

/// Context-acquired one-frame buffer carrying a separately publishable visible count.
pub struct CounterBuffer<T: bytemuck::Pod> {
    inner: Rc<BufferInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> CounterBuffer<T> {
    /// Replaces a typed element range.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is dispatching a callback or the range is invalid.
    pub fn write(&self, start_index: usize, values: &[T]) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        self.inner.write(start_index, values)
    }

    /// Publishes the visible element count.
    ///
    /// # Errors
    /// Returns [`Error`] during callback dispatch or when the count exceeds capacity.
    pub fn publish_count(&self, count: u32) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        // Publication after a frame claim would diverge from the staged contents.
        if self.inner.usage.get() != BufferUse::Available {
            return Err(Error::NotReady);
        }
        if count > self.inner.element_count {
            return Err(Error::InvalidArgument);
        }
        self.inner.published_count.set(count);
        Ok(())
    }
}

impl Context {
    fn allocate_buffer<T: bytemuck::Pod>(
        &self,
        element_count: usize,
        initial: Option<&[T]>,
        published_count: u32,
    ) -> Result<Rc<BufferInner>> {
        // Empty and zero-sized buffers cannot produce valid native bindings.
        self.check_entry()?;
        let element_size =
            u32::try_from(core::mem::size_of::<T>()).map_err(|_| Error::InvalidArgument)?;
        let element_count = u32::try_from(element_count).map_err(|_| Error::InvalidArgument)?;
        let byte_count = (element_size as usize)
            .checked_mul(element_count as usize)
            .filter(|size| element_size != 0 && element_count != 0 && *size <= 16 * 1024 * 1024)
            .ok_or(Error::InvalidArgument)?;
        let bytes = match initial {
            Some(values) if values.len() == element_count as usize => {
                bytemuck::cast_slice(values).to_vec()
            }
            Some(_) => return Err(Error::InvalidArgument),
            None => vec![0; byte_count],
        };
        Ok(Rc::new(BufferInner {
            context: Rc::clone(&self.inner),
            element_size,
            element_count,
            bytes: RefCell::new(bytes),
            published_count: Cell::new(published_count),
            usage: Cell::new(BufferUse::Available),
        }))
    }

    /// Acquires a one-frame typed buffer by element count.
    ///
    /// # Errors
    /// Returns [`Error`] when the count, type size, or context is invalid.
    pub fn acquire_buffer<T: bytemuck::Pod>(&self, element_count: usize) -> Result<Buffer<T>> {
        Ok(Buffer {
            inner: self.allocate_buffer::<T>(element_count, None, 0)?,
            marker: PhantomData,
        })
    }

    /// Acquires a correctly-sized one-frame typed buffer initialized from POD data.
    ///
    /// # Errors
    /// Returns [`Error`] when the input is empty, oversized, zero-sized, or the context is invalid.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the source type carries element inference and prevents scalar/array ambiguity"
    )]
    pub fn acquire_buffer_from<D: BufferData>(&self, data: D) -> Result<Buffer<D::Element>> {
        let values = buffer_data_slice(&data);
        Ok(Buffer {
            inner: self.allocate_buffer(values.len(), Some(values), 0)?,
            marker: PhantomData,
        })
    }

    /// Acquires a one-frame typed counter buffer by element count.
    ///
    /// # Errors
    /// Returns [`Error`] when the count, type size, or context is invalid.
    pub fn acquire_counter_buffer<T: bytemuck::Pod>(
        &self,
        element_count: usize,
    ) -> Result<CounterBuffer<T>> {
        Ok(CounterBuffer {
            inner: self.allocate_buffer::<T>(element_count, None, 0)?,
            marker: PhantomData,
        })
    }

    /// Acquires a correctly-sized one-frame counter buffer initialized from POD data.
    ///
    /// The initialized element count is published automatically.
    ///
    /// # Errors
    /// Returns [`Error`] when the input is empty, oversized, zero-sized, or the context is invalid.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the source type carries element inference and prevents scalar/array ambiguity"
    )]
    pub fn acquire_counter_buffer_from<D: BufferData>(
        &self,
        data: D,
    ) -> Result<CounterBuffer<D::Element>> {
        let values = buffer_data_slice(&data);
        let published_count = u32::try_from(values.len()).map_err(|_| Error::InvalidArgument)?;
        Ok(CounterBuffer {
            inner: self.allocate_buffer(values.len(), Some(values), published_count)?,
            marker: PhantomData,
        })
    }
}

enum BindingResource<'a> {
    Structured(&'a Rc<BufferInner>),
    Indirect(&'a Rc<BufferInner>),
    RenderTarget(&'a Rc<RenderTargetInner>),
}

/// One named safe resource binding retained by a recorded frame.
pub struct Binding<'a> {
    name: String,
    resource: BindingResource<'a>,
}

impl<'a> Binding<'a> {
    /// Binds a typed buffer.
    pub fn buffer<T: bytemuck::Pod>(name: impl Into<String>, buffer: &'a Buffer<T>) -> Self {
        Self {
            name: name.into(),
            resource: BindingResource::Structured(&buffer.inner),
        }
    }

    /// Binds a counter buffer.
    pub fn counter_buffer<T: bytemuck::Pod>(
        name: impl Into<String>,
        buffer: &'a CounterBuffer<T>,
    ) -> Self {
        Self {
            name: name.into(),
            resource: BindingResource::Indirect(&buffer.inner),
        }
    }

    /// Binds a managed render target.
    pub fn render_target(name: impl Into<String>, target: &'a RenderTarget) -> Self {
        Self {
            name: name.into(),
            resource: BindingResource::RenderTarget(&target.inner),
        }
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


/// One explicit recording transaction.
///
/// [`finish`](Self::finish) consumes and submits exactly once. Dropping an
/// unfinished frame aborts and rolls back because `Drop` cannot return errors.
pub struct Frame {
    context: Rc<ContextInner>,
    target: FrameTarget,
    surface: Option<Rc<SurfaceInner>>,
    retained: Vec<Rc<dyn Any>>,
    transients: Vec<Rc<TransientInner>>,
    poison: Option<Error>,
    terminal: bool,
    readbacks: Vec<PendingReadback>,
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
                    TransientHandle::Structured(handle) => {
                        state::release_structured(self.context.handle, handle);
                    }
                    TransientHandle::Indirect(handle) => {
                        state::release_indirect(self.context.handle, handle);
                    }
                }
            }
            transient.buffer.usage.set(BufferUse::Consumed);
        }
    }

    fn materialize_structured(&mut self, inner: &Rc<BufferInner>) -> Result<StructuredBufferHandle> {
        self.ensure_context(&inner.context)?;
        if let Some(handle) = self.transients.iter().find_map(|transient| {
            Rc::ptr_eq(&transient.buffer, inner).then_some(&transient.handle)
        }) {
            return match handle {
                TransientHandle::Structured(handle) => Ok(*handle),
                TransientHandle::Indirect(_) => self.fail(Error::InvalidContext),
            };
        }
        if inner.usage.get() != BufferUse::Available {
            return self.fail(Error::NotReady);
        }
        let handle = match state::acquire_structured_sized(
            self.context.handle,
            inner.element_size,
            inner.element_count,
        ) {
            Ok(handle) => handle,
            Err(error) => return self.fail(error),
        };
        if let Err(error) = state::write_structured_bytes(
            self.context.handle,
            handle,
            inner.element_size,
            &inner.bytes.borrow(),
        ) {
            state::release_structured(self.context.handle, handle);
            return self.fail(error);
        }
        inner.usage.set(BufferUse::Claimed);
        let transient = Rc::new(TransientInner {
            context: Rc::clone(&self.context),
            buffer: Rc::clone(inner),
            handle: TransientHandle::Structured(handle),
            state: Cell::new(TransientState::Live),
        });
        self.transients.push(transient);
        self.retain(inner);
        Ok(handle)
    }

    fn materialize_counter(&mut self, inner: &Rc<BufferInner>) -> Result<IndirectBufferHandle> {
        self.ensure_context(&inner.context)?;
        if inner.element_size as usize
            != core::mem::size_of::<ez_gfx_runtime::indirect::DrawIndexedCommand>()
        {
            return self.fail(Error::InvalidArgument);
        }
        if let Some(handle) = self.transients.iter().find_map(|transient| {
            Rc::ptr_eq(&transient.buffer, inner).then_some(&transient.handle)
        }) {
            return match handle {
                TransientHandle::Indirect(handle) => Ok(*handle),
                TransientHandle::Structured(_) => self.fail(Error::InvalidContext),
            };
        }
        if inner.usage.get() != BufferUse::Available {
            return self.fail(Error::NotReady);
        }
        let handle = match state::acquire_indirect(self.context.handle, inner.element_count) {
            Ok(handle) => handle,
            Err(error) => return self.fail(error),
        };
        let staged =
            state::write_indirect_bytes(self.context.handle, handle, &inner.bytes.borrow()).and_then(
                |()| {
                    state::publish_compute_indirect_count(
                        self.context.handle,
                        handle,
                        inner.published_count.get(),
                    )
                },
            );
        if let Err(error) = staged {
            state::release_indirect(self.context.handle, handle);
            return self.fail(error);
        }
        inner.usage.set(BufferUse::Claimed);
        let transient = Rc::new(TransientInner {
            context: Rc::clone(&self.context),
            buffer: Rc::clone(inner),
            handle: TransientHandle::Indirect(handle),
            state: Cell::new(TransientState::Live),
        });
        self.transients.push(transient);
        self.retain(inner);
        Ok(handle)
    }

    fn raw_bindings(&mut self, bindings: &[Binding<'_>]) -> Result<Vec<RawBinding>> {
        let mut raw = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let resource = match binding.resource {
                BindingResource::Structured(inner) => {
                    ResourceIdentity::Structured(self.materialize_structured(inner)?)
                }
                BindingResource::Indirect(inner) => {
                    ResourceIdentity::Indirect(self.materialize_counter(inner)?)
                }
                BindingResource::RenderTarget(inner) => {
                    self.ensure_context(&inner.context)?;
                    let handle = match inner.managed_handle() {
                        Ok(handle) => handle,
                        Err(error) => return self.fail(error),
                    };
                    self.retain(inner);
                    ResourceIdentity::RenderTarget(handle)
                }
            };
            raw.push(RawBinding {
                name: binding.name.clone(),
                resource,
            });
        }
        Ok(raw)
    }

    /// Records an indexed graphics operation.
    ///
    /// # Errors
    /// Returns [`Error`] when resources, bindings, constants, or recording state are invalid.
    pub fn add_graphics(
        &mut self,
        shader: &Shader,
        indirect: &CounterBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand>,
        bindings: &[Binding<'_>],
        state_desc: ez_gfx_hal::DynamicPipelineState,
        push_constants: &[u8],
    ) -> Result<()> {
        self.ensure_context(&shader.inner.context)?;
        let indirect_handle = self.materialize_counter(&indirect.inner)?;
        let raw_bindings = self.raw_bindings(bindings)?;
        self.retain(&shader.inner);
        self.record(|context| {
            state::render_add_graphics(
                context,
                shader.inner.handle,
                indirect_handle,
                &raw_bindings,
                state_desc,
                push_constants,
            )
        })
    }

    /// Records a compute dispatch.
    ///
    /// # Errors
    /// Returns [`Error`] when resources, bindings, dispatch, or recording state are invalid.
    pub fn add_compute(
        &mut self,
        shader: &Shader,
        groups: [u32; 3],
        bindings: &[Binding<'_>],
        push_constants: &[u8],
    ) -> Result<()> {
        self.ensure_context(&shader.inner.context)?;
        let raw_bindings = self.raw_bindings(bindings)?;
        self.retain(&shader.inner);
        self.record(|context| {
            state::render_add_compute(
                context,
                shader.inner.handle,
                groups,
                &raw_bindings,
                push_constants,
            )
        })
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
            // Presented images have no stable raw target handle; surface capture is emitted last.
            RenderTargetBacking::Surface { .. } => {
                let Some(surface) = target.surface_lease.as_ref() else {
                    return self.fail(Error::InvalidContext);
                };
                self.record(|context| {
                    state::set_snapshot_cache(context, surface.handle, true)
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

    /// Explicitly retains a texture used through the bindless heap.
    ///
    /// # Errors
    /// Returns [`Error`] when the texture belongs to another context.
    pub fn retain_texture(&mut self, texture: &Texture) -> Result<()> {
        self.ensure_context(&texture.inner.context)?;
        self.retain(&texture.inner);
        Ok(())
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
            let handle = match state::create_render_target(
                self.context.handle,
                &declaration,
                width,
                height,
            ) {
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
    ///
    /// # Errors
    /// Returns [`Error`] when the frame is already configured, the extent is zero,
    /// or surface resize/acquisition fails.
    pub fn configure_swapchain(
        &mut self,
        size: [u32; 2],
        format: ez_gfx_runtime::target::Format,
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
        if let Err(error) = state::configure_surface(self.context.handle, surface.handle) {
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
        let outputs = if result.is_ok() {
            match state::frame_readbacks(self.context.handle) {
                Ok(outputs) => Some(Ok(outputs)),
                Err(Error::NotReady) if self.readbacks.is_empty() => None,
                Err(Error::NotReady) => Some(Err(Error::NativeFailure)),
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
        let has_presented = readbacks.iter().any(|readback| {
            matches!(readback, PendingReadback::RequestedPresentation(_))
        });
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
        };
        context.check_entry()?;
        context.dispatch_events()?;
        state::frame_begin(context.raw())?;
        let target_lease: Rc<dyn Any> = self.inner.clone();
        Ok(Frame {
            context: Rc::clone(&context.inner),
            target: FrameTarget::Unconfigured,
            surface: Some(Rc::clone(&self.inner)),
            retained: vec![target_lease],
            transients: Vec::new(),
            poison: None,
            terminal: false,
            readbacks: Vec::new(),
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
        Ok(Frame {
            context: Rc::clone(&self.inner),
            target: FrameTarget::Unconfigured,
            surface: None,
            retained: Vec::new(),
            transients: Vec::new(),
            poison: None,
            terminal: false,
            readbacks: Vec::new(),
        })
    }
}
