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
    Buffer(BufferHandle),
    Counter(CounterBufferHandle),
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
            TransientHandle::Buffer(handle) => {
                state::release_buffer(self.context.handle, handle);
            }
            TransientHandle::Counter(handle) => {
                state::release_counter(self.context.handle, handle);
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
        let context = Context { inner: Rc::clone(&self.inner.context), owner: false, };
        context.check_entry()?;
        self.inner.write(start_index, values)
    }
}

/// Context-acquired one-frame buffer with a shader-writable count.
pub struct CounterBuffer<T: bytemuck::Pod> {
    inner: Rc<BufferInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> CounterBuffer<T> {
    /// Replaces a typed element range and advances the initial visible count.
    ///
    /// GPU producers can replace that count with `set_count` or `add_count`.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is dispatching a callback or the range is invalid.
    pub fn write(&self, start_index: usize, values: &[T]) -> Result<()> {
        let context = Context { inner: Rc::clone(&self.inner.context), owner: false, };
        context.check_entry()?;
        self.inner.write(start_index, values)?;
        let end = start_index
            .checked_add(values.len())
            .and_then(|end| u32::try_from(end).ok())
            .ok_or(Error::InvalidArgument)?;
        self.inner.published_count.set(self.inner.published_count.get().max(end));
        Ok(())
    }
}

/// Context-acquired one-frame buffer containing exactly one POD value.
pub struct ValueBuffer<T: bytemuck::Pod> {
    inner: Rc<BufferInner>,
    marker: PhantomData<T>,
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

    /// Acquires a one-frame buffer containing exactly one POD value.
    ///
    /// # Errors
    /// Returns [`Error`] when the value is zero-sized, oversized, or the context is invalid.
    pub fn acquire_value_buffer<T: bytemuck::Pod>(&self, value: T) -> Result<ValueBuffer<T>> {
        Ok(ValueBuffer {
            inner: self.allocate_buffer(1, Some(core::slice::from_ref(&value)), 0)?,
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
    /// The initialized element count becomes the GPU-visible initial count.
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
    fn bind_to_frame(&self, frame: &mut Frame, name: String) -> Result<()>;
}

impl<T: bytemuck::Pod> bindable_buffer::Sealed for Buffer<T> {}
impl<T: bytemuck::Pod> BindableBuffer for Buffer<T> {
    fn bind_to_frame(&self, frame: &mut Frame, name: String) -> Result<()> {
        frame.bind_draft(name, DraftBinding::Buffer(Rc::clone(&self.inner)))
    }
}

impl<T: bytemuck::Pod> bindable_buffer::Sealed for ValueBuffer<T> {}
impl<T: bytemuck::Pod> BindableBuffer for ValueBuffer<T> {
    fn bind_to_frame(&self, frame: &mut Frame, name: String) -> Result<()> {
        frame.bind_draft(name, DraftBinding::Buffer(Rc::clone(&self.inner)))
    }
}

impl<T: bytemuck::Pod> bindable_buffer::Sealed for CounterBuffer<T> {}
impl<T: bytemuck::Pod> BindableBuffer for CounterBuffer<T> {
    fn bind_to_frame(&self, frame: &mut Frame, name: String) -> Result<()> {
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
    bindings: HashMap<String, DraftBinding>,
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
        transients: &mut Vec<Rc<TransientInner>>,
        retained: &mut Vec<Rc<dyn Any>>,
        inner: &Rc<BufferInner>,
    ) -> Result<BufferHandle> {
        if !Rc::ptr_eq(context, &inner.context) {
            return Err(Error::InvalidContext);
        }
        if let Some(handle) = transients.iter().find_map(|transient| {
            Rc::ptr_eq(&transient.buffer, inner).then_some(&transient.handle)
        }) {
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
        transients.push(Rc::new(TransientInner {
            context: Rc::clone(context),
            buffer: Rc::clone(inner),
            handle: TransientHandle::Buffer(handle),
            state: Cell::new(TransientState::Live),
        }));
        let retained_inner: Rc<dyn Any> = inner.clone();
        retained.push(retained_inner);
        Ok(handle)
    }


    fn materialize_counter_state(
        context: &Rc<ContextInner>,
        transients: &mut Vec<Rc<TransientInner>>,
        retained: &mut Vec<Rc<dyn Any>>,
        inner: &Rc<BufferInner>,
    ) -> Result<CounterBufferHandle> {
        if !Rc::ptr_eq(context, &inner.context) {
            return Err(Error::InvalidContext);
        }
        if inner.element_size as usize
            != core::mem::size_of::<ez_gfx_runtime::indirect::DrawIndexedCommand>()
        {
            return Err(Error::InvalidArgument);
        }
        if let Some(handle) = transients.iter().find_map(|transient| {
            Rc::ptr_eq(&transient.buffer, inner).then_some(&transient.handle)
        }) {
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
        transients.push(Rc::new(TransientInner {
            context: Rc::clone(context),
            buffer: Rc::clone(inner),
            handle: TransientHandle::Counter(handle),
            state: Cell::new(TransientState::Live),
        }));
        let retained_inner: Rc<dyn Any> = inner.clone();
        retained.push(retained_inner);
        Ok(handle)
    }

    fn materialize_counter(&mut self, inner: &Rc<BufferInner>) -> Result<CounterBufferHandle> {
        match Self::materialize_counter_state(
            &self.context,
            &mut self.transients,
            &mut self.retained,
            inner,
        ) {
            Ok(handle) => Ok(handle),
            Err(error) => self.fail(error),
        }
    }

    fn raw_bindings(&mut self) -> Result<Vec<RawBinding>> {
        let mut raw = Vec::with_capacity(self.bindings.len());
        for (name, binding) in &self.bindings {
            let resource = match binding {
                DraftBinding::Buffer(inner) => Self::materialize_buffer_state(
                    &self.context,
                    &mut self.transients,
                    &mut self.retained,
                    inner,
                )
                .map(ResourceIdentity::Buffer),
                DraftBinding::Counter(inner) => Self::materialize_counter_state(
                    &self.context,
                    &mut self.transients,
                    &mut self.retained,
                    inner,
                )
                .map(ResourceIdentity::Counter),
            };
            let resource = match resource {
                Ok(resource) => resource,
                Err(error) => return self.fail(error),
            };
            raw.push(RawBinding {
                name: name.clone(),
                resource,
            });
        }
        Ok(raw)
    }

    /// Adds or replaces one named buffer in the frame binding set.
    ///
    /// # Errors
    /// Returns [`Error`] when the name, frame, resource state, or ownership is invalid.
    pub fn bind_buffer<B: BindableBuffer + ?Sized>(
        &mut self,
        name: impl Into<String>,
        buffer: &B,
    ) -> Result<()> {
        buffer.bind_to_frame(self, name.into())
    }

    fn bind_draft(&mut self, name: String, binding: DraftBinding) -> Result<()> {
        if name.is_empty() || name.len() > 255 || name.as_bytes().contains(&0) {
            return self.fail(Error::InvalidArgument);
        }
        let inner = match &binding {
            DraftBinding::Buffer(inner) | DraftBinding::Counter(inner) => inner,
        };
        self.ensure_context(&inner.context)?;
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
        self.bindings.insert(name, binding);
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
        let raw_bindings = self.raw_bindings()?;
        self.retain(&vertex_shader.inner);
        self.retain(&fragment_shader.inner);
        self.record(|context| {
            state::execute_graphics(
                context,
                vertex_shader.inner.handle,
                fragment_shader.inner.handle,
                counter_handle,
                &raw_bindings,
                state_desc,
            )
        })
    }

    /// Executes a compute dispatch with one exact compute entry point.
    ///
    /// # Errors
    /// Returns [`Error`] when resources, bindings, dispatch, stage ownership, or recording state are invalid.
    pub fn execute_compute(&mut self, shader: &ComputeShader, groups: [u32; 3]) -> Result<()> {
        self.ensure_context(&shader.inner.context)?;
        let raw_bindings = self.raw_bindings()?;
        self.retain(&shader.inner);
        self.record(|context| {
            state::execute_compute(context, shader.inner.handle, groups, &raw_bindings)
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
        let context = Context { inner: Rc::clone(&self.context), owner: false, };
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
        let context = Context { inner: Rc::clone(&self.inner.context), owner: false, };
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
            bindings: HashMap::new(),
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
            bindings: HashMap::new(),
        })
    }
}
impl Surface {
    /// Returns the presentation modes available for this initialized surface.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale or the backend query fails.
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
    /// Returns [`Error`] when the surface query fails or FIFO is unavailable.
    pub fn resolve_presentation_mode(
        &self,
        requested: PresentationMode,
    ) -> Result<PresentationMode> {
        self.presentation_modes()?
            .resolve(requested)
            .ok_or(Error::Unsupported)
    }
}
