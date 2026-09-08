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
    handle: TransientHandle,
    state: Cell<TransientState>,
}

impl Drop for TransientInner {
    fn drop(&mut self) {
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

/// Frame-local typed buffer.
pub struct Buffer<T: bytemuck::Pod> {
    inner: Rc<TransientInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    /// Writes a typed element range while this buffer's frame is live.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame or buffer is stale, foreign, or out of range.
    pub fn write(&self, frame: &mut Frame, start_index: usize, values: &[T]) -> Result<()> {
        frame.ensure_transient(&self.inner)?;
        let TransientHandle::Structured(handle) = self.inner.handle else {
            return frame.fail(Error::InvalidContext);
        };
        frame.record(|context| state::write_structured(context, handle, start_index, values))
    }
}

/// Frame-local typed buffer carrying a separately publishable visible count.
pub struct CountedBuffer<T> {
    inner: Rc<TransientInner>,
    marker: PhantomData<T>,
}

impl CountedBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand> {
    /// Writes indexed draw commands.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame or buffer is stale, foreign, or out of range.
    pub fn write(
        &self,
        frame: &mut Frame,
        start_index: u32,
        commands: &[ez_gfx_runtime::indirect::DrawIndexedCommand],
    ) -> Result<()> {
        frame.ensure_transient(&self.inner)?;
        let TransientHandle::Indirect(handle) = self.inner.handle else {
            return frame.fail(Error::InvalidContext);
        };
        frame.record(|context| state::write_indirect(context, handle, start_index, commands))
    }

    /// Publishes the visible command count.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame or buffer is stale, foreign, or out of range.
    pub fn publish_count(&self, frame: &mut Frame, count: u32) -> Result<()> {
        frame.ensure_transient(&self.inner)?;
        let TransientHandle::Indirect(handle) = self.inner.handle else {
            return frame.fail(Error::InvalidContext);
        };
        frame.record(|context| state::publish_compute_indirect_count(context, handle, count))
    }
}

enum BindingResource<'a> {
    Structured(&'a Rc<TransientInner>),
    Indirect(&'a Rc<TransientInner>),
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

    /// Binds a counted buffer.
    pub fn counted_buffer<T>(name: impl Into<String>, buffer: &'a CountedBuffer<T>) -> Self {
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
    readback_queued: bool,
}

impl Frame {
    fn ensure_context(&mut self, context: &Rc<ContextInner>) -> Result<()> {
        if Rc::ptr_eq(&self.context, context) {
            Ok(())
        } else {
            self.fail(Error::InvalidContext)
        }
    }

    fn ensure_transient(&mut self, transient: &Rc<TransientInner>) -> Result<()> {
        self.ensure_context(&transient.context)?;
        if transient.state.get() == TransientState::Live {
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
        }
    }

    fn release_aborted_transients(&self) {
        for transient in &self.transients {
            if transient.state.replace(TransientState::Consumed) != TransientState::Live {
                continue;
            }
            match transient.handle {
                TransientHandle::Structured(handle) => {
                    state::release_structured(self.context.handle, handle);
                }
                TransientHandle::Indirect(handle) => {
                    state::release_indirect(self.context.handle, handle);
                }
            }
        }
    }

    /// Acquires a typed transient buffer by element count.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame, count, type size, or allocation is invalid.
    pub fn acquire_buffer<T: bytemuck::Pod>(
        &mut self,
        element_count: usize,
    ) -> Result<Buffer<T>> {
        if let Some(error) = self.poison {
            return Err(error);
        }
        let handle = match state::acquire_structured::<T>(self.context.handle, element_count) {
            Ok(handle) => handle,
            Err(error) => return self.fail(error),
        };
        let inner = Rc::new(TransientInner {
            context: Rc::clone(&self.context),
            handle: TransientHandle::Structured(handle),
            state: Cell::new(TransientState::Live),
        });
        self.transients.push(Rc::clone(&inner));
        Ok(Buffer {
            inner,
            marker: PhantomData,
        })
    }

    /// Acquires an indexed-command counted buffer by element count.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame, count, or allocation is invalid.
    pub fn acquire_counted_buffer(
        &mut self,
        element_count: u32,
    ) -> Result<CountedBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand>> {
        if let Some(error) = self.poison {
            return Err(error);
        }
        let handle = match state::acquire_indirect(self.context.handle, element_count) {
            Ok(handle) => handle,
            Err(error) => return self.fail(error),
        };
        let inner = Rc::new(TransientInner {
            context: Rc::clone(&self.context),
            handle: TransientHandle::Indirect(handle),
            state: Cell::new(TransientState::Live),
        });
        self.transients.push(Rc::clone(&inner));
        Ok(CountedBuffer {
            inner,
            marker: PhantomData,
        })
    }

    fn raw_bindings(&mut self, bindings: &[Binding<'_>]) -> Result<Vec<RawBinding>> {
        let mut raw = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let resource = match binding.resource {
                BindingResource::Structured(inner) => {
                    self.ensure_transient(inner)?;
                    self.retain(inner);
                    let TransientHandle::Structured(handle) = inner.handle else {
                        return self.fail(Error::InvalidContext);
                    };
                    ResourceIdentity::Structured(handle)
                }
                BindingResource::Indirect(inner) => {
                    self.ensure_transient(inner)?;
                    self.retain(inner);
                    let TransientHandle::Indirect(handle) = inner.handle else {
                        return self.fail(Error::InvalidContext);
                    };
                    ResourceIdentity::Indirect(handle)
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
        indirect: &CountedBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand>,
        bindings: &[Binding<'_>],
        state_desc: ez_gfx_hal::DynamicPipelineState,
        push_constants: &[u8],
    ) -> Result<()> {
        self.ensure_context(&shader.inner.context)?;
        self.ensure_transient(&indirect.inner)?;
        let raw_bindings = self.raw_bindings(bindings)?;
        self.retain(&shader.inner);
        self.retain(&indirect.inner);
        let TransientHandle::Indirect(indirect_handle) = indirect.inner.handle else {
            return self.fail(Error::InvalidContext);
        };
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
    /// Completed bytes are delivered through [`Event::Readback`] and are valid
    /// only for that callback invocation.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame or texture is stale, foreign, or not ready.
    pub fn enqueue_texture_readback(&mut self, texture: &Texture) -> Result<()> {
        self.ensure_context(&texture.inner.context)?;
        self.retain(&texture.inner);
        let result =
            self.record(|context| state::frame_enqueue_readback(context, texture.inner.handle));
        if result.is_ok() {
            self.readback_queued = true;
        }
        result
    }

    /// Enqueues an opaque managed-target readback request.
    ///
    /// # Errors
    /// Returns [`Error`] when the request is stale, foreign, or not renderable.
    pub fn enqueue_readback(&mut self, request: &RenderTargetReadback) -> Result<()> {
        self.ensure_context(&request.target.context)?;
        let handle = match request.target.managed_handle() {
            Ok(handle) => handle,
            Err(error) => return self.fail(error),
        };
        self.retain(&request.target);
        let result =
            self.record(|context| state::frame_enqueue_render_target_readback(context, handle));
        if result.is_ok() {
            self.readback_queued = true;
        }
        result
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

    /// Retains a vertex allocation and its owning heap through completion.
    ///
    /// # Errors
    /// Returns [`Error`] when the allocation belongs to another context.
    pub fn retain_vertex_allocation<T: bytemuck::Pod>(
        &mut self,
        allocation: &VertexAllocation<T>,
    ) -> Result<()> {
        self.ensure_context(&allocation.inner.heap.context)?;
        self.retain(&allocation.inner);
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
            _surface_lease: None,
        })
    }


    /// Retains an index allocation through completion.
    ///
    /// # Errors
    /// Returns [`Error`] when the allocation belongs to another context.
    pub fn retain_index_allocation(&mut self, allocation: &IndexAllocation) -> Result<()> {
        self.ensure_context(&allocation.inner.context)?;
        self.retain(&allocation.inner);
        Ok(())
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
            _surface_lease: Some(surface),
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
        let readback = if result.is_ok()
            && (self.readback_queued || matches!(self.target, FrameTarget::Surface))
        {
            match state::frame_readback(self.context.handle) {
                Ok(bytes) => Some(Ok(bytes)),
                Err(Error::NotReady) if !self.readback_queued => None,
                Err(error) => Some(Err(error)),
            }
        } else {
            None
        };
        self.terminal = true;
        drop(self);
        let result = context.complete(result);
        match (result, readback) {
            (Err(error), _) | (Ok(()), Some(Err(error))) => Err(error),
            (Ok(()), Some(Ok(bytes))) => context.dispatch_readback(&bytes),
            (Ok(()), None) => Ok(()),
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if !self.terminal {
            // Rollback precedes release so Interned(frame_serial) becomes releasable.
            let _ = state::frame_abort(self.context.handle);
            self.release_aborted_transients();
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
            readback_queued: false,
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
            readback_queued: false,
        })
    }
}
