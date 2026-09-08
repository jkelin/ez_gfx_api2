use std::{any::Any, cell::Cell, marker::PhantomData, rc::Rc};

use ez_gfx_core::{
    capability::CapabilityError,
    handle::{
        ContextHandle, IndexAllocationHandle, IndirectBufferHandle, RenderTargetHandle,
        ShaderHandle, StructuredBufferHandle, SurfaceHandle, TextureHandle, VertexAllocationHandle,
        VertexHeapHandle,
    },
};
use ez_gfx_runtime::{
    LifecycleError,
    binding::{PublicBinding as RawBinding, ResourceIdentity},
};

use crate::state;

/// Error returned by the safe Rust facade.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An argument violates the operation contract.
    #[error("invalid argument")]
    InvalidArgument,
    /// A context or resource handle is invalid or stale.
    #[error("invalid or stale context/resource handle")]
    InvalidContext,
    /// The native graphics backend failed.
    #[error("native graphics backend failure")]
    NativeFailure,
    /// Completion or output is not yet available.
    #[error("operation is not ready")]
    NotReady,
    /// The requested capability is unavailable.
    #[error("unsupported operation or capability")]
    Unsupported,
    /// The graphics device was lost.
    #[error("graphics device lost")]
    DeviceLost,
    /// Asynchronous scheduling or staging capacity is unavailable.
    #[error("asynchronous scheduling capacity unavailable")]
    QueueFull,
    /// An asynchronous operation was cancelled before completion.
    #[error("asynchronous operation cancelled")]
    Cancelled,
    /// Preserves a lifecycle or handle-validation cause.
    #[error(transparent)]
    Lifecycle(#[from] LifecycleError),
    /// Preserves an adapter capability cause.
    #[error(transparent)]
    Capability(#[from] CapabilityError),
}

/// Result returned by the safe Rust facade.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Diagnostic item and dropped-record count returned by [`Context::poll_diagnostic`].
pub type DiagnosticPoll = (
    Option<(
        ez_gfx_runtime::observability::DiagnosticLevel,
        ez_gfx_runtime::observability::RuntimeRecord,
    )>,
    u64,
);

struct ContextInner {
    handle: ContextHandle,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // Context destruction performs its own creator-thread check. Drop cannot
        // report teardown failures, so explicit callers should use `wait_idle` first.
        let _ = state::destroy_context(self.handle);
    }
}

/// Creator-thread graphics context.
///
/// The `Rc` ownership marker deliberately makes this type `!Send` and `!Sync`.
/// Dropping the last context or resource lease performs creator-thread teardown.
#[derive(Clone)]
pub struct Context {
    inner: Rc<ContextInner>,
}

impl Context {
    pub(crate) fn raw(&self) -> ContextHandle {
        self.inner.handle
    }

    /// Waits for all submitted work to complete.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale, unhealthy, or native waiting fails.
    pub fn wait_idle(&self) -> Result<()> {
        state::wait_idle(self.raw())
    }

    /// Polls the next runtime event and dropped-event count.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale or asynchronous progress fails.
    pub fn poll_runtime_event(
        &self,
    ) -> Result<(Option<ez_gfx_runtime::observability::RuntimeRecord>, u64)> {
        state::poll_runtime_event(self.raw())
    }

    /// Polls the next upload lifecycle event.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale or asynchronous progress fails.
    pub fn poll_upload_event(&self) -> Result<Option<ez_gfx_runtime::upload::UploadEvent>> {
        state::poll_upload_event(self.raw())
    }

    /// Polls the next diagnostic and dropped-record count.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale.
    pub fn poll_diagnostic(&self) -> Result<DiagnosticPoll> {
        state::poll_diagnostic(self.raw())
    }

    /// Returns the configured asynchronous texture worker count.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale.
    pub fn texture_decode_worker_count(&self) -> Result<u32> {
        state::texture_decode_worker_count(self.raw())
    }

    /// Returns context-wide texture upload counters.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale or unhealthy.
    pub fn texture_upload_telemetry(
        &self,
    ) -> Result<ez_gfx_runtime::texture::TextureUploadTelemetrySnapshot> {
        state::texture_upload_telemetry(self.raw())
    }

    /// Takes the last completed texture readback.
    ///
    /// # Errors
    /// Returns [`Error`] when no completed readback exists or the context is stale.
    pub fn frame_readback(&self) -> Result<Vec<u8>> {
        state::frame_readback(self.raw())
    }
}

/// Creates a creator-thread graphics context.
///
/// # Errors
/// Returns [`Error`] when native context creation or validation fails.
pub fn create_context(options: ez_gfx_runtime::ContextOptions) -> Result<Context> {
    state::create_context(options).map(|handle| Context {
        inner: Rc::new(ContextInner { handle }),
    })
}

/// Enumerates adapters visible to all compiled backends.
#[must_use]
pub fn enumerate_adapters() -> Vec<ez_gfx_core::capability::AdapterInfo> {
    state::enumerate_adapters()
}

/// Diagnoses adapter admission without creating a context.
#[must_use]
pub fn query_adapter_report(allow_software: bool) -> Vec<ez_gfx_runtime::AdapterReport> {
    state::query_adapter_report(allow_software)
}

struct SurfaceInner {
    context: Rc<ContextInner>,
    handle: SurfaceHandle,
}

impl Drop for SurfaceInner {
    fn drop(&mut self) {
        state::destroy_surface(self.context.handle, self.handle);
    }
}

/// Owning presentation surface.
pub struct Surface {
    inner: Rc<SurfaceInner>,
}

impl Surface {
    /// Requests a surface resize.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale or the extent is invalid.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        state::resize_surface(self.inner.context.handle, self.inner.handle, width, height)
    }

    /// Returns the current nonzero surface extent.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale or has no ready extent.
    pub fn extent(&self) -> Result<(u32, u32)> {
        state::surface_extent(self.inner.context.handle, self.inner.handle)
    }

    /// Reports whether a resize still awaits the next render.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale.
    pub fn resize_pending(&self) -> Result<bool> {
        state::surface_resize_pending(self.inner.context.handle, self.inner.handle)
    }

    /// Enables or disables presented snapshot caching.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale.
    pub fn set_snapshot_cache(&self, enabled: bool) -> Result<()> {
        state::set_snapshot_cache(self.inner.context.handle, self.inner.handle, enabled)
    }
}

/// Creates, initializes, and sizes a surface as one safe operation.
///
/// Every failure destroys the unpublished raw surface before returning.
///
/// # Errors
/// Returns [`Error`] when creation, device initialization, or initial sizing fails.
pub fn create_surface(
    context: &Context,
    options: ez_gfx_runtime::SurfaceOptions,
) -> Result<Surface> {
    let handle = state::create_surface(context.raw(), options)?;
    let initialized = state::init_device(context.raw(), handle)
        .and_then(|()| state::resize_surface(context.raw(), handle, options.width, options.height));
    if let Err(error) = initialized {
        // Even partial native initialization never publishes a safe wrapper.
        state::destroy_surface(context.raw(), handle);
        return Err(error);
    }
    Ok(Surface {
        inner: Rc::new(SurfaceInner {
            context: Rc::clone(&context.inner),
            handle,
        }),
    })
}

struct ShaderInner {
    context: Rc<ContextInner>,
    handle: ShaderHandle,
}

impl Drop for ShaderInner {
    fn drop(&mut self) {
        state::destroy_shader(self.context.handle, self.handle);
    }
}

/// Owning shader artifact.
pub struct Shader {
    inner: Rc<ShaderInner>,
}

/// Loads a validated shader artifact.
///
/// # Errors
/// Returns [`Error`] when the artifact or native shader is invalid.
pub fn load_shader(context: &Context, artifact: &[u8]) -> Result<Shader> {
    state::load_shader(context.raw(), artifact).map(|handle| Shader {
        inner: Rc::new(ShaderInner {
            context: Rc::clone(&context.inner),
            handle,
        }),
    })
}

struct TextureInner {
    context: Rc<ContextInner>,
    handle: TextureHandle,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        state::unload_texture(self.context.handle, self.handle);
    }
}

/// Owning bindless texture.
pub struct Texture {
    inner: Rc<TextureInner>,
}

impl Texture {
    /// Returns this texture's stable shader binding index.
    ///
    /// # Errors
    /// Returns [`Error`] until the texture is resident or when it is stale.
    pub fn binding(&self) -> Result<u32> {
        state::texture_binding(self.inner.context.handle, self.inner.handle)
    }

    /// Returns resident and total mip counts.
    ///
    /// # Errors
    /// Returns [`Error`] when the texture is stale.
    pub fn residency(&self) -> Result<(u32, u32)> {
        state::texture_residency(self.inner.context.handle, self.inner.handle)
    }

    /// Changes the contiguous resident mip count.
    ///
    /// # Errors
    /// Returns [`Error`] when the mip request or texture state is invalid.
    pub fn set_residency(&self, resident_mips: u32) -> Result<()> {
        state::set_texture_residency(self.inner.context.handle, self.inner.handle, resident_mips)
    }

    /// Uploads one validated texture region.
    ///
    /// # Errors
    /// Returns [`Error`] when the region, texture state, or upload fails.
    pub fn update_region(&self, region: ez_gfx_hal::TextureRegion<'_>) -> Result<()> {
        state::update_texture_region(self.inner.context.handle, self.inner.handle, region)
    }

    /// Cancels pending decode or transfer work.
    ///
    /// # Errors
    /// Returns [`Error`] when cancellation or texture ownership validation fails.
    pub fn cancel_load(&self) -> Result<()> {
        state::cancel_texture_load(self.inner.context.handle, self.inner.handle)
    }
}

/// Queues texture decode and upload while owning the copied input.
///
/// # Errors
/// Returns [`Error`] when input, scheduling, decode admission, or ownership validation fails.
pub fn load_texture(
    context: &Context,
    source: ez_gfx_runtime::texture::TextureSource,
    bytes: &[u8],
    generate_mips: bool,
    config: &state::TextureConfig,
) -> Result<Texture> {
    state::load_texture(context.raw(), source, bytes, generate_mips, config).map(|handle| Texture {
        inner: Rc::new(TextureInner {
            context: Rc::clone(&context.inner),
            handle,
        }),
    })
}

struct RenderTargetInner {
    context: Rc<ContextInner>,
    handle: RenderTargetHandle,
}

impl Drop for RenderTargetInner {
    fn drop(&mut self) {
        state::destroy_render_target(self.context.handle, self.handle);
    }
}

/// Owning managed render target.
pub struct RenderTarget {
    inner: Rc<RenderTargetInner>,
}

impl RenderTarget {
    /// Returns the resolved target format.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn format(&self) -> Result<ez_gfx_runtime::target::Format> {
        state::render_target_format(self.inner.context.handle, self.inner.handle)
    }

    /// Returns the target extent.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn extent(&self) -> Result<(u32, u32)> {
        state::render_target_extent(self.inner.context.handle, self.inner.handle)
    }

    /// Returns the target clear value.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn clear(&self) -> Result<ez_gfx_runtime::target::ClearValue> {
        state::render_target_clear(self.inner.context.handle, self.inner.handle)
    }
}

/// Creates an owning render target.
///
/// # Errors
/// Returns [`Error`] when the declaration, extent, capability, or allocation is invalid.
pub fn create_render_target(
    context: &Context,
    declaration: &ez_gfx_runtime::target::TargetDeclaration,
    width: u32,
    height: u32,
) -> Result<RenderTarget> {
    state::create_render_target(context.raw(), declaration, width, height).map(|handle| {
        RenderTarget {
            inner: Rc::new(RenderTargetInner {
                context: Rc::clone(&context.inner),
                handle,
            }),
        }
    })
}

/// Probes one render-target format without allocating it.
///
/// # Errors
/// Returns [`Error`] when the format or sample count is unsupported.
pub fn probe_render_target_format(
    context: &Context,
    format: ez_gfx_runtime::target::Format,
    samples: u8,
) -> Result<()> {
    state::probe_render_target_format(context.raw(), format, samples)
}

struct VertexHeapInner {
    context: Rc<ContextInner>,
    handle: VertexHeapHandle,
}

impl Drop for VertexHeapInner {
    fn drop(&mut self) {
        state::destroy_vertex_heap(self.context.handle, self.handle);
    }
}

/// Owning named vertex heap.
pub struct VertexHeap {
    inner: Rc<VertexHeapInner>,
}

/// Creates an owning named vertex heap.
///
/// # Errors
/// Returns [`Error`] when the heap contract or native allocation is invalid.
pub fn create_vertex_heap(
    context: &Context,
    name: &str,
    capacity: u64,
    stride: u64,
) -> Result<VertexHeap> {
    state::create_vertex_heap(context.raw(), name, capacity, stride).map(|handle| VertexHeap {
        inner: Rc::new(VertexHeapInner {
            context: Rc::clone(&context.inner),
            handle,
        }),
    })
}

/// Creates the context-owned singleton index heap.
///
/// # Errors
/// Returns [`Error`] when the singleton already exists or allocation fails.
pub fn create_index_heap(context: &Context, capacity: u64) -> Result<()> {
    state::create_index_heap(context.raw(), capacity)
}

struct VertexAllocationInner {
    heap: Rc<VertexHeapInner>,
    handle: VertexAllocationHandle,
}

impl Drop for VertexAllocationInner {
    fn drop(&mut self) {
        let _ = state::remove_vertices(self.heap.context.handle, self.handle);
    }
}

/// Owning vertex allocation that retains its heap.
pub struct VertexAllocation {
    inner: Rc<VertexAllocationInner>,
}

impl VertexAllocation {
    /// Returns `(first_element, element_count)`.
    ///
    /// # Errors
    /// Returns [`Error`] when the allocation is stale.
    pub fn range(&self) -> Result<(u32, u32)> {
        state::vertex_allocation_range(self.inner.heap.context.handle, self.inner.handle)
    }
}

/// Uploads typed vertices and retains the destination heap.
///
/// # Errors
/// Returns [`Error`] when the slice, heap ownership, capacity, or upload is invalid.
pub fn upload_vertices<T: bytemuck::Pod>(
    heap: &VertexHeap,
    vertices: &[T],
) -> Result<VertexAllocation> {
    state::upload_vertices(heap.inner.context.handle, heap.inner.handle, vertices).map(|handle| {
        VertexAllocation {
            inner: Rc::new(VertexAllocationInner {
                heap: Rc::clone(&heap.inner),
                handle,
            }),
        }
    })
}

struct IndexAllocationInner {
    context: Rc<ContextInner>,
    handle: IndexAllocationHandle,
}

impl Drop for IndexAllocationInner {
    fn drop(&mut self) {
        let _ = state::remove_indices(self.context.handle, self.handle);
    }
}

/// Owning allocation from the context index heap.
pub struct IndexAllocation {
    inner: Rc<IndexAllocationInner>,
}

impl IndexAllocation {
    /// Returns `(first_index, index_count)`.
    ///
    /// # Errors
    /// Returns [`Error`] when the allocation is stale.
    pub fn range(&self) -> Result<(u32, u32)> {
        state::index_allocation_range(self.inner.context.handle, self.inner.handle)
    }
}

/// Uploads packed indices to the context singleton heap.
///
/// # Errors
/// Returns [`Error`] when the slice, heap capacity, or upload is invalid.
pub fn upload_indices(context: &Context, indices: &[u32]) -> Result<IndexAllocation> {
    state::upload_indices(context.raw(), indices).map(|handle| IndexAllocation {
        inner: Rc::new(IndexAllocationInner {
            context: Rc::clone(&context.inner),
            handle,
        }),
    })
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

/// Frame-local typed structured buffer.
pub struct StructuredBuffer<T: bytemuck::Pod> {
    inner: Rc<TransientInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> StructuredBuffer<T> {
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

/// Frame-local indexed-indirect command buffer.
pub struct IndirectBuffer {
    inner: Rc<TransientInner>,
}

impl IndirectBuffer {
    /// Writes and publishes indexed draw commands.
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

    /// Publishes a CPU-known draw count for compute-generated commands.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame or buffer is stale, foreign, or out of range.
    pub fn publish_compute_count(&self, frame: &mut Frame, count: u32) -> Result<()> {
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
    /// Binds a typed structured buffer.
    pub fn structured<T: bytemuck::Pod>(
        name: impl Into<String>,
        buffer: &'a StructuredBuffer<T>,
    ) -> Self {
        Self {
            name: name.into(),
            resource: BindingResource::Structured(&buffer.inner),
        }
    }

    /// Binds an indirect buffer.
    pub fn indirect(name: impl Into<String>, buffer: &'a IndirectBuffer) -> Self {
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

#[derive(Clone, Copy)]
enum FrameTarget {
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
    retained: Vec<Rc<dyn Any>>,
    transients: Vec<Rc<TransientInner>>,
    poison: Option<Error>,
    terminal: bool,
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

    /// Acquires a typed transient structured buffer.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame, count, type size, or allocation is invalid.
    pub fn acquire_structured<T: bytemuck::Pod>(
        &mut self,
        element_count: usize,
    ) -> Result<StructuredBuffer<T>> {
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
        Ok(StructuredBuffer {
            inner,
            marker: PhantomData,
        })
    }

    /// Acquires a transient indexed-indirect buffer.
    ///
    /// # Errors
    /// Returns [`Error`] when the frame, capacity, or allocation is invalid.
    pub fn acquire_indirect(&mut self, capacity: u32) -> Result<IndirectBuffer> {
        if let Some(error) = self.poison {
            return Err(error);
        }
        let handle = match state::acquire_indirect(self.context.handle, capacity) {
            Ok(handle) => handle,
            Err(error) => return self.fail(error),
        };
        let inner = Rc::new(TransientInner {
            context: Rc::clone(&self.context),
            handle: TransientHandle::Indirect(handle),
            state: Cell::new(TransientState::Live),
        });
        self.transients.push(Rc::clone(&inner));
        Ok(IndirectBuffer { inner })
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
                    self.retain(inner);
                    ResourceIdentity::RenderTarget(inner.handle)
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
        indirect: &IndirectBuffer,
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
    /// # Errors
    /// Returns [`Error`] when the frame or texture is stale, foreign, or not ready.
    pub fn enqueue_readback(&mut self, texture: &Texture) -> Result<()> {
        self.ensure_context(&texture.inner.context)?;
        self.retain(&texture.inner);
        self.record(|context| state::frame_enqueue_readback(context, texture.inner.handle))
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
    pub fn retain_vertex_allocation(&mut self, allocation: &VertexAllocation) -> Result<()> {
        self.ensure_context(&allocation.inner.heap.context)?;
        self.retain(&allocation.inner);
        Ok(())
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

    /// Submits and, for surface frames, presents this transaction.
    ///
    /// A prior recording error poisons the transaction: this method aborts and
    /// returns that exact error without attempting submission.
    ///
    /// # Errors
    /// Returns the exact first recording, submission, or presentation error.
    pub fn finish(mut self) -> Result<()> {
        if let Some(error) = self.poison {
            let _ = state::frame_abort(self.context.handle);
            self.release_aborted_transients();
            self.terminal = true;
            return Err(error);
        }

        match state::frame_submit(self.context.handle) {
            Ok(()) => self.consume_transients(),
            Err(error) => {
                self.release_aborted_transients();
                self.terminal = true;
                return Err(error);
            }
        }
        self.terminal = true;
        match self.target {
            FrameTarget::Surface => state::present(self.context.handle),
            FrameTarget::RenderTarget => Ok(()),
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

/// Begins an explicit surface recording transaction.
///
/// # Errors
/// Returns [`Error`] when ownership, readiness, or concurrent recording validation fails.
pub fn begin_frame(context: &Context, surface: &Surface) -> Result<Frame> {
    if !Rc::ptr_eq(&context.inner, &surface.inner.context) {
        return Err(Error::InvalidContext);
    }
    state::begin_render(context.raw(), surface.inner.handle)?;
    let target_lease: Rc<dyn Any> = surface.inner.clone();
    Ok(Frame {
        context: Rc::clone(&context.inner),
        target: FrameTarget::Surface,
        retained: vec![target_lease],
        transients: Vec::new(),
        poison: None,
        terminal: false,
    })
}

/// Begins an explicit managed render-target transaction.
///
/// # Errors
/// Returns [`Error`] when ownership, target support, or concurrent recording validation fails.
pub fn begin_render_target_frame(context: &Context, target: &RenderTarget) -> Result<Frame> {
    if !Rc::ptr_eq(&context.inner, &target.inner.context) {
        return Err(Error::InvalidContext);
    }
    state::begin_render_target(context.raw(), target.inner.handle)?;
    let target_lease: Rc<dyn Any> = target.inner.clone();
    Ok(Frame {
        context: Rc::clone(&context.inner),
        target: FrameTarget::RenderTarget,
        retained: vec![target_lease],
        transients: Vec::new(),
        poison: None,
        terminal: false,
    })
}

#[cfg(test)]
#[path = "api_tests.rs"]
mod tests;
