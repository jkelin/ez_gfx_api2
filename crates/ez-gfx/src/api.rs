use std::{
    any::Any,
    cell::{Cell, RefCell},
    collections::HashMap,
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
};

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
    /// A callback attempted a graphics operation recursively.
    #[error("graphics operations are unavailable during event callback dispatch")]
    ReentrantCallback,
    /// A registered event callback panicked and was removed.
    #[error("graphics event callback panicked")]
    CallbackPanicked,
    /// Preserves a lifecycle or handle-validation cause.
    #[error(transparent)]
    Lifecycle(#[from] LifecycleError),
    /// Preserves an adapter capability cause.
    #[error(transparent)]
    Capability(#[from] CapabilityError),
}

/// Result returned by the safe Rust facade.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Creator-thread event delivered by [`Context::register_callback`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Event<'a> {
    /// One asynchronous upload state transition.
    Upload(ez_gfx_runtime::upload::UploadEvent),
    /// One runtime observation.
    Runtime(ez_gfx_runtime::observability::RuntimeRecord),
    /// One diagnostic observation.
    Diagnostic {
        /// Diagnostic severity.
        level: ez_gfx_runtime::observability::DiagnosticLevel,
        /// Runtime operation that produced the diagnostic.
        record: ez_gfx_runtime::observability::RuntimeRecord,
    },
    /// Bounded observability storage discarded records.
    ObservationsDropped(u64),
    /// Completed host-visible readback bytes, valid only for this callback.
    Readback(&'a [u8]),
}

type EventCallback = dyn for<'a> FnMut(Event<'a>);
#[derive(Clone, Copy)]
struct CachedRenderTarget {
    handle: RenderTargetHandle,
    format: ez_gfx_runtime::target::Format,
    extent: (u32, u32),
}

struct ContextInner {
    handle: ContextHandle,
    callback: RefCell<Option<Box<EventCallback>>>,
    dispatching: Cell<bool>,
    render_targets: RefCell<HashMap<String, CachedRenderTarget>>,
    closed: Cell<bool>,
}

struct DispatchGuard<'a>(&'a Cell<bool>);

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.closed.get() {
            state::abandon_context(self.handle);
        }
    }
}

/// Creator-thread graphics context.
///
/// The `Rc` ownership marker deliberately makes this type `!Send` and `!Sync`.
/// Dropping the last lease abandons native state without waiting; call [`Context::close`] for deterministic teardown.
#[derive(Clone)]
pub struct Context {
    inner: Rc<ContextInner>,
}

/// Creates a creator-thread graphics context.
///
/// # Errors
/// Returns [`Error`] when native context creation or validation fails.
pub fn create_context(options: ez_gfx_runtime::ContextOptions) -> Result<Context> {
    state::create_context(options).map(|handle| Context {
        inner: Rc::new(ContextInner {
            handle,
            callback: RefCell::new(None),
            render_targets: RefCell::new(HashMap::new()),
            dispatching: Cell::new(false),
            closed: Cell::new(false),
        }),
    })
}

impl Context {
    pub(crate) fn raw(&self) -> ContextHandle {
        self.inner.handle
    }

    fn check_entry(&self) -> Result<()> {
        if self.inner.dispatching.get() {
            Err(Error::ReentrantCallback)
        } else {
            Ok(())
        }
    }

    fn dispatch_events(&self) -> Result<()> {
        if self.inner.callback.borrow().is_none() {
            return Ok(());
        }
        if self.inner.dispatching.replace(true) {
            return Err(Error::ReentrantCallback);
        }
        let dispatch_guard = DispatchGuard(&self.inner.dispatching);

        let mut callback_panicked = false;
        for _ in 0..4096 {
            let upload = state::poll_upload_event(self.raw())?.map(Event::Upload);
            let (runtime, runtime_dropped) = state::poll_runtime_event(self.raw())?;
            let (diagnostic, diagnostic_dropped) = state::poll_diagnostic(self.raw())?;
            let dropped = runtime_dropped.saturating_add(diagnostic_dropped);
            let mut events = [
                upload,
                runtime.map(Event::Runtime),
                diagnostic.map(|(level, record)| Event::Diagnostic { level, record }),
                (dropped != 0).then_some(Event::ObservationsDropped(dropped)),
            ];
            if events.iter().all(Option::is_none) {
                break;
            }
            for event in events.iter_mut().filter_map(Option::take) {
                let mut slot = self.inner.callback.borrow_mut();
                let Some(callback) = slot.as_mut() else {
                    continue;
                };
                if catch_unwind(AssertUnwindSafe(|| callback(event))).is_err() {
                    *slot = None;
                    callback_panicked = true;
                }
            }
        }
        drop(dispatch_guard);
        if callback_panicked {
            Err(Error::CallbackPanicked)
        } else {
            Ok(())
        }
    }

    fn complete<T>(&self, result: Result<T>) -> Result<T> {
        let dispatch = self.dispatch_events();
        match result {
            Err(error) => Err(error),
            Ok(value) => dispatch.map(|()| value),
        }
    }

    fn dispatch_readback(&self, bytes: &[u8]) -> Result<()> {
        if self.inner.callback.borrow().is_none() {
            return Ok(());
        }
        if self.inner.dispatching.replace(true) {
            return Err(Error::ReentrantCallback);
        }
        let dispatch_guard = DispatchGuard(&self.inner.dispatching);
        let mut slot = self.inner.callback.borrow_mut();
        let result = if let Some(callback) = slot.as_mut() {
            catch_unwind(AssertUnwindSafe(|| callback(Event::Readback(bytes))))
        } else {
            Ok(())
        };
        if result.is_err() {
            *slot = None;
        }
        drop(slot);
        drop(dispatch_guard);
        result.map_err(|_| Error::CallbackPanicked)
    }

    /// Replaces the creator-thread event callback.
    ///
    /// Callback-scoped borrowed data is invalid after the callback returns.
    ///
    /// # Errors
    /// Returns [`Error::ReentrantCallback`] during callback dispatch or a
    /// callback error raised while delivering already-pending events.
    pub fn register_callback(
        &self,
        callback: impl for<'a> FnMut(Event<'a>) + 'static,
    ) -> Result<()> {
        self.check_entry()?;
        *self.inner.callback.borrow_mut() = Some(Box::new(callback));
        self.dispatch_events()
    }

    /// Waits for submitted work and dispatches pending events.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale, unhealthy, or native waiting fails.
    pub fn wait_idle(&self) -> Result<()> {
        self.check_entry()?;
        self.complete(state::wait_idle(self.raw()))
    }

    /// Returns the configured asynchronous texture worker count.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale.
    pub fn texture_decode_worker_count(&self) -> Result<u32> {
        self.check_entry()?;
        self.complete(state::texture_decode_worker_count(self.raw()))
    }

    /// Returns context-wide texture upload counters.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is stale or unhealthy.
    pub fn texture_upload_telemetry(
        &self,
    ) -> Result<ez_gfx_runtime::texture::TextureUploadTelemetrySnapshot> {
        self.check_entry()?;
        self.complete(state::texture_upload_telemetry(self.raw()))
    }

    /// Deterministically destroys a uniquely owned context.
    ///
    /// # Errors
    /// Returns `(self, Error::NotReady)` while child resource leases remain.
    /// Native teardown errors are terminal and therefore do not return ownership.
    pub fn close(self) -> std::result::Result<(), (Option<Self>, Error)> {
        self.check_entry()
            .map_err(|error| (Some(self.clone()), error))?;
        let inner = match Rc::try_unwrap(self.inner) {
            Ok(inner) => inner,
            Err(inner) => return Err((Some(Self { inner }), Error::NotReady)),
        };
        inner.closed.set(true);
        state::destroy_context(inner.handle).map_err(|error| (None, error))
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
}

struct SurfaceInner {
    context: Rc<ContextInner>,
    handle: SurfaceHandle,
    target: RefCell<Option<Rc<RenderTargetInner>>>,
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
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::resize_surface(
            self.inner.context.handle,
            self.inner.handle,
            width,
            height,
        ))
    }

    /// Returns the current nonzero surface extent.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale or has no ready extent.
    pub fn extent(&self) -> Result<(u32, u32)> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::surface_extent(
            self.inner.context.handle,
            self.inner.handle,
        ))
    }

    /// Reports whether a resize still awaits the next render.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale.
    pub fn resize_pending(&self) -> Result<bool> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::surface_resize_pending(
            self.inner.context.handle,
            self.inner.handle,
        ))
    }

    /// Enables or disables presented snapshot caching.
    ///
    /// # Errors
    /// Returns [`Error`] when the surface is stale.
    pub fn set_snapshot_cache(&self, enabled: bool) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::set_snapshot_cache(
            self.inner.context.handle,
            self.inner.handle,
            enabled,
        ))
    }
}

/// Creates, initializes, and sizes a surface atomically.
///
/// # Errors
/// Returns [`Error`] when creation, device initialization, or initial sizing fails.
pub fn create_surface(
    context: &Context,
    options: ez_gfx_runtime::SurfaceOptions,
) -> Result<Surface> {
    context.check_entry()?;
    let handle = state::create_surface(context.raw(), options)?;
    let initialized = state::init_device(context.raw(), handle)
        .and_then(|()| state::resize_surface(context.raw(), handle, options.width, options.height));
    if let Err(error) = initialized {
        // A partially initialized surface is never published into the owning interface.
        state::destroy_surface(context.raw(), handle);
        return Err(error);
    }
    context.complete(Ok(Surface {
        inner: Rc::new(SurfaceInner {
            context: Rc::clone(&context.inner),
            handle,
            target: RefCell::new(None),
        }),
    }))
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

impl Context {
    /// Loads a validated shader artifact.
    ///
    /// # Errors
    /// Returns [`Error`] when the artifact or native shader is invalid.
    pub fn load_shader(&self, artifact: &[u8]) -> Result<Shader> {
        self.check_entry()?;
        let result = state::load_shader(self.raw(), artifact).map(|handle| Shader {
            inner: Rc::new(ShaderInner {
                context: Rc::clone(&self.inner),
                handle,
            }),
        });
        self.complete(result)
    }
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
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::texture_binding(
            self.inner.context.handle,
            self.inner.handle,
        ))
    }

    /// Returns `(resident_mips, total_mips)`.
    ///
    /// # Errors
    /// Returns [`Error`] when the texture is stale.
    pub fn residency(&self) -> Result<(u32, u32)> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::texture_residency(
            self.inner.context.handle,
            self.inner.handle,
        ))
    }

    /// Changes the desired resident mip prefix.
    ///
    /// # Errors
    /// Returns [`Error`] when the mip request or texture state is invalid.
    pub fn set_residency(&self, resident_mips: u32) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::set_texture_residency(
            self.inner.context.handle,
            self.inner.handle,
            resident_mips,
        ))
    }

    /// Queues an in-place texture region update.
    ///
    /// # Errors
    /// Returns [`Error`] when the region, texture state, or upload fails.
    pub fn update_region(&self, region: ez_gfx_hal::TextureRegion<'_>) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::update_texture_region(
            self.inner.context.handle,
            self.inner.handle,
            region,
        ))
    }

    /// Cancels an asynchronous texture load.
    ///
    /// # Errors
    /// Returns [`Error`] when cancellation or texture ownership validation fails.
    pub fn cancel_load(&self) -> Result<()> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::cancel_texture_load(
            self.inner.context.handle,
            self.inner.handle,
        ))
    }
}

impl Context {
    /// Queues texture decode and upload while owning the copied input.
    ///
    /// # Errors
    /// Returns [`Error`] when input, scheduling, decode admission, or ownership validation fails.
    pub fn load_texture(
        &self,
        source: ez_gfx_runtime::texture::TextureSource,
        bytes: &[u8],
        generate_mips: bool,
        config: &state::TextureConfig,
    ) -> Result<Texture> {
        self.check_entry()?;
        let result =
            state::load_texture(self.raw(), source, bytes, generate_mips, config).map(|handle| {
                Texture {
                    inner: Rc::new(TextureInner {
                        context: Rc::clone(&self.inner),
                        handle,
                    }),
                }
            });
        self.complete(result)
    }
}

enum RenderTargetBacking {
    Managed(String),
    Surface {
        format: ez_gfx_runtime::target::Format,
        extent: (u32, u32),
    },
}

struct RenderTargetInner {
    context: Rc<ContextInner>,
    backing: RenderTargetBacking,
}
impl RenderTargetInner {
    fn managed_handle(&self) -> Result<RenderTargetHandle> {
        let RenderTargetBacking::Managed(name) = &self.backing else {
            return Err(Error::Unsupported);
        };
        self.context
            .render_targets
            .borrow()
            .get(name)
            .map(|target| target.handle)
            .ok_or(Error::InvalidContext)
    }
}

/// Owning logical render target.
pub struct RenderTarget {
    inner: Rc<RenderTargetInner>,
    _surface_lease: Option<Rc<SurfaceInner>>,
}

impl RenderTarget {
    /// Returns the resolved target format.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn format(&self) -> Result<ez_gfx_runtime::target::Format> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(name) => self
                .inner
                .context
                .render_targets
                .borrow()
                .get(name)
                .map(|target| target.format)
                .ok_or(Error::InvalidContext),
            RenderTargetBacking::Surface { format, .. } => Ok(*format),
        }
    }

    /// Returns the target extent.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn extent(&self) -> Result<(u32, u32)> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(name) => self
                .inner
                .context
                .render_targets
                .borrow()
                .get(name)
                .map(|target| target.extent)
                .ok_or(Error::InvalidContext),
            RenderTargetBacking::Surface { extent, .. } => Ok(*extent),
        }
    }

    /// Returns the target clear value.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn clear(&self) -> Result<ez_gfx_runtime::target::ClearValue> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(_) | RenderTargetBacking::Surface { .. } => {
                Ok(ez_gfx_runtime::target::ClearValue::Color([
                    0.1, 0.1, 0.1, 1.0,
                ]))
            }
        }
    }

    /// Creates an opaque request for callback-scoped readback after this target is rendered.
    #[must_use]
    pub fn prepare_readback(&self) -> RenderTargetReadback {
        RenderTargetReadback {
            target: Rc::clone(&self.inner),
        }
    }
}

/// Opaque request to capture a managed render target through the context callback.
pub struct RenderTargetReadback {
    target: Rc<RenderTargetInner>,
}

impl Context {
    /// Probes one render-target format without allocating it.
    ///
    /// # Errors
    /// Returns [`Error`] when the format or sample count is unsupported.
    pub fn probe_render_target_format(
        &self,
        format: ez_gfx_runtime::target::Format,
        samples: u8,
    ) -> Result<()> {
        self.check_entry()?;
        self.complete(state::probe_render_target_format(
            self.raw(),
            format,
            samples,
        ))
    }
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

/// Owning typed auto-growing named vertex heap.
pub struct VertexHeap<T: bytemuck::Pod> {
    inner: Rc<VertexHeapInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> VertexHeap<T> {
    /// Uploads typed vertices and retains this heap.
    ///
    /// Heap storage grows without compacting, so returned logical ranges remain stable.
    ///
    /// # Errors
    /// Returns [`Error`] when the slice, heap ownership, growth, or upload is invalid.
    pub fn upload(&self, vertices: &[T]) -> Result<VertexAllocation<T>> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        let result = state::upload_vertices(self.inner.context.handle, self.inner.handle, vertices)
            .map(|handle| VertexAllocation {
                inner: Rc::new(VertexAllocationInner {
                    heap: Rc::clone(&self.inner),
                    handle,
                }),
                marker: PhantomData,
            });
        context.complete(result)
    }
}

impl Context {
    /// Creates a typed auto-growing named vertex heap.
    ///
    /// # Errors
    /// Returns [`Error`] when the name, element type, or native allocation is invalid.
    pub fn create_vertex_heap<T: bytemuck::Pod>(&self, name: &str) -> Result<VertexHeap<T>> {
        self.check_entry()?;
        let stride =
            u64::try_from(core::mem::size_of::<T>()).map_err(|_| Error::InvalidArgument)?;
        let result = state::create_vertex_heap(self.raw(), name, stride).map(|handle| VertexHeap {
            inner: Rc::new(VertexHeapInner {
                context: Rc::clone(&self.inner),
                handle,
            }),
            marker: PhantomData,
        });
        self.complete(result)
    }

    /// Uploads packed indices to the lazily-created auto-growing index heap.
    ///
    /// # Errors
    /// Returns [`Error`] when the slice, heap growth, or upload is invalid.
    pub fn upload_indices(&self, indices: &[u32]) -> Result<IndexAllocation> {
        self.check_entry()?;
        let result = state::upload_indices(self.raw(), indices).map(|handle| IndexAllocation {
            inner: Rc::new(IndexAllocationInner {
                context: Rc::clone(&self.inner),
                handle,
            }),
        });
        self.complete(result)
    }
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

/// Owning typed vertex allocation that retains its heap.
pub struct VertexAllocation<T: bytemuck::Pod> {
    inner: Rc<VertexAllocationInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> VertexAllocation<T> {
    /// Returns the stable `(first_element, element_count)` logical range.
    ///
    /// # Errors
    /// Returns [`Error`] when the allocation is stale.
    pub fn range(&self) -> Result<(u32, u32)> {
        let context = Context {
            inner: Rc::clone(&self.inner.heap.context),
        };
        context.check_entry()?;
        context.complete(state::vertex_allocation_range(
            self.inner.heap.context.handle,
            self.inner.handle,
        ))
    }
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
    /// Returns the stable `(first_index, index_count)` logical range.
    ///
    /// # Errors
    /// Returns [`Error`] when the allocation is stale.
    pub fn range(&self) -> Result<(u32, u32)> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
        };
        context.check_entry()?;
        context.complete(state::index_allocation_range(
            self.inner.context.handle,
            self.inner.handle,
        ))
    }
}

include!("api_frame.rs");

#[cfg(test)]
#[path = "api_tests.rs"]
mod tests;
