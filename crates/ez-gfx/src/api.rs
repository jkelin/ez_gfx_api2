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
        BufferHandle, ContextHandle, CounterBufferHandle, IndexAllocationHandle,
        RenderTargetHandle, ShaderHandle, SurfaceHandle, TextureHandle, VertexAllocationHandle,
        VertexHeapHandle,
    },
};
use ez_gfx_runtime::{
    LifecycleError,
    binding::{PublicBinding as RawBinding, ResourceIdentity},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

trait SurfaceHost: HasWindowHandle + HasDisplayHandle {}

impl<T> SurfaceHost for T where T: HasWindowHandle + HasDisplayHandle {}

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
    /// Native teardown was abandoned because submitted work could not be proven complete.
    #[error("native teardown abandoned; borrowed host handles must remain alive")]
    TeardownAbandoned,
    /// Preserves a lifecycle or handle-validation cause.
    #[error(transparent)]
    Lifecycle(#[from] LifecycleError),
    /// Preserves an adapter capability cause.
    #[error(transparent)]
    Capability(#[from] CapabilityError),
}

/// Result returned by the safe Rust facade.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Opaque owner-and-generation identity of one submitted readback request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReadbackId {
    owner: u64,
    generation: u64,
}

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
    /// Completed requested readback; bytes are valid only for this callback.
    Readback {
        /// Request identity.
        request: ReadbackId,
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
        /// Tightly packed RGBA8 pixels.
        bytes: &'a [u8],
    },
    /// Completed implicit terminal-frame snapshot.
    Snapshot(&'a [u8]),
}

type EventCallback = dyn for<'a> FnMut(Event<'a>);
#[derive(Clone, Copy)]
struct CachedRenderTarget {
    handle: RenderTargetHandle,
    format: ez_gfx_runtime::target::Format,
    extent: (u32, u32),
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ContextTeardownDisposition {
    Live,
    Released,
    Unproven,
}

fn terminal_teardown_disposition(result: &Result<()>) -> ContextTeardownDisposition {
    if matches!(result, Err(Error::TeardownAbandoned)) {
        ContextTeardownDisposition::Unproven
    } else {
        // Every other result is returned only after terminal cleanup released native ownership.
        ContextTeardownDisposition::Released
    }
}

struct ContextInner {
    handle: ContextHandle,
    callback: RefCell<Option<Box<EventCallback>>>,
    dispatching: Cell<bool>,
    render_targets: RefCell<HashMap<String, CachedRenderTarget>>,
    closed: Cell<bool>,
    next_readback: Cell<u64>,
    teardown: Cell<ContextTeardownDisposition>,
}

struct DispatchGuard<'a>(&'a Cell<bool>);

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.closed.replace(true) {
            // TLS state may already have run its own destructor during thread exit.
            let result = state::drop_context(self.handle);
            self.teardown.set(terminal_teardown_disposition(&result));
        }
    }
}

/// Creator-thread graphics context.
///
/// The `Rc` ownership marker deliberately makes this type `!Send` and `!Sync`.
/// Explicit destruction and owner drop invalidate every context-owned resource.
pub struct Context {
    inner: Rc<ContextInner>,
    owner: bool,
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.owner && !self.inner.closed.replace(true) {
            // Drop cannot report native teardown failures; `destroy` remains the
            // deterministic path when the caller needs the result.
            let result = state::drop_context(self.inner.handle);
            self.inner
                .teardown
                .set(terminal_teardown_disposition(&result));
        }
    }
}

impl Context {
    /// Creates a creator-thread graphics context.
    ///
    /// # Errors
    /// Returns [`Error`] when native context creation or validation fails.
    pub fn new(options: ez_gfx_runtime::ContextOptions) -> Result<Self> {
        state::create_context(options).map(|handle| Self {
            inner: Rc::new(ContextInner {
                handle,
                callback: RefCell::new(None),
                render_targets: RefCell::new(HashMap::new()),
                dispatching: Cell::new(false),
                next_readback: Cell::new(1),
                closed: Cell::new(false),
                teardown: Cell::new(ContextTeardownDisposition::Live),
            }),
            owner: true,
        })
    }

    /// Registers one process-wide custom texture decoder.
    ///
    /// # Errors
    /// Returns [`TextureError`] when `source_format` is reserved or already registered.
    pub fn register_texture_decoder(
        source_format: u8,
        decoder: ez_gfx_runtime::texture::TextureDecodeCallback,
    ) -> std::result::Result<(), ez_gfx_runtime::texture::TextureError> {
        ez_gfx_runtime::texture::register_texture_decoder(source_format, decoder)
    }

    /// Removes one process-wide custom texture decoder.
    ///
    /// Already-admitted texture loads retain their decoder.
    ///
    /// # Errors
    /// Returns [`TextureError`] when `source_format` is reserved or not registered.
    pub fn unregister_texture_decoder(
        source_format: u8,
    ) -> std::result::Result<(), ez_gfx_runtime::texture::TextureError> {
        ez_gfx_runtime::texture::unregister_texture_decoder(source_format)
    }
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
        let mut events: Vec<Event<'static>> = Vec::new();

        // Drain raw queues into owned records before invoking user code. Each poll
        // releases the context TLS borrow before any callback can run.
        for _ in 0..4096 {
            let upload = state::poll_upload_event(self.raw())?.map(Event::Upload);
            let (runtime, runtime_dropped) = state::poll_runtime_event(self.raw())?;
            let (diagnostic, diagnostic_dropped) = state::poll_diagnostic(self.raw())?;
            let dropped = runtime_dropped.saturating_add(diagnostic_dropped);
            let pending = [
                upload,
                runtime.map(Event::Runtime),
                diagnostic.map(|(level, record)| Event::Diagnostic { level, record }),
                (dropped != 0).then_some(Event::ObservationsDropped(dropped)),
            ];
            if pending.iter().all(Option::is_none) {
                break;
            }
            events.extend(pending.into_iter().flatten());
        }

        let mut callback_panicked = false;
        for event in events {
            let mut slot = self.inner.callback.borrow_mut();
            let Some(callback) = slot.as_mut() else {
                continue;
            };
            if catch_unwind(AssertUnwindSafe(|| callback(event))).is_err() {
                *slot = None;
                callback_panicked = true;
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

    fn dispatch_readback(
        &self,
        request: ReadbackId,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> Result<()> {
        self.dispatch_callback(Event::Readback {
            request,
            width,
            height,
            bytes,
        })
    }

    fn dispatch_snapshot(&self, bytes: &[u8]) -> Result<()> {
        self.dispatch_callback(Event::Snapshot(bytes))
    }

    fn dispatch_callback(&self, event: Event<'_>) -> Result<()> {
        if self.inner.callback.borrow().is_none() {
            return Ok(());
        }
        if self.inner.dispatching.replace(true) {
            return Err(Error::ReentrantCallback);
        }
        let dispatch_guard = DispatchGuard(&self.inner.dispatching);
        let mut slot = self.inner.callback.borrow_mut();
        let result = if let Some(callback) = slot.as_mut() {
            catch_unwind(AssertUnwindSafe(|| callback(event)))
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

    /// Deterministically destroys this context and every resource it owns.
    ///
    /// All outstanding resource wrappers become stale.
    ///
    /// # Errors
    /// Returns [`Error`] when teardown or pending event dispatch fails.
    pub fn destroy(mut self) -> Result<()> {
        self.check_entry()?;
        state::validate_context_owner(self.inner.handle)?;
        if self.inner.closed.replace(true) {
            return Err(Error::InvalidContext);
        }
        self.owner = false;
        let result = state::destroy_context(self.inner.handle);
        self.inner
            .teardown
            .set(terminal_teardown_disposition(&result));
        result
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
    // Keep the host alive until backend surface destruction releases every borrowed handle.
    host: Option<Box<dyn SurfaceHost>>,
}

fn failed_window_surface_creation<W>(host: W, error: Error) -> Error {
    if error == Error::TeardownAbandoned {
        // Failed native rollback may still borrow the host, so it must share the native leak.
        core::mem::forget(host);
    }
    error
}

fn teardown_owned_host<T>(host: &mut Option<T>, destroy_native: impl FnOnce() -> Result<()>) {
    match destroy_native() {
        Ok(()) => {
            // Native handles are gone, so releasing the owner is now safe.
            drop(host.take());
        }
        Err(Error::TeardownAbandoned) => {
            if let Some(host) = host.take() {
                // Abandoned native objects still borrow this host and intentionally share their leak.
                core::mem::forget(host);
            }
        }
        Err(_) => {
            if let Some(host) = host.take() {
                // A valid safe surface cannot become stale or wrong-thread during drop. If that
                // invariant is ever broken, preserve the possibly borrowed host conservatively.
                core::mem::forget(host);
            }
        }
    }
}

impl Drop for SurfaceInner {
    fn drop(&mut self) {
        match self.context.teardown.get() {
            ContextTeardownDisposition::Live => teardown_owned_host(&mut self.host, || {
                state::destroy_surface(self.context.handle, self.handle)
            }),
            ContextTeardownDisposition::Released => {
                teardown_owned_host(&mut self.host, || Ok(()));
            }
            ContextTeardownDisposition::Unproven => {
                teardown_owned_host(&mut self.host, || Err(Error::TeardownAbandoned));
            }
        }
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
            owner: false,
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
            owner: false,
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
            owner: false,
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
            owner: false,
        };
        context.check_entry()?;
        context.complete(state::set_snapshot_cache(
            self.inner.context.handle,
            self.inner.handle,
            enabled,
        ))
    }
}

impl Context {
    /// Creates and initializes a window surface from `raw-window-handle`.
    ///
    /// An authoritative native extent is queried after device initialization. Window systems that
    /// report a host-managed extent, notably Wayland, publish the surface unready until the
    /// toolkit-neutral caller forwards its configure-event dimensions through [`Surface::resize`].
    /// Callers never provide dimensions in the creation descriptor.
    ///
    /// # Errors
    /// Returns [`Error`] when the handle is unavailable or unsupported, native creation or device
    /// initialization fails, or the authoritative native extent is minimized. If post-native
    /// publication rollback returns [`Error::TeardownAbandoned`], the host is intentionally retained
    /// for the process lifetime.
    pub fn create_surface_window<W>(
        &self,
        host: W,
        cache_presented_snapshots: bool,
    ) -> Result<Surface>
    where
        W: HasWindowHandle + HasDisplayHandle + 'static,
    {
        self.check_entry()?;
        let display = host
            .display_handle()
            .map_err(|_| Error::InvalidArgument)?
            .as_raw();
        let window = host
            .window_handle()
            .map_err(|_| Error::InvalidArgument)?
            .as_raw();
        let handle = match state::create_surface_window(
            self.raw(),
            state::SurfaceWindow { display, window },
            cache_presented_snapshots,
        ) {
            Ok(handle) => handle,
            Err(error) => return Err(failed_window_surface_creation(host, error)),
        };
        let initialized = state::init_device(self.raw(), handle)
            .and_then(|()| state::sync_window_surface_extent(self.raw(), handle));
        self.publish_surface(handle, initialized, Some(Box::new(host)))
    }

    /// Creates, initializes, and sizes a headless surface atomically.
    ///
    /// # Errors
    /// Returns [`Error`] when creation, device initialization, or sizing fails.
    pub fn create_surface_headless(
        &self,
        options: ez_gfx_runtime::HeadlessSurfaceOptions,
    ) -> Result<Surface> {
        self.check_entry()?;
        let handle = state::create_surface_headless(self.raw(), options)?;
        let initialized = state::init_device(self.raw(), handle);
        self.publish_surface(handle, initialized, None)
    }

    fn publish_surface(
        &self,
        handle: SurfaceHandle,
        initialized: Result<()>,
        mut host: Option<Box<dyn SurfaceHost>>,
    ) -> Result<Surface> {
        if let Err(error) = initialized {
            // Preserve the initialization failure while disposing of the host according to the
            // typed native teardown result.
            teardown_owned_host(&mut host, || state::destroy_surface(self.raw(), handle));
            return Err(error);
        }
        self.complete(Ok(Surface {
            inner: Rc::new(SurfaceInner {
                context: Rc::clone(&self.inner),
                handle,
                target: RefCell::new(None),
                host,
            }),
        }))
    }
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
            owner: false,
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
            owner: false,
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
            owner: false,
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
            owner: false,
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
            owner: false,
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
    surface_lease: Option<Rc<SurfaceInner>>,
}

impl RenderTarget {
    /// Returns the resolved target format.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn format(&self) -> Result<ez_gfx_runtime::target::Format> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
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
            owner: false,
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
            owner: false,
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

    /// Creates and schedules a unique readback request on `frame`.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale, foreign, already released, or not renderable.
    pub fn prepare_readback(&self, frame: &mut Frame) -> Result<Readback> {
        frame.prepare_target_readback(self)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ReadbackState {
    Queued,
    Complete,
    Aborted,
}

struct ReadbackInner {
    _context: Rc<ContextInner>,
    _target: Rc<RenderTargetInner>,
    id: ReadbackId,
    width: u32,
    height: u32,
    state: Cell<ReadbackState>,
}

/// Owning identity and target lease for one submitted readback.
pub struct Readback {
    inner: Rc<ReadbackInner>,
}

impl Readback {
    /// Returns the opaque request identity delivered with its callback.
    pub fn id(&self) -> ReadbackId {
        self.inner.id
    }
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
            owner: false,
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
            owner: false,
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
            owner: false,
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
