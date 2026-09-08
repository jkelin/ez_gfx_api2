use crate::Result;

use super::{
    AdapterCatalog, AdapterInfo, AdapterReport, AdapterSelection, Arc, AsyncTextureState, Backend,
    CONTEXT_HANDLES, CONTEXTS, ContextHandle, ContextIdentity, ContextOptions, ContextState,
    DiagnosticLevel, Error, FrameRecorder, GeometryManager, HalError, HashMap,
    IndexAllocationHandle, NativeContext, NativeSurface, Observability, Ordering,
    RenderTargetHandle, ResourceKind, RuntimeError, RuntimePhase, RuntimeRecord, RuntimeStatus,
    SurfaceHandle, SurfaceOptions, SurfacePlatform, SurfaceRecord, SurfaceState, TextureRegistry,
    TextureUploadTelemetry, UploadEvent, UploadResource, UploadStatus, VertexAllocationHandle,
    VulkanContext, VulkanPlatform, admission_report, completed_transfer_native, context_local,
    destroy_native_pipeline, destroy_native_shader, destroy_native_texture, free_native_allocation,
    map_allocation, map_hal, map_lifecycle, map_native_loss, map_texture,
    progress_texture_upload_events, pump_async_textures, render_target::destroy_all_render_targets,
    result_status, wait_native_idle, with_context_mut, with_surface_mut,
};
#[cfg(windows)]
use super::{Dx12Context, Dx12Surface};
#[cfg(target_vendor = "apple")]
use super::{MetalContext, MetalSurface};

/// Creates a graphics context.
///
/// # Errors
///
/// Returns an error when the backend/platform pair is unsupported or native context creation fails.
pub fn create_context(options: ContextOptions) -> Result<ContextHandle> {
    let native = match options.adapter_selection {
        Some(selection) => build_native_context_for_adapter(&options, selection)?,
        None => build_native_context(&options)?,
    };
    let texture_registry = TextureRegistry::new(
        ez_gfx_runtime::binding::MAX_TEXTURE_HEAP_CAPACITY,
        ez_gfx_runtime::binding::MAX_TEXTURE_HEAP_CAPACITY,
    )
    .map_err(|_| Error::NativeFailure)?;
    let frame = FrameRecorder::new(1024).map_err(|_| Error::NativeFailure)?;
    let observability = Observability::new(1024, 256).map_err(|_| Error::NativeFailure)?;
    let async_textures = AsyncTextureState::new_with_workers(options.texture_decode_workers)?;
    let local = CONTEXT_HANDLES
        .lock()
        .map_err(|_| Error::NativeFailure)?
        .insert(())
        .map_err(|_| Error::NativeFailure)?;
    let Ok(identity) = ContextIdentity::new(local) else {
        if let Ok(mut handles) = CONTEXT_HANDLES.lock() {
            let _ = handles.remove(local);
        }
        return Err(Error::NativeFailure);
    };
    let handle = identity.context_handle();
    let state = ContextState {
        identity,
        options,
        native,
        surfaces: HashMap::new(),
        allocations: HashMap::new(),
        allocation_ready: HashMap::new(),
        shaders: HashMap::new(),
        textures: HashMap::new(),
        render_targets: HashMap::new(),
        texture_formats: HashMap::new(),
        texture_published_mips: HashMap::new(),
        texture_residency_targets: HashMap::new(),
        texture_last_transfer: HashMap::new(),
        retired_textures: Vec::new(),
        pipelines: HashMap::new(),
        graphics_format: None,
        indirects: HashMap::new(),
        transient_buffers: HashMap::new(),
        structured_pool: HashMap::new(),
        indirect_pool: ez_gfx_hal::ReusableStagingPool::new(256),
        texture_registry,
        texture_ready: HashMap::new(),
        pending_textures: HashMap::new(),
        texture_handoffs: HashMap::new(),
        texture_telemetry: Arc::new(TextureUploadTelemetry::default()),
        async_textures,
        texture_failures: HashMap::new(),
        geometry: GeometryManager::new(),
        vertex_heaps: HashMap::new(),
        vertex_heap_handles: HashMap::new(),
        next_vertex_heap_id: 0,
        retired_geometry_ranges: Vec::new(),
        retired_vertex_heaps: Vec::new(),
        index_heap: None,
        retired_geometry: Vec::new(),
        geometry_uploads: HashMap::new(),
        geometry_last_transfer: HashMap::new(),
        upload_events: ez_gfx_runtime::upload::UploadEventQueue::new(),
        staging: ez_gfx_hal::ReusableStagingPool::new(256),
        frame_vertex_heaps: HashMap::new(),
        frame_serial: 0,
        frame,
        frame_resources: HashMap::new(),
        frame_native_resources: HashMap::new(),
        frame_index: None,
        frame_surface: None,
        frame_depth: None,
        frame_has_graphics: false,
        frame_render_target: None,
        last_readbacks: Vec::new(),
        active_surface: None,
        frame_presented: false,
        observability,
    };
    let inserted = CONTEXTS.with(|contexts| {
        let mut contexts = contexts
            .try_borrow_mut()
            .map_err(|_| Error::NativeFailure)?;
        if contexts.states.contains_key(&local) {
            return Err(Error::NativeFailure);
        }
        contexts.states.insert(local, state);
        Ok(())
    });
    if let Err(error) = inserted {
        if let Ok(mut handles) = CONTEXT_HANDLES.lock() {
            let _ = handles.remove(local);
        }
        return Err(error);
    }
    Ok(handle)
}

/// Builds the backend-native context with legacy first-fit device selection.
///
/// # Errors
///
/// Returns an error when the backend/platform pair is unsupported or native context creation fails.
fn build_native_context(options: &ContextOptions) -> Result<NativeContext> {
    Ok(match options.backend {
        Backend::Vulkan if options.surface_platform == SurfacePlatform::Win32 => {
            NativeContext::Vulkan(Box::new(
                VulkanContext::create(
                    options.enable_debug,
                    options.enable_validation,
                    VulkanPlatform::Win32,
                )
                .map_err(map_hal)?,
            ))
        }
        Backend::Vulkan if options.surface_platform == SurfacePlatform::Headless => {
            NativeContext::Vulkan(Box::new(
                VulkanContext::create(
                    options.enable_debug,
                    options.enable_validation,
                    VulkanPlatform::Headless,
                )
                .map_err(map_hal)?,
            ))
        }
        Backend::Vulkan => return Err(Error::Unsupported),
        Backend::Dx12 => {
            if options.surface_platform != SurfacePlatform::Win32 {
                return Err(Error::InvalidArgument);
            }
            #[cfg(windows)]
            {
                NativeContext::Dx12(Box::new(
                    Dx12Context::create_default(false).map_err(|_| Error::NativeFailure)?,
                ))
            }
            #[cfg(not(windows))]
            {
                return Err(Error::Unsupported);
            }
        }
        Backend::Metal => {
            if options.surface_platform != SurfacePlatform::MetalLayer {
                return Err(Error::InvalidArgument);
            }
            #[cfg(target_vendor = "apple")]
            {
                NativeContext::Metal(Box::new(MetalContext::create_default().map_err(map_hal)?))
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                return Err(Error::Unsupported);
            }
        }
    })
}

/// Builds the backend-native context for one explicitly selected adapter.
///
/// The request bypasses ranking but never bypasses admission: the catalog
/// resolves the stable identity first, so unknown identities fail
/// `InvalidArgument` before any native call. Vulkan creates its instance here
/// and enforces the selection at device initialization; DX12 and Metal create
/// the selected device directly.
///
/// # Errors
///
/// Returns an error when the backend/platform pair is unsupported, the
/// stable identity is unknown or rejected by admission policy, enumeration
/// fails, or native context creation fails.
fn build_native_context_for_adapter(
    options: &ContextOptions,
    selection: AdapterSelection,
) -> Result<NativeContext> {
    Ok(match options.backend {
        Backend::Vulkan if options.surface_platform == SurfacePlatform::Win32 => {
            let catalog =
                AdapterCatalog::new(backend_adapters(Backend::Vulkan)).map_err(map_runtime)?;
            catalog
                .select(selection.stable_id, selection.allow_software)
                .map_err(map_runtime)?;
            NativeContext::Vulkan(Box::new(
                VulkanContext::create(
                    options.enable_debug,
                    options.enable_validation,
                    VulkanPlatform::Win32,
                )
                .map_err(map_hal)?,
            ))
        }
        Backend::Vulkan if options.surface_platform == SurfacePlatform::Headless => {
            let catalog =
                AdapterCatalog::new(backend_adapters(Backend::Vulkan)).map_err(map_runtime)?;
            catalog
                .select(selection.stable_id, selection.allow_software)
                .map_err(map_runtime)?;
            NativeContext::Vulkan(Box::new(
                VulkanContext::create(
                    options.enable_debug,
                    options.enable_validation,
                    VulkanPlatform::Headless,
                )
                .map_err(map_hal)?,
            ))
        }
        Backend::Vulkan => return Err(Error::Unsupported),
        Backend::Dx12 => {
            if options.surface_platform != SurfacePlatform::Win32 {
                return Err(Error::InvalidArgument);
            }
            #[cfg(windows)]
            {
                let catalog =
                    AdapterCatalog::new(backend_adapters(Backend::Dx12)).map_err(map_runtime)?;
                catalog
                    .select(selection.stable_id, selection.allow_software)
                    .map_err(map_runtime)?;
                NativeContext::Dx12(Box::new(
                    Dx12Context::create_for_adapter(selection.stable_id, selection.allow_software)
                        .map_err(map_hal)?,
                ))
            }
            #[cfg(not(windows))]
            {
                return Err(Error::Unsupported);
            }
        }
        Backend::Metal => {
            if options.surface_platform != SurfacePlatform::MetalLayer {
                return Err(Error::InvalidArgument);
            }
            #[cfg(target_vendor = "apple")]
            {
                let catalog =
                    AdapterCatalog::new(backend_adapters(Backend::Metal)).map_err(map_runtime)?;
                catalog
                    .select(selection.stable_id, selection.allow_software)
                    .map_err(map_runtime)?;
                NativeContext::Metal(Box::new(
                    MetalContext::create_for_adapter(selection.stable_id).map_err(map_hal)?,
                ))
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                return Err(Error::Unsupported);
            }
        }
    })
}

/// Enumerates adapters for one backend. Backends that fail discovery
/// contribute nothing; cross-backend enumeration never fails because of one
/// missing loader.
fn backend_adapters(backend: Backend) -> Vec<AdapterInfo> {
    match backend {
        Backend::Vulkan => VulkanContext::enumerate_adapters().unwrap_or_default(),
        #[cfg(windows)]
        Backend::Dx12 => Dx12Context::enumerate_adapters().unwrap_or_default(),
        #[cfg(not(windows))]
        Backend::Dx12 => Vec::new(),
        #[cfg(target_vendor = "apple")]
        Backend::Metal => MetalContext::enumerate_adapters().unwrap_or_default(),
        #[cfg(not(target_vendor = "apple"))]
        Backend::Metal => Vec::new(),
    }
}

/// Maps catalog admission failures to safe results. Unknown identities and
/// disallowed software are caller errors; inadmissible hardware is
/// unsupported. A duplicate identity across backends is a native defect.
fn map_runtime(error: RuntimeError) -> Error {
    match error {
        RuntimeError::AdapterNotFound | RuntimeError::SoftwareAdapterNotAllowed => {
            Error::InvalidArgument
        }
        RuntimeError::UnsupportedAdapter | RuntimeError::NoAdmittedAdapter => Error::Unsupported,
        RuntimeError::DuplicateAdapterIdentity => Error::NativeFailure,
    }
}

/// Enumerates every adapter visible to the supported backends in backend
/// order (Vulkan, DX12, Metal). Backends that fail discovery contribute
/// nothing. Admitted or not, every entry carries its normalized limits so
/// [`query_adapter_report`] can diagnose rejections.
#[must_use]
pub fn enumerate_adapters() -> Vec<AdapterInfo> {
    let mut adapters = backend_adapters(Backend::Vulkan);
    adapters.extend(backend_adapters(Backend::Dx12));
    adapters.extend(backend_adapters(Backend::Metal));
    adapters
}

/// Diagnoses every enumerated adapter against admission policy without
/// ranking or creating anything. Reports with empty errors and no software
/// rejection are admissible; all others name the exact unmet requirements.
#[must_use]
pub fn query_adapter_report(allow_software: bool) -> Vec<AdapterReport> {
    enumerate_adapters()
        .iter()
        .map(|info| admission_report(info, allow_software))
        .collect()
}

pub(super) fn runtime_status(result: Result<()>) -> RuntimeStatus {
    match result {
        Ok(()) => RuntimeStatus::Ok,
        Err(Error::InvalidArgument | Error::InvalidContext | Error::Lifecycle(_)) => {
            RuntimeStatus::InvalidArgument
        }
        Err(Error::NotReady | Error::QueueFull) => RuntimeStatus::NotReady,
        Err(Error::Cancelled) => RuntimeStatus::Cancelled,
        Err(Error::Unsupported | Error::Capability(_)) => RuntimeStatus::Unsupported,
        Err(Error::NativeFailure | Error::ReentrantCallback | Error::CallbackPanicked) => {
            RuntimeStatus::NativeFailure
        }
        Err(Error::DeviceLost) => RuntimeStatus::DeviceLost,
    }
}

pub(super) fn runtime_record(
    context: &mut ContextState,
    resource: u64,
    phase: RuntimePhase,
    result: Result<()>,
) -> RuntimeRecord {
    RuntimeRecord {
        correlation_id: context.observability.next_correlation(),
        resource,
        backend: context.options.backend,
        phase,
        status: runtime_status(result),
    }
}

type DiagnosticPoll = (Option<(DiagnosticLevel, RuntimeRecord)>, u64);

/// Polls the next runtime event and dropped-event count.
///
/// # Errors
///
/// Returns an error when the context is invalid, stale, unhealthy, or asynchronous texture progress fails.
pub fn poll_runtime_event(context: ContextHandle) -> Result<(Option<RuntimeRecord>, u64)> {
    with_context_mut(context, |context| {
        pump_async_textures(context)?;
        Ok(context.observability.poll_event())
    })
}
/// Polls the lossless upload queue after progressing owner-thread texture and geometry work.
///
/// # Errors
///
/// Returns an error only for an invalid or stale context handle.
pub fn poll_upload_event(context: ContextHandle) -> Result<Option<UploadEvent>> {
    with_context_mut(context, |context| {
        let texture_progress = progress_texture_upload_events(context);
        // Preserve queued ownership events ahead of terminal loss, but sweep immediately;
        // returning an older event must not mask cleanup or omit the terminal event.
        if matches!(&texture_progress, Err(Error::DeviceLost)) {
            super::texture::note_device_lost(context);
        }
        match completed_transfer_native(&mut context.native) {
            Ok(completed) => {
                let ready: Vec<_> = context
                    .geometry_uploads
                    .iter()
                    .filter_map(|(handle, token)| (token.value <= completed).then_some(*handle))
                    .collect();
                for handle in ready {
                    let resource = match context
                        .identity
                        .resource_kind(handle)
                        .map_err(map_lifecycle)?
                    {
                        ResourceKind::VertexAllocation => UploadResource::Vertex(
                            VertexAllocationHandle::from_packed(handle)
                                .map_err(|_| Error::NativeFailure)?,
                        ),
                        ResourceKind::IndexAllocation => UploadResource::Index(
                            IndexAllocationHandle::from_packed(handle)
                                .map_err(|_| Error::NativeFailure)?,
                        ),
                        _ => return Err(Error::NativeFailure),
                    };
                    context.geometry_uploads.remove(&handle);
                    context.upload_events.push(UploadEvent {
                        resource,
                        status: UploadStatus::DeviceReady,
                    });
                }
            }
            Err(error) => {
                let status = UploadStatus::Failed(runtime_status(Err(map_allocation(error))));
                let pending: Vec<_> = context.geometry_uploads.keys().copied().collect();
                for handle in pending {
                    let resource = match context
                        .identity
                        .resource_kind(handle)
                        .map_err(map_lifecycle)?
                    {
                        ResourceKind::VertexAllocation => UploadResource::Vertex(
                            VertexAllocationHandle::from_packed(handle)
                                .map_err(|_| Error::NativeFailure)?,
                        ),
                        ResourceKind::IndexAllocation => UploadResource::Index(
                            IndexAllocationHandle::from_packed(handle)
                                .map_err(|_| Error::NativeFailure)?,
                        ),
                        _ => return Err(Error::NativeFailure),
                    };
                    context.geometry_uploads.remove(&handle);
                    context.upload_events.push(UploadEvent { resource, status });
                }
            }
        }
        if let Some(event) = context.upload_events.pop() {
            Ok(Some(event))
        } else {
            texture_progress.map(|()| None)
        }
    })
}

/// Polls the next diagnostic and dropped-record count.
///
/// # Errors
///
/// Returns an error when the context handle is invalid or stale.
pub fn poll_diagnostic(context: ContextHandle) -> Result<DiagnosticPoll> {
    with_context_mut(context, |context| {
        Ok(context.observability.poll_diagnostic())
    })
}

/// Waits for all context work to finish without destroying resources.
///
/// # Errors
///
/// Returns an invalid-context, not-ready, native-failure, or device-loss result.
pub fn wait_idle(context: ContextHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        while !context.pending_textures.is_empty() {
            pump_async_textures(context)?;
            if !context.pending_textures.is_empty() {
                std::thread::yield_now();
            }
        }
        let result = match &mut context.native {
            NativeContext::Vulkan(native) => native.wait_idle(),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native.wait_idle(),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native.wait_idle(),
        };
        result.map_err(|error| map_native_loss(&context.identity, error))?;

        // Native idle advances completion counters, but publication remains safe-core state.
        // Progress once more so callers may use completed uploads immediately after this call.
        progress_texture_upload_events(context)
    }))
}

/// Destroys a graphics context and every resource it owns.
///
/// Device initialization is optional: a context destroyed before `init_device` has no GPU work to
/// wait for. Once a native device exists, teardown waits for it before releasing resources. The
/// context is removed even when a native wait or release fails. If draining cannot establish
/// completion, the terminal context retains GPU-owned resources rather than destroying live
/// storage. Otherwise all remaining releases are attempted and the first failure is returned.
///
/// Thread affinity is kept: call only on the creator thread. Callers under Windows loader
/// lock (DllMain/TLS teardown) must use the abandon path only (thread-exit invalidation,
/// which forgets GPU owners without native cleanup or joins) and never call this or
/// `wait_idle` there.
///
/// # Errors
///
/// Returns [`Error::InvalidContext`] for an invalid, stale, repeated, or wrong-thread
/// destroy. Native wait/release failures are returned after terminal cleanup.
pub fn destroy_context(context: ContextHandle) -> Result<()> {
    let (local, _) = context_local(context)?;
    let owned = CONTEXTS.with(|contexts| {
        let mut contexts = contexts
            .try_borrow_mut()
            .map_err(|_| Error::NativeFailure)?;
        contexts
            .states
            .get(&local)
            .ok_or(Error::InvalidContext)?
            .identity
            .check_thread()
            .map_err(map_lifecycle)?;
        contexts.states.remove(&local).ok_or(Error::InvalidContext)
    })?;
    let failure = match CONTEXT_HANDLES.lock() {
        Ok(mut handles) => handles.remove(local).err().map(|_| Error::NativeFailure),
        Err(_) => Some(Error::NativeFailure),
    };

    cleanup_context_state(owned, failure)
}
pub(super) fn cleanup_context_state(
    mut owned: ContextState,
    mut failure: Option<Error>,
) -> Result<()> {
    // Handles become terminal before cleanup begins; later failures cannot expose partial state.
    owned.identity.invalidate_resources();
    owned.async_textures.pool.shutdown();
    for (_, pending) in owned.pending_textures.drain() {
        pending.cancelled.store(true, Ordering::Release);
        if let Err(error) = owned.texture_registry.cancel_upload(pending.id) {
            failure.get_or_insert_with(|| map_texture(error));
        }
    }
    owned.texture_failures.clear();
    // Vulkan reports `NotReady` only when no native device or GPU work exists before `init_device`.
    if let Err(error) = wait_native_idle(&mut owned.native)
        && error != HalError::NotReady
    {
        failure.get_or_insert_with(|| map_native_loss(&owned.identity, error));
    }
    let drained = match &owned.native {
        NativeContext::Vulkan(native) => native.is_drained(),
        #[cfg(windows)]
        NativeContext::Dx12(native) => native.is_drained(),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(native) => native.is_drained(),
    };
    if !drained {
        // The handle is already terminal. Keep this bounded owner intact: even shader, frame,
        // staging, and descriptor resources may still be referenced by an undrained live queue.
        let error = failure.unwrap_or(Error::NativeFailure);
        std::mem::forget(owned);
        return Err(error);
    }

    for (_, pipeline) in owned.pipelines.drain() {
        destroy_native_pipeline(&mut owned.native, pipeline);
    }
    for (_, shader) in owned.shaders.drain() {
        destroy_native_shader(&mut owned.native, shader.native);
    }
    for (handle, (id, texture, _, _, _)) in owned.textures.drain() {
        owned.texture_ready.remove(&handle);
        if let Err(error) = owned.texture_registry.unload(id) {
            failure.get_or_insert_with(|| map_texture(error));
        }
        if let Err(error) = destroy_native_texture(&mut owned.native, texture) {
            failure.get_or_insert_with(|| map_allocation(error));
        }
    }
    for retired in owned.retired_textures.drain(..) {
        if let Err(error) = owned.texture_registry.release_retired(retired.id) {
            failure.get_or_insert_with(|| map_texture(error));
        }
        if let Err(error) = destroy_native_texture(&mut owned.native, retired.native) {
            failure.get_or_insert_with(|| map_allocation(error));
        }
    }
    destroy_all_render_targets(&mut owned);
    owned.texture_formats.clear();
    owned.texture_published_mips.clear();
    owned.texture_residency_targets.clear();
    owned.texture_last_transfer.clear();
    owned.texture_ready.clear();
    owned.texture_handoffs.clear();
    if let Err(error) = owned.texture_registry.clear() {
        failure.get_or_insert_with(|| map_texture(error));
    }
    destroy_buffer_state(&mut owned, &mut failure);
    owned.graphics_format = None;
    owned.frame_resources.clear();
    owned.frame_native_resources.clear();
    owned.frame_vertex_heaps.clear();
    owned.frame_index = None;
    owned.frame_surface = None;
    owned.frame_depth = None;
    owned.frame_has_graphics = false;
    owned.last_readbacks.clear();
    owned.active_surface = None;
    owned.frame_presented = false;

    for (_, surface) in owned.surfaces.drain() {
        destroy_native_surface(&mut owned.native, surface.native);
    }

    // FrameRecorder, GeometryManager, Observability, options, and emptied collections are CPU-only;
    // NativeContext drops last, after every object created from it.
    drop(owned);
    failure.map_or(Ok(()), Err)
}

fn destroy_buffer_state(owned: &mut ContextState, failure: &mut Option<Error>) {
    let mut free = |allocation| {
        if let Err(error) = free_native_allocation(&mut owned.native, allocation) {
            failure.get_or_insert_with(|| map_allocation(error));
        }
    };

    for (_, (_, allocation)) in owned.allocations.drain() {
        free(allocation);
    }
    owned.transient_buffers.clear();
    for (_, mut pool) in owned.structured_pool.drain() {
        for allocation in pool.drain() {
            free(allocation);
        }
    }
    for allocation in owned.indirect_pool.drain() {
        free(allocation);
    }
    owned.indirects.clear();
    for (_, heap) in owned.vertex_heaps.drain() {
        free(heap.allocation);
    }
    owned.vertex_heap_handles.clear();
    if let Some(heap) = owned.index_heap.take() {
        free(heap.allocation);
    }
    for retired in owned.retired_geometry.drain(..) {
        free(retired.allocation);
    }
    owned.retired_geometry_ranges.clear();
    for retired in owned.retired_vertex_heaps.drain(..) {
        free(retired.allocation);
    }
    for allocation in owned.staging.drain() {
        free(allocation);
    }
}

/// Creates a presentation surface.
///
/// # Errors
///
/// Returns an error for an invalid context or surface options, exhausted handles, or native surface failure.
pub fn create_surface(context: ContextHandle, options: SurfaceOptions) -> Result<SurfaceHandle> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if options.platform != context.options.surface_platform {
            return Err(Error::InvalidArgument);
        }
        let headless = options.platform == SurfacePlatform::Headless;
        let native = match &mut context.native {
            NativeContext::Vulkan(native) => NativeSurface::Vulkan(if headless {
                native.create_headless_surface().map_err(map_hal)?
            } else {
                native
                    .create_win32_surface(options.window as *mut _, options.display as *mut _)
                    .map_err(map_hal)?
            }),
            #[cfg(windows)]
            NativeContext::Dx12(_) => {
                NativeSurface::Dx12(Dx12Surface::new(options.window as *mut _).map_err(map_hal)?)
            }
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(_) => NativeSurface::Metal(
                MetalSurface::new(options.window as *mut _, options.cache_presented_snapshots)
                    .map_err(map_hal)?,
            ),
        };
        let handle = match context.identity.insert(ResourceKind::Surface) {
            Ok(handle) => handle,
            Err(error) => {
                destroy_native_surface(&mut context.native, native);
                return Err(map_lifecycle(error));
            }
        };
        let Ok(state) = SurfaceState::new(
            options.width,
            options.height,
            options.cache_presented_snapshots,
        ) else {
            let _ = context.identity.remove(handle, ResourceKind::Surface);
            destroy_native_surface(&mut context.native, native);
            return Err(Error::InvalidArgument);
        };
        let Ok(handle) = SurfaceHandle::from_packed(handle) else {
            let _ = context.identity.remove(handle, ResourceKind::Surface);
            destroy_native_surface(&mut context.native, native);
            return Err(Error::NativeFailure);
        };
        context
            .surfaces
            .insert(handle, SurfaceRecord { native, state });
        Ok(handle)
    })
}

/// Initializes a context device for a surface.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn init_device(context: ContextHandle, surface: SurfaceHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = surface.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Surface)
            .map_err(map_lifecycle)?;
        let record = context
            .surfaces
            .get(&surface)
            .ok_or(Error::InvalidContext)?;
        let first_initialization = context.active_surface.is_none();
        // Explicit selection is enforced at device creation: Vulkan instances
        // are adapter-agnostic, so the stable identity resolves here.
        let selection = context.options.adapter_selection;
        let adapter = match (&mut context.native, &record.native) {
            (NativeContext::Vulkan(native), NativeSurface::Vulkan(surface)) => {
                let presentation_surface = (!surface.is_headless()).then_some(surface);
                match selection {
                    Some(selected) => native.init_device_for_adapter(
                        presentation_surface,
                        selected.stable_id,
                        selected.allow_software,
                    ),
                    None => native.init_device(presentation_surface),
                }
            }
            #[cfg(windows)]
            (NativeContext::Dx12(native), NativeSurface::Dx12(surface)) => {
                native.init_device(surface)
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(native), NativeSurface::Metal(surface)) => {
                native.init_device(surface)
            }
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(HalError::InvalidArgument),
        }
        .map_err(|error| map_native_loss(&context.identity, error))?;
        context.active_surface = Some(surface);
        if first_initialization {
            let backend = match adapter.backend() {
                Backend::Vulkan => "Vulkan",
                Backend::Dx12 => "DirectX 12",
                Backend::Metal => "Metal",
            };
            eprintln!(
                "ez-gfx: initialized GPU `{}` with {backend}",
                adapter.name()
            );
        }
        Ok(())
    }))
}

/// Requests a surface resize.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn resize_surface(
    context: ContextHandle,
    surface: SurfaceHandle,
    width: u32,
    height: u32,
) -> Result<()> {
    result_status(with_surface_mut(context, surface, |record| {
        record
            .state
            .resize(width, height)
            .map_err(|error| match error {
                ez_gfx_runtime::PublicApiError::NotReady => Error::NotReady,
                _ => Error::InvalidArgument,
            })
    }))
}
/// Returns the current surface extent.
///
/// # Errors
///
/// Returns an error when either handle is invalid or the extent is not ready.
pub fn surface_extent(context: ContextHandle, surface: SurfaceHandle) -> Result<(u32, u32)> {
    with_surface_mut(context, surface, |record| {
        record.state.extent().ok_or(Error::NotReady)
    })
}
/// Reports whether a surface resize is pending.
///
/// # Errors
///
/// Returns an error when either handle is invalid or stale.
pub fn surface_resize_pending(context: ContextHandle, surface: SurfaceHandle) -> Result<bool> {
    with_surface_mut(context, surface, |record| Ok(record.state.resize_pending()))
}
/// Enables or disables presented snapshot caching.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn set_snapshot_cache(
    context: ContextHandle,
    surface: SurfaceHandle,
    enabled: bool,
) -> Result<()> {
    result_status(with_surface_mut(context, surface, |record| {
        record.state.set_snapshot_cache(enabled);
        Ok(())
    }))
}

/// Destroys a presentation surface.
pub fn destroy_surface(context: ContextHandle, surface: SurfaceHandle) {
    let _ = with_context_mut(context, |context| {
        let handle = surface.packed();
        context
            .identity
            .remove(handle, ResourceKind::Surface)
            .map_err(map_lifecycle)?;
        let record = context
            .surfaces
            .remove(&surface)
            .ok_or(Error::InvalidContext)?;
        destroy_native_surface(&mut context.native, record.native);
        if context.active_surface == Some(surface) {
            context.active_surface = None;
        }
        Ok(())
    });
}
/// Begins rendering to a surface.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
#[allow(
    dead_code,
    reason = "the C raw seam begins an already-configured surface frame"
)]
pub fn begin_render(context: ContextHandle, surface: SurfaceHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        super::frame::start_recording(context)?;
        if let Err(error) = configure_surface_recording(context, surface) {
            context.frame.abort();
            return Err(error);
        }
        Ok(())
    }))
}

/// Selects the presentation surface for the active recording transaction.
///
/// # Errors
///
/// Returns an error when no frame is recording or the surface is invalid or not ready.
pub(crate) fn configure_surface(context: ContextHandle, surface: SurfaceHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        configure_surface_recording(context, surface)
    }))
}

fn configure_surface_recording(context: &mut ContextState, surface: SurfaceHandle) -> Result<()> {
    context
        .identity
        .check_thread_and_health()
        .map_err(map_lifecycle)?;
    if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording
        || context.active_surface.is_some()
        || context.frame_render_target.is_some()
    {
        return Err(Error::NotReady);
    }
    context
        .identity
        .resolve(surface.packed(), ResourceKind::Surface)
        .map_err(map_lifecycle)?;
    let record = context
        .surfaces
        .get(&surface)
        .ok_or(Error::InvalidContext)?;
    if record.state.extent().is_none() {
        return Err(Error::NotReady);
    }
    context.active_surface = Some(surface);
    Ok(())
}
/// Begins rendering to a managed color target.
///
/// # Errors
/// Returns an error when the target is invalid, unsupported, or another frame is active.
#[allow(
    dead_code,
    reason = "the C raw seam begins a preconfigured managed-target frame"
)]
pub fn begin_render_target(context: ContextHandle, target: RenderTargetHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        super::frame::start_recording(context)?;
        if let Err(error) = configure_render_target_recording(context, target) {
            context.frame.abort();
            return Err(error);
        }
        Ok(())
    }))
}

/// Selects a managed target for the active recording transaction.
///
/// # Errors
/// Returns an error when no frame is recording or the target is invalid.
pub(crate) fn configure_render_target(
    context: ContextHandle,
    target: RenderTargetHandle,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        configure_render_target_recording(context, target)
    }))
}

fn configure_render_target_recording(
    context: &mut ContextState,
    target: RenderTargetHandle,
) -> Result<()> {
    context
        .identity
        .check_thread_and_health()
        .map_err(map_lifecycle)?;
    if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording
        || context.active_surface.is_some()
        || context.frame_render_target.is_some()
    {
        return Err(Error::NotReady);
    }
    context
        .identity
        .resolve(target.packed(), ResourceKind::RenderTarget)
        .map_err(map_lifecycle)?;
    let record = context
        .render_targets
        .get(&target)
        .ok_or(Error::InvalidContext)?;
    if record.declaration.usage() != ez_gfx_runtime::target::TargetUsage::Color {
        return Err(Error::Unsupported);
    }
    if record.width == 0 || record.height == 0 {
        return Err(Error::InvalidArgument);
    }
    context.active_surface = None;
    context.frame_render_target = Some(target);
    Ok(())
}

/// Presents the recorded frame.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn present(context: ContextHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        if context.frame_presented {
            context.frame_presented = false;
            return Ok(());
        }
        let surface_handle = context.active_surface.ok_or(Error::NotReady)?;
        let mut record = context
            .surfaces
            .remove(&surface_handle)
            .ok_or(Error::InvalidContext)?;
        let result = match record.state.extent() {
            None => Err(HalError::NotReady),
            Some((width, height)) => match (&mut context.native, &mut record.native) {
                (NativeContext::Vulkan(_), NativeSurface::Vulkan(surface))
                    if surface.is_headless() =>
                {
                    Err(HalError::Unsupported)
                }
                (NativeContext::Vulkan(native), NativeSurface::Vulkan(surface)) => {
                    native.acquire_present(surface, width, height)
                }
                #[cfg(windows)]
                (NativeContext::Dx12(native), NativeSurface::Dx12(surface)) => {
                    native.acquire_present(surface, width, height)
                }
                #[cfg(target_vendor = "apple")]
                (NativeContext::Metal(native), NativeSurface::Metal(surface)) => {
                    native.acquire_present(surface, width, height)
                }
                #[cfg(any(windows, target_vendor = "apple"))]
                _ => Err(HalError::InvalidArgument),
            },
        };
        context.surfaces.insert(surface_handle, record);
        result.map_err(|error| map_native_loss(&context.identity, error))
    }))
}

pub(super) fn destroy_native_surface(context: &mut NativeContext, surface: NativeSurface) {
    match (context, surface) {
        (NativeContext::Vulkan(context), NativeSurface::Vulkan(surface)) => {
            context.destroy_surface(surface);
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeSurface::Dx12(surface)) => {
            context.destroy_surface(surface);
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeSurface::Metal(surface)) => {
            context.destroy_surface(surface);
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => {}
    }
}
