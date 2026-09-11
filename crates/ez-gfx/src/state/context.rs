use crate::Result;

#[cfg(windows)]
use super::Dx12Context;
#[cfg(target_vendor = "apple")]
use super::MetalContext;
use super::{
    AdapterCatalog, AdapterInfo, AdapterReport, AdapterSelection, Arc, AsyncTextureState, Backend,
    CONTEXT_HANDLES, CONTEXTS, ContextHandle, ContextIdentity, ContextOptions, ContextState,
    DiagnosticLevel, Error, FrameRecorder, GeometryManager, HalError, HashMap,
    IndexAllocationHandle, LocalHandle, NativeContext, NativeSurface, Observability, Ordering,
    PresentationMode, RenderTargetHandle, ResourceKind, RuntimeError, RuntimePhase, RuntimeRecord,
    RuntimeStatus, SurfaceHandle, TextureRegistry, TextureUploadTelemetry, UploadEvent,
    UploadResource, UploadStatus, VertexAllocationHandle, VulkanContext, admission_report,
    completed_transfer_native, context_local, destroy_native_pipeline, destroy_native_shader,
    destroy_native_surface, destroy_native_texture, free_native_allocation, map_allocation,
    map_hal, map_lifecycle, map_native_loss, map_texture, progress_texture_upload_events,
    pump_async_textures, render_target::destroy_all_render_targets, result_status,
    wait_native_idle, with_context_mut,
};
#[cfg(test)]
use super::{CleanupTestOutcome, SurfaceInsertTestFailure};

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
    // Finite retention budgets apply from creation; idle and pressure trims
    // enforce them explicitly, while `put` itself never evicts implicitly.
    let mut staging = ez_gfx_hal::ReusableStagingPool::new(256);
    staging.set_byte_budget(ez_gfx_hal::DEFAULT_SHARED_STAGING_BUDGET);
    let mut counter_pool = ez_gfx_hal::ReusableStagingPool::new(256);
    counter_pool.set_byte_budget(ez_gfx_hal::DEFAULT_COUNTER_STAGING_BUDGET);
    let state = ContextState {
        identity,
        options,
        native,
        surfaces: HashMap::new(),
        allocations: HashMap::new(),
        allocation_ready: HashMap::new(),
        shaders: HashMap::new(),
        frame_shaders: std::collections::HashSet::new(),
        pending_shader_destroys: std::collections::HashSet::new(),
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
        buffer_pool: HashMap::new(),
        counter_pool,
        counter_scratch: Vec::new(),
        texture_registry,
        texture_ready: HashMap::new(),
        pending_textures: HashMap::new(),
        texture_transfer_bytes: HashMap::new(),
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
        staging,
        staging_high_water_bytes: 0,
        frame_vertex_heaps: HashMap::new(),
        frame_serial: 0,
        frame,
        frame_pipeline_keys: Vec::new(),
        frame_action_indices: Vec::new(),
        frame_binding_scratch: Vec::new(),
        frame_binding_ranges: Vec::new(),
        #[cfg(target_vendor = "apple")]
        frame_texture_heaps: Vec::new(),
        #[cfg(target_vendor = "apple")]
        frame_workgroup_sizes: Vec::new(),
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
        frame_capture_surface: None,
        observability,
        #[cfg(test)]
        cleanup_test_outcome: None,
        #[cfg(test)]
        surface_insert_test_failure: None,
        #[cfg(test)]
        surface_rollback_test_abandoned: false,
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
        Backend::Vulkan => NativeContext::Vulkan(Box::new(
            VulkanContext::create(options.enable_debug, options.enable_validation)
                .map_err(map_hal)?,
        )),
        Backend::Dx12 => {
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
        Backend::Vulkan => {
            let catalog =
                AdapterCatalog::new(backend_adapters(Backend::Vulkan)).map_err(map_runtime)?;
            catalog
                .select(selection.stable_id, selection.allow_software)
                .map_err(map_runtime)?;
            NativeContext::Vulkan(Box::new(
                VulkanContext::create(options.enable_debug, options.enable_validation)
                    .map_err(map_hal)?,
            ))
        }
        Backend::Dx12 => {
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
        Err(
            Error::NativeFailure
            | Error::ReentrantCallback
            | Error::CallbackPanicked
            | Error::TeardownAbandoned,
        ) => RuntimeStatus::NativeFailure,
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
        // Native idle retires in-flight work, so completions are fresh: enforce
        // the finite staging budgets now rather than letting retention ride
        // until the next upload happens to sweep it.
        super::buffers::trim_staging_caches(context)?;

        // Native idle advances completion counters, but publication remains safe-core state.
        // Progress once more so callers may use completed uploads immediately after this call.
        progress_texture_upload_events(context)
    }))
}

/// Validates that a raw context belongs to the calling creator thread before terminal teardown.
///
/// # Errors
///
/// Returns an error for an invalid, stale, unavailable, or wrong-thread context.
pub fn validate_context_owner(context: ContextHandle) -> Result<()> {
    with_context_mut(context, |owned| {
        owned.identity.check_thread().map_err(map_lifecycle)
    })
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
    let (local, owned) = remove_context(context)?;
    cleanup_context_state(owned, remove_context_handle(local))
}

/// Best-effort owner-drop teardown; unavailable thread-local state means its
/// own destructor has already invalidated and reclaimed the context.
///
/// # Errors
///
/// Returns the typed terminal cleanup disposition.
pub fn drop_context(context: ContextHandle) -> Result<()> {
    // A safe owner reaches this only once. Failure to recover its state leaves cleanup unproven.
    let (local, owned) = remove_context(context).map_err(|_| Error::TeardownAbandoned)?;
    cleanup_context_state(owned, remove_context_handle(local))
}

#[cfg(test)]
pub(crate) fn inject_cleanup_outcome(
    context: ContextHandle,
    outcome: CleanupTestOutcome,
) -> Result<()> {
    with_context_mut(context, |state| {
        state.cleanup_test_outcome = Some(outcome);
        Ok(())
    })
}

#[cfg(test)]
pub(crate) fn inject_surface_insert_failure(
    context: ContextHandle,
    failure: SurfaceInsertTestFailure,
    rollback_abandoned: bool,
) -> Result<()> {
    with_context_mut(context, |state| {
        state.surface_insert_test_failure = Some(failure);
        state.surface_rollback_test_abandoned = rollback_abandoned;
        Ok(())
    })
}

fn remove_context(context: ContextHandle) -> Result<(LocalHandle, ContextState)> {
    let (local, _) = context_local(context)?;
    let owned = CONTEXTS
        .try_with(|contexts| {
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
        })
        .map_err(|_| Error::InvalidContext)??;
    Ok((local, owned))
}

fn remove_context_handle(local: LocalHandle) -> Option<Error> {
    match CONTEXT_HANDLES.lock() {
        Ok(mut handles) => handles.remove(local).err().map(|_| Error::NativeFailure),
        Err(_) => Some(Error::NativeFailure),
    }
}

pub(super) fn cleanup_context_state(
    mut owned: ContextState,
    mut failure: Option<Error>,
) -> Result<()> {
    // Handles become terminal before cleanup begins; later failures cannot expose partial state.
    owned.identity.invalidate_resources();
    owned.async_textures.shutdown();
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
    #[cfg(test)]
    let drained = drained && owned.cleanup_test_outcome != Some(CleanupTestOutcome::Undrained);
    if !drained {
        // The handle is already terminal. Keep this bounded owner intact: even shader, frame,
        // staging, and descriptor resources may still be referenced by an undrained live queue.
        // Forgetting native ownership makes abandonment authoritative over the triggering failure.
        std::mem::forget(owned);
        return Err(Error::TeardownAbandoned);
    }
    #[cfg(test)]
    if let Some(CleanupTestOutcome::DrainedFailure(error)) = owned.cleanup_test_outcome.take() {
        failure.get_or_insert(error);
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
    owned.texture_transfer_bytes.clear();
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

    while let Some(handle) = owned.surfaces.keys().next().copied() {
        let surface = owned
            .surfaces
            .remove(&handle)
            .expect("surface key came from this map");
        if let Err(error) = destroy_native_surface(&mut owned.native, surface.native) {
            if error == Error::TeardownAbandoned {
                // Native work may still reference this surface and its host-backed objects.
                std::mem::forget(owned);
                return Err(Error::TeardownAbandoned);
            }
            failure.get_or_insert(error);
        }
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
    for (_, mut pool) in owned.buffer_pool.drain() {
        for allocation in pool.drain() {
            free(allocation);
        }
    }
    for allocation in owned.counter_pool.drain() {
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

/// Begins rendering to a surface.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
#[allow(
    dead_code,
    reason = "the C raw seam begins an already-configured surface frame"
)]
pub fn begin_render(
    context: ContextHandle,
    surface: SurfaceHandle,
    presentation_mode: PresentationMode,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        super::frame::start_recording(context)?;
        if let Err(error) = configure_surface_recording(context, surface, presentation_mode) {
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
pub(crate) fn configure_surface(
    context: ContextHandle,
    surface: SurfaceHandle,
    requested: PresentationMode,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        configure_surface_recording(context, surface, requested)
    }))
}

fn configure_surface_recording(
    context: &mut ContextState,
    surface: SurfaceHandle,
    requested: PresentationMode,
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
        .resolve(surface.packed(), ResourceKind::Surface)
        .map_err(map_lifecycle)?;
    let available = context
        .surfaces
        .get(&surface)
        .and_then(|record| record.presentation_modes)
        .map_or_else(
            || super::surface::presentation_modes_for_record(context, surface),
            Ok,
        )?;
    let effective = available.resolve(requested).ok_or(Error::Unsupported)?;
    let record = context
        .surfaces
        .get_mut(&surface)
        .ok_or(Error::InvalidContext)?;
    if record.state.extent().is_none() {
        return Err(Error::NotReady);
    }
    // Keep same-mode frame configuration allocation-free until resize or reinitialization.
    record.presentation_modes = Some(available);
    record.presentation_mode = effective;
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
                    native.acquire_present(surface, width, height, record.presentation_mode)
                }
                #[cfg(windows)]
                (NativeContext::Dx12(native), NativeSurface::Dx12(surface)) => {
                    native.acquire_present(surface, width, height, record.presentation_mode)
                }
                #[cfg(target_vendor = "apple")]
                (NativeContext::Metal(native), NativeSurface::Metal(surface)) => {
                    native.acquire_present(surface, width, height, record.presentation_mode)
                }
                #[cfg(any(windows, target_vendor = "apple"))]
                _ => Err(HalError::InvalidArgument),
            },
        };
        context.surfaces.insert(surface_handle, record);
        result.map_err(|error| map_native_loss(&context.identity, error))
    }))
}
