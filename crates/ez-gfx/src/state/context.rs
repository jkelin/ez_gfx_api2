use super::{
    AllocationRequest, Arc, AsyncTextureState, Backend, CONTEXT_HANDLES, CONTEXTS, CompletionToken,
    ContextHandle, ContextIdentity, ContextOptions, ContextState, DEFAULT_STAGING_POLICY,
    DiagnosticLevel, EzGfxResult, FrameRecorder, GeometryAllocation, GeometryManager, HalError,
    HashMap, MemoryClass, NativeAllocation, NativeContext, NativeSurface, Observability, Ordering,
    ResourceKind, RuntimePhase, RuntimeRecord, RuntimeStatus, SurfaceHandle, SurfaceOptions,
    SurfacePlatform, SurfaceRecord, SurfaceState, TextureRegistry, TextureUploadTelemetry,
    VulkanContext, VulkanPlatform, allocate_native, completed_transfer_native, context_local,
    copy_native, destroy_native_pipeline, destroy_native_shader, destroy_native_texture,
    free_native_allocation, map_allocation, map_frame, map_geometry, map_hal, map_lifecycle,
    map_native_loss, map_texture, pump_async_textures, result_status, staging_bucket_size,
    wait_native_idle, with_context_mut, with_surface_mut, write_native,
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
pub fn create_context(options: ContextOptions) -> Result<ContextHandle, EzGfxResult> {
    let native = match options.backend {
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
        Backend::Vulkan => return Err(EzGfxResult::Unsupported),
        Backend::Dx12 => {
            if options.surface_platform != SurfacePlatform::Win32 {
                return Err(EzGfxResult::InvalidArgument);
            }
            #[cfg(windows)]
            {
                NativeContext::Dx12(Box::new(
                    Dx12Context::create_default(false).map_err(|_| EzGfxResult::NativeFailure)?,
                ))
            }
            #[cfg(not(windows))]
            {
                return Err(EzGfxResult::Unsupported);
            }
        }
        Backend::Metal => {
            if options.surface_platform != SurfacePlatform::MetalLayer {
                return Err(EzGfxResult::InvalidArgument);
            }
            #[cfg(target_vendor = "apple")]
            {
                NativeContext::Metal(Box::new(MetalContext::create_default().map_err(map_hal)?))
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                return Err(EzGfxResult::Unsupported);
            }
        }
    };
    let texture_registry = TextureRegistry::new(
        ez_gfx_runtime::binding::MAX_TEXTURE_HEAP_CAPACITY,
        ez_gfx_runtime::binding::MAX_TEXTURE_HEAP_CAPACITY,
    )
    .map_err(|_| EzGfxResult::NativeFailure)?;
    let frame = FrameRecorder::new(1024).map_err(|_| EzGfxResult::NativeFailure)?;
    let observability = Observability::new(1024, 256).map_err(|_| EzGfxResult::NativeFailure)?;
    let async_textures = AsyncTextureState::new_with_workers(options.texture_decode_workers)?;
    let local = CONTEXT_HANDLES
        .lock()
        .map_err(|_| EzGfxResult::NativeFailure)?
        .insert(())
        .map_err(|_| EzGfxResult::NativeFailure)?;
    let Ok(identity) = ContextIdentity::new(local) else {
        if let Ok(mut handles) = CONTEXT_HANDLES.lock() {
            let _ = handles.remove(local);
        }
        return Err(EzGfxResult::NativeFailure);
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
        texture_formats: HashMap::new(),
        texture_published_mips: HashMap::new(),
        texture_residency_targets: HashMap::new(),
        texture_last_transfer: HashMap::new(),
        retired_textures: Vec::new(),
        pipelines: HashMap::new(),
        graphics_format: None,
        indirects: HashMap::new(),
        texture_registry,
        texture_ready: HashMap::new(),
        pending_textures: HashMap::new(),
        texture_handoffs: HashMap::new(),
        texture_telemetry: Arc::new(TextureUploadTelemetry::default()),
        async_textures,
        texture_failures: HashMap::new(),
        geometry: GeometryManager::new(),
        vertex_heaps: HashMap::new(),
        index_heap: None,
        staging: ez_gfx_hal::ReusableStagingPool::new(256),
        frame,
        frame_resources: HashMap::new(),
        frame_native_resources: HashMap::new(),
        frame_index: None,
        frame_surface: None,
        frame_depth: None,
        frame_has_graphics: false,
        last_readback: Vec::new(),
        active_surface: None,
        frame_presented: false,
        observability,
    };
    let inserted = CONTEXTS.with(|contexts| {
        let mut contexts = contexts
            .try_borrow_mut()
            .map_err(|_| EzGfxResult::NativeFailure)?;
        if contexts.states.contains_key(&local) {
            return Err(EzGfxResult::NativeFailure);
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

pub(super) fn runtime_status(status: EzGfxResult) -> RuntimeStatus {
    match status {
        EzGfxResult::Ok => RuntimeStatus::Ok,
        EzGfxResult::InvalidArgument | EzGfxResult::InvalidContext => {
            RuntimeStatus::InvalidArgument
        }
        EzGfxResult::NotReady | EzGfxResult::QueueFull => RuntimeStatus::NotReady,
        EzGfxResult::Cancelled => RuntimeStatus::Cancelled,
        EzGfxResult::Unsupported => RuntimeStatus::Unsupported,
        EzGfxResult::NativeFailure => RuntimeStatus::NativeFailure,
        EzGfxResult::DeviceLost => RuntimeStatus::DeviceLost,
    }
}

pub(super) fn runtime_record(
    context: &mut ContextState,
    resource: u64,
    phase: RuntimePhase,
    status: EzGfxResult,
) -> RuntimeRecord {
    RuntimeRecord {
        correlation_id: context.observability.next_correlation(),
        resource,
        backend: context.options.backend,
        phase,
        status: runtime_status(status),
    }
}

type DiagnosticPoll = (Option<(DiagnosticLevel, RuntimeRecord)>, u64);

/// Polls the next runtime event and dropped-event count.
///
/// # Errors
///
/// Returns an error when the context is invalid, stale, unhealthy, or asynchronous texture progress fails.
pub fn poll_runtime_event(
    context: ContextHandle,
) -> Result<(Option<RuntimeRecord>, u64), EzGfxResult> {
    with_context_mut(context, |context| {
        pump_async_textures(context)?;
        Ok(context.observability.poll_event())
    })
}

/// Polls the next diagnostic and dropped-record count.
///
/// # Errors
///
/// Returns an error when the context handle is invalid or stale.
pub fn poll_diagnostic(context: ContextHandle) -> Result<DiagnosticPoll, EzGfxResult> {
    with_context_mut(context, |context| {
        Ok(context.observability.poll_diagnostic())
    })
}

/// Waits for all context work to finish without destroying resources.
///
/// # Errors
///
/// Returns an invalid-context, not-ready, native-failure, or device-loss result.
pub fn wait_idle(context: ContextHandle) -> EzGfxResult {
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
        result.map_err(|error| map_native_loss(&context.identity, error))
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
/// # Errors
///
/// Returns [`EzGfxResult::InvalidContext`] for an invalid, stale, repeated, or wrong-thread
/// destroy. Native wait/release failures are returned after terminal cleanup.
pub fn destroy_context(context: ContextHandle) -> EzGfxResult {
    let Ok((local, _)) = context_local(context) else {
        return EzGfxResult::InvalidContext;
    };
    let owned = CONTEXTS.with(|contexts| {
        let mut contexts = contexts
            .try_borrow_mut()
            .map_err(|_| EzGfxResult::NativeFailure)?;
        contexts
            .states
            .get(&local)
            .ok_or(EzGfxResult::InvalidContext)?
            .identity
            .check_thread()
            .map_err(map_lifecycle)?;
        contexts
            .states
            .remove(&local)
            .ok_or(EzGfxResult::InvalidContext)
    });
    let owned = match owned {
        Ok(owned) => owned,
        Err(error) => return error,
    };
    let failure = match CONTEXT_HANDLES.lock() {
        Ok(mut handles) => handles
            .remove(local)
            .err()
            .map(|_| EzGfxResult::NativeFailure),
        Err(_) => Some(EzGfxResult::NativeFailure),
    };

    cleanup_context_state(owned, failure)
}
pub(super) fn cleanup_context_state(
    mut owned: ContextState,
    mut failure: Option<EzGfxResult>,
) -> EzGfxResult {
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
        let result = failure.unwrap_or(EzGfxResult::NativeFailure);
        std::mem::forget(owned);
        return result;
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
    owned.texture_formats.clear();
    owned.texture_published_mips.clear();
    owned.texture_residency_targets.clear();
    owned.texture_last_transfer.clear();
    owned.texture_ready.clear();
    owned.texture_handoffs.clear();
    if let Err(error) = owned.texture_registry.clear() {
        failure.get_or_insert_with(|| map_texture(error));
    }
    for (_, (_, allocation)) in owned.allocations.drain() {
        if let Err(error) = free_native_allocation(&mut owned.native, allocation) {
            failure.get_or_insert_with(|| map_allocation(error));
        }
    }
    owned.indirects.clear();
    for (_, heap) in owned.vertex_heaps.drain() {
        if let Err(error) = free_native_allocation(&mut owned.native, heap.allocation) {
            failure.get_or_insert_with(|| map_allocation(error));
        }
    }
    if let Some(heap) = owned.index_heap.take()
        && let Err(error) = free_native_allocation(&mut owned.native, heap.allocation)
    {
        failure.get_or_insert_with(|| map_allocation(error));
    }
    for allocation in owned.staging.drain() {
        if let Err(error) = free_native_allocation(&mut owned.native, allocation) {
            failure.get_or_insert_with(|| map_allocation(error));
        }
    }
    owned.graphics_format = None;
    owned.frame_resources.clear();
    owned.frame_native_resources.clear();
    owned.frame_index = None;
    owned.frame_surface = None;
    owned.frame_depth = None;
    owned.frame_has_graphics = false;
    owned.last_readback.clear();
    owned.active_surface = None;
    owned.frame_presented = false;

    for (_, surface) in owned.surfaces.drain() {
        destroy_native_surface(&mut owned.native, surface.native);
    }

    // FrameRecorder, GeometryManager, Observability, options, and emptied collections are CPU-only;
    // NativeContext drops last, after every object created from it.
    drop(owned);
    failure.unwrap_or(EzGfxResult::Ok)
}

/// Creates a presentation surface.
///
/// # Errors
///
/// Returns an error for an invalid context or surface options, exhausted handles, or native surface failure.
pub fn create_surface(
    context: ContextHandle,
    options: SurfaceOptions,
) -> Result<SurfaceHandle, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if options.platform != context.options.surface_platform {
            return Err(EzGfxResult::InvalidArgument);
        }
        let native = match &mut context.native {
            NativeContext::Vulkan(native) => NativeSurface::Vulkan(
                native
                    .create_win32_surface(options.window as *mut _, options.display as *mut _)
                    .map_err(map_hal)?,
            ),
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
        let state = SurfaceState::new(
            options.width,
            options.height,
            options.cache_presented_snapshots,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        let handle = SurfaceHandle::from_packed(handle).map_err(|_| EzGfxResult::NativeFailure)?;
        context
            .surfaces
            .insert(handle, SurfaceRecord { native, state });
        Ok(handle)
    })
}

/// Initializes a context device for a surface.
pub fn init_device(context: ContextHandle, surface: SurfaceHandle) -> EzGfxResult {
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
            .ok_or(EzGfxResult::InvalidContext)?;
        let first_initialization = context.active_surface.is_none();
        let adapter = match (&mut context.native, &record.native) {
            (NativeContext::Vulkan(native), NativeSurface::Vulkan(surface)) => {
                native.init_device(Some(surface))
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
pub fn resize_surface(
    context: ContextHandle,
    surface: SurfaceHandle,
    width: u32,
    height: u32,
) -> EzGfxResult {
    result_status(with_surface_mut(context, surface, |record| {
        record
            .state
            .resize(width, height)
            .map_err(|error| match error {
                ez_gfx_runtime::PublicApiError::NotReady => EzGfxResult::NotReady,
                _ => EzGfxResult::InvalidArgument,
            })
    }))
}
/// Returns the current surface extent.
///
/// # Errors
///
/// Returns an error when either handle is invalid or the extent is not ready.
pub fn surface_extent(
    context: ContextHandle,
    surface: SurfaceHandle,
) -> Result<(u32, u32), EzGfxResult> {
    with_surface_mut(context, surface, |record| {
        record.state.extent().ok_or(EzGfxResult::NotReady)
    })
}
/// Reports whether a surface resize is pending.
///
/// # Errors
///
/// Returns an error when either handle is invalid or stale.
pub fn surface_resize_pending(
    context: ContextHandle,
    surface: SurfaceHandle,
) -> Result<bool, EzGfxResult> {
    with_surface_mut(context, surface, |record| Ok(record.state.resize_pending()))
}
/// Enables or disables presented snapshot caching.
pub fn set_snapshot_cache(
    context: ContextHandle,
    surface: SurfaceHandle,
    enabled: bool,
) -> EzGfxResult {
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
            .ok_or(EzGfxResult::InvalidContext)?;
        destroy_native_surface(&mut context.native, record.native);
        if context.active_surface == Some(surface) {
            context.active_surface = None;
        }
        Ok(())
    });
}
/// Begins rendering to a surface.
pub fn begin_render(context: ContextHandle, surface: SurfaceHandle) -> EzGfxResult {
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
            .ok_or(EzGfxResult::InvalidContext)?;
        if record.state.extent().is_none() {
            return Err(EzGfxResult::NotReady);
        }
        context.active_surface = Some(surface);
        context.frame_resources.clear();
        context.frame_native_resources.clear();
        context.frame_index = None;
        context.frame_surface = None;
        context.frame_depth = None;
        context.frame_has_graphics = false;
        context.last_readback.clear();
        context.frame_presented = false;
        context.frame.begin().map_err(|error| map_frame(&error))
    }))
}

/// Presents the recorded frame.
pub fn present(context: ContextHandle) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        if context.frame_presented {
            context.frame_presented = false;
            return Ok(());
        }
        let surface_handle = context.active_surface.ok_or(EzGfxResult::NotReady)?;
        let mut record = context
            .surfaces
            .remove(&surface_handle)
            .ok_or(EzGfxResult::InvalidContext)?;
        let result = match record.state.extent() {
            None => Err(HalError::NotReady),
            Some((width, height)) => match (&mut context.native, &mut record.native) {
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

/// Creates a named vertex heap.
pub fn create_vertex_heap(
    context: ContextHandle,
    name: &str,
    capacity: u64,
    stride: u64,
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .geometry
            .create_vertex_heap(name, capacity, stride)
            .map_err(map_geometry)?;
        let request = AllocationRequest::new(capacity, 16, MemoryClass::Device, false, None)
            .map_err(map_allocation)?;
        match allocate_native(&mut context.native, request) {
            Ok(allocation) => {
                context.vertex_heaps.insert(
                    name.to_owned(),
                    GeometryAllocation {
                        allocation,
                        ready: None,
                        size: capacity,
                    },
                );
                Ok(())
            }
            Err(error) => {
                let _ = context.geometry.remove_vertex_heap(name);
                Err(map_allocation(error))
            }
        }
    }))
}

/// Destroys a named vertex heap.
pub fn destroy_vertex_heap(context: ContextHandle, name: &str) {
    let _ = with_context_mut(context, |context| {
        context
            .geometry
            .remove_vertex_heap(name)
            .map_err(map_geometry)?;
        let heap = context
            .vertex_heaps
            .remove(name)
            .ok_or(EzGfxResult::InvalidArgument)?;
        free_native_allocation(&mut context.native, heap.allocation).map_err(map_allocation)
    });
}

/// Creates the context index heap.
pub fn create_index_heap(context: ContextHandle, capacity: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .geometry
            .create_index_heap(capacity)
            .map_err(map_geometry)?;
        let request = AllocationRequest::new(capacity, 16, MemoryClass::Device, false, None)
            .map_err(map_allocation)?;
        match allocate_native(&mut context.native, request) {
            Ok(allocation) => {
                context.index_heap = Some(GeometryAllocation {
                    allocation,
                    ready: None,
                    size: capacity,
                });
                Ok(())
            }
            Err(error) => {
                let _ = context.geometry.remove_index_heap();
                Err(map_allocation(error))
            }
        }
    }))
}

/// Destroys the context index heap.
pub fn destroy_index_heap(context: ContextHandle) {
    let _ = with_context_mut(context, |context| {
        context.geometry.remove_index_heap().map_err(map_geometry)?;
        let heap = context
            .index_heap
            .take()
            .ok_or(EzGfxResult::InvalidArgument)?;
        free_native_allocation(&mut context.native, heap.allocation).map_err(map_allocation)
    });
}

/// Uploads elements to a named vertex heap.
///
/// # Errors
///
/// Returns an error for invalid handles, sizes, ranges, capacity, or native upload failure.
pub fn upload_vertices(
    context: ContextHandle,
    name: &str,
    count: u32,
    element_size: u64,
    bytes: &[u8],
) -> Result<u32, EzGfxResult> {
    with_context_mut(context, |context| {
        let upload = context
            .geometry
            .reserve_vertices(name, count, element_size)
            .map_err(map_geometry)?;
        if bytes.len() as u64 != upload.byte_size {
            let _ = context.geometry.rollback_vertices(name, upload);
            return Err(EzGfxResult::InvalidArgument);
        }
        let mut heap = context
            .vertex_heaps
            .remove(name)
            .ok_or(EzGfxResult::InvalidArgument)?;
        let result = stage_upload(
            &mut context.native,
            &mut context.staging,
            &heap.allocation,
            upload.byte_offset,
            bytes,
        );
        match result {
            Ok(token) => {
                heap.ready = Some(token);
                context
                    .geometry
                    .mark_vertex_ready(name, token)
                    .map_err(map_geometry)?;
                context.vertex_heaps.insert(name.to_owned(), heap);
                Ok(upload.first_element)
            }
            Err(error) => {
                context.vertex_heaps.insert(name.to_owned(), heap);
                let _ = context.geometry.rollback_vertices(name, upload);
                Err(map_allocation(error))
            }
        }
    })
}

/// Uploads indices to the context index heap.
///
/// # Errors
///
/// Returns an error for invalid handles, sizes, ranges, capacity, or native upload failure.
pub fn upload_indices(
    context: ContextHandle,
    count: u32,
    bytes: &[u8],
) -> Result<u32, EzGfxResult> {
    with_context_mut(context, |context| {
        let upload = context
            .geometry
            .reserve_indices(count)
            .map_err(map_geometry)?;
        if bytes.len() as u64 != upload.byte_size {
            let _ = context.geometry.rollback_indices(upload);
            return Err(EzGfxResult::InvalidArgument);
        }
        let mut heap = context
            .index_heap
            .take()
            .ok_or(EzGfxResult::InvalidArgument)?;
        let result = stage_upload(
            &mut context.native,
            &mut context.staging,
            &heap.allocation,
            upload.byte_offset,
            bytes,
        );
        match result {
            Ok(token) => {
                heap.ready = Some(token);
                context
                    .geometry
                    .mark_index_ready(token)
                    .map_err(map_geometry)?;
                context.index_heap = Some(heap);
                Ok(upload.first_element)
            }
            Err(error) => {
                context.index_heap = Some(heap);
                let _ = context.geometry.rollback_indices(upload);
                Err(map_allocation(error))
            }
        }
    })
}

pub(super) fn stage_upload(
    context: &mut NativeContext,
    pool: &mut ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    destination: &NativeAllocation,
    destination_offset: u64,
    bytes: &[u8],
) -> Result<CompletionToken, ez_gfx_hal::AllocationError> {
    let completed = completed_transfer_native(context)?;
    for stale in pool.trim(completed) {
        free_native_allocation(context, stale)?;
    }
    let requested = bytes.len() as u64;
    let (capacity, mut allocation) = if let Some(entry) = pool.take(requested, completed) {
        entry
    } else {
        if pool.len() >= 8 {
            return Err(ez_gfx_hal::AllocationError::OutOfMemory);
        }
        let bucket = staging_bucket_size(requested, DEFAULT_STAGING_POLICY)
            .map_err(|_| ez_gfx_hal::AllocationError::OutOfMemory)?;
        let request = AllocationRequest::new(bucket, 16, MemoryClass::Upload, true, None)?;
        (bucket, allocate_native(context, request)?)
    };
    if let Err(error) = write_native(context, &mut allocation, bytes) {
        pool.put(capacity, allocation, None);
        return Err(error);
    }
    let token = match copy_native(
        context,
        &allocation,
        destination,
        0,
        destination_offset,
        requested,
    ) {
        Ok(token) => token,
        Err(error) => {
            pool.put(capacity, allocation, None);
            return Err(error);
        }
    };
    pool.put(capacity, allocation, Some(token));
    Ok(token)
}
