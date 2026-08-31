use super::{
    AllocationRequest, Backend, CONTEXTS, CompletionToken, ContextIdentity, ContextOptions,
    DiagnosticLevel, Dx12Context, Dx12Surface, EzGfxResult, FfiContext, FrameRecorder,
    GeometryAllocation, GeometryManager, HalError, HashMap, MemoryClass, NativeAllocation,
    NativeContext, NativeSurface, Observability, PackedHandle, ResourceKind, RuntimePhase,
    RuntimeRecord, RuntimeStatus, StagingAllocation, SurfaceOptions, SurfacePlatform,
    SurfaceRecord, SurfaceState, TextureRegistry, VulkanContext, VulkanPlatform, allocate_native,
    completed_transfer_native, context_local, copy_native, destroy_native_pipeline,
    destroy_native_shader, destroy_native_texture, free_native_allocation, map_allocation,
    map_frame, map_geometry, map_hal, map_lifecycle, map_native_loss, result_status,
    wait_native_idle, with_context_mut, with_surface_mut, write_native,
};

pub fn create_context(options: ContextOptions) -> Result<PackedHandle, EzGfxResult> {
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
    let mut arena = CONTEXTS.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    let local = arena.insert(None).map_err(|_| EzGfxResult::NativeFailure)?;
    let Ok(identity) = ContextIdentity::new(local) else {
        let _ = arena.remove(local);
        return Err(EzGfxResult::NativeFailure);
    };
    let handle = identity.context_handle();
    *arena
        .get_mut(local)
        .map_err(|_| EzGfxResult::NativeFailure)? = Some(FfiContext {
        identity,
        options,
        native,
        surfaces: HashMap::new(),
        allocations: HashMap::new(),
        shaders: HashMap::new(),
        textures: HashMap::new(),
        pipelines: HashMap::new(),
        graphics_format: None,
        indirects: HashMap::new(),
        texture_registry: TextureRegistry::new(
            ez_gfx_runtime::binding::MAX_TEXTURE_HEAP_CAPACITY,
            ez_gfx_runtime::binding::MAX_TEXTURE_HEAP_CAPACITY,
        )
        .map_err(|_| EzGfxResult::NativeFailure)?,
        texture_ready: HashMap::new(),
        geometry: GeometryManager::new(),
        vertex_heaps: HashMap::new(),
        index_heap: None,
        staging: Vec::new(),
        frame: FrameRecorder::new(1024).map_err(|_| EzGfxResult::NativeFailure)?,
        frame_resources: HashMap::new(),
        frame_native_resources: HashMap::new(),
        frame_index: None,
        frame_surface: None,
        frame_depth: None,
        frame_has_graphics: false,
        last_readback: Vec::new(),
        active_surface: None,
        frame_presented: false,
        observability: Observability::new(1024, 256).map_err(|_| EzGfxResult::NativeFailure)?,
    });
    Ok(handle)
}

pub(super) fn runtime_status(status: EzGfxResult) -> RuntimeStatus {
    match status {
        EzGfxResult::Ok => RuntimeStatus::Ok,
        EzGfxResult::InvalidArgument | EzGfxResult::InvalidContext => {
            RuntimeStatus::InvalidArgument
        }
        EzGfxResult::NotReady => RuntimeStatus::NotReady,
        EzGfxResult::Unsupported => RuntimeStatus::Unsupported,
        EzGfxResult::NativeFailure => RuntimeStatus::NativeFailure,
        EzGfxResult::DeviceLost => RuntimeStatus::DeviceLost,
    }
}

pub(super) fn runtime_record(
    context: &mut FfiContext,
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

pub fn poll_runtime_event(context: u64) -> Result<(Option<RuntimeRecord>, u64), EzGfxResult> {
    with_context_mut(context, |context| Ok(context.observability.poll_event()))
}

pub fn poll_diagnostic(context: u64) -> Result<DiagnosticPoll, EzGfxResult> {
    with_context_mut(context, |context| {
        Ok(context.observability.poll_diagnostic())
    })
}

pub fn wait_idle(context: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
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

pub fn destroy_context(context: u64) {
    if context == 0 {
        return;
    }
    let Ok((local, _)) = context_local(context) else {
        return;
    };
    let Ok(mut arena) = CONTEXTS.lock() else {
        return;
    };
    let Ok(slot) = arena.get_mut(local) else {
        return;
    };
    // A void destroy from the wrong thread cannot report failure, so preserve the live context.
    if slot
        .as_ref()
        .is_some_and(|owned| owned.identity.check_thread_and_health().is_err())
    {
        return;
    }
    let Some(mut owned) = slot.take() else {
        return;
    };
    let _ = wait_native_idle(&mut owned.native);
    for (_, surface) in owned.surfaces.drain() {
        destroy_native_surface(&mut owned.native, surface.native);
    }
    for (_, pipeline) in owned.pipelines.drain() {
        destroy_native_pipeline(&mut owned.native, pipeline);
    }
    for (_, shader) in owned.shaders.drain() {
        destroy_native_shader(&mut owned.native, shader.native);
    }
    for (_, (_, texture, _, _, _)) in owned.textures.drain() {
        let _ = destroy_native_texture(&mut owned.native, texture);
    }
    for (_, (_, allocation)) in owned.allocations.drain() {
        let _ = free_native_allocation(&mut owned.native, allocation);
    }
    for (_, heap) in owned.vertex_heaps.drain() {
        let _ = free_native_allocation(&mut owned.native, heap.allocation);
    }
    if let Some(heap) = owned.index_heap.take() {
        let _ = free_native_allocation(&mut owned.native, heap.allocation);
    }
    for staging in owned.staging.drain(..) {
        let _ = free_native_allocation(&mut owned.native, staging.allocation);
    }
    owned.identity.invalidate_resources();
    drop(owned);
    let _ = arena.remove(local);
}
pub fn create_surface(context: u64, options: SurfaceOptions) -> Result<PackedHandle, EzGfxResult> {
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
        context
            .surfaces
            .insert(handle.get(), SurfaceRecord { native, state });
        Ok(handle)
    })
}

pub fn init_device(context: u64, surface: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(surface).map_err(|_| EzGfxResult::InvalidContext)?;
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

pub fn resize_surface(context: u64, surface: u64, width: u32, height: u32) -> EzGfxResult {
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
pub fn surface_extent(context: u64, surface: u64) -> Result<(u32, u32), EzGfxResult> {
    with_surface_mut(context, surface, |record| {
        record.state.extent().ok_or(EzGfxResult::NotReady)
    })
}
pub fn surface_resize_pending(context: u64, surface: u64) -> Result<bool, EzGfxResult> {
    with_surface_mut(context, surface, |record| Ok(record.state.resize_pending()))
}
pub fn set_snapshot_cache(context: u64, surface: u64, enabled: bool) -> EzGfxResult {
    result_status(with_surface_mut(context, surface, |record| {
        record.state.set_snapshot_cache(enabled);
        Ok(())
    }))
}

pub fn destroy_surface(context: u64, surface: u64) {
    if surface == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(surface).map_err(|_| EzGfxResult::InvalidContext)?;
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
pub fn begin_render(context: u64, surface: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(surface).map_err(|_| EzGfxResult::InvalidContext)?;
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

pub fn present(context: u64) -> EzGfxResult {
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
        _ => {}
    }
}

pub fn create_vertex_heap(context: u64, name: &str, capacity: u64, stride: u64) -> EzGfxResult {
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

pub fn destroy_vertex_heap(context: u64, name: &str) {
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

pub fn create_index_heap(context: u64, capacity: u64) -> EzGfxResult {
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

pub fn destroy_index_heap(context: u64) {
    let _ = with_context_mut(context, |context| {
        context.geometry.remove_index_heap().map_err(map_geometry)?;
        let heap = context
            .index_heap
            .take()
            .ok_or(EzGfxResult::InvalidArgument)?;
        free_native_allocation(&mut context.native, heap.allocation).map_err(map_allocation)
    });
}

pub fn upload_vertices(
    context: u64,
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

pub fn upload_indices(context: u64, count: u32, bytes: &[u8]) -> Result<u32, EzGfxResult> {
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
    pool: &mut Vec<StagingAllocation>,
    destination: &NativeAllocation,
    destination_offset: u64,
    bytes: &[u8],
) -> Result<CompletionToken, ez_gfx_hal::AllocationError> {
    let completed = completed_transfer_native(context)?;
    let index = if let Some(index) = pool.iter().position(|entry| {
        entry.capacity >= bytes.len() as u64
            && entry
                .retirement
                .is_none_or(|token| token.value <= completed)
    }) {
        index
    } else {
        if pool.len() >= 8 {
            return Err(ez_gfx_hal::AllocationError::OutOfMemory);
        }
        let request =
            AllocationRequest::new(bytes.len() as u64, 16, MemoryClass::Upload, true, None)?;
        let allocation = allocate_native(context, request)?;
        pool.push(StagingAllocation {
            capacity: bytes.len() as u64,
            allocation,
            retirement: None,
        });
        pool.len() - 1
    };
    let staging = &mut pool[index];
    write_native(context, &mut staging.allocation, bytes)?;
    let token = copy_native(
        context,
        &staging.allocation,
        destination,
        0,
        destination_offset,
        bytes.len() as u64,
    )?;
    staging.retirement = Some(token);
    Ok(token)
}
