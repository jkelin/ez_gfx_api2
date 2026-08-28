use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

#[cfg(windows)]
use ez_gfx_backend_dx12::native::{NativeContext as Dx12Context, NativeSurface as Dx12Surface};
#[cfg(target_vendor = "apple")]
use ez_gfx_backend_metal::native::{NativeContext as MetalContext, NativeSurface as MetalSurface};
use ez_gfx_backend_vulkan::{
    NativeContext as VulkanContext, NativeSurface as VulkanSurface,
    SurfacePlatform as VulkanPlatform,
};
use ez_gfx_core::{
    Backend,
    handle::{GenerationalArena, HandleParts, LocalHandle, PackedHandle},
};
use ez_gfx_hal::{
    AllocationRequest, BufferTransfer, CompletionToken, DynamicPipelineState, HalError, ImageMip,
    MemoryAllocator, MemoryClass,
};
use ez_gfx_runtime::{
    ContextIdentity, ContextOptions, LifecycleError, ResourceKind, SurfaceOptions, SurfacePlatform,
    SurfaceState,
    frame::FrameRecorder,
    geometry::{GeometryError, GeometryManager},
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer},
    observability::{DiagnosticLevel, Observability, RuntimePhase, RuntimeRecord, RuntimeStatus},
    texture::{
        TextureDecoder, TextureError, TextureId, TextureRegistry, TextureSource, generate_mips,
    },
};

use crate::api::EzGfxResult;

enum NativeContext {
    Vulkan(Box<VulkanContext>),
    #[cfg(windows)]
    Dx12(Box<Dx12Context>),
    #[cfg(target_vendor = "apple")]
    Metal(Box<MetalContext>),
}

enum NativeSurface {
    Vulkan(VulkanSurface),
    #[cfg(windows)]
    Dx12(Dx12Surface),
    #[cfg(target_vendor = "apple")]
    Metal(MetalSurface),
}

enum NativeAllocation {
    Vulkan(ez_gfx_backend_vulkan::NativeAllocation),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativeAllocation),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativeAllocation),
}

enum NativeShader {
    Vulkan(ez_gfx_backend_vulkan::NativeShader),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativeShader),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativeShader),
}

struct ShaderRecord {
    native: NativeShader,
    graphics: Option<(usize, String, usize, String)>,
    compute: Option<(usize, String)>,
    runtime: ez_gfx_runtime::shader::RuntimeShader,
}

enum PendingPipeline {
    Graphics {
        shader: u64,
        commands: Vec<DrawIndexedCommand>,
        bindings: Vec<ez_gfx_runtime::binding::PublicBinding>,
        layout: ez_gfx_runtime::binding::ReflectedBindings,
        state: DynamicPipelineState,
        push_constants: Vec<u8>,
    },
    Compute {
        shader: u64,
        groups: [u32; 3],
        bindings: Vec<ez_gfx_runtime::binding::PublicBinding>,
        layout: ez_gfx_runtime::binding::ReflectedBindings,
        push_constants: Vec<u8>,
    },
}

enum NativeTexture {
    Vulkan(ez_gfx_backend_vulkan::NativeTexture),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativeTexture),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativeTexture),
}

struct SurfaceRecord {
    native: NativeSurface,
    state: SurfaceState,
}

struct GeometryAllocation {
    allocation: NativeAllocation,
    ready: Option<CompletionToken>,
}

struct StagingAllocation {
    capacity: u64,
    allocation: NativeAllocation,
    retirement: Option<CompletionToken>,
}
struct FfiContext {
    identity: ContextIdentity,
    options: ContextOptions,
    native: NativeContext,
    surfaces: HashMap<u64, SurfaceRecord>,
    allocations: HashMap<u64, (u64, NativeAllocation)>,
    shaders: HashMap<u64, ShaderRecord>,
    indirects: HashMap<u64, IndexedIndirectBuffer>,
    textures: HashMap<u64, (TextureId, NativeTexture, u32, u32, u32)>,
    texture_registry: TextureRegistry,
    geometry: GeometryManager,
    vertex_heaps: HashMap<String, GeometryAllocation>,
    index_heap: Option<GeometryAllocation>,
    staging: Vec<StagingAllocation>,
    frame: FrameRecorder,
    frame_indirect: Option<NativeAllocation>,
    last_readback: Vec<u8>,
    frame_readback_texture: Option<u64>,
    active_surface: Option<u64>,
    pending_pipelines: Vec<PendingPipeline>,
    frame_presented: bool,
    observability: Observability,
}

type ContextArena = GenerationalArena<Option<FfiContext>>;
static CONTEXTS: LazyLock<Mutex<ContextArena>> =
    LazyLock::new(|| Mutex::new(GenerationalArena::new()));

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
    let identity = match ContextIdentity::new(local) {
        Ok(identity) => identity,
        Err(_) => {
            let _ = arena.remove(local);
            return Err(EzGfxResult::NativeFailure);
        }
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
        indirects: HashMap::new(),
        texture_registry: TextureRegistry::new(4096, 4096)
            .map_err(|_| EzGfxResult::NativeFailure)?,
        geometry: GeometryManager::new(),
        vertex_heaps: HashMap::new(),
        index_heap: None,
        staging: Vec::new(),
        frame: FrameRecorder::new(1024).map_err(|_| EzGfxResult::NativeFailure)?,
        frame_indirect: None,
        last_readback: Vec::new(),
        frame_readback_texture: None,
        active_surface: None,
        pending_pipelines: Vec::new(),
        frame_presented: false,
        observability: Observability::new(1024, 256).map_err(|_| EzGfxResult::NativeFailure)?,
    });
    Ok(handle)
}

fn runtime_status(status: EzGfxResult) -> RuntimeStatus {
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

fn runtime_record(
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

pub fn poll_runtime_event(context: u64) -> Result<(Option<RuntimeRecord>, u64), EzGfxResult> {
    with_context_mut(context, |context| Ok(context.observability.poll_event()))
}

pub fn poll_diagnostic(
    context: u64,
) -> Result<(Option<(DiagnosticLevel, RuntimeRecord)>, u64), EzGfxResult> {
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
    let Some(mut owned) = slot.take() else {
        return;
    };
    let _ = wait_native_idle(&mut owned.native);
    for (_, surface) in owned.surfaces.drain() {
        destroy_native_surface(&mut owned.native, surface.native);
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
    if let Some(indirect) = owned.frame_indirect.take() {
        let _ = free_native_allocation(&mut owned.native, indirect);
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
            NativeContext::Metal(_) => {
                NativeSurface::Metal(MetalSurface::new(options.window as *mut _).map_err(map_hal)?)
            }
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
        let result = match (&mut context.native, &record.native) {
            (NativeContext::Vulkan(native), NativeSurface::Vulkan(surface)) => {
                native.init_device(Some(surface)).map(|_| ())
            }
            #[cfg(windows)]
            (NativeContext::Dx12(native), NativeSurface::Dx12(surface)) => {
                native.init_device(surface).map(|_| ())
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(native), NativeSurface::Metal(surface)) => {
                native.init_device(surface).map(|_| ())
            }
            _ => Err(HalError::InvalidArgument),
        };
        if result.is_ok() {
            context.active_surface = Some(surface);
        }
        result.map_err(|error| map_native_loss(&context.identity, error))
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
        context.pending_pipelines.clear();
        context.frame_readback_texture = None;
        context.last_readback.clear();
        context.frame_presented = false;
        context.frame.begin().map_err(map_frame)
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

fn destroy_native_surface(context: &mut NativeContext, surface: NativeSurface) {
    if let (NativeContext::Vulkan(context), NativeSurface::Vulkan(surface)) = (context, surface) {
        context.destroy_surface(surface);
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

fn stage_upload(
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

fn with_surface_mut<T>(
    context: u64,
    surface: u64,
    operation: impl FnOnce(&mut SurfaceRecord) -> Result<T, EzGfxResult>,
) -> Result<T, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(surface).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Surface)
            .map_err(map_lifecycle)?;
        operation(
            context
                .surfaces
                .get_mut(&surface)
                .ok_or(EzGfxResult::InvalidContext)?,
        )
    })
}
fn with_context_mut<T>(
    raw: u64,
    operation: impl FnOnce(&mut FfiContext) -> Result<T, EzGfxResult>,
) -> Result<T, EzGfxResult> {
    let (local, _) = context_local(raw)?;
    let mut arena = CONTEXTS.lock().map_err(|_| EzGfxResult::NativeFailure)?;
    let context = arena
        .get_mut(local)
        .map_err(|_| EzGfxResult::InvalidContext)?
        .as_mut()
        .ok_or(EzGfxResult::InvalidContext)?;
    operation(context)
}
fn context_local(raw: u64) -> Result<(LocalHandle, PackedHandle), EzGfxResult> {
    let packed = PackedHandle::from_raw(raw).map_err(|_| EzGfxResult::InvalidContext)?;

    match packed.parts().map_err(|_| EzGfxResult::InvalidContext)? {
        HandleParts::Context(local) => Ok((local, packed)),
        HandleParts::Child { .. } => Err(EzGfxResult::InvalidContext),
    }
}
pub fn acquire_structured(context: u64, size: u64) -> Result<PackedHandle, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let request = AllocationRequest::new(size, 16, MemoryClass::Device, false, None)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let allocation = allocate_native(&mut context.native, request).map_err(map_allocation)?;
        let handle = match context.identity.insert(ResourceKind::Structured) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = free_native_allocation(&mut context.native, allocation);
                return Err(map_lifecycle(error));
            }
        };
        context.allocations.insert(handle.get(), (size, allocation));
        Ok(handle)
    })
}

pub fn acquire_indirect(context: u64, capacity: u32) -> Result<PackedHandle, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let buffer =
            IndexedIndirectBuffer::new(capacity).map_err(|_| EzGfxResult::InvalidArgument)?;
        let handle = context
            .identity
            .insert(ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context.indirects.insert(handle.get(), buffer);
        Ok(handle)
    })
}

pub fn write_indirect(
    context: u64,
    indirect: u64,
    index: u32,
    command: DrawIndexedCommand,
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(indirect).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context
            .indirects
            .get_mut(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?
            .write(index, command)
            .map_err(|_| EzGfxResult::InvalidArgument)
    }))
}

pub fn set_indirect_count(context: u64, indirect: u64, count: u32) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(indirect).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context
            .indirects
            .get_mut(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?
            .set_draw_count(count)
            .map_err(|_| EzGfxResult::InvalidArgument)
    }))
}

pub fn release_indirect(context: u64, indirect: u64) {
    if indirect == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(indirect).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context
            .indirects
            .remove(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?;
        Ok(())
    });
}
pub fn write_structured(context: u64, structured: u64, bytes: &[u8]) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(structured).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Structured)
            .map_err(map_lifecycle)?;
        let FfiContext {
            native,
            staging,
            allocations,
            ..
        } = context;
        let (capacity, allocation) = allocations
            .get(&structured)
            .ok_or(EzGfxResult::InvalidContext)?;
        if bytes.len() as u64 > *capacity {
            return Err(EzGfxResult::InvalidArgument);
        }
        stage_upload(native, staging, allocation, 0, bytes)
            .map(|_| ())
            .map_err(map_allocation)
    }))
}

pub fn release_structured(context: u64, structured: u64) {
    if structured == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(structured).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Structured)
            .map_err(map_lifecycle)?;
        let (_, allocation) = context
            .allocations
            .remove(&structured)
            .ok_or(EzGfxResult::InvalidContext)?;
        free_native_allocation(&mut context.native, allocation).map_err(map_allocation)
    });
}

pub fn load_shader(
    context: u64,
    artifact: &[u8],
    requests: &[ez_gfx_runtime::shader::ShaderRequest],
) -> Result<PackedHandle, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let shader = ez_gfx_runtime::shader::RuntimeShader::load(
            artifact,
            context.options.backend,
            ez_gfx_core::capability::SemanticProfile::V1,
            requests,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        let graphics = shader.graphics_pair().ok().map(|(vertex, fragment)| {
            (
                vertex.0,
                vertex.2.to_owned(),
                fragment.0,
                fragment.2.to_owned(),
            )
        });
        let compute = shader
            .compute_product()
            .ok()
            .map(|product| (product.0, product.2.to_owned()));
        if graphics.is_none() && compute.is_none() {
            return Err(EzGfxResult::InvalidArgument);
        }
        let products = shader
            .products()
            .map(|(_, bytes)| bytes)
            .collect::<Vec<_>>();
        let native = match &context.native {
            NativeContext::Vulkan(native) => {
                NativeShader::Vulkan(native.create_shader(&products).map_err(map_hal)?)
            }
            #[cfg(windows)]
            NativeContext::Dx12(native) => {
                NativeShader::Dx12(native.create_shader(&products).map_err(map_hal)?)
            }
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => {
                NativeShader::Metal(native.create_shader(&products).map_err(map_hal)?)
            }
        };
        let handle = match context.identity.insert(ResourceKind::Shader) {
            Ok(handle) => handle,
            Err(error) => {
                destroy_native_shader(&mut context.native, native);
                return Err(map_lifecycle(error));
            }
        };
        context.shaders.insert(
            handle.get(),
            ShaderRecord {
                native,
                graphics,
                compute,
                runtime: shader,
            },
        );
        Ok(handle)
    })
}

pub fn destroy_shader(context: u64, shader: u64) {
    if shader == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(shader).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        let native = context
            .shaders
            .remove(&shader)
            .ok_or(EzGfxResult::InvalidContext)?;
        destroy_native_shader(&mut context.native, native.native);
        Ok(())
    });
}

fn destroy_native_shader(context: &mut NativeContext, shader: NativeShader) {
    match (context, shader) {
        (NativeContext::Vulkan(context), NativeShader::Vulkan(shader)) => {
            context.destroy_shader(shader)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeShader::Dx12(shader)) => {
            context.destroy_shader(shader)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeShader::Metal(shader)) => {
            context.destroy_shader(shader)
        }
        _ => {}
    }
}

pub fn load_texture(
    context: u64,
    source: TextureSource,
    bytes: &[u8],
    generate: bool,
) -> Result<PackedHandle, EzGfxResult> {
    let decoded = TextureDecoder::decode(source, bytes)
        .and_then(|texture| {
            if generate {
                generate_mips(texture)
            } else {
                Ok(texture)
            }
        })
        .map_err(map_texture)?;
    let mips = decoded
        .mips
        .iter()
        .map(|mip| ImageMip {
            width: mip.width,
            height: mip.height,
            bytes: &mip.rgba8,
        })
        .collect::<Vec<_>>();
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let texture = context
            .texture_registry
            .begin_upload()
            .map_err(map_texture)?;
        let binding = context
            .texture_registry
            .reserved_binding(texture)
            .map_err(map_texture)?;
        let (native, completions) = match &mut context.native {
            NativeContext::Vulkan(native) => native
                .create_texture_rgba8(&mips, binding)
                .map(|(texture, tokens)| (NativeTexture::Vulkan(texture), tokens)),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native
                .create_texture_rgba8(&mips, binding)
                .map(|(texture, tokens)| (NativeTexture::Dx12(texture), tokens)),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native
                .create_texture_rgba8(&mips, binding)
                .map(|(texture, tokens)| (NativeTexture::Metal(texture), tokens)),
        }
        .map_err(map_allocation)?;
        let mut completions = completions.into_iter();
        context
            .texture_registry
            .mark_submitted(
                texture,
                completions.next().ok_or(EzGfxResult::NativeFailure)?,
            )
            .map_err(map_texture)?;
        for (index, completion) in completions.enumerate() {
            context
                .texture_registry
                .mark_mips_submitted(texture, index as u32 + 2, completion)
                .map_err(map_texture)?;
        }
        let handle = match context.identity.insert(ResourceKind::Texture) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = destroy_native_texture(&mut context.native, native);
                return Err(map_lifecycle(error));
            }
        };
        context.textures.insert(
            handle.get(),
            (
                texture,
                native,
                decoded.width,
                decoded.height,
                decoded.mip_count,
            ),
        );
        let record = runtime_record(context, handle.get(), RuntimePhase::Upload, EzGfxResult::Ok);
        context.observability.push_event(record);
        Ok(handle)
    })
}

pub fn texture_binding(context: u64, texture: u64) -> Result<u32, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let completed = completed_transfer_native(&context.native).map_err(map_allocation)?;
        context
            .texture_registry
            .poll(ez_gfx_hal::QueueKind::Transfer, completed)
            .map_err(map_texture)?;
        let (id, _, _, _, _) = context
            .textures
            .get(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        context
            .texture_registry
            .binding_index(*id)
            .map_err(map_texture)
    })
}

pub fn texture_residency(context: u64, texture: u64) -> Result<(u32, u32), EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let completed = completed_transfer_native(&context.native).map_err(map_allocation)?;
        context
            .texture_registry
            .poll(ez_gfx_hal::QueueKind::Transfer, completed)
            .map_err(map_texture)?;
        let (id, _, _, _, total) = context
            .textures
            .get(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        let resident = match context.texture_registry.resident_mips(*id) {
            Ok(resident) => resident,
            Err(TextureError::NotReady) => 0,
            Err(error) => return Err(map_texture(error)),
        };
        Ok((resident, *total))
    })
}

pub fn unload_texture(context: u64, texture: u64) {
    if texture == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        wait_native_idle(&mut context.native).map_err(map_hal)?;
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let (id, allocation, _, _, _) = context
            .textures
            .remove(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        context.texture_registry.unload(id).map_err(map_texture)?;
        destroy_native_texture(&mut context.native, allocation).map_err(map_allocation)
    });
}

pub fn frame_begin(context: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        context.pending_pipelines.clear();
        context.frame.begin().map_err(map_frame)?;
        context.frame_readback_texture = None;
        context.last_readback.clear();
        context.frame_presented = false;
        Ok(())
    }))
}

pub fn frame_enqueue_readback(context: u64, texture: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let (_, _, width, height, _) = context
            .textures
            .get(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        let mut graph = ez_gfx_runtime::graph::FrameGraph::new();
        let resource = graph
            .add_resource(
                ez_gfx_runtime::graph::ResourceDesc::image(
                    *width,
                    *height,
                    1,
                    1,
                    ez_gfx_runtime::target::Format::Rgba8Unorm,
                    1,
                    ez_gfx_runtime::graph::ResourceLifetime::External,
                )
                .map_err(|_| EzGfxResult::InvalidArgument)?,
            )
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let range = ez_gfx_runtime::graph::ImageRange::all(1, 1)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let state = ez_gfx_hal::ResourceState::new(
            ez_gfx_hal::QueueKind::Transfer,
            ez_gfx_hal::ShaderStage::None,
            ez_gfx_hal::ResourceAccess::TransferRead,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        graph
            .add_node(
                ez_gfx_runtime::graph::NodeDesc::new(
                    "texture-readback",
                    ez_gfx_hal::QueueKind::Transfer,
                )
                .access(ez_gfx_runtime::graph::Access::image(resource, range, state)),
            )
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        context
            .frame
            .enqueue(graph.compile().map_err(|_| EzGfxResult::InvalidArgument)?)
            .map_err(map_frame)?;
        context.frame_readback_texture = Some(texture);
        Ok(())
    }))
}
pub fn render_add_graphics(
    context: u64,
    shader: u64,
    indirect: u64,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    state: DynamicPipelineState,
    push_constants: &[u8],
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let shader_handle =
            PackedHandle::from_raw(shader).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(shader_handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        let indirect_handle =
            PackedHandle::from_raw(indirect).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(indirect_handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        validate_binding_handles(context, bindings)?;
        let record = context
            .shaders
            .get(&shader)
            .ok_or(EzGfxResult::InvalidContext)?;
        let graphics = record
            .graphics
            .as_ref()
            .ok_or(EzGfxResult::InvalidArgument)?;
        let layout = record
            .runtime
            .bindings(&graphics.1, ez_gfx_artifact::Stage::Vertex)
            .and_then(|vertex| {
                record
                    .runtime
                    .bindings(&graphics.3, ez_gfx_artifact::Stage::Fragment)
                    .and_then(|fragment| vertex.merge(&fragment))
            })
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        layout
            .validate(bindings)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let commands = context
            .indirects
            .get(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?
            .commands()
            .to_vec();
        if commands.is_empty() || push_constants.len() > 128 || push_constants.len() % 4 != 0 {
            return Err(EzGfxResult::InvalidArgument);
        }
        context.frame.mark_work_enqueued().map_err(map_frame)?;
        context.pending_pipelines.push(PendingPipeline::Graphics {
            shader,
            commands,
            bindings: bindings.to_vec(),
            layout,
            state,
            push_constants: push_constants.to_vec(),
        });
        Ok(())
    }))
}
pub fn render_add_compute(
    context: u64,
    shader: u64,
    groups: [u32; 3],
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    push_constants: &[u8],
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(shader).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        validate_binding_handles(context, bindings)?;
        let record = context
            .shaders
            .get(&shader)
            .ok_or(EzGfxResult::InvalidContext)?;
        let compute = record
            .compute
            .as_ref()
            .ok_or(EzGfxResult::InvalidArgument)?;
        if groups.contains(&0) || push_constants.len() > 128 || push_constants.len() % 4 != 0 {
            return Err(EzGfxResult::InvalidArgument);
        }
        let layout = record
            .runtime
            .bindings(&compute.1, ez_gfx_artifact::Stage::Compute)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        layout
            .validate(bindings)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        context.frame.mark_work_enqueued().map_err(map_frame)?;
        context.pending_pipelines.push(PendingPipeline::Compute {
            shader,
            groups,
            bindings: bindings.to_vec(),
            layout,
            push_constants: push_constants.to_vec(),
        });
        Ok(())
    }))
}

fn validate_binding_handles(
    context: &FfiContext,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
) -> Result<(), EzGfxResult> {
    for binding in bindings {
        let (handle, kind) = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle) => {
                (handle, ResourceKind::Structured)
            }
            ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) => {
                (handle, ResourceKind::Indirect)
            }
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(handle) => {
                (handle, ResourceKind::RenderTarget)
            }
        };
        let packed = PackedHandle::from_raw(handle).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(packed, kind)
            .map_err(map_lifecycle)?;
    }
    Ok(())
}

pub fn frame_submit(context: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let result = (|| {
            let _submission = context.frame.submit().map_err(map_frame)?;
            if let Some(old) = context.frame_indirect.take() {
                free_native_allocation(&mut context.native, old).map_err(map_allocation)?;
            }
            let work = core::mem::take(&mut context.pending_pipelines);
            for pipeline in work {
                match pipeline {
                    PendingPipeline::Compute {
                        shader,
                        groups,
                        bindings,
                        layout,
                        push_constants,
                    } => {
                        let record = context
                            .shaders
                            .get(&shader)
                            .ok_or(EzGfxResult::InvalidContext)?;
                        let (product, entry) = record
                            .compute
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        execute_compute(
                            &mut context.native,
                            &context.allocations,
                            ComputeExecution {
                                shader: &record.native,
                                product: *product,
                                entry,
                                groups,
                                push_constants: &push_constants,
                                layout: &layout,
                                bindings: &bindings,
                            },
                        )
                        .map_err(map_hal)?;
                    }
                    PendingPipeline::Graphics {
                        shader,
                        commands,
                        bindings,
                        layout,
                        state,
                        push_constants,
                    } => {
                        if let Some(old) = context.frame_indirect.take() {
                            free_native_allocation(&mut context.native, old)
                                .map_err(map_allocation)?;
                        }
                        let mut bytes = Vec::with_capacity(commands.len() * 20);
                        for command in &commands {
                            bytes.extend_from_slice(&command.index_count.to_le_bytes());
                            bytes.extend_from_slice(&command.instance_count.to_le_bytes());
                            bytes.extend_from_slice(&command.first_index.to_le_bytes());
                            bytes.extend_from_slice(&command.vertex_offset.to_le_bytes());
                            bytes.extend_from_slice(&command.first_instance.to_le_bytes());
                        }
                        let allocation = allocate_native(
                            &mut context.native,
                            AllocationRequest::new(
                                bytes.len() as u64,
                                4,
                                MemoryClass::Device,
                                false,
                                None,
                            )
                            .map_err(|_| EzGfxResult::InvalidArgument)?,
                        )
                        .map_err(map_allocation)?;
                        if let Err(error) = stage_upload(
                            &mut context.native,
                            &mut context.staging,
                            &allocation,
                            0,
                            &bytes,
                        ) {
                            let _ = free_native_allocation(&mut context.native, allocation);
                            return Err(map_allocation(error));
                        }
                        wait_native_idle(&mut context.native).map_err(map_hal)?;
                        context.frame_indirect = Some(allocation);
                        let indirect = context.frame_indirect.as_ref().expect("just assigned");
                        let index = &context
                            .index_heap
                            .as_ref()
                            .ok_or(EzGfxResult::NotReady)?
                            .allocation;
                        let surface_handle = context.active_surface.ok_or(EzGfxResult::NotReady)?;
                        let extent = context
                            .surfaces
                            .get(&surface_handle)
                            .ok_or(EzGfxResult::InvalidContext)?
                            .state
                            .extent()
                            .ok_or(EzGfxResult::NotReady)?;
                        let record = context
                            .shaders
                            .get(&shader)
                            .ok_or(EzGfxResult::InvalidContext)?;
                        let graphics = record
                            .graphics
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        let mut surface = context
                            .surfaces
                            .remove(&surface_handle)
                            .expect("validated above");
                        let capture_presented = surface.state.snapshot_cache();
                        let result = execute_graphics(
                            &mut context.native,
                            &context.allocations,
                            GraphicsExecution {
                                surface: &mut surface.native,
                                shader: &record.native,
                                graphics,
                                state,
                                index,
                                indirect,
                                draw_count: commands.len() as u32,
                                push_constants: &push_constants,
                                extent,
                                capture_presented,
                                layout: &layout,
                                bindings: &bindings,
                            },
                        );
                        if result.is_ok() && surface.state.snapshot_cache() {
                            context.last_readback = match &surface.native {
                                NativeSurface::Vulkan(surface) => {
                                    surface.presented_rgba8().to_vec()
                                }
                                #[cfg(windows)]
                                NativeSurface::Dx12(surface) => surface.presented_rgba8().to_vec(),
                                #[cfg(target_vendor = "apple")]
                                NativeSurface::Metal(_) => Vec::new(),
                            };
                        }
                        context.surfaces.insert(surface_handle, surface);
                        result.map_err(map_hal)?;
                        context.frame_presented = true;
                    }
                }
            }
            if let Some(texture) = context.frame_readback_texture {
                let (_, native_texture, width, height, _) = context
                    .textures
                    .get(&texture)
                    .ok_or(EzGfxResult::InvalidContext)?;
                context.last_readback = match (&mut context.native, native_texture) {
                    (NativeContext::Vulkan(native), NativeTexture::Vulkan(texture)) => {
                        native.readback_texture_rgba8(texture, *width, *height)
                    }
                    #[cfg(windows)]
                    (NativeContext::Dx12(native), NativeTexture::Dx12(texture)) => {
                        native.readback_texture_rgba8(texture, *width, *height)
                    }
                    #[cfg(target_vendor = "apple")]
                    (NativeContext::Metal(native), NativeTexture::Metal(texture)) => {
                        native.readback_texture_rgba8(texture, *width, *height)
                    }
                    _ => return Err(EzGfxResult::NativeFailure),
                }
                .map_err(map_allocation)?;
            }
            context.frame.finish().map_err(map_frame)?;
            let record = runtime_record(context, 0, RuntimePhase::Submit, EzGfxResult::Ok);
            context.observability.push_event(record);
            Ok(())
        })();
        if let Err(status) = result {
            let record = runtime_record(context, 0, RuntimePhase::Submit, status);
            context
                .observability
                .push_diagnostic(DiagnosticLevel::Error, record);
        }
        result
    }))
}

pub fn frame_readback(context: u64) -> Result<Vec<u8>, EzGfxResult> {
    with_context_mut(context, |context| {
        if context.last_readback.is_empty() {
            return Err(EzGfxResult::NotReady);
        }
        Ok(context.last_readback.clone())
    })
}

fn map_frame(error: ez_gfx_runtime::frame::FrameError) -> EzGfxResult {
    match error {
        ez_gfx_runtime::frame::FrameError::NotRecording
        | ez_gfx_runtime::frame::FrameError::MissingGraph
        | ez_gfx_runtime::frame::FrameError::NotSubmitted => EzGfxResult::NotReady,
        _ => EzGfxResult::InvalidArgument,
    }
}

fn map_texture(error: ez_gfx_runtime::texture::TextureError) -> EzGfxResult {
    use ez_gfx_runtime::texture::TextureError;
    match error {
        TextureError::Unsupported => EzGfxResult::Unsupported,
        TextureError::NotReady => EzGfxResult::NotReady,
        TextureError::TooLarge
        | TextureError::CapacityExceeded
        | TextureError::GenerationExhausted => EzGfxResult::NativeFailure,
        _ => EzGfxResult::InvalidArgument,
    }
}

fn destroy_native_texture(
    context: &mut NativeContext,
    texture: NativeTexture,
) -> Result<(), ez_gfx_hal::AllocationError> {
    match (context, texture) {
        (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
            context.destroy_texture(texture)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
            context.destroy_texture(texture)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
            context.destroy_texture(texture)
        }
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

struct ComputeExecution<'a> {
    shader: &'a NativeShader,
    product: usize,
    entry: &'a str,
    groups: [u32; 3],
    push_constants: &'a [u8],
    layout: &'a ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &'a [ez_gfx_runtime::binding::PublicBinding],
}

struct GraphicsExecution<'a> {
    surface: &'a mut NativeSurface,
    shader: &'a NativeShader,
    graphics: &'a (usize, String, usize, String),
    state: DynamicPipelineState,
    index: &'a NativeAllocation,
    indirect: &'a NativeAllocation,
    draw_count: u32,
    push_constants: &'a [u8],
    extent: (u32, u32),
    capture_presented: bool,
    layout: &'a ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &'a [ez_gfx_runtime::binding::PublicBinding],
}

fn execute_compute(
    context: &mut NativeContext,
    allocations: &HashMap<u64, (u64, NativeAllocation)>,
    request: ComputeExecution<'_>,
) -> Result<(), HalError> {
    let ComputeExecution {
        shader,
        product,
        entry,
        groups,
        push_constants,
        layout,
        bindings,
    } = request;
    let layouts = native_layouts(layout)?;
    match (context, shader) {
        (NativeContext::Vulkan(context), NativeShader::Vulkan(shader)) => {
            let native_bindings = vulkan_bindings(layout, bindings, allocations)?;
            let pipeline = context.create_compute_pipeline(shader, product, entry, &layouts)?;
            let result =
                context.dispatch_compute(&pipeline, groups, push_constants, &native_bindings);
            context.destroy_pipeline(pipeline);
            result
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeShader::Dx12(shader)) => {
            let native_bindings = dx12_bindings(layout, bindings, allocations)?;
            let pipeline = context.create_compute_pipeline(shader, product, &layouts)?;
            context.dispatch_compute(&pipeline, groups, push_constants, &native_bindings)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeShader::Metal(shader)) => {
            let native_bindings = metal_bindings(layout, bindings, allocations)?;
            context.dispatch_compute_shader(
                shader,
                product,
                entry,
                groups,
                push_constants,
                &native_bindings,
            )
        }
        _ => Err(HalError::InvalidArgument),
    }
}
fn execute_graphics(
    context: &mut NativeContext,
    allocations: &HashMap<u64, (u64, NativeAllocation)>,
    request: GraphicsExecution<'_>,
) -> Result<(), HalError> {
    let GraphicsExecution {
        surface,
        shader,
        graphics,
        state,
        index,
        indirect,
        draw_count,
        push_constants,
        extent,
        capture_presented,
        layout,
        bindings,
    } = request;
    let layouts = native_layouts(layout)?;
    match (context, surface, shader, index, indirect) {
        (
            NativeContext::Vulkan(context),
            NativeSurface::Vulkan(surface),
            NativeShader::Vulkan(shader),
            NativeAllocation::Vulkan(index),
            NativeAllocation::Vulkan(indirect),
        ) => {
            context.prepare_surface(surface, extent.0, extent.1)?;
            let native_bindings = vulkan_bindings(layout, bindings, allocations)?;
            let pipeline = context.create_graphics_pipeline(
                shader,
                ez_gfx_backend_vulkan::NativeGraphicsPipelineDesc {
                    vertex_index: graphics.0,
                    fragment_index: graphics.2,
                    state,
                    layouts: &layouts,
                },
            )?;
            let result = context.draw_indexed_present(
                surface,
                ez_gfx_backend_vulkan::NativeDrawIndexed {
                    width: extent.0,
                    height: extent.1,
                    pipeline: &pipeline,
                    index_buffer: index,
                    indirect_buffer: indirect,
                    draw_count,
                    push_constants,
                    bindings: &native_bindings,
                    capture_presented,
                },
            );
            context.destroy_pipeline(pipeline);
            result
        }
        #[cfg(windows)]
        (
            NativeContext::Dx12(context),
            NativeSurface::Dx12(surface),
            NativeShader::Dx12(shader),
            NativeAllocation::Dx12(index),
            NativeAllocation::Dx12(indirect),
        ) => {
            let native_bindings = dx12_bindings(layout, bindings, allocations)?;
            let pipeline = context
                .create_graphics_pipeline(shader, graphics.0, graphics.2, state, &layouts)?;
            context.draw_indexed_present(
                surface,
                ez_gfx_backend_dx12::native::NativeDrawIndexed {
                    width: extent.0,
                    height: extent.1,
                    pipeline: &pipeline,
                    index_buffer: index,
                    indirect_buffer: indirect,
                    draw_count,
                    push_constants,
                    bindings: &native_bindings,
                    capture_presented,
                },
            )
        }
        #[cfg(target_vendor = "apple")]
        (
            NativeContext::Metal(context),
            NativeSurface::Metal(surface),
            NativeShader::Metal(shader),
            NativeAllocation::Metal(index),
            NativeAllocation::Metal(indirect),
        ) => {
            let native_bindings = metal_bindings(layout, bindings, allocations)?;
            context.draw_indexed_shader(
                surface,
                shader,
                graphics,
                state,
                index,
                indirect,
                draw_count,
                push_constants,
                extent,
                &native_bindings,
            )
        }
        _ => Err(HalError::InvalidArgument),
    }
}
fn native_layouts(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
) -> Result<Vec<ez_gfx_hal::ShaderBufferLayout>, HalError> {
    layout
        .requirements()
        .iter()
        .map(|requirement| {
            ez_gfx_hal::ShaderBufferLayout::new(
                requirement.space,
                requirement.binding,
                requirement.descriptor_count,
                requirement.writable,
            )
        })
        .collect()
}

fn vulkan_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<u64, (u64, NativeAllocation)>,
) -> Result<Vec<ez_gfx_backend_vulkan::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle)
                if requirement.descriptor_count == 1 =>
            {
                let (size, allocation) =
                    allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
                let NativeAllocation::Vulkan(allocation) = allocation else {
                    return Err(HalError::InvalidArgument);
                };
                native.push(ez_gfx_backend_vulkan::NativeBufferBinding {
                    allocation,
                    offset: 0,
                    range: *size,
                    writable: requirement.writable,
                });
            }
            _ => return Err(HalError::Unsupported),
        }
    }
    Ok(native)
}

#[cfg(target_vendor = "apple")]
fn metal_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<u64, (u64, NativeAllocation)>,
) -> Result<Vec<ez_gfx_backend_metal::native::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle)
                if requirement.descriptor_count == 1 =>
            {
                let (_, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
                let NativeAllocation::Metal(allocation) = allocation else {
                    return Err(HalError::InvalidArgument);
                };
                native.push(ez_gfx_backend_metal::native::NativeBufferBinding {
                    allocation,
                    offset: 0,
                    index: requirement.binding as usize,
                });
            }
            _ => return Err(HalError::Unsupported),
        }
    }
    Ok(native)
}

#[cfg(windows)]
fn dx12_bindings<'a>(
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    allocations: &'a HashMap<u64, (u64, NativeAllocation)>,
) -> Result<Vec<ez_gfx_backend_dx12::native::NativeBufferBinding<'a>>, HalError> {
    let mut native = Vec::new();
    for requirement in layout.requirements() {
        let public = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(HalError::InvalidArgument)?;
        match public.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle)
                if requirement.descriptor_count == 1 =>
            {
                let (_, allocation) = allocations.get(&handle).ok_or(HalError::InvalidArgument)?;
                let NativeAllocation::Dx12(allocation) = allocation else {
                    return Err(HalError::InvalidArgument);
                };
                native.push(ez_gfx_backend_dx12::native::NativeBufferBinding {
                    allocation,
                    offset: 0,
                    writable: requirement.writable,
                });
            }
            _ => return Err(HalError::Unsupported),
        }
    }
    Ok(native)
}

fn wait_native_idle(context: &mut NativeContext) -> Result<(), HalError> {
    match context {
        NativeContext::Vulkan(context) => context.wait_idle(),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.wait_idle(),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.wait_idle(),
    }
}

fn allocate_native(
    context: &mut NativeContext,
    request: AllocationRequest,
) -> Result<NativeAllocation, ez_gfx_hal::AllocationError> {
    match context {
        NativeContext::Vulkan(context) => context.allocate(request).map(NativeAllocation::Vulkan),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.allocate(request).map(NativeAllocation::Dx12),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.allocate(request).map(NativeAllocation::Metal),
    }
}

fn write_native(
    context: &mut NativeContext,
    allocation: &mut NativeAllocation,
    bytes: &[u8],
) -> Result<(), ez_gfx_hal::AllocationError> {
    match (context, allocation) {
        (NativeContext::Vulkan(context), NativeAllocation::Vulkan(allocation)) => {
            let target = context.mapped_slice_mut(allocation)?;
            if bytes.len() > target.len() {
                return Err(ez_gfx_hal::AllocationError::NativeFailure);
            }
            target[..bytes.len()].copy_from_slice(bytes);
            context.flush(allocation, 0, bytes.len() as u64)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeAllocation::Dx12(allocation)) => {
            let target = context.mapped_slice_mut(allocation)?;
            if bytes.len() > target.len() {
                return Err(ez_gfx_hal::AllocationError::NativeFailure);
            }
            target[..bytes.len()].copy_from_slice(bytes);
            context.flush(allocation, 0, bytes.len() as u64)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeAllocation::Metal(allocation)) => {
            let target = context.mapped_slice_mut(allocation)?;
            if bytes.len() > target.len() {
                return Err(ez_gfx_hal::AllocationError::NativeFailure);
            }
            target[..bytes.len()].copy_from_slice(bytes);
            context.flush(allocation, 0, bytes.len() as u64)
        }
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

fn copy_native(
    context: &mut NativeContext,
    source: &NativeAllocation,
    destination: &NativeAllocation,
    source_offset: u64,
    destination_offset: u64,
    size: u64,
) -> Result<CompletionToken, ez_gfx_hal::AllocationError> {
    match (context, source, destination) {
        (
            NativeContext::Vulkan(context),
            NativeAllocation::Vulkan(source),
            NativeAllocation::Vulkan(destination),
        ) => context.copy_buffer(source, destination, source_offset, destination_offset, size),
        #[cfg(windows)]
        (
            NativeContext::Dx12(context),
            NativeAllocation::Dx12(source),
            NativeAllocation::Dx12(destination),
        ) => context.copy_buffer(source, destination, source_offset, destination_offset, size),
        #[cfg(target_vendor = "apple")]
        (
            NativeContext::Metal(context),
            NativeAllocation::Metal(source),
            NativeAllocation::Metal(destination),
        ) => context.copy_buffer(source, destination, source_offset, destination_offset, size),
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

fn completed_transfer_native(context: &NativeContext) -> Result<u64, ez_gfx_hal::AllocationError> {
    match context {
        NativeContext::Vulkan(context) => context.completed_transfer_value(),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context.completed_transfer_value(),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.completed_transfer_value(),
    }
}

fn free_native_allocation(
    context: &mut NativeContext,
    allocation: NativeAllocation,
) -> Result<(), ez_gfx_hal::AllocationError> {
    match (context, allocation) {
        (NativeContext::Vulkan(context), NativeAllocation::Vulkan(allocation)) => {
            context.free(allocation)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeAllocation::Dx12(allocation)) => {
            context.free(allocation)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeAllocation::Metal(allocation)) => {
            context.free(allocation)
        }
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}
fn map_allocation(error: ez_gfx_hal::AllocationError) -> EzGfxResult {
    match error {
        ez_gfx_hal::AllocationError::ZeroSize
        | ez_gfx_hal::AllocationError::InvalidAlignment
        | ez_gfx_hal::AllocationError::NotHostVisible
        | ez_gfx_hal::AllocationError::InvalidAliasClass => EzGfxResult::InvalidArgument,
        ez_gfx_hal::AllocationError::DeviceLost => EzGfxResult::DeviceLost,
        ez_gfx_hal::AllocationError::OutOfMemory | ez_gfx_hal::AllocationError::NativeFailure => {
            EzGfxResult::NativeFailure
        }
    }
}

fn map_geometry(error: GeometryError) -> EzGfxResult {
    match error {
        GeometryError::UnknownHeap => EzGfxResult::InvalidArgument,
        GeometryError::StagingPoolExhausted => EzGfxResult::NotReady,
        _ => EzGfxResult::InvalidArgument,
    }
}

fn map_lifecycle(error: LifecycleError) -> EzGfxResult {
    match error {
        LifecycleError::DeviceLost | LifecycleError::AlreadyLost => EzGfxResult::DeviceLost,
        _ => EzGfxResult::InvalidContext,
    }
}
fn map_native_loss(identity: &ContextIdentity, error: HalError) -> EzGfxResult {
    if error == HalError::DeviceLost {
        let _ = identity.mark_lost();
    }
    map_hal(error)
}
fn result_status(result: Result<(), EzGfxResult>) -> EzGfxResult {
    match result {
        Ok(()) => EzGfxResult::Ok,
        Err(status) => status,
    }
}
fn map_hal(error: HalError) -> EzGfxResult {
    match error {
        HalError::InvalidArgument => EzGfxResult::InvalidArgument,
        HalError::Unsupported => EzGfxResult::Unsupported,
        HalError::NotReady => EzGfxResult::NotReady,
        HalError::DeviceLost => EzGfxResult::DeviceLost,
        HalError::OutOfMemory | HalError::NativeFailure => EzGfxResult::NativeFailure,
    }
}
