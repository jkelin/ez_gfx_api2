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
    AllocationRequest, BufferRange, BufferTransfer, CompletionToken, DynamicPipelineState,
    ExecutionAction, FrameExecutionBackend, FrameExecutionPlan, HalError, ImageMip,
    MemoryAllocator, MemoryClass, QueueKind, ResourceAccess, ResourceState, ShaderStage,
};
use ez_gfx_runtime::render::{ExecutionError, execute_compiled_graph};
use ez_gfx_runtime::{
    ContextIdentity, ContextOptions, LifecycleError, ResourceKind, SurfaceOptions, SurfacePlatform,
    SurfaceState,
    frame::{ExecutableNode, FrameRecorder},
    geometry::{GeometryError, GeometryManager},
    graph::{
        Access, ImageRange, LoadOp, NodeDesc, PassInfo, ResourceDesc, ResourceId, ResourceLifetime,
        StoreOp,
    },
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer},
    observability::{DiagnosticLevel, Observability, RuntimePhase, RuntimeRecord, RuntimeStatus},
    target::Format,
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
    graphics_layout: Option<ez_gfx_runtime::binding::PipelineLayout>,
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
    size: u64,
}

struct StagingAllocation {
    capacity: u64,
    allocation: NativeAllocation,
    retirement: Option<CompletionToken>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameNativeResource {
    Buffer(u64),
    Texture(u64),
    Surface(u64),
    Depth,
    Index,
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
    texture_ready: HashMap<u64, CompletionToken>,
    geometry: GeometryManager,
    vertex_heaps: HashMap<String, GeometryAllocation>,
    index_heap: Option<GeometryAllocation>,
    staging: Vec<StagingAllocation>,
    frame: FrameRecorder,
    frame_resources: HashMap<u64, ResourceId>,
    frame_native_resources: HashMap<ResourceId, FrameNativeResource>,
    frame_index: Option<ResourceId>,
    frame_surface: Option<ResourceId>,
    frame_depth: Option<ResourceId>,
    frame_has_graphics: bool,
    last_readback: Vec<u8>,
    active_surface: Option<u64>,
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
        context.frame_resources.clear();
        context.frame_native_resources.clear();
        context.frame_index = None;
        context.frame_surface = None;
        context.frame_depth = None;
        context.frame_has_graphics = false;
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
    match (context, surface) {
        (NativeContext::Vulkan(context), NativeSurface::Vulkan(surface)) => {
            context.destroy_surface(surface);
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeSurface::Dx12(surface)) => {
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
        let size = u64::from(capacity)
            .checked_mul(20)
            .ok_or(EzGfxResult::InvalidArgument)?;
        let request = AllocationRequest::new(size, 4, MemoryClass::Device, false, None)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let allocation = allocate_native(&mut context.native, request).map_err(map_allocation)?;
        let handle = match context.identity.insert(ResourceKind::Indirect) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = free_native_allocation(&mut context.native, allocation);
                return Err(map_lifecycle(error));
            }
        };
        context.allocations.insert(handle.get(), (size, allocation));
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
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&command.index_count.to_le_bytes());
        bytes.extend_from_slice(&command.instance_count.to_le_bytes());
        bytes.extend_from_slice(&command.first_index.to_le_bytes());
        bytes.extend_from_slice(&command.vertex_offset.to_le_bytes());
        bytes.extend_from_slice(&command.first_instance.to_le_bytes());
        let (_, allocation) = context
            .allocations
            .get(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?;
        stage_upload(
            &mut context.native,
            &mut context.staging,
            allocation,
            u64::from(index) * 20,
            &bytes,
        )
        .map(|_| ())
        .map_err(map_allocation)
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
        let (_, allocation) = context
            .allocations
            .remove(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?;
        free_native_allocation(&mut context.native, allocation).map_err(map_allocation)
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
        let graphics_layout = graphics
            .as_ref()
            .map(|graphics| {
                ez_gfx_runtime::binding::PipelineLayout::parse(
                    shader.metadata(),
                    context.options.backend,
                    &graphics.3,
                    ez_gfx_artifact::Stage::Fragment,
                )
            })
            .transpose()
            .map_err(|_| EzGfxResult::InvalidArgument)?;
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
                graphics_layout,
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

pub struct TextureConfig {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub sampler: ez_gfx_hal::TextureSamplerDesc,
}

pub fn load_texture(
    context: u64,
    source: TextureSource,
    bytes: &[u8],
    generate: bool,
    config: TextureConfig,
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
    if (config.width != 0 && decoded.width != config.width)
        || (config.height != 0 && decoded.height != config.height)
        || (config.mip_count != 0 && decoded.mip_count != config.mip_count)
    {
        return Err(EzGfxResult::InvalidArgument);
    }
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
        let binding = match context.texture_registry.reserved_binding(texture) {
            Ok(binding) => binding,
            Err(error) => {
                let _ = context.texture_registry.cancel_upload(texture);
                return Err(map_texture(error));
            }
        };
        let created = match &mut context.native {
            NativeContext::Vulkan(native) => native
                .create_texture_rgba8(&mips, binding, config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Vulkan(texture), tokens)),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native
                .create_texture_rgba8(&mips, binding, config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Dx12(texture), tokens)),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native
                .create_texture_rgba8(&mips, binding, config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Metal(texture), tokens)),
        };
        let (native, completions) = match created {
            Ok(created) => created,
            Err(error) => {
                let _ = context.texture_registry.cancel_upload(texture);
                return Err(map_allocation(error));
            }
        };
        let ready = completions
            .last()
            .copied()
            .ok_or(EzGfxResult::NativeFailure)?;
        let mut completions = completions.into_iter();
        let submitted = completions
            .next()
            .ok_or(EzGfxResult::NativeFailure)
            .and_then(|completion| {
                context
                    .texture_registry
                    .mark_submitted(texture, completion)
                    .map_err(map_texture)
            })
            .and_then(|()| {
                for (index, completion) in completions.enumerate() {
                    context
                        .texture_registry
                        .mark_mips_submitted(texture, index as u32 + 2, completion)
                        .map_err(map_texture)?;
                }
                Ok(())
            });
        if let Err(error) = submitted {
            rollback_texture_upload(context, texture, native)?;
            return Err(error);
        }
        let handle = match context.identity.insert(ResourceKind::Texture) {
            Ok(handle) => handle,
            Err(error) => {
                rollback_texture_upload(context, texture, native)?;
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
        context.texture_ready.insert(handle.get(), ready);
        let record = runtime_record(context, handle.get(), RuntimePhase::Upload, EzGfxResult::Ok);
        context.observability.push_event(record);
        Ok(handle)
    })
}

fn rollback_texture_upload(
    context: &mut FfiContext,
    texture: TextureId,
    native: NativeTexture,
) -> Result<(), EzGfxResult> {
    let idle = wait_native_idle(&mut context.native).map_err(map_hal);
    let destroyed = destroy_native_texture(&mut context.native, native).map_err(map_allocation);
    let canceled = context
        .texture_registry
        .cancel_upload(texture)
        .map_err(map_texture);
    idle.and(destroyed).and(canceled)
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
        context.texture_ready.remove(&texture);
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
        context.frame.begin().map_err(map_frame)?;
        context.frame_resources.clear();
        context.frame_native_resources.clear();
        context.frame_index = None;
        context.frame_surface = None;
        context.frame_depth = None;
        context.frame_has_graphics = false;
        context.last_readback.clear();
        context.frame_presented = false;
        Ok(())
    }))
}

fn intern_buffer_resource(
    context: &mut FfiContext,
    handle: u64,
) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_resources.get(&handle) {
        return Ok(*resource);
    }
    let size = context
        .allocations
        .get(&handle)
        .map(|(size, _)| *size)
        .ok_or(EzGfxResult::InvalidContext)?;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context.frame.add_resource(desc).map_err(map_frame)?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(map_frame)?;
    context.frame_resources.insert(handle, resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Buffer(handle));
    Ok(resource)
}

fn intern_surface_resource(context: &mut FfiContext) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_surface {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(EzGfxResult::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(EzGfxResult::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Bgra8Srgb,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context.frame.add_resource(desc).map_err(map_frame)?;
    let present = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::None,
        ResourceAccess::Present,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, present)
        .map_err(map_frame)?;
    context.frame_surface = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Surface(surface));
    Ok(resource)
}

fn intern_depth_resource(context: &mut FfiContext) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_depth {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(EzGfxResult::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(EzGfxResult::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Depth32Float,
        1,
        ResourceLifetime::Transient,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context.frame.add_resource(desc).map_err(map_frame)?;
    context.frame_depth = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Depth);
    Ok(resource)
}

fn intern_index_resource(context: &mut FfiContext) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_index {
        return Ok(resource);
    }
    let heap = context.index_heap.as_ref().ok_or(EzGfxResult::NotReady)?;
    let size = heap.size;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context.frame.add_resource(desc).map_err(map_frame)?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(map_frame)?;
    if let Some(ready) = heap.ready {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(map_frame)?;
    }
    context.frame_index = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Index);
    Ok(resource)
}

fn add_binding_accesses(
    context: &mut FfiContext,
    mut node: NodeDesc,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    queue: QueueKind,
    stage: ShaderStage,
    combined_indirect: Option<u64>,
) -> Result<NodeDesc, EzGfxResult> {
    for requirement in layout.requirements() {
        let binding = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(EzGfxResult::InvalidArgument)?;
        let handle = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle)
            | ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) => handle,
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => {
                return Err(EzGfxResult::Unsupported);
            }
        };
        if combined_indirect == Some(handle) {
            continue;
        }
        let size = context
            .allocations
            .get(&handle)
            .map(|(size, _)| *size)
            .ok_or(EzGfxResult::InvalidContext)?;
        let resource = intern_buffer_resource(context, handle)?;
        let writable = requirement.writable;
        let state = ResourceState::new(
            queue,
            stage,
            if writable {
                ResourceAccess::StorageReadWrite
            } else {
                ResourceAccess::StorageRead
            },
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        node = node.access(Access::buffer(
            resource,
            BufferRange::new(0, size).map_err(|_| EzGfxResult::InvalidArgument)?,
            state,
        ));
    }
    Ok(node)
}

fn intern_texture_resource(
    context: &mut FfiContext,
    texture: u64,
) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_resources.get(&texture) {
        return Ok(*resource);
    }
    let (_, _, width, height, _) = context
        .textures
        .get(&texture)
        .ok_or(EzGfxResult::InvalidContext)?;
    let desc = ResourceDesc::image(
        *width,
        *height,
        1,
        1,
        Format::Rgba8Unorm,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context.frame.add_resource(desc).map_err(map_frame)?;
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, sampled)
        .map_err(map_frame)?;
    if let Some(ready) = context.texture_ready.get(&texture).copied() {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(map_frame)?;
    }
    context.frame_resources.insert(texture, resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Texture(texture));
    Ok(resource)
}

pub fn frame_enqueue_readback(context: u64, texture: u64) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let resource = intern_texture_resource(context, texture)?;
        let range = ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?;
        let state = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferRead,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        context
            .frame
            .record_node(
                NodeDesc::new("texture-readback", QueueKind::Transfer)
                    .access(Access::image(resource, range, state)),
                ExecutableNode::TextureReadback { texture },
            )
            .map_err(map_frame)?;
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
        let draw_count = context
            .indirects
            .get(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?
            .draw_count();
        if draw_count == 0 || push_constants.len() > 128 || !push_constants.len().is_multiple_of(4)
        {
            return Err(EzGfxResult::InvalidArgument);
        }
        let pipeline_layout = *record
            .graphics_layout
            .as_ref()
            .ok_or(EzGfxResult::InvalidArgument)?;
        let surface = intern_surface_resource(context)?;
        let depth = if pipeline_layout.depth_required() {
            Some(intern_depth_resource(context)?)
        } else {
            None
        };
        let (width, height) = context
            .active_surface
            .and_then(|surface| context.surfaces.get(&surface))
            .and_then(|surface| surface.state.extent())
            .ok_or(EzGfxResult::NotReady)?;
        let load = if context.frame_has_graphics {
            LoadOp::Load
        } else {
            LoadOp::Clear
        };
        let pass = PassInfo::new(
            vec![surface],
            depth,
            [0, 0, width, height],
            1,
            load,
            StoreOp::Store,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        let color_state = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::Fragment,
            ResourceAccess::ColorAttachmentWrite,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        let mut node = NodeDesc::new("graphics", QueueKind::Graphics)
            .access(Access::image(
                surface,
                ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
                color_state,
            ))
            .pass(pass);
        if let Some(depth) = depth {
            let depth_state = ResourceState::new(
                QueueKind::Graphics,
                ShaderStage::Fragment,
                ResourceAccess::DepthStencilWrite,
            )
            .map_err(|_| EzGfxResult::InvalidArgument)?;
            node = node.access(Access::image(
                depth,
                ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
                depth_state,
            ));
        }
        let index_resource = intern_index_resource(context)?;
        let index_size = context
            .index_heap
            .as_ref()
            .ok_or(EzGfxResult::NotReady)?
            .size;
        let index_state = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::AllGraphics,
            ResourceAccess::IndexRead,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        node = node.access(Access::buffer(
            index_resource,
            BufferRange::new(0, index_size).map_err(|_| EzGfxResult::InvalidArgument)?,
            index_state,
        ));
        let indirect_binding = layout.requirements().iter().find_map(|requirement| {
            let binding = bindings
                .iter()
                .find(|binding| binding.name == requirement.name)?;
            matches!(
                binding.resource,
                ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) if handle == indirect
            )
            .then_some(requirement.writable)
        });
        let indirect_size = context
            .allocations
            .get(&indirect)
            .map(|(size, _)| *size)
            .ok_or(EzGfxResult::InvalidContext)?;
        let indirect_resource = intern_buffer_resource(context, indirect)?;
        let indirect_state = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::AllGraphics,
            match indirect_binding {
                Some(true) => ResourceAccess::IndirectStorageReadWrite,
                Some(false) => ResourceAccess::IndirectStorageRead,
                None => ResourceAccess::IndirectRead,
            },
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        node = node.access(Access::buffer(
            indirect_resource,
            BufferRange::new(0, indirect_size).map_err(|_| EzGfxResult::InvalidArgument)?,
            indirect_state,
        ));
        node = add_binding_accesses(
            context,
            node,
            &layout,
            bindings,
            QueueKind::Graphics,
            ShaderStage::AllGraphics,
            Some(indirect),
        )?;
        let texture_handles: Vec<_> = context.textures.keys().copied().collect();
        for texture in texture_handles {
            let resource = intern_texture_resource(context, texture)?;
            let sampled = ResourceState::new(
                QueueKind::Graphics,
                ShaderStage::Fragment,
                ResourceAccess::SampledRead,
            )
            .map_err(|_| EzGfxResult::InvalidArgument)?;
            node = node.access(Access::image(
                resource,
                ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
                sampled,
            ));
        }
        context
            .frame
            .record_node(
                node,
                ExecutableNode::Graphics {
                    shader,
                    indirect,
                    draw_count,
                    bindings: bindings.to_vec(),
                    layout,
                    pipeline_layout,
                    state,
                    push_constants: push_constants.to_vec(),
                },
            )
            .map_err(map_frame)?;
        context.frame_has_graphics = true;
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
        if groups.contains(&0)
            || push_constants.len() > 128
            || !push_constants.len().is_multiple_of(4)
        {
            return Err(EzGfxResult::InvalidArgument);
        }
        let layout = record
            .runtime
            .bindings(&compute.1, ez_gfx_artifact::Stage::Compute)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        layout
            .validate(bindings)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let node = add_binding_accesses(
            context,
            NodeDesc::new("compute", QueueKind::Compute),
            &layout,
            bindings,
            QueueKind::Compute,
            ShaderStage::Compute,
            None,
        )?;
        context
            .frame
            .record_node(
                node,
                ExecutableNode::Compute {
                    shader,
                    groups,
                    bindings: bindings.to_vec(),
                    layout,
                    push_constants: push_constants.to_vec(),
                },
            )
            .map_err(map_frame)?;
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
            if context.frame_has_graphics {
                let surface = context.active_surface.ok_or(EzGfxResult::NotReady)?;
                let resource = context.frame_surface.ok_or(EzGfxResult::NotReady)?;
                let present = ResourceState::new(
                    QueueKind::Graphics,
                    ShaderStage::None,
                    ResourceAccess::Present,
                )
                .map_err(|_| EzGfxResult::InvalidArgument)?;
                context
                    .frame
                    .record_node(
                        NodeDesc::new("present", QueueKind::Graphics).access(Access::image(
                            resource,
                            ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
                            present,
                        )),
                        ExecutableNode::Present { surface },
                    )
                    .map_err(map_frame)?;
            }
            let submission = context.frame.submit().map_err(map_frame)?;
            let mut adapter = NativeFrameAdapter { context };
            execute_compiled_graph(&submission.graph, &submission.nodes, &mut adapter)
                .map_err(map_execution)?;
            adapter.context.frame.finish().map_err(map_frame)?;
            let record = runtime_record(adapter.context, 0, RuntimePhase::Submit, EzGfxResult::Ok);
            adapter.context.observability.push_event(record);
            Ok(())
        })();
        if let Err(status) = result {
            context.frame.abort();
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

fn map_execution(error: ExecutionError<EzGfxResult>) -> EzGfxResult {
    match error {
        ExecutionError::Backend(error) => error,
        ExecutionError::MissingPayload { .. }
        | ExecutionError::UnexpectedPayloads
        | ExecutionError::InvalidCompiledRange => EzGfxResult::InvalidArgument,
    }
}

struct NativeFrameAdapter<'a> {
    context: &'a mut FfiContext,
}

impl FrameExecutionBackend<ExecutableNode> for NativeFrameAdapter<'_> {
    type Error = EzGfxResult;

    fn execute(
        &mut self,
        plan: &FrameExecutionPlan,
        payloads: &[ExecutableNode],
    ) -> Result<(), Self::Error> {
        if matches!(self.context.native, NativeContext::Vulkan(_)) {
            return execute_vulkan_frame_plan(self.context, plan, payloads);
        }
        #[cfg(windows)]
        if matches!(self.context.native, NativeContext::Dx12(_)) {
            return execute_dx12_frame_plan(self.context, plan, payloads);
        }
        #[cfg(target_vendor = "apple")]
        if matches!(self.context.native, NativeContext::Metal(_)) {
            return execute_metal_frame_plan(self.context, plan, payloads);
        }
        Err(EzGfxResult::NativeFailure)
    }
}
fn execute_vulkan_frame_plan(
    context: &mut FfiContext,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
) -> Result<(), EzGfxResult> {
    let surface_handle = payloads.iter().find_map(|payload| match payload {
        ExecutableNode::Present { surface } => Some(*surface),
        _ => None,
    });
    let mut surface = surface_handle
        .map(|handle| {
            context
                .surfaces
                .remove(&handle)
                .ok_or(EzGfxResult::InvalidContext)
        })
        .transpose()?;
    let extent = surface
        .as_ref()
        .and_then(|surface| surface.state.extent())
        .unwrap_or((0, 0));
    let capture = surface
        .as_ref()
        .is_some_and(|surface| surface.state.snapshot_cache());

    let index = match context.index_heap.as_ref().map(|heap| &heap.allocation) {
        Some(NativeAllocation::Vulkan(index)) => Some(index),
        Some(_) => return Err(EzGfxResult::NativeFailure),
        None => None,
    };
    if surface
        .as_ref()
        .is_some_and(|surface| !matches!(surface.native, NativeSurface::Vulkan(_)))
    {
        if let (Some(handle), Some(surface)) = (surface_handle, surface) {
            context.surfaces.insert(handle, surface);
        }
        return Err(EzGfxResult::NativeFailure);
    }
    let mut native_surface = surface.as_mut().map(|surface| {
        let NativeSurface::Vulkan(surface) = &mut surface.native else {
            unreachable!("surface variant validated");
        };
        surface
    });
    let NativeContext::Vulkan(native) = &mut context.native else {
        return Err(EzGfxResult::NativeFailure);
    };

    let mut pipelines: Vec<Option<ez_gfx_backend_vulkan::NativePipeline>> =
        (0..payloads.len()).map(|_| None).collect();
    let execution = (|| -> Result<Vec<Vec<u8>>, EzGfxResult> {
        for (node_index, payload) in payloads.iter().enumerate() {
            let pipeline = match payload {
                ExecutableNode::Compute { shader, layout, .. } => {
                    let record = context
                        .shaders
                        .get(shader)
                        .ok_or(EzGfxResult::InvalidContext)?;
                    let compute = record
                        .compute
                        .as_ref()
                        .ok_or(EzGfxResult::InvalidArgument)?;
                    let NativeShader::Vulkan(shader) = &record.native else {
                        return Err(EzGfxResult::NativeFailure);
                    };
                    let layouts = native_layouts(layout).map_err(map_hal)?;
                    native
                        .create_compute_pipeline(shader, compute.0, &compute.1, &layouts)
                        .map_err(map_hal)?
                }
                ExecutableNode::Graphics {
                    shader,
                    layout,
                    pipeline_layout,
                    state,
                    ..
                } => {
                    let record = context
                        .shaders
                        .get(shader)
                        .ok_or(EzGfxResult::InvalidContext)?;
                    let graphics = record
                        .graphics
                        .as_ref()
                        .ok_or(EzGfxResult::InvalidArgument)?;
                    let NativeShader::Vulkan(shader) = &record.native else {
                        return Err(EzGfxResult::NativeFailure);
                    };
                    let layouts = native_layouts(layout).map_err(map_hal)?;
                    native
                        .create_graphics_pipeline(
                            shader,
                            ez_gfx_backend_vulkan::NativeGraphicsPipelineDesc {
                                vertex_index: graphics.0,
                                fragment_index: graphics.2,
                                state: *state,
                                depth_required: pipeline_layout.depth_required(),
                                layouts: &layouts,
                            },
                        )
                        .map_err(map_hal)?
                }
                ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => continue,
            };
            pipelines[node_index] = Some(pipeline);
        }

        let binding_sets = payloads
            .iter()
            .map(|payload| match payload {
                ExecutableNode::Compute {
                    layout, bindings, ..
                }
                | ExecutableNode::Graphics {
                    layout, bindings, ..
                } => vulkan_bindings(layout, bindings, &context.allocations).map_err(map_hal),
                ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => {
                    Ok(Vec::new())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut actions = Vec::with_capacity(plan.actions.len());
        for action in &plan.actions {
            match action {
                ExecutionAction::Wait(wait) => {
                    if let Some(token) = wait.external {
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Wait(token));
                    }
                }
                ExecutionAction::Barrier(barrier) => {
                    let resource = context
                        .frame_native_resources
                        .get(&ResourceId::from_index(barrier.resource))
                        .ok_or(EzGfxResult::InvalidArgument)?;
                    let resource = match *resource {
                        FrameNativeResource::Buffer(handle) => {
                            let NativeAllocation::Vulkan(allocation) = &context
                                .allocations
                                .get(&handle)
                                .ok_or(EzGfxResult::InvalidContext)?
                                .1
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            ez_gfx_backend_vulkan::NativeFrameResource::Buffer(allocation)
                        }
                        FrameNativeResource::Texture(handle) => {
                            let (_, NativeTexture::Vulkan(texture), _, _, _) = context
                                .textures
                                .get(&handle)
                                .ok_or(EzGfxResult::InvalidContext)?
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            ez_gfx_backend_vulkan::NativeFrameResource::Texture(texture)
                        }
                        FrameNativeResource::Surface(_) => {
                            ez_gfx_backend_vulkan::NativeFrameResource::Surface
                        }
                        FrameNativeResource::Depth => {
                            ez_gfx_backend_vulkan::NativeFrameResource::Depth
                        }
                        FrameNativeResource::Index => {
                            ez_gfx_backend_vulkan::NativeFrameResource::Buffer(
                                index.ok_or(EzGfxResult::NotReady)?,
                            )
                        }
                    };
                    actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Barrier {
                        barrier: *barrier,
                        resource,
                    });
                }
                ExecutionAction::BeginPass(pass) => {
                    actions.push(ez_gfx_backend_vulkan::NativeFrameAction::BeginPass(pass));
                }
                ExecutionAction::ExecuteNode(node) => {
                    let index_node = *node as usize;
                    let payload = payloads
                        .get(index_node)
                        .ok_or(EzGfxResult::InvalidArgument)?;
                    match payload {
                        ExecutableNode::Compute {
                            groups,
                            push_constants,
                            ..
                        } => actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Compute(
                            ez_gfx_backend_vulkan::NativeComputeDispatch {
                                pipeline: pipelines[index_node]
                                    .as_ref()
                                    .ok_or(EzGfxResult::InvalidArgument)?,
                                groups: *groups,
                                push_constants,
                                bindings: &binding_sets[index_node],
                            },
                        )),
                        ExecutableNode::Graphics {
                            indirect,
                            draw_count,
                            push_constants,
                            ..
                        } => {
                            let NativeAllocation::Vulkan(indirect) = &context
                                .allocations
                                .get(indirect)
                                .ok_or(EzGfxResult::InvalidContext)?
                                .1
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Graphics(
                                ez_gfx_backend_vulkan::NativeDrawIndexed {
                                    width: extent.0,
                                    height: extent.1,
                                    pipeline: pipelines[index_node]
                                        .as_ref()
                                        .ok_or(EzGfxResult::InvalidArgument)?,
                                    index_buffer: index.ok_or(EzGfxResult::NotReady)?,
                                    indirect_buffer: indirect,
                                    draw_count: *draw_count,
                                    push_constants,
                                    bindings: &binding_sets[index_node],
                                },
                            ));
                        }
                        ExecutableNode::TextureReadback { texture } => {
                            let (_, NativeTexture::Vulkan(texture), width, height, _) = context
                                .textures
                                .get(texture)
                                .ok_or(EzGfxResult::InvalidContext)?
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            actions.push(
                                ez_gfx_backend_vulkan::NativeFrameAction::TextureReadback {
                                    texture,
                                    width: *width,
                                    height: *height,
                                },
                            );
                        }
                        ExecutableNode::Present { .. } => {
                            actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Present);
                        }
                    }
                }
                ExecutionAction::EndPass => {
                    actions.push(ez_gfx_backend_vulkan::NativeFrameAction::EndPass);
                }
            }
        }

        let result = native
            .execute_frame(
                native_surface
                    .as_deref_mut()
                    .map(|surface| (surface, extent)),
                &actions,
                capture,
            )
            .map_err(map_hal);
        drop(actions);
        result
    })();
    for pipeline in pipelines.into_iter().flatten() {
        native.destroy_pipeline(pipeline);
    }
    let outcome = match execution {
        Ok(outputs) => {
            let texture_readbacks = payloads
                .iter()
                .filter(|payload| matches!(payload, ExecutableNode::TextureReadback { .. }))
                .count();
            if texture_readbacks != 0 {
                let Some(readback) = outputs.get(texture_readbacks - 1) else {
                    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
                        context.surfaces.insert(handle, surface);
                    }
                    return Err(EzGfxResult::NativeFailure);
                };
                context.last_readback = readback.clone();
            }
            if capture && let Some(native_surface) = native_surface.as_deref() {
                context.last_readback = native_surface.presented_rgba8().to_vec();
            }
            context.frame_presented = payloads
                .iter()
                .any(|payload| matches!(payload, ExecutableNode::Present { .. }));
            Ok(())
        }
        Err(error) => Err(error),
    };
    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
        context.surfaces.insert(handle, surface);
    }
    outcome
}
#[cfg(windows)]
fn execute_dx12_frame_plan(
    context: &mut FfiContext,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
) -> Result<(), EzGfxResult> {
    let surface_handle = payloads.iter().find_map(|payload| match payload {
        ExecutableNode::Present { surface } => Some(*surface),
        _ => None,
    });
    let mut surface = surface_handle
        .map(|handle| {
            context
                .surfaces
                .remove(&handle)
                .ok_or(EzGfxResult::InvalidContext)
        })
        .transpose()?;
    let extent = surface
        .as_ref()
        .and_then(|surface| surface.state.extent())
        .unwrap_or((0, 0));
    let capture = surface
        .as_ref()
        .is_some_and(|surface| surface.state.snapshot_cache());
    let index = match context.index_heap.as_ref().map(|heap| &heap.allocation) {
        Some(NativeAllocation::Dx12(index)) => Some(index),
        Some(_) => return Err(EzGfxResult::NativeFailure),
        None => None,
    };
    if surface
        .as_ref()
        .is_some_and(|surface| !matches!(surface.native, NativeSurface::Dx12(_)))
    {
        if let (Some(handle), Some(surface)) = (surface_handle, surface) {
            context.surfaces.insert(handle, surface);
        }
        return Err(EzGfxResult::NativeFailure);
    }
    let mut native_surface = surface.as_mut().map(|surface| {
        let NativeSurface::Dx12(surface) = &mut surface.native else {
            unreachable!("surface variant validated");
        };
        surface
    });
    let NativeContext::Dx12(native) = &mut context.native else {
        return Err(EzGfxResult::NativeFailure);
    };

    let execution = (|| -> Result<Vec<Vec<u8>>, EzGfxResult> {
        let binding_sets = payloads
            .iter()
            .map(|payload| match payload {
                ExecutableNode::Compute {
                    layout, bindings, ..
                }
                | ExecutableNode::Graphics {
                    layout, bindings, ..
                } => dx12_bindings(layout, bindings, &context.allocations).map_err(map_hal),
                ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => {
                    Ok(Vec::new())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut pipelines: Vec<Option<ez_gfx_backend_dx12::native::NativePipeline>> =
            (0..payloads.len()).map(|_| None).collect();
        for (node_index, payload) in payloads.iter().enumerate() {
            let pipeline = match payload {
                ExecutableNode::Compute { shader, layout, .. } => {
                    let record = context
                        .shaders
                        .get(shader)
                        .ok_or(EzGfxResult::InvalidContext)?;
                    let compute = record
                        .compute
                        .as_ref()
                        .ok_or(EzGfxResult::InvalidArgument)?;
                    let NativeShader::Dx12(shader) = &record.native else {
                        return Err(EzGfxResult::NativeFailure);
                    };
                    let layouts = native_layouts(layout).map_err(map_hal)?;
                    native
                        .create_compute_pipeline(shader, compute.0, &layouts)
                        .map_err(map_hal)?
                }
                ExecutableNode::Graphics {
                    shader,
                    layout,
                    pipeline_layout,
                    state,
                    ..
                } => {
                    let record = context
                        .shaders
                        .get(shader)
                        .ok_or(EzGfxResult::InvalidContext)?;
                    let graphics = record
                        .graphics
                        .as_ref()
                        .ok_or(EzGfxResult::InvalidArgument)?;
                    let NativeShader::Dx12(shader) = &record.native else {
                        return Err(EzGfxResult::NativeFailure);
                    };
                    let layouts = native_layouts(layout).map_err(map_hal)?;
                    native
                        .create_graphics_pipeline(
                            shader,
                            graphics.0,
                            graphics.2,
                            *state,
                            pipeline_layout.depth_required(),
                            &layouts,
                        )
                        .map_err(map_hal)?
                }
                ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => {
                    continue;
                }
            };
            pipelines[node_index] = Some(pipeline);
        }

        let mut actions = Vec::with_capacity(plan.actions.len());
        for action in &plan.actions {
            match action {
                ExecutionAction::Wait(wait) => {
                    if let Some(token) = wait.external {
                        actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Wait(token));
                    }
                }
                ExecutionAction::Barrier(barrier) => {
                    let resource = context
                        .frame_native_resources
                        .get(&ResourceId::from_index(barrier.resource))
                        .ok_or(EzGfxResult::InvalidArgument)?;
                    let resource = match *resource {
                        FrameNativeResource::Buffer(handle) => {
                            let NativeAllocation::Dx12(allocation) = &context
                                .allocations
                                .get(&handle)
                                .ok_or(EzGfxResult::InvalidContext)?
                                .1
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            ez_gfx_backend_dx12::native::NativeFrameResource::Buffer(allocation)
                        }
                        FrameNativeResource::Texture(handle) => {
                            let (_, NativeTexture::Dx12(texture), _, _, _) = context
                                .textures
                                .get(&handle)
                                .ok_or(EzGfxResult::InvalidContext)?
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            ez_gfx_backend_dx12::native::NativeFrameResource::Texture(texture)
                        }
                        FrameNativeResource::Surface(_) => {
                            ez_gfx_backend_dx12::native::NativeFrameResource::Surface
                        }
                        FrameNativeResource::Depth => {
                            ez_gfx_backend_dx12::native::NativeFrameResource::Depth
                        }
                        FrameNativeResource::Index => {
                            ez_gfx_backend_dx12::native::NativeFrameResource::Buffer(
                                index.ok_or(EzGfxResult::NotReady)?,
                            )
                        }
                    };
                    actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Barrier {
                        barrier: *barrier,
                        resource,
                    });
                }
                ExecutionAction::BeginPass(pass) => {
                    actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::BeginPass(
                        pass,
                    ));
                }
                ExecutionAction::ExecuteNode(node) => {
                    let index_node = *node as usize;
                    match payloads
                        .get(index_node)
                        .ok_or(EzGfxResult::InvalidArgument)?
                    {
                        ExecutableNode::Compute {
                            groups,
                            push_constants,
                            ..
                        } => actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Compute(
                            ez_gfx_backend_dx12::native::NativeComputeDispatch {
                                pipeline: pipelines[index_node]
                                    .as_ref()
                                    .ok_or(EzGfxResult::InvalidArgument)?,
                                groups: *groups,
                                push_constants,
                                bindings: &binding_sets[index_node],
                            },
                        )),
                        ExecutableNode::Graphics {
                            indirect,
                            draw_count,
                            push_constants,
                            ..
                        } => {
                            let NativeAllocation::Dx12(indirect) = &context
                                .allocations
                                .get(indirect)
                                .ok_or(EzGfxResult::InvalidContext)?
                                .1
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Graphics(
                                ez_gfx_backend_dx12::native::NativeDrawIndexed {
                                    width: extent.0,
                                    height: extent.1,
                                    pipeline: pipelines[index_node]
                                        .as_ref()
                                        .ok_or(EzGfxResult::InvalidArgument)?,
                                    index_buffer: index.ok_or(EzGfxResult::NotReady)?,
                                    indirect_buffer: indirect,
                                    draw_count: *draw_count,
                                    push_constants,
                                    bindings: &binding_sets[index_node],
                                },
                            ));
                        }
                        ExecutableNode::TextureReadback { texture } => {
                            let (_, NativeTexture::Dx12(texture), width, height, _) = context
                                .textures
                                .get(texture)
                                .ok_or(EzGfxResult::InvalidContext)?
                            else {
                                return Err(EzGfxResult::NativeFailure);
                            };
                            actions.push(
                                ez_gfx_backend_dx12::native::NativeFrameAction::TextureReadback {
                                    texture,
                                    width: *width,
                                    height: *height,
                                },
                            );
                        }
                        ExecutableNode::Present { .. } => {
                            actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Present);
                        }
                    }
                }
                ExecutionAction::EndPass => {
                    actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::EndPass);
                }
            }
        }
        native
            .execute_frame(
                native_surface
                    .as_deref_mut()
                    .map(|surface| (surface, extent)),
                &actions,
                capture,
            )
            .map_err(map_hal)
    })();

    let outcome = match execution {
        Ok(outputs) => {
            let texture_readbacks = payloads
                .iter()
                .filter(|payload| matches!(payload, ExecutableNode::TextureReadback { .. }))
                .count();
            if texture_readbacks != 0 {
                let Some(readback) = outputs.get(texture_readbacks - 1) else {
                    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
                        context.surfaces.insert(handle, surface);
                    }
                    return Err(EzGfxResult::NativeFailure);
                };
                context.last_readback = readback.clone();
            }
            if capture && let Some(native_surface) = native_surface.as_deref() {
                context.last_readback = native_surface.presented_rgba8().to_vec();
            }
            context.frame_presented = payloads
                .iter()
                .any(|payload| matches!(payload, ExecutableNode::Present { .. }));
            Ok(())
        }
        Err(error) => Err(error),
    };
    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
        context.surfaces.insert(handle, surface);
    }
    outcome
}

#[cfg(target_vendor = "apple")]
fn execute_metal_frame_plan(
    context: &mut FfiContext,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
) -> Result<(), EzGfxResult> {
    let surface_handle = payloads.iter().find_map(|payload| match payload {
        ExecutableNode::Present { surface } => Some(*surface),
        _ => None,
    });
    let mut surface = surface_handle
        .map(|handle| {
            context
                .surfaces
                .remove(&handle)
                .ok_or(EzGfxResult::InvalidContext)
        })
        .transpose()?;
    let extent = surface
        .as_ref()
        .and_then(|surface| surface.state.extent())
        .unwrap_or((0, 0));
    let capture = surface
        .as_ref()
        .is_some_and(|surface| surface.state.snapshot_cache());

    let mut binding_sets = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let bindings = match payload {
            ExecutableNode::Graphics {
                layout, bindings, ..
            }
            | ExecutableNode::Compute {
                layout, bindings, ..
            } => metal_bindings(layout, bindings, &context.allocations).map_err(map_hal)?,
            ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => Vec::new(),
        };
        binding_sets.push(bindings);
    }
    let native_textures: Vec<_> = context
        .textures
        .values()
        .map(|(_, texture, _, _, _)| match texture {
            NativeTexture::Metal(texture) => Ok(texture),
            _ => Err(EzGfxResult::NativeFailure),
        })
        .collect::<Result<_, _>>()?;
    let index = match context.index_heap.as_ref().map(|heap| &heap.allocation) {
        Some(NativeAllocation::Metal(index)) => Some(index),
        Some(_) => return Err(EzGfxResult::NativeFailure),
        None => None,
    };

    let mut actions = Vec::with_capacity(plan.actions.len());
    for action in &plan.actions {
        match action {
            ExecutionAction::Wait(wait) => {
                if let Some(token) = wait.external {
                    actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Wait(token));
                }
            }
            ExecutionAction::Barrier(barrier) => {
                let resource = context
                    .frame_native_resources
                    .get(&ResourceId::from_index(barrier.resource))
                    .ok_or(EzGfxResult::InvalidArgument)?;
                let resource = match *resource {
                    FrameNativeResource::Buffer(handle) => {
                        let NativeAllocation::Metal(allocation) = &context
                            .allocations
                            .get(&handle)
                            .ok_or(EzGfxResult::InvalidContext)?
                            .1
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        ez_gfx_backend_metal::native::NativeFrameResource::Buffer(allocation)
                    }
                    FrameNativeResource::Texture(handle) => {
                        let (_, NativeTexture::Metal(texture), _, _, _) = context
                            .textures
                            .get(&handle)
                            .ok_or(EzGfxResult::InvalidContext)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        ez_gfx_backend_metal::native::NativeFrameResource::Texture(texture)
                    }
                    FrameNativeResource::Surface(_) => {
                        ez_gfx_backend_metal::native::NativeFrameResource::Surface
                    }
                    FrameNativeResource::Depth => {
                        ez_gfx_backend_metal::native::NativeFrameResource::Depth
                    }
                    FrameNativeResource::Index => {
                        ez_gfx_backend_metal::native::NativeFrameResource::Buffer(
                            index.ok_or(EzGfxResult::NotReady)?,
                        )
                    }
                };
                actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                });
            }
            ExecutionAction::BeginPass(pass) => {
                actions.push(ez_gfx_backend_metal::native::NativeFrameAction::BeginPass(
                    pass,
                ));
            }
            ExecutionAction::ExecuteNode(node) => {
                let index_node = *node as usize;
                let payload = payloads
                    .get(index_node)
                    .ok_or(EzGfxResult::InvalidArgument)?;
                match payload {
                    ExecutableNode::Graphics {
                        shader,
                        indirect,
                        draw_count,
                        layout: _,
                        pipeline_layout,
                        state,
                        push_constants,
                        ..
                    } => {
                        let record = context
                            .shaders
                            .get(shader)
                            .ok_or(EzGfxResult::InvalidContext)?;
                        let graphics = record
                            .graphics
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        let NativeShader::Metal(shader) = &record.native else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        let NativeAllocation::Metal(indirect) = &context
                            .allocations
                            .get(indirect)
                            .ok_or(EzGfxResult::InvalidContext)?
                            .1
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        let texture_heap = pipeline_layout
                            .texture_heap()
                            .map(|layout| {
                                ez_gfx_hal::ShaderTextureHeapLayout::new(
                                    layout.space,
                                    layout.binding,
                                    layout.capacity,
                                    layout.argument_stride,
                                    layout.texture_argument_offset,
                                    layout.sampler_argument_offset,
                                )
                            })
                            .transpose()
                            .map_err(map_hal)?;
                        actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Graphics(
                            ez_gfx_backend_metal::native::NativeGraphicsDraw {
                                shader,
                                graphics,
                                depth_required: pipeline_layout.depth_required(),
                                texture_heap,
                                state: *state,
                                index: index.ok_or(EzGfxResult::NotReady)?,
                                indirect,
                                draw_count: *draw_count,
                                push_constants,
                                bindings: &binding_sets[index_node],
                                textures: &native_textures,
                            },
                        ));
                    }
                    ExecutableNode::Compute {
                        shader,
                        groups,
                        push_constants,
                        ..
                    } => {
                        let record = context
                            .shaders
                            .get(shader)
                            .ok_or(EzGfxResult::InvalidContext)?;
                        let (product, entry) = record
                            .compute
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        let NativeShader::Metal(shader) = &record.native else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Compute(
                            ez_gfx_backend_metal::native::NativeComputeDispatch {
                                shader,
                                product_index: *product,
                                entry,
                                groups: *groups,
                                push_constants,
                                bindings: &binding_sets[index_node],
                            },
                        ));
                    }
                    ExecutableNode::TextureReadback { texture } => {
                        let (_, NativeTexture::Metal(texture), width, height, _) = context
                            .textures
                            .get(texture)
                            .ok_or(EzGfxResult::InvalidContext)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        actions.push(
                            ez_gfx_backend_metal::native::NativeFrameAction::TextureReadback {
                                texture,
                                width: *width,
                                height: *height,
                            },
                        );
                    }
                    ExecutableNode::Present { surface } => {
                        if Some(*surface) != surface_handle {
                            return Err(EzGfxResult::InvalidArgument);
                        }
                        actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Present);
                    }
                }
            }
            ExecutionAction::EndPass => {
                actions.push(ez_gfx_backend_metal::native::NativeFrameAction::EndPass);
            }
        }
    }

    let result = match (
        &mut context.native,
        surface.as_mut().map(|surface| &mut surface.native),
    ) {
        (NativeContext::Metal(native), Some(NativeSurface::Metal(surface))) => native
            .execute_frame(Some((surface, extent)), &actions, capture)
            .map_err(map_hal),
        (NativeContext::Metal(native), None) => {
            native.execute_frame(None, &actions, false).map_err(map_hal)
        }
        _ => Err(EzGfxResult::NativeFailure),
    };
    if let Ok(Some(readback)) = &result {
        context.last_readback = readback.clone();
    }
    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
        context.surfaces.insert(handle, surface);
    }
    result?;
    context.frame_presented = payloads
        .iter()
        .any(|payload| matches!(payload, ExecutableNode::Present { .. }));
    Ok(())
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
            | ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle)
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
            | ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle)
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
            | ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle)
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
