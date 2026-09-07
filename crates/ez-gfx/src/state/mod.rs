use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
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
    capability::AdapterInfo,
    handle::{
        ContextHandle, GenerationalArena, HandleParts, IndirectBufferHandle, LocalHandle,
        PackedHandle, RenderTargetHandle, ShaderHandle, StructuredBufferHandle, SurfaceHandle,
        TextureHandle,
    },
};
use ez_gfx_hal::{
    AllocationRequest, BufferRange, BufferTransfer, CompletionToken, DEFAULT_STAGING_POLICY,
    DynamicPipelineState, ExecutionAction, ExecutionBarrier, ExecutionPass, FrameExecutionBackend,
    FrameExecutionPlan, HalError, ImageMip, MemoryAllocator, MemoryClass, QueueKind,
    ResourceAccess, ResourceState, SURFACE_DEFAULT_CLEAR, ShaderStage, TextureFormat,
    TextureRegion, staging_bucket_size,
};
use ez_gfx_runtime::render::{ExecutionError, execute_compiled_graph};
use ez_gfx_runtime::{
    AdapterCatalog, AdapterReport, AdapterSelection, ContextIdentity, ContextOptions,
    LifecycleError, ResourceKind, RuntimeError, SurfaceOptions, SurfacePlatform, SurfaceState,
    admission_report,
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
        DecodedTexture, TextureDecoder, TextureDestination, TextureError, TextureId,
        TextureRegistry, TextureSource, TextureUploadTelemetry, TextureUploadTelemetrySnapshot,
        generate_mips,
    },
};

use crate::EzGfxResult;

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

enum NativePipeline {
    Vulkan(ez_gfx_backend_vulkan::NativePipeline),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativePipeline),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativePipeline),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum PipelineKey {
    Compute {
        backend: Backend,
        shader: ShaderHandle,
        shader_digest: [u8; 32],
        product: usize,
        entry: String,
        layouts: Vec<ez_gfx_hal::ShaderBufferLayout>,
    },
    Graphics {
        backend: Backend,
        shader: ShaderHandle,
        shader_digest: [u8; 32],
        vertex_product: usize,
        vertex_entry: String,
        fragment_product: usize,
        fragment_entry: String,
        layouts: Vec<ez_gfx_hal::ShaderBufferLayout>,
        texture_heap: Option<ez_gfx_hal::ShaderTextureHeapLayout>,
        state: DynamicPipelineState,
        depth_required: bool,
        color_format: u32,
        depth_format: u32,
        sample_count: u8,
    },
}

impl PipelineKey {
    fn shader(&self) -> ShaderHandle {
        // Both variants always carry their owning generational shader handle.
        match self {
            Self::Compute { shader, .. } | Self::Graphics { shader, .. } => *shader,
        }
    }
}

const MAX_PIPELINE_CACHE_ENTRIES: usize = 1024;

struct ShaderRecord {
    native: NativeShader,
    digest: [u8; 32],
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

struct RetiredTexture {
    id: TextureId,
    native: NativeTexture,
    completion: CompletionToken,
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

struct PendingTexture {
    id: TextureId,
    cancelled: Arc<AtomicBool>,
    config: TextureConfig,
    admitted_at: Instant,
}

struct DecodedTextureJob {
    handle: TextureHandle,
    decoded: Result<DecodedTexture, TextureError>,
}

struct AsyncTextureState {
    pool: ez_gfx_assets::CpuPool,
    ready_tx: crossbeam_channel::Sender<DecodedTextureJob>,
    ready_rx: crossbeam_channel::Receiver<DecodedTextureJob>,
    #[cfg(test)]
    decode_gate: Option<Arc<std::sync::Barrier>>,
}

impl AsyncTextureState {
    fn new_with_workers(workers: u32) -> Result<Self, EzGfxResult> {
        // Zero preserves the historical default topology; an explicit count is
        // honored verbatim so embedders can pin decode concurrency. Counts above
        // the pool admission cap fail here, before Rayon spawns one OS thread
        // per worker, so both the Rust option and the C descriptor fail fast
        // with InvalidArgument instead of grinding thread creation.
        let threads = if workers == 0 {
            std::thread::available_parallelism()
                .map_or(2, usize::from)
                .saturating_sub(1)
                .max(1)
        } else {
            let threads = usize::try_from(workers).map_err(|_| EzGfxResult::InvalidArgument)?;
            if threads > ez_gfx_assets::MAX_CPU_POOL_THREADS {
                return Err(EzGfxResult::InvalidArgument);
            }
            threads
        };
        let (ready_tx, ready_rx) = crossbeam_channel::bounded(64);
        Ok(Self {
            pool: ez_gfx_assets::CpuPool::new(
                threads,
                64,
                ez_gfx_runtime::texture::MAX_TEXTURE_BYTES,
            )
            .map_err(|_| EzGfxResult::NativeFailure)?,
            ready_tx,
            ready_rx,
            #[cfg(test)]
            decode_gate: None,
        })
    }

    #[cfg(test)]
    fn worker_count(&self) -> usize {
        self.pool.thread_count()
    }
}

impl Drop for AsyncTextureState {
    fn drop(&mut self) {
        self.pool.shutdown();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameNativeResource {
    Buffer(PackedHandle),
    Texture(TextureHandle),
    Surface(SurfaceHandle),
    Depth,
    Index,
    RenderTarget(RenderTargetHandle),
}
struct ContextState {
    identity: ContextIdentity,
    options: ContextOptions,
    native: NativeContext,
    surfaces: HashMap<SurfaceHandle, SurfaceRecord>,
    allocations: HashMap<PackedHandle, (u64, NativeAllocation)>,
    allocation_ready: HashMap<PackedHandle, CompletionToken>,
    shaders: HashMap<ShaderHandle, ShaderRecord>,
    indirects: HashMap<IndirectBufferHandle, IndexedIndirectBuffer>,
    textures: HashMap<TextureHandle, (TextureId, NativeTexture, u32, u32, u32)>,
    render_targets: HashMap<RenderTargetHandle, render_target::RenderTargetRecord>,
    texture_formats: HashMap<TextureHandle, TextureFormat>,
    texture_published_mips: HashMap<TextureHandle, u32>,
    texture_residency_targets: HashMap<TextureHandle, u32>,
    texture_last_transfer: HashMap<TextureHandle, CompletionToken>,
    retired_textures: Vec<RetiredTexture>,
    pipelines: HashMap<PipelineKey, NativePipeline>,
    graphics_format: Option<u32>,
    texture_registry: TextureRegistry,
    texture_ready: HashMap<TextureHandle, CompletionToken>,
    pending_textures: HashMap<TextureHandle, PendingTexture>,
    texture_handoffs: HashMap<TextureHandle, Instant>,
    texture_telemetry: Arc<TextureUploadTelemetry>,
    async_textures: AsyncTextureState,
    texture_failures: HashMap<TextureHandle, EzGfxResult>,
    geometry: GeometryManager,
    vertex_heaps: HashMap<String, GeometryAllocation>,
    index_heap: Option<GeometryAllocation>,
    staging: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    frame: FrameRecorder,
    frame_resources: HashMap<PackedHandle, ResourceId>,
    frame_native_resources: HashMap<ResourceId, FrameNativeResource>,
    frame_index: Option<ResourceId>,
    frame_surface: Option<ResourceId>,
    frame_depth: Option<ResourceId>,
    frame_has_graphics: bool,
    frame_presented: bool,
    active_surface: Option<SurfaceHandle>,
    frame_render_target: Option<RenderTargetHandle>,
    last_readback: Vec<u8>,
    observability: Observability,
}

type ContextHandleArena = GenerationalArena<()>;
static CONTEXT_HANDLES: LazyLock<Mutex<ContextHandleArena>> = LazyLock::new(|| {
    // Context generations and slots must retire before exceeding their packed 20-bit fields.
    Mutex::new(
        GenerationalArena::with_limits(
            PackedHandle::MAX_CONTEXT_SLOT,
            PackedHandle::MAX_CONTEXT_GENERATION,
        )
        .expect("packed context limits have a nonzero generation"),
    )
});
struct ThreadContexts {
    states: HashMap<LocalHandle, ContextState>,
}

impl ThreadContexts {
    fn cleanup_for_thread_exit(&mut self) -> EzGfxResult {
        // Handles invalidate synchronously before platform-specific abandonment handling.
        let result = match CONTEXT_HANDLES.lock() {
            Ok(mut handles) => {
                let mut result = EzGfxResult::Ok;
                for local in self.states.keys() {
                    if handles.remove(*local).is_err() {
                        result = EzGfxResult::NativeFailure;
                    }
                }
                result
            }
            Err(_) => EzGfxResult::NativeFailure,
        };

        #[cfg(windows)]
        {
            for (_, state) in self.states.drain() {
                // Windows TLS destructors run under loader lock; native cleanup or joining can deadlock.
                // Loader-lock callers must use this abandon path only, never destroy/wait_idle.
                std::mem::forget(state);
            }
            result
        }

        #[cfg(not(windows))]
        {
            let mut result = result;
            for (_, state) in self.states.drain() {
                let cleanup = context::cleanup_context_state(state, None);
                if result == EzGfxResult::Ok {
                    result = cleanup;
                }
            }
            result
        }
    }
}

impl Drop for ThreadContexts {
    fn drop(&mut self) {
        let _ = self.cleanup_for_thread_exit();
    }
}

thread_local! {
    static CONTEXTS: RefCell<ThreadContexts> = RefCell::new(ThreadContexts {
        states: HashMap::new(),
    });
}

mod buffers;
mod context;
mod frame;
mod native;
#[cfg(windows)]
use native::dx12_bindings;
#[cfg(target_vendor = "apple")]
use native::metal_bindings;
use native::{
    allocate_native, completed_texture_transfer_native, completed_transfer_native, copy_native,
    destroy_native_texture, free_native_allocation, map_allocation, map_frame, map_geometry,
    map_hal, map_lifecycle, map_native_loss, map_texture, native_layouts,
    native_texture_compression, pipeline_layout_key, poll_native_frame_completion, result_status,
    retire_native_allocation, vulkan_bindings, wait_native_idle, write_native,
};
mod render_target;
mod shader;
mod texture;

pub use buffers::*;
pub use context::*;
pub use frame::*;
pub use render_target::*;
pub use shader::*;
pub use texture::*;

fn with_surface_mut<T>(
    context: ContextHandle,
    surface: SurfaceHandle,
    operation: impl FnOnce(&mut SurfaceRecord) -> Result<T, EzGfxResult>,
) -> Result<T, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = surface.packed();
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
    context: ContextHandle,
    operation: impl FnOnce(&mut ContextState) -> Result<T, EzGfxResult>,
) -> Result<T, EzGfxResult> {
    let (local, _) = context_local(context)?;
    CONTEXTS.with(|contexts| {
        let mut contexts = contexts
            .try_borrow_mut()
            .map_err(|_| EzGfxResult::NativeFailure)?;
        let context = contexts
            .states
            .get_mut(&local)
            .ok_or(EzGfxResult::InvalidContext)?;
        let result = operation(context);
        if matches!(&result, Err(EzGfxResult::DeviceLost)) {
            // Terminal-loss sweep: the first DeviceLost synchronously cancels queued
            // decodes so later polls return DeviceLost fast instead of NotReady.
            texture::note_device_lost(context);
        }
        result
    })
}
fn context_local(handle: ContextHandle) -> Result<(LocalHandle, PackedHandle), EzGfxResult> {
    let packed = handle.packed();

    match packed.parts().map_err(|_| EzGfxResult::InvalidContext)? {
        HandleParts::Context(local) => Ok((local, packed)),
        HandleParts::Child { .. } => Err(EzGfxResult::InvalidContext),
    }
}

#[cfg(test)]
mod tests;
