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
        ContextHandle, GenerationalArena, HandleParts, IndexAllocationHandle, IndirectBufferHandle,
        LocalHandle, PackedHandle, RenderTargetHandle, ShaderHandle, StructuredBufferHandle,
        SurfaceHandle, TextureHandle, VertexAllocationHandle, VertexHeapHandle,
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
    upload::{UploadEvent, UploadEventQueue, UploadResource, UploadStatus},
};

use crate::{Error, Result};

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
    heap_id: Option<u32>,
}

struct RetiredGeometry {
    allocation: NativeAllocation,
    transfer: CompletionToken,
    graphics: Option<CompletionToken>,
}

enum RetiredRangeKind {
    Vertex,
    Index,
}

enum RetiredRangeGraphics {
    Prior(Option<CompletionToken>),
    Recording(u64),
}

struct RetiredGeometryRange {
    handle: PackedHandle,
    kind: RetiredRangeKind,
    transfer: CompletionToken,
    graphics: RetiredRangeGraphics,
}

struct RetiredVertexHeap {
    name: String,
    allocation: NativeAllocation,
}

struct PendingTexture {
    id: TextureId,
    cancelled: Arc<AtomicBool>,
    config: TextureConfig,
    admitted_at: Instant,
}

struct DecodedTextureJob {
    handle: TextureHandle,
    decoded: std::result::Result<DecodedTexture, TextureError>,
}

struct AsyncTextureState {
    pool: ez_gfx_assets::CpuPool,
    ready_tx: crossbeam_channel::Sender<DecodedTextureJob>,
    ready_rx: crossbeam_channel::Receiver<DecodedTextureJob>,
    #[cfg(test)]
    decode_gate: Option<Arc<std::sync::Barrier>>,
}

impl AsyncTextureState {
    fn new_with_workers(workers: u32) -> Result<Self> {
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
            let threads = usize::try_from(workers).map_err(|_| Error::InvalidArgument)?;
            if threads > ez_gfx_assets::MAX_CPU_POOL_THREADS {
                return Err(Error::InvalidArgument);
            }
            threads
        };
        let (ready_tx, ready_rx) = crossbeam_channel::unbounded();
        Ok(Self {
            pool: ez_gfx_assets::CpuPool::new(threads).map_err(|_| Error::NativeFailure)?,
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
    VertexHeap(u32),
    RenderTarget(RenderTargetHandle),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransientUse {
    Available,
    Interned(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TransientBuffer {
    element_size: u32,
    element_count: u32,
    byte_capacity: u64,
    usage: TransientUse,
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
    transient_buffers: HashMap<PackedHandle, TransientBuffer>,
    structured_pool: HashMap<u32, ez_gfx_hal::ReusableStagingPool<NativeAllocation>>,
    indirect_pool: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
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
    texture_failures: HashMap<TextureHandle, Error>,
    geometry: GeometryManager,
    vertex_heaps: HashMap<String, GeometryAllocation>,
    vertex_heap_handles: HashMap<VertexHeapHandle, String>,
    index_heap: Option<GeometryAllocation>,
    retired_geometry: Vec<RetiredGeometry>,
    retired_geometry_ranges: Vec<RetiredGeometryRange>,
    retired_vertex_heaps: Vec<RetiredVertexHeap>,
    next_vertex_heap_id: u32,
    geometry_uploads: HashMap<PackedHandle, CompletionToken>,
    geometry_last_transfer: HashMap<PackedHandle, CompletionToken>,
    upload_events: UploadEventQueue,
    staging: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    frame: FrameRecorder,
    frame_resources: HashMap<PackedHandle, ResourceId>,
    frame_vertex_heaps: HashMap<u32, ResourceId>,
    frame_serial: u64,
    frame_native_resources: HashMap<ResourceId, FrameNativeResource>,
    frame_index: Option<ResourceId>,
    frame_surface: Option<ResourceId>,
    frame_depth: Option<ResourceId>,
    frame_has_graphics: bool,
    frame_presented: bool,
    active_surface: Option<SurfaceHandle>,
    frame_render_target: Option<RenderTargetHandle>,
    last_readbacks: Vec<Vec<u8>>,
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
    fn cleanup_for_thread_exit(&mut self) -> Result<()> {
        // Handles invalidate synchronously before platform-specific abandonment handling.
        let result = match CONTEXT_HANDLES.lock() {
            Ok(mut handles) => {
                let mut result = Ok(());
                for local in self.states.keys() {
                    if handles.remove(*local).is_err() {
                        result = Err(Error::NativeFailure);
                    }
                }
                result
            }
            Err(_) => Err(Error::NativeFailure),
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
                if result.is_ok() {
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
mod geometry;
mod native;
#[cfg(windows)]
use native::dx12_bindings;
#[cfg(target_vendor = "apple")]
use native::metal_bindings;
use native::{
    allocate_native, completed_native_frame_value, completed_texture_transfer_native,
    completed_transfer_native, copy_native, destroy_native_texture, free_native_allocation,
    last_native_frame_completion, map_allocation, map_frame, map_geometry, map_hal, map_lifecycle,
    map_native_loss, map_texture, native_layouts, native_texture_compression, pipeline_layout_key,
    poll_native_frame_completion, result_status, retire_native_allocation, vulkan_bindings,
    wait_native_idle, write_native,
};
mod render_target;
mod shader;
mod texture;

pub use buffers::*;
pub use context::*;
pub use frame::*;
pub use geometry::*;
pub use render_target::*;
pub use shader::*;
pub use texture::*;

fn with_surface_mut<T>(
    context: ContextHandle,
    surface: SurfaceHandle,
    operation: impl FnOnce(&mut SurfaceRecord) -> Result<T>,
) -> Result<T> {
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
                .ok_or(Error::InvalidContext)?,
        )
    })
}

pub(crate) fn abandon_context(context: ContextHandle) {
    let Ok((local, _)) = context_local(context) else {
        return;
    };
    if let Ok(mut handles) = CONTEXT_HANDLES.lock() {
        let _ = handles.remove(local);
    }
    let _ = CONTEXTS.try_with(|contexts| {
        let Ok(mut contexts) = contexts.try_borrow_mut() else {
            return;
        };
        if let Some(state) = contexts.states.remove(&local) {
            // Drop is nonblocking and may run during Windows loader/TLS teardown.
            // Explicit `Context::close` is the only native destruction path.
            std::mem::forget(state);
        }
    });
}

fn with_context_mut<T>(
    context: ContextHandle,
    operation: impl FnOnce(&mut ContextState) -> Result<T>,
) -> Result<T> {
    let (local, _) = context_local(context)?;
    CONTEXTS
        .try_with(|contexts| {
            let mut contexts = contexts
                .try_borrow_mut()
                .map_err(|_| Error::NativeFailure)?;
            let context = contexts
                .states
                .get_mut(&local)
                .ok_or(Error::InvalidContext)?;
            let result = operation(context);
            if matches!(&result, Err(Error::DeviceLost)) {
                texture::note_device_lost(context);
            }
            result
        })
        .map_err(|_| Error::InvalidContext)?
}
fn context_local(handle: ContextHandle) -> Result<(LocalHandle, PackedHandle)> {
    let packed = handle.packed();

    match packed.parts().map_err(|_| Error::InvalidContext)? {
        HandleParts::Context(local) => Ok((local, packed)),
        HandleParts::Child { .. } => Err(Error::InvalidContext),
    }
}

#[cfg(test)]
mod tests;
