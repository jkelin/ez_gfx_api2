use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
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
use ez_gfx_backend_vulkan::{NativeContext as VulkanContext, NativeSurface as VulkanSurface};
use ez_gfx_core::{
    Backend,
    capability::{AdapterInfo, PresentationMode, PresentationModes},
    handle::{
        BufferHandle, ContextHandle, CounterBufferHandle, GenerationalArena, HandleParts,
        IndexAllocationHandle, LocalHandle, PackedHandle, RenderTargetHandle, ShaderHandle,
        SurfaceHandle, TextureHandle, VertexAllocationHandle, VertexHeapHandle,
    },
};
use ez_gfx_hal::{
    AllocationRequest, BufferRange, BufferTransfer, COUNTER_BUFFER_ELEMENT_OFFSET, CompletionToken,
    DEFAULT_STAGING_POLICY, DynamicPipelineState, ExecutionAction, ExecutionBarrier, ExecutionPass,
    FrameExecutionBackend, FrameExecutionPlan, HalError, ImageMip, MemoryAllocator, MemoryClass,
    QueueKind, ResourceAccess, ResourceState, SURFACE_DEFAULT_CLEAR, ShaderStage, TextureFormat,
    TextureRegion, staging_bucket_size,
};
use ez_gfx_runtime::{
    AdapterCatalog, AdapterReport, AdapterSelection, ContextIdentity, ContextOptions,
    HeadlessSurfaceOptions, LifecycleError, ResourceKind, RuntimeError, SurfaceState,
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

#[derive(Clone, Copy)]
pub(crate) struct SurfaceWindow {
    pub(crate) display: raw_window_handle::RawDisplayHandle,
    pub(crate) window: raw_window_handle::RawWindowHandle,
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
        entry: String,
        layouts: Vec<ez_gfx_hal::ShaderBufferLayout>,
    },
    Graphics {
        backend: Backend,
        vertex_shader: ShaderHandle,
        vertex_digest: [u8; 32],
        vertex_entry: String,
        fragment_shader: ShaderHandle,
        fragment_digest: [u8; 32],
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
    fn involves_shader(&self, shader: ShaderHandle) -> bool {
        match self {
            Self::Compute {
                shader: candidate, ..
            } => *candidate == shader,
            Self::Graphics {
                vertex_shader,
                fragment_shader,
                ..
            } => *vertex_shader == shader || *fragment_shader == shader,
        }
    }
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Compute { entry, layouts, .. } => entry.capacity().saturating_add(
                layouts
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ez_gfx_hal::ShaderBufferLayout>()),
            ),
            Self::Graphics {
                vertex_entry,
                fragment_entry,
                layouts,
                ..
            } => vertex_entry
                .capacity()
                .saturating_add(fragment_entry.capacity())
                .saturating_add(
                    layouts
                        .capacity()
                        .saturating_mul(core::mem::size_of::<ez_gfx_hal::ShaderBufferLayout>()),
                ),
        }
    }
}

const MAX_PIPELINE_CACHE_ENTRIES: usize = 1024;

struct ShaderRecord {
    native: NativeShader,
    digest: [u8; 32],
    product: usize,
    entry: String,
    stage: ez_gfx_artifact::Stage,
    runtime: ez_gfx_runtime::shader::RuntimeShader,
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
    initialized: bool,
    presentation_mode: PresentationMode,
    is_window: bool,
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
    // Admitted caller source bytes still awaiting decode; the owned copy lives on the
    // decode closure, so this count is the only retained size until transfer takes over.
    source_bytes: u64,
    admitted_at: Instant,
}

struct DecodedTextureJob {
    handle: TextureHandle,
    decoded: std::result::Result<DecodedTexture, TextureError>,
}

struct AsyncTextureState {
    /// Validated worker policy resolved at creation; the Rayon pool is built on
    /// first decode so context creation never spawns threads for textureless use.
    threads: usize,
    /// Decode pool built lazily by `decode_pool`; `None` until first decode.
    pool: Option<ez_gfx_assets::CpuPool>,
    ready_tx: crossbeam_channel::Sender<DecodedTextureJob>,
    ready_rx: crossbeam_channel::Receiver<DecodedTextureJob>,
    #[cfg(test)]
    decode_gate: Option<Arc<std::sync::Barrier>>,
}

impl AsyncTextureState {
    fn new_with_workers(workers: u32) -> Result<Self> {
        // Zero preserves the historical default topology; an explicit count is
        // honored verbatim so embedders can pin decode concurrency. Counts above
        // the pool admission cap fail here, before any thread exists, so both the
        // Rust option and the C descriptor fail fast with InvalidArgument instead
        // of grinding thread creation. Pool construction itself waits for first use.
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
            threads,
            pool: None,
            ready_tx,
            ready_rx,
            #[cfg(test)]
            decode_gate: None,
        })
    }

    /// Returns the decode pool, constructing it on the first decode.
    ///
    /// The count was validated at creation, so late construction only fails when
    /// the OS refuses threads; that surfaces as `NativeFailure` at admission,
    /// matching the previous eager-construction failure at the same boundary.
    fn decode_pool(&mut self) -> Result<&ez_gfx_assets::CpuPool> {
        if self.pool.is_none() {
            let pool =
                ez_gfx_assets::CpuPool::new(self.threads).map_err(|_| Error::NativeFailure)?;
            self.pool = Some(pool);
        }
        // Inserted above when absent, so `None` is unreachable without a borrow break.
        self.pool.as_ref().ok_or(Error::NativeFailure)
    }

    /// Builds the lazy decode pool before any admission that would need rollback.
    ///
    /// Callers must invoke this before registering registry, identity, or
    /// pending-texture state: late OS thread refusal then fails with nothing to
    /// unwind. The creator-thread model means no other thread can drop the pool
    /// between this call and submission.
    fn ensure_decode_pool(&mut self) -> Result<()> {
        self.decode_pool().map(|_| ())
    }
    /// Cancels queued and future CPU work when a pool was ever built.
    fn shutdown(&self) {
        // An unbuilt pool owns no threads or jobs, so skipping shutdown preserves
        // the lazy savings instead of constructing a pool only to cancel it.
        if let Some(pool) = self.pool.as_ref() {
            pool.shutdown();
        }
    }

    /// Configured worker count; reports policy before the first decode builds the pool.
    fn worker_count(&self) -> usize {
        self.pool
            .as_ref()
            .map_or(self.threads, ez_gfx_assets::CpuPool::thread_count)
    }
}

impl Drop for AsyncTextureState {
    fn drop(&mut self) {
        self.shutdown();
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CleanupTestOutcome {
    DrainedFailure(Error),
    Undrained,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SurfaceInsertTestFailure {
    IdentityInsertion,
    InvalidPackedHandle,
}

struct ContextState {
    identity: ContextIdentity,
    options: ContextOptions,
    native: NativeContext,
    surfaces: HashMap<SurfaceHandle, SurfaceRecord>,
    allocations: HashMap<PackedHandle, (u64, NativeAllocation)>,
    allocation_ready: HashMap<PackedHandle, CompletionToken>,
    shaders: HashMap<ShaderHandle, ShaderRecord>,
    frame_shaders: HashSet<ShaderHandle>,
    pending_shader_destroys: HashSet<ShaderHandle>,
    indirects: HashMap<CounterBufferHandle, IndexedIndirectBuffer>,
    textures: HashMap<TextureHandle, (TextureId, NativeTexture, u32, u32, u32)>,
    transient_buffers: HashMap<PackedHandle, TransientBuffer>,
    buffer_pool: HashMap<u32, ez_gfx_hal::ReusableStagingPool<NativeAllocation>>,
    counter_pool: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    /// Retained command/payload serialization buffer; cleared per counter write
    /// so steady-state uploads reuse capacity instead of allocating per frame.
    counter_scratch: Vec<u8>,
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
    // Decoded staging bytes per transfer-pending texture, kept in lockstep with
    // `texture_ready`: inserted at native submission, removed at publication, cancel,
    // loss, or teardown. Region updates overwrite with their latest transfer size.
    texture_transfer_bytes: HashMap<TextureHandle, u64>,
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
    /// Peak aggregate staging retention recorded at pool mutation boundaries.
    ///
    /// Telemetry reads this value without mutating state or reconstructing a
    /// peak from per-pool maxima that may not have occurred simultaneously.
    staging_high_water_bytes: u64,
    frame: FrameRecorder,
    /// Reusable graph-indexed pipeline lookup storage for native lowering.
    frame_pipeline_keys: Vec<Option<PipelineKey>>,
    /// Reusable mapping from emitted native actions to frame-plan records.
    frame_action_indices: Vec<usize>,
    /// Reusable owned binding coordinates for synchronous native lowering.
    frame_binding_scratch: Vec<native::FrameBufferBindingRecord>,
    /// Binding ranges aligned with frame payloads.
    frame_binding_ranges: Vec<core::ops::Range<usize>>,
    #[cfg(target_vendor = "apple")]
    /// Reusable Metal texture-heap metadata aligned with frame nodes.
    frame_texture_heaps: Vec<Option<ez_gfx_hal::ShaderTextureHeapLayout>>,
    #[cfg(target_vendor = "apple")]
    /// Reusable Metal workgroup metadata aligned with frame nodes.
    frame_workgroup_sizes: Vec<Option<[u32; 3]>>,
    frame_resources: HashMap<PackedHandle, ResourceId>,
    frame_vertex_heaps: HashMap<u32, ResourceId>,
    frame_serial: u64,
    frame_native_resources: HashMap<ResourceId, FrameNativeResource>,
    frame_index: Option<ResourceId>,
    frame_surface: Option<ResourceId>,
    frame_depth: Option<ResourceId>,
    frame_has_graphics: bool,
    frame_presented: bool,
    frame_capture_surface: Option<SurfaceHandle>,
    active_surface: Option<SurfaceHandle>,
    frame_render_target: Option<RenderTargetHandle>,
    last_readbacks: Vec<Vec<u8>>,
    observability: Observability,
    #[cfg(test)]
    cleanup_test_outcome: Option<CleanupTestOutcome>,
    #[cfg(test)]
    surface_insert_test_failure: Option<SurfaceInsertTestFailure>,
    #[cfg(test)]
    surface_rollback_test_abandoned: bool,
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
            let abandoned = !self.states.is_empty();
            for (_, state) in self.states.drain() {
                // Windows TLS destructors run under loader lock; native cleanup or joining can deadlock.
                // Loader-lock callers must abandon GPU owners rather than claim successful cleanup.
                std::mem::forget(state);
            }
            if abandoned {
                Err(Error::TeardownAbandoned)
            } else {
                result
            }
        }

        #[cfg(not(windows))]
        {
            let mut result = result;
            for (_, state) in self.states.drain() {
                let cleanup = context::cleanup_context_state(state, None);
                if cleanup == Err(Error::TeardownAbandoned) || result.is_ok() {
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

#[cfg(test)]
pub(crate) fn cleanup_context_for_thread_exit() -> Result<()> {
    CONTEXTS.with(|contexts| contexts.borrow_mut().cleanup_for_thread_exit())
}

mod buffers;
mod context;
mod diagnostics;
mod frame;
mod geometry;
mod native;
use native::{
    FrameBindingSource, FrameBufferBindingRecord, allocate_native,
    completed_native_frame_value, completed_texture_transfer_native, completed_transfer_native,
    copy_native, destroy_native_texture, free_native_allocation, last_native_frame_completion,
    map_allocation, map_frame, map_geometry, map_hal, map_lifecycle, map_native_loss, map_texture,
    native_device_initialized, native_layouts, native_texture_compression, pipeline_layout_key,
    poll_native_frame_completion, prepare_frame_binding_scratch, result_status,
    retire_native_allocation, wait_native_idle, write_native,
};
mod render_target;
mod shader;
mod surface;
use surface::destroy_native_surface;
mod texture;

pub use buffers::*;
pub use context::*;
pub use diagnostics::*;
pub use frame::*;
pub use geometry::*;
pub use render_target::*;
pub use shader::*;
pub use surface::*;
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
