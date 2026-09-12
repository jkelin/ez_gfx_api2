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
    capability::{AdapterInfo, PresentationMode, PresentationModes, ShaderCapabilities},
    handle::{
        BufferHandle, ContextHandle, CounterBufferHandle, GenerationalArena, HandleParts,
        IndexAllocationHandle, LocalHandle, PackedHandle, RenderTargetHandle, ShaderHandle,
        SurfaceHandle, TextureHandle, VertexAllocationHandle, VertexHeapHandle,
    },
};
use ez_gfx_geometry_manager::{GeometryError, GeometryManager};
use ez_gfx_hal::{
    AllocationRequest, BufferRange, BufferTransfer, COUNTER_BUFFER_ELEMENT_OFFSET, CompletionToken,
    DEFAULT_STAGING_POLICY, DynamicPipelineState, ExecutionAction, ExecutionBarrier, ExecutionPass,
    FrameExecutionBackend, FrameExecutionPlan, HalError, MemoryAllocator, MemoryClass, QueueKind,
    ResourceAccess, ResourceState, SURFACE_DEFAULT_CLEAR, ShaderStage, TextureFormat,
    TextureRegion, staging_bucket_size,
};
use ez_gfx_runtime::{
    AdapterCatalog, AdapterReport, AdapterSelection, ContextIdentity, ContextOptions,
    HeadlessSurfaceOptions, LifecycleError, ResourceKind, RuntimeError, SurfaceState,
    admission_report,
    frame::{ExecutableNode, FrameRecorder},
    graph::{
        Access, ImageRange, LoadOp, NodeDesc, PassInfo, ResourceDesc, ResourceId, ResourceLifetime,
        StoreOp,
    },
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer},
    observability::{DiagnosticLevel, Observability, RuntimePhase, RuntimeRecord, RuntimeStatus},
    target::Format,
    upload::{UploadEvent, UploadEventQueue, UploadResource, UploadStatus},
};
use ez_gfx_texture_manager::pipeline::{
    PendingUpload, QueuedDecode, SubmittedInfo, TexturePipeline, UploadFailure,
};
use ez_gfx_texture_manager::texture::{
    MAX_TEXTURE_BYTES, TextureDecoder, TextureUploadTelemetrySnapshot,
};
use ez_gfx_texture_manager::{
    DECODE_RESERVATION_BYTES, DecodeDriver, DecodeDriverError, MipTransferValues,
    ReclaimableStaging, SharedTransferPool, TextureBackendContext, TextureBackendTexture,
    TextureId, TextureRegistry, WORKING_SET_BUDGET_BYTES,
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
    Mesh {
        backend: Backend,
        task_shader: Option<ShaderHandle>,
        task_digest: Option<[u8; 32]>,
        task_entry: String,
        has_task_entry: bool,
        mesh_shader: ShaderHandle,
        mesh_digest: [u8; 32],
        mesh_entry: String,
        fragment_shader: ShaderHandle,
        fragment_digest: [u8; 32],
        fragment_entry: String,
        stage_layouts: [Option<ez_gfx_runtime::binding::StageLayoutIdentity>; 3],
        state: ez_gfx_hal::MeshPipelineState,
        depth_required: bool,
        color_format: u32,
        depth_format: u32,
        sample_count: u8,
    },
}

#[derive(Clone, Copy)]
struct MeshPipelineKeyDesc<'a> {
    backend: Backend,
    task_shader: Option<ShaderHandle>,
    task_digest: Option<[u8; 32]>,
    task_entry: Option<&'a str>,
    mesh_shader: ShaderHandle,
    mesh_digest: [u8; 32],
    mesh_entry: &'a str,
    fragment_shader: ShaderHandle,
    fragment_digest: [u8; 32],
    fragment_entry: &'a str,
    stage_layouts: &'a ez_gfx_hal::MeshStages<ez_gfx_runtime::binding::StageLayoutIdentity>,
    state: ez_gfx_hal::MeshPipelineState,
    depth_required: bool,
    color_format: u32,
    depth_format: u32,
    sample_count: u8,
}

impl PipelineKey {
    fn prepare_mesh_slot<'a>(
        slot: &'a mut Option<Self>,
        desc: MeshPipelineKeyDesc<'_>,
    ) -> &'a Self {
        if !matches!(slot, Some(Self::Mesh { .. })) {
            *slot = Some(Self::Mesh {
                backend: desc.backend,
                task_shader: None,
                task_digest: None,
                task_entry: String::new(),
                has_task_entry: false,
                mesh_shader: desc.mesh_shader,
                mesh_digest: desc.mesh_digest,
                mesh_entry: String::new(),
                fragment_shader: desc.fragment_shader,
                fragment_digest: desc.fragment_digest,
                fragment_entry: String::new(),
                stage_layouts: [None; 3],
                state: desc.state,
                depth_required: desc.depth_required,
                color_format: desc.color_format,
                depth_format: desc.depth_format,
                sample_count: desc.sample_count,
            });
        }
        let Some(Self::Mesh {
            backend,
            task_shader,
            task_digest,
            task_entry,
            has_task_entry,
            mesh_shader,
            mesh_digest,
            mesh_entry,
            fragment_shader,
            fragment_digest,
            fragment_entry,
            stage_layouts,
            state,
            depth_required,
            color_format,
            depth_format,
            sample_count,
        }) = slot
        else {
            unreachable!("mesh slot initialized above")
        };

        *backend = desc.backend;
        *task_shader = desc.task_shader;
        *task_digest = desc.task_digest;
        *has_task_entry = desc.task_entry.is_some();
        // A mesh-only use clears the logical task identity but retains the
        // allocation for a later task-and-mesh use of this bounded scratch slot.
        task_entry.clear();
        if let Some(entry) = desc.task_entry {
            task_entry.push_str(entry);
        }
        *mesh_shader = desc.mesh_shader;
        *mesh_digest = desc.mesh_digest;
        mesh_entry.clear();
        mesh_entry.push_str(desc.mesh_entry);
        *fragment_shader = desc.fragment_shader;
        *fragment_digest = desc.fragment_digest;
        fragment_entry.clear();
        fragment_entry.push_str(desc.fragment_entry);
        *stage_layouts = [
            desc.stage_layouts.task,
            Some(desc.stage_layouts.mesh),
            Some(desc.stage_layouts.fragment),
        ];
        *state = desc.state;
        *depth_required = desc.depth_required;
        *color_format = desc.color_format;
        *depth_format = desc.depth_format;
        *sample_count = desc.sample_count;
        slot.as_ref().expect("mesh slot initialized above")
    }

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
            Self::Mesh {
                task_shader,
                mesh_shader,
                fragment_shader,
                ..
            } => {
                task_shader.is_some_and(|candidate| candidate == shader)
                    || *mesh_shader == shader
                    || *fragment_shader == shader
            }
        }
    }

    const fn is_render(&self) -> bool {
        matches!(self, Self::Graphics { .. } | Self::Mesh { .. })
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
            Self::Mesh {
                task_entry,
                mesh_entry,
                fragment_entry,
                ..
            } => task_entry
                .capacity()
                .saturating_add(mesh_entry.capacity())
                .saturating_add(fragment_entry.capacity()),
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
    binding: u32,
    native: NativeTexture,
    completion: CompletionToken,
}

struct RetiredTextureBinding {
    id: TextureId,
}

struct SurfaceRecord {
    native: NativeSurface,
    state: SurfaceState,
    initialized: bool,
    presentation_mode: PresentationMode,
    presentation_modes: Option<PresentationModes>,
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
    #[cfg(not(target_vendor = "apple"))]
    DrainedFailure(Error),
    Undrained,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SurfaceInsertTestFailure {
    IdentityInsertion,
    InvalidPackedHandle,
}

#[cfg(all(test, windows))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RawNativeFrameTestProbe {
    enabled: bool,
    submits: usize,
    mesh_executions: usize,
    graphics_executions: usize,
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
    textures: HashMap<TextureHandle, NativeTexture>,
    transient_buffers: HashMap<PackedHandle, TransientBuffer>,
    buffer_pool: HashMap<u32, ez_gfx_hal::ReusableStagingPool<NativeAllocation>>,
    counter_pool: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    /// Retained command/payload serialization buffer; cleared per counter write
    /// so steady-state uploads reuse capacity instead of allocating per frame.
    counter_scratch: Vec<u8>,
    render_targets: HashMap<RenderTargetHandle, render_target::RenderTargetRecord>,
    retired_textures: Vec<RetiredTexture>,
    pipelines: HashMap<PipelineKey, NativePipeline>,
    retired_texture_bindings: Vec<RetiredTextureBinding>,
    graphics_format: Option<u32>,
    /// Backend-neutral upload pipeline: pending decodes, FIFO order, decoded
    /// payloads, submitted residency, retained fine mips, publication
    /// progress, transfer ledgers, and telemetry. Native textures stay in
    /// [`ContextState::textures`]; transitions run through manager-owned
    texture_pipeline: TexturePipeline,
    texture_registry: TextureRegistry,
    /// Explicit ownership and publication state for the shared opaque-magenta texture.
    texture_fallback: TextureFallback,
    /// One byte budget shared by texture decode reservations and geometry and
    /// buffer uploads, so a single hardware transfer queue sees one admission
    /// domain. Required reservations retire at prefix completion; fine,
    /// geometry, and buffer bytes retire at their transfer tokens.
    transfer_pool: SharedTransferPool,
    decode_textures: DecodeDriver,
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
    /// Reusable texture-handle snapshot used while recording sampled heap hazards.
    frame_texture_handles: Vec<TextureHandle>,
    #[cfg(target_vendor = "apple")]
    /// Reusable Metal texture-heap metadata aligned with frame nodes.
    frame_texture_heaps: Vec<Option<ez_gfx_hal::ShaderTextureHeapLayout>>,
    #[cfg(target_vendor = "apple")]
    /// Reusable Metal workgroup metadata aligned with frame nodes.
    frame_workgroup_sizes: Vec<Option<MetalWorkgroupSizes>>,
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
    #[cfg(test)]
    texture_fallback_alias_test_failure_after: Option<usize>,
    #[cfg(test)]
    shader_capabilities_override: Option<ez_gfx_core::capability::ShaderCapabilities>,
    #[cfg(test)]
    native_shader_allocation_attempts: usize,
    #[cfg(all(test, not(target_vendor = "apple")))]
    shader_destroy_requests: usize,
    #[cfg(test)]
    native_shader_destroys: usize,
    #[cfg(all(test, windows))]
    raw_native_frame_test_probe: RawNativeFrameTestProbe,
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

#[cfg(all(test, not(target_vendor = "apple")))]
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
    FrameBindingSource, FrameBufferBindingRecord, allocate_native, completed_native_frame_value,
    completed_texture_transfer_native, completed_transfer_native, copy_native,
    destroy_native_texture, free_native_allocation, largest_native_texture_staging,
    last_native_frame_completion, map_allocation, map_frame, map_geometry, map_hal, map_lifecycle,
    map_native_loss, map_schedule, map_texture, native_device_initialized, native_layouts,
    native_mesh_dispatch_limits, native_texture_compression, pipeline_layout_key,
    poll_native_frame_completion, pop_largest_native_texture_staging, prepare_frame_binding_scratch,
    published_texture_handles_into, result_status, retained_native_texture_staging,
    retire_native_allocation, wait_native_idle, write_native,
};
mod render_target;
mod shader;
mod surface;
use surface::destroy_native_surface;
mod texture;
mod texture_fallback;
use texture_fallback::{TextureFallback, initialize_texture_fallback, publish_reserved_fallback};
mod texture_manager;
use texture_manager::pump_async_textures;

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
