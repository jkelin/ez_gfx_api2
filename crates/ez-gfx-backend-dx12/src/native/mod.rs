use crate::{BACKEND, TEXTURE_DESCRIPTOR_CAPACITY};
use arrayvec::ArrayVec;
use core::{ffi::c_void, ptr};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use ez_gfx_core::capability::{
    AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport, PresentationMode,
    PresentationModes, SemanticProfile, ShaderCapabilities,
};
use ez_gfx_hal::{
    AllocationError, AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BlendMode,
    BufferTransfer, CompletionToken, CullMode, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DynamicPipelineState, ExecutionBarrier, ExecutionPass, FrontFace, HalError, ImageMip,
    MemoryAllocator, MemoryClass, MeshDispatchLimits, MeshPipelineState, PrimitiveTopology,
    QueueKind, ResourceAccess, SamplerAddressMode, SamplerFilter, ShaderBufferLayout,
    TextureFormat, TextureRegion, TextureSamplerDesc, TransferWorker, validate_mesh_dispatch,
    validate_texture_mips, validate_texture_region,
};
use gpu_allocator::{
    AllocationSizes, MemoryLocation,
    d3d12::{
        Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc, ID3D12DeviceVersion,
    },
};
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D_SHADER_MODEL_6_5, D3D12_BLEND_DESC,
    D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_BLEND_SRC_ALPHA,
    D3D12_BLEND_ZERO, D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0,
    D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COMMAND_SIGNATURE_DESC, D3D12_COMPARISON_FUNC_ALWAYS,
    D3D12_COMPARISON_FUNC_LESS, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_CULL_MODE_BACK, D3D12_CULL_MODE_FRONT,
    D3D12_CULL_MODE_NONE, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DEPTH_STENCIL_DESC,
    D3D12_DEPTH_STENCIL_VALUE, D3D12_DEPTH_STENCILOP_DESC, D3D12_DEPTH_WRITE_MASK_ALL,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_RANGE,
    D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND, D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
    D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_FEATURE_D3D12_OPTIONS7,
    D3D12_FEATURE_DATA_D3D12_OPTIONS7, D3D12_FEATURE_DATA_SHADER_MODEL, D3D12_FEATURE_SHADER_MODEL,
    D3D12_FILL_MODE_SOLID, D3D12_FILTER_ANISOTROPIC, D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
    D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_FILTER_MIN_MAG_MIP_POINT,
    D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT, D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    D3D12_INDEX_BUFFER_VIEW, D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_DESC_0,
    D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, D3D12_LOGIC_OP_NOOP, D3D12_MESH_SHADER_TIER,
    D3D12_MESH_SHADER_TIER_1, D3D12_MESH_SHADER_TIER_NOT_SUPPORTED,
    D3D12_PIPELINE_STATE_STREAM_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE, D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_READ,
    D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_INDEX_BUFFER,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER,
    D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
    D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_RT_FORMAT_ARRAY, D3D12_SAMPLER_DESC, D3D12_SHADER_BYTECODE,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
    D3D12_SHADER_VISIBILITY_ALL, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_STENCIL_OP_KEEP,
    D3D12_TEX2D_SRV, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_VIEWPORT, D3D12SerializeRootSignature,
    ID3D12CommandSignature, ID3D12DescriptorHeap, ID3D12PipelineState, ID3D12RootSignature,
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, E_INVALIDARG, HANDLE, HWND, RECT, WAIT_FAILED, WAIT_OBJECT_0},
        Graphics::{
            Direct3D::{D3D_FEATURE_LEVEL_12_1, ID3DBlob},
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_DIRECT,
                D3D12_COMMAND_QUEUE_DESC, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS,
                D3D12_FENCE_FLAG_NONE, D3D12_RANGE, D3D12_RENDER_TARGET_VIEW_DESC,
                D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RTV_DIMENSION_TEXTURE2D,
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12CreateDevice, ID3D12CommandAllocator,
                ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Device2, ID3D12Fence,
                ID3D12GraphicsCommandList, ID3D12GraphicsCommandList6, ID3D12Resource,
            },
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT, DXGI_FORMAT_D32_FLOAT,
                    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                    DXGI_FORMAT_R32_UINT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
                },
                CreateDXGIFactory1, DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_ERROR_NOT_FOUND,
                DXGI_ERROR_UNSUPPORTED, DXGI_FEATURE_PRESENT_ALLOW_TEARING, DXGI_PRESENT,
                DXGI_PRESENT_ALLOW_TEARING, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
                DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
                DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIAdapter4,
                IDXGIFactory4, IDXGIFactory5, IDXGISwapChain4,
            },
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
        UI::WindowsAndMessaging::GetClientRect,
    },
    core::Interface,
};

/// Returns the DXGI presentation parameters for one supported normalized mode.
const fn presentation_parameters(mode: PresentationMode) -> Option<(u32, DXGI_PRESENT)> {
    match mode {
        PresentationMode::Fifo => Some((1, DXGI_PRESENT(0))),
        PresentationMode::Paced => Some((0, DXGI_PRESENT(0))),
        PresentationMode::Immediate => Some((0, DXGI_PRESENT_ALLOW_TEARING)),
        PresentationMode::Mailbox | PresentationMode::Relaxed => None,
    }
}

const fn dx_presentation_modes(allow_tearing: bool) -> PresentationModes {
    let modes = PresentationModes::FIFO.union(PresentationModes::PACED);
    if allow_tearing {
        modes.union(PresentationModes::IMMEDIATE)
    } else {
        modes
    }
}

/// D3D12 buffer resource and its allocator ownership record.
pub struct NativeAllocation {
    resource: ID3D12Resource,
    allocation: Allocation,
    mapped_address: usize,
}

/// Root buffer binding resolved for one dispatch or draw.
pub struct NativeBufferBinding<'a> {
    /// Allocation supplying the root descriptor.
    pub allocation: &'a NativeAllocation,
    /// Byte offset added to the resource GPU address.
    pub offset: u64,
    /// Whether the root descriptor permits unordered writes.
    pub writable: bool,
}
/// Synchronous provider for resolved root-buffer bindings.
pub trait NativeBufferBindingSource {
    /// Number of bindings supplied to the pipeline.
    fn len(&self) -> usize;

    /// Returns whether the source has no bindings.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Visits each binding; borrowed allocation views cannot escape the call.
    ///
    /// # Errors
    ///
    /// Returns an error when a binding cannot resolve its native allocation or the visitor fails.
    #[expect(
        clippy::type_complexity,
        reason = "the object-safe callback keeps borrowed native allocation views scoped to each visit"
    )]
    fn visit(
        &self,
        visitor: &mut dyn FnMut(usize, &NativeBufferBinding<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError>;
}

impl NativeBufferBindingSource for [NativeBufferBinding<'_>] {
    fn len(&self) -> usize {
        <[NativeBufferBinding<'_>]>::len(self)
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(usize, &NativeBufferBinding<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError> {
        for (index, binding) in self.iter().enumerate() {
            visitor(index, binding)?;
        }
        Ok(())
    }
}

impl NativeBufferBindingSource for &[NativeBufferBinding<'_>] {
    fn len(&self) -> usize {
        <[NativeBufferBinding<'_>]>::len(self)
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(usize, &NativeBufferBinding<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError> {
        <[NativeBufferBinding<'_>] as NativeBufferBindingSource>::visit(self, visitor)
    }
}
impl<const N: usize> NativeBufferBindingSource for [NativeBufferBinding<'_>; N] {
    fn len(&self) -> usize {
        N
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(usize, &NativeBufferBinding<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError> {
        self.as_slice().visit(visitor)
    }
}

/// Fully resolved indexed draw consumed by command-list recording.
pub struct NativeDrawIndexed<'a> {
    /// Render width in pixels.
    pub width: u32,
    /// Render height in pixels.
    pub height: u32,
    /// Graphics pipeline used by the draw.
    pub pipeline: &'a NativePipeline,
    /// Buffer containing 32-bit indices.
    pub index_buffer: &'a NativeAllocation,
    /// Logical byte length of the index resource.
    pub index_size: u64,
    /// Buffer containing indexed indirect commands.
    pub indirect_buffer: &'a NativeAllocation,
    /// Logical byte length of the indirect resource.
    pub indirect_size: u64,
    /// Number of indirect commands to execute.
    pub draw_count: u32,
    /// Reflected root buffer bindings.
    pub bindings: &'a dyn NativeBufferBindingSource,
}

/// Fully resolved compute dispatch consumed by command-list recording.
pub struct NativeComputeDispatch<'a> {
    /// Compute pipeline used by the dispatch.
    pub pipeline: &'a NativePipeline,
    /// Workgroup count for each dimension.
    pub groups: [u32; 3],
    /// Reflected root buffer bindings.
    pub bindings: &'a dyn NativeBufferBindingSource,
}

/// Immutable inputs used to create a mesh pipeline.
///
/// Backend-local shape of the shared mesh seam: exact task/mesh/fragment DXIL
/// products, one entry per stage, and merged stage layouts. Rasterization comes
/// from the shared [`MeshPipelineState`], which carries no topology because mesh
/// pipelines hold no input-assembler state. Workgroup sizes come from stage
/// reflection and are checked before any D3D12 state creation.
pub struct NativeMeshPipelineDesc<'a> {
    /// Optional task stage as owning shader plus product index.
    pub task: Option<(&'a NativeShader, usize)>,
    /// Required mesh stage as owning shader plus product index.
    pub mesh: (&'a NativeShader, usize),
    /// Required fragment stage as owning shader plus product index.
    pub fragment: (&'a NativeShader, usize),
    /// Rasterization and blend state fixed by the pipeline.
    pub state: MeshPipelineState,
    /// Offscreen color format, or the default surface format.
    pub color_format: Option<ez_gfx_runtime::target::Format>,
    /// Reflected root buffer layout merged across the selected stages.
    pub layouts: &'a [ShaderBufferLayout],
    /// Whether the pipeline requires a depth attachment.
    pub depth_required: bool,
    /// Optional task workgroup size from reflection.
    pub task_workgroup_size: Option<[u32; 3]>,
    /// Mesh workgroup size from reflection.
    pub mesh_workgroup_size: [u32; 3],
}

/// Fully resolved mesh dispatch consumed by command-list recording.
pub struct NativeMeshDispatch<'a> {
    /// Mesh pipeline used by the dispatch.
    pub pipeline: &'a NativePipeline,
    /// Task workgroup count for each dispatch dimension.
    pub groups: [u32; 3],
    /// Whether the pipeline holds a task stage.
    pub has_task: bool,
    /// Mesh workgroup size from reflection, rechecked at record time.
    pub mesh_workgroup_size: [u32; 3],
    /// Optional task workgroup size from reflection; required with a task stage.
    pub task_workgroup_size: Option<[u32; 3]>,
    /// Reflected root buffer bindings.
    pub bindings: &'a dyn NativeBufferBindingSource,
}

/// Native limits used to validate a direct mesh dispatch.
///
/// The per-dimension grid ceiling is the documented D3D12 dispatch ceiling, and
/// the total grid ceiling is the mesh-shader specification's `DispatchMesh`
/// requirement that the workgroup-count product not exceed 2^22. The
/// specification caps amplification and mesh threadgroup sizes at 128 threads
/// each; D3D12 exposes no limit query, so the backend enforces the specified
/// constants through the shared value.
pub(super) fn retained_mesh_dispatch_limits() -> MeshDispatchLimits {
    MeshDispatchLimits {
        max_groups: [D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION; 3],
        max_total_groups: 1 << 22,
        max_mesh_threads: 128,
        max_task_threads: 128,
    }
}

/// Checks a mesh workgroup dispatch without allocation.
///
/// A task threadgroup size without a task stage, or a missing one with it, fails
/// as an inconsistent dispatch description before the shared grid validation,
/// which covers per-dimension, total, and threadgroup ceilings.
///
/// # Errors
///
/// Returns [`HalError::InvalidArgument`] for an inconsistent stage selection, a
/// zero dimension, an overflowing product, or an exceeded grid ceiling.
pub(super) fn check_mesh_dispatch(
    has_task: bool,
    groups: [u32; 3],
    mesh_threads: [u32; 3],
    task_threads: Option<[u32; 3]>,
) -> Result<(), HalError> {
    if task_threads.is_some() != has_task {
        return Err(HalError::InvalidArgument);
    }
    validate_mesh_dispatch(
        groups,
        mesh_threads,
        task_threads,
        retained_mesh_dispatch_limits(),
    )
    .map_err(|_| HalError::InvalidArgument)
}

/// D3D12 resource referenced by a compiled frame barrier.
pub enum NativeFrameResource<'a> {
    /// Buffer resource.
    Buffer(&'a NativeAllocation),
    /// Texture resource.
    Texture(&'a NativeTexture),
    /// Current swapchain back buffer.
    Surface,
    /// Current depth resource.
    Depth,
    /// Managed single-mip color render target.
    RenderTarget(&'a NativeTexture),
}

/// One resolved pass color attachment: its native resource plus the clear
/// value applied when the pass load op clears. Surfaces carry the legacy
/// default; render targets carry their stored declaration clear.
pub struct PassAttachment<'a> {
    /// Resolved native color resource.
    pub resource: NativeFrameResource<'a>,
    /// Clear color applied for a clearing load op.
    pub clear: [f32; 4],
}

/// Validated D3D12 action emitted by the frame-plan adapter.
pub enum NativeFrameAction<'a> {
    /// Wait for external transfer completion.
    Wait(CompletionToken),
    /// Transition or order access to a resource.
    Barrier {
        /// Backend-neutral barrier description.
        barrier: ExecutionBarrier,
        /// Resolved resource affected by the barrier.
        resource: NativeFrameResource<'a>,
    },
    /// Begin the declared render pass with resolved color attachments.
    BeginPass {
        /// Backend-neutral pass description.
        pass: &'a ExecutionPass,
        /// One attachment per pass color, in order.
        colors: ArrayVec<PassAttachment<'a>, 1>,
    },
    /// Encode a compute dispatch.
    Compute(NativeComputeDispatch<'a>),
    /// Encode indexed indirect graphics work.
    Graphics(NativeDrawIndexed<'a>),
    /// Encode a mesh workgroup dispatch inside the active render pass.
    Mesh(NativeMeshDispatch<'a>),
    /// Copy a texture into a readback allocation.
    TextureReadback {
        /// Texture to copy.
        texture: &'a NativeTexture,
        /// Readback width in pixels.
        width: u32,
        /// Readback height in pixels.
        height: u32,
    },
    /// End the active render pass.
    EndPass,
    /// Present the active swapchain buffer.
    Present,
}

/// Synchronous provider whose borrowed native actions live only for each visit.
pub trait NativeFrameActionSource {
    /// Number of stable action records.
    fn len(&self) -> usize;

    /// Visits one action by stable index.
    ///
    /// # Errors
    ///
    /// Returns an error when the index is invalid, an action cannot resolve its native owner, or
    /// the visitor rejects the action.
    fn with_action(
        &self,
        index: usize,
        visitor: &mut dyn FnMut(&NativeFrameAction<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError>;

    /// Visits all actions in order.
    ///
    /// # Errors
    ///
    /// Returns an error when an action cannot resolve its native owner or the visitor rejects it.
    #[expect(
        clippy::type_complexity,
        reason = "the object-safe callback keeps borrowed native action views scoped to each visit"
    )]
    fn visit(
        &self,
        visitor: &mut dyn FnMut(usize, &NativeFrameAction<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError> {
        for index in 0..self.len() {
            self.with_action(index, &mut |action| visitor(index, action))?;
        }
        Ok(())
    }

    /// Returns whether no action records exist.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl NativeFrameActionSource for [NativeFrameAction<'_>] {
    fn len(&self) -> usize {
        <[NativeFrameAction<'_>]>::len(self)
    }

    fn with_action(
        &self,
        index: usize,
        visitor: &mut dyn FnMut(&NativeFrameAction<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError> {
        visitor(self.get(index).ok_or(HalError::InvalidArgument)?)
    }
}

impl NativeFrameActionSource for &[NativeFrameAction<'_>] {
    fn len(&self) -> usize {
        <[NativeFrameAction<'_>]>::len(self)
    }

    fn with_action(
        &self,
        index: usize,
        visitor: &mut dyn FnMut(&NativeFrameAction<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError> {
        self.as_ref().with_action(index, visitor)
    }
}

impl<const N: usize> NativeFrameActionSource for [NativeFrameAction<'_>; N] {
    fn len(&self) -> usize {
        N
    }

    fn with_action(
        &self,
        index: usize,
        visitor: &mut dyn FnMut(&NativeFrameAction<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError> {
        self.as_slice().with_action(index, visitor)
    }
}

/// Validated DXIL products retained until pipeline creation.
pub struct NativeShader {
    products: Vec<Vec<u8>>,
}

impl NativeShader {
    /// Returns the DXIL products in artifact order.
    pub fn products(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.products.iter().map(Vec::as_slice)
    }
}

/// D3D12 pipeline, root signature, and indirect execution metadata.
pub struct NativePipeline {
    state: ID3D12PipelineState,
    root: ID3D12RootSignature,
    topology: Option<D3D_PRIMITIVE_TOPOLOGY>,
    signature: Option<ID3D12CommandSignature>,
    buffer_writable: Vec<bool>,
    /// Whether this state object is a mesh pipeline; mesh dispatches reject any
    /// other pipeline instead of misrecording through it.
    mesh: bool,
    /// Whether the pipeline was created with a task stage; mesh dispatches
    /// must match it instead of trusting the caller's stage flag.
    task_stage: bool,
}

/// Multisampled render storage owned by a managed render target.
///
/// The single-sample `NativeTexture` resource stays the resolve destination,
/// so barriers, readback, descriptors, and destruction keep working unchanged;
/// only the render pass binds this storage and resolves into it at end of pass.
pub struct MsaaStorage {
    resource: ID3D12Resource,
    allocation: Allocation,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    /// Render sample count; passes must request exactly this count.
    samples: u8,
}

/// Texture resource and allocator ownership record.
pub struct NativeTexture {
    resource: ID3D12Resource,
    allocation: Allocation,
    format: TextureFormat,
    /// Sampler installed with the first sampled-view publication; absent for render targets.
    sampler_desc: Option<TextureSamplerDesc>,
    width: u32,
    height: u32,
    mip_count: u32,
    resident_mips: u32,
    mip_completions: Vec<u64>,
    cancellation: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Slot in the shader-visible texture descriptor heap.
    pub binding: u32,
    /// Render-target view for managed color targets; `None` for uploads.
    /// The retaining heap keeps the CPU handle valid until destruction.
    pub rtv: Option<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE)>,
    /// Multisampled render storage plus its view and allocation; `None` for
    /// uploads and single-sample targets. Only render-target entry points
    /// touch this; the resource above stays the resolve destination.
    msaa: Option<MsaaStorage>,
}

impl NativeTexture {
    /// Returns the last transfer value that may reference this texture.
    pub fn last_transfer_value(&self) -> u64 {
        self.mip_completions.iter().copied().max().unwrap_or(0)
    }

    /// Latest copy values ordered from finest to coarsest mip; zero means never submitted.
    pub fn mip_transfer_values(&self) -> &[u64] {
        &self.mip_completions
    }
}

struct RetiredAllocation {
    allocation: NativeAllocation,
    completion: CompletionToken,
}

const FRAMES_IN_FLIGHT: usize = 3;

struct FrameSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_value: u64,
    garbage: Vec<NativeAllocation>,
    /// Retained indirect-copy entry shell; entries own GPU allocations while in
    /// flight, so only the emptied shell returns here, completion-gated by the
    /// slot fence wait in `prepare_frame`.
    indirect_scratch: Vec<frame::IndirectCopyEntry>,
}

enum DeferredResource {
    Allocation(NativeAllocation),
    Pipeline(NativePipeline),
    Texture(NativeTexture),
}

struct DeferredNativeResource {
    fence_value: u64,
    resource: DeferredResource,
}

struct SurfaceDepth {
    resource: ID3D12Resource,
    allocation: Allocation,
    heap: ID3D12DescriptorHeap,
}

/// Borrowed HWND, swapchain state, and optional captured frame.
pub struct NativeSurface {
    window: usize,
    swapchain: Option<IDXGISwapChain4>,
    buffers: Vec<ID3D12Resource>,
    allow_tearing: bool,
    rtv_heap: Option<ID3D12DescriptorHeap>,
    width: u32,
    height: u32,
    presented: Vec<u8>,
    depth: Option<SurfaceDepth>,
}

impl NativeSurface {
    /// Extracts a validated HWND from a matched Windows/Win32 raw handle pair.
    ///
    /// # Safety
    ///
    /// `display` and `window` must belong to the same live Windows host, and this function must run
    /// on the host's creator thread. The host must remain alive until the returned surface is
    /// destroyed by [`NativeContext::destroy_surface`] or native teardown is abandoned, after
    /// which it must remain alive for the process lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::InvalidArgument`] for every non-Windows or mismatched pair.
    pub unsafe fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<Self, HalError> {
        let (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) = (display, window)
        else {
            return Err(HalError::InvalidArgument);
        };
        Ok(Self {
            // Preserve the opaque HWND bit pattern even when its pointer-sized integer is negative.
            window: window.hwnd.get().cast_unsigned(),
            swapchain: None,
            allow_tearing: false,
            width: 0,
            height: 0,
            buffers: Vec::new(),
            rtv_heap: None,
            presented: Vec::new(),
            depth: None,
        })
    }

    /// Returns the normalized presentation modes available for this windowed flip-model surface.
    pub const fn presentation_modes(&self) -> PresentationModes {
        dx_presentation_modes(self.allow_tearing)
    }

    /// Returns the borrowed HWND value.
    pub const fn window(&self) -> usize {
        self.window
    }

    /// Reads the current drawable size from the borrowed HWND.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::NativeFailure`] when Win32 rejects the query.
    pub fn window_extent(&self) -> Result<Option<(u32, u32)>, HalError> {
        let mut rect = RECT::default();
        // SAFETY: `self.window` remains borrowed and valid for this surface's lifetime.
        unsafe { GetClientRect(HWND(self.window as *mut _), &raw mut rect) }
            .map_err(|_| HalError::NativeFailure)?;
        let width = rect.right.saturating_sub(rect.left);
        let height = rect.bottom.saturating_sub(rect.top);
        if width == 0 || height == 0 {
            return Ok(None);
        }
        Ok(Some((
            u32::try_from(width).map_err(|_| HalError::NativeFailure)?,
            u32::try_from(height).map_err(|_| HalError::NativeFailure)?,
        )))
    }

    /// Returns the most recently captured RGBA8 frame, or an empty slice before capture.
    pub fn presented_rgba8(&self) -> &[u8] {
        &self.presented
    }

    /// Reports retained swapchain images for on-demand memory telemetry.
    ///
    /// Buffers die with resize, so this observes only currently retained images.
    pub fn telemetry_images(&self) -> u32 {
        // Back-buffer counts are small; the fallback only guards the conversion.
        u32::try_from(self.buffers.len()).unwrap_or(u32::MAX)
    }

    /// Reports the last configured swapchain extent for memory telemetry.
    ///
    /// Zero means no swapchain was ever configured for this surface.
    pub const fn telemetry_extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Reports the swapchain format code for memory telemetry.
    ///
    /// Swapchains are created as `DXGI_FORMAT_R8G8B8A8_UNORM` (28); a missing
    /// swapchain still reports the configured format so format alignment never
    /// depends on presentation having occurred.
    pub const fn telemetry_format(&self) -> u32 {
        28
    }

    /// Reports whether depth storage is retained for memory telemetry.
    pub const fn telemetry_has_depth(&self) -> bool {
        self.depth.is_some()
    }
}

/// Admitted D3D12 device, queues, allocators, descriptors, and frame state.
pub struct NativeContext {
    /// Adapter selected during context creation.
    pub adapter: IDXGIAdapter4,
    /// Feature-level 12.1 device created from the selected adapter.
    pub device: ID3D12Device,
    /// Mesh-shader tier probed at admission; gates later `DispatchMesh` recording.
    mesh_shader_tier: D3D12_MESH_SHADER_TIER,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    transfer_fence: ID3D12Fence,
    texture_fence: ID3D12Fence,
    transfer_worker: Option<TransferWorker<transfer::Dx12TransferJob>>,
    texture_worker: Option<TransferWorker<transfer::Dx12TransferJob>>,
    fence_event: HANDLE,
    next_fence: u64,
    idle_drained: bool,
    #[cfg(test)]
    wait_idle_failure: Option<HalError>,
    next_transfer_fence: u64,
    next_texture_fence: u64,
    allocator: Option<Allocator>,
    retired: Vec<RetiredAllocation>,
    texture_staging: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    frame_slots: Vec<FrameSlot>,
    frame_cursor: usize,
    deferred: Vec<DeferredNativeResource>,
    adapter_info: AdapterInfo,
    descriptors: ID3D12DescriptorHeap,
    descriptor_stride: u32,
    samplers: ID3D12DescriptorHeap,
    sampler_stride: u32,
    /// Slots whose descriptors currently alias the shared fallback texture and sampler.
    /// Every sampled-heap writer must set `true` for aliases and `false` for real descriptors.
    texture_fallback_bindings: Vec<bool>,
}

// SAFETY: D3D12/DXGI interfaces are agile and the event handle is process-wide; higher layers
// serialize mutation and enforce frame-recording thread affinity.
unsafe impl Send for NativeContext {}

mod commands;
use commands::{
    bind_dx12_compute_buffers, bind_dx12_graphics_buffers, copy_texture_to_readback,
    create_frame_slots, dx12_resource_state, record_resource_barriers, transition_barrier,
    uav_barrier,
};
mod device;
mod frame;
mod memory;
use memory::{adapter_id, map_allocation_windows, map_allocator, map_allocator_hal, map_windows};
mod pipeline;
mod surface;
mod texture;
mod transfer;

#[cfg(test)]
mod texture_tests;

#[cfg(test)]
mod surface_tests {
    use std::num::NonZeroIsize;

    use raw_window_handle::{
        RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
        XlibDisplayHandle,
    };

    use super::*;

    #[test]
    fn surface_accepts_only_windows_win32_pairs() {
        let window = RawWindowHandle::Win32(Win32WindowHandle::new(NonZeroIsize::new(1).unwrap()));

        let surface =
            // SAFETY: the synthetic matched pair is inspected only; no native call dereferences it.
            unsafe {
                NativeSurface::new(
                    RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
                    window,
                )
            }
            .unwrap();
        assert_eq!(surface.window(), 1);

        let mismatch = RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0));
        assert_eq!(
            // SAFETY: mismatched handles are rejected before either synthetic value is used.
            unsafe { NativeSurface::new(mismatch, window) }.err(),
            Some(HalError::InvalidArgument)
        );
    }
}
