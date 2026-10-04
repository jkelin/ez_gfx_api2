//! Backend-neutral allocation, binding, synchronization, and command contracts.
#![forbid(unsafe_code)]

use core::fmt;
use ez_gfx_core::capability::{AdapterInfo, MAX_BINDLESS_SAMPLED_TEXTURES};
mod telemetry;
pub use telemetry::{
    AllocationBlockPolicy, AllocationBlockPolicyError, AllocatorTelemetry, BackendMemoryTelemetry,
    COUNTER_BUFFER_ELEMENT_OFFSET, DEFAULT_ALLOCATION_BLOCK_POLICY, DEFAULT_BUFFER_STAGING_BUDGET,
    DEFAULT_COUNTER_STAGING_BUDGET, DEFAULT_SHARED_STAGING_BUDGET,
    DEFAULT_STAGING_AGGREGATE_BUDGET, DEFAULT_STAGING_POLICY, StagingPolicy, StagingPolicyError,
    rgba8_image_bytes, staging_bucket_size,
};

mod texture;
pub use texture::{
    ImageMip, TextureCopyRegion, TextureFormat, TextureRegion, validate_rgba8_mips,
    validate_texture_copy, validate_texture_mips, validate_texture_region,
};

mod transfer;
pub use transfer::{
    AllocationError, ReusableStagingPool, StagingEntry, TransferWorker, TransferWorkerError,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Placement and CPU-visibility class requested for an allocation.
pub enum MemoryClass {
    /// Device-local memory.
    Device,
    /// Host-visible upload memory.
    Upload,
    /// Host-visible readback memory.
    Readback,
    /// Transient aliasable memory.
    Transient,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Size, alignment, placement, mapping, and optional aliasing constraints for one allocation.
pub struct AllocationRequest {
    /// Requested byte size.
    pub size: u64,
    /// Required byte alignment.
    pub alignment: u64,
    /// Placement class.
    pub memory_class: MemoryClass,
    /// Whether the allocation must be mapped.
    pub mapped: bool,
    /// Optional transient aliasing class.
    pub alias_class: Option<u64>,
}
impl AllocationRequest {
    ///
    /// # Errors
    ///
    /// Returns an error for zero size, invalid alignment, incompatible mapping,
    /// or an invalid alias class.
    /// Alignment must be a nonzero power of two; alias classes are transient-only.
    pub fn new(
        size: u64,
        alignment: u64,
        memory_class: MemoryClass,
        mapped: bool,
        alias_class: Option<u64>,
    ) -> Result<Self, AllocationError> {
        if size == 0 {
            return Err(AllocationError::ZeroSize);
        }
        if !alignment.is_power_of_two() {
            return Err(AllocationError::InvalidAlignment);
        }
        if mapped && !matches!(memory_class, MemoryClass::Upload | MemoryClass::Readback) {
            return Err(AllocationError::NotHostVisible);
        }
        match (memory_class, alias_class) {
            (MemoryClass::Transient, Some(0) | None) => {
                return Err(AllocationError::InvalidAliasClass);
            }
            (MemoryClass::Transient, Some(_)) | (_, None) => {}
            (_, Some(_)) => return Err(AllocationError::InvalidAliasClass),
        }

        Ok(Self {
            size,
            alignment,
            memory_class,
            mapped,
            alias_class,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Reflected descriptor interval for one public shader-buffer namespace.
pub struct ShaderBufferLayout {
    /// Descriptor namespace containing the buffer bindings.
    pub space: u32,
    /// First buffer binding in the descriptor namespace.
    pub binding: u32,
    /// Number of contiguous buffer descriptors, from one through two.
    pub descriptor_count: u32,
    /// Whether shaders may write through these buffer descriptors.
    pub writable: bool,
}

impl ShaderBufferLayout {
    /// Descriptor arrays are bounded to the two-buffer indirect ABI; physical-range overflow is rejected.
    /// Creates a validated layout for a bounded buffer descriptor array.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::InvalidArgument`] for a zero or excessive count or a binding-range overflow.
    pub fn new(
        space: u32,
        binding: u32,
        descriptor_count: u32,
        writable: bool,
    ) -> Result<Self, HalError> {
        if descriptor_count == 0
            || descriptor_count > 2
            || binding.checked_add(descriptor_count).is_none()
        {
            return Err(HalError::InvalidArgument);
        }
        Ok(Self {
            space,
            binding,
            descriptor_count,
            writable,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Reflected descriptor and argument-buffer layout for a bindless sampled-texture heap.
pub struct ShaderTextureHeapLayout {
    /// Descriptor namespace containing the texture heap.
    pub space: u32,
    /// Binding assigned to the interleaved texture and sampler heap.
    pub binding: u32,
    /// Maximum number of texture and sampler entries, capped at 1024.
    pub capacity: u32,
    /// Byte stride between consecutive argument entries.
    pub argument_stride: u32,
    /// Byte offset of the texture argument within each entry.
    pub texture_argument_offset: u32,
    /// Byte offset of the sampler argument within each entry.
    pub sampler_argument_offset: u32,
}

impl ShaderTextureHeapLayout {
    /// Texture heaps use one bounded interleaved texture/sampler argument array.
    /// Creates a validated interleaved texture and sampler heap layout.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::InvalidArgument`] for an invalid capacity, stride, or argument offset.
    pub fn new(
        space: u32,
        binding: u32,
        capacity: u32,
        argument_stride: u32,
        texture_argument_offset: u32,
        sampler_argument_offset: u32,
    ) -> Result<Self, HalError> {
        if capacity == 0
            || capacity > MAX_BINDLESS_SAMPLED_TEXTURES
            || argument_stride == 0
            || texture_argument_offset >= argument_stride
            || sampler_argument_offset >= argument_stride
            || texture_argument_offset == sampler_argument_offset
        {
            return Err(HalError::InvalidArgument);
        }
        Ok(Self {
            space,
            binding,
            capacity,
            argument_stride,
            texture_argument_offset,
            sampler_argument_offset,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Texture filtering mode used for minification or magnification.
pub enum SamplerFilter {
    /// Selects the nearest texel without interpolation.
    Nearest,
    /// Interpolates neighboring texels linearly.
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Texture-coordinate behavior outside the normalized image extent.
pub enum SamplerAddressMode {
    /// Clamps texture coordinates to the edge texels.
    Clamp,
    /// Wraps texture coordinates periodically.
    Repeat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Filtering, anisotropy, and coordinate addressing for a sampled texture.
pub struct TextureSamplerDesc {
    /// Filtering applied when a texture is minified.
    pub min_filter: SamplerFilter,
    /// Filtering applied when a texture is magnified.
    pub mag_filter: SamplerFilter,
    /// Maximum anisotropy ratio requested for texture filtering.
    pub max_anisotropy: f32,
    /// Addressing mode for the U texture coordinate.
    pub address_u: SamplerAddressMode,
    /// Addressing mode for the V texture coordinate.
    pub address_v: SamplerAddressMode,
    /// Addressing mode for the W texture coordinate.
    pub address_w: SamplerAddressMode,
}

/// Backend-owned allocator boundary. Native implementations derive physical requirements, while this
/// contract makes mapping, cache visibility, immediate free, and timeline retirement explicit.
pub trait MemoryAllocator {
    /// Backend-owned memory record passed to mapping, transfer, and release operations.
    type Allocation;

    /// Allocates memory satisfying the validated request.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError`] when the request cannot be allocated.
    fn allocate(&mut self, request: AllocationRequest)
    -> Result<Self::Allocation, AllocationError>;
    /// Maps an allocation to an immutable byte slice.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation error when mapping is unavailable.
    fn mapped_slice<'a>(
        &self,
        allocation: &'a Self::Allocation,
    ) -> Result<&'a [u8], AllocationError>;
    /// Maps an allocation to a mutable byte slice.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation error when mapping is unavailable.
    fn mapped_slice_mut<'a>(
        &mut self,
        allocation: &'a mut Self::Allocation,
    ) -> Result<&'a mut [u8], AllocationError>;
    /// Makes host writes in the byte range visible to the device.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError`] when the byte range cannot be flushed.
    fn flush(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError>;
    /// Invalidates an allocation range for host reads.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation error when the range is invalid.
    fn invalidate(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError>;
    /// Releases an allocation immediately.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError`] when the owned allocation cannot be released immediately.
    fn free(&mut self, allocation: Self::Allocation) -> Result<(), AllocationError>;
    /// Defers allocation release until completion.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation error when retirement cannot be recorded.
    fn retire(
        &mut self,
        allocation: Self::Allocation,
        completion: CompletionToken,
    ) -> Result<(), AllocationError>;
    /// Releases retired allocations whose queue timeline has reached `completed`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError`] when retired allocations cannot be reclaimed through the completed timeline counter.
    fn reclaim(&mut self, queue: QueueKind, completed: u64) -> Result<usize, AllocationError>;
}

/// Records asynchronous buffer copies and exposes the real queue timeline used to retire staging.
pub trait BufferTransfer: MemoryAllocator {
    /// Records a buffer copy.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation error when ranges are invalid.
    fn copy_buffer(
        &mut self,
        source: &Self::Allocation,
        destination: &Self::Allocation,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    ) -> Result<CompletionToken, AllocationError>;

    /// Returns the latest completed transfer-queue timeline value.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError`] when the completed transfer timeline cannot be queried.
    fn completed_transfer_value(&self) -> Result<u64, AllocationError>;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// A queue timeline point that becomes complete after submitted work retires.
pub struct CompletionToken {
    /// Queue whose timeline tracks completion.
    pub queue: QueueKind,
    /// Nonzero timeline counter reached when the submitted work completes.
    pub value: u64,
}

impl CompletionToken {
    /// Timeline value zero is reserved for work that has not been submitted.
    /// # Errors
    ///
    /// Returns [`ContractError::ZeroTimeline`] when the timeline counter is zero.
    pub fn new(queue: QueueKind, value: u64) -> Result<Self, ContractError> {
        if value == 0 {
            return Err(ContractError::ZeroTimeline);
        }
        Ok(Self { queue, value })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Backend execution domain that owns work and completion timelines.
pub enum QueueKind {
    /// Queue for rendering, compute dispatches, and copies.
    Graphics,
    /// Queue dedicated to compute dispatches and compatible transfers.
    Compute,
    /// Queue dedicated to data transfers.
    Transfer,
    /// Queue dedicated to texture transfers and layout finalization.
    TextureTransfer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Nonempty byte interval within a buffer allocation.
pub struct BufferRange {
    /// Starting byte offset in the buffer.
    pub offset: u64,
    /// Length of the buffer interval in bytes.
    pub size: u64,
}

impl BufferRange {
    /// Empty ranges and ranges whose end overflows are rejected before backend lowering.
    /// # Errors
    ///
    /// Returns [`ContractError::EmptyRange`] for zero size or [`ContractError::RangeOverflow`] when the end is not representable.
    pub fn new(offset: u64, size: u64) -> Result<Self, ContractError> {
        if size == 0 {
            return Err(ContractError::EmptyRange);
        }
        offset
            .checked_add(size)
            .ok_or(ContractError::RangeOverflow)?;
        Ok(Self { offset, size })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Contiguous mip-level and array-layer interval within an image.
pub struct ImageSubresources {
    /// Index of the first mip level.
    pub first_mip: u32,
    /// Number of consecutive mip levels.
    pub mip_count: u32,
    /// Index of the first array layer.
    pub first_layer: u32,
    /// Number of consecutive array layers.
    pub layer_count: u32,
}

impl ImageSubresources {
    /// Creates validated half-open mip-level and array-layer ranges.
    /// # Errors
    ///
    /// Returns [`ContractError::EmptyRange`] for a zero count or [`ContractError::RangeOverflow`] for an unrepresentable range end.
    pub fn new(
        first_mip: u32,
        mip_count: u32,
        first_layer: u32,
        layer_count: u32,
    ) -> Result<Self, ContractError> {
        if mip_count == 0 || layer_count == 0 {
            return Err(ContractError::EmptyRange);
        }
        first_mip
            .checked_add(mip_count)
            .ok_or(ContractError::RangeOverflow)?;
        first_layer
            .checked_add(layer_count)
            .ok_or(ContractError::RangeOverflow)?;
        Ok(Self {
            first_mip,
            mip_count,
            first_layer,
            layer_count,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Rasterizer face-selection mode.
pub enum CullMode {
    /// Disables face culling.
    None,
    /// Culls front-facing primitives.
    Front,
    /// Culls back-facing primitives.
    Back,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Vertex winding interpreted as the front face.
pub enum FrontFace {
    /// Treats counter-clockwise winding as front-facing.
    CounterClockwise,
    /// Treats clockwise winding as front-facing.
    Clockwise,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Vertex assembly used by a graphics pipeline.
pub enum PrimitiveTopology {
    /// Interprets each group of three vertices as an independent triangle.
    TriangleList,
    /// Interprets each vertex as an independent point.
    PointList,
    /// Interprets each pair of vertices as an independent line.
    LineList,
    /// Connects each vertex after the first to the preceding vertex.
    LineStrip,
    /// Forms a triangle from each vertex after the first two and its two predecessors.
    TriangleStrip,
    /// Forms triangles sharing the first vertex as the fan center.
    TriangleFan,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Color blending equation selected for a graphics pipeline.
pub enum BlendMode {
    /// Disables color blending.
    None,
    /// Blends source color using source alpha.
    Alpha,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Depth-test and depth-write behavior selected for a graphics pipeline.
pub enum DepthMode {
    /// Disables depth testing and writes.
    Disabled,
    /// Tests against the depth attachment without modifying it.
    ReadOnly,
    /// Tests against and writes the depth attachment.
    Write,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Shader stages selected for a mesh graphics pipeline.
pub struct MeshStages<T> {
    /// Optional task shader.
    pub task: Option<T>,
    /// Required mesh shader.
    pub mesh: T,
    /// Required fragment shader.
    pub fragment: T,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Rasterization and blending choices supplied for a mesh graphics pipeline.
pub struct MeshPipelineState {
    /// Face-culling mode for rasterization.
    pub cull: CullMode,
    /// Vertex winding interpreted as front-facing.
    pub front_face: FrontFace,
    /// Color blending mode.
    pub blend: BlendMode,
    /// Depth test/write mode.
    pub depth: DepthMode,
}

impl MeshPipelineState {
    /// Resolves explicit state with a shader-declared depth attachment requirement.
    #[must_use]
    pub const fn resolved_depth(self, shader_requires_depth: bool) -> DepthMode {
        if shader_requires_depth && matches!(self.depth, DepthMode::Disabled) {
            DepthMode::Write
        } else {
            self.depth
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Failure class for direct mesh dispatch validation.
pub enum MeshDispatchError {
    /// Caller-supplied dispatch counts are zero, overflow, or exceed a grid limit.
    InvalidGroups,
    /// A reflected task or mesh workgroup shape is zero or overflows.
    InvalidWorkgroup,
    /// A valid reflected workgroup shape exceeds the selected device limit.
    UnsupportedWorkgroup,
}

impl fmt::Display for MeshDispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MeshDispatchError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Native limits used to validate a direct mesh dispatch.
pub struct MeshDispatchLimits {
    /// Maximum grid count in each dimension.
    pub max_groups: [u32; 3],
    /// Maximum total grid count across all dimensions.
    pub max_total_groups: u64,
    /// Maximum threads in one mesh threadgroup.
    pub max_mesh_threads: u32,
    /// Maximum threads in one task/object threadgroup.
    pub max_task_threads: u32,
}

/// Validates direct mesh grid and reflected threadgroup dimensions without allocation.
///
/// # Errors
///
/// Returns [`MeshDispatchError::InvalidGroups`] for invalid caller dispatch counts,
/// [`MeshDispatchError::InvalidWorkgroup`] for malformed reflected dimensions, or
/// [`MeshDispatchError::UnsupportedWorkgroup`] when valid reflection exceeds native limits.
pub fn validate_mesh_dispatch(
    groups: [u32; 3],
    mesh_threads: [u32; 3],
    task_threads: Option<[u32; 3]>,
    limits: MeshDispatchLimits,
) -> Result<(), MeshDispatchError> {
    fn product(dimensions: [u32; 3], invalid: MeshDispatchError) -> Result<u64, MeshDispatchError> {
        if dimensions.contains(&0) {
            return Err(invalid);
        }
        dimensions.into_iter().try_fold(1_u64, |total, value| {
            total.checked_mul(u64::from(value)).ok_or(invalid)
        })
    }

    let group_count = product(groups, MeshDispatchError::InvalidGroups)?;
    if groups
        .into_iter()
        .zip(limits.max_groups)
        .any(|(count, limit)| count > limit)
        || group_count > limits.max_total_groups
    {
        return Err(MeshDispatchError::InvalidGroups);
    }
    if product(mesh_threads, MeshDispatchError::InvalidWorkgroup)?
        > u64::from(limits.max_mesh_threads)
    {
        return Err(MeshDispatchError::UnsupportedWorkgroup);
    }
    if let Some(threads) = task_threads
        && product(threads, MeshDispatchError::InvalidWorkgroup)?
            > u64::from(limits.max_task_threads)
    {
        return Err(MeshDispatchError::UnsupportedWorkgroup);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Rasterization, topology, blending, and depth choices supplied at pipeline creation.
pub struct DynamicPipelineState {
    /// Face-culling mode for rasterization.
    pub cull: CullMode,
    /// Vertex winding interpreted as front-facing.
    pub front_face: FrontFace,
    /// Primitive assembly topology.
    pub topology: PrimitiveTopology,
    /// Color blending mode.
    pub blend: BlendMode,
    /// Depth test/write mode.
    pub depth: DepthMode,
}

impl DynamicPipelineState {
    /// Decodes the legacy render state. Depth stays disabled.
    ///
    /// # Errors
    ///
    /// Returns [`RenderStateError::InvalidDiscriminant`] when any ABI byte has no defined encoding.
    pub fn from_abi(
        cull: u8,
        front_face: u8,
        topology: u8,
        blend: u8,
    ) -> Result<Self, RenderStateError> {
        Self::from_abi_with_depth(cull, front_face, topology, blend, 0)
    }

    /// Decodes render state including explicit depth behavior.
    ///
    /// # Errors
    ///
    /// Returns [`RenderStateError::InvalidDiscriminant`] when any ABI byte has no defined encoding.
    pub fn from_abi_with_depth(
        cull: u8,
        front_face: u8,
        topology: u8,
        blend: u8,
        depth: u8,
    ) -> Result<Self, RenderStateError> {
        Ok(Self {
            cull: match cull {
                0 => CullMode::None,
                1 => CullMode::Front,
                2 => CullMode::Back,
                _ => return Err(RenderStateError::InvalidDiscriminant),
            },
            front_face: match front_face {
                0 => FrontFace::CounterClockwise,
                1 => FrontFace::Clockwise,
                _ => return Err(RenderStateError::InvalidDiscriminant),
            },
            topology: match topology {
                0 => PrimitiveTopology::TriangleList,
                1 => PrimitiveTopology::PointList,
                2 => PrimitiveTopology::LineList,
                3 => PrimitiveTopology::LineStrip,
                4 => PrimitiveTopology::TriangleStrip,
                5 => PrimitiveTopology::TriangleFan,
                _ => return Err(RenderStateError::InvalidDiscriminant),
            },
            blend: match blend {
                0 => BlendMode::None,
                1 => BlendMode::Alpha,
                _ => return Err(RenderStateError::InvalidDiscriminant),
            },
            depth: match depth {
                0 => DepthMode::Disabled,
                1 => DepthMode::ReadOnly,
                2 => DepthMode::Write,
                _ => return Err(RenderStateError::InvalidDiscriminant),
            },
        })
    }

    /// Resolves explicit state with the legacy shader depth requirement.
    #[must_use]
    pub const fn resolved_depth(self, shader_requires_depth: bool) -> DepthMode {
        if shader_requires_depth && matches!(self.depth, DepthMode::Disabled) {
            DepthMode::Write
        } else {
            self.depth
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Failures while decoding render state from the C ABI.
pub enum RenderStateError {
    /// A C ABI byte does not encode a recognized render-state choice.
    InvalidDiscriminant,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Programmable pipeline stage associated with a resource access.
pub enum ShaderStage {
    /// No programmable shader stage participates.
    None = 0,
    /// Vertex shader stage.
    Vertex = 1,
    /// Fragment shader stage.
    Fragment = 2,
    /// Compute shader stage.
    Compute = 3,
    /// All programmable graphics stages.
    AllGraphics = 4,
    /// Task/object shader stage.
    Task = 5,
    /// Mesh shader stage.
    Mesh = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Semantic read or write performed on a GPU resource.
pub enum ResourceAccess {
    /// Shader sampling reads from a texture or buffer.
    SampledRead,
    /// Shader storage reads without writes.
    StorageRead,
    /// Shader storage writes without preserving prior contents.
    StorageWrite,
    /// Shader storage reads and writes.
    StorageReadWrite,
    /// Index data is read during indexed drawing.
    IndexRead,
    /// Indirect command arguments are read by command processing.
    IndirectRead,
    /// Indirect arguments are read by command processing and exposed as read-only shader storage.
    IndirectStorageRead,
    /// Indirect arguments are read by command processing and exposed as read-write shader storage.
    IndirectStorageReadWrite,
    /// A color attachment receives render writes.
    ColorAttachmentWrite,
    /// A depth-stencil attachment is read without modification.
    DepthStencilRead,
    /// A depth-stencil attachment receives render writes.
    DepthStencilWrite,
    /// A transfer reads from the resource.
    TransferRead,
    /// A transfer writes to the resource.
    TransferWrite,
    /// An image is read by presentation.
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Queue ownership, access mode, and shader stage for a GPU resource.
pub struct ResourceState {
    /// Queue that owns and accesses the resource.
    pub queue: QueueKind,
    /// Shader stage participating in the access.
    pub stage: ShaderStage,
    /// Intended resource access category.
    pub access: ResourceAccess,
}

impl ResourceState {
    /// Transfer queues carry no shader stage; index and attachment/present access require graphics.
    /// Creates a resource state after checking queue, stage, and access compatibility.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::InvalidState`] when the queue, shader stage, and access combination is incompatible.
    pub fn new(
        queue: QueueKind,
        stage: ShaderStage,
        access: ResourceAccess,
    ) -> Result<Self, ContractError> {
        let transfer_access = matches!(
            access,
            ResourceAccess::TransferRead | ResourceAccess::TransferWrite
        );
        let shader_access = matches!(
            access,
            ResourceAccess::SampledRead
                | ResourceAccess::StorageRead
                | ResourceAccess::StorageWrite
                | ResourceAccess::StorageReadWrite
                | ResourceAccess::IndirectStorageRead
                | ResourceAccess::IndirectStorageReadWrite
        );
        // Transfer queues cannot execute fixed-function or programmable-stage accesses.
        if matches!(queue, QueueKind::Transfer | QueueKind::TextureTransfer)
            && (stage != ShaderStage::None || !transfer_access)
        {
            return Err(ContractError::InvalidState);
        }
        if queue == QueueKind::Compute
            && matches!(
                stage,
                ShaderStage::Vertex
                    | ShaderStage::Fragment
                    | ShaderStage::AllGraphics
                    | ShaderStage::Task
                    | ShaderStage::Mesh
            )
        {
            return Err(ContractError::InvalidState);
        }
        if shader_access && stage == ShaderStage::None {
            return Err(ContractError::InvalidState);
        }
        // Index buffers lower to graphics-only vertex-input/index-buffer states on every backend.
        if access == ResourceAccess::IndexRead && queue != QueueKind::Graphics {
            return Err(ContractError::InvalidState);
        }
        if matches!(
            access,
            ResourceAccess::ColorAttachmentWrite
                | ResourceAccess::DepthStencilRead
                | ResourceAccess::DepthStencilWrite
                | ResourceAccess::Present
        ) && queue != QueueKind::Graphics
        {
            return Err(ContractError::InvalidState);
        }
        if matches!(
            access,
            ResourceAccess::ColorAttachmentWrite
                | ResourceAccess::DepthStencilRead
                | ResourceAccess::DepthStencilWrite
        ) && !matches!(stage, ShaderStage::Fragment | ShaderStage::AllGraphics)
        {
            return Err(ContractError::InvalidState);
        }
        if transfer_access && stage != ShaderStage::None {
            return Err(ContractError::InvalidState);
        }
        if access == ResourceAccess::Present && stage != ShaderStage::None {
            return Err(ContractError::InvalidState);
        }

        Ok(Self {
            queue,
            stage,
            access,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Violations of backend-neutral range, state, and execution contracts.
pub enum ContractError {
    /// A byte or subresource range has zero length.
    EmptyRange,
    /// A range end exceeds the integer representation.
    RangeOverflow,
    /// A queue, shader stage, or access combination is incompatible.
    InvalidState,
    /// A completion token uses the reserved zero timeline counter.
    ZeroTimeline,
    /// An RGBA8 mip chain has invalid dimensions, sequence, or byte length.
    InvalidImage,
}

impl fmt::Display for ContractError {
    /// Formats the validation error using its debug name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for ContractError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Portable backend operation failures exposed to runtime callers.
pub enum HalError {
    /// An argument fails HAL validation.
    InvalidArgument,
    /// The backend lacks the requested capability.
    Unsupported,
    /// The requested result is not yet available.
    NotReady,
    /// Insufficient memory prevented completion.
    OutOfMemory,
    /// A native API reported an unclassified failure.
    NativeFailure,
    /// The device is no longer available.
    DeviceLost,
}

/// Static-dispatch backend contract. Native handle types never cross this package boundary.
pub trait Backend: Sized {
    /// Backend device and its owned queues.
    type Device;
    /// Backend-native buffer handle.
    type Buffer;
    /// Backend-native image handle.
    type Image;
    /// Backend-native command recording handle.
    type CommandBuffer;

    /// Enumerates adapters visible to this backend.
    ///
    /// # Errors
    ///
    /// Returns [`HalError`] when adapter discovery cannot complete.
    fn enumerate_adapters() -> Result<Vec<AdapterInfo>, HalError>;
    /// Creates a device for an admitted adapter.
    ///
    /// # Errors
    ///
    /// Returns [`HalError`] when device creation fails.
    fn create_device(adapter: &AdapterInfo) -> Result<Self::Device, HalError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Buffer bytes or image subresources affected by an execution transition.
pub enum ExecutionRange {
    /// Selects a byte interval within a buffer.
    Buffer(BufferRange),
    /// Selects mip levels and array layers within an image.
    Image(ImageSubresources),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Dependency consumed before one execution node begins.
pub struct ExecutionWait {
    /// Execution-node index that consumes the dependency.
    pub node: u32,
    /// Optional execution-node index providing an internal dependency.
    pub source: Option<u32>,
    /// Optional queue timeline completion required from external work.
    pub external: Option<CompletionToken>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Resource transition applied before one execution node.
pub struct ExecutionBarrier {
    /// Execution-node index preceded by this resource transition.
    pub node: u32,
    /// Resource index to transition.
    pub resource: u32,
    /// Buffer bytes or image subresources covered by the transition.
    pub range: ExecutionRange,
    /// Known prior resource state, or `None` when no prior state is declared.
    pub before: Option<ResourceState>,
    /// Resource state required before the node executes.
    pub after: ResourceState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Treatment of attachment contents when a render pass begins.
pub enum AttachmentLoadOp {
    /// Preserves existing attachment contents at pass start.
    Load,
    /// Initializes the attachment with its clear data at pass start.
    Clear,
    /// Leaves prior attachment contents undefined at pass start.
    Discard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Treatment of attachment contents when a render pass ends.
pub enum AttachmentStoreOp {
    /// Preserves attachment contents after the pass.
    Store,
    /// Allows attachment contents to become undefined after the pass.
    Discard,
}

/// Legacy clear color applied to surface passes.
///
/// Render targets carry their stored declaration clear instead; surfaces
/// have no declaration, so every backend preserves this value for them.
pub const SURFACE_DEFAULT_CLEAR: [f32; 4] = [0.1, 0.1, 0.1, 1.0];

#[derive(Clone, Debug, Eq, PartialEq)]
/// Ordered render nodes and attachment policy encoded as one native render pass.
pub struct ExecutionPass {
    /// Execution-node indices recorded inside the render pass.
    pub nodes: Vec<u32>,
    /// Resource indices used as color attachments.
    pub colors: Vec<u32>,
    /// Optional resource index used as the depth attachment.
    pub depth: Option<u32>,
    /// Render area encoded as `[x, y, width, height]` in pixels.
    pub area: [u32; 4],
    /// Rasterization sample count.
    pub samples: u8,
    /// Attachment behavior at pass start.
    pub load: AttachmentLoadOp,
    /// Attachment behavior at pass end.
    pub store: AttachmentStoreOp,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Backend command emitted while executing a compiled frame graph.
pub enum ExecutionAction {
    /// Waits for an internal dependency or external queue timeline.
    Wait(ExecutionWait),
    /// Transitions a resource range between access states.
    Barrier(ExecutionBarrier),
    /// Begins a render pass with the declared attachments and nodes.
    BeginPass(ExecutionPass),
    /// Executes the payload for the indexed node.
    ExecuteNode(u32),
    /// Ends the current render pass.
    EndPass,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Fully validated backend command stream for one frame.
pub struct FrameExecutionPlan {
    /// Ordered commands forming the frame command stream.
    pub actions: Vec<ExecutionAction>,
}

/// Records a validated frame plan against backend-specific payloads.
pub trait FrameExecutionBackend<P> {
    /// Failure returned when plan recording or completion fails.
    type Error;

    /// Records and submits every action in `plan` using the corresponding payloads.
    ///
    /// # Errors
    ///
    /// Returns the backend-defined execution error when the preflighted plan cannot be recorded or completed.
    fn execute(&mut self, plan: &FrameExecutionPlan, payloads: &[P]) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod telemetry_tests {
    use super::{AllocatorTelemetry, rgba8_image_bytes};

    #[test]
    fn waste_is_capacity_minus_live_without_wrapping() {
        let telemetry = AllocatorTelemetry {
            live_bytes: 100,
            block_bytes: 256,
            block_count: 2,
            allocation_count: 3,
        };
        assert_eq!(telemetry.waste_bytes(), 156);
        // A corrupt snapshot with live above capacity saturates instead of wrapping.
        let inverted = AllocatorTelemetry {
            live_bytes: 300,
            ..telemetry
        };
        assert_eq!(inverted.waste_bytes(), 0);
    }

    #[test]
    fn image_estimates_scale_and_saturate() {
        assert_eq!(rgba8_image_bytes(3, 1920, 1080), 3 * 1920 * 1080 * 4);
        assert_eq!(rgba8_image_bytes(0, 1920, 1080), 0);
        assert_eq!(rgba8_image_bytes(u32::MAX, u32::MAX, u32::MAX), u64::MAX);
    }
}
