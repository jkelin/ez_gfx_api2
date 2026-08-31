#![forbid(unsafe_code)]

use core::fmt;
use ez_gfx_core::capability::AdapterInfo;

const ALLOCATION_BLOCK_ALIGNMENT: u64 = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationBlockPolicy {
    pub initial_device: u64,
    pub maximum_device: u64,
    pub initial_host: u64,
    pub maximum_host: u64,
}

impl AllocationBlockPolicy {
    pub const fn new(
        initial_device: u64,
        maximum_device: u64,
        initial_host: u64,
        maximum_host: u64,
    ) -> Result<Self, AllocationBlockPolicyError> {
        // gpu-allocator accepts only 4 MiB block increments; reject instead of allowing it to clamp.
        if initial_device == 0 || maximum_device == 0 || initial_host == 0 || maximum_host == 0 {
            return Err(AllocationBlockPolicyError::ZeroSize);
        }
        if !initial_device.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
            || !maximum_device.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
            || !initial_host.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
            || !maximum_host.is_multiple_of(ALLOCATION_BLOCK_ALIGNMENT)
        {
            return Err(AllocationBlockPolicyError::InvalidAlignment);
        }
        if initial_device > maximum_device || initial_host > maximum_host {
            return Err(AllocationBlockPolicyError::InvalidRange);
        }
        Ok(Self {
            initial_device,
            maximum_device,
            initial_host,
            maximum_host,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationBlockPolicyError {
    ZeroSize,
    InvalidAlignment,
    InvalidRange,
}

pub const DEFAULT_ALLOCATION_BLOCK_POLICY: AllocationBlockPolicy = AllocationBlockPolicy {
    initial_device: 16 * 1024 * 1024,
    maximum_device: 256 * 1024 * 1024,
    initial_host: 8 * 1024 * 1024,
    maximum_host: 64 * 1024 * 1024,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryClass {
    Device,
    Upload,
    Readback,
    Transient,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationRequest {
    pub size: u64,
    pub alignment: u64,
    pub memory_class: MemoryClass,
    pub mapped: bool,
    pub alias_class: Option<u64>,
}
impl AllocationRequest {
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
            (MemoryClass::Transient, Some(0)) | (MemoryClass::Transient, None) => {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    ZeroSize,
    InvalidAlignment,
    NotHostVisible,
    InvalidAliasClass,
    OutOfMemory,
    DeviceLost,
    NativeFailure,
}

impl fmt::Display for AllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for AllocationError {}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderBufferLayout {
    pub space: u32,
    pub binding: u32,
    pub descriptor_count: u32,
    pub writable: bool,
}

impl ShaderBufferLayout {
    /// Descriptor arrays are bounded to the two-buffer indirect ABI; physical-range overflow is rejected.
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
pub struct ShaderTextureHeapLayout {
    pub space: u32,
    pub binding: u32,
    pub capacity: u32,
    pub argument_stride: u32,
    pub texture_argument_offset: u32,
    pub sampler_argument_offset: u32,
}

impl ShaderTextureHeapLayout {
    /// Texture heaps use one bounded interleaved texture/sampler argument array.
    pub fn new(
        space: u32,
        binding: u32,
        capacity: u32,
        argument_stride: u32,
        texture_argument_offset: u32,
        sampler_argument_offset: u32,
    ) -> Result<Self, HalError> {
        if capacity == 0
            || capacity > 1024
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
pub enum SamplerFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SamplerAddressMode {
    Clamp,
    Repeat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureSamplerDesc {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub max_anisotropy: f32,
    pub address_u: SamplerAddressMode,
    pub address_v: SamplerAddressMode,
    pub address_w: SamplerAddressMode,
}

/// Backend-owned allocator seam. Native implementations derive physical requirements, while this
/// contract makes mapping, cache visibility, immediate free, and timeline retirement explicit.
pub trait MemoryAllocator {
    type Allocation;

    fn allocate(&mut self, request: AllocationRequest)
    -> Result<Self::Allocation, AllocationError>;
    fn mapped_slice<'a>(
        &self,
        allocation: &'a Self::Allocation,
    ) -> Result<&'a [u8], AllocationError>;
    fn mapped_slice_mut<'a>(
        &mut self,
        allocation: &'a mut Self::Allocation,
    ) -> Result<&'a mut [u8], AllocationError>;
    fn flush(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError>;
    fn invalidate(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError>;
    fn free(&mut self, allocation: Self::Allocation) -> Result<(), AllocationError>;
    fn retire(
        &mut self,
        allocation: Self::Allocation,
        completion: CompletionToken,
    ) -> Result<(), AllocationError>;
    fn reclaim(&mut self, queue: QueueKind, completed: u64) -> Result<usize, AllocationError>;
}

/// Records asynchronous buffer copies and exposes the real queue timeline used to retire staging.
pub trait BufferTransfer: MemoryAllocator {
    /// Source and destination ranges must fit their allocations; zero-byte copies are rejected.
    fn copy_buffer(
        &mut self,
        source: &Self::Allocation,
        destination: &Self::Allocation,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    ) -> Result<CompletionToken, AllocationError>;

    fn completed_transfer_value(&self) -> Result<u64, AllocationError>;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CompletionToken {
    pub queue: QueueKind,
    pub value: u64,
}

impl CompletionToken {
    /// Timeline value zero is reserved for work that has not been submitted.
    pub fn new(queue: QueueKind, value: u64) -> Result<Self, ContractError> {
        if value == 0 {
            return Err(ContractError::ZeroTimeline);
        }
        Ok(Self { queue, value })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QueueKind {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferRange {
    pub offset: u64,
    pub size: u64,
}

impl BufferRange {
    /// Empty ranges and ranges whose end overflows are rejected before backend lowering.
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
pub struct ImageSubresources {
    pub first_mip: u32,
    pub mip_count: u32,
    pub first_layer: u32,
    pub layer_count: u32,
}

impl ImageSubresources {
    /// Counts are nonzero and both half-open range ends must remain representable.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageMip<'a> {
    pub width: u32,
    pub height: u32,
    pub bytes: &'a [u8],
}

/// Validates a complete RGBA8 mip chain without allocating; dimensions clamp at one.
pub fn validate_rgba8_mips(mips: &[ImageMip<'_>]) -> Result<(), ContractError> {
    let Some(first) = mips.first() else {
        return Err(ContractError::InvalidImage);
    };
    if first.width == 0 || first.height == 0 {
        return Err(ContractError::InvalidImage);
    }
    let mut width = first.width;
    let mut height = first.height;
    for mip in mips {
        let expected = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|value| value.checked_mul(4))
            .ok_or(ContractError::RangeOverflow)?;
        if mip.width != width || mip.height != height || mip.bytes.len() as u64 != expected {
            return Err(ContractError::InvalidImage);
        }
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CullMode {
    None,
    Front,
    Back,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FrontFace {
    CounterClockwise,
    Clockwise,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PrimitiveTopology {
    TriangleList,
    PointList,
    LineList,
    LineStrip,
    TriangleStrip,
    TriangleFan,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlendMode {
    None,
    Alpha,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DynamicPipelineState {
    pub cull: CullMode,
    pub front_face: FrontFace,
    pub topology: PrimitiveTopology,
    pub blend: BlendMode,
}

impl DynamicPipelineState {
    /// Every C discriminant is checked; unknown future values fail instead of changing pipeline state.
    pub fn from_abi(
        cull: u8,
        front_face: u8,
        topology: u8,
        blend: u8,
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
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderStateError {
    InvalidDiscriminant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderStage {
    None,
    Vertex,
    Fragment,
    Compute,
    AllGraphics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceAccess {
    SampledRead,
    StorageRead,
    StorageWrite,
    StorageReadWrite,
    IndexRead,
    IndirectRead,
    IndirectStorageRead,
    IndirectStorageReadWrite,
    ColorAttachmentWrite,
    DepthStencilRead,
    DepthStencilWrite,
    TransferRead,
    TransferWrite,
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceState {
    pub queue: QueueKind,
    pub stage: ShaderStage,
    pub access: ResourceAccess,
}

impl ResourceState {
    /// Transfer queues carry no shader stage; attachment and present access require graphics.
    pub fn new(
        queue: QueueKind,
        stage: ShaderStage,
        access: ResourceAccess,
    ) -> Result<Self, ContractError> {
        if queue == QueueKind::Transfer && stage != ShaderStage::None {
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
        if matches!(
            access,
            ResourceAccess::TransferRead | ResourceAccess::TransferWrite
        ) && stage != ShaderStage::None
        {
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
pub enum ContractError {
    EmptyRange,
    RangeOverflow,
    InvalidState,
    ZeroTimeline,
    InvalidImage,
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for ContractError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalError {
    InvalidArgument,
    Unsupported,
    NotReady,
    OutOfMemory,
    NativeFailure,
    DeviceLost,
}

/// Static-dispatch backend contract. Native handle types never cross this package boundary.
pub trait Backend: Sized {
    type Device;
    type Buffer;
    type Image;
    type CommandBuffer;

    fn enumerate_adapters() -> Result<Vec<AdapterInfo>, HalError>;
    fn create_device(adapter: &AdapterInfo) -> Result<Self::Device, HalError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionRange {
    Buffer(BufferRange),
    Image(ImageSubresources),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionWait {
    pub node: u32,
    pub source: Option<u32>,
    pub external: Option<CompletionToken>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionBarrier {
    pub node: u32,
    pub resource: u32,
    pub range: ExecutionRange,
    pub before: Option<ResourceState>,
    pub after: ResourceState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentLoadOp {
    Load,
    Clear,
    Discard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentStoreOp {
    Store,
    Discard,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPass {
    pub nodes: Vec<u32>,
    pub colors: Vec<u32>,
    pub depth: Option<u32>,
    pub area: [u32; 4],
    pub samples: u8,
    pub load: AttachmentLoadOp,
    pub store: AttachmentStoreOp,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionAction {
    Wait(ExecutionWait),
    Barrier(ExecutionBarrier),
    BeginPass(ExecutionPass),
    ExecuteNode(u32),
    EndPass,
}

/// Fully preflighted backend-neutral work. Native implementations record every action into one
/// frame command stream and may present only after all recording succeeds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameExecutionPlan {
    pub actions: Vec<ExecutionAction>,
}

pub trait FrameExecutionBackend<P> {
    type Error;

    fn execute(&mut self, plan: &FrameExecutionPlan, payloads: &[P]) -> Result<(), Self::Error>;
}
