use core::{cell::Cell, ffi::c_void, marker::PhantomData, mem::ManuallyDrop, ops::Deref};

use crate::{
    BACKEND, TEXTURE_DESCRIPTOR_CAPACITY,
    frame_slots::{FRAMES_IN_FLIGHT, FrameSlotTracker, complete_deferred_slot},
};
use ez_gfx_core::capability::{
    AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport, SemanticProfile,
};

const MAX_ARGUMENT_BUFFERS_PER_SLOT: usize = 1024;
use ez_gfx_hal::{
    AllocationError, AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BlendMode,
    BufferTransfer, CompletionToken, CullMode, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DynamicPipelineState, ExecutionBarrier, ExecutionPass, FrontFace, HalError, ImageMip,
    MemoryAllocator, MemoryClass, PrimitiveTopology, QueueKind, SamplerAddressMode, SamplerFilter,
    ShaderTextureHeapLayout, TextureFormat, TextureRegion, TextureSamplerDesc,
    validate_texture_mips, validate_texture_region,
};
use gpu_allocator::{
    AllocationSizes, MemoryLocation,
    metal::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLArgumentBuffersTier, MTLArgumentEncoder, MTLBlendFactor, MTLBlitCommandEncoder, MTLBuffer,
    MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompareFunction, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLCullMode, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLEvent, MTLFunction, MTLHeap, MTLIndexType, MTLLibrary, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLRenderStages, MTLResource,
    MTLResourceOptions, MTLResourceUsage, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage, MTLWinding,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
/// Retains a non-`Send` Objective-C value for access and destruction on its creating thread.
#[doc(hidden)]
pub struct ThreadBound<T> {
    owner: std::thread::ThreadId,
    value: ManuallyDrop<T>,
    not_sync: PhantomData<Cell<()>>,
}

impl<T> ThreadBound<T> {
    fn new(value: T) -> Self {
        Self {
            owner: std::thread::current().id(),
            value: ManuallyDrop::new(value),
            not_sync: PhantomData,
        }
    }

    fn is_current(&self) -> bool {
        self.owner == std::thread::current().id()
    }
}

impl<T> Deref for ThreadBound<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        if !self.is_current() {
            std::process::abort();
        }
        &self.value
    }
}

impl<T> Drop for ThreadBound<T> {
    fn drop(&mut self) {
        if self.is_current() {
            // SAFETY: owner-thread access is enforced before the wrapped value is released.
            unsafe { ManuallyDrop::drop(&mut self.value) };
        }
    }
}

// SAFETY: moving the wrapper cannot expose `T`: dereference is restricted to the creating
// thread, and dropping it elsewhere leaks rather than accessing or releasing `T`.
unsafe impl<T> Send for ThreadBound<T> {}

/// Metal buffer and allocator ownership record.
pub struct NativeAllocation {
    buffer: ThreadBound<Retained<ProtocolObject<dyn MTLBuffer>>>,
    allocation: ThreadBound<Allocation>,
    mapped_address: usize,
}
/// Buffer argument resolved for one dispatch or draw.
pub struct NativeBufferBinding<'a> {
    /// Allocation supplying the buffer.
    pub allocation: &'a NativeAllocation,
    /// Byte offset passed to the shader stage.
    pub offset: usize,
    /// Metal buffer argument index.
    pub index: usize,
}

/// Fully resolved indexed draw consumed by Metal encoding.
pub struct NativeGraphicsDraw<'a> {
    /// Graphics pipeline used by the draw.
    pub pipeline: &'a NativePipeline,
    /// Whether the draw requires a depth attachment.
    pub depth_required: bool,
    /// Reflected texture argument-buffer layout, when present.
    pub texture_heap: Option<ShaderTextureHeapLayout>,
    /// Dynamic rasterization and blend state.
    pub state: DynamicPipelineState,
    /// Buffer containing 32-bit indices.
    pub index: &'a NativeAllocation,
    /// Logical byte length of the index resource.
    pub index_size: u64,
    /// Buffer containing indexed indirect commands.
    pub indirect: &'a NativeAllocation,
    /// Logical byte length of the indirect resource.
    pub indirect_size: u64,
    /// Number of indirect commands to encode.
    pub draw_count: u32,
    /// Inline constant payload.
    pub push_constants: &'a [u8],
    /// Reflected public buffer bindings.
    pub bindings: &'a [NativeBufferBinding<'a>],
    /// Textures referenced by the argument buffer.
    pub textures: &'a [&'a NativeTexture],
}

/// Fully resolved compute dispatch consumed by Metal encoding.
pub struct NativeComputeDispatch<'a> {
    /// Compute pipeline used by the dispatch.
    pub pipeline: &'a NativePipeline,
    /// Workgroup count for each dimension.
    pub groups: [u32; 3],
    /// Threads launched in each workgroup, reflected from the compute entry point.
    pub threads_per_group: [u32; 3],
    /// Inline constant payload.
    pub push_constants: &'a [u8],
    /// Reflected public buffer bindings.
    pub bindings: &'a [NativeBufferBinding<'a>],
    /// Reflected compute texture argument-buffer layout, when present.
    pub texture_heap: Option<ShaderTextureHeapLayout>,
    /// Textures referenced by the compute argument buffer.
    pub textures: &'a [&'a NativeTexture],
}

/// Metal resource referenced by a compiled frame barrier.
pub enum NativeFrameResource<'a> {
    /// Buffer allocation.
    Buffer(&'a NativeAllocation),
    /// Texture resource.
    Texture(&'a NativeTexture),
    /// Current drawable texture.
    Surface,
    /// Current depth attachment.
    Depth,
}

/// Validated Metal action emitted by the frame-plan adapter.
pub enum NativeFrameAction<'a> {
    /// Wait for external transfer completion.
    Wait(CompletionToken),
    /// Declare ordering and usage for a resource.
    Barrier {
        /// Backend-neutral barrier description.
        barrier: ExecutionBarrier,
        /// Resolved resource affected by the barrier.
        resource: NativeFrameResource<'a>,
    },
    /// Begin the declared render pass.
    BeginPass(&'a ExecutionPass),
    /// Encode a compute dispatch.
    Compute(NativeComputeDispatch<'a>),
    /// Encode indexed indirect graphics work.
    Graphics(NativeGraphicsDraw<'a>),
    /// Copy a texture into shared host-readable storage.
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
    /// Present the current drawable.
    Present,
}

/// Validated metallib products retained as one shader artifact.
pub struct NativeShader {
    libraries: ThreadBound<Vec<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>>>,
}
/// Texture, sampler, and allocation published as one resource.
pub struct NativeTexture {
    texture: ThreadBound<Retained<ProtocolObject<dyn MTLTexture>>>,
    allocation: ThreadBound<Allocation>,
    sampler: ThreadBound<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    format: TextureFormat,
    width: u32,
    height: u32,
    mip_count: u32,
    resident_mips: u32,
    mip_completions: Vec<u64>,
    cancellation: std::sync::Arc<transfer::TransferCancellation>,
    /// Slot in the bindless texture argument buffer.
    pub binding: u32,
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

/// Metal compute or graphics pipeline state.
pub enum NativePipeline {
    /// Compute pipeline state and optional texture argument encoder.
    Compute {
        /// Retained compute pipeline object.
        state: ThreadBound<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        /// Encoder for the compute-stage bindless texture heap.
        argument_encoder: Option<ThreadBound<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>>,
    },
    /// Graphics pipeline state and stage-specific texture argument encoders.
    Graphics {
        /// Retained graphics pipeline object.
        state: ThreadBound<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
        /// Encoder for a vertex-stage bindless texture heap.
        vertex_argument_encoder:
            Option<ThreadBound<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>>,
        /// Encoder for a fragment-stage bindless texture heap.
        fragment_argument_encoder:
            Option<ThreadBound<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>>,
    },
}
struct FrameSlot {
    command: Option<ThreadBound<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    argument_buffers: Vec<ThreadBound<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

enum DeferredResource {
    Allocation(NativeAllocation),
    Pipeline(NativePipeline),
    Depth(SurfaceDepth),
    Shader(NativeShader),
    Surface(NativeSurface),
    Texture(NativeTexture),
}

struct DeferredNativeResource {
    pending_slots: u8,
    resource: DeferredResource,
}

struct RetiredAllocation {
    allocation: NativeAllocation,
    completion: CompletionToken,
}

struct PendingTransfer {
    value: u64,
    command: ThreadBound<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

struct SurfaceDepth {
    texture: ThreadBound<Retained<ProtocolObject<dyn MTLTexture>>>,
    state: ThreadBound<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    extent: (u32, u32),
}

/// Borrowed `CAMetalLayer` and optional captured frame/depth state.
pub struct NativeSurface {
    layer: usize,
    presented_rgba8: Vec<u8>,
    depth: Option<SurfaceDepth>,
}
impl NativeSurface {
    /// The `CAMetalLayer` pointer is borrowed; this crate never releases the host's layer.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::InvalidArgument`] when `layer` is null.
    pub fn new(layer: *mut c_void, capture_presented: bool) -> Result<Self, HalError> {
        if layer.is_null() {
            return Err(HalError::InvalidArgument);
        }
        // SAFETY: the non-null layer is retained by the caller for this surface's lifetime.
        let metal_layer = unsafe { &*(layer as *const CAMetalLayer) };
        // Every surface graph resource is BGRA8 sRGB; there is no linear fallback.
        metal_layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
        // Capture needs shader-readable drawable textures; set this before nextDrawable.
        if capture_presented {
            metal_layer.setFramebufferOnly(false);
        }
        Ok(Self {
            layer: layer as usize,
            presented_rgba8: Vec::new(),
            depth: None,
        })
    }

    /// Empty before the first cached presentation; successful captures replace the full frame.
    pub fn presented_rgba8(&self) -> &[u8] {
        &self.presented_rgba8
    }
}

/// Admitted Metal device, queue, allocator, and frame state.
pub struct NativeContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    transfer_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    texture_graphics_event: Retained<ProtocolObject<dyn MTLEvent>>,
    texture_completion_event: Retained<ProtocolObject<dyn MTLEvent>>,
    allocator: Option<Allocator>,
    retired: Vec<RetiredAllocation>,
    frame_slots: Vec<FrameSlot>,
    frame_tracker: FrameSlotTracker,
    deferred: Vec<DeferredNativeResource>,
    adapter: AdapterInfo,
    drain_complete: bool,
    next_transfer_value: u64,
    completed_transfer_value: u64,
    pending_transfers: Vec<PendingTransfer>,
    transfer_worker: Option<ez_gfx_hal::TransferWorker<transfer::MetalTransferJob>>,
    next_texture_value: u64,
    completed_texture_value: u64,
    pending_texture_transfers: Vec<transfer::PendingTextureTransfer>,
    texture_worker: Option<ez_gfx_hal::TransferWorker<transfer::TextureTransferJob>>,
    texture_staging: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    #[cfg(test)]
    buffer_wait_observer: Option<std::sync::mpsc::Sender<()>>,
}

// Metal device and command-queue protocol objects are `Send + Sync` in objc2-metal. Every
// remaining Objective-C object is held by `ThreadBound`, so `NativeContext` derives `Send`.

mod device;
mod frame;
mod memory;
use memory::{map_allocation_hal, map_allocator, map_allocator_hal};
mod pipeline;
mod surface;
mod texture;
mod transfer;

#[cfg(test)]
mod texture_tests;
