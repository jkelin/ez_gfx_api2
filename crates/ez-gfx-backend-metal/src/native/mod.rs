use core::{cell::Cell, marker::PhantomData, mem::ManuallyDrop, ops::Deref};

use crate::{
    BACKEND, TEXTURE_DESCRIPTOR_CAPACITY,
    frame_slots::{FRAMES_IN_FLIGHT, FrameSlotTracker, complete_deferred_slot},
};
use ez_gfx_core::capability::{
    AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport, PresentationMode,
    PresentationModes, SemanticProfile,
};
use arrayvec::ArrayVec;

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
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use raw_window_metal::Layer;

use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLArgumentBuffersTier, MTLArgumentEncoder, MTLBlendFactor, MTLBlitCommandEncoder, MTLBuffer,
    MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompareFunction, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices,
    MTLCreateSystemDefaultDevice, MTLCullMode, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLEvent, MTLFunction, MTLHeap, MTLIndexType, MTLLibrary, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLRenderStages, MTLResource,
    MTLResourceOptions, MTLResourceUsage, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState, MTLSize, MTLStorageMode,
    MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLWinding,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
const fn detach_backend_layer(pre_existing: bool) -> bool {
    !pre_existing
}

const fn metal_presentation_modes() -> PresentationModes {
    PresentationModes::FIFO.union(PresentationModes::IMMEDIATE)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DrawableExtent {
    Existing(u32, u32),
    Derived(u32, u32),
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite, rounded dimensions are proven positive and within u32 before casting"
)]
fn validated_pixel_extent(dimensions: [f64; 2]) -> Option<(u32, u32)> {
    let [width, height] = dimensions.map(f64::round);
    // Zero/negative values are minimized; nonfinite and post-rounding overflow cannot name pixels.
    if !width.is_finite()
        || !height.is_finite()
        || width <= 0.0
        || height <= 0.0
        || width > f64::from(u32::MAX)
        || height > f64::from(u32::MAX)
    {
        return None;
    }

    Some((width as u32, height as u32))
}

fn resolve_drawable_extent(
    drawable: [f64; 2],
    bounds: [f64; 2],
    contents_scale: f64,
) -> Option<DrawableExtent> {
    if drawable != [0.0, 0.0] {
        return validated_pixel_extent(drawable)
            .map(|(width, height)| DrawableExtent::Existing(width, height));
    }

    // A zero drawable may precede the first drawable acquisition. Invalid scale or bounds must
    // remain NotReady rather than manufacturing a size for a minimized or malformed native layer.
    if !contents_scale.is_finite() || contents_scale <= 0.0 {
        return None;
    }
    validated_pixel_extent([bounds[0] * contents_scale, bounds[1] * contents_scale])
        .map(|(width, height)| DrawableExtent::Derived(width, height))
}

#[cfg(test)]
mod surface_extent_tests {
    use super::{DrawableExtent, resolve_drawable_extent};

    #[test]
    fn drawable_extent_takes_precedence_over_layer_geometry() {
        assert_eq!(
            resolve_drawable_extent([640.0, 480.0], [10.0, 20.0], 3.0),
            Some(DrawableExtent::Existing(640, 480))
        );
    }

    #[test]
    fn zero_drawable_extent_falls_back_to_scaled_layer_bounds() {
        assert_eq!(
            resolve_drawable_extent([0.0, 0.0], [320.0, 240.0], 2.0),
            Some(DrawableExtent::Derived(640, 480))
        );
    }

    #[test]
    fn zero_or_partial_extents_remain_minimized() {
        for (drawable, bounds, scale) in [
            ([0.0, 0.0], [0.0, 480.0], 1.0),
            ([0.0, 0.0], [640.0, 0.0], 1.0),
            ([640.0, 0.0], [640.0, 480.0], 1.0),
            ([0.0, 480.0], [640.0, 480.0], 1.0),
        ] {
            assert_eq!(resolve_drawable_extent(drawable, bounds, scale), None);
        }
    }

    #[test]
    fn nonfinite_or_overflowing_geometry_is_rejected() {
        for (drawable, bounds, scale) in [
            ([f64::NAN, 480.0], [640.0, 480.0], 1.0),
            ([640.0, f64::INFINITY], [640.0, 480.0], 1.0),
            ([0.0, 0.0], [f64::NAN, 480.0], 1.0),
            ([0.0, 0.0], [640.0, 480.0], f64::INFINITY),
            ([0.0, 0.0], [f64::from(u32::MAX), 480.0], 2.0),
        ] {
            assert_eq!(resolve_drawable_extent(drawable, bounds, scale), None);
        }
    }
}

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

/// Synchronous provider for resolved Metal buffer arguments.
pub trait NativeBufferBindingSource {
    /// Number of bindings supplied to the pipeline.
    fn len(&self) -> usize;

    /// Visits each binding; borrowed allocation views cannot escape the call.
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
    /// Reflected public buffer bindings.
    pub bindings: &'a dyn NativeBufferBindingSource,
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
    /// Reflected public buffer bindings.
    pub bindings: &'a dyn NativeBufferBindingSource,
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

/// Synchronous provider whose borrowed native actions live only for each visit.
pub trait NativeFrameActionSource {
    /// Number of stable action records.
    fn len(&self) -> usize;

    /// Visits one action by stable index.
    fn with_action(
        &self,
        index: usize,
        visitor: &mut dyn FnMut(&NativeFrameAction<'_>) -> Result<(), HalError>,
    ) -> Result<(), HalError>;

    /// Visits all actions in order.
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
    /// Multisampled render storage plus its allocation; `None` for uploads
    /// and single-sample targets. Only render-target entry points touch this;
    /// the texture above stays the resolve destination.
    msaa: Option<MsaaStorage>,
}

/// Private multisampled storage resolved into a render target's sampled texture.
pub struct MsaaStorage {
    texture: ThreadBound<Retained<ProtocolObject<dyn MTLTexture>>>,
    allocation: ThreadBound<Allocation>,
    /// Render sample count; passes must request exactly this count.
    samples: u8,
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
    /// Retained prepared-argument entries; pure CPU pairs with no GPU lifetime,
    /// reused across frames once the slot's submission completes.
    prepared_scratch: Vec<(u32, usize)>,
    submission_value: u64,
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

/// Retained `CAMetalLayer`; a created observer layer weakly references its host until detached.
pub struct NativeSurface {
    layer: ThreadBound<Layer>,
    presented_rgba8: Vec<u8>,
    detach_on_destroy: bool,
    original_display_sync_enabled: bool,
    presentation_mode: PresentationMode,
    depth: Option<SurfaceDepth>,
}
impl NativeSurface {
    /// Obtains and retains a Metal layer from a matched `AppKit` handle pair.
    ///
    /// # Safety
    ///
    /// The handles must belong to the same live `AppKit` host and this must run on the main thread.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::InvalidArgument`] for mismatched handles or a non-main-thread call.
    pub unsafe fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        capture_presented: bool,
    ) -> Result<Self, HalError> {
        let (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) = (display, window)
        else {
            return Err(HalError::InvalidArgument);
        };
        if objc2::MainThreadMarker::new().is_none() {
            return Err(HalError::InvalidArgument);
        }
        // SAFETY: the raw handle borrows a live NSView retained by the safe surface host.
        let layer = unsafe { Layer::from_ns_view(window.ns_view) };
        let detach_on_destroy = detach_backend_layer(layer.pre_existing());
        // SAFETY: Layer retains a non-null CAMetalLayer.
        let metal_layer = unsafe { &*layer.as_ptr().cast::<CAMetalLayer>().as_ptr() };
        let original_display_sync_enabled = metal_layer.displaySyncEnabled();
        metal_layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
        if capture_presented {
            metal_layer.setFramebufferOnly(false);
        }
        Ok(Self {
            layer: ThreadBound::new(layer),
            presented_rgba8: Vec::new(),
            detach_on_destroy,
            original_display_sync_enabled,
            presentation_mode: if original_display_sync_enabled {
                PresentationMode::Fifo
            } else {
                PresentationMode::Immediate
            },
            depth: None,
        })
    }

    fn metal_layer(&self) -> &CAMetalLayer {
        // SAFETY: Layer retains a non-null CAMetalLayer.
        unsafe { &*self.layer.as_ptr().cast::<CAMetalLayer>().as_ptr() }
    }

    /// Returns the normalized modes exposed by `CAMetalLayer`.
    pub const fn presentation_modes(&self) -> PresentationModes {
        metal_presentation_modes()
    }

    /// Applies one supported mode before the next drawable is acquired.
    fn set_presentation_mode(&mut self, mode: PresentationMode) -> Result<(), HalError> {
        if !self.presentation_modes().contains(mode) {
            return Err(HalError::Unsupported);
        }
        if self.presentation_mode != mode {
            self.metal_layer()
                .setDisplaySyncEnabled(mode == PresentationMode::Fifo);
            self.presentation_mode = mode;
        }
        Ok(())
    }
    fn detach_from_host(&mut self) {
        self.metal_layer()
            .setDisplaySyncEnabled(self.original_display_sync_enabled);
        if self.detach_on_destroy {
            self.metal_layer().removeFromSuperlayer();
            self.detach_on_destroy = false;
        }
    }
    /// Reads the current drawable extent, deriving an uninitialized size from native layer geometry.
    #[must_use]
    pub fn window_extent(&self) -> Option<(u32, u32)> {
        let layer = self.metal_layer();
        let drawable = layer.drawableSize();
        let resolved = if drawable.width == 0.0 && drawable.height == 0.0 {
            let bounds = layer.bounds();
            resolve_drawable_extent(
                [drawable.width, drawable.height],
                [bounds.size.width, bounds.size.height],
                layer.contentsScale(),
            )
        } else {
            resolve_drawable_extent([drawable.width, drawable.height], [0.0, 0.0], 1.0)
        }?;

        match resolved {
            DrawableExtent::Existing(width, height) => Some((width, height)),
            DrawableExtent::Derived(width, height) => {
                layer.setDrawableSize(objc2_core_foundation::CGSize {
                    width: f64::from(width),
                    height: f64::from(height),
                });
                Some((width, height))
            }
        }
    }

    /// Updates the layer drawable extent after a host resize or DPI change.
    pub fn resize(&self, width: u32, height: u32) {
        self.metal_layer()
            .setDrawableSize(objc2_core_foundation::CGSize {
                width: f64::from(width),
                height: f64::from(height),
            });
    }

    /// Empty before the first cached presentation.
    pub fn presented_rgba8(&self) -> &[u8] {
        &self.presented_rgba8
    }

    /// Reports retained drawable images for on-demand memory telemetry.
    ///
    /// Core Animation owns the drawables; the layer's maximum drawable count is
    /// the only retained-image bound the backend can observe without presenting.
    pub fn telemetry_images(&self) -> u32 {
        // Drawable counts are small; the fallback only guards the conversion.
        u32::try_from(self.metal_layer().maximumDrawableCount()).unwrap_or(u32::MAX)
    }

    /// Reports the swapchain format code for memory telemetry.
    ///
    /// Layers are configured as `BGRA8Unorm_sRGB` (125); the code is constant
    /// so format alignment never depends on presentation having occurred.
    pub const fn telemetry_format(&self) -> u32 {
        125
    }

    /// Reports retained depth storage for memory telemetry.
    ///
    pub fn telemetry_depth_extent(&self) -> Option<(u32, u32)> {
        match &self.depth {
            Some(depth) => Some(depth.extent),
            None => None,
        }
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
    next_frame_value: u64,
    last_frame_value: u64,
    completed_frame_value: u64,
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
mod presentation_tests {
    use super::{PresentationMode, metal_presentation_modes};

    #[test]
    fn presentation_modes_match_metal_layer_controls() {
        let modes = metal_presentation_modes();
        assert!(modes.contains(PresentationMode::Fifo));
        assert!(modes.contains(PresentationMode::Immediate));
        assert!(!modes.contains(PresentationMode::Mailbox));
        assert!(!modes.contains(PresentationMode::Relaxed));
        assert!(!modes.contains(PresentationMode::Paced));
    }
}
