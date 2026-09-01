//! Vulkan backend for ez-gfx native resource creation and frame execution.
//!
//! The context owns every Vulkan object it creates; host window handles are borrowed.

use std::ffi::{CStr, CString};

use ash::{Entry, Instance, khr, vk};

use ez_gfx_core::{
    Backend,
    capability::{
        AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport,
        MAX_BINDLESS_SAMPLED_TEXTURES, SemanticProfile,
    },
};
use ez_gfx_hal::{
    AllocationError, AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BlendMode,
    BufferTransfer, CompletionToken, CullMode, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DynamicPipelineState, ExecutionBarrier, ExecutionPass, FrontFace, HalError, ImageMip,
    MemoryAllocator, MemoryClass, PrimitiveTopology, QueueKind, ResourceAccess, ResourceState,
    SamplerAddressMode, SamplerFilter, ShaderBufferLayout, ShaderStage, TextureSamplerDesc,
    validate_rgba8_mips,
};
use gpu_allocator::{
    AllocationSizes, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};

/// Identifies resources and contexts from this adapter as Vulkan-backed.
pub const BACKEND: Backend = Backend::Vulkan;
/// Descriptor set reserved for the bindless texture heap.
pub const TEXTURE_DESCRIPTOR_SET: u32 = 1;
/// Binding containing sampled images in the texture heap.
pub const TEXTURE_DESCRIPTOR_BINDING: u32 = 0;
/// Binding containing samplers paired with sampled images.
pub const SAMPLER_DESCRIPTOR_BINDING: u32 = 1;
/// Maximum number of sampled textures admitted by the Vulkan adapter.
pub const TEXTURE_DESCRIPTOR_CAPACITY: u32 = MAX_BINDLESS_SAMPLED_TEXTURES;

fn texture_descriptor_layout_bindings() -> [vk::DescriptorSetLayoutBinding<'static>; 2] {
    [
        vk::DescriptorSetLayoutBinding::default()
            .binding(TEXTURE_DESCRIPTOR_BINDING)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(TEXTURE_DESCRIPTOR_CAPACITY)
            .stage_flags(vk::ShaderStageFlags::ALL),
        vk::DescriptorSetLayoutBinding::default()
            .binding(SAMPLER_DESCRIPTOR_BINDING)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(TEXTURE_DESCRIPTOR_CAPACITY)
            .stage_flags(vk::ShaderStageFlags::ALL),
    ]
}

fn paired_texture_capacity(limits: &vk::PhysicalDeviceDescriptorIndexingProperties<'_>) -> u32 {
    limits
        .max_descriptor_set_update_after_bind_sampled_images
        .min(limits.max_descriptor_set_update_after_bind_samplers)
        .min(limits.max_per_stage_descriptor_update_after_bind_sampled_images)
        .min(limits.max_per_stage_descriptor_update_after_bind_samplers)
        .min(limits.max_per_stage_update_after_bind_resources / 2)
        .min(limits.max_update_after_bind_descriptors_in_all_pools / 2)
        .min(TEXTURE_DESCRIPTOR_CAPACITY)
}

fn texture_heap_rejection(
    features12: &vk::PhysicalDeviceVulkan12Features<'_>,
) -> Option<&'static str> {
    if features12.descriptor_binding_sampled_image_update_after_bind == 0 {
        Some("descriptor_binding_sampled_image_update_after_bind")
    } else if features12.descriptor_binding_partially_bound == 0 {
        Some("descriptor_binding_partially_bound")
    } else if features12.shader_sampled_image_array_non_uniform_indexing == 0 {
        Some("shader_sampled_image_array_non_uniform_indexing")
    } else {
        None
    }
}

fn sampler_create_info(desc: TextureSamplerDesc, mip_count: u32) -> vk::SamplerCreateInfo<'static> {
    let filter = |value| match value {
        SamplerFilter::Nearest => vk::Filter::NEAREST,
        SamplerFilter::Linear => vk::Filter::LINEAR,
    };
    let address = |value| match value {
        SamplerAddressMode::Clamp => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
    };

    vk::SamplerCreateInfo::default()
        .min_filter(filter(desc.min_filter))
        .mag_filter(filter(desc.mag_filter))
        .mipmap_mode(match desc.min_filter {
            SamplerFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
            SamplerFilter::Linear => vk::SamplerMipmapMode::LINEAR,
        })
        .address_mode_u(address(desc.address_u))
        .address_mode_v(address(desc.address_v))
        .address_mode_w(address(desc.address_w))
        .anisotropy_enable(desc.max_anisotropy > 1.0)
        .max_anisotropy(desc.max_anisotropy)
        .max_lod(f32::from(u16::try_from(mip_count).unwrap_or(u16::MAX)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Host window system accepted when creating a Vulkan context.
pub enum SurfacePlatform {
    /// A Win32 window and application instance.
    Win32,
}

/// Borrowed Vulkan presentation surface and optional captured frame.
pub struct NativeSurface {
    handle: vk::SurfaceKHR,
    presented_rgba8: Vec<u8>,
}

impl NativeSurface {
    /// Returns the owned Vulkan surface handle.
    pub const fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Empty before the first cached presentation; successful draws replace the complete RGBA8 frame.
    pub fn presented_rgba8(&self) -> &[u8] {
        &self.presented_rgba8
    }
}

/// Device or host-visible buffer allocation owned by the Vulkan context.
pub struct NativeAllocation {
    buffer: vk::Buffer,
    allocation: Allocation,
}
/// Buffer descriptor bound while recording one dispatch or draw.
pub struct NativeBufferBinding<'a> {
    /// Allocation supplying the descriptor.
    pub allocation: &'a NativeAllocation,
    /// First byte exposed through the descriptor.
    pub offset: u64,
    /// Number of bytes exposed through the descriptor.
    pub range: u64,
    /// Whether shaders may write through the descriptor.
    pub writable: bool,
}

/// Immutable inputs used to create a graphics pipeline.
pub struct NativeGraphicsPipelineDesc<'a> {
    /// Index of the vertex module in the owning shader.
    pub vertex_index: usize,
    /// Index of the fragment module in the owning shader.
    pub fragment_index: usize,
    /// Rasterization and blend state fixed by the pipeline.
    pub state: DynamicPipelineState,
    /// Reflected public buffer layout.
    pub layouts: &'a [ShaderBufferLayout],
    /// Whether the pipeline requires a depth attachment.
    pub depth_required: bool,
}

/// Fully resolved indexed draw consumed by frame recording.
pub struct NativeDrawIndexed<'a> {
    /// Render width in pixels.
    pub width: u32,
    /// Render height in pixels.
    pub height: u32,
    /// Graphics pipeline used by the draw.
    pub pipeline: &'a NativePipeline,
    /// Buffer containing 32-bit indices.
    pub index_buffer: &'a NativeAllocation,
    /// Buffer containing indexed indirect commands.
    pub indirect_buffer: &'a NativeAllocation,
    /// Number of indirect commands to execute.
    pub draw_count: u32,
    /// Raw push-constant payload, limited by frame validation.
    pub push_constants: &'a [u8],
    /// Reflected public buffer bindings.
    pub bindings: &'a [NativeBufferBinding<'a>],
}

/// Fully resolved compute dispatch consumed by frame recording.
pub struct NativeComputeDispatch<'a> {
    /// Compute pipeline used by the dispatch.
    pub pipeline: &'a NativePipeline,
    /// Workgroup count for each dispatch dimension.
    pub groups: [u32; 3],
    /// Raw push-constant payload.
    pub push_constants: &'a [u8],
    /// Reflected public buffer bindings.
    pub bindings: &'a [NativeBufferBinding<'a>],
}

/// Backend resource referenced by a compiled frame barrier.
pub enum NativeFrameResource<'a> {
    /// Buffer allocation.
    Buffer(&'a NativeAllocation),
    /// Sampled or storage texture.
    Texture(&'a NativeTexture),
    /// Current presentation image.
    Surface,
    /// Current depth attachment.
    Depth,
}

/// Validated backend action emitted by the frame-plan adapter.
pub enum NativeFrameAction<'a> {
    /// Wait for an external queue completion token.
    Wait(CompletionToken),
    /// Transition a resource between execution states.
    Barrier {
        /// Backend-neutral transition description.
        barrier: ExecutionBarrier,
        /// Resolved native resource to transition.
        resource: NativeFrameResource<'a>,
    },
    /// Begin the declared render pass.
    BeginPass(&'a ExecutionPass),
    /// Encode a compute dispatch.
    Compute(NativeComputeDispatch<'a>),
    /// Encode indexed indirect graphics work.
    Graphics(NativeDrawIndexed<'a>),
    /// Copy a texture into host-readable memory.
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
    /// Present the current surface image.
    Present,
}

/// Validated Vulkan shader modules owned as one artifact.
pub struct NativeShader {
    modules: Vec<vk::ShaderModule>,
}
/// Image, view, sampler, and allocation published as one texture.
pub struct NativeTexture {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Allocation,
    sampler: vk::Sampler,
    /// Slot in the bindless texture descriptor heap.
    pub binding: u32,
}

/// Vulkan pipeline and descriptor layout required to bind it.
pub struct NativePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    public_descriptor_layout: vk::DescriptorSetLayout,
    buffer_writable: Vec<bool>,
    buffer_bindings: Vec<u32>,
}

struct DepthTarget {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Allocation,
    extent: vk::Extent2D,
}

struct RetiredAllocation {
    allocation: NativeAllocation,
    completion: CompletionToken,
}

enum DeferredResource {
    Allocation(NativeAllocation),
    Pipeline(NativePipeline),
    Shader(NativeShader),
    Texture(NativeTexture),
}

struct DeferredNativeResource {
    pending_slots: u8,
    resource: DeferredResource,
}

const FRAMES_IN_FLIGHT: usize = 3;
const FRAME_DESCRIPTOR_SET_CAPACITY: u32 = 1024;

struct FrameSlot {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    in_flight: bool,
}

struct DeviceProbe {
    adapter: AdapterInfo,
    queue_family: u32,
    features12: vk::PhysicalDeviceVulkan12Features<'static>,
    features13: vk::PhysicalDeviceVulkan13Features<'static>,
    vertex_storage: bool,
    multi_draw: bool,
}

struct PendingDevice {
    device: Option<ash::Device>,
    allocator: Option<Allocator>,
    command_pool: Option<vk::CommandPool>,
    timeline: Option<vk::Semaphore>,
    descriptor_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    swapchain_loader: Option<khr::swapchain::Device>,
    image_available: Option<vk::Semaphore>,
}

impl PendingDevice {
    fn new(device: ash::Device) -> Self {
        Self {
            device: Some(device),
            allocator: None,
            command_pool: None,
            timeline: None,
            descriptor_layout: None,
            descriptor_pool: None,
            swapchain_loader: None,
            image_available: None,
        }
    }
}

impl Drop for PendingDevice {
    fn drop(&mut self) {
        let Some(device) = self.device.take() else {
            return;
        };
        // SAFETY: The taken semaphores, descriptor pools/layout, and command pool are rollback-only children of `device`; no commands were submitted from the pool, and `device` outlives each destroy call.
        unsafe {
            if let Some(semaphore) = self.image_available.take() {
                device.destroy_semaphore(semaphore, None);
            }
            if let Some(pool) = self.descriptor_pool.take() {
                device.destroy_descriptor_pool(pool, None);
            }
            if let Some(layout) = self.descriptor_layout.take() {
                device.destroy_descriptor_set_layout(layout, None);
            }
            if let Some(pool) = self.command_pool.take() {
                device.destroy_command_pool(pool, None);
            }
            if let Some(semaphore) = self.timeline.take() {
                device.destroy_semaphore(semaphore, None);
            }
        }
        self.swapchain_loader.take();
        drop(self.allocator.take());
        // SAFETY: `&mut self` excludes concurrent device use, and every device child retained by `PendingDevice` was destroyed or dropped above before `vkDestroyDevice(device)`.
        unsafe { device.destroy_device(None) };
    }
}

/// Vulkan instance, admitted device, queues, allocators, and frame state.
pub struct NativeContext {
    // Retain ash's dynamically loaded Vulkan library until all instance/device function pointers are dropped.
    _entry_loader: Entry,
    instance: Instance,
    surface_loader: khr::surface::Instance,
    physical_device: Option<vk::PhysicalDevice>,
    adapter_info: Option<AdapterInfo>,
    device: Option<ash::Device>,
    allocator: Option<Allocator>,
    retired: Vec<RetiredAllocation>,
    deferred: Vec<DeferredNativeResource>,
    graphics_queue: Option<vk::Queue>,
    graphics_queue_family: Option<u32>,
    transfer_timeline: Option<vk::Semaphore>,
    transfer_command_pool: Option<vk::CommandPool>,
    next_transfer_value: u64,
    texture_descriptor_pool: Option<vk::DescriptorPool>,
    texture_descriptor_layout: Option<vk::DescriptorSetLayout>,
    texture_descriptor_set: Option<vk::DescriptorSet>,
    sampler_anisotropy: bool,
    swapchain_loader: Option<khr::swapchain::Device>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_views: Vec<vk::ImageView>,
    swapchain_initialized: Vec<bool>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    frame_slots: Vec<FrameSlot>,
    frame_cursor: usize,
    image_available: Option<vk::Semaphore>,
    depth_target: Option<DepthTarget>,
}

mod device;
mod frame;
mod memory;
use memory::{
    create_frame_slots, map_allocation_hal, map_allocation_vk, map_allocator, map_allocator_hal,
    map_vk, vulkan_state,
};
mod pipeline;
mod surface;
mod texture;

#[cfg(test)]
mod tests;
