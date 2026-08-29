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
    AllocationError, AllocationRequest, BlendMode, BufferTransfer, CompletionToken, CullMode,
    DynamicPipelineState, FrontFace, HalError, ImageMip, MemoryAllocator, MemoryClass,
    PrimitiveTopology, QueueKind, SamplerAddressMode, SamplerFilter, ShaderBufferLayout,
    TextureSamplerDesc, validate_rgba8_mips,
};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};

pub const BACKEND: Backend = Backend::Vulkan;
pub const TEXTURE_DESCRIPTOR_SET: u32 = 1;
pub const TEXTURE_DESCRIPTOR_BINDING: u32 = 0;
pub const SAMPLER_DESCRIPTOR_BINDING: u32 = 1;
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
        .max_lod(mip_count as f32)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfacePlatform {
    Win32,
}

pub struct NativeSurface {
    handle: vk::SurfaceKHR,
    presented_rgba8: Vec<u8>,
}

impl NativeSurface {
    pub const fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Empty before the first cached presentation; successful draws replace the complete RGBA8 frame.
    pub fn presented_rgba8(&self) -> &[u8] {
        &self.presented_rgba8
    }
}

pub struct NativeAllocation {
    buffer: vk::Buffer,
    allocation: Allocation,
}
pub struct NativeBufferBinding<'a> {
    pub allocation: &'a NativeAllocation,
    pub offset: u64,
    pub range: u64,
    pub writable: bool,
}

pub struct NativeGraphicsPipelineDesc<'a> {
    pub vertex_index: usize,
    pub fragment_index: usize,
    pub state: DynamicPipelineState,
    pub layouts: &'a [ShaderBufferLayout],
    pub depth_required: bool,
}

pub struct NativeDrawIndexed<'a> {
    pub width: u32,
    pub height: u32,
    pub pipeline: &'a NativePipeline,
    pub index_buffer: &'a NativeAllocation,
    pub indirect_buffer: &'a NativeAllocation,
    pub draw_count: u32,
    pub push_constants: &'a [u8],
    pub bindings: &'a [NativeBufferBinding<'a>],
    pub capture_presented: bool,
}

pub struct NativeShader {
    modules: Vec<vk::ShaderModule>,
}
pub struct NativeTexture {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Allocation,
    sampler: vk::Sampler,
    pub binding: u32,
}

pub struct NativePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    bind_point: vk::PipelineBindPoint,
    public_descriptor_layout: vk::DescriptorSetLayout,
    buffer_writable: Vec<bool>,
    buffer_bindings: Vec<u32>,
    depth_required: bool,
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
        unsafe { device.destroy_device(None) };
    }
}

#[allow(dead_code)]
pub struct NativeContext {
    entry: Entry,
    instance: Instance,
    surface_loader: khr::surface::Instance,
    physical_device: Option<vk::PhysicalDevice>,
    adapter_info: Option<AdapterInfo>,
    device: Option<ash::Device>,
    allocator: Option<Allocator>,
    retired: Vec<RetiredAllocation>,
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
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    image_available: Option<vk::Semaphore>,
    depth_target: Option<DepthTarget>,
}

impl NativeContext {
    /// Validation is enabled only when requested and installed; a missing requested layer is unsupported.
    pub fn create(
        enable_debug: bool,
        enable_validation: bool,
        platform: SurfacePlatform,
    ) -> Result<Self, HalError> {
        // SAFETY: loading performs symbol lookup only and errors when the Vulkan loader is unavailable.
        let entry = unsafe { Entry::load() }.map_err(|_| HalError::Unsupported)?;
        let app_name = CString::new("ez_gfx_api").map_err(|_| HalError::NativeFailure)?;
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&app_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut extensions = vec![khr::surface::NAME.as_ptr()];
        match platform {
            SurfacePlatform::Win32 => extensions.push(khr::win32_surface::NAME.as_ptr()),
        }
        if enable_debug {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let validation =
            CString::new("VK_LAYER_KHRONOS_validation").map_err(|_| HalError::NativeFailure)?;
        let mut layers = Vec::new();
        if enable_validation {
            // SAFETY: the entry is live and the call only returns owned property data.
            let available =
                unsafe { entry.enumerate_instance_layer_properties() }.map_err(map_vk)?;
            let found = available.iter().any(|layer| {
                // SAFETY: Vulkan guarantees a NUL-terminated fixed-size layer name.
                (unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) }) == validation.as_c_str()
            });
            if !found {
                return Err(HalError::Unsupported);
            }
            layers.push(validation.as_ptr());
        }

        let create = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        // SAFETY: pointers in `create` remain live for the duration of the call and use Vulkan-owned names.
        let instance = unsafe { entry.create_instance(&create, None) }.map_err(map_vk)?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let mut context = Self {
            entry,
            instance,
            surface_loader,
            physical_device: None,
            device: None,
            adapter_info: None,
            allocator: None,
            retired: Vec::new(),
            graphics_queue: None,
            graphics_queue_family: None,
            transfer_timeline: None,
            transfer_command_pool: None,
            next_transfer_value: 1,
            texture_descriptor_pool: None,
            texture_descriptor_layout: None,
            texture_descriptor_set: None,
            sampler_anisotropy: false,
            swapchain_loader: None,
            swapchain: None,
            swapchain_views: Vec::new(),
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D::default(),
            image_available: None,
            depth_target: None,
        };
        context.init_device(None)?;
        Ok(context)
    }

    /// Win32 handles are borrowed; null handles are rejected before the native call.
    #[cfg(windows)]
    pub fn create_win32_surface(
        &self,
        window: *mut core::ffi::c_void,
        display: *mut core::ffi::c_void,
    ) -> Result<NativeSurface, HalError> {
        if window.is_null() || display.is_null() {
            return Err(HalError::InvalidArgument);
        }
        let loader = khr::win32_surface::Instance::new(&self.entry, &self.instance);
        let create = vk::Win32SurfaceCreateInfoKHR::default()
            .hwnd(window as isize)
            .hinstance(display as isize);
        // SAFETY: validated handles are borrowed from the host and Vulkan copies them during creation.
        let handle = unsafe { loader.create_win32_surface(&create, None) }.map_err(map_vk)?;
        Ok(NativeSurface {
            handle,
            presented_rgba8: Vec::new(),
        })
    }

    #[cfg(not(windows))]
    pub fn create_win32_surface(
        &self,
        _window: *mut core::ffi::c_void,
        _display: *mut core::ffi::c_void,
    ) -> Result<NativeSurface, HalError> {
        Err(HalError::Unsupported)
    }

    /// Device admission checks the semantic floor before creating queues or manager-visible state.
    pub fn init_device(
        &mut self,
        surface: Option<&NativeSurface>,
    ) -> Result<AdapterInfo, HalError> {
        if let (Some(physical), Some(queue_family), Some(adapter)) = (
            self.physical_device,
            self.graphics_queue_family,
            self.adapter_info.as_ref(),
        ) {
            if let Some(surface) = surface {
                let present = unsafe {
                    self.surface_loader.get_physical_device_surface_support(
                        physical,
                        queue_family,
                        surface.handle,
                    )
                }
                .map_err(map_vk)?;
                if !present {
                    return Err(HalError::Unsupported);
                }
            }
            return Ok(adapter.clone());
        }
        // SAFETY: the instance is live and owns returned physical-device handles.
        let devices = unsafe { self.instance.enumerate_physical_devices() }.map_err(map_vk)?;
        for physical in devices {
            if let Some(DeviceProbe {
                adapter,
                queue_family,
                features12,
                features13,
                vertex_storage,
                multi_draw,
            }) = self.probe_device(physical, surface)?
            {
                if !vertex_storage || !multi_draw {
                    continue;
                }
                if SemanticProfile::V1.admit(adapter.capabilities()).is_err() {
                    continue;
                }
                let priority = [1.0_f32];
                let queue = vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family)
                    .queue_priorities(&priority);
                let mut enabled12 = vk::PhysicalDeviceVulkan12Features::default()
                    .timeline_semaphore(features12.timeline_semaphore != 0)
                    .buffer_device_address(features12.buffer_device_address != 0)
                    .descriptor_indexing(features12.descriptor_indexing != 0)
                    .runtime_descriptor_array(features12.runtime_descriptor_array != 0)
                    .descriptor_binding_partially_bound(
                        features12.descriptor_binding_partially_bound != 0,
                    )
                    .descriptor_binding_sampled_image_update_after_bind(
                        features12.descriptor_binding_sampled_image_update_after_bind != 0,
                    )
                    .descriptor_binding_storage_buffer_update_after_bind(
                        features12.descriptor_binding_storage_buffer_update_after_bind != 0,
                    )
                    .descriptor_binding_storage_image_update_after_bind(
                        features12.descriptor_binding_storage_image_update_after_bind != 0,
                    )
                    .shader_sampled_image_array_non_uniform_indexing(
                        features12.shader_sampled_image_array_non_uniform_indexing != 0,
                    );
                let mut enabled13 = vk::PhysicalDeviceVulkan13Features::default()
                    .dynamic_rendering(features13.dynamic_rendering != 0)
                    .synchronization2(features13.synchronization2 != 0);
                let mut enabled11 =
                    vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
                let core_features = unsafe { self.instance.get_physical_device_features(physical) };
                let enabled_core = vk::PhysicalDeviceFeatures::default()
                    .vertex_pipeline_stores_and_atomics(vertex_storage)
                    .multi_draw_indirect(multi_draw)
                    .sampler_anisotropy(core_features.sampler_anisotropy != 0);
                let swapchain_extensions = [khr::swapchain::NAME.as_ptr()];
                let enabled_extensions = swapchain_extensions.as_slice();
                let create = vk::DeviceCreateInfo::default()
                    .enabled_features(&enabled_core)
                    .enabled_extension_names(enabled_extensions)
                    .queue_create_infos(core::slice::from_ref(&queue))
                    .push_next(&mut enabled11)
                    .push_next(&mut enabled12)
                    .push_next(&mut enabled13);
                // SAFETY: the physical device and queue family were queried from this live instance.
                let device = unsafe { self.instance.create_device(physical, &create, None) }
                    .map_err(map_vk)?;
                let mut pending = PendingDevice::new(device);
                let device = pending
                    .device
                    .as_ref()
                    .expect("pending device is initialized");
                // SAFETY: queue zero was requested from `queue_family` above.
                let graphics_queue = unsafe { device.get_device_queue(queue_family, 0) };
                pending.allocator = Some(
                    Allocator::new(&AllocatorCreateDesc {
                        instance: self.instance.clone(),
                        device: device.clone(),
                        physical_device: physical,
                        debug_settings: Default::default(),
                        buffer_device_address: true,
                        allocation_sizes: Default::default(),
                    })
                    .map_err(map_allocator_hal)?,
                );
                pending.command_pool = Some(
                    unsafe {
                        device.create_command_pool(
                            &vk::CommandPoolCreateInfo::default()
                                .queue_family_index(queue_family)
                                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                            None,
                        )
                    }
                    .map_err(map_vk)?,
                );
                let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(0);
                pending.timeline = Some(
                    unsafe {
                        device.create_semaphore(
                            &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                            None,
                        )
                    }
                    .map_err(map_vk)?,
                );
                let bindings = texture_descriptor_layout_bindings();
                let binding_flags = [
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                ];
                let mut binding_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(&binding_flags);
                pending.descriptor_layout = Some(
                    unsafe {
                        device.create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::default()
                                .bindings(&bindings)
                                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                                .push_next(&mut binding_info),
                            None,
                        )
                    }
                    .map_err(map_vk)?,
                );
                let descriptor_layout = pending
                    .descriptor_layout
                    .expect("descriptor layout was created");
                pending.descriptor_pool = Some(
                    unsafe {
                        device.create_descriptor_pool(
                            &vk::DescriptorPoolCreateInfo::default()
                                .max_sets(1)
                                .pool_sizes(&[
                                    vk::DescriptorPoolSize {
                                        ty: vk::DescriptorType::SAMPLED_IMAGE,
                                        descriptor_count: TEXTURE_DESCRIPTOR_CAPACITY,
                                    },
                                    vk::DescriptorPoolSize {
                                        ty: vk::DescriptorType::SAMPLER,
                                        descriptor_count: TEXTURE_DESCRIPTOR_CAPACITY,
                                    },
                                ])
                                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                            None,
                        )
                    }
                    .map_err(map_vk)?,
                );
                let descriptor_pool = pending
                    .descriptor_pool
                    .expect("descriptor pool was created");
                let descriptor_set = unsafe {
                    device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(core::slice::from_ref(&descriptor_layout)),
                    )
                }
                .map_err(map_vk)?[0];
                pending.swapchain_loader =
                    Some(khr::swapchain::Device::new(&self.instance, device));
                pending.image_available = Some(
                    unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                        .map_err(map_vk)?,
                );
                self.physical_device = Some(physical);
                self.adapter_info = Some(adapter.clone());
                self.graphics_queue_family = Some(queue_family);
                self.graphics_queue = Some(graphics_queue);
                self.allocator = pending.allocator.take();
                self.sampler_anisotropy = core_features.sampler_anisotropy != 0;
                self.device = pending.device.take();
                self.transfer_timeline = pending.timeline.take();
                self.transfer_command_pool = pending.command_pool.take();
                self.texture_descriptor_pool = pending.descriptor_pool.take();
                self.texture_descriptor_layout = pending.descriptor_layout.take();
                self.texture_descriptor_set = Some(descriptor_set);
                self.swapchain_loader = pending.swapchain_loader.take();
                self.image_available = pending.image_available.take();
                return Ok(adapter);
            }
        }
        Err(HalError::Unsupported)
    }
    pub fn wait_idle(&self) -> Result<(), HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        unsafe { device.device_wait_idle() }.map_err(map_vk)
    }

    pub fn destroy_surface(&mut self, surface: NativeSurface) {
        let _ = self.wait_idle();
        let _ = self.destroy_depth_target();
        if let Some(device) = self.device.as_ref() {
            for view in self.swapchain_views.drain(..) {
                unsafe { device.destroy_image_view(view, None) };
            }
        }
        if let (Some(loader), Some(swapchain)) =
            (self.swapchain_loader.as_ref(), self.swapchain.take())
        {
            unsafe { loader.destroy_swapchain(swapchain, None) };
        }
        self.swapchain_format = vk::Format::UNDEFINED;
        self.swapchain_extent = vk::Extent2D::default();
        // SAFETY: the host window is still live and no swapchain references this surface.
        unsafe { self.surface_loader.destroy_surface(surface.handle, None) };
    }

    /// Acquires and presents a FIFO swapchain image; minimized and out-of-date states are explicit.
    pub fn acquire_present(
        &mut self,
        surface: &NativeSurface,
        width: u32,
        height: u32,
    ) -> Result<(), HalError> {
        if width == 0 || height == 0 {
            return Err(HalError::NotReady);
        }
        if self.swapchain.is_none()
            || self.swapchain_extent.width != width
            || self.swapchain_extent.height != height
        {
            self.recreate_swapchain(surface, width, height)?;
        }
        let loader = self.swapchain_loader.as_ref().ok_or(HalError::NotReady)?;
        let semaphore = self.image_available.ok_or(HalError::NotReady)?;
        let swapchain = self.swapchain.ok_or(HalError::NotReady)?;
        let (image, suboptimal) = match unsafe {
            loader.acquire_next_image(swapchain, u64::MAX, semaphore, vk::Fence::null())
        } {
            Ok(value) => value,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(surface, width, height)?;
                let loader = self.swapchain_loader.as_ref().ok_or(HalError::NotReady)?;
                unsafe {
                    loader.acquire_next_image(
                        self.swapchain.ok_or(HalError::NotReady)?,
                        u64::MAX,
                        semaphore,
                        vk::Fence::null(),
                    )
                }
                .map_err(map_vk)?
            }
            Err(error) => return Err(map_vk(error)),
        };
        let swapchain = self.swapchain.ok_or(HalError::NotReady)?;
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(core::slice::from_ref(&semaphore))
            .swapchains(core::slice::from_ref(&swapchain))
            .image_indices(core::slice::from_ref(&image));
        match unsafe {
            self.swapchain_loader
                .as_ref()
                .ok_or(HalError::NotReady)?
                .queue_present(self.graphics_queue.ok_or(HalError::NotReady)?, &present)
        } {
            Ok(present_suboptimal) => {
                if suboptimal || present_suboptimal {
                    self.recreate_swapchain(surface, width, height)?;
                }
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(surface, width, height)
            }
            Err(error) => Err(map_vk(error)),
        }
    }

    fn recreate_swapchain(
        &mut self,
        surface: &NativeSurface,
        requested_width: u32,
        requested_height: u32,
    ) -> Result<(), HalError> {
        self.wait_idle()?;
        self.destroy_depth_target().map_err(map_allocation_hal)?;
        let physical = self.physical_device.ok_or(HalError::NotReady)?;
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(physical, surface.handle)
        }
        .map_err(map_vk)?;
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(physical, surface.handle)
        }
        .map_err(map_vk)?;
        let chosen = formats
            .iter()
            .copied()
            .find(|format| format.format == vk::Format::B8G8R8A8_SRGB)
            .or_else(|| formats.first().copied())
            .ok_or(HalError::Unsupported)?;
        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: requested_width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: requested_height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Err(HalError::NotReady);
        }
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        if !capabilities.supported_usage_flags.contains(usage) {
            return Err(HalError::Unsupported);
        }
        let alpha_bits = capabilities.supported_composite_alpha.as_raw();
        if alpha_bits == 0 {
            return Err(HalError::Unsupported);
        }
        let composite_alpha =
            vk::CompositeAlphaFlagsKHR::from_raw(alpha_bits & alpha_bits.wrapping_neg());
        for view in self.swapchain_views.drain(..) {
            unsafe { device.destroy_image_view(view, None) };
        }
        let old = self.swapchain.unwrap_or(vk::SwapchainKHR::null());
        let count = capabilities.min_image_count.saturating_add(1).min(
            if capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                capabilities.max_image_count
            },
        );
        let create = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(count)
            .image_format(chosen.format)
            .image_color_space(chosen.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old);
        let loader = self.swapchain_loader.as_ref().ok_or(HalError::NotReady)?;
        let swapchain = unsafe { loader.create_swapchain(&create, None) }.map_err(map_vk)?;
        let images = unsafe { loader.get_swapchain_images(swapchain) }.map_err(map_vk)?;
        let mut views = Vec::with_capacity(images.len());
        for image in images {
            match unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(chosen.format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
            } {
                Ok(view) => views.push(view),
                Err(error) => {
                    for view in views {
                        unsafe { device.destroy_image_view(view, None) };
                    }
                    unsafe { loader.destroy_swapchain(swapchain, None) };
                    return Err(map_vk(error));
                }
            }
        }
        if old != vk::SwapchainKHR::null() {
            unsafe { loader.destroy_swapchain(old, None) };
        }
        self.swapchain = Some(swapchain);
        self.swapchain_views = views;
        self.swapchain_format = chosen.format;
        self.swapchain_extent = extent;
        Ok(())
    }

    fn ensure_depth_target(&mut self, extent: vk::Extent2D) -> Result<(), HalError> {
        if self
            .depth_target
            .as_ref()
            .is_some_and(|target| target.extent == extent)
        {
            return Ok(());
        }
        self.wait_idle()?;
        self.destroy_depth_target().map_err(map_allocation_hal)?;
        let device = self.device.as_ref().ok_or(HalError::NotReady)?.clone();
        let create = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&create, None) }.map_err(map_vk)?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = match self.allocator.as_mut().ok_or(HalError::NotReady)?.allocate(
            &AllocationCreateDesc {
                name: "ez-gfx-depth",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            },
        ) {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(map_allocation_hal(map_allocator(error)));
            }
        };
        if let Err(error) =
            unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }
        {
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
            unsafe { device.destroy_image(image, None) };
            return Err(map_vk(error));
        }
        let view = match unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        } {
            Ok(view) => view,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator remains initialized")
                    .free(allocation);
                return Err(map_vk(error));
            }
        };
        self.depth_target = Some(DepthTarget {
            image,
            view,
            allocation,
            extent,
        });
        Ok(())
    }

    fn destroy_depth_target(&mut self) -> Result<(), AllocationError> {
        let Some(target) = self.depth_target.take() else {
            return Ok(());
        };
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        unsafe {
            device.destroy_image_view(target.view, None);
            device.destroy_image(target.image, None);
        }
        self.allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .free(target.allocation)
            .map_err(map_allocator)
    }
    /// Creates validated SPIR-V shader modules; each product must be nonempty and word-aligned.
    pub fn create_shader(&self, products: &[&[u8]]) -> Result<NativeShader, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let mut modules = Vec::with_capacity(products.len());
        for product in products {
            if product.is_empty() || product.len() % 4 != 0 {
                self.destroy_shader(NativeShader { modules });
                return Err(HalError::InvalidArgument);
            }
            let words = product
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            match unsafe {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            } {
                Ok(module) => modules.push(module),
                Err(error) => {
                    self.destroy_shader(NativeShader { modules });
                    return Err(map_vk(error));
                }
            }
        }
        if modules.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        Ok(NativeShader { modules })
    }

    pub fn destroy_shader(&self, shader: NativeShader) {
        if let Some(device) = self.device.as_ref() {
            for module in shader.modules {
                unsafe { device.destroy_shader_module(module, None) };
            }
        }
    }

    /// Reflected public buffers occupy descriptor set zero; the bindless texture table remains set one.
    fn create_pipeline_layout(
        &self,
        layouts: &[ShaderBufferLayout],
    ) -> Result<
        (
            vk::PipelineLayout,
            vk::DescriptorSetLayout,
            Vec<bool>,
            Vec<u32>,
        ),
        HalError,
    > {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let texture = self.texture_descriptor_layout.ok_or(HalError::NotReady)?;
        let mut bindings = Vec::new();
        let mut writable = Vec::new();
        let mut physical_bindings = Vec::new();
        for layout in layouts {
            if layout.space != 0 || layout.descriptor_count == 0 || layout.descriptor_count > 2 {
                return Err(HalError::Unsupported);
            }
            for offset in 0..layout.descriptor_count {
                let binding = layout
                    .binding
                    .checked_add(offset)
                    .ok_or(HalError::InvalidArgument)?;
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::ALL),
                );
                writable.push(layout.writable);
                physical_bindings.push(binding);
            }
        }
        let public = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(map_vk)?;
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(128);
        let set_layouts = [public, texture];
        match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(core::slice::from_ref(&range)),
                None,
            )
        } {
            Ok(layout) => Ok((layout, public, writable, physical_bindings)),
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(public, None) };
                Err(map_vk(error))
            }
        }
    }

    /// Entry names reject interior NUL before reaching Vulkan.
    pub fn create_compute_pipeline(
        &self,
        shader: &NativeShader,
        module_index: usize,
        _entry: &str,
        layouts: &[ShaderBufferLayout],
    ) -> Result<NativePipeline, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let module = *shader
            .modules
            .get(module_index)
            .ok_or(HalError::InvalidArgument)?;
        let entry = CString::new("main").unwrap();
        let (layout, public_descriptor_layout, buffer_writable, buffer_bindings) =
            self.create_pipeline_layout(layouts)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        match unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
        } {
            Ok(pipelines) => Ok(NativePipeline {
                pipeline: pipelines[0],
                layout,
                bind_point: vk::PipelineBindPoint::COMPUTE,
                public_descriptor_layout,
                buffer_writable,
                buffer_bindings,
                depth_required: false,
            }),
            Err((_, error)) => {
                unsafe {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(public_descriptor_layout, None)
                };
                Err(map_vk(error))
            }
        }
    }

    pub fn destroy_pipeline(&self, pipeline: NativePipeline) {
        if let Some(device) = self.device.as_ref() {
            unsafe {
                device.destroy_pipeline(pipeline.pipeline, None);
                device.destroy_pipeline_layout(pipeline.layout, None);
                device.destroy_descriptor_set_layout(pipeline.public_descriptor_layout, None)
            };
        }
    }
    /// Creates a dynamic-rendering graphics pipeline from an exact vertex/fragment artifact pair.
    pub fn create_graphics_pipeline(
        &self,
        shader: &NativeShader,
        desc: NativeGraphicsPipelineDesc<'_>,
    ) -> Result<NativePipeline, HalError> {
        let NativeGraphicsPipelineDesc {
            vertex_index,
            fragment_index,
            state,
            layouts,
            depth_required,
        } = desc;
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        if self.swapchain_format == vk::Format::UNDEFINED {
            return Err(HalError::NotReady);
        }
        let vertex = *shader
            .modules
            .get(vertex_index)
            .ok_or(HalError::InvalidArgument)?;
        let fragment = *shader
            .modules
            .get(fragment_index)
            .ok_or(HalError::InvalidArgument)?;
        let entry = CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(&entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(&entry),
        ];
        let topology = match state.topology {
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
        };
        let cull = match state.cull {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::Back => vk::CullModeFlags::BACK,
        };
        let front = match state.front_face {
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        };
        let blend = match state.blend {
            BlendMode::None => vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            BlendMode::Alpha => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
        };
        let (layout, public_descriptor_layout, buffer_writable, buffer_bindings) =
            self.create_pipeline_layout(layouts)?;
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(cull)
            .front_face(front)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(core::slice::from_ref(&blend));
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_required)
            .depth_write_enable(depth_required)
            .depth_compare_op(vk::CompareOp::LESS);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(core::slice::from_ref(&self.swapchain_format))
            .depth_attachment_format(if depth_required {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::UNDEFINED
            });
        let create = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        match unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[create], None)
        } {
            Ok(pipelines) => Ok(NativePipeline {
                pipeline: pipelines[0],
                layout,
                bind_point: vk::PipelineBindPoint::GRAPHICS,
                public_descriptor_layout,
                buffer_writable,
                buffer_bindings,
                depth_required,
            }),
            Err((_, error)) => {
                unsafe {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(public_descriptor_layout, None)
                };
                Err(map_vk(error))
            }
        }
    }

    fn create_public_descriptor_set(
        &self,
        pipeline: &NativePipeline,
        bindings: &[NativeBufferBinding<'_>],
    ) -> Result<(vk::DescriptorPool, vk::DescriptorSet), HalError> {
        if bindings.len() != pipeline.buffer_writable.len() {
            return Err(HalError::InvalidArgument);
        }
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(bindings.len().max(1) as u32);
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(core::slice::from_ref(&pool_size)),
                None,
            )
        }
        .map_err(map_vk)?;
        let set = match unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(core::slice::from_ref(&pipeline.public_descriptor_layout)),
            )
        } {
            Ok(sets) => sets[0],
            Err(error) => {
                unsafe { device.destroy_descriptor_pool(pool, None) };
                return Err(map_vk(error));
            }
        };
        let mut infos = Vec::with_capacity(bindings.len());
        for (binding, writable) in bindings.iter().zip(&pipeline.buffer_writable) {
            if binding.writable != *writable
                || binding.range == 0
                || binding
                    .offset
                    .checked_add(binding.range)
                    .is_none_or(|end| end > binding.allocation.allocation.size())
            {
                unsafe { device.destroy_descriptor_pool(pool, None) };
                return Err(HalError::InvalidArgument);
            }
            infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(binding.allocation.buffer)
                    .offset(binding.offset)
                    .range(binding.range),
            );
        }
        let writes = infos
            .iter()
            .enumerate()
            .map(|(index, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(pipeline.buffer_bindings[index])
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(core::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok((pool, set))
    }

    /// Ensures target format and extent exist before a graphics pipeline is created.
    pub fn prepare_surface(
        &mut self,
        surface: &NativeSurface,
        width: u32,
        height: u32,
    ) -> Result<(), HalError> {
        if width == 0 || height == 0 {
            return Err(HalError::NotReady);
        }
        if self.swapchain.is_none()
            || self.swapchain_extent.width != width
            || self.swapchain_extent.height != height
        {
            self.recreate_swapchain(surface, width, height)?;
        }
        Ok(())
    }

    /// Executes indexed indirect draws, optionally captures the rendered image, then presents it.
    pub fn draw_indexed_present(
        &mut self,
        surface: &mut NativeSurface,
        request: NativeDrawIndexed<'_>,
    ) -> Result<(), HalError> {
        let NativeDrawIndexed {
            width,
            height,
            pipeline,
            index_buffer,
            indirect_buffer,
            draw_count,
            push_constants,
            bindings,
            capture_presented,
        } = request;
        if pipeline.bind_point != vk::PipelineBindPoint::GRAPHICS
            || width == 0
            || height == 0
            || draw_count == 0
            || push_constants.len() > 128
        {
            return Err(HalError::InvalidArgument);
        }
        if self.swapchain.is_none()
            || self.swapchain_extent.width != width
            || self.swapchain_extent.height != height
        {
            self.recreate_swapchain(surface, width, height)?;
        }
        if pipeline.depth_required {
            self.ensure_depth_target(self.swapchain_extent)?;
        }
        let device = self.device.as_ref().ok_or(HalError::NotReady)?.clone();
        let loader = self
            .swapchain_loader
            .as_ref()
            .ok_or(HalError::NotReady)?
            .clone();
        let swapchain = self.swapchain.ok_or(HalError::NotReady)?;
        let available = self.image_available.ok_or(HalError::NotReady)?;
        let pool = self.transfer_command_pool.ok_or(HalError::NotReady)?;
        let queue = self.graphics_queue.ok_or(HalError::NotReady)?;
        let texture_set = self.texture_descriptor_set.ok_or(HalError::NotReady)?;
        let capture = if capture_presented {
            let size = u64::from(width)
                .checked_mul(u64::from(height))
                .and_then(|value| value.checked_mul(4))
                .ok_or(HalError::InvalidArgument)?;
            let allocation = self
                .allocate(
                    AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                        .map_err(|_| HalError::InvalidArgument)?,
                )
                .map_err(map_allocation_hal)?;
            Some((size, allocation))
        } else {
            None
        };
        let (public_pool, public_set) = match self.create_public_descriptor_set(pipeline, bindings)
        {
            Ok(public) => public,
            Err(error) => {
                if let Some((_, capture)) = capture {
                    let _ = self.free(capture);
                }
                return Err(error);
            }
        };
        let mut finished = None;
        let mut command = None;
        let mut submitted = false;
        let rendered = (|| {
            let (image_index, _) = unsafe {
                loader.acquire_next_image(swapchain, u64::MAX, available, vk::Fence::null())
            }
            .map_err(map_vk)?;
            let images = unsafe { loader.get_swapchain_images(swapchain) }.map_err(map_vk)?;
            let image = *images
                .get(image_index as usize)
                .ok_or(HalError::NativeFailure)?;
            let finished_handle =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .map_err(map_vk)?;
            finished = Some(finished_handle);
            let command_handle = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(map_vk)?[0];
            command = Some(command_handle);
            let finished = finished_handle;
            let command = command_handle;
            unsafe {
                device
                    .begin_command_buffer(
                        command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(map_vk)?;
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let to_color = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_color],
                );
                if let Some(depth) = self
                    .depth_target
                    .as_ref()
                    .filter(|_| pipeline.depth_required)
                {
                    let depth_range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let to_depth = vk::ImageMemoryBarrier::default()
                        .image(depth.image)
                        .subresource_range(depth_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        );
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_depth],
                    );
                }
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(self.swapchain_views[image_index as usize])
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.1, 0.1, 1.0],
                        },
                    });
                let depth_attachment = self
                    .depth_target
                    .as_ref()
                    .filter(|_| pipeline.depth_required)
                    .map(|depth| {
                        vk::RenderingAttachmentInfo::default()
                            .image_view(depth.view)
                            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 1.0,
                                    stencil: 0,
                                },
                            })
                    });
                let mut rendering = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: self.swapchain_extent,
                    })
                    .layer_count(1)
                    .color_attachments(core::slice::from_ref(&attachment));
                if let Some(depth) = depth_attachment.as_ref() {
                    rendering = rendering.depth_attachment(depth);
                }
                device.cmd_begin_rendering(command, &rendering);
                device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline,
                );
                let sets = [public_set, texture_set];
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout,
                    0,
                    &sets,
                    &[],
                );
                device.cmd_set_viewport(
                    command,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: width as f32,
                        height: height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                device.cmd_set_scissor(
                    command,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: self.swapchain_extent,
                    }],
                );
                device.cmd_bind_index_buffer(
                    command,
                    index_buffer.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                if !push_constants.is_empty() {
                    device.cmd_push_constants(
                        command,
                        pipeline.layout,
                        vk::ShaderStageFlags::ALL,
                        0,
                        push_constants,
                    );
                }
                device.cmd_draw_indexed_indirect(
                    command,
                    indirect_buffer.buffer,
                    0,
                    draw_count,
                    20,
                );
                device.cmd_end_rendering(command);
                if let Some((_, capture)) = &capture {
                    let to_copy = vk::ImageMemoryBarrier::default()
                        .image(image)
                        .subresource_range(range)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_copy],
                    );
                    let copy = vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        });
                    device.cmd_copy_image_to_buffer(
                        command,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        capture.buffer,
                        core::slice::from_ref(&copy),
                    );
                    let to_present = vk::ImageMemoryBarrier::default()
                        .image(image)
                        .subresource_range(range)
                        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ);
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_present],
                    );
                } else {
                    let to_present = vk::ImageMemoryBarrier::default()
                        .image(image)
                        .subresource_range(range)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_present],
                    );
                }
                device.end_command_buffer(command).map_err(map_vk)?;
                let wait_stage = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let submit = vk::SubmitInfo::default()
                    .wait_semaphores(core::slice::from_ref(&available))
                    .wait_dst_stage_mask(&wait_stage)
                    .command_buffers(core::slice::from_ref(&command))
                    .signal_semaphores(core::slice::from_ref(&finished));
                let queue = self.graphics_queue.ok_or(HalError::NotReady)?;
                device
                    .queue_submit(queue, &[submit], vk::Fence::null())
                    .map_err(map_vk)?;
                submitted = true;
                let present = vk::PresentInfoKHR::default()
                    .wait_semaphores(core::slice::from_ref(&finished))
                    .swapchains(core::slice::from_ref(&swapchain))
                    .image_indices(core::slice::from_ref(&image_index));
                loader.queue_present(queue, &present).map_err(map_vk)?;
                device.queue_wait_idle(queue).map_err(map_vk)?;
            }
            Ok::<_, HalError>(())
        })();
        if submitted && rendered.is_err() {
            let _ = unsafe { device.queue_wait_idle(queue) };
        }
        unsafe {
            if let Some(command) = command {
                device.free_command_buffers(pool, &[command]);
            }
            if let Some(finished) = finished {
                device.destroy_semaphore(finished, None);
            }
            device.destroy_descriptor_pool(public_pool, None);
        }
        if let Err(error) = rendered {
            if let Some((_, capture)) = capture {
                let _ = self.free(capture);
            }
            return Err(error);
        }
        let Some((capture_size, mut capture)) = capture else {
            return Ok(());
        };
        let captured = (|| {
            self.invalidate(&mut capture, 0, capture_size)
                .map_err(map_allocation_hal)?;
            surface.presented_rgba8 = self.mapped_slice(&capture).map_err(map_allocation_hal)?
                [..capture_size as usize]
                .to_vec();
            if matches!(
                self.swapchain_format,
                vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
            ) {
                for pixel in surface.presented_rgba8.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
            }
            Ok(())
        })();
        let freed = self.free(capture).map_err(map_allocation_hal);
        match (captured, freed) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    /// Dispatches a validated compute grid and waits before the transient pipeline may be destroyed.
    pub fn dispatch_compute(
        &self,
        pipeline: &NativePipeline,
        groups: [u32; 3],
        push_constants: &[u8],
        bindings: &[NativeBufferBinding<'_>],
    ) -> Result<(), HalError> {
        if pipeline.bind_point != vk::PipelineBindPoint::COMPUTE
            || groups.contains(&0)
            || push_constants.len() > 128
        {
            return Err(HalError::InvalidArgument);
        }
        let (public_pool, public_set) = self.create_public_descriptor_set(pipeline, bindings)?;
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let pool = self.transfer_command_pool.ok_or(HalError::NotReady)?;
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(map_vk)?[0];
        unsafe {
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(map_vk)?;
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            let sets = [
                public_set,
                self.texture_descriptor_set.ok_or(HalError::NotReady)?,
            ];
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &sets,
                &[],
            );
            if !push_constants.is_empty() {
                device.cmd_push_constants(
                    command,
                    pipeline.layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    push_constants,
                );
            }
            device.cmd_dispatch(command, groups[0], groups[1], groups[2]);
            device.end_command_buffer(command).map_err(map_vk)?;
            device
                .queue_submit(
                    self.graphics_queue.ok_or(HalError::NotReady)?,
                    &[vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command))],
                    vk::Fence::null(),
                )
                .map_err(map_vk)?;
            device
                .queue_wait_idle(self.graphics_queue.ok_or(HalError::NotReady)?)
                .map_err(map_vk)?;
            device.free_command_buffers(pool, &[command]);
        }
        unsafe { device.destroy_descriptor_pool(public_pool, None) };
        Ok(())
    }

    fn destroy_unpublished_texture(
        &mut self,
        device: &ash::Device,
        image: vk::Image,
        view: Option<vk::ImageView>,
        sampler: Option<vk::Sampler>,
        allocation: Allocation,
    ) {
        unsafe {
            if let Some(sampler) = sampler {
                device.destroy_sampler(sampler, None);
            }
            if let Some(view) = view {
                device.destroy_image_view(view, None);
            }
            device.destroy_image(image, None);
        }
        if let Some(allocator) = self.allocator.as_mut() {
            let _ = allocator.free(allocation);
        }
    }

    /// Creates an RGBA8 mip chain and submits each level under a distinct timeline value.
    pub fn create_texture_rgba8(
        &mut self,
        mips: &[ImageMip<'_>],
        binding: u32,
        sampler_desc: TextureSamplerDesc,
    ) -> Result<(NativeTexture, Vec<CompletionToken>), AllocationError> {
        validate_rgba8_mips(mips).map_err(|_| AllocationError::ZeroSize)?;
        if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
            return Err(AllocationError::ZeroSize);
        }
        if sampler_desc.max_anisotropy > 1.0 && !self.sampler_anisotropy {
            return Err(AllocationError::NativeFailure);
        }
        let descriptor_set = self
            .texture_descriptor_set
            .ok_or(AllocationError::NativeFailure)?;
        let pool = self
            .transfer_command_pool
            .ok_or(AllocationError::NativeFailure)?;
        let queue = self.graphics_queue.ok_or(AllocationError::NativeFailure)?;
        let semaphore = self
            .transfer_timeline
            .ok_or(AllocationError::NativeFailure)?;
        let width = mips[0].width;
        let height = mips[0].height;
        let mip_count = u32::try_from(mips.len()).map_err(|_| AllocationError::NativeFailure)?;
        let total = mips
            .iter()
            .try_fold(0_u64, |sum, mip| sum.checked_add(mip.bytes.len() as u64))
            .ok_or(AllocationError::NativeFailure)?;
        let upload_request = AllocationRequest::new(total, 4, MemoryClass::Upload, true, None)
            .map_err(|_| AllocationError::ZeroSize)?;
        if self.allocator.is_none() {
            return Err(AllocationError::NativeFailure);
        }
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        let create = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&create, None) }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = match self
            .allocator
            .as_mut()
            .expect("allocator checked before image creation")
            .allocate(&AllocationCreateDesc {
                name: "ez-gfx-texture",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(map_allocator(error));
            }
        };
        if let Err(error) =
            unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }
        {
            unsafe { device.destroy_image(image, None) };
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator initialized")
                .free(allocation);
            return Err(map_allocation_vk(map_vk(error)));
        }
        let view = match unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: mip_count,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        } {
            Ok(view) => view,
            Err(error) => {
                self.destroy_unpublished_texture(&device, image, None, None, allocation);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let sampler_info = sampler_create_info(sampler_desc, mip_count);
        let sampler = match unsafe { device.create_sampler(&sampler_info, None) } {
            Ok(sampler) => sampler,
            Err(error) => {
                self.destroy_unpublished_texture(&device, image, Some(view), None, allocation);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let mut upload = match self.allocate(upload_request) {
            Ok(upload) => upload,
            Err(error) => {
                self.destroy_unpublished_texture(
                    &device,
                    image,
                    Some(view),
                    Some(sampler),
                    allocation,
                );
                return Err(error);
            }
        };
        let mut commands = Vec::with_capacity(mips.len());
        let submitted = (|| {
            let target = self.mapped_slice_mut(&mut upload)?;
            let mut offset = 0_usize;
            for mip in mips {
                target[offset..offset + mip.bytes.len()].copy_from_slice(mip.bytes);
                offset += mip.bytes.len();
            }
            self.flush(&mut upload, 0, total)?;

            let mut completions = Vec::with_capacity(mips.len());
            let mut source_offset = 0_u64;
            for (level, mip) in mips.iter().enumerate() {
                let command = unsafe {
                    device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                }
                .map_err(|error| map_allocation_vk(map_vk(error)))?[0];
                commands.push(command);
                unsafe {
                    device.begin_command_buffer(
                        command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: level as u32,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let to_copy = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                unsafe {
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        core::slice::from_ref(&to_copy),
                    )
                };
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(source_offset)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level as u32,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: mip.width,
                        height: mip.height,
                        depth: 1,
                    });
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command,
                        upload.buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        core::slice::from_ref(&region),
                    )
                };
                let to_shader = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                unsafe {
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_GRAPHICS
                            | vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        core::slice::from_ref(&to_shader),
                    );
                    device.end_command_buffer(command)
                }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
                let value = self.next_transfer_value;
                self.next_transfer_value =
                    value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
                let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                    .signal_semaphore_values(core::slice::from_ref(&value));
                let submit = vk::SubmitInfo::default()
                    .command_buffers(core::slice::from_ref(&command))
                    .signal_semaphores(core::slice::from_ref(&semaphore))
                    .push_next(&mut timeline);
                unsafe {
                    device.queue_submit(queue, core::slice::from_ref(&submit), vk::Fence::null())
                }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
                completions.push(
                    CompletionToken::new(QueueKind::Transfer, value)
                        .map_err(|_| AllocationError::NativeFailure)?,
                );
                source_offset += mip.bytes.len() as u64;
            }
            unsafe { device.queue_wait_idle(queue) }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
            Ok(completions)
        })();
        if submitted.is_err() {
            let _ = unsafe { device.queue_wait_idle(queue) };
        }
        if !commands.is_empty() {
            unsafe { device.free_command_buffers(pool, &commands) };
        }
        let upload_freed = self.free(upload);
        let completions = match (submitted, upload_freed) {
            (Ok(completions), Ok(())) => completions,
            (Err(error), _) | (_, Err(error)) => {
                let _ = self.wait_idle();
                self.destroy_unpublished_texture(
                    &device,
                    image,
                    Some(view),
                    Some(sampler),
                    allocation,
                );
                return Err(error);
            }
        };
        let image_descriptor = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let sampler_descriptor = vk::DescriptorImageInfo::default().sampler(sampler);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(TEXTURE_DESCRIPTOR_BINDING)
                .dst_array_element(binding)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(core::slice::from_ref(&image_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(SAMPLER_DESCRIPTOR_BINDING)
                .dst_array_element(binding)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(core::slice::from_ref(&sampler_descriptor)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok((
            NativeTexture {
                image,
                view,
                allocation,
                binding,
                sampler,
            },
            completions,
        ))
    }

    pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        unsafe {
            device.destroy_image_view(texture.view, None);
            device.destroy_sampler(texture.sampler, None);
            device.destroy_image(texture.image, None)
        };
        self.allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .free(texture.allocation)
            .map_err(map_allocator)
    }

    /// Copies a shader-readable image to host-visible memory and returns tightly packed RGBA8 pixels.
    pub fn readback_texture_rgba8(
        &mut self,
        texture: &NativeTexture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, AllocationError> {
        let size = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|value| value.checked_mul(4))
            .ok_or(AllocationError::ZeroSize)?;
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        let pool = self
            .transfer_command_pool
            .ok_or(AllocationError::NativeFailure)?;
        let semaphore = self
            .transfer_timeline
            .ok_or(AllocationError::NativeFailure)?;
        let queue = self.graphics_queue.ok_or(AllocationError::NativeFailure)?;
        let value = self.next_transfer_value;
        let next_value = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        let mut readback = self.allocate(
            AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                .map_err(|_| AllocationError::ZeroSize)?,
        )?;
        let command = match unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(commands) => commands[0],
            Err(error) => {
                let _ = self.free(readback);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let mut submitted = false;
        let result = (|| {
            unsafe {
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_copy = vk::ImageMemoryBarrier::default()
                .image(texture.image)
                .subresource_range(range)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    core::slice::from_ref(&to_copy),
                )
            };
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            unsafe {
                device.cmd_copy_image_to_buffer(
                    command,
                    texture.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    core::slice::from_ref(&region),
                )
            };
            let to_shader = vk::ImageMemoryBarrier::default()
                .image(texture.image)
                .subresource_range(range)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    core::slice::from_ref(&to_shader),
                );
                device.end_command_buffer(command)
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .signal_semaphore_values(core::slice::from_ref(&value));
            let submit = vk::SubmitInfo::default()
                .command_buffers(core::slice::from_ref(&command))
                .signal_semaphores(core::slice::from_ref(&semaphore))
                .push_next(&mut timeline);
            unsafe {
                device.queue_submit(queue, core::slice::from_ref(&submit), vk::Fence::null())
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            submitted = true;
            self.next_transfer_value = next_value;
            unsafe {
                device.wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(core::slice::from_ref(&semaphore))
                        .values(core::slice::from_ref(&value)),
                    u64::MAX,
                )
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            self.invalidate(&mut readback, 0, size)?;
            Ok(self.mapped_slice(&readback)?[..size as usize].to_vec())
        })();
        if submitted && result.is_err() {
            let _ = unsafe { device.queue_wait_idle(queue) };
        }
        unsafe { device.free_command_buffers(pool, &[command]) };
        let freed = self.free(readback);
        match (result, freed) {
            (Ok(pixels), Ok(())) => Ok(pixels),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    fn probe_device(
        &self,
        physical: vk::PhysicalDevice,
        surface: Option<&NativeSurface>,
    ) -> Result<Option<DeviceProbe>, HalError> {
        // SAFETY: all queries use a physical device returned by this instance.
        let queues = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(physical)
        };
        let mut queue_family = None;
        for (index, queue) in queues.iter().enumerate() {
            if !queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                continue;
            }
            if let Some(surface) = surface {
                // SAFETY: surface and physical device belong to this instance.
                let present = unsafe {
                    self.surface_loader.get_physical_device_surface_support(
                        physical,
                        index as u32,
                        surface.handle,
                    )
                }
                .map_err(map_vk)?;
                if !present {
                    continue;
                }
            }
            queue_family = Some(index as u32);
            break;
        }
        let Some(queue_family) = queue_family else {
            return Ok(None);
        };

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let (vertex_storage, multi_draw, compression) = {
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features12)
                .push_next(&mut features13);
            // SAFETY: output feature chains are valid for the duration of the query.
            unsafe {
                self.instance
                    .get_physical_device_features2(physical, &mut features)
            };
            let vertex_storage = features.features.vertex_pipeline_stores_and_atomics != 0;
            let compression = (if features.features.texture_compression_bc != 0 {
                CompressionSupport::BC
            } else {
                CompressionSupport::NONE
            })
            .union(if features.features.texture_compression_astc_ldr != 0 {
                CompressionSupport::ASTC
            } else {
                CompressionSupport::NONE
            });
            (
                vertex_storage,
                features.features.multi_draw_indirect != 0,
                compression,
            )
        };
        if features12.descriptor_indexing == 0
            || features12.descriptor_binding_partially_bound == 0
            || features12.descriptor_binding_sampled_image_update_after_bind == 0
        {
            return Ok(None);
        }
        let properties = unsafe { self.instance.get_physical_device_properties(physical) };
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut indexing = vk::PhysicalDeviceDescriptorIndexingProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut id)
            .push_next(&mut indexing);
        unsafe {
            self.instance
                .get_physical_device_properties2(physical, &mut properties2)
        };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let class = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => AdapterClass::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => AdapterClass::Integrated,
            vk::PhysicalDeviceType::CPU => AdapterClass::Software,
            _ => AdapterClass::Other,
        };
        let limits = properties.limits;
        let caps = AdapterCapabilities {
            bindless_sampled_textures: paired_texture_capacity(&indexing),
            bindless_storage_resources: indexing
                .max_descriptor_set_update_after_bind_storage_buffers
                .max(indexing.max_descriptor_set_update_after_bind_storage_images),
            bindless_samplers: paired_texture_capacity(&indexing),
            max_indirect_draw_count: limits.max_draw_indirect_count,
            shader_model: 0x0605,
            timeline_synchronization: features12.timeline_semaphore != 0,
            resource_aliasing: true,
            dynamic_rendering: features13.dynamic_rendering != 0
                && features13.synchronization2 != 0,
            presentation: true,
            compression,
        };
        let adapter = AdapterInfo::new(
            Backend::Vulkan,
            id.device_uuid,
            name,
            format!("{}", properties.driver_version),
            class,
            caps,
        )
        .map_err(|_| HalError::Unsupported)?;
        Ok(Some(DeviceProbe {
            adapter,
            queue_family,
            features12,
            features13,
            vertex_storage,
            multi_draw,
        }))
    }
}

impl MemoryAllocator for NativeContext {
    type Allocation = NativeAllocation;

    fn allocate(
        &mut self,
        request: AllocationRequest,
    ) -> Result<Self::Allocation, AllocationError> {
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let create = vk::BufferCreateInfo::default()
            .size(request.size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live and the descriptor contains no borrowed arrays.
        let buffer = unsafe { device.create_buffer(&create, None) }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
        // SAFETY: the buffer was created by this device and remains live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        if requirements.alignment < request.alignment {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(AllocationError::InvalidAlignment);
        }
        let location = match request.memory_class {
            MemoryClass::Device | MemoryClass::Transient => MemoryLocation::GpuOnly,
            MemoryClass::Upload => MemoryLocation::CpuToGpu,
            MemoryClass::Readback => MemoryLocation::GpuToCpu,
        };
        let allocator = self
            .allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?;
        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name: "ez-gfx-buffer",
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(map_allocator(error));
            }
        };
        if request.mapped && allocation.mapped_ptr().is_none() {
            let _ = allocator.free(allocation);
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(AllocationError::NotHostVisible);
        }
        // SAFETY: the allocation is live, compatible with these queried requirements, and outlives the buffer.
        if let Err(error) =
            unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
        {
            let _ = allocator.free(allocation);
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(map_allocation_vk(map_vk(error)));
        }
        Ok(NativeAllocation { buffer, allocation })
    }

    fn mapped_slice<'a>(
        &self,
        allocation: &'a Self::Allocation,
    ) -> Result<&'a [u8], AllocationError> {
        allocation
            .allocation
            .mapped_slice()
            .ok_or(AllocationError::NotHostVisible)
    }

    fn mapped_slice_mut<'a>(
        &mut self,
        allocation: &'a mut Self::Allocation,
    ) -> Result<&'a mut [u8], AllocationError> {
        allocation
            .allocation
            .mapped_slice_mut()
            .ok_or(AllocationError::NotHostVisible)
    }

    fn flush(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError> {
        validate_allocation_range(allocation.allocation.size(), offset, size)?;
        if allocation
            .allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(());
        }
        if allocation.allocation.mapped_ptr().is_none() {
            return Err(AllocationError::NotHostVisible);
        }
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let range = vk::MappedMemoryRange::default()
            .memory(unsafe { allocation.allocation.memory() })
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { device.flush_mapped_memory_ranges(core::slice::from_ref(&range)) }
            .map_err(|error| map_allocation_vk(map_vk(error)))
    }

    fn invalidate(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError> {
        validate_allocation_range(allocation.allocation.size(), offset, size)?;
        if allocation
            .allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(());
        }
        if allocation.allocation.mapped_ptr().is_none() {
            return Err(AllocationError::NotHostVisible);
        }
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let range = vk::MappedMemoryRange::default()
            .memory(unsafe { allocation.allocation.memory() })
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { device.invalidate_mapped_memory_ranges(core::slice::from_ref(&range)) }
            .map_err(|error| map_allocation_vk(map_vk(error)))
    }

    fn free(&mut self, allocation: Self::Allocation) -> Result<(), AllocationError> {
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        unsafe { device.destroy_buffer(allocation.buffer, None) };
        self.allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .free(allocation.allocation)
            .map_err(map_allocator)
    }

    fn retire(
        &mut self,
        allocation: Self::Allocation,
        completion: CompletionToken,
    ) -> Result<(), AllocationError> {
        self.retired.push(RetiredAllocation {
            allocation,
            completion,
        });
        Ok(())
    }

    fn reclaim(&mut self, queue: QueueKind, completed: u64) -> Result<usize, AllocationError> {
        let mut reclaimed = 0;
        let mut index = 0;
        while index < self.retired.len() {
            if self.retired[index].completion.queue == queue
                && self.retired[index].completion.value <= completed
            {
                let retired = self.retired.swap_remove(index);
                self.free(retired.allocation)?;
                reclaimed += 1;
            } else {
                index += 1;
            }
        }
        Ok(reclaimed)
    }
}

impl BufferTransfer for NativeContext {
    fn copy_buffer(
        &mut self,
        source: &Self::Allocation,
        destination: &Self::Allocation,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    ) -> Result<CompletionToken, AllocationError> {
        validate_allocation_range(source.allocation.size(), source_offset, size)?;
        validate_allocation_range(destination.allocation.size(), destination_offset, size)?;
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let pool = self
            .transfer_command_pool
            .ok_or(AllocationError::NativeFailure)?;
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|error| map_allocation_vk(map_vk(error)))?[0];
        unsafe {
            device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|error| map_allocation_vk(map_vk(error)))?;
        let copy = vk::BufferCopy {
            src_offset: source_offset,
            dst_offset: destination_offset,
            size,
        };
        unsafe {
            device.cmd_copy_buffer(
                command,
                source.buffer,
                destination.buffer,
                core::slice::from_ref(&copy),
            )
        };
        unsafe { device.end_command_buffer(command) }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
        let value = self.next_transfer_value;
        self.next_transfer_value = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        let semaphore = self
            .transfer_timeline
            .ok_or(AllocationError::NativeFailure)?;
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(core::slice::from_ref(&value));
        let submit = vk::SubmitInfo::default()
            .command_buffers(core::slice::from_ref(&command))
            .signal_semaphores(core::slice::from_ref(&semaphore))
            .push_next(&mut timeline);
        unsafe {
            device.queue_submit(
                self.graphics_queue.ok_or(AllocationError::NativeFailure)?,
                core::slice::from_ref(&submit),
                vk::Fence::null(),
            )
        }
        .map_err(|error| map_allocation_vk(map_vk(error)))?;
        CompletionToken::new(QueueKind::Transfer, value).map_err(|_| AllocationError::NativeFailure)
    }

    fn completed_transfer_value(&self) -> Result<u64, AllocationError> {
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        unsafe {
            device.get_semaphore_counter_value(
                self.transfer_timeline
                    .ok_or(AllocationError::NativeFailure)?,
            )
        }
        .map_err(|error| map_allocation_vk(map_vk(error)))
    }
}
fn validate_allocation_range(length: u64, offset: u64, size: u64) -> Result<(), AllocationError> {
    if size == 0 || offset.checked_add(size).is_none_or(|end| end > length) {
        return Err(AllocationError::NativeFailure);
    }
    Ok(())
}

fn map_allocation_hal(error: AllocationError) -> HalError {
    match error {
        AllocationError::DeviceLost => HalError::DeviceLost,
        AllocationError::OutOfMemory => HalError::OutOfMemory,
        _ => HalError::NativeFailure,
    }
}

fn map_allocation_vk(error: HalError) -> AllocationError {
    match error {
        HalError::DeviceLost => AllocationError::DeviceLost,
        HalError::OutOfMemory => AllocationError::OutOfMemory,
        _ => AllocationError::NativeFailure,
    }
}

fn map_allocator_hal(error: gpu_allocator::AllocationError) -> HalError {
    match error {
        gpu_allocator::AllocationError::OutOfMemory => HalError::OutOfMemory,
        _ => HalError::NativeFailure,
    }
}

fn map_allocator(error: gpu_allocator::AllocationError) -> AllocationError {
    match error {
        gpu_allocator::AllocationError::OutOfMemory => AllocationError::OutOfMemory,
        _ => AllocationError::NativeFailure,
    }
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        if let Some(device) = self.device.as_ref() {
            // SAFETY: context ownership prevents concurrent use during drop.
            unsafe {
                let _ = device.device_wait_idle();
            }
        }
        let _ = self.destroy_depth_target();
        while let Some(retired) = self.retired.pop() {
            let _ = self.free(retired.allocation);
        }
        drop(self.allocator.take());
        if let Some(device) = self.device.take() {
            for view in self.swapchain_views.drain(..) {
                unsafe { device.destroy_image_view(view, None) };
            }
            if let (Some(loader), Some(swapchain)) =
                (self.swapchain_loader.as_ref(), self.swapchain.take())
            {
                unsafe { loader.destroy_swapchain(swapchain, None) };
            }
            if let Some(semaphore) = self.image_available.take() {
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            if let Some(pool) = self.texture_descriptor_pool.take() {
                unsafe { device.destroy_descriptor_pool(pool, None) };
            }
            if let Some(layout) = self.texture_descriptor_layout.take() {
                unsafe { device.destroy_descriptor_set_layout(layout, None) };
            }
            if let Some(pool) = self.transfer_command_pool.take() {
                unsafe { device.destroy_command_pool(pool, None) };
            }
            if let Some(semaphore) = self.transfer_timeline.take() {
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            // SAFETY: all allocator-owned memory and child resources were released above.
            unsafe {
                device.destroy_device(None);
            }
        }
        // SAFETY: every child owned by the context is destroyed before the instance.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn map_vk(error: vk::Result) -> HalError {
    match error {
        vk::Result::ERROR_DEVICE_LOST => HalError::DeviceLost,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
            HalError::OutOfMemory
        }
        vk::Result::NOT_READY | vk::Result::TIMEOUT => HalError::NotReady,
        vk::Result::ERROR_EXTENSION_NOT_PRESENT
        | vk::Result::ERROR_FEATURE_NOT_PRESENT
        | vk::Result::ERROR_INCOMPATIBLE_DRIVER => HalError::Unsupported,
        _ => HalError::NativeFailure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_heap_layout_matches_slang_bindless_contract() {
        let [texture, sampler] = texture_descriptor_layout_bindings();

        assert_eq!(TEXTURE_DESCRIPTOR_SET, 1);
        assert_eq!(texture.binding, TEXTURE_DESCRIPTOR_BINDING);
        assert_eq!(texture.descriptor_type, vk::DescriptorType::SAMPLED_IMAGE);
        assert_eq!(texture.descriptor_count, TEXTURE_DESCRIPTOR_CAPACITY);
        assert_eq!(texture.stage_flags, vk::ShaderStageFlags::ALL);
        assert_eq!(sampler.binding, SAMPLER_DESCRIPTOR_BINDING);
        assert_eq!(sampler.descriptor_type, vk::DescriptorType::SAMPLER);
        assert_eq!(sampler.descriptor_count, TEXTURE_DESCRIPTOR_CAPACITY);
        assert_eq!(sampler.stage_flags, vk::ShaderStageFlags::ALL);
    }

    #[test]
    fn paired_texture_capacity_honors_every_update_after_bind_limit() {
        let mut limits = vk::PhysicalDeviceDescriptorIndexingProperties {
            max_update_after_bind_descriptors_in_all_pools: 2048,
            max_per_stage_descriptor_update_after_bind_samplers: 1024,
            max_per_stage_descriptor_update_after_bind_sampled_images: 1024,
            max_per_stage_update_after_bind_resources: 2048,
            max_descriptor_set_update_after_bind_samplers: 1024,
            max_descriptor_set_update_after_bind_sampled_images: 1024,
            ..Default::default()
        };

        assert_eq!(paired_texture_capacity(&limits), 1024);
        limits.max_per_stage_update_after_bind_resources = 2046;
        assert_eq!(paired_texture_capacity(&limits), 1023);
    }

    #[test]
    fn sampler_state_preserves_filter_address_and_mip_configuration() {
        let info = sampler_create_info(
            TextureSamplerDesc {
                min_filter: SamplerFilter::Linear,
                mag_filter: SamplerFilter::Nearest,
                max_anisotropy: 16.0,
                address_u: SamplerAddressMode::Repeat,
                address_v: SamplerAddressMode::Clamp,
                address_w: SamplerAddressMode::Repeat,
            },
            5,
        );

        assert_eq!(info.min_filter, vk::Filter::LINEAR);
        assert_eq!(info.mag_filter, vk::Filter::NEAREST);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert_eq!(info.address_mode_u, vk::SamplerAddressMode::REPEAT);
        assert_eq!(info.address_mode_v, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(info.address_mode_w, vk::SamplerAddressMode::REPEAT);
        assert_eq!(info.anisotropy_enable, vk::TRUE);
        assert_eq!(info.max_anisotropy, 16.0);
        assert_eq!(info.max_lod, 5.0);
    }
}
