use super::surface::{WsiCapabilities, instance_extensions};
#[path = "device_presentation.rs"]
mod presentation;
pub(super) use presentation::available_extension;
pub(crate) use presentation::device_extensions;
use presentation::{
    FIFO_LATEST_READY_EXT, FIFO_LATEST_READY_KHR, PhysicalDevicePresentModeFifoLatestReadyFeatures,
    supports_fifo_latest_ready,
};

use super::{
    AdapterCapabilities, AdapterClass, AdapterInfo, AllocationError, AllocationSizes, Allocator,
    AllocatorCreateDesc, Backend, CStr, CString, CompressionSupport,
    DEFAULT_ALLOCATION_BLOCK_POLICY, DeferredNativeResource, DeferredResource, DeviceProbe, Entry,
    FrameSlot, HalError, MemoryAllocator, NativeContext, NativeSurface, PendingDevice,
    PresentationMode, PresentationSupport, SemanticProfile, TEXTURE_DESCRIPTOR_CAPACITY,
    create_frame_slots, khr, map_allocation_hal, map_allocation_vk, map_allocator,
    map_allocator_hal, map_vk, paired_texture_capacity, texture_descriptor_layout_bindings,
    texture_heap_rejection, transfer, vk,
};

fn create_device_frame_state(
    instance: &ash::Instance,
    pending: &mut PendingDevice,
    queue_family: u32,
    swapchain_enabled: bool,
) -> Result<(vk::DescriptorSet, Vec<FrameSlot>), HalError> {
    let device = pending.device.as_ref().ok_or(HalError::NativeFailure)?;
    let bindings = texture_descriptor_layout_bindings();
    let binding_flags = [
        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
    ];
    let mut binding_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
    pending.descriptor_layout = Some(
        // SAFETY: `create_descriptor_set_layout` reads the initialized `bindings` and `binding_info`/`binding_flags` stack storage only for this call on `pending.device`.
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
    let descriptor_layout = pending.descriptor_layout.ok_or(HalError::NativeFailure)?;
    pending.descriptor_pool = Some(
        // SAFETY: `create_descriptor_pool` reads the create info and its inline pool-size array only during this call on `pending.device`.
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
    let descriptor_pool = pending.descriptor_pool.ok_or(HalError::NativeFailure)?;
    // SAFETY: `descriptor_pool` and `descriptor_layout` were created above by `device` with matching update-after-bind flags, and the one-element layout slice lasts through allocation.
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(core::slice::from_ref(&descriptor_layout)),
        )
    }
    .map_err(map_vk)?
    .into_iter()
    .next()
    .ok_or(HalError::NativeFailure)?;
    if swapchain_enabled {
        pending.swapchain_loader = Some(khr::swapchain::Device::new(instance, device));
        pending.image_available = Some(
            // SAFETY: `create_semaphore` uses `pending.device`, no custom allocator, and a default create-info value whose storage lasts through the call.
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(map_vk)?,
        );
    }
    let frame_slots = create_frame_slots(device, queue_family)?;

    Ok((descriptor_set, frame_slots))
}

fn select_transfer_family(properties: &[vk::QueueFamilyProperties], graphics: u32) -> u32 {
    properties
        .iter()
        .position(|family| {
            family.queue_count != 0
                && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .and_then(|index| u32::try_from(index).ok())
        .unwrap_or(graphics)
}

fn draw_feature_rejection(
    vertex_storage: bool,
    multi_draw: bool,
    features12: &vk::PhysicalDeviceVulkan12Features<'_>,
) -> Option<&'static str> {
    if !vertex_storage {
        Some("vertex_pipeline_stores_and_atomics")
    } else if !multi_draw {
        Some("multi_draw_indirect")
    } else if features12.draw_indirect_count == 0 {
        Some("draw_indirect_count")
    } else {
        None
    }
}

pub(crate) const fn cached_device_supports_surface(
    wants_surface: bool,
    swapchain_enabled: bool,
) -> bool {
    !wants_surface || swapchain_enabled
}

impl NativeContext {
    /// Creates a Vulkan context with every surface extension supported on this target.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader or requested validation layer is unavailable, or instance setup fails.
    pub fn create(enable_debug: bool, enable_validation: bool) -> Result<Self, HalError> {
        // SAFETY: loading performs symbol lookup only and errors when the Vulkan loader is unavailable.
        let entry = unsafe { Entry::load() }.map_err(|_| HalError::Unsupported)?;
        let app_name = CString::new("ez_gfx_api").map_err(|_| HalError::NativeFailure)?;
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&app_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        // SAFETY: successful `Entry::load` initialized instance-extension enumeration.
        let available_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(map_vk)?;
        let (extensions, instance_flags, headless_surface_enabled) =
            instance_extensions(&available_extensions, enable_debug);

        let validation =
            CString::new("VK_LAYER_KHRONOS_validation").map_err(|_| HalError::NativeFailure)?;
        let mut layers = Vec::new();
        if enable_validation {
            let available =
                // SAFETY: successful `Entry::load` initialized the instance-layer enumeration function, and ash manages the output storage returned by the call.
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
            .flags(instance_flags)
            .application_info(&app)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        // SAFETY: pointers in `create` remain live for the duration of the call and use Vulkan-owned names.
        let instance = unsafe { entry.create_instance(&create, None) }.map_err(map_vk)?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let context = Self {
            entry_loader: entry,
            instance,
            surface_loader,
            headless_surface_enabled,
            wsi_capabilities: WsiCapabilities::from_enabled(&extensions),
            physical_device: None,
            device: None,
            idle_drained: true,
            adapter_info: None,
            allocator: None,
            retired: Vec::new(),
            graphics_queue: None,
            graphics_queue_lock: std::sync::Arc::new(parking_lot::Mutex::new(())),
            transfer_queue: None,
            deferred: Vec::new(),
            graphics_queue_family: None,
            transfer_queue_family: None,
            transfer_timeline: None,
            transfer_worker: None,
            transfer_command_pool: None,
            transfer_acquire_pool: None,
            texture_staging: ez_gfx_hal::ReusableStagingPool::new(256),
            transfer_ownership_timeline: None,
            texture_timeline: None,
            texture_worker: None,
            texture_command_pool: None,
            texture_acquire_pool: None,
            texture_ownership_timeline: None,
            next_transfer_value: 1,
            next_texture_value: 1,
            texture_descriptor_pool: None,
            texture_descriptor_layout: None,
            texture_descriptor_set: None,
            texture_fallback_bindings: Vec::new(),
            sampler_anisotropy: false,
            swapchain_loader: None,
            presentation_support: PresentationSupport::default(),
            swapchain: None,
            swapchain_images: Vec::new(),
            swapchain_views: Vec::new(),
            swapchain_finished: Vec::new(),
            swapchain_initialized: Vec::new(),
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D::default(),
            swapchain_presentation_mode: PresentationMode::Fifo,
            frame_slots: Vec::new(),
            frame_cursor: 0,
            next_frame_value: 1,
            last_frame_value: 0,
            completed_frame_value: 0,
            image_available: None,
            depth_target: None,
        };
        Ok(context)
    }

    /// Returns the admitted adapter after device initialization.
    pub fn adapter_info(&self) -> Option<&AdapterInfo> {
        self.adapter_info.as_ref()
    }

    /// Device admission checks the semantic floor before creating queues or manager-visible state.
    ///
    /// # Errors
    ///
    /// Returns an error if Vulkan device queries or setup fail, the allocator cannot be created, the surface is unsupported, or no suitable adapter is found.
    ///
    /// # Panics
    ///
    /// Panics if `PendingDevice::new` leaves its `device` field empty.
    pub fn init_device(
        &mut self,
        surface: Option<&NativeSurface>,
    ) -> Result<AdapterInfo, HalError> {
        self.init_device_inner(surface, None)
    }

    /// Creates the device for one explicitly selected adapter.
    ///
    /// Ranking is bypassed but admission never is: an unknown identity fails
    /// `InvalidArgument`, disallowed software fails `InvalidArgument`, and a
    /// matched but inadmissible adapter fails `Unsupported`. A matched adapter
    /// lacking required draw or queue features also fails `Unsupported`.
    ///
    /// # Errors
    ///
    /// Returns an error if device queries or setup fail or no enumerated
    /// adapter matches the requested identity under admission policy.
    pub fn init_device_for_adapter(
        &mut self,
        surface: Option<&NativeSurface>,
        stable_id: [u8; 16],
        allow_software: bool,
    ) -> Result<AdapterInfo, HalError> {
        self.init_device_inner(surface, Some((stable_id, allow_software)))
    }

    /// Enumerates every physical device with a graphics queue, admitted or not.
    ///
    /// Rejected adapters stay listed so rejection diagnostics can name them;
    /// devices without usable queues or required draw features are skipped
    /// because they can never back a context. Surface presentation is checked at
    /// device-creation time, not here.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader is unavailable or physical-device
    /// enumeration fails.
    pub fn enumerate_adapters() -> Result<Vec<AdapterInfo>, HalError> {
        // A throwaway instance suffices: adapter enumeration is independent of
        // surface creation, and dropping the probe reclaims the instance.
        let probe = NativeContext::create(false, false)?;
        // SAFETY: the instance is live and owns returned physical-device handles.
        let devices = unsafe { probe.instance.enumerate_physical_devices() }.map_err(map_vk)?;
        let mut adapters = Vec::new();
        for physical in devices {
            // One undescribable adapter skips itself, never the enumeration.
            let Ok(Some(candidate)) = probe.probe_device(physical, None) else {
                continue;
            };
            if draw_feature_rejection(
                candidate.vertex_storage,
                candidate.multi_draw,
                &candidate.features12,
            )
            .is_some()
            {
                continue;
            }
            adapters.push(candidate.adapter);
        }
        Ok(adapters)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "device admission and rollback remain local across optional native objects"
    )]
    fn init_device_inner(
        &mut self,
        surface: Option<&NativeSurface>,
        selection: Option<([u8; 16], bool)>,
    ) -> Result<AdapterInfo, HalError> {
        if let (Some(physical), Some(queue_family), Some(adapter)) = (
            self.physical_device,
            self.graphics_queue_family,
            self.adapter_info.as_ref(),
        ) {
            if !cached_device_supports_surface(surface.is_some(), self.swapchain_loader.is_some()) {
                return Err(HalError::Unsupported);
            }
            if let Some((wanted, _)) = selection
                && adapter.stable_id() != wanted
            {
                return Err(HalError::InvalidArgument);
            }
            if let Some(surface) = surface {
                // SAFETY: `physical` and `queue_family` come from this instance, but same-instance provenance of the caller-supplied `surface.handle` is not established here.
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
            let Some(candidate) = self.probe_device(physical, surface)? else {
                continue;
            };
            if draw_feature_rejection(
                candidate.vertex_storage,
                candidate.multi_draw,
                &candidate.features12,
            )
            .is_some()
            {
                if selection.is_some_and(|(wanted, _)| wanted == candidate.adapter.stable_id()) {
                    return Err(HalError::Unsupported);
                }
                continue;
            }
            if let Some((wanted, allow_software)) = selection {
                if candidate.adapter.stable_id() != wanted {
                    continue;
                }
                // Explicit selection bypasses ranking but never bypasses admission.
                if candidate.adapter.class() == AdapterClass::Software && !allow_software {
                    return Err(HalError::InvalidArgument);
                }
                if SemanticProfile::V1
                    .admit(candidate.adapter.capabilities())
                    .is_err()
                {
                    return Err(HalError::Unsupported);
                }
            } else if SemanticProfile::V1
                .admit(candidate.adapter.capabilities())
                .is_err()
            {
                continue;
            }
            let DeviceProbe {
                adapter,
                queue_family,
                features12,
                features13,
                vertex_storage,
                multi_draw,
            } = candidate;
            // Prefer a transfer-only family; otherwise use a second queue from the graphics
            // family when available, with queue zero as the universal fallback.
            // SAFETY: the enumerated physical device belongs to this live instance.
            let queue_properties = unsafe {
                self.instance
                    .get_physical_device_queue_family_properties(physical)
            };
            let transfer_family = select_transfer_family(&queue_properties, queue_family);
            let graphics_queue_count =
                queue_properties[queue_family as usize].queue_count.min(3) as usize;
            let transfer_queue_count = queue_properties[transfer_family as usize]
                .queue_count
                .min(2) as usize;
            let priorities = [1.0_f32, 1.0_f32, 1.0_f32];
            let queue_infos = if transfer_family == queue_family {
                vec![
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family)
                        .queue_priorities(&priorities[..graphics_queue_count]),
                ]
            } else {
                vec![
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family)
                        .queue_priorities(&priorities[..1]),
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(transfer_family)
                        .queue_priorities(&priorities[..transfer_queue_count]),
                ]
            };
            let mut enabled12 = vk::PhysicalDeviceVulkan12Features::default()
                .timeline_semaphore(features12.timeline_semaphore != 0)
                .draw_indirect_count(features12.draw_indirect_count != 0)
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
            // SAFETY: `physical` was returned by `self.instance.enumerate_physical_devices` in this initialization pass and remains usable for the feature query.
            let core_features = unsafe { self.instance.get_physical_device_features(physical) };
            let enabled_core = vk::PhysicalDeviceFeatures::default()
                .vertex_pipeline_stores_and_atomics(vertex_storage)
                .multi_draw_indirect(multi_draw)
                .sampler_anisotropy(core_features.sampler_anisotropy != 0);
            // Portability devices require `VK_KHR_portability_subset`; ordinary devices omit it.
            // SAFETY: `physical` belongs to this live instance.
            let available_device_extensions = unsafe {
                self.instance
                    .enumerate_device_extension_properties(physical)
            }
            .map_err(map_vk)?;
            let (mut enabled_extensions, swapchain_enabled, fifo_latest_ready_extension) =
                device_extensions(&available_device_extensions);
            if surface.is_some() && !swapchain_enabled {
                continue;
            }
            let fifo_latest_ready_enabled =
                supports_fifo_latest_ready(&self.instance, physical, fifo_latest_ready_extension);
            if !fifo_latest_ready_enabled {
                enabled_extensions.pop_if(|extension| {
                    *extension == FIFO_LATEST_READY_KHR.as_ptr()
                        || *extension == FIFO_LATEST_READY_EXT.as_ptr()
                });
            }
            let mut latest_ready = PhysicalDevicePresentModeFifoLatestReadyFeatures {
                present_mode_fifo_latest_ready: vk::TRUE,
                ..Default::default()
            };
            let mut create = vk::DeviceCreateInfo::default()
                .enabled_features(&enabled_core)
                .enabled_extension_names(&enabled_extensions)
                .queue_create_infos(&queue_infos)
                .push_next(&mut enabled11)
                .push_next(&mut enabled12)
                .push_next(&mut enabled13);
            if fifo_latest_ready_enabled {
                latest_ready.p_next = create.p_next.cast_mut();
                create.p_next = (&raw const latest_ready).cast();
            }
            // SAFETY: the physical device and queue family were queried from this live instance.
            let device =
                unsafe { self.instance.create_device(physical, &create, None) }.map_err(map_vk)?;
            let mut pending = PendingDevice::new(device);
            let device = pending
                .device
                .as_ref()
                .expect("pending device is initialized");
            // SAFETY: queue zero was requested from the selected graphics family above.
            let graphics_queue = unsafe { device.get_device_queue(queue_family, 0) };
            let transfer_index =
                u32::from(transfer_family == queue_family && graphics_queue_count > 1);
            // SAFETY: the selected transfer family and index were included in `queue_infos`.
            let transfer_queue =
                unsafe { device.get_device_queue(transfer_family, transfer_index) };
            let texture_index = if transfer_family == queue_family {
                if graphics_queue_count > 2 {
                    2
                } else {
                    transfer_index
                }
            } else {
                u32::from(transfer_queue_count > 1)
            };
            // SAFETY: the selected texture family and index were included in `queue_infos`.
            let texture_queue = unsafe { device.get_device_queue(transfer_family, texture_index) };
            pending.allocator = Some(
                Allocator::new(&AllocatorCreateDesc {
                    instance: self.instance.clone(),
                    device: device.clone(),
                    physical_device: physical,
                    debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                    buffer_device_address: true,
                    allocation_sizes: AllocationSizes::new(
                        DEFAULT_ALLOCATION_BLOCK_POLICY.initial_device,
                        DEFAULT_ALLOCATION_BLOCK_POLICY.initial_host,
                    )
                    .with_max_device_memblock_size(DEFAULT_ALLOCATION_BLOCK_POLICY.maximum_device)
                    .with_max_host_memblock_size(DEFAULT_ALLOCATION_BLOCK_POLICY.maximum_host),
                })
                .map_err(|error| map_allocator_hal(&error))?,
            );
            pending.command_pool = Some(
                // SAFETY: `transfer_family` belongs to the physical device used to create
                // `device`, and the command-pool create-info storage spans the call.
                unsafe {
                    device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(transfer_family)
                            .flags(
                                vk::CommandPoolCreateFlags::TRANSIENT
                                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            ),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            pending.texture_command_pool = Some(
                // SAFETY: this pool uses the admitted transfer family and live device.
                unsafe {
                    device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(transfer_family)
                            .flags(
                                vk::CommandPoolCreateFlags::TRANSIENT
                                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            ),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            pending.acquire_pool = Some(
                // SAFETY: `queue_family` belongs to this device and the create-info spans the
                // call. Same-family texture updates also use this pool for queue handoff.
                unsafe {
                    device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(queue_family)
                            .flags(
                                vk::CommandPoolCreateFlags::TRANSIENT
                                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            ),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            pending.texture_acquire_pool = Some(
                // SAFETY: this pool uses the admitted graphics family and live device.
                unsafe {
                    device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(queue_family)
                            .flags(
                                vk::CommandPoolCreateFlags::TRANSIENT
                                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            ),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            pending.timeline = Some(
                // SAFETY: profile admission required and device creation enabled timeline semaphores, while the `timeline` pNext storage lasts through `create_semaphore`.
                unsafe {
                    device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            let mut texture_timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            pending.texture_timeline = Some(
                // SAFETY: timeline semaphores are enabled and pNext storage spans the call.
                unsafe {
                    device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut texture_timeline),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            let mut ownership_timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            pending.ownership_timeline = Some(
                // SAFETY: timeline semaphores were enabled and pNext storage spans the call.
                unsafe {
                    device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut ownership_timeline),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            let mut texture_ownership_timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            pending.texture_ownership_timeline = Some(
                // SAFETY: the timeline serializes graphics-to-transfer texture updates even
                // when both queues belong to the same family.
                unsafe {
                    device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default()
                            .push_next(&mut texture_ownership_timeline),
                        None,
                    )
                }
                .map_err(map_vk)?,
            );
            let worker_device = device.clone();
            let transfer_lock = std::sync::Arc::new(parking_lot::Mutex::new(()));
            let texture_lock = if texture_queue == transfer_queue {
                transfer_lock.clone()
            } else {
                std::sync::Arc::new(parking_lot::Mutex::new(()))
            };
            let graphics_lock = self.graphics_queue_lock.clone();
            let (descriptor_set, frame_slots) = create_device_frame_state(
                &self.instance,
                &mut pending,
                queue_family,
                swapchain_enabled,
            )?;

            let transfer_worker = transfer::start_worker(
                worker_device.clone(),
                transfer_queue,
                graphics_queue,
                transfer_family,
                queue_family,
                pending.command_pool.ok_or(HalError::NativeFailure)?,
                pending.acquire_pool,
                pending.timeline.ok_or(HalError::NativeFailure)?,
                pending.ownership_timeline,
                transfer_lock,
                graphics_lock.clone(),
            )
            .map_err(|_| HalError::NativeFailure)?;
            let texture_worker = transfer::start_worker(
                worker_device.clone(),
                texture_queue,
                graphics_queue,
                transfer_family,
                queue_family,
                pending
                    .texture_command_pool
                    .ok_or(HalError::NativeFailure)?,
                pending.texture_acquire_pool,
                pending.texture_timeline.ok_or(HalError::NativeFailure)?,
                pending.texture_ownership_timeline,
                texture_lock,
                graphics_lock,
            )
            .map_err(|_| HalError::NativeFailure)?;
            self.physical_device = Some(physical);
            self.adapter_info = Some(adapter.clone());
            self.graphics_queue_family = Some(queue_family);
            self.transfer_queue_family = Some(transfer_family);
            self.graphics_queue = Some(graphics_queue);
            self.transfer_queue = Some(transfer_queue);
            self.allocator = pending.allocator.take();
            self.sampler_anisotropy = core_features.sampler_anisotropy != 0;
            self.device = pending.device.take();
            self.transfer_timeline = pending.timeline.take();
            self.transfer_worker = Some(transfer_worker);
            self.transfer_command_pool = pending.command_pool.take();
            self.transfer_acquire_pool = pending.acquire_pool.take();
            self.transfer_ownership_timeline = pending.ownership_timeline.take();
            self.texture_timeline = pending.texture_timeline.take();
            self.texture_worker = Some(texture_worker);
            self.texture_command_pool = pending.texture_command_pool.take();
            self.texture_acquire_pool = pending.texture_acquire_pool.take();
            self.texture_ownership_timeline = pending.texture_ownership_timeline.take();
            self.texture_descriptor_pool = pending.descriptor_pool.take();
            self.texture_descriptor_layout = pending.descriptor_layout.take();
            self.texture_descriptor_set = Some(descriptor_set);
            self.presentation_support.fifo_latest_ready = fifo_latest_ready_enabled;
            self.swapchain_loader = pending.swapchain_loader.take();
            self.image_available = pending.image_available.take();
            self.frame_slots = frame_slots;
            return Ok(adapter);
        }
        // An unmatched explicit identity names no adapter; default first-fit
        // keeps the legacy admission failure.
        Err(if selection.is_some() {
            HalError::InvalidArgument
        } else {
            HalError::Unsupported
        })
    }
    /// Waits for submitted native work and reclaims completed deferred resources.
    ///
    /// # Errors
    ///
    /// Returns an error if no device is initialized, waiting for the device fails, or reclaiming a deferred allocation fails.
    pub fn wait_idle(&mut self) -> Result<(), HalError> {
        self.idle_drained = self.device.is_none();
        let device = self.device.as_ref().ok_or(HalError::NotReady)?.clone();
        let mut worker_failed = false;
        let mut worker_lost = false;
        for worker in [&mut self.transfer_worker, &mut self.texture_worker] {
            if let Some(worker) = worker {
                if let Err(error) = worker.flush() {
                    // Join terminal cleanup before native idle; failed callbacks may
                    // have submitted only the first half of a queue handoff.
                    worker.shutdown();
                    worker_failed = true;
                    worker_lost |= error == ez_gfx_hal::TransferWorkerError::DeviceLost
                        || worker.device_lost();
                }
            } else {
                worker_failed = true;
            }
        }
        // SAFETY: `device` is the initialized logical device, and exclusive `&mut self` access prevents this context from concurrently submitting through its queues during `device_wait_idle`.
        unsafe { device.device_wait_idle() }.map_err(map_vk)?;
        self.idle_drained = true;
        if worker_failed {
            return Err(if worker_lost {
                HalError::DeviceLost
            } else {
                HalError::NativeFailure
            });
        }
        self.reclaim(ez_gfx_hal::QueueKind::Transfer, u64::MAX)
            .map_err(map_allocation_hal)?;
        self.reclaim(ez_gfx_hal::QueueKind::TextureTransfer, u64::MAX)
            .map_err(map_allocation_hal)?;
        for slot in &mut self.frame_slots {
            slot.in_flight = false;
        }
        self.completed_frame_value = self.last_frame_value;
        let resources = self
            .deferred
            .drain(..)
            .map(|item| item.resource)
            .collect::<Vec<_>>();
        for resource in resources {
            self.destroy_deferred_now(resource)
                .map_err(map_allocation_hal)?;
        }
        Ok(())
    }

    /// Reports on-demand allocator and device-memory telemetry.
    ///
    /// Calls `generate_report`, which allocates; never call per frame. Counts
    /// and sizes saturate instead of wrapping, and unknown context-level values
    /// report zero. Swapchain and depth sizes are resolution-scaled estimates,
    /// not driver measurements.
    pub fn memory_telemetry(&self) -> ez_gfx_hal::BackendMemoryTelemetry {
        // Report generation walks live blocks, so this observes the allocator
        // exactly once per explicit query rather than sampling per frame.
        let allocator = self.allocator.as_ref().map(|allocator| {
            let report = allocator.generate_report();
            ez_gfx_hal::AllocatorTelemetry {
                live_bytes: report.total_allocated_bytes,
                block_bytes: report.total_capacity_bytes,
                block_count: u32::try_from(report.blocks.len()).unwrap_or(u32::MAX),
                allocation_count: u32::try_from(report.allocations.len()).unwrap_or(u32::MAX),
            }
        });
        let images = u32::try_from(self.swapchain_images.len()).unwrap_or(u32::MAX);
        let depth_bytes = self.depth_target.as_ref().map_or(0, |target| {
            ez_gfx_hal::rgba8_image_bytes(1, target.extent.width, target.extent.height)
        });
        ez_gfx_hal::BackendMemoryTelemetry {
            allocator,
            swapchain_images: images,
            swapchain_extent: (self.swapchain_extent.width, self.swapchain_extent.height),
            swapchain_format: u32::try_from(self.swapchain_format.as_raw()).unwrap_or(u32::MAX),
            swapchain_bytes: ez_gfx_hal::rgba8_image_bytes(
                images,
                self.swapchain_extent.width,
                self.swapchain_extent.height,
            ),
            depth_bytes,
            frame_slots: u32::try_from(self.frame_slots.len()).unwrap_or(u32::MAX),
        }
    }

    /// Reports whether the last idle attempt proved native storage safe to release.
    /// Call `wait_idle` immediately before consulting this terminal-cleanup status.
    pub const fn is_drained(&self) -> bool {
        self.idle_drained
    }

    pub(super) fn in_flight_mask(&self) -> u8 {
        self.frame_slots
            .iter()
            .enumerate()
            .fold(0_u8, |mask, (index, slot)| {
                if slot.in_flight {
                    mask | (1 << index)
                } else {
                    mask
                }
            })
    }
    /// Reaps frame slots whose fences already signaled without blocking.
    ///
    /// Polling texture paths call this before consulting the descriptor gate so
    /// completed submissions unblock publication during sustained rendering.
    /// Slot reuse already reclaims the same state on wrap-around; this only
    /// observes fence status and never waits.
    ///
    /// # Errors
    ///
    /// Returns an error when the device is missing or fence status cannot be queried.
    pub fn poll_frame_completion(&mut self) -> Result<(), AllocationError> {
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        for slot_index in 0..self.frame_slots.len() {
            if !self.frame_slots[slot_index].in_flight {
                continue;
            }
            // SAFETY: the fence belongs to this slot on the retained live device,
            // and a status query performs no wait and mutates no command state.
            let signaled = unsafe { device.get_fence_status(self.frame_slots[slot_index].fence) }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
            if signaled {
                self.frame_slots[slot_index].in_flight = false;
                self.complete_frame_slot(slot_index)?;
            }
        }
        Ok(())
    }

    /// Returns the most recently submitted graphics-frame token.
    pub fn last_frame_completion(&self) -> Option<ez_gfx_hal::CompletionToken> {
        ez_gfx_hal::CompletionToken::new(ez_gfx_hal::QueueKind::Graphics, self.last_frame_value)
            .ok()
    }

    /// Polls frame fences and returns the completed graphics-frame prefix.
    ///
    /// # Errors
    ///
    /// Returns an allocation error when a fence status cannot be queried.
    pub fn completed_frame_value(&mut self) -> Result<u64, AllocationError> {
        self.poll_frame_completion()?;
        Ok(self.completed_frame_value)
    }
    ///
    /// # Errors
    ///
    /// Returns an error if immediate destruction requires a missing device or allocator, or freeing the allocation fails.
    pub(super) fn defer_resource(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        let pending_slots = self.in_flight_mask();
        if pending_slots == 0 {
            return self.destroy_deferred_now(resource);
        }
        self.deferred.push(DeferredNativeResource {
            pending_slots,
            resource,
        });
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if `slot_index` cannot select a bit in the `u8` slot mask or destroying a newly unblocked resource fails.
    pub(super) fn complete_frame_slot(&mut self, slot_index: usize) -> Result<(), AllocationError> {
        let bit = 1_u8
            .checked_shl(u32::try_from(slot_index).map_err(|_| AllocationError::NativeFailure)?)
            .ok_or(AllocationError::NativeFailure)?;
        let submission_value = self
            .frame_slots
            .get(slot_index)
            .ok_or(AllocationError::NativeFailure)?
            .submission_value;
        self.completed_frame_value = self.completed_frame_value.max(submission_value);
        let mut ready = Vec::new();
        let mut index = 0;
        while index < self.deferred.len() {
            self.deferred[index].pending_slots &= !bit;
            if self.deferred[index].pending_slots == 0 {
                ready.push(self.deferred.swap_remove(index).resource);
            } else {
                index += 1;
            }
        }
        for resource in ready {
            self.destroy_deferred_now(resource)?;
        }
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if the device or a required allocator is missing, or freeing an allocation fails.
    pub(super) fn destroy_deferred_now(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        match resource {
            DeferredResource::Allocation(allocation) => {
                // SAFETY: deferred reclamation consumes `allocation.buffer` only after its pending frame-slot mask clears, before freeing its paired allocation storage.
                unsafe { device.destroy_buffer(allocation.buffer, None) };
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(allocation.allocation)
                    .map_err(|error| map_allocator(&error))
            }
            DeferredResource::Pipeline(pipeline) => {
                // SAFETY: the deferred `pipeline` resource is consumed once after its pending frame-slot mask clears, so its pipeline and layout handles have no remaining submitted uses.
                unsafe {
                    device.destroy_pipeline(pipeline.pipeline, None);
                    device.destroy_pipeline_layout(pipeline.layout, None);
                    device.destroy_descriptor_set_layout(pipeline.public_descriptor_layout, None);
                }
                Ok(())
            }
            DeferredResource::Shader(shader) => {
                for module in shader.modules {
                    // SAFETY: each `module` is consumed once from deferred shader storage, and Vulkan pipelines retain no `VkShaderModule` storage after pipeline creation.
                    unsafe { device.destroy_shader_module(module, None) };
                }
                Ok(())
            }
            DeferredResource::TextureView(view) => {
                // SAFETY: the replaced view is destroyed only after every frame slot that could
                // have observed its descriptor has completed.
                unsafe { device.destroy_image_view(view, None) };
                Ok(())
            }
            DeferredResource::Texture(texture) => {
                // The MSAA render storage is never sampled or described, so
                // only its view and image retire alongside the sampled image.
                let msaa = texture.msaa;
                // SAFETY: the deferred texture is consumed after its pending frame-slot mask clears; its view and sampler are destroyed before its image and allocation storage.
                unsafe {
                    device.destroy_image_view(texture.view, None);
                    device.destroy_sampler(texture.sampler, None);
                    device.destroy_image(texture.image, None);
                    if let Some(storage) = msaa.as_ref() {
                        device.destroy_image_view(storage.view, None);
                        device.destroy_image(storage.image, None);
                    }
                }
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(texture.allocation)
                    .map_err(|error| map_allocator(&error))?;
                if let Some(storage) = msaa {
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(storage.allocation)
                        .map_err(|error| map_allocator(&error))?;
                }
                Ok(())
            }
        }
    }

    /// Destroys backend-owned state associated with a borrowed host surface.
    ///
    /// Returns `false` only when native work could not drain and the surface was abandoned.
    pub fn destroy_surface(&mut self, surface: NativeSurface) -> bool {
        let _ = self.wait_idle();
        if !self.is_drained() {
            // Keep the surface and swapchain alive when submitted uses cannot retire.
            core::mem::forget(surface);
            return false;
        }
        let _ = self.destroy_depth_target();
        if let Some(device) = self.device.as_ref() {
            for view in self.swapchain_views.drain(..) {
                // SAFETY: native idle proved all uses complete; each view is destroyed once before its swapchain.
                unsafe { device.destroy_image_view(view, None) };
            }
        }
        if let (Some(loader), Some(swapchain)) =
            (self.swapchain_loader.as_ref(), self.swapchain.take())
        {
            // SAFETY: native idle proved all uses complete and the swapchain's views were destroyed first.
            unsafe { loader.destroy_swapchain(swapchain, None) };
        }
        self.swapchain_format = vk::Format::UNDEFINED;
        // Images die with their swapchain; dropping the cache here keeps stale
        // handles from surviving past destruction.
        self.swapchain_images.clear();
        self.swapchain_initialized.clear();
        // Images, views, and the swapchain are gone; the extent must die with
        // them or telemetry keeps reporting a resolution for a dead surface.
        self.swapchain_extent = vk::Extent2D::default();
        if surface.handle != vk::SurfaceKHR::null() {
            // SAFETY: the host window is still live and no swapchain references this surface.
            unsafe { self.surface_loader.destroy_surface(surface.handle, None) };
        }
        drop(surface);
        true
    }

    ///
    /// # Errors
    ///
    /// Returns an error if a queue-family index does not fit in `u32`, a surface-support query fails, or adapter metadata is rejected.
    pub(super) fn probe_device(
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
                        u32::try_from(index).map_err(|_| HalError::NativeFailure)?,
                        surface.handle,
                    )
                }
                .map_err(map_vk)?;
                if !present {
                    continue;
                }
            }
            queue_family = Some(u32::try_from(index).map_err(|_| HalError::NativeFailure)?);
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
                    .get_physical_device_features2(physical, &mut features);
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
        if texture_heap_rejection(&features12).is_some() {
            return Ok(None);
        }
        // SAFETY: `physical` was obtained from `self.instance` during this probe pass and remains usable for `get_physical_device_properties`.
        let properties = unsafe { self.instance.get_physical_device_properties(physical) };
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut indexing = vk::PhysicalDeviceDescriptorIndexingProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut id)
            .push_next(&mut indexing);
        // SAFETY: `properties2` links initialized `id` and `indexing` output storage, all of which remains mutable and allocated for the properties query.
        unsafe {
            self.instance
                .get_physical_device_properties2(physical, &mut properties2);
        };
        // SAFETY: Vulkan filled `properties.device_name` as a NUL-terminated fixed-size array, and that array remains allocated through the `CStr` conversion.
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

#[cfg(test)]
#[path = "device_tests.rs"]
mod tests;
