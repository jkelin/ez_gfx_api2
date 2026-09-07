use super::{
    AdapterCapabilities, AdapterClass, AdapterInfo, AllocationError, AllocationSizes, Allocator,
    AllocatorCreateDesc, Backend, CStr, CString, CompressionSupport,
    DEFAULT_ALLOCATION_BLOCK_POLICY, DeferredNativeResource, DeferredResource, DeviceProbe, Entry,
    FrameSlot, HalError, MemoryAllocator, NativeContext, NativeSurface, PendingDevice,
    SemanticProfile, SurfacePlatform, TEXTURE_DESCRIPTOR_CAPACITY, create_frame_slots, khr,
    map_allocation_hal, map_allocation_vk, map_allocator, map_allocator_hal, map_vk,
    paired_texture_capacity, texture_descriptor_layout_bindings, texture_heap_rejection, transfer,
    vk,
};

fn create_device_frame_state(
    instance: &ash::Instance,
    pending: &mut PendingDevice,
    queue_family: u32,
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
    pending.swapchain_loader = Some(khr::swapchain::Device::new(instance, device));
    pending.image_available = Some(
        // SAFETY: `create_semaphore` uses `pending.device`, no custom allocator, and a default create-info value whose storage lasts through the call.
        unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
            .map_err(map_vk)?,
    );
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

impl NativeContext {
    /// Creates a Vulkan context after validating the requested platform and optional validation layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader or requested validation layer is unavailable, or instance setup fails.
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

        let headless_surface_enabled = if platform == SurfacePlatform::Headless {
            // SAFETY: successful `Entry::load` initialized instance-extension enumeration.
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(map_vk)?
                .iter()
                .any(|extension| {
                    // SAFETY: Vulkan guarantees a NUL-terminated fixed-size extension name.
                    (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
                        == ash::ext::headless_surface::NAME
                })
        } else {
            false
        };
        let mut extensions = Vec::new();
        match platform {
            SurfacePlatform::Win32 => {
                extensions.push(khr::surface::NAME.as_ptr());
                extensions.push(khr::win32_surface::NAME.as_ptr());
            }
            SurfacePlatform::Headless if headless_surface_enabled => {
                extensions.push(khr::surface::NAME.as_ptr());
                extensions.push(ash::ext::headless_surface::NAME.as_ptr());
            }
            // Capable drivers may omit VK_EXT_headless_surface. Contexts remain
            // usable for target-only work through a logical surfaceless target.
            SurfacePlatform::Headless => {}
        }
        if enable_debug {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

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
            sampler_anisotropy: false,
            swapchain_loader: None,
            swapchain: None,
            swapchain_views: Vec::new(),
            swapchain_finished: Vec::new(),
            swapchain_initialized: Vec::new(),
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D::default(),
            frame_slots: Vec::new(),
            frame_cursor: 0,
            image_available: None,
            depth_target: None,
        };
        Ok(context)
    }

    /// Win32 handles are borrowed; null handles are rejected before the native call.
    #[cfg(windows)]
    ///
    /// # Errors
    ///
    /// Returns an error if either Win32 handle is null or Vulkan surface creation fails.
    pub fn create_win32_surface(
        &self,
        window: *mut core::ffi::c_void,
        display: *mut core::ffi::c_void,
    ) -> Result<NativeSurface, HalError> {
        if window.is_null() || display.is_null() {
            return Err(HalError::InvalidArgument);
        }
        let Self {
            entry_loader,
            instance,
            ..
        } = self;
        let loader = khr::win32_surface::Instance::new(entry_loader, instance);
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
    /// Creates a Vulkan surface from borrowed Win32 handles.
    ///
    /// # Errors
    ///
    /// Returns `HalError::Unsupported` on non-Windows platforms.
    pub fn create_win32_surface(
        &self,
        _window: *mut core::ffi::c_void,
        _display: *mut core::ffi::c_void,
    ) -> Result<NativeSurface, HalError> {
        Err(HalError::Unsupported)
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
    /// lacking core queue features also fails `Unsupported`.
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
    /// devices without usable queues or core features are skipped because they
    /// can never back a context. Surface presentation is checked at
    /// device-creation time, not here.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader is unavailable or physical-device
    /// enumeration fails.
    pub fn enumerate_adapters() -> Result<Vec<AdapterInfo>, HalError> {
        // A throwaway instance suffices: description needs properties, features,
        // and queue families only. Drop reclaims context state; the instance
        // handle follows the existing context lifecycle.
        // Win32 instances need a Win32 loader; every other host probes headless.
        let platform = if cfg!(windows) {
            SurfacePlatform::Win32
        } else {
            SurfacePlatform::Headless
        };
        let probe = NativeContext::create(false, false, platform)?;
        // SAFETY: the instance is live and owns returned physical-device handles.
        let devices = unsafe { probe.instance.enumerate_physical_devices() }.map_err(map_vk)?;
        let mut adapters = Vec::new();
        for physical in devices {
            // One undescribable adapter skips itself, never the enumeration.
            let Ok(Some(candidate)) = probe.probe_device(physical, None) else {
                continue;
            };
            if !candidate.vertex_storage || !candidate.multi_draw {
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
            if !candidate.vertex_storage || !candidate.multi_draw {
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
            let swapchain_extensions = [khr::swapchain::NAME.as_ptr()];
            let enabled_extensions = if surface.is_some() {
                swapchain_extensions.as_slice()
            } else {
                &[]
            };
            let create = vk::DeviceCreateInfo::default()
                .enabled_features(&enabled_core)
                .enabled_extension_names(enabled_extensions)
                .queue_create_infos(&queue_infos)
                .push_next(&mut enabled11)
                .push_next(&mut enabled12)
                .push_next(&mut enabled13);
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
            let (descriptor_set, frame_slots) =
                create_device_frame_state(&self.instance, &mut pending, queue_family)?;

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
    pub fn destroy_surface(&mut self, surface: NativeSurface) {
        let _ = self.wait_idle();
        if !self.is_drained() {
            // Keep the surface and swapchain alive when submitted uses cannot retire.
            core::mem::forget(surface);
            return;
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
        self.swapchain_initialized.clear();
        self.swapchain_extent = vk::Extent2D::default();
        if surface.handle != vk::SurfaceKHR::null() {
            // SAFETY: the host window is still live and no swapchain references this surface.
            unsafe { self.surface_loader.destroy_surface(surface.handle, None) };
        }
        drop(surface);
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
mod tests {
    use super::*;

    fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            ..Default::default()
        }
    }

    #[test]
    fn transfer_family_prefers_non_graphics_hardware_queue() {
        let families = [
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER, 2),
            family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 1),
        ];

        assert_eq!(select_transfer_family(&families, 0), 1);
    }

    #[test]
    fn transfer_family_falls_back_when_specialized_queue_is_unavailable() {
        let families = [
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER, 1),
            family(vk::QueueFlags::TRANSFER, 0),
        ];

        assert_eq!(select_transfer_family(&families, 0), 0);
    }
}

#[cfg(test)]
mod adapter_tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn enumeration_reports_unique_named_adapters() {
        // No surface is created, shown, or activated by this test.
        let adapters = NativeContext::enumerate_adapters().expect("Vulkan enumerates adapters");
        assert!(!adapters.is_empty());
        let mut identities = BTreeSet::new();
        for adapter in &adapters {
            assert_ne!(adapter.stable_id(), [0; 16]);
            assert!(!adapter.name().is_empty());
            assert!(!adapter.driver().is_empty());
            assert!(identities.insert(adapter.stable_id()));
        }
    }

    #[test]
    fn explicit_selection_rejects_unknown_identity() {
        // No surface is created, shown, or activated by this test.
        // Win32 instances need a Win32 loader; every other host probes headless.
        let platform = if cfg!(windows) {
            SurfacePlatform::Win32
        } else {
            SurfacePlatform::Headless
        };
        let mut context = NativeContext::create(false, false, platform).expect("Vulkan instance");
        assert_eq!(
            context.init_device_for_adapter(None, [0xA5; 16], false),
            Err(HalError::InvalidArgument)
        );
    }

    #[test]
    fn explicit_selection_admits_enumerated_adapter() {
        // No surface is created, shown, or activated by this test.
        // Win32 instances need a Win32 loader; every other host probes headless.
        let platform = if cfg!(windows) {
            SurfacePlatform::Win32
        } else {
            SurfacePlatform::Headless
        };
        let adapters = NativeContext::enumerate_adapters().expect("Vulkan enumerates adapters");
        let wanted = adapters.first().expect("at least one adapter").stable_id();
        let mut context = NativeContext::create(false, false, platform).expect("Vulkan instance");
        let admitted = context
            .init_device_for_adapter(None, wanted, true)
            .expect("enumerated adapter initializes");
        assert_eq!(admitted.stable_id(), wanted);
    }
}
