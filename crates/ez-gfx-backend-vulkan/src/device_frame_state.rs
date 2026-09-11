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
