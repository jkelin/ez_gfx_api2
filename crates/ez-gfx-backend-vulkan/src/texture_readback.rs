use super::{
    AllocationError, AllocationRequest, MemoryAllocator, MemoryClass, NativeContext, NativeTexture,
    map_allocation_vk, map_vk, vk,
};

impl NativeContext {
    /// Copies a shader-readable image to host-visible memory and returns tightly packed RGBA8 pixels.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions, allocation, synchronization, or the native copy fails.
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
        let queue = self.graphics_queue.ok_or(AllocationError::NativeFailure)?;
        let family = self
            .graphics_queue_family
            .ok_or(AllocationError::NativeFailure)?;
        // SAFETY: `family` belongs to this initialized device and create-info storage spans the call.
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
        }
        .map_err(|error| map_allocation_vk(map_vk(error)))?;
        let mut readback = match self.allocate(
            AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                .map_err(|_| AllocationError::ZeroSize)?,
        ) {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: no command buffer was allocated from this new pool.
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(error);
            }
        };
        // SAFETY: this pool is graphics-family local and exclusively owned by this call.
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
                // SAFETY: allocation failed before any command buffer became pending.
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let mut submitted = false;
        let graphics_queue_lock = self.graphics_queue_lock.clone();
        let result = (|| {
            // SAFETY: `command` was just allocated from `pool` in the initial state, host access is serialized by `&mut self`, and the begin-info storage lives through `begin_command_buffer`.
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
            // SAFETY: `command` is recording, `texture.image` is retained by the shared texture reference, and the `to_copy` barrier slice lives through `cmd_pipeline_barrier`.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    core::slice::from_ref(&to_copy),
                );
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
            // SAFETY: `command` is recording, the preceding barrier establishes transfer-source layout, the image and `readback.buffer` are retained through completion, and `region` lives through the call.
            unsafe {
                device.cmd_copy_image_to_buffer(
                    command,
                    texture.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    core::slice::from_ref(&region),
                );
            };
            let to_shader = vk::ImageMemoryBarrier::default()
                .image(texture.image)
                .subresource_range(range)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            // SAFETY: `command` is recording, `texture.image` is retained through execution, and `to_shader` lives through barrier recording, after which the command remains recording for `end_command_buffer`.
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
            let submit = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command));
            let _queue_guard = graphics_queue_lock.lock();
            // SAFETY: the closed command buffer and graphics queue belong to this live device.
            unsafe {
                device.queue_submit(queue, core::slice::from_ref(&submit), vk::Fence::null())
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            submitted = true;
            // SAFETY: the graphics queue remains live and locked through this wait.
            unsafe { device.queue_wait_idle(queue) }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
            self.invalidate(&mut readback, 0, size)?;
            Ok(self.mapped_slice(&readback)?
                [..usize::try_from(size).map_err(|_| AllocationError::NativeFailure)?]
                .to_vec())
        })();
        if submitted && result.is_err() {
            // SAFETY: `submitted` means `queue_submit` succeeded; queue access is serialized by `&mut self`, and this idle wait precedes freeing the command and readback storage.
            let _ = unsafe { device.queue_wait_idle(queue) };
        }
        // SAFETY: `command` came from `pool`; unsubmitted paths never made it pending, submitted paths wait for completion, and the slice lives through `free_command_buffers`.
        unsafe { device.free_command_buffers(pool, &[command]) };
        // SAFETY: the only command buffer was freed above and no pool work remains pending.
        unsafe { device.destroy_command_pool(pool, None) };
        let freed = self.free(readback);
        match (result, freed) {
            (Ok(pixels), Ok(())) => Ok(pixels),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }
}
