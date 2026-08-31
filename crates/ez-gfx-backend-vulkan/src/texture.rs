use super::{
    Allocation, AllocationCreateDesc, AllocationError, AllocationRequest, AllocationScheme,
    CompletionToken, DeferredResource, ImageMip, MemoryAllocator, MemoryClass, MemoryLocation,
    NativeAllocation, NativeContext, NativeTexture, QueueKind, SAMPLER_DESCRIPTOR_BINDING,
    TEXTURE_DESCRIPTOR_BINDING, TEXTURE_DESCRIPTOR_CAPACITY, TextureSamplerDesc, map_allocation_vk,
    map_allocator, map_vk, sampler_create_info, validate_rgba8_mips, vk,
};

fn publish_texture(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    binding: u32,
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Allocation,
) -> NativeTexture {
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
    // SAFETY: the descriptor set and image objects belong to this live device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
    NativeTexture {
        image,
        view,
        allocation,
        sampler,
        binding,
    }
}

struct TextureUpload<'a> {
    device: &'a ash::Device,
    mips: &'a [ImageMip<'a>],
    pool: vk::CommandPool,
    queue: vk::Queue,
    semaphore: vk::Semaphore,
    image: vk::Image,
    total: u64,
}

impl NativeContext {
    fn upload_texture_mips(
        &mut self,
        mut upload: NativeAllocation,
        request: &TextureUpload<'_>,
    ) -> Result<Vec<CompletionToken>, AllocationError> {
        let TextureUpload {
            device,
            mips,
            pool,
            queue,
            semaphore,
            image,
            total,
        } = *request;
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
                // SAFETY: `pool` is this device's transfer command pool, host access is serialized by `&mut self`, and the allocate-info storage lives through `allocate_command_buffers`.
                let command = unsafe {
                    device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                }
                .map_err(|error| map_allocation_vk(map_vk(error)))?
                .into_iter()
                .next()
                .ok_or(AllocationError::NativeFailure)?;
                commands.push(command);
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
                    base_mip_level: u32::try_from(level)
                        .map_err(|_| AllocationError::NativeFailure)?,
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
                // SAFETY: `command` is recording, `image` memory is bound and retained through submission, and the `to_copy` barrier slice lives through `cmd_pipeline_barrier`.
                unsafe {
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        core::slice::from_ref(&to_copy),
                    );
                };
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(source_offset)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: u32::try_from(level)
                            .map_err(|_| AllocationError::NativeFailure)?,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: mip.width,
                        height: mip.height,
                        depth: 1,
                    });
                // SAFETY: `command` is recording, `upload.buffer` and `image` are retained until queue idle, and `region` storage lives through `cmd_copy_buffer_to_image`.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command,
                        upload.buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        core::slice::from_ref(&region),
                    );
                };
                let to_shader = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                // SAFETY: `command` is recording, `image` is retained through execution, and `to_shader` lives through barrier recording, after which the command remains recording for `end_command_buffer`.
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
                // SAFETY: `command` is executable, queue access is serialized by `&mut self`, submit-chain storage lives through `queue_submit`, and the upload and image are retained until queue idle.
                unsafe {
                    device.queue_submit(queue, core::slice::from_ref(&submit), vk::Fence::null())
                }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
                completions.push(
                    CompletionToken::new(QueueKind::Transfer, value)
                        .map_err(|_| AllocationError::NativeFailure)?,
                );
                source_offset = source_offset
                    .checked_add(
                        u64::try_from(mip.bytes.len())
                            .map_err(|_| AllocationError::NativeFailure)?,
                    )
                    .ok_or(AllocationError::NativeFailure)?;
            }
            // SAFETY: host access to `queue` is serialized by `&mut self`, and `queue_wait_idle` completes submitted command-buffer use before cleanup.
            unsafe { device.queue_wait_idle(queue) }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
            Ok(completions)
        })();
        if submitted.is_err() {
            // SAFETY: host access to `queue` is serialized by `&mut self`, and `queue_wait_idle` is issued before any submitted command buffer or upload storage is freed.
            let _ = unsafe { device.queue_wait_idle(queue) };
        }
        if !commands.is_empty() {
            // SAFETY: every `command` was allocated from `pool`; the normal or error wait path has finished queue use, and the slice lives through `free_command_buffers`.
            unsafe { device.free_command_buffers(pool, &commands) };
        }
        let upload_freed = self.free(upload);
        match (submitted, upload_freed) {
            (Ok(completions), Ok(())) => Ok(completions),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }
}

impl NativeContext {
    pub(super) fn destroy_unpublished_texture(
        &mut self,
        device: &ash::Device,
        image: vk::Image,
        view: Option<vk::ImageView>,
        sampler: Option<vk::Sampler>,
        allocation: Allocation,
    ) {
        // SAFETY: the unpublished sampler, view, and image have no descriptor references or pending queue use, and the image allocation is freed only after these destroy operations.
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

    fn create_texture_image(
        &mut self,
        device: &ash::Device,
        width: u32,
        height: u32,
        mip_count: u32,
    ) -> Result<(vk::Image, Allocation), AllocationError> {
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
        // SAFETY: `create` is initialized without dangling extension pointers and its stack storage lives through `create_image`; no allocation callbacks are supplied.
        let image = unsafe { device.create_image(&create, None) }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
        // SAFETY: `image` is the undestroyed result of `device.create_image` immediately above, so `get_image_memory_requirements` may query it before binding.
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
                // SAFETY: allocation failed before `image` was bound, published, or submitted, so `destroy_image` cannot race pending device use.
                unsafe { device.destroy_image(image, None) };
                return Err(map_allocator(&error));
            }
        };
        if let Err(error) =
            // SAFETY: `allocation` was created from this `image`'s requirements, so its memory type, aligned offset, and range satisfy `bind_image_memory` and remain allocated through the call.
            unsafe {
                device.bind_image_memory(image, allocation.memory(), allocation.offset())
            }
        {
            // SAFETY: `bind_image_memory` failed before publication or submission, so `image` has no pending use and is destroyed before `allocation` is freed.
            unsafe { device.destroy_image(image, None) };
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator initialized")
                .free(allocation);
            return Err(map_allocation_vk(map_vk(error)));
        }
        Ok((image, allocation))
    }

    /// Creates an RGBA8 mip chain and submits each level under a distinct timeline value.
    ///
    /// # Errors
    ///
    /// Returns an error if the mip chain or binding is invalid, anisotropy or required context state is unavailable, size or timeline arithmetic overflows, or image creation, upload allocation, synchronization, or Vulkan submission fails.
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
        let (image, allocation) = self.create_texture_image(&device, width, height, mip_count)?;
        // SAFETY: `image` is a bound 2D RGBA8 image with `mip_count` levels, and the initialized view-info storage selects exactly that color range for `create_image_view`.
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
        // SAFETY: `sampler_info` is initialized and lives through `create_sampler`, and anisotropy is requested only after the enabled-feature check above.
        let sampler = match unsafe { device.create_sampler(&sampler_info, None) } {
            Ok(sampler) => sampler,
            Err(error) => {
                self.destroy_unpublished_texture(&device, image, Some(view), None, allocation);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let upload = match self.allocate(upload_request) {
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
        let completions = match self.upload_texture_mips(
            upload,
            &TextureUpload {
                device: &device,
                mips,
                pool,
                queue,
                semaphore,
                image,
                total,
            },
        ) {
            Ok(completions) => completions,
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
        let texture = publish_texture(
            &device,
            descriptor_set,
            binding,
            image,
            view,
            sampler,
            allocation,
        );
        Ok((texture, completions))
    }

    /// Defers texture destruction until every referencing frame completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the texture cannot be queued for deferred destruction.
    pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
        self.defer_resource(DeferredResource::Texture(texture))
    }

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
        // SAFETY: `pool` is this device's transfer command pool, host access is serialized by `&mut self`, and the allocate-info storage lives through `allocate_command_buffers`.
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
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .signal_semaphore_values(core::slice::from_ref(&value));
            let submit = vk::SubmitInfo::default()
                .command_buffers(core::slice::from_ref(&command))
                .signal_semaphores(core::slice::from_ref(&semaphore))
                .push_next(&mut timeline);
            // SAFETY: `command` is executable, queue access is serialized by `&mut self`, submit-chain storage lives through `queue_submit`, and image and readback storage are retained until the wait.
            unsafe {
                device.queue_submit(queue, core::slice::from_ref(&submit), vk::Fence::null())
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            submitted = true;
            self.next_transfer_value = next_value;
            // SAFETY: the preceding successful submit enqueued signal `value` on timeline `semaphore`, and the equal-length wait-info slices live through `wait_semaphores`.
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
        let freed = self.free(readback);
        match (result, freed) {
            (Ok(pixels), Ok(())) => Ok(pixels),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }
}
