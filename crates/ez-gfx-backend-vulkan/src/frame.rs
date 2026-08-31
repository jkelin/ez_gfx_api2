use super::{
    AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, FRAMES_IN_FLIGHT, HalError,
    MemoryAllocator, MemoryClass, NativeContext, NativeFrameAction, NativeFrameResource,
    NativeSurface, QueueKind, ResourceAccess, map_allocation_hal, map_vk, vk, vulkan_state,
};

struct VulkanEncoding<'a> {
    device: &'a ash::Device,
    command: vk::CommandBuffer,
    image_index: u32,
    surface_image: vk::Image,
}

impl NativeContext {
    fn record_barrier(
        &self,
        encoding: &VulkanEncoding<'_>,
        barrier: &ez_gfx_hal::ExecutionBarrier,
        resource: &NativeFrameResource<'_>,
    ) -> Result<(), HalError> {
        // SAFETY: all handles belong to the live context and ranges were validated during preflight.
        unsafe {
            let (src_stage, src_access, old_layout) = barrier.before.map_or(
                (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                    vk::ImageLayout::UNDEFINED,
                ),
                vulkan_state,
            );
            let (dst_stage, dst_access, new_layout) = vulkan_state(barrier.after);
            match resource {
                NativeFrameResource::Buffer(allocation) => {
                    let ez_gfx_hal::ExecutionRange::Buffer(range) = barrier.range else {
                        return Err(HalError::InvalidArgument);
                    };
                    let buffer = vk::BufferMemoryBarrier::default()
                        .src_access_mask(src_access)
                        .dst_access_mask(dst_access)
                        .buffer(allocation.buffer)
                        .offset(range.offset)
                        .size(range.size);
                    encoding.device.cmd_pipeline_barrier(
                        encoding.command,
                        src_stage,
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        core::slice::from_ref(&buffer),
                        &[],
                    );
                }
                NativeFrameResource::Texture(texture) => {
                    let ez_gfx_hal::ExecutionRange::Image(range) = barrier.range else {
                        return Err(HalError::InvalidArgument);
                    };
                    let image = vk::ImageMemoryBarrier::default()
                        .src_access_mask(src_access)
                        .dst_access_mask(dst_access)
                        .old_layout(old_layout)
                        .new_layout(new_layout)
                        .image(texture.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: range.first_mip,
                            level_count: range.mip_count,
                            base_array_layer: range.first_layer,
                            layer_count: range.layer_count,
                        });
                    encoding.device.cmd_pipeline_barrier(
                        encoding.command,
                        src_stage,
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        core::slice::from_ref(&image),
                    );
                }
                NativeFrameResource::Surface => {
                    let first_present = barrier
                        .before
                        .is_some_and(|state| state.access == ResourceAccess::Present)
                        && !self
                            .swapchain_initialized
                            .get(encoding.image_index as usize)
                            .copied()
                            .ok_or(HalError::NativeFailure)?;
                    let image = vk::ImageMemoryBarrier::default()
                        .src_access_mask(if first_present {
                            vk::AccessFlags::empty()
                        } else {
                            src_access
                        })
                        .dst_access_mask(dst_access)
                        .old_layout(if first_present {
                            vk::ImageLayout::UNDEFINED
                        } else {
                            old_layout
                        })
                        .new_layout(new_layout)
                        .image(encoding.surface_image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    encoding.device.cmd_pipeline_barrier(
                        encoding.command,
                        if first_present {
                            vk::PipelineStageFlags::TOP_OF_PIPE
                        } else {
                            src_stage
                        },
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        core::slice::from_ref(&image),
                    );
                }
                NativeFrameResource::Depth => {
                    let depth = self.depth_target.as_ref().ok_or(HalError::NotReady)?;
                    let image = vk::ImageMemoryBarrier::default()
                        .src_access_mask(src_access)
                        .dst_access_mask(dst_access)
                        .old_layout(old_layout)
                        .new_layout(new_layout)
                        .image(depth.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    encoding.device.cmd_pipeline_barrier(
                        encoding.command,
                        src_stage,
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        core::slice::from_ref(&image),
                    );
                }
            }
        }
        Ok(())
    }
}

impl NativeContext {
    fn begin_render_pass(
        &self,
        encoding: &VulkanEncoding<'_>,
        pass: &ez_gfx_hal::ExecutionPass,
        extent: (u32, u32),
        pass_active: bool,
    ) -> Result<(), HalError> {
        // SAFETY: attachments belong to the current swapchain image and the pass was validated during preflight.
        unsafe {
            if pass_active
                || pass.colors.len() != 1
                || pass.samples != 1
                || pass.area[0]
                    .checked_add(pass.area[2])
                    .is_none_or(|end| end > extent.0)
                || pass.area[1]
                    .checked_add(pass.area[3])
                    .is_none_or(|end| end > extent.1)
            {
                return Err(HalError::InvalidArgument);
            }
            let color = vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain_views[encoding.image_index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(match pass.load {
                    AttachmentLoadOp::Load => vk::AttachmentLoadOp::LOAD,
                    AttachmentLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
                    AttachmentLoadOp::Discard => vk::AttachmentLoadOp::DONT_CARE,
                })
                .store_op(match pass.store {
                    AttachmentStoreOp::Store => vk::AttachmentStoreOp::STORE,
                    AttachmentStoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
                })
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                });
            let depth = pass.depth.map(|_| {
                let target = self.depth_target.as_ref().expect("preflighted depth");
                vk::RenderingAttachmentInfo::default()
                    .image_view(target.view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(match pass.load {
                        AttachmentLoadOp::Load => vk::AttachmentLoadOp::LOAD,
                        AttachmentLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
                        AttachmentLoadOp::Discard => vk::AttachmentLoadOp::DONT_CARE,
                    })
                    .store_op(match pass.store {
                        AttachmentStoreOp::Store => vk::AttachmentStoreOp::STORE,
                        AttachmentStoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
                    })
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    })
            });
            let mut rendering = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: i32::try_from(pass.area[0]).map_err(|_| HalError::InvalidArgument)?,
                        y: i32::try_from(pass.area[1]).map_err(|_| HalError::InvalidArgument)?,
                    },
                    extent: vk::Extent2D {
                        width: pass.area[2],
                        height: pass.area[3],
                    },
                })
                .layer_count(1)
                .color_attachments(core::slice::from_ref(&color));
            if let Some(depth) = depth.as_ref() {
                rendering = rendering.depth_attachment(depth);
            }
            encoding
                .device
                .cmd_begin_rendering(encoding.command, &rendering);
        }
        Ok(())
    }
}

fn record_compute(
    encoding: &VulkanEncoding<'_>,
    dispatch: &super::NativeComputeDispatch<'_>,
    public_set: Option<&vk::DescriptorSet>,
    texture_set: vk::DescriptorSet,
    pass_active: bool,
) -> Result<(), HalError> {
    // SAFETY: pipeline, descriptor sets, and command buffer belong to the live context.
    unsafe {
        if pass_active || dispatch.groups.contains(&0) {
            return Err(HalError::InvalidArgument);
        }
        let public = *public_set.ok_or(HalError::InvalidArgument)?;
        encoding.device.cmd_bind_pipeline(
            encoding.command,
            vk::PipelineBindPoint::COMPUTE,
            dispatch.pipeline.pipeline,
        );
        encoding.device.cmd_bind_descriptor_sets(
            encoding.command,
            vk::PipelineBindPoint::COMPUTE,
            dispatch.pipeline.layout,
            0,
            &[public, texture_set],
            &[],
        );
        if !dispatch.push_constants.is_empty() {
            encoding.device.cmd_push_constants(
                encoding.command,
                dispatch.pipeline.layout,
                vk::ShaderStageFlags::ALL,
                0,
                dispatch.push_constants,
            );
        }
        encoding.device.cmd_dispatch(
            encoding.command,
            dispatch.groups[0],
            dispatch.groups[1],
            dispatch.groups[2],
        );
    }
    Ok(())
}

fn record_graphics(
    encoding: &VulkanEncoding<'_>,
    draw: &super::NativeDrawIndexed<'_>,
    public_set: Option<&vk::DescriptorSet>,
    texture_set: vk::DescriptorSet,
    swapchain_extent: vk::Extent2D,
    pass_active: bool,
) -> Result<(), HalError> {
    // SAFETY: pipeline, descriptor sets, buffers, and command buffer belong to the live context.
    unsafe {
        if !pass_active {
            return Err(HalError::InvalidArgument);
        }
        let public = *public_set.ok_or(HalError::InvalidArgument)?;
        encoding.device.cmd_bind_pipeline(
            encoding.command,
            vk::PipelineBindPoint::GRAPHICS,
            draw.pipeline.pipeline,
        );
        encoding.device.cmd_bind_descriptor_sets(
            encoding.command,
            vk::PipelineBindPoint::GRAPHICS,
            draw.pipeline.layout,
            0,
            &[public, texture_set],
            &[],
        );
        encoding.device.cmd_set_viewport(
            encoding.command,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: f32::from(u16::try_from(draw.width).map_err(|_| HalError::InvalidArgument)?),
                height: f32::from(
                    u16::try_from(draw.height).map_err(|_| HalError::InvalidArgument)?,
                ),
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        encoding.device.cmd_set_scissor(
            encoding.command,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: swapchain_extent,
            }],
        );
        encoding.device.cmd_bind_index_buffer(
            encoding.command,
            draw.index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        if !draw.push_constants.is_empty() {
            encoding.device.cmd_push_constants(
                encoding.command,
                draw.pipeline.layout,
                vk::ShaderStageFlags::ALL,
                0,
                draw.push_constants,
            );
        }
        encoding.device.cmd_draw_indexed_indirect(
            encoding.command,
            draw.indirect_buffer.buffer,
            0,
            draw.draw_count,
            20,
        );
    }
    Ok(())
}

fn record_texture_readback(
    encoding: &VulkanEncoding<'_>,
    texture: &super::NativeTexture,
    readbacks: &[(u32, u32, u64, bool, super::NativeAllocation)],
    readback_index: &mut usize,
    pass_active: bool,
) -> Result<(), HalError> {
    // SAFETY: the source image and destination allocation remain live through command submission.
    unsafe {
        if pass_active {
            return Err(HalError::InvalidArgument);
        }
        let (width, height, _, _, allocation) = readbacks
            .get(*readback_index)
            .ok_or(HalError::InvalidArgument)?;
        encoding.device.cmd_copy_image_to_buffer(
            encoding.command,
            texture.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            allocation.buffer,
            &[vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: *width,
                    height: *height,
                    depth: 1,
                })],
        );
        *readback_index += 1;
    }
    Ok(())
}

fn record_present_readback(
    encoding: &VulkanEncoding<'_>,
    capture_presented: bool,
    readbacks: &[(u32, u32, u64, bool, super::NativeAllocation)],
    readback_index: &mut usize,
    pass_active: bool,
) -> Result<(), HalError> {
    // SAFETY: the acquired image and readback allocation remain live through command submission.
    unsafe {
        if pass_active {
            return Err(HalError::InvalidArgument);
        }
        if capture_presented {
            let (width, height, _, _, allocation) = readbacks
                .get(*readback_index)
                .ok_or(HalError::InvalidArgument)?;
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_copy = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(encoding.surface_image)
                .subresource_range(range);
            encoding.device.cmd_pipeline_barrier(
                encoding.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                core::slice::from_ref(&to_copy),
            );
            encoding.device.cmd_copy_image_to_buffer(
                encoding.command,
                encoding.surface_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                allocation.buffer,
                &[vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: *width,
                        height: *height,
                        depth: 1,
                    })],
            );
            let to_present = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::empty())
                .image(encoding.surface_image)
                .subresource_range(range);
            encoding.device.cmd_pipeline_barrier(
                encoding.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                core::slice::from_ref(&to_present),
            );
            *readback_index += 1;
        }
    }
    Ok(())
}

type FrameReadback = (u32, u32, u64, bool, super::NativeAllocation);
type FrameBindings = (Vec<FrameReadback>, Vec<Option<vk::DescriptorSet>>);

impl NativeContext {
    fn allocate_frame_bindings(
        &mut self,
        actions: &[NativeFrameAction<'_>],
        descriptor_pool: vk::DescriptorPool,
        extent: (u32, u32),
        capture_presented: bool,
    ) -> Result<FrameBindings, HalError> {
        let mut readbacks = Vec::new();
        for action in actions {
            let dimensions = match action {
                NativeFrameAction::TextureReadback { width, height, .. } => {
                    Some((*width, *height, false))
                }
                NativeFrameAction::Present if capture_presented => Some((extent.0, extent.1, true)),
                _ => None,
            };
            if let Some((width, height, surface_readback)) = dimensions {
                let created = (|| {
                    let size = u64::from(width)
                        .checked_mul(u64::from(height))
                        .and_then(|value| value.checked_mul(4))
                        .ok_or(HalError::InvalidArgument)?;
                    let request =
                        AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                            .map_err(|_| HalError::InvalidArgument)?;
                    let allocation = self.allocate(request).map_err(map_allocation_hal)?;
                    Ok((width, height, size, surface_readback, allocation))
                })();
                match created {
                    Ok(readback) => readbacks.push(readback),
                    Err(error) => {
                        for (_, _, _, _, allocation) in readbacks {
                            let _ = self.free(allocation);
                        }
                        return Err(error);
                    }
                }
            }
        }
        let mut public_sets = Vec::with_capacity(actions.len());
        for action in actions {
            let created = match action {
                NativeFrameAction::Compute(dispatch) => self
                    .create_public_descriptor_set(
                        descriptor_pool,
                        dispatch.pipeline,
                        dispatch.bindings,
                    )
                    .map(Some),
                NativeFrameAction::Graphics(draw) => self
                    .create_public_descriptor_set(descriptor_pool, draw.pipeline, draw.bindings)
                    .map(Some),
                _ => Ok(None),
            };
            match created {
                Ok(set) => public_sets.push(set),
                Err(error) => {
                    for (_, _, _, _, allocation) in readbacks {
                        let _ = self.free(allocation);
                    }
                    return Err(error);
                }
            }
        }
        Ok((readbacks, public_sets))
    }
}

impl NativeContext {
    fn collect_frame_readbacks(
        &mut self,
        readbacks: Vec<FrameReadback>,
        capture_presented: bool,
        surface: Option<&mut NativeSurface>,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        let mut outputs = Vec::with_capacity(readbacks.len());
        let mut remaining = readbacks.into_iter();
        while let Some((_, _, size, surface_readback, mut allocation)) = remaining.next() {
            let copied = (|| {
                self.invalidate(&mut allocation, 0, size)
                    .map_err(map_allocation_hal)?;
                Ok(self.mapped_slice(&allocation).map_err(map_allocation_hal)?
                    [..usize::try_from(size).map_err(|_| HalError::InvalidArgument)?]
                    .to_vec())
            })();
            let freed = self.free(allocation).map_err(map_allocation_hal);
            let mut pixels = match (copied, freed) {
                (Ok(pixels), Ok(())) => pixels,
                (Err(error), _) | (_, Err(error)) => {
                    for (_, _, _, _, allocation) in remaining {
                        let _ = self.free(allocation);
                    }
                    return Err(error);
                }
            };
            if matches!(
                self.swapchain_format,
                vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
            ) && surface_readback
            {
                for pixel in pixels.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
            }
            outputs.push(pixels);
        }
        if capture_presented && let Some(pixels) = outputs.last() {
            surface
                .ok_or(HalError::InvalidArgument)?
                .presented_rgba8
                .clone_from(pixels);
        }
        Ok(outputs)
    }
}

struct PreparedFrame {
    device: ash::Device,
    loader: Option<ash::khr::swapchain::Device>,
    swapchain: vk::SwapchainKHR,
    slot_index: usize,
    available: vk::Semaphore,
    finished_handle: vk::Semaphore,
    command_handle: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    fence: vk::Fence,
    queue: vk::Queue,
    texture_set: vk::DescriptorSet,
}

impl NativeContext {
    fn prepare_frame_slot(&mut self, uses_surface: bool) -> Result<PreparedFrame, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?.clone();
        let loader = if uses_surface {
            Some(
                self.swapchain_loader
                    .as_ref()
                    .ok_or(HalError::NotReady)?
                    .clone(),
            )
        } else {
            None
        };
        let swapchain = if uses_surface {
            self.swapchain.ok_or(HalError::NotReady)?
        } else {
            vk::SwapchainKHR::null()
        };
        let slot_index = self.frame_cursor;
        self.frame_cursor = (self.frame_cursor + 1) % FRAMES_IN_FLIGHT;
        let completed_slot = {
            let slot = self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NotReady)?;
            let completed = slot.in_flight;
            if completed {
                // SAFETY: `slot.in_flight` is set only after this `device` submits the slot with `slot.fence`, and the one-element fence slice lasts through `wait_for_fences`.
                unsafe { device.wait_for_fences(&[slot.fence], true, u64::MAX) }.map_err(map_vk)?;
                slot.in_flight = false;
            }
            // SAFETY: `slot.command_buffer` and `slot.descriptor_pool` are this `device`'s per-slot objects, and the slot is not in flight after the conditional wait, so resetting them cannot race GPU use.
            unsafe {
                device
                    .reset_command_buffer(slot.command_buffer, vk::CommandBufferResetFlags::empty())
                    .map_err(map_vk)?;
                device
                    .reset_descriptor_pool(
                        slot.descriptor_pool,
                        vk::DescriptorPoolResetFlags::empty(),
                    )
                    .map_err(map_vk)?;
            }
            completed
        };
        if completed_slot {
            self.complete_frame_slot(slot_index)
                .map_err(map_allocation_hal)?;
        }
        let slot = self.frame_slots.get(slot_index).ok_or(HalError::NotReady)?;
        let available = slot.image_available;
        let finished_handle = slot.render_finished;
        let command_handle = slot.command_buffer;
        let descriptor_pool = slot.descriptor_pool;
        let fence = slot.fence;
        let queue = self.graphics_queue.ok_or(HalError::NotReady)?;
        let texture_set = self.texture_descriptor_set.ok_or(HalError::NotReady)?;
        Ok(PreparedFrame {
            device,
            loader,
            swapchain,
            slot_index,
            available,
            finished_handle,
            command_handle,
            descriptor_pool,
            fence,
            queue,
            texture_set,
        })
    }
}

struct FrameRecordRequest<'a> {
    plan: &'a FramePlan,
    actions: &'a [NativeFrameAction<'a>],
    readbacks: &'a [FrameReadback],
    public_sets: &'a [Option<vk::DescriptorSet>],
    extent: (u32, u32),
    capture_presented: bool,
}

impl NativeContext {
    fn submit_recorded_frame(
        &mut self,
        prepared: &PreparedFrame,
        plan: &FramePlan,
        image_index: u32,
    ) -> Result<(), HalError> {
        // SAFETY: `prepared.command_handle` is recording; the queue, fence, semaphores, and conditional swapchain share `prepared.device`, and every submit/present slice remains allocated through its call.
        unsafe {
            prepared
                .device
                .end_command_buffer(prepared.command_handle)
                .map_err(map_vk)?;
            let mut wait_semaphores = Vec::with_capacity(2);
            let mut wait_values = Vec::with_capacity(2);
            let mut wait_stages = Vec::with_capacity(2);
            if plan.uses_surface {
                wait_semaphores.push(prepared.available);
                wait_values.push(0);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            if let Some(value) = plan.external_wait {
                wait_semaphores.push(self.transfer_timeline.ok_or(HalError::NotReady)?);
                wait_values.push(value);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            let signal_values = [0_u64];
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(core::slice::from_ref(&prepared.command_handle))
                .signal_semaphores(core::slice::from_ref(&prepared.finished_handle))
                .push_next(&mut timeline);
            prepared
                .device
                .reset_fences(&[prepared.fence])
                .map_err(map_vk)?;
            prepared
                .device
                .queue_submit(prepared.queue, &[submit], prepared.fence)
                .map_err(map_vk)?;
            if plan.presents {
                prepared
                    .loader
                    .as_ref()
                    .ok_or(HalError::InvalidArgument)?
                    .queue_present(
                        prepared.queue,
                        &vk::PresentInfoKHR::default()
                            .wait_semaphores(core::slice::from_ref(&prepared.finished_handle))
                            .swapchains(core::slice::from_ref(&prepared.swapchain))
                            .image_indices(core::slice::from_ref(&image_index)),
                    )
                    .map_err(map_vk)?;
            }
        }
        Ok(())
    }

    fn record_frame(
        &mut self,
        prepared: &PreparedFrame,
        request: &FrameRecordRequest<'_>,
    ) -> (Option<u32>, bool, Result<(), HalError>) {
        let FrameRecordRequest {
            plan,
            actions,
            readbacks,
            public_sets,
            extent,
            capture_presented,
        } = *request;
        let mut acquired_image_index = None;
        let mut submitted = false;
        let mut readback_index = 0_usize;
        let recorded = (|| {
            let (image_index, surface_image) = if plan.uses_surface {
                let loader = prepared.loader.as_ref().ok_or(HalError::InvalidArgument)?;
                // SAFETY: `prepared.swapchain` belongs to `loader`, and slot completion leaves the per-slot `prepared.available` semaphore available for `acquire_next_image` to signal.
                let (image_index, _) = unsafe {
                    loader.acquire_next_image(
                        prepared.swapchain,
                        u64::MAX,
                        prepared.available,
                        vk::Fence::null(),
                    )
                }
                .map_err(map_vk)?;
                acquired_image_index = Some(image_index);
                // SAFETY: `loader` was cloned from the context's swapchain loader and `prepared.swapchain` was selected from that same swapchain setup for `get_swapchain_images`.
                let images =
                    unsafe { loader.get_swapchain_images(prepared.swapchain) }.map_err(map_vk)?;
                let image = *images
                    .get(usize::try_from(image_index).map_err(|_| HalError::NativeFailure)?)
                    .ok_or(HalError::NativeFailure)?;
                (image_index, image)
            } else {
                (u32::MAX, vk::Image::null())
            };
            let encoding = VulkanEncoding {
                device: &prepared.device,
                command: prepared.command_handle,
                image_index,
                surface_image,
            };
            let mut pass_active = false;
            for (action_index, action) in actions.iter().enumerate() {
                match action {
                    NativeFrameAction::Wait(_) => {}
                    NativeFrameAction::Barrier { barrier, resource } => {
                        self.record_barrier(&encoding, barrier, resource)?;
                    }
                    NativeFrameAction::BeginPass(pass) => {
                        self.begin_render_pass(&encoding, pass, extent, pass_active)?;
                        pass_active = true;
                    }
                    NativeFrameAction::Compute(dispatch) => {
                        record_compute(
                            &encoding,
                            dispatch,
                            public_sets[action_index].as_ref(),
                            prepared.texture_set,
                            pass_active,
                        )?;
                    }
                    NativeFrameAction::Graphics(draw) => {
                        record_graphics(
                            &encoding,
                            draw,
                            public_sets[action_index].as_ref(),
                            prepared.texture_set,
                            self.swapchain_extent,
                            pass_active,
                        )?;
                    }
                    NativeFrameAction::TextureReadback { texture, .. } => {
                        record_texture_readback(
                            &encoding,
                            texture,
                            readbacks,
                            &mut readback_index,
                            pass_active,
                        )?;
                    }
                    // SAFETY: `prepared.command_handle` is recording, and `pass_active` is set only after its unmatched `cmd_begin_rendering`, so `cmd_end_rendering` closes that rendering scope.
                    NativeFrameAction::EndPass => unsafe {
                        if !pass_active {
                            return Err(HalError::InvalidArgument);
                        }
                        prepared.device.cmd_end_rendering(prepared.command_handle);
                        pass_active = false;
                    },
                    NativeFrameAction::Present => {
                        record_present_readback(
                            &encoding,
                            capture_presented,
                            readbacks,
                            &mut readback_index,
                            pass_active,
                        )?;
                    }
                }
            }
            if pass_active {
                return Err(HalError::InvalidArgument);
            }
            self.submit_recorded_frame(prepared, plan, image_index)?;
            submitted = true;
            Ok(())
        })();
        (acquired_image_index, submitted, recorded)
    }
}

struct FramePlan {
    uses_surface: bool,
    presents: bool,
    external_wait: Option<u64>,
}

fn validate_frame_plan(
    actions: &[NativeFrameAction<'_>],
    extent: (u32, u32),
    surface_available: bool,
    capture_presented: bool,
) -> Result<FramePlan, HalError> {
    let present_count = actions
        .iter()
        .filter(|action| matches!(action, NativeFrameAction::Present))
        .count();
    let presents = present_count == 1;
    let uses_surface = actions.iter().any(|action| {
        matches!(
            action,
            NativeFrameAction::BeginPass(_)
                | NativeFrameAction::Graphics(_)
                | NativeFrameAction::Present
                | NativeFrameAction::Barrier {
                    resource: NativeFrameResource::Surface | NativeFrameResource::Depth,
                    ..
                }
        )
    });
    if present_count > 1
        || uses_surface && !surface_available
        || (uses_surface || capture_presented) && !presents
    {
        return Err(HalError::InvalidArgument);
    }
    let external_wait = actions
        .iter()
        .filter_map(|action| match action {
            NativeFrameAction::Wait(token) if token.queue == QueueKind::Transfer => {
                Some(Ok(token.value))
            }
            NativeFrameAction::Wait(_) => Some(Err(HalError::InvalidArgument)),
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max();
    let mut pass_active = false;
    let mut saw_present = false;
    for action in actions {
        if saw_present {
            return Err(HalError::InvalidArgument);
        }
        match action {
            NativeFrameAction::Wait(_) => {}
            NativeFrameAction::Barrier { barrier, resource } => match (resource, barrier.range) {
                (
                    NativeFrameResource::Buffer(allocation),
                    ez_gfx_hal::ExecutionRange::Buffer(range),
                ) if range
                    .offset
                    .checked_add(range.size)
                    .is_some_and(|end| end <= allocation.allocation.size()) => {}
                (
                    NativeFrameResource::Texture(_)
                    | NativeFrameResource::Surface
                    | NativeFrameResource::Depth,
                    ez_gfx_hal::ExecutionRange::Image(_),
                ) => {}
                _ => return Err(HalError::InvalidArgument),
            },
            NativeFrameAction::BeginPass(pass) => {
                if pass_active
                    || pass.colors.len() != 1
                    || pass.samples != 1
                    || pass.area[0]
                        .checked_add(pass.area[2])
                        .is_none_or(|end| end > extent.0)
                    || pass.area[1]
                        .checked_add(pass.area[3])
                        .is_none_or(|end| end > extent.1)
                {
                    return Err(HalError::InvalidArgument);
                }
                pass_active = true;
            }
            NativeFrameAction::Compute(dispatch) => {
                if pass_active
                    || dispatch.groups.contains(&0)
                    || dispatch.push_constants.len() > 128
                    || !dispatch.push_constants.len().is_multiple_of(4)
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::Graphics(draw) => {
                let indirect_size = u64::from(draw.draw_count)
                    .checked_mul(20)
                    .ok_or(HalError::InvalidArgument)?;
                if !pass_active
                    || draw.width == 0
                    || draw.height == 0
                    || draw.width > extent.0
                    || draw.height > extent.1
                    || draw.draw_count == 0
                    || draw.indirect_buffer.allocation.size() < indirect_size
                    || draw.push_constants.len() > 128
                    || !draw.push_constants.len().is_multiple_of(4)
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::TextureReadback { width, height, .. } => {
                if pass_active || *width == 0 || *height == 0 {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::EndPass => {
                if !pass_active {
                    return Err(HalError::InvalidArgument);
                }
                pass_active = false;
            }
            NativeFrameAction::Present => {
                if pass_active {
                    return Err(HalError::InvalidArgument);
                }
                saw_present = true;
            }
        }
    }
    if pass_active {
        return Err(HalError::InvalidArgument);
    }
    Ok(FramePlan {
        uses_surface,
        presents,
        external_wait,
    })
}

impl NativeContext {
    /// Validates, records, submits, and optionally presents one complete frame plan.
    ///
    /// # Panics
    ///
    /// Panics if an acquired swapchain image has no corresponding view or an action has no corresponding descriptor-set entry.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid frame plan, unavailable required context resources, readback or descriptor setup failures, or failed Vulkan frame operations.
    pub fn execute_frame(
        &mut self,
        surface: Option<(&mut NativeSurface, (u32, u32))>,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        if actions.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        let (mut surface, extent) = match surface {
            Some((surface, extent)) if extent.0 != 0 && extent.1 != 0 => (Some(surface), extent),
            Some(_) => return Err(HalError::InvalidArgument),
            None => (None, (0, 0)),
        };
        let FramePlan {
            uses_surface,
            presents,
            external_wait,
        } = validate_frame_plan(actions, extent, surface.is_some(), capture_presented)?;
        if uses_surface {
            self.prepare_surface(
                surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                extent.0,
                extent.1,
            )?;
        }
        if actions.iter().any(|action| {
            matches!(
                action,
                NativeFrameAction::BeginPass(pass) if pass.depth.is_some()
            )
        }) {
            self.ensure_depth_target(self.swapchain_extent)?;
        }
        let prepared = self.prepare_frame_slot(uses_surface)?;

        let (readbacks, public_sets) = self.allocate_frame_bindings(
            actions,
            prepared.descriptor_pool,
            extent,
            capture_presented,
        )?;

        // SAFETY: `prepare_frame_slot` reset `prepared.command_handle` on `prepared.device` after any prior submission completed, and the `CommandBufferBeginInfo` temporary lasts through `begin_command_buffer`.
        if let Err(error) = unsafe {
            prepared.device.begin_command_buffer(
                prepared.command_handle,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        } {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(map_vk(error));
        }
        let plan = FramePlan {
            uses_surface,
            presents,
            external_wait,
        };
        let (acquired_image_index, submitted, recorded) = self.record_frame(
            &prepared,
            &FrameRecordRequest {
                plan: &plan,
                actions,
                readbacks: &readbacks,
                public_sets: &public_sets,
                extent,
                capture_presented,
            },
        );

        if submitted {
            self.frame_slots
                .get_mut(prepared.slot_index)
                .ok_or(HalError::NativeFailure)?
                .in_flight = true;
        }
        if submitted && (recorded.is_err() || !readbacks.is_empty()) {
            // SAFETY: when `submitted` is true, `prepared.fence` was passed to `prepared.device.queue_submit`, and the one-element fence slice lasts through `wait_for_fences`.
            unsafe {
                prepared
                    .device
                    .wait_for_fences(&[prepared.fence], true, u64::MAX)
            }
            .map_err(map_vk)?;
            self.frame_slots
                .get_mut(prepared.slot_index)
                .ok_or(HalError::NativeFailure)?
                .in_flight = false;
            self.complete_frame_slot(prepared.slot_index)
                .map_err(map_allocation_hal)?;
        }
        if let Err(error) = recorded {
            if acquired_image_index.is_some()
                && let Some(surface) = surface.as_deref_mut()
            {
                let _ = self.recreate_swapchain(surface, extent.0, extent.1);
            }
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(error);
        }
        if let Some(image_index) = acquired_image_index {
            *self
                .swapchain_initialized
                .get_mut(image_index as usize)
                .ok_or(HalError::NativeFailure)? = true;
        }
        self.collect_frame_readbacks(readbacks, capture_presented, surface)
    }
}
