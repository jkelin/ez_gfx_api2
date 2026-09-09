//! Vulkan frame command recording for compute, graphics, readback, and present.

use super::{HalError, VulkanEncoding, vk};
use ez_gfx_hal::COUNTER_BUFFER_ELEMENT_OFFSET;

pub(super) fn record_compute(
    encoding: &VulkanEncoding<'_>,
    dispatch: &super::super::NativeComputeDispatch<'_>,
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
        encoding.device.cmd_dispatch(
            encoding.command,
            dispatch.groups[0],
            dispatch.groups[1],
            dispatch.groups[2],
        );
    }
    Ok(())
}

pub(super) fn record_graphics(
    encoding: &VulkanEncoding<'_>,
    draw: &super::super::NativeDrawIndexed<'_>,
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
        encoding.device.cmd_draw_indexed_indirect_count(
            encoding.command,
            draw.indirect_buffer.buffer,
            COUNTER_BUFFER_ELEMENT_OFFSET,
            draw.indirect_buffer.buffer,
            0,
            draw.draw_count,
            20,
        );
    }
    Ok(())
}

pub(super) fn record_texture_readback(
    encoding: &VulkanEncoding<'_>,
    texture: &super::super::NativeTexture,
    readbacks: &[(u32, u32, u64, bool, super::super::NativeAllocation)],
    readback_index: &mut usize,
    pass_active: bool,
) -> Result<(), HalError> {
    // SAFETY: the source image and destination allocation remain live through command submission.
    // Layout comes from the graph's transfer-read barrier, which runs before this
    // action and restores shader readability afterward; the copy itself needs no barrier.
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
        // The graph emits the entry barrier but no matching restore: leave the
        // image shader-readable like the present path does for swapchain images.
        let to_shader = vk::ImageMemoryBarrier::default()
            .image(texture.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        encoding.device.cmd_pipeline_barrier(
            encoding.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            core::slice::from_ref(&to_shader),
        );
    }
    Ok(())
}

pub(super) fn record_present_readback(
    encoding: &VulkanEncoding<'_>,
    capture_presented: bool,
    readbacks: &[(u32, u32, u64, bool, super::super::NativeAllocation)],
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
