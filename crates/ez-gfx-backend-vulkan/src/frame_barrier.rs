//! Vulkan execution barriers for frame resources.

use super::frame::VulkanEncoding;
use super::{
    HalError, NativeContext, NativeFrameResource, NativeTexture, ResourceAccess, vk, vulkan_state,
};

impl NativeContext {
    pub(super) fn record_barrier(
        &self,
        encoding: &VulkanEncoding<'_>,
        barrier: &ez_gfx_hal::ExecutionBarrier,
        resource: &NativeFrameResource<'_>,
    ) -> Result<(), HalError> {
        let capabilities = self
            .adapter_info()
            .ok_or(HalError::NotReady)?
            .capabilities()
            .shader_stages;
        // SAFETY: all handles belong to the live context and ranges were validated during preflight.
        unsafe {
            let (src_stage, src_access, old_layout) = match barrier.before {
                Some(before) => vulkan_state(before, capabilities)?,
                None => (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                    vk::ImageLayout::UNDEFINED,
                ),
            };
            let (dst_stage, dst_access, new_layout) = vulkan_state(barrier.after, capabilities)?;
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
                NativeFrameResource::Texture(texture)
                | NativeFrameResource::RenderTarget(texture) => Self::record_texture_barrier(
                    encoding,
                    barrier,
                    texture,
                    false,
                    (src_stage, src_access, old_layout),
                    (dst_stage, dst_access, new_layout),
                )?,
                NativeFrameResource::RenderTargetDepth(texture) => Self::record_texture_barrier(
                    encoding,
                    barrier,
                    texture,
                    true,
                    (src_stage, src_access, old_layout),
                    (dst_stage, dst_access, new_layout),
                )?,
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

    fn record_texture_barrier(
        encoding: &VulkanEncoding<'_>,
        barrier: &ez_gfx_hal::ExecutionBarrier,
        texture: &NativeTexture,
        depth: bool,
        before: (vk::PipelineStageFlags, vk::AccessFlags, vk::ImageLayout),
        after: (vk::PipelineStageFlags, vk::AccessFlags, vk::ImageLayout),
    ) -> Result<(), HalError> {
        let ez_gfx_hal::ExecutionRange::Image(range) = barrier.range else {
            return Err(HalError::InvalidArgument);
        };
        let (src_stage, src_access, old_layout) = before;
        let (dst_stage, dst_access, new_layout) = after;
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: if depth {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            },
            base_mip_level: range.first_mip,
            level_count: range.mip_count,
            base_array_layer: range.first_layer,
            layer_count: range.layer_count,
        };
        let image = if depth {
            texture.depth.as_ref().ok_or(HalError::NotReady)?.image
        } else {
            texture.image
        };
        let sampled_image = vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(subresource_range);
        // SAFETY: both images belong to `texture`; the validated range covers
        // their single color layer and only the sampled image exposes mips.
        unsafe {
            encoding.device.cmd_pipeline_barrier(
                encoding.command,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                core::slice::from_ref(&sampled_image),
            );
            // MSAA storage is never sampled: it enters color-attachment
            if !depth
                && let Some(msaa) = texture.msaa.as_ref()
                && new_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            {
                let first_use = old_layout == vk::ImageLayout::UNDEFINED;
                let msaa_image = vk::ImageMemoryBarrier::default()
                    .src_access_mask(if first_use {
                        vk::AccessFlags::empty()
                    } else {
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    })
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(if first_use {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    })
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(msaa.image)
                    .subresource_range(subresource_range);
                encoding.device.cmd_pipeline_barrier(
                    encoding.command,
                    if first_use {
                        vk::PipelineStageFlags::TOP_OF_PIPE
                    } else {
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    },
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    core::slice::from_ref(&msaa_image),
                );
            }
        }
        Ok(())
    }
}
