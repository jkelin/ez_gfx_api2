use super::{
    AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, CompletionToken, FRAMES_IN_FLIGHT,
    HalError, MemoryAllocator, MemoryClass, NativeContext, NativeFrameAction, NativeFrameResource,
    NativeSurface, NativeTexture, PresentationMode, QueueKind, ResourceAccess, map_allocation_hal,
    map_vk, vk, vulkan_state,
};
use arrayvec::ArrayVec;
#[path = "frame_plan.rs"]
mod plan;
#[path = "frame_record.rs"]
mod record;
use plan::{FramePlan, validate_frame_plan};

type FrameSurface<'a> = (&'a mut NativeSurface, (u32, u32), PresentationMode);
type ResolvedFrameSurface<'a> = (Option<&'a mut NativeSurface>, (u32, u32), PresentationMode);

fn validate_surface_request(
    surface: Option<FrameSurface<'_>>,
) -> Result<ResolvedFrameSurface<'_>, HalError> {
    match surface {
        Some((surface, extent, mode)) if extent.0 != 0 && extent.1 != 0 => {
            Ok((Some(surface), extent, mode))
        }
        Some(_) => Err(HalError::InvalidArgument),
        None => Ok((None, (0, 0), PresentationMode::Fifo)),
    }
}

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
        before: (vk::PipelineStageFlags, vk::AccessFlags, vk::ImageLayout),
        after: (vk::PipelineStageFlags, vk::AccessFlags, vk::ImageLayout),
    ) -> Result<(), HalError> {
        let ez_gfx_hal::ExecutionRange::Image(range) = barrier.range else {
            return Err(HalError::InvalidArgument);
        };
        let (src_stage, src_access, old_layout) = before;
        let (dst_stage, dst_access, new_layout) = after;
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: range.first_mip,
            level_count: range.mip_count,
            base_array_layer: range.first_layer,
            layer_count: range.layer_count,
        };
        let sampled_image = vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(texture.image)
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
            // layout before rendering and stays there after resolve.
            if let Some(msaa) = texture.msaa.as_ref()
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

impl NativeContext {
    fn begin_render_pass(
        &self,
        encoding: &VulkanEncoding<'_>,
        pass: &ez_gfx_hal::ExecutionPass,
        colors: &[super::PassAttachment<'_>],
        extent: (u32, u32),
        pass_active: bool,
    ) -> Result<(), HalError> {
        // SAFETY: the surface attachment belongs to the current swapchain image,
        // render-target attachments belong to live context images, and the pass
        // was validated during preflight.
        unsafe {
            if pass_active || pass.colors.len() != 1 || colors.len() != 1 {
                return Err(HalError::InvalidArgument);
            }
            if !matches!(pass.samples, 1 | 2 | 4 | 8) {
                return Err(HalError::InvalidArgument);
            }
            let attachment = colors.first().ok_or(HalError::InvalidArgument)?;
            // Textures, buffers, and depth images are never color attachments.
            // A multisampled target renders into its MSAA storage and resolves
            // into the sampled image; single-sample targets render directly.
            let (view, resolve_view, clear) = match attachment.resource {
                super::NativeFrameResource::Surface => {
                    if pass.samples != 1 {
                        return Err(HalError::InvalidArgument);
                    }
                    (
                        self.swapchain_views[encoding.image_index as usize],
                        None,
                        attachment.clear,
                    )
                }
                super::NativeFrameResource::RenderTarget(texture) => {
                    if pass.depth.is_some() {
                        return Err(HalError::InvalidArgument);
                    }
                    if let Some(msaa) = texture.msaa.as_ref() {
                        if msaa.samples != pass.samples {
                            return Err(HalError::InvalidArgument);
                        }
                        (msaa.view, Some(texture.view), attachment.clear)
                    } else {
                        if pass.samples != 1 {
                            return Err(HalError::InvalidArgument);
                        }
                        (texture.view, None, attachment.clear)
                    }
                }
                _ => return Err(HalError::InvalidArgument),
            };
            let (target_width, target_height) = match attachment.resource {
                super::NativeFrameResource::Surface => extent,
                super::NativeFrameResource::RenderTarget(texture) => {
                    (texture.width, texture.height)
                }
                _ => return Err(HalError::InvalidArgument),
            };
            if pass.area[0]
                .checked_add(pass.area[2])
                .is_none_or(|end| end > target_width)
                || pass.area[1]
                    .checked_add(pass.area[3])
                    .is_none_or(|end| end > target_height)
            {
                return Err(HalError::InvalidArgument);
            }
            let mut color = vk::RenderingAttachmentInfo::default()
                .image_view(view)
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
                    color: vk::ClearColorValue { float32: clear },
                });
            // A resolve view is present exactly when the attachment renders
            // multisampled; the pass then averages into the sampled image.
            if let Some(resolve) = resolve_view {
                color = color
                    .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                    .resolve_image_view(resolve)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            }
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

type FrameReadback = (u32, u32, u64, bool, super::NativeAllocation);
/// Allocated public descriptor sets for pipeline actions, in ascending action
/// order; every other action needs no entry.
type FrameBindings = (Vec<FrameReadback>, Vec<(u32, vk::DescriptorSet)>);

/// Returns the public descriptor set for one action, if it uses a pipeline.
///
/// Entries are recorded in ascending action order, so binary search is valid.
fn public_set(sets: &[(u32, vk::DescriptorSet)], action: usize) -> Option<&vk::DescriptorSet> {
    // Action counts always fit `u32`; the fallback only guards the conversion.
    let action = u32::try_from(action).ok()?;
    sets.binary_search_by_key(&action, |(action, _)| *action)
        .ok()
        .map(|index| &sets[index].1)
}
impl NativeContext {
    /// Returns public-set result storage to its frame slot, keeping capacity.
    ///
    /// Sets are pool-allocated per frame and die with the pool reset, so only
    /// the emptied shell is retained. A missing slot is unreachable after
    /// preparation; dropping instead preserves behavior and only loses capacity.
    fn restore_public_sets(&mut self, slot_index: usize, sets: Vec<(u32, vk::DescriptorSet)>) {
        if let Some(slot) = self.frame_slots.get_mut(slot_index) {
            slot.public_sets_scratch = sets;
        }
    }
    /// Returns descriptor construction shells to their frame slot, keeping capacity.
    ///
    /// Shells hold no GPU work; they are cleared per pipeline action and only
    /// retain high-water storage. A missing slot is unreachable after
    /// preparation; dropping instead preserves behavior and only loses capacity.
    fn restore_descriptor_scratch(
        &mut self,
        slot_index: usize,
        infos: Vec<vk::DescriptorBufferInfo>,
        writes: Vec<vk::WriteDescriptorSet<'static>>,
    ) {
        if let Some(slot) = self.frame_slots.get_mut(slot_index) {
            slot.descriptor_info_scratch = infos;
            slot.descriptor_write_scratch = writes;
        }
    }

    /// Grows a retired slot's descriptor pool to the preflighted need.
    ///
    /// The slot just passed the fence wait and pool reset in `prepare_frame_slot`
    /// and nothing has submitted from it yet, so destroying and recreating its
    /// pool cannot race GPU use. Capacities and high-water marks update only on
    /// successful recreation; a failed recreation leaves the old pool destroyed
    /// and reports the error, and the next frame retries from current capacity.
    /// Crate-visible for the descriptor-count test alongside production use.
    pub(super) fn ensure_descriptor_capacity(
        &mut self,
        slot_index: usize,
        needed_sets: u32,
        needed_descriptors: u32,
    ) -> Result<vk::DescriptorPool, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?.clone();
        let slot = self
            .frame_slots
            .get_mut(slot_index)
            .ok_or(HalError::NotReady)?;
        if needed_sets <= slot.descriptor_sets_capacity
            && needed_descriptors <= slot.descriptor_count_capacity
        {
            return Ok(slot.descriptor_pool);
        }
        let (sets, descriptors) = super::memory::next_descriptor_capacity(
            slot.descriptor_sets_capacity,
            slot.descriptor_count_capacity,
            needed_sets,
            needed_descriptors,
        )?;
        // The replacement is created before the old pool is destroyed, so a
        // failed growth leaves a working pool behind and the next frame simply
        // retries; capacities update only on success.
        let pool = super::memory::create_descriptor_pool(&device, sets, descriptors)?;
        // SAFETY: the slot is retired (fence-waited and pool-reset) with no
        // submission yet this frame, so no command references the old pool.
        unsafe { device.destroy_descriptor_pool(slot.descriptor_pool, None) };
        slot.descriptor_pool = pool;
        slot.descriptor_sets_capacity = sets;
        slot.descriptor_count_capacity = descriptors;
        Ok(pool)
    }

    fn allocate_frame_bindings(
        &mut self,
        slot_index: usize,
        actions: &(impl super::NativeFrameActionSource + ?Sized),
        extent: (u32, u32),
        capture_presented: bool,
    ) -> Result<FrameBindings, HalError> {
        let mut readbacks = Vec::new();
        let allocation_result = actions.visit(&mut |_, action| {
            let dimensions = match action {
                NativeFrameAction::TextureReadback { width, height, .. } => {
                    Some((*width, *height, false))
                }
                NativeFrameAction::Present if capture_presented => Some((extent.0, extent.1, true)),
                _ => None,
            };
            if let Some((width, height, surface_readback)) = dimensions {
                let size = u64::from(width)
                    .checked_mul(u64::from(height))
                    .and_then(|value| value.checked_mul(4))
                    .ok_or(HalError::InvalidArgument)?;
                let request = AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                    .map_err(|_| HalError::InvalidArgument)?;
                let allocation = self.allocate(request).map_err(map_allocation_hal)?;
                readbacks.push((width, height, size, surface_readback, allocation));
            }
            Ok(())
        });
        if let Err(error) = allocation_result {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(error);
        }
        // Preflight: one set per pipeline action, descriptors from its bindings.
        // Counts are tiny frameside values; saturation guards pathological plans.
        let mut needed_sets = 0_u32;
        let mut needed_descriptors = 0_u32;
        actions.visit(&mut |_, action| {
            let bindings = match action {
                NativeFrameAction::Compute(dispatch) => Some(dispatch.bindings.len()),
                NativeFrameAction::Graphics(draw) => Some(draw.bindings.len()),
                NativeFrameAction::Mesh(draw) => Some(draw.bindings.len()),
                _ => None,
            };
            if let Some(count) = bindings {
                needed_sets = needed_sets.saturating_add(1);
                needed_descriptors =
                    needed_descriptors.saturating_add(u32::try_from(count).unwrap_or(u32::MAX));
            }
            Ok(())
        })?;
        let descriptor_pool =
            self.ensure_descriptor_capacity(slot_index, needed_sets, needed_descriptors)?;
        // Slot scratch is retired: the pool was just reset after the fence wait
        // and nothing has submitted from this slot yet this frame.
        let mut public_sets = core::mem::take(
            &mut self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NotReady)?
                .public_sets_scratch,
        );
        public_sets.clear();
        let mut descriptor_infos = core::mem::take(
            &mut self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NotReady)?
                .descriptor_info_scratch,
        );
        let mut descriptor_writes = core::mem::take(
            &mut self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NotReady)?
                .descriptor_write_scratch,
        );
        let mut used_sets = 0_u32;
        let mut used_descriptors = 0_u32;
        let descriptor_result = actions.visit(&mut |action_index, action| {
            let bindings = match action {
                NativeFrameAction::Compute(dispatch) => {
                    Some((dispatch.pipeline, dispatch.bindings))
                }
                NativeFrameAction::Graphics(draw) => Some((draw.pipeline, draw.bindings)),
                NativeFrameAction::Mesh(draw) => Some((draw.pipeline, draw.bindings)),
                _ => None,
            };
            let Some((pipeline, bindings)) = bindings else {
                return Ok(());
            };
            let action = u32::try_from(action_index).map_err(|_| HalError::InvalidArgument)?;
            let set = self.create_public_descriptor_set(
                descriptor_pool,
                pipeline,
                bindings,
                &mut descriptor_infos,
                &mut descriptor_writes,
            )?;
            public_sets.push((action, set));
            used_sets = used_sets.saturating_add(1);
            used_descriptors =
                used_descriptors.saturating_add(u32::try_from(bindings.len()).unwrap_or(u32::MAX));
            Ok(())
        });
        if let Err(error) = descriptor_result {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            self.restore_public_sets(slot_index, public_sets);
            self.restore_descriptor_scratch(slot_index, descriptor_infos, descriptor_writes);
            return Err(error);
        }
        self.restore_descriptor_scratch(slot_index, descriptor_infos, descriptor_writes);
        if let Some(slot) = self.frame_slots.get_mut(slot_index) {
            // High-water records proven need; growth decisions compare against it
            // indirectly through capacity, which only rises on preflight misses.
            slot.descriptor_sets_high_water = slot.descriptor_sets_high_water.max(used_sets);
            slot.descriptor_count_high_water =
                slot.descriptor_count_high_water.max(used_descriptors);
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
    command_handle: vk::CommandBuffer,
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
            // SAFETY: `slot.command_buffer` is this `device`'s per-slot object, and the slot is not in flight after the conditional wait, so resetting it cannot race GPU use. A null descriptor pool was deferred and needs no reset; the preflight creates it on first need.
            unsafe {
                device
                    .reset_command_buffer(slot.command_buffer, vk::CommandBufferResetFlags::empty())
                    .map_err(map_vk)?;
                if slot.descriptor_pool != vk::DescriptorPool::null() {
                    device
                        .reset_descriptor_pool(
                            slot.descriptor_pool,
                            vk::DescriptorPoolResetFlags::empty(),
                        )
                        .map_err(map_vk)?;
                }
            }
            completed
        };
        if completed_slot {
            self.complete_frame_slot(slot_index)
                .map_err(map_allocation_hal)?;
        }
        let slot = self.frame_slots.get(slot_index).ok_or(HalError::NotReady)?;
        let available = slot.image_available;
        let command_handle = slot.command_buffer;
        let fence = slot.fence;
        let queue = self.graphics_queue.ok_or(HalError::NotReady)?;
        let texture_set = self.texture_descriptor_set.ok_or(HalError::NotReady)?;
        Ok(PreparedFrame {
            device,
            loader,
            swapchain,
            slot_index,
            available,
            command_handle,
            fence,
            queue,
            texture_set,
        })
    }
}

struct FrameRecordRequest<'a> {
    plan: &'a FramePlan,
    actions: &'a dyn super::NativeFrameActionSource,
    readbacks: &'a [FrameReadback],
    public_sets: &'a [(u32, vk::DescriptorSet)],
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
        // A graphics wait must not precede the worker's graphics release/acquire submissions:
        // otherwise that wait would block the queue needed to produce its completion value.
        for queue in [QueueKind::Transfer, QueueKind::TextureTransfer] {
            if let Some(required) = plan
                .external_wait
                .iter()
                .filter(|token| token.queue == queue)
                .map(|token| token.value)
                .max()
            {
                let worker = if queue == QueueKind::Transfer {
                    self.transfer_worker.as_ref()
                } else {
                    self.texture_worker.as_ref()
                };
                worker
                    .ok_or(HalError::NotReady)?
                    .flush_through(required)
                    .map_err(ez_gfx_hal::TransferWorkerError::to_hal_error)?;
            }
        }
        // Reacquiring this image retires its previous presentation semaphore wait. A frame
        // fence alone cannot prove that presentation has consumed a slot-owned semaphore.
        let finished = if plan.presents {
            Some(
                *self
                    .swapchain_finished
                    .get(image_index as usize)
                    .ok_or(HalError::NativeFailure)?,
            )
        } else {
            None
        };
        // SAFETY: `prepared.command_handle` is recording; the queue, fence, semaphores, and conditional swapchain share `prepared.device`, and every submit/present slice remains allocated through its call.
        unsafe {
            prepared
                .device
                .end_command_buffer(prepared.command_handle)
                .map_err(map_vk)?;
            let _queue_guard = self.graphics_queue_lock.lock();
            let mut wait_semaphores = ArrayVec::<_, 3>::new();
            let mut wait_values = ArrayVec::<_, 3>::new();
            let mut wait_stages = ArrayVec::<_, 3>::new();
            if plan.uses_surface {
                wait_semaphores.push(prepared.available);
                wait_values.push(0);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            for token in &plan.external_wait {
                let semaphore = match token.queue {
                    QueueKind::Transfer => self.transfer_timeline,
                    QueueKind::TextureTransfer => self.texture_timeline,
                    _ => None,
                }
                .ok_or(HalError::NotReady)?;
                wait_semaphores.push(semaphore);
                wait_values.push(token.value);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            let signal_values = [0_u64];
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values[..finished.as_slice().len()]);
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(core::slice::from_ref(&prepared.command_handle))
                .signal_semaphores(finished.as_slice())
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
                            .wait_semaphores(finished.as_slice())
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
                // Images are cached at swapchain creation/recreation in presentation
                // order; `prepared.swapchain` was read from the context at prepare
                // time and nothing recreates it before this fetch, so the cache is
                // current. The copy ends the borrow before encoding begins.
                debug_assert_eq!(
                    prepared.swapchain,
                    self.swapchain.unwrap_or(vk::SwapchainKHR::null())
                );
                let image = *self
                    .swapchain_images
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
            actions.visit(&mut |action_index, action| {
                match action {
                    NativeFrameAction::Wait(_) => {}
                    NativeFrameAction::Barrier { barrier, resource } => {
                        self.record_barrier(&encoding, barrier, resource)?;
                    }
                    NativeFrameAction::BeginPass { pass, colors } => {
                        self.begin_render_pass(&encoding, pass, colors, extent, pass_active)?;
                        pass_active = true;
                    }
                    NativeFrameAction::Compute(dispatch) => {
                        record::record_compute(
                            &encoding,
                            dispatch,
                            public_set(public_sets, action_index),
                            prepared.texture_set,
                            pass_active,
                        )?;
                    }
                    NativeFrameAction::Graphics(draw) => {
                        record::record_graphics(
                            &encoding,
                            draw,
                            public_set(public_sets, action_index),
                            prepared.texture_set,
                            pass_active,
                        )?;
                    }
                    NativeFrameAction::Mesh(draw) => {
                        // Mesh support was rejected at pipeline creation; a missing
                        // loader or limits here means the pipeline never existed.
                        let loader = self
                            .mesh_shader_loader
                            .as_ref()
                            .ok_or(HalError::Unsupported)?;
                        let limits = self
                            .mesh_shader_limits
                            .as_ref()
                            .ok_or(HalError::Unsupported)?;
                        let support = record::MeshSupport { loader, limits };
                        record::record_mesh(
                            &encoding,
                            &support,
                            draw,
                            public_set(public_sets, action_index),
                            prepared.texture_set,
                            pass_active,
                        )?;
                    }
                    NativeFrameAction::TextureReadback { texture, .. } => {
                        record::record_texture_readback(
                            &encoding,
                            texture,
                            readbacks,
                            &mut readback_index,
                            pass_active,
                        )?;
                    }
                    NativeFrameAction::EndPass => {
                        if !pass_active {
                            return Err(HalError::InvalidArgument);
                        }
                        // SAFETY: this command buffer owns the active dynamic-rendering pass.
                        unsafe {
                            prepared.device.cmd_end_rendering(prepared.command_handle);
                        }
                        pass_active = false;
                    }
                    NativeFrameAction::Present => {
                        record::record_present_readback(
                            &encoding,
                            capture_presented,
                            readbacks,
                            &mut readback_index,
                            pass_active,
                        )?;
                    }
                }
                Ok(())
            })?;
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

impl NativeContext {
    /// Validates, records, submits, and optionally presents one complete frame plan.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid frame plan, unavailable resources, allocation
    /// failures, or failed Vulkan operations.
    pub fn execute_frame(
        &mut self,
        surface: Option<FrameSurface<'_>>,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        self.execute_frame_source(surface, &actions, capture_presented)
    }

    /// Records a synchronous action source without retaining borrowed views.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::execute_frame`].
    pub fn execute_frame_source(
        &mut self,
        surface: Option<FrameSurface<'_>>,
        actions: &impl super::NativeFrameActionSource,
        capture_presented: bool,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        if actions.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        let (mut surface, extent, presentation_mode) = validate_surface_request(surface)?;
        let shader_capabilities = self
            .adapter_info()
            .ok_or(HalError::NotReady)?
            .capabilities()
            .shader_stages;
        actions.visit(&mut |_, action| {
            if let NativeFrameAction::Barrier { barrier, .. } = action {
                if let Some(before) = barrier.before {
                    vulkan_state(before, shader_capabilities)?;
                }
                vulkan_state(barrier.after, shader_capabilities)?;
            }
            Ok(())
        })?;
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
                presentation_mode,
            )?;
        }
        let mut requires_depth = false;
        actions.visit(&mut |_, action| {
            requires_depth |= matches!(
                action,
                NativeFrameAction::BeginPass { pass, .. } if pass.depth.is_some()
            );
            Ok(())
        })?;
        if requires_depth {
            self.ensure_depth_target(self.swapchain_extent)?;
        }
        let prepared = self.prepare_frame_slot(uses_surface)?;
        let frame_value = self.next_frame_value;
        self.next_frame_value = frame_value.checked_add(1).ok_or(HalError::NativeFailure)?;
        let (readbacks, public_sets) =
            self.allocate_frame_bindings(prepared.slot_index, actions, extent, capture_presented)?;
        // SAFETY: the reset primary command buffer is idle and recording begins once.
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
            self.restore_public_sets(prepared.slot_index, public_sets);
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
            let slot = self
                .frame_slots
                .get_mut(prepared.slot_index)
                .ok_or(HalError::NativeFailure)?;
            slot.in_flight = true;
            slot.submission_value = frame_value;
            self.last_frame_value = frame_value;
        }
        if submitted && (recorded.is_err() || !readbacks.is_empty()) {
            // SAFETY: submitted work owns this fence until it signals.
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
                let _ = self.recreate_swapchain(surface, extent.0, extent.1, presentation_mode);
            }
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            self.restore_public_sets(prepared.slot_index, public_sets);
            return Err(error);
        }
        if let Some(image_index) = acquired_image_index {
            *self
                .swapchain_initialized
                .get_mut(image_index as usize)
                .ok_or(HalError::NativeFailure)? = true;
        }
        self.restore_public_sets(prepared.slot_index, public_sets);
        self.collect_frame_readbacks(readbacks, capture_presented, surface)
    }
}

#[cfg(test)]
mod wait_tests {
    use super::{HalError, NativeFrameAction, validate_frame_plan};
    use ez_gfx_hal::{CompletionToken, QueueKind};

    #[test]
    fn frame_waits_collapse_to_one_maximum_per_transfer_queue() {
        let actions = [
            NativeFrameAction::Wait(CompletionToken {
                queue: QueueKind::Transfer,
                value: 1,
            }),
            NativeFrameAction::Wait(CompletionToken {
                queue: QueueKind::TextureTransfer,
                value: 2,
            }),
            NativeFrameAction::Wait(CompletionToken {
                queue: QueueKind::Transfer,
                value: 3,
            }),
        ];
        let plan = validate_frame_plan(&actions, (1, 1), false, false).unwrap();

        assert_eq!(plan.external_wait.len(), 2);
        assert!(
            plan.external_wait
                .iter()
                .any(|token| { token.queue == QueueKind::Transfer && token.value == 3 })
        );
        assert!(
            plan.external_wait
                .iter()
                .any(|token| { token.queue == QueueKind::TextureTransfer && token.value == 2 })
        );
        assert!(matches!(
            validate_frame_plan(
                &[NativeFrameAction::Wait(CompletionToken {
                    queue: QueueKind::Graphics,
                    value: 1,
                })],
                (1, 1),
                false,
                false,
            ),
            Err(HalError::InvalidArgument)
        ));
    }
}

#[cfg(test)]
mod public_set_tests {
    use super::public_set;
    use ash::vk::{self, Handle};

    #[test]
    fn compact_sets_resolve_only_pipeline_actions() {
        let first = vk::DescriptorSet::null();
        let second = vk::DescriptorSet::from_raw(7);
        let sets = [(1_u32, first), (4_u32, second)];
        assert_eq!(public_set(&sets, 1), Some(&first));
        assert_eq!(public_set(&sets, 4), Some(&second));
        // Barrier, pass, and readback actions hold no entry.
        assert_eq!(public_set(&sets, 0), None);
        assert_eq!(public_set(&sets, 2), None);
        assert_eq!(public_set(&sets, 9), None);
        assert_eq!(public_set(&[], 1), None);
    }
}

include!("frame_mesh_tests.rs");
