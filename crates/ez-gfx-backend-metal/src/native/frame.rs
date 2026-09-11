use super::{
    AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BufferTransfer, CAMetalDrawable,
    CullMode, FrontFace, HalError, MAX_ARGUMENT_BUFFERS_PER_SLOT, MTLArgumentEncoder,
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCullMode, MTLDevice, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderStages,
    MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize, MTLStoreAction, MTLTexture,
    MTLWinding, MemoryAllocator, MemoryClass, NativeAllocation, NativeContext, NativeFrameAction,
    NativeFrameResource, NativeGraphicsDraw, NativePipeline, NativeSurface, NativeTexture,
    PresentationMode, PrimitiveTopology, ProtocolObject, QueueKind, ThreadBound,
    map_allocation_hal,
};
use ez_gfx_hal::COUNTER_BUFFER_ELEMENT_OFFSET;

type MetalDrawable = super::Retained<ProtocolObject<dyn CAMetalDrawable>>;

type MetalFrameReadback = (NativeAllocation, u64, u64, u64, u32);
type MetalArgumentEncoder = ThreadBound<super::Retained<ProtocolObject<dyn MTLArgumentEncoder>>>;

type FrameSurface<'a> = (&'a mut NativeSurface, (u32, u32), PresentationMode);
type ResolvedFrameSurface<'a> = (Option<&'a mut NativeSurface>, (u32, u32));

fn validate_surface_request(
    surface: Option<FrameSurface<'_>>,
) -> Result<ResolvedFrameSurface<'_>, HalError> {
    match surface {
        Some((surface, extent, mode)) if extent.0 != 0 && extent.1 != 0 => {
            surface.set_presentation_mode(mode)?;
            Ok((Some(surface), extent))
        }
        Some(_) => Err(HalError::InvalidArgument),
        None => Ok((None, (0, 0))),
    }
}

fn buffer_range_fits(allocation_size: u64, range: ez_gfx_hal::BufferRange) -> bool {
    range
        .offset
        .checked_add(range.size)
        .is_some_and(|end| end <= allocation_size)
}

fn draw_ranges_fit(
    index_physical_size: u64,
    index_logical_size: u64,
    indirect_physical_size: u64,
    indirect_logical_size: u64,
    draw_count: u32,
) -> bool {
    let required_indirect = u64::from(draw_count)
        .checked_mul(20)
        .and_then(|size| size.checked_add(COUNTER_BUFFER_ELEMENT_OFFSET));
    index_logical_size != 0
        && index_logical_size <= index_physical_size
        && indirect_logical_size <= indirect_physical_size
        && required_indirect.is_some_and(|required| required <= indirect_logical_size)
}

fn bindings_fit(bindings: &dyn super::NativeBufferBindingSource) -> Result<bool, HalError> {
    let mut fits = true;
    bindings.visit(&mut |_, binding| {
        fits &= (binding.offset as u64) < binding.allocation.allocation.size();
        Ok(())
    })?;
    Ok(fits)
}

fn metal_size(size: [u32; 3]) -> MTLSize {
    MTLSize {
        width: size[0] as usize,
        height: size[1] as usize,
        depth: size[2] as usize,
    }
}

struct MetalFrameResources {
    slot_index: usize,
    /// Prepared argument-buffer slots for aliased actions, in ascending action
    /// order; every other action needs no entry.
    prepared_arguments: Vec<(u32, usize)>,
    readbacks: Vec<MetalFrameReadback>,
}

/// Returns the prepared argument-buffer slot for one action, if it is aliased.
///
/// Entries are recorded in ascending action order, so binary search is valid.
fn prepared_argument(prepared: &[(u32, usize)], action: usize) -> Option<usize> {
    // Action counts always fit `u32`; the fallback only guards the conversion.
    let action = u32::try_from(action).ok()?;
    prepared
        .binary_search_by_key(&action, |(action, _)| *action)
        .ok()
        .map(|index| prepared[index].1)
}

struct MetalFrameEncoder<'a> {
    command: &'a ProtocolObject<dyn MTLCommandBuffer>,
    surface: Option<&'a NativeSurface>,
    extent: (u32, u32),
    drawable: Option<&'a ProtocolObject<dyn CAMetalDrawable>>,
    drawable_texture: Option<&'a ProtocolObject<dyn MTLTexture>>,
    prepared_arguments: &'a [(u32, usize)],
    frame_slot: &'a super::FrameSlot,
    readbacks: &'a [MetalFrameReadback],
    render_encoder: Option<super::Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>>,
    render_area: [u32; 4],
    presented: bool,
    readback_index: usize,
}

impl MetalFrameEncoder<'_> {
    fn encode_barrier(
        &mut self,
        barrier: &super::ExecutionBarrier,
        resource: &super::NativeFrameResource<'_>,
    ) -> Result<(), HalError> {
        if self.render_encoder.is_some() {
            return Err(HalError::InvalidArgument);
        }
        match resource {
            NativeFrameResource::Buffer(allocation) => {
                let ez_gfx_hal::ExecutionRange::Buffer(range) = barrier.range else {
                    return Err(HalError::InvalidArgument);
                };
                if !buffer_range_fits(allocation.allocation.size(), range) {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameResource::Texture(texture) | NativeFrameResource::RenderTarget(texture) => {
                if !matches!(barrier.range, ez_gfx_hal::ExecutionRange::Image(_))
                    || texture.allocation.size() == 0
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameResource::Surface | NativeFrameResource::Depth => {
                if !matches!(barrier.range, ez_gfx_hal::ExecutionRange::Image(_)) {
                    return Err(HalError::InvalidArgument);
                }
            }
        }
        // Separate encoders in one self.command buffer are ordered Metal hazard
        // boundaries. Every transition action is therefore lowered by requiring
        // the prior encoder to be closed before the next node is opened.

        Ok(())
    }
    fn begin_pass(
        &mut self,
        pass: &super::ExecutionPass,
        colors: &[super::PassAttachment<'_>],
    ) -> Result<(), HalError> {
        if self.render_encoder.is_some()
            || pass.colors.len() != 1
            || colors.len() != 1
            || !matches!(pass.samples, 1 | 2 | 4 | 8)
        {
            return Err(HalError::InvalidArgument);
        }
        let attachment = colors.first().ok_or(HalError::InvalidArgument)?;
        // Textures, buffers, and depth images are never color attachments.
        enum Target<'a> {
            Surface(&'a ProtocolObject<dyn MTLTexture>),
            Target(&'a super::NativeTexture),
        }
        let target = match attachment.resource {
            super::NativeFrameResource::Surface => {
                if pass.samples != 1 {
                    return Err(HalError::InvalidArgument);
                }
                Target::Surface(
                    self.drawable_texture
                        .as_ref()
                        .ok_or(HalError::InvalidArgument)?,
                )
            }
            super::NativeFrameResource::RenderTarget(texture) => {
                if pass.depth.is_some() {
                    return Err(HalError::InvalidArgument);
                }
                // A multisampled texture renders exactly its count and
                // resolves into the sampled texture; single-sample textures
                // render directly.
                let expected = texture.msaa.as_ref().map_or(1, |msaa| msaa.samples);
                if expected != pass.samples {
                    return Err(HalError::InvalidArgument);
                }
                Target::Target(texture)
            }
            _ => return Err(HalError::InvalidArgument),
        };
        let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: Metal render-pass descriptors define color-attachment slot 0, so `objectAtIndexedSubscript(0)` is in bounds, and `descriptor` owns that attachment for the descriptor's lifetime.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        match target {
            // SAFETY: the drawable texture is retained by the surface for encoding.
            Target::Surface(texture) => color.setTexture(Some(texture)),
            // SAFETY: the render-target textures are retained by their context
            // record for encoding.
            Target::Target(texture) => match texture.msaa.as_ref() {
                Some(msaa) => {
                    color.setTexture(Some(&*msaa.texture));
                    color.setResolveTexture(Some(&*texture.texture));
                }
                None => color.setTexture(Some(&*texture.texture)),
            },
        }
        color.setLoadAction(match pass.load {
            AttachmentLoadOp::Load => MTLLoadAction::Load,
            AttachmentLoadOp::Clear => MTLLoadAction::Clear,
            AttachmentLoadOp::Discard => MTLLoadAction::DontCare,
        });
        // A multisampled target resolves into the sampled texture instead of
        // storing its storage.
        let resolving = matches!(target, Target::Target(texture) if texture.msaa.is_some());
        color.setStoreAction(match pass.store {
            AttachmentStoreOp::Store if resolving => MTLStoreAction::MultisampleResolve,
            AttachmentStoreOp::Store => MTLStoreAction::Store,
            AttachmentStoreOp::Discard => MTLStoreAction::DontCare,
        });
        color.setClearColor(MTLClearColor {
            red: f64::from(attachment.clear[0]),
            green: f64::from(attachment.clear[1]),
            blue: f64::from(attachment.clear[2]),
            alpha: f64::from(attachment.clear[3]),
        });
        if pass.depth.is_some() {
            let depth = self
                .surface
                .as_ref()
                .and_then(|surface| surface.depth.as_ref())
                .ok_or(HalError::NotReady)?;
            let attachment = descriptor.depthAttachment();
            attachment.setTexture(Some(&depth.texture));
            attachment.setLoadAction(match pass.load {
                AttachmentLoadOp::Load => MTLLoadAction::Load,
                AttachmentLoadOp::Clear => MTLLoadAction::Clear,
                AttachmentLoadOp::Discard => MTLLoadAction::DontCare,
            });
            attachment.setStoreAction(match pass.store {
                AttachmentStoreOp::Store => MTLStoreAction::Store,
                AttachmentStoreOp::Discard => MTLStoreAction::DontCare,
            });
            attachment.setClearDepth(1.0);
        }
        self.render_area = pass.area;
        self.render_encoder = Some(
            self.command
                .renderCommandEncoderWithDescriptor(&descriptor)
                .ok_or(HalError::NativeFailure)?,
        );

        Ok(())
    }
    fn compute(
        &mut self,
        action_index: usize,
        dispatch: &super::NativeComputeDispatch<'_>,
    ) -> Result<(), HalError> {
        if self.render_encoder.is_some()
            || dispatch.groups.contains(&0)
            || dispatch.threads_per_group.contains(&0)
        {
            return Err(HalError::InvalidArgument);
        }
        let NativePipeline::Compute {
            state,
            argument_encoder,
        } = dispatch.pipeline
        else {
            return Err(HalError::InvalidArgument);
        };
        let encoder = self
            .command
            .computeCommandEncoder()
            .ok_or(HalError::NativeFailure)?;
        encoder.setComputePipelineState(state);
        dispatch.bindings.visit(&mut |_, binding| {
            if binding.offset as u64 >= binding.allocation.allocation.size() {
                return Err(HalError::InvalidArgument);
            }
            // SAFETY: the checked offset lies inside the retained allocation.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&binding.allocation.buffer),
                    binding.offset,
                    binding.index,
                );
            }
            Ok(())
        })?;
        match (
            dispatch.texture_heap,
            argument_encoder.as_ref(),
            prepared_argument(self.prepared_arguments, action_index),
        ) {
            (Some(heap), Some(_), Some(index)) => {
                let buffer = &self.frame_slot.argument_buffers[index];
                for texture in dispatch.textures {
                    let resource = <ProtocolObject<dyn MTLTexture> as AsRef<
                        ProtocolObject<dyn MTLResource>,
                    >>::as_ref(&*texture.texture);
                    encoder.useResource_usage(resource, MTLResourceUsage::Read);
                }
                // SAFETY: frame preparation encoded the complete compute argument buffer and retains it through command completion.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, heap.binding as usize);
                }
            }
            (None, None, None) => {}
            _ => return Err(HalError::InvalidArgument),
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            metal_size(dispatch.groups),
            metal_size(dispatch.threads_per_group),
        );
        encoder.endEncoding();

        Ok(())
    }
    fn present(&mut self, capture_presented: bool) -> Result<(), HalError> {
        if self.render_encoder.is_some() || self.presented {
            return Err(HalError::InvalidArgument);
        }
        if capture_presented {
            let (allocation, _, row_stride, size, _) = self
                .readbacks
                .get(self.readback_index)
                .ok_or(HalError::InvalidArgument)?;
            let blit = self
                .command
                .blitCommandEncoder()
                .ok_or(HalError::NativeFailure)?;
            // SAFETY: `prepare_drawable` checked the source texture against `extent`, and the readback buffer has `row_stride * extent.1` bytes with `row_stride >= extent.0 * 4`, so this blit's source and destination ranges are in bounds and retained through encoding.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                                self.drawable_texture.as_ref().ok_or(HalError::InvalidArgument)?,
                                0,
                                0,
                                MTLOrigin { x: 0, y: 0, z: 0 },
                                MTLSize {
                                    width: usize::try_from(self.extent.0).map_err(|_| HalError::InvalidArgument)?,
                                    height: usize::try_from(self.extent.1).map_err(|_| HalError::InvalidArgument)?,
                                    depth: 1,
                                },
                                &allocation.buffer,
                                0,
                                usize::try_from(*row_stride).map_err(|_| HalError::InvalidArgument)?,
                                usize::try_from(*size).map_err(|_| HalError::InvalidArgument)?,
                            );
                blit.endEncoding();
            }
            self.readback_index += 1;
        }
        let drawable = self.drawable.ok_or(HalError::InvalidArgument)?;
        let drawable_ref = <ProtocolObject<dyn CAMetalDrawable> as AsRef<
            ProtocolObject<dyn objc2_metal::MTLDrawable>,
        >>::as_ref(drawable);
        self.command.presentDrawable(drawable_ref);
        self.presented = true;

        Ok(())
    }
}
#[path = "frame_encode.rs"]
mod encode;

impl NativeContext {
    fn prepare_texture_argument_buffer(
        &mut self,
        slot: usize,
        heap: Option<ez_gfx_hal::ShaderTextureHeapLayout>,
        textures: &[&NativeTexture],
        encoders: &[Option<&MetalArgumentEncoder>],
        argument_index: usize,
    ) -> Result<Option<usize>, HalError> {
        let Some(heap) = heap else {
            return if encoders.iter().all(Option::is_none) {
                Ok(None)
            } else {
                Err(HalError::InvalidArgument)
            };
        };
        if encoders.iter().all(Option::is_none)
            || argument_index >= MAX_ARGUMENT_BUFFERS_PER_SLOT
            || textures
                .iter()
                .any(|texture| texture.binding >= heap.capacity)
        {
            return Err(HalError::InvalidArgument);
        }
        let required = encoders
            .iter()
            .flatten()
            .map(|encoder| encoder.encodedLength())
            .max()
            .ok_or(HalError::InvalidArgument)?;
        let slot = self
            .frame_slots
            .get_mut(slot)
            .ok_or(HalError::NativeFailure)?;
        if argument_index == slot.argument_buffers.len() {
            slot.argument_buffers.push(ThreadBound::new(
                self.device
                    .newBufferWithLength_options(required, MTLResourceOptions::StorageModeShared)
                    .ok_or(HalError::NativeFailure)?,
            ));
        } else if slot.argument_buffers[argument_index].length() < required {
            slot.argument_buffers[argument_index] = ThreadBound::new(
                self.device
                    .newBufferWithLength_options(required, MTLResourceOptions::StorageModeShared)
                    .ok_or(HalError::NativeFailure)?,
            );
        }
        let buffer = &slot.argument_buffers[argument_index];
        for encoder in encoders.iter().flatten() {
            // SAFETY: `buffer` is at least the maximum encoded length across the declaring stage encoders and is retained by the frame slot.
            unsafe { encoder.setArgumentBuffer_offset(Some(buffer), 0) };
            for texture in textures {
                let texture_index = texture.binding as usize * heap.argument_stride as usize
                    + heap.texture_argument_offset as usize;
                let sampler_index = texture.binding as usize * heap.argument_stride as usize
                    + heap.sampler_argument_offset as usize;
                // SAFETY: validated heap capacity and stride/offset metadata place both argument indices in the encoder-declared layout.
                unsafe {
                    encoder.setTexture_atIndex(Some(&texture.texture), texture_index);
                    encoder.setSamplerState_atIndex(Some(&texture.sampler), sampler_index);
                }
            }
        }
        Ok(Some(argument_index))
    }

    fn prepare_graphics_argument_buffer(
        &mut self,
        slot: usize,
        draw: &NativeGraphicsDraw<'_>,
        argument_index: usize,
    ) -> Result<Option<usize>, HalError> {
        if draw.draw_count == 0
            || draw.state.topology == PrimitiveTopology::TriangleFan
            || !draw_ranges_fit(
                draw.index.allocation.size(),
                draw.index_size,
                draw.indirect.allocation.size(),
                draw.indirect_size,
                draw.draw_count,
            )
            || !bindings_fit(draw.bindings)?
        {
            return Err(HalError::InvalidArgument);
        }
        let NativePipeline::Graphics {
            vertex_argument_encoder,
            fragment_argument_encoder,
            ..
        } = draw.pipeline
        else {
            return Err(HalError::InvalidArgument);
        };
        let encoders = [
            vertex_argument_encoder.as_ref(),
            fragment_argument_encoder.as_ref(),
        ];
        self.prepare_texture_argument_buffer(
            slot,
            draw.texture_heap,
            draw.textures,
            &encoders,
            argument_index,
        )
    }

    fn prepare_compute_argument_buffer(
        &mut self,
        slot: usize,
        dispatch: &super::NativeComputeDispatch<'_>,
        argument_index: usize,
    ) -> Result<Option<usize>, HalError> {
        let NativePipeline::Compute {
            argument_encoder, ..
        } = dispatch.pipeline
        else {
            return Err(HalError::InvalidArgument);
        };
        let encoders = [argument_encoder.as_ref()];
        self.prepare_texture_argument_buffer(
            slot,
            dispatch.texture_heap,
            dispatch.textures,
            &encoders,
            argument_index,
        )
    }

    fn prepare_compute_action(
        &mut self,
        slot: usize,
        dispatch: &super::NativeComputeDispatch<'_>,
        argument_index: usize,
    ) -> Result<Option<usize>, HalError> {
        let NativePipeline::Compute { state, .. } = dispatch.pipeline else {
            return Err(HalError::InvalidArgument);
        };
        let thread_count = dispatch
            .threads_per_group
            .into_iter()
            .try_fold(1_u64, |total, value| total.checked_mul(u64::from(value)));
        if dispatch.groups.contains(&0)
            || dispatch.threads_per_group.contains(&0)
            || thread_count.is_none_or(|count| count > state.maxTotalThreadsPerThreadgroup() as u64)
            || !bindings_fit(dispatch.bindings)?
        {
            return Err(HalError::InvalidArgument);
        }
        self.prepare_compute_argument_buffer(slot, dispatch, argument_index)
    }

    /// Level-zero extent of a published Metal texture view.
    ///
    /// The view covers the `resident_mips` coarse tail, so its level zero is storage
    /// mip `mip_count - resident_mips`. Returns `None` when no level is published.
    fn published_view_extent(
        width: u32,
        height: u32,
        mip_count: u32,
        resident_mips: u32,
    ) -> Option<(usize, usize)> {
        if resident_mips == 0 || resident_mips > mip_count {
            return None;
        }
        let level = mip_count - resident_mips;
        let extent = |base: u32| {
            u64::from(base)
                .checked_shr(level)
                .map(|value| value.max(1))
                .and_then(|value| usize::try_from(value).ok())
        };
        Some((extent(width)?, extent(height)?))
    }
    pub(super) fn allocate_frame_readback(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(NativeAllocation, u64, u64, u64, u32), HalError> {
        let tight_row = u64::from(width)
            .checked_mul(4)
            .ok_or(HalError::InvalidArgument)?;
        let row_stride = tight_row
            .checked_add(255)
            .map(|value| value & !255)
            .ok_or(HalError::InvalidArgument)?;
        let size = row_stride
            .checked_mul(u64::from(height))
            .ok_or(HalError::InvalidArgument)?;
        let request = AllocationRequest::new(size, 256, MemoryClass::Readback, true, None)
            .map_err(|_| HalError::InvalidArgument)?;
        let allocation = self.allocate(request).map_err(map_allocation_hal)?;
        Ok((allocation, tight_row, row_stride, size, height))
    }

    fn validate_frame_plan(
        &mut self,
        surface: &mut Option<&mut NativeSurface>,
        extent: (u32, u32),
        actions: &(impl super::NativeFrameActionSource + ?Sized),
        capture_presented: bool,
    ) -> Result<(bool, bool), HalError> {
        let mut presents = false;
        let mut uses_surface = false;
        let mut requires_depth = false;
        actions.visit(&mut |_, action| {
            presents |= matches!(action, NativeFrameAction::Present);
            uses_surface |= match action {
                NativeFrameAction::BeginPass { colors, .. } => colors
                    .iter()
                    .any(|attachment| matches!(attachment.resource, NativeFrameResource::Surface)),
                NativeFrameAction::Present
                | NativeFrameAction::Barrier {
                    resource: NativeFrameResource::Surface | NativeFrameResource::Depth,
                    ..
                } => true,
                _ => false,
            };
            requires_depth |= matches!(
                action,
                NativeFrameAction::BeginPass { pass, .. } if pass.depth.is_some()
            );
            if let NativeFrameAction::Wait(token) = action {
                if token.queue == QueueKind::TextureTransfer {
                    self.completed_texture_transfer_value()
                        .map_err(map_allocation_hal)?;
                    if token.value >= self.next_texture_value {
                        return Err(HalError::InvalidArgument);
                    }
                    return Ok(());
                }
                if token.queue != QueueKind::Transfer || token.value >= self.next_transfer_value {
                    return Err(HalError::InvalidArgument);
                }
                self.transfer_worker
                    .as_ref()
                    .ok_or(HalError::NotReady)?
                    .flush_through(token.value)
                    .map_err(ez_gfx_hal::TransferWorkerError::to_hal_error)?;
                if let Some(pending) = self
                    .pending_transfers
                    .iter()
                    .find(|pending| pending.value == token.value)
                {
                    #[cfg(test)]
                    if let Some(observer) = self.buffer_wait_observer.take() {
                        let _ = observer.send(());
                    }
                    pending.command.waitUntilCompleted();
                }
                if token.value
                    > self
                        .completed_transfer_value()
                        .map_err(map_allocation_hal)?
                {
                    return Err(HalError::NativeFailure);
                }
            }
            Ok(())
        })?;
        if uses_surface && surface.is_none() || (uses_surface || capture_presented) && !presents {
            return Err(HalError::InvalidArgument);
        }
        if requires_depth {
            self.ensure_surface_depth(
                surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                extent,
            )?;
        }
        Ok((presents, uses_surface))
    }

    /// Returns prepared-argument scratch to its frame slot, keeping capacity.
    ///
    /// Entries are pure CPU pairs with no GPU lifetime, so reuse needs no
    /// completion wait of its own; the slot itself stays completion-gated.
    /// A missing slot is unreachable after preparation, so falling back to a
    /// drop preserves behavior and only loses retained capacity.
    fn reclaim_prepared_scratch(&mut self, slot_index: usize, prepared: Vec<(u32, usize)>) {
        if let Some(slot) = self.frame_slots.get_mut(slot_index) {
            slot.prepared_scratch = prepared;
        }
    }

    fn prepare_frame_resources(
        &mut self,
        surface: Option<&NativeSurface>,
        extent: (u32, u32),
        actions: &(impl super::NativeFrameActionSource + ?Sized),
        capture_presented: bool,
        presents: bool,
    ) -> Result<MetalFrameResources, HalError> {
        let (slot_index, must_wait) = self.frame_tracker.acquire();
        if must_wait {
            self.complete_frame_slot(slot_index)?;
        }
        // The slot is retired: the tracker waited on reuse and this frame has not
        // submitted yet. Scratch capacity persists across frames; entries are pure
        // CPU pairs with no GPU lifetime. Every error return below restores it.
        let mut prepared_arguments = core::mem::take(
            &mut self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NativeFailure)?
                .prepared_scratch,
        );
        prepared_arguments.clear();
        let mut argument_count = 0;
        let mut readbacks = Vec::new();
        let mut pass_active = false;
        let mut saw_present = false;
        let preparation = actions.visit(&mut |action_index, action| {
            if saw_present {
                return Err(HalError::InvalidArgument);
            }
            let item = match action {
                NativeFrameAction::Wait(_) => Ok(None),
                NativeFrameAction::Barrier { barrier, resource } => {
                    let valid = match resource {
                        NativeFrameResource::Buffer(allocation) => {
                            let ez_gfx_hal::ExecutionRange::Buffer(range) = barrier.range else {
                                return Err(HalError::InvalidArgument);
                            };
                            buffer_range_fits(allocation.allocation.size(), range)
                        }
                        NativeFrameResource::Texture(texture)
                        | NativeFrameResource::RenderTarget(texture) => {
                            matches!(barrier.range, ez_gfx_hal::ExecutionRange::Image(_))
                                && texture.allocation.size() != 0
                        }
                        NativeFrameResource::Surface | NativeFrameResource::Depth => {
                            matches!(barrier.range, ez_gfx_hal::ExecutionRange::Image(_))
                        }
                    };
                    if pass_active || !valid {
                        Err(HalError::InvalidArgument)
                    } else {
                        Ok(None)
                    }
                }
                NativeFrameAction::BeginPass { pass, colors } => {
                    // Textures, buffers, and depth images are never color
                    // attachments; depth with a render target stays unsupported.
                    // A multisampled pass needs a multisampled target and vice
                    // versa; surfaces stay single-sample.
                    let mut target_extent = None;
                    let mut valid = !pass_active
                        && pass.colors.len() == 1
                        && colors.len() == 1
                        && matches!(pass.samples, 1 | 2 | 4 | 8);
                    if let Some(attachment) = colors.first() {
                        target_extent = match attachment.resource {
                            NativeFrameResource::Surface => {
                                valid &= pass.samples == 1;
                                Some(extent)
                            }
                            NativeFrameResource::RenderTarget(texture) => {
                                valid &= pass.depth.is_none();
                                valid &= texture.msaa.as_ref().map_or(1, |msaa| msaa.samples)
                                    == pass.samples;
                                Some((texture.width, texture.height))
                            }
                            _ => None,
                        };
                    }
                    let Some((target_width, target_height)) = target_extent else {
                        return Err(HalError::InvalidArgument);
                    };
                    let invalid = !valid
                        || pass.area[0]
                            .checked_add(pass.area[2])
                            .is_none_or(|end| end > target_width)
                        || pass.area[1]
                            .checked_add(pass.area[3])
                            .is_none_or(|end| end > target_height);
                    if invalid {
                        Err(HalError::InvalidArgument)
                    } else {
                        pass_active = true;
                        Ok(None)
                    }
                }
                NativeFrameAction::Compute(dispatch) => {
                    if pass_active {
                        Err(HalError::InvalidArgument)
                    } else {
                        self.prepare_compute_action(slot_index, dispatch, argument_count)
                            .inspect(|prepared| {
                                if prepared.is_some() {
                                    argument_count += 1;
                                }
                            })
                    }
                }
                NativeFrameAction::Graphics(draw) => {
                    if !pass_active
                        || draw.depth_required
                            && surface.is_none_or(|surface| surface.depth.is_none())
                    {
                        Err(HalError::InvalidArgument)
                    } else {
                        self.prepare_graphics_argument_buffer(slot_index, draw, argument_count)
                            .inspect(|prepared| {
                                if prepared.is_some() {
                                    argument_count += 1;
                                }
                            })
                    }
                }
                NativeFrameAction::TextureReadback { width, height, .. } => {
                    if pass_active || *width == 0 || *height == 0 {
                        Err(HalError::InvalidArgument)
                    } else {
                        self.allocate_frame_readback(*width, *height)
                            .map(|readback| {
                                readbacks.push(readback);
                                None
                            })
                    }
                }
                NativeFrameAction::EndPass => {
                    if pass_active {
                        pass_active = false;
                        Ok(None)
                    } else {
                        Err(HalError::InvalidArgument)
                    }
                }
                NativeFrameAction::Present => {
                    if pass_active || saw_present {
                        Err(HalError::InvalidArgument)
                    } else {
                        saw_present = true;
                        if capture_presented {
                            let readback = self.allocate_frame_readback(extent.0, extent.1)?;
                            readbacks.push(readback);
                        }
                        Ok(None)
                    }
                }
            };
            match item {
                Ok(Some(index)) => {
                    let action =
                        u32::try_from(action_index).map_err(|_| HalError::InvalidArgument)?;
                    prepared_arguments.push((action, index));
                }
                Ok(None) => {}
                Err(error) => return Err(error),
            }
            Ok(())
        });
        if let Err(error) = preparation {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            self.reclaim_prepared_scratch(slot_index, prepared_arguments);
            return Err(error);
        }
        if pass_active || saw_present != presents {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            self.reclaim_prepared_scratch(slot_index, prepared_arguments);
            return Err(HalError::InvalidArgument);
        }
        Ok(MetalFrameResources {
            slot_index,
            prepared_arguments,
            readbacks,
        })
    }

    fn finish_frame(
        &mut self,
        command: super::Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        readbacks: Vec<MetalFrameReadback>,
        slot_index: usize,
        capture_presented: bool,
        mut surface: Option<&mut NativeSurface>,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        let frame_value = self.next_frame_value;
        self.next_frame_value = frame_value.checked_add(1).ok_or(HalError::NativeFailure)?;
        self.drain_complete = false;
        command.commit();
        self.last_frame_value = frame_value;
        if readbacks.is_empty() {
            self.frame_slots[slot_index].command = Some(ThreadBound::new(command));
            self.frame_slots[slot_index].submission_value = frame_value;
            self.frame_tracker.mark_submitted(slot_index);
            return Ok(Vec::new());
        }
        command.waitUntilCompleted();
        self.completed_frame_value = self.completed_frame_value.max(frame_value);
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(HalError::NativeFailure);
        }
        let mut outputs = Vec::with_capacity(readbacks.len());
        let mut remaining = readbacks.into_iter();
        while let Some((mut allocation, tight_row, row_stride, size, height)) = remaining.next() {
            let copied = (|| {
                self.invalidate(&mut allocation, 0, size)
                    .map_err(map_allocation_hal)?;
                let source = self.mapped_slice(&allocation).map_err(map_allocation_hal)?;
                let packed_size = tight_row
                    .checked_mul(u64::from(height))
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(HalError::InvalidArgument)?;
                let mut packed = Vec::with_capacity(packed_size);
                for row in 0..usize::try_from(height).map_err(|_| HalError::InvalidArgument)? {
                    let start = row
                        .checked_mul(
                            usize::try_from(row_stride).map_err(|_| HalError::InvalidArgument)?,
                        )
                        .ok_or(HalError::InvalidArgument)?;
                    let end = start
                        .checked_add(
                            usize::try_from(tight_row).map_err(|_| HalError::InvalidArgument)?,
                        )
                        .ok_or(HalError::InvalidArgument)?;
                    packed
                        .extend_from_slice(source.get(start..end).ok_or(HalError::NativeFailure)?);
                }
                Ok(packed)
            })();
            let freed = self.free(allocation).map_err(map_allocation_hal);
            let mut packed = match (copied, freed) {
                (Ok(packed), Ok(())) => packed,
                (Err(error), _) | (_, Err(error)) => {
                    for (allocation, _, _, _, _) in remaining {
                        let _ = self.free(allocation);
                    }
                    return Err(error);
                }
            };
            if capture_presented && remaining.len() == 0 {
                for pixel in packed.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
                surface
                    .as_deref_mut()
                    .ok_or(HalError::InvalidArgument)?
                    .presented_rgba8
                    .clone_from(&packed);
            }
            outputs.push(packed);
        }
        Ok(outputs)
    }

    fn prepare_drawable(
        &self,
        surface: Option<&NativeSurface>,
        extent: (u32, u32),
        uses_surface: bool,
        capture_presented: bool,
    ) -> Result<Option<MetalDrawable>, HalError> {
        if uses_surface {
            let surface = surface.ok_or(HalError::InvalidArgument)?;
            let layer = surface.metal_layer();
            layer.setDevice(Some(&self.device));
            layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
            // Core Animation defaults to framebuffer-only drawables. Disable that restriction
            // before the first requested capture; leaving it disabled supports later one-frame
            // captures without recreating the host layer.
            if capture_presented {
                layer.setFramebufferOnly(false);
            }
            let drawable = layer.nextDrawable().ok_or(HalError::NotReady)?;
            let texture = drawable.texture();
            if texture.width()
                != usize::try_from(extent.0).map_err(|_| HalError::InvalidArgument)?
                || texture.height()
                    != usize::try_from(extent.1).map_err(|_| HalError::InvalidArgument)?
            {
                return Err(HalError::NotReady);
            }
            Ok(Some(drawable))
        } else {
            Ok(None)
        }
    }

    /// Records one complete frame into one Metal command buffer.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid plans, allocation failures, or rejected commands.
    pub fn execute_frame(
        &mut self,
        surface: Option<FrameSurface<'_>>,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        self.execute_frame_source(surface, &actions, capture_presented)
    }

    /// Records a synchronous source without retaining borrowed native views.
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
        let (mut surface, extent) = validate_surface_request(surface)?;
        let (presents, uses_surface) =
            self.validate_frame_plan(&mut surface, extent, actions, capture_presented)?;
        // A failed producer must be rejected before allocating frame resources or queuing
        // an event wait that could otherwise remain permanently unsignaled.
        let mut required = None::<u64>;
        actions.visit(&mut |_, action| {
            if let NativeFrameAction::Wait(token) = action
                && token.queue == QueueKind::TextureTransfer
            {
                required = Some(required.map_or(token.value, |value| value.max(token.value)));
            }
            Ok(())
        })?;
        if let Some(required) = required {
            self.texture_worker
                .as_ref()
                .ok_or(HalError::NotReady)?
                .flush_through(required)
                .map_err(ez_gfx_hal::TransferWorkerError::to_hal_error)?;
        }
        let MetalFrameResources {
            slot_index,
            prepared_arguments,
            readbacks,
        } = self.prepare_frame_resources(
            surface.as_deref(),
            extent,
            actions,
            capture_presented,
            presents,
        )?;
        let drawable =
            self.prepare_drawable(surface.as_deref(), extent, uses_surface, capture_presented);
        let drawable = match drawable {
            Ok(drawable) => drawable,
            Err(error) => {
                for (allocation, _, _, _, _) in readbacks {
                    let _ = self.free(allocation);
                }
                self.reclaim_prepared_scratch(slot_index, prepared_arguments);
                return Err(error);
            }
        };
        let drawable_texture = drawable.as_ref().map(|drawable| drawable.texture());
        let Some(command) = self.queue.commandBuffer() else {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            self.reclaim_prepared_scratch(slot_index, prepared_arguments);
            return Err(HalError::NativeFailure);
        };
        // Encode waits before opening any encoder. Updates have already committed their
        // graphics release marker, so this cannot wait ahead of its own producer.
        actions.visit(&mut |_, action| {
            if let NativeFrameAction::Wait(token) = action
                && token.queue == QueueKind::TextureTransfer
            {
                command.encodeWaitForEvent_value(&self.texture_completion_event, token.value);
            }
            Ok(())
        })?;
        let mut encoder = MetalFrameEncoder {
            command: &command,
            surface: surface.as_deref(),
            extent,
            drawable: drawable.as_deref(),
            drawable_texture: drawable_texture.as_deref(),
            prepared_arguments: &prepared_arguments,
            frame_slot: &self.frame_slots[slot_index],
            readbacks: &readbacks,
            render_encoder: None,
            render_area: [0; 4],
            presented: false,
            readback_index: 0,
        };
        let recording_result = (|| -> Result<(), HalError> {
            actions.visit(&mut |action_index, action| {
                match action {
                    NativeFrameAction::Wait(_) => {}
                    NativeFrameAction::Barrier { barrier, resource } => {
                        encoder.encode_barrier(barrier, resource)?;
                    }
                    NativeFrameAction::BeginPass { pass, colors } => {
                        encoder.begin_pass(pass, colors)?;
                    }
                    NativeFrameAction::Compute(dispatch) => {
                        encoder.compute(action_index, dispatch)?;
                    }
                    NativeFrameAction::Graphics(draw) => encoder.graphics(action_index, draw)?,
                    NativeFrameAction::TextureReadback {
                        texture,
                        width,
                        height,
                    } => {
                        let (allocation, _, row_stride, size, _) = readbacks
                            .get(encoder.readback_index)
                            .ok_or(HalError::InvalidArgument)?;
                        let blit = command
                            .blitCommandEncoder()
                            .ok_or(HalError::NativeFailure)?;
                        // The shared boundary only admits fully resident readbacks, but the
                        // copy must still honor the published view: a demoted view's level
                        // zero is a coarser storage mip with a smaller extent.
                        let (exposed_width, exposed_height) = Self::published_view_extent(
                            texture.width,
                            texture.height,
                            texture.mip_count,
                            texture.resident_mips,
                        )
                        .ok_or(HalError::InvalidArgument)?;
                        // SAFETY: the copy is clamped to the published view's real level-zero
                        // extent, and its readback buffer has `row_stride * height` bytes with
                        // `row_stride >= width * 4`, so the blit's ranges are in bounds and both
                        // resources remain referenced through encoding.
                        unsafe {
                            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                            &texture.texture,
                            0,
                            0,
                            MTLOrigin { x: 0, y: 0, z: 0 },
                            MTLSize {
                                width: (*width as usize).min(exposed_width),
                                height: (*height as usize).min(exposed_height),
                                depth: 1,
                            },
                            &allocation.buffer,
                            0,
                            usize::try_from(*row_stride).map_err(|_| HalError::InvalidArgument)?,
                            usize::try_from(*size).map_err(|_| HalError::InvalidArgument)?,
                        );
                            blit.endEncoding();
                        }
                        encoder.readback_index += 1;
                    }
                    NativeFrameAction::EndPass => {
                        let encoder = encoder
                            .render_encoder
                            .take()
                            .ok_or(HalError::InvalidArgument)?;
                        encoder.endEncoding();
                    }
                    NativeFrameAction::Present => encoder.present(capture_presented)?,
                }
                Ok(())
            })?;
            if encoder.render_encoder.is_some()
                || encoder.presented != presents
                || encoder.readback_index != readbacks.len()
            {
                return Err(HalError::InvalidArgument);
            }
            Ok(())
        })();
        // Encoder borrows end with the recording closure, so scratch returns to
        // the slot on both exits; readback failures below never lose capacity.
        self.reclaim_prepared_scratch(slot_index, prepared_arguments);
        if let Err(error) = recording_result {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(error);
        }
        self.finish_frame(command, readbacks, slot_index, capture_presented, surface)
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeContext, buffer_range_fits, draw_ranges_fit, metal_size};
    use ez_gfx_hal::{BufferRange, COUNTER_BUFFER_ELEMENT_OFFSET};

    #[test]
    fn buffer_barrier_range_must_fit_allocation() {
        assert!(buffer_range_fits(64, BufferRange::new(16, 48).unwrap()));
        assert!(!buffer_range_fits(63, BufferRange::new(16, 48).unwrap()));
        assert!(!buffer_range_fits(
            u64::MAX,
            BufferRange {
                offset: u64::MAX - 3,
                size: 4,
            }
        ));
    }

    #[test]
    fn indexed_indirect_logical_ranges_include_aligned_element_offset() {
        let required = COUNTER_BUFFER_ELEMENT_OFFSET + 40;
        assert!(draw_ranges_fit(64, 64, required, required, 2));
        assert!(!draw_ranges_fit(64, 0, required, required, 2));
        assert!(!draw_ranges_fit(64, 65, required, required, 2));
        assert!(!draw_ranges_fit(64, 64, required, required - 1, 2));
        assert!(!draw_ranges_fit(64, 64, required - 1, required, 2));
    }

    #[test]
    fn reflected_threadgroup_dimensions_reach_metal_dispatch_shape() {
        let size = metal_size([8, 2, 1]);
        assert_eq!((size.width, size.height, size.depth), (8, 2, 1));
    }

    #[test]
    fn demoted_view_extent_follows_published_coarse_tail() {
        use NativeContext as Context;
        // A fully resident view exposes the stored base extent.
        assert_eq!(Context::published_view_extent(64, 64, 3, 3), Some((64, 64)));
        // Demotion drops fine levels: level one of a 64-wide chain is 16 wide.
        assert_eq!(Context::published_view_extent(64, 64, 3, 1), Some((16, 16)));
        // Odd edges clamp at one texel rather than shifting to zero.
        assert_eq!(Context::published_view_extent(7, 3, 3, 1), Some((1, 1)));
        // An unpublished or over-published view has no readable extent.
        assert_eq!(Context::published_view_extent(64, 64, 3, 0), None);
        assert_eq!(Context::published_view_extent(64, 64, 3, 4), None);
    }
}
