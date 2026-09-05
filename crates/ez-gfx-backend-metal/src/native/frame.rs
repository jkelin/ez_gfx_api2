use super::{
    AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BufferTransfer, CAMetalDrawable,
    CAMetalLayer, CullMode, FrontFace, HalError, MAX_ARGUMENT_BUFFERS_PER_SLOT, MTLArgumentEncoder,
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCullMode, MTLDevice, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderStages,
    MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize, MTLStoreAction, MTLTexture,
    MTLWinding, MemoryAllocator, MemoryClass, NativeAllocation, NativeContext, NativeFrameAction,
    NativeFrameResource, NativeGraphicsDraw, NativePipeline, NativeSurface, NativeTexture,
    PrimitiveTopology, ProtocolObject, QueueKind, ThreadBound, c_void, map_allocation_hal,
};

type MetalDrawable = super::Retained<ProtocolObject<dyn CAMetalDrawable>>;

type MetalFrameReadback = (NativeAllocation, u64, u64, u64, u32);
type MetalArgumentEncoder = ThreadBound<super::Retained<ProtocolObject<dyn MTLArgumentEncoder>>>;

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
    let required_indirect = u64::from(draw_count).checked_mul(20);
    index_logical_size != 0
        && index_logical_size <= index_physical_size
        && indirect_logical_size <= indirect_physical_size
        && required_indirect.is_some_and(|required| required <= indirect_logical_size)
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
    prepared_arguments: Vec<Option<usize>>,
    readbacks: Vec<MetalFrameReadback>,
}

struct MetalFrameEncoder<'a> {
    command: &'a ProtocolObject<dyn MTLCommandBuffer>,
    surface: Option<&'a NativeSurface>,
    extent: (u32, u32),
    drawable: Option<&'a ProtocolObject<dyn CAMetalDrawable>>,
    drawable_texture: Option<&'a ProtocolObject<dyn MTLTexture>>,
    prepared_arguments: &'a [Option<usize>],
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
            NativeFrameResource::Texture(texture) => {
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
    fn begin_pass(&mut self, pass: &super::ExecutionPass) -> Result<(), HalError> {
        if self.render_encoder.is_some()
            || pass.colors.len() != 1
            || pass.samples != 1
            || pass.area[0]
                .checked_add(pass.area[2])
                .is_none_or(|end| end > self.extent.0)
            || pass.area[1]
                .checked_add(pass.area[3])
                .is_none_or(|end| end > self.extent.1)
        {
            return Err(HalError::InvalidArgument);
        }
        let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: Metal render-pass descriptors define color-attachment slot 0, so `objectAtIndexedSubscript(0)` is in bounds, and `descriptor` owns that attachment for the descriptor's lifetime.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(
            self.drawable_texture
                .as_ref()
                .ok_or(HalError::InvalidArgument)?,
        ));
        color.setLoadAction(match pass.load {
            AttachmentLoadOp::Load => MTLLoadAction::Load,
            AttachmentLoadOp::Clear => MTLLoadAction::Clear,
            AttachmentLoadOp::Discard => MTLLoadAction::DontCare,
        });
        color.setStoreAction(match pass.store {
            AttachmentStoreOp::Store => MTLStoreAction::Store,
            AttachmentStoreOp::Discard => MTLStoreAction::DontCare,
        });
        color.setClearColor(MTLClearColor {
            red: 0.1,
            green: 0.1,
            blue: 0.1,
            alpha: 1.0,
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
            || dispatch.push_constants.len() > 128
            || !dispatch.push_constants.len().is_multiple_of(4)
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
        for binding in dispatch.bindings {
            if binding.offset as u64 >= binding.allocation.allocation.size() {
                return Err(HalError::InvalidArgument);
            }
            // SAFETY: the preceding check places `binding.offset` within the allocation backing `binding.allocation.buffer`, which remains stored in `binding.allocation` while `setBuffer_offset_atIndex` records the binding.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&binding.allocation.buffer),
                    binding.offset,
                    binding.index,
                );
            };
        }
        match (
            dispatch.texture_heap,
            argument_encoder.as_ref(),
            self.prepared_arguments
                .get(action_index)
                .and_then(|prepared| *prepared),
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
        if let Some(bytes) =
            core::ptr::NonNull::new(dispatch.push_constants.as_ptr() as *mut c_void)
            && !dispatch.push_constants.is_empty()
        {
            // SAFETY: `bytes` points to the nonempty `dispatch.push_constants` storage for exactly `len()` bytes, and `setBytes_length_atIndex` copies those bytes before that storage can be released.
            unsafe {
                encoder.setBytes_length_atIndex(bytes, dispatch.push_constants.len(), 0);
            };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            metal_size(dispatch.groups),
            metal_size(dispatch.threads_per_group),
        );
        encoder.endEncoding();

        Ok(())
    }
    fn graphics(
        &mut self,
        action_index: usize,
        draw: &super::NativeGraphicsDraw<'_>,
    ) -> Result<(), HalError> {
        let encoder = self
            .render_encoder
            .as_ref()
            .ok_or(HalError::InvalidArgument)?;
        let NativePipeline::Graphics {
            state,
            vertex_argument_encoder,
            fragment_argument_encoder,
        } = draw.pipeline
        else {
            return Err(HalError::InvalidArgument);
        };
        let argument_buffer = self
            .prepared_arguments
            .get(action_index)
            .and_then(|prepared| prepared.map(|index| &self.frame_slot.argument_buffers[index]));
        encoder.setRenderPipelineState(state);
        if draw.depth_required {
            let depth = self
                .surface
                .as_ref()
                .and_then(|surface| surface.depth.as_ref())
                .ok_or(HalError::InvalidArgument)?;
            encoder.setDepthStencilState(Some(&depth.state));
        }
        encoder.setCullMode(match draw.state.cull {
            CullMode::None => MTLCullMode::None,
            CullMode::Front => MTLCullMode::Front,
            CullMode::Back => MTLCullMode::Back,
        });
        encoder.setFrontFacingWinding(match draw.state.front_face {
            FrontFace::CounterClockwise => MTLWinding::CounterClockwise,
            FrontFace::Clockwise => MTLWinding::Clockwise,
        });
        encoder.setViewport(objc2_metal::MTLViewport {
            originX: f64::from(self.render_area[0]),
            originY: f64::from(self.render_area[1]),
            width: f64::from(self.render_area[2]),
            height: f64::from(self.render_area[3]),
            znear: 0.0,
            zfar: 1.0,
        });
        for binding in draw.bindings {
            if binding.offset as u64 >= binding.allocation.allocation.size() {
                return Err(HalError::InvalidArgument);
            }
            // SAFETY: the preceding check places `binding.offset` within the allocation backing `binding.allocation.buffer`, which remains stored in `binding.allocation` while the vertex and fragment buffer bindings are recorded.
            unsafe {
                encoder.setVertexBuffer_offset_atIndex(
                    Some(&binding.allocation.buffer),
                    binding.offset,
                    binding.index,
                );
                encoder.setFragmentBuffer_offset_atIndex(
                    Some(&binding.allocation.buffer),
                    binding.offset,
                    binding.index,
                );
            }
        }
        match (draw.texture_heap, argument_buffer) {
            (Some(heap), Some(buffer)) => {
                for texture in draw.textures {
                    let resource = <ProtocolObject<dyn MTLTexture> as AsRef<
                        ProtocolObject<dyn MTLResource>,
                    >>::as_ref(&*texture.texture);
                    if vertex_argument_encoder.is_some() {
                        encoder.useResource_usage_stages(
                            resource,
                            MTLResourceUsage::Read,
                            MTLRenderStages::Vertex,
                        );
                    }
                    if fragment_argument_encoder.is_some() {
                        encoder.useResource_usage_stages(
                            resource,
                            MTLResourceUsage::Read,
                            MTLRenderStages::Fragment,
                        );
                    }
                }
                // SAFETY: frame preparation sized and encoded this buffer with every declaring stage encoder, and the frame slot retains it through completion.
                unsafe {
                    if vertex_argument_encoder.is_some() {
                        encoder.setVertexBuffer_offset_atIndex(
                            Some(buffer),
                            0,
                            heap.binding as usize,
                        );
                    }
                    if fragment_argument_encoder.is_some() {
                        encoder.setFragmentBuffer_offset_atIndex(
                            Some(buffer),
                            0,
                            heap.binding as usize,
                        );
                    }
                }
            }
            (None, None)
                if vertex_argument_encoder.is_none() && fragment_argument_encoder.is_none() => {}
            _ => return Err(HalError::InvalidArgument),
        }
        if let Some(bytes) = core::ptr::NonNull::new(draw.push_constants.as_ptr() as *mut c_void)
            && !draw.push_constants.is_empty()
        {
            // SAFETY: `bytes` points to the nonempty `draw.push_constants` storage for exactly `len()` bytes, and both `set*Bytes_length_atIndex` calls copy those bytes before that storage can be released.
            unsafe {
                encoder.setVertexBytes_length_atIndex(bytes, draw.push_constants.len(), 0);
                encoder.setFragmentBytes_length_atIndex(bytes, draw.push_constants.len(), 0);
            }
        }
        let primitive = match draw.state.topology {
            PrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
            PrimitiveTopology::PointList => MTLPrimitiveType::Point,
            PrimitiveTopology::LineList => MTLPrimitiveType::Line,
            PrimitiveTopology::LineStrip => MTLPrimitiveType::LineStrip,
            PrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
            PrimitiveTopology::TriangleFan => {
                return Err(HalError::Unsupported);
            }
        };
        for command_index in 0..draw.draw_count {
            // SAFETY: `NativeGraphicsDraw` supplies `draw_count` contiguous 20-byte indirect records and the referenced UInt32 index storage, so offsets `command_index * 20` and 0 are aligned and in bounds while `draw` holds both buffers.
            unsafe {
                encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                                primitive,
                                MTLIndexType::UInt32,
                                &draw.index.buffer,
                                0,
                                &draw.indirect.buffer,
                                command_index as usize * 20,
                            );
            };
        }

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
            || draw.push_constants.len() > 128
            || !draw.push_constants.len().is_multiple_of(4)
            || draw.state.topology == PrimitiveTopology::TriangleFan
            || !draw_ranges_fit(
                draw.index.allocation.size(),
                draw.index_size,
                draw.indirect.allocation.size(),
                draw.indirect_size,
                draw.draw_count,
            )
            || draw
                .bindings
                .iter()
                .any(|binding| binding.offset as u64 >= binding.allocation.allocation.size())
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
            || dispatch.push_constants.len() > 128
            || !dispatch.push_constants.len().is_multiple_of(4)
            || dispatch
                .bindings
                .iter()
                .any(|binding| binding.offset as u64 >= binding.allocation.allocation.size())
        {
            return Err(HalError::InvalidArgument);
        }
        self.prepare_compute_argument_buffer(slot, dispatch, argument_index)
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
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
    ) -> Result<(bool, bool), HalError> {
        let presents = actions
            .iter()
            .any(|action| matches!(action, NativeFrameAction::Present));
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
        if uses_surface && surface.is_none() || (uses_surface || capture_presented) && !presents {
            return Err(HalError::InvalidArgument);
        }
        for action in actions {
            if let NativeFrameAction::Wait(token) = action {
                if token.queue == QueueKind::TextureTransfer {
                    // Texture work has a GPU event dependency; pending is valid, but fabricated
                    // values and failed worker submission must fail before command encoding.
                    self.completed_texture_transfer_value()
                        .map_err(map_allocation_hal)?;
                    if token.value >= self.next_texture_value {
                        return Err(HalError::InvalidArgument);
                    }
                    continue;
                }
                let completed = match token.queue {
                    QueueKind::Transfer => self.completed_transfer_value(),
                    _ => return Err(HalError::InvalidArgument),
                }
                .map_err(map_allocation_hal)?;
                if token.value > completed {
                    return Err(HalError::InvalidArgument);
                }
            }
        }
        if actions.iter().any(|action| {
            matches!(
                action,
                NativeFrameAction::BeginPass(pass) if pass.depth.is_some()
            )
        }) {
            self.ensure_surface_depth(
                surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                extent,
            )?;
        }
        Ok((presents, uses_surface))
    }

    fn prepare_frame_resources(
        &mut self,
        surface: Option<&NativeSurface>,
        extent: (u32, u32),
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
        presents: bool,
    ) -> Result<MetalFrameResources, HalError> {
        let (slot_index, must_wait) = self.frame_tracker.acquire();
        if must_wait {
            self.complete_frame_slot(slot_index)?;
        }
        let mut prepared_arguments = Vec::with_capacity(actions.len());
        let mut argument_count = 0;
        let mut readbacks = Vec::new();
        let mut pass_active = false;
        let mut saw_present = false;
        for action in actions {
            if saw_present {
                for (allocation, _, _, _, _) in readbacks {
                    let _ = self.free(allocation);
                }
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
                        NativeFrameResource::Texture(texture) => {
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
                NativeFrameAction::BeginPass(pass) => {
                    let invalid = pass_active
                        || pass.colors.len() != 1
                        || pass.samples != 1
                        || pass.area[0]
                            .checked_add(pass.area[2])
                            .is_none_or(|end| end > extent.0)
                        || pass.area[1]
                            .checked_add(pass.area[3])
                            .is_none_or(|end| end > extent.1);
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
                            match self.allocate_frame_readback(extent.0, extent.1) {
                                Ok(readback) => readbacks.push(readback),
                                Err(error) => {
                                    for (allocation, _, _, _, _) in readbacks {
                                        let _ = self.free(allocation);
                                    }
                                    return Err(error);
                                }
                            }
                        }
                        Ok(None)
                    }
                }
            };
            match item {
                Ok(item) => prepared_arguments.push(item),
                Err(error) => {
                    for (allocation, _, _, _, _) in readbacks {
                        let _ = self.free(allocation);
                    }
                    return Err(error);
                }
            }
        }
        if pass_active || saw_present != presents {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
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
    ) -> Result<Option<Vec<u8>>, HalError> {
        self.drain_complete = false;
        command.commit();
        if readbacks.is_empty() {
            self.frame_slots[slot_index].command = Some(ThreadBound::new(command));
            self.frame_tracker.mark_submitted(slot_index);
            return Ok(None);
        }
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(HalError::NativeFailure);
        }
        let mut final_pixels = None;
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
            final_pixels = Some(packed);
        }
        Ok(final_pixels)
    }

    fn prepare_drawable(
        &self,
        surface: Option<&NativeSurface>,
        extent: (u32, u32),
        uses_surface: bool,
    ) -> Result<Option<MetalDrawable>, HalError> {
        if uses_surface {
            let surface = surface.ok_or(HalError::InvalidArgument)?;
            // SAFETY: `NativeSurface::layer` is a non-null, properly aligned pointer to a `CAMetalLayer` retained by `surface`, so dereferencing it for the lifetime of this shared surface borrow is sound.
            let layer = unsafe { &*(surface.layer as *const CAMetalLayer) };
            layer.setDevice(Some(&self.device));
            layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
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

    /// Records one complete frame into one Metal command buffer and presents only after every
    /// action has been encoded successfully.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid plan, allocation failure, or rejected Metal commands.
    pub fn execute_frame(
        &mut self,
        surface: Option<(&mut NativeSurface, (u32, u32))>,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
    ) -> Result<Option<Vec<u8>>, HalError> {
        if actions.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        let (mut surface, extent) = match surface {
            Some((surface, extent)) if extent.0 != 0 && extent.1 != 0 => (Some(surface), extent),
            Some(_) => return Err(HalError::InvalidArgument),
            None => (None, (0, 0)),
        };
        let (presents, uses_surface) =
            self.validate_frame_plan(&mut surface, extent, actions, capture_presented)?;
        // A failed producer must be rejected before allocating frame resources or queuing
        // an event wait that could otherwise remain permanently unsignaled.
        if let Some(required) = actions
            .iter()
            .filter_map(|action| match action {
                NativeFrameAction::Wait(token) if token.queue == QueueKind::TextureTransfer => {
                    Some(token.value)
                }
                _ => None,
            })
            .max()
        {
            self.texture_worker
                .as_ref()
                .ok_or(HalError::NotReady)?
                .flush_through(required)
                .map_err(|_| HalError::NativeFailure)?;
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
        let drawable = self.prepare_drawable(surface.as_deref(), extent, uses_surface);
        let drawable = match drawable {
            Ok(drawable) => drawable,
            Err(error) => {
                for (allocation, _, _, _, _) in readbacks {
                    let _ = self.free(allocation);
                }
                return Err(error);
            }
        };
        let drawable_texture = drawable.as_ref().map(|drawable| drawable.texture());
        let Some(command) = self.queue.commandBuffer() else {
            for (allocation, _, _, _, _) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(HalError::NativeFailure);
        };
        // Encode waits before opening any encoder. Updates have already committed their
        // graphics release marker, so this cannot wait ahead of its own producer.
        for action in actions {
            if let NativeFrameAction::Wait(token) = action
                && token.queue == QueueKind::TextureTransfer
            {
                command.encodeWaitForEvent_value(&self.texture_completion_event, token.value);
            }
        }
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
            for (action_index, action) in actions.iter().enumerate() {
                match action {
                    NativeFrameAction::Wait(_) => {}
                    NativeFrameAction::Barrier { barrier, resource } => {
                        encoder.encode_barrier(barrier, resource)?;
                    }
                    NativeFrameAction::BeginPass(pass) => encoder.begin_pass(pass)?,
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
                        // SAFETY: the readback action's dimensions bound the source region of `texture.texture`, and its readback buffer has `row_stride * height` bytes with `row_stride >= width * 4`, so the blit's ranges are in bounds and both resources remain referenced through encoding.
                        unsafe {
                            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                            &texture.texture,
                            0,
                            0,
                            MTLOrigin { x: 0, y: 0, z: 0 },
                            MTLSize {
                                width: *width as usize,
                                height: *height as usize,
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
            }
            if encoder.render_encoder.is_some()
                || encoder.presented != presents
                || encoder.readback_index != readbacks.len()
            {
                return Err(HalError::InvalidArgument);
            }
            Ok(())
        })();
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
    use super::{buffer_range_fits, draw_ranges_fit, metal_size};
    use ez_gfx_hal::BufferRange;

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
    fn indexed_indirect_logical_ranges_bound_native_reads() {
        assert!(draw_ranges_fit(64, 64, 40, 40, 2));
        assert!(!draw_ranges_fit(64, 0, 40, 40, 2));
        assert!(!draw_ranges_fit(64, 65, 40, 40, 2));
        assert!(!draw_ranges_fit(64, 64, 40, 39, 2));
        assert!(!draw_ranges_fit(64, 64, 39, 40, 2));
    }

    #[test]
    fn reflected_threadgroup_dimensions_reach_metal_dispatch_shape() {
        let size = metal_size([8, 2, 1]);
        assert_eq!((size.width, size.height, size.depth), (8, 2, 1));
    }
}
