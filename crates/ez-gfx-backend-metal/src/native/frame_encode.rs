//! Metal frame graphics-draw encoding on one render command encoder.

use super::{
    CullMode, FrontFace, HalError, MTLCullMode, MTLIndexType, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderStages, MTLResource, MTLResourceUsage, MTLTexture,
    MTLWinding, MetalFrameEncoder, NativePipeline, PrimitiveTopology, ProtocolObject,
};
use ez_gfx_hal::COUNTER_BUFFER_ELEMENT_OFFSET;

pub(super) const fn object_threadgroup_size(task: Option<[u32; 3]>) -> [u32; 3] {
    match task {
        Some(size) => size,
        None => [1, 1, 1],
    }
}

pub(super) fn buffer_stage_uses_binding(
    layouts: &[ez_gfx_hal::ShaderBufferLayout],
    binding: usize,
) -> bool {
    u32::try_from(binding).is_ok_and(|binding| {
        layouts.iter().any(|layout| {
            layout
                .binding
                .checked_add(layout.descriptor_count)
                .is_some_and(|end| binding >= layout.binding && binding < end)
        })
    })
}

impl MetalFrameEncoder<'_> {
    pub(super) fn graphics(
        &mut self,
        action_index: usize,
        draw: &super::super::NativeGraphicsDraw<'_>,
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
        let argument_buffer = super::prepared_argument(self.prepared_arguments, action_index)
            .map(|index| &self.frame_slot.argument_buffers[index]);
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
        draw.bindings.visit(&mut |_, binding| {
            if binding.offset as u64 >= binding.allocation.allocation.size() {
                return Err(HalError::InvalidArgument);
            }
            // SAFETY: the checked offset lies inside the retained allocation.
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
            Ok(())
        })?;
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
        let element_offset = usize::try_from(COUNTER_BUFFER_ELEMENT_OFFSET)
            .map_err(|_| HalError::InvalidArgument)?;
        for command_index in 0..draw.draw_count {
            // SAFETY: `NativeGraphicsDraw` supplies a four-byte count followed by
            // padding to the shared element offset and `draw_count` contiguous
            // 20-byte indirect records, while retaining both buffers.
            unsafe {
                encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                    primitive,
                    MTLIndexType::UInt32,
                    &draw.index.buffer,
                    0,
                    &draw.indirect.buffer,
                    element_offset + command_index as usize * 20,
                );
            };
        }

        Ok(())
    }

    pub(super) fn mesh(
        &mut self,
        action_index: usize,
        draw: &super::super::NativeMeshDraw<'_>,
    ) -> Result<(), HalError> {
        let encoder = self
            .render_encoder
            .as_ref()
            .ok_or(HalError::InvalidArgument)?;
        let NativePipeline::Mesh {
            state,
            object_argument_encoder,
            mesh_argument_encoder,
            fragment_argument_encoder,
            task_buffer_layouts,
            mesh_buffer_layouts,
            fragment_buffer_layouts,
            ..
        } = draw.pipeline
        else {
            return Err(HalError::InvalidArgument);
        };
        let argument_buffer = super::prepared_argument(self.prepared_arguments, action_index)
            .map(|index| &self.frame_slot.argument_buffers[index]);
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
        draw.bindings.visit(&mut |_, binding| {
            if binding.offset as u64 >= binding.allocation.allocation.size() {
                return Err(HalError::InvalidArgument);
            }
            // SAFETY: the offset lies inside the retained allocation. Each stage receives only
            // the physical binding intervals declared by its own reflection.
            unsafe {
                if buffer_stage_uses_binding(task_buffer_layouts, binding.index) {
                    encoder.setObjectBuffer_offset_atIndex(
                        Some(&binding.allocation.buffer),
                        binding.offset,
                        binding.index,
                    );
                }
                if buffer_stage_uses_binding(mesh_buffer_layouts, binding.index) {
                    encoder.setMeshBuffer_offset_atIndex(
                        Some(&binding.allocation.buffer),
                        binding.offset,
                        binding.index,
                    );
                }
                if buffer_stage_uses_binding(fragment_buffer_layouts, binding.index) {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(&binding.allocation.buffer),
                        binding.offset,
                        binding.index,
                    );
                }
            }
            Ok(())
        })?;
        match (draw.texture_heap, argument_buffer) {
            (Some(heap), Some(buffer)) => {
                for texture in draw.textures {
                    let resource = <ProtocolObject<dyn MTLTexture> as AsRef<
                        ProtocolObject<dyn MTLResource>,
                    >>::as_ref(&*texture.texture);
                    if object_argument_encoder.is_some() {
                        encoder.useResource_usage_stages(
                            resource,
                            MTLResourceUsage::Read,
                            MTLRenderStages::Object,
                        );
                    }
                    if mesh_argument_encoder.is_some() {
                        encoder.useResource_usage_stages(
                            resource,
                            MTLResourceUsage::Read,
                            MTLRenderStages::Mesh,
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
                // SAFETY: frame preparation encoded one complete argument buffer and retains it
                // through command completion; only stages with reflected encoders receive it.
                unsafe {
                    if object_argument_encoder.is_some() {
                        encoder.setObjectBuffer_offset_atIndex(
                            Some(buffer),
                            0,
                            heap.binding as usize,
                        );
                    }
                    if mesh_argument_encoder.is_some() {
                        encoder.setMeshBuffer_offset_atIndex(
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
                if object_argument_encoder.is_none()
                    && mesh_argument_encoder.is_none()
                    && fragment_argument_encoder.is_none() => {}
            _ => return Err(HalError::InvalidArgument),
        }
        encoder.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            super::metal_size(draw.groups),
            super::metal_size(object_threadgroup_size(draw.task_threads_per_group)),
            super::metal_size(draw.mesh_threads_per_group),
        );
        Ok(())
    }
}
