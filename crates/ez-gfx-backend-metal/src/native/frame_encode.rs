//! Metal frame graphics-draw encoding on one render command encoder.

use super::{
    CullMode, FrontFace, HalError, MTLCullMode, MTLIndexType, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderStages, MTLResource, MTLResourceUsage, MTLTexture,
    MTLWinding, MetalFrameEncoder, NativePipeline, PrimitiveTopology, ProtocolObject,
};
use ez_gfx_hal::COUNTER_BUFFER_ELEMENT_OFFSET;

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
}
