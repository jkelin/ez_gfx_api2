use super::{
    BlendMode, DeferredResource, DynamicPipelineState, HalError, MTLBlendFactor, MTLDevice,
    MTLFunction, MTLLibrary, MTLPixelFormat, MTLRenderPipelineDescriptor, NSString, NativeContext,
    NativePipeline, NativeShader, PrimitiveTopology, ShaderTextureHeapLayout, ThreadBound,
};

impl NativeContext {
    /// Loads precompiled metallib products directly; source compilation is intentionally absent.
    ///
    /// # Errors
    ///
    /// Returns an error for empty products or rejected Metal libraries.
    pub fn create_shader(&self, products: &[&[u8]]) -> Result<NativeShader, HalError> {
        if products.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        let mut libraries = Vec::with_capacity(products.len());
        for product in products {
            if product.is_empty() {
                return Err(HalError::InvalidArgument);
            }
            let data = dispatch2::DispatchData::from_bytes(product);
            libraries.push(
                self.device
                    .newLibraryWithData_error(&data)
                    .map_err(|_| HalError::NativeFailure)?,
            );
        }
        Ok(NativeShader {
            libraries: ThreadBound::new(libraries),
        })
    }

    /// Defers shader destruction until every referencing frame completes.
    pub fn destroy_shader(&mut self, shader: NativeShader) {
        let _ = self.defer_resource(DeferredResource::Shader(shader));
    }

    /// Resolves a named function from a precompiled metallib and creates its compute PSO.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid entry or rejected pipeline state.
    pub fn create_compute_pipeline(
        &self,
        shader: &NativeShader,
        product_index: usize,
        entry: &str,
        texture_heap: Option<ShaderTextureHeapLayout>,
    ) -> Result<NativePipeline, HalError> {
        if entry.is_empty() || entry.as_bytes().contains(&0) {
            return Err(HalError::InvalidArgument);
        }
        let library = shader
            .libraries
            .get(product_index)
            .ok_or(HalError::InvalidArgument)?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or(HalError::InvalidArgument)?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| HalError::NativeFailure)?;
        let argument_encoder = texture_heap.map(|heap| {
            // SAFETY: reflection validated the compute argument-buffer index against the selected entry point.
            unsafe { function.newArgumentEncoderWithBufferIndex(heap.binding as usize) }
        });
        Ok(NativePipeline::Compute {
            state: ThreadBound::new(state),
            argument_encoder: argument_encoder.map(ThreadBound::new),
        })
    }

    /// Creates a graphics pipeline from the requested shader entries and render state.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid entries, unsupported state, or rejected pipeline state.
    pub fn create_graphics_pipeline(
        &self,
        shader: &NativeShader,
        graphics: &(usize, String, usize, String),
        state: DynamicPipelineState,
        depth_required: bool,
        vertex_texture_heap: Option<ShaderTextureHeapLayout>,
        fragment_texture_heap: Option<ShaderTextureHeapLayout>,
    ) -> Result<NativePipeline, HalError> {
        if graphics.1.is_empty()
            || graphics.1.as_bytes().contains(&0)
            || graphics.3.is_empty()
            || graphics.3.as_bytes().contains(&0)
            || state.topology == PrimitiveTopology::TriangleFan
        {
            return Err(HalError::InvalidArgument);
        }
        let vertex_library = shader
            .libraries
            .get(graphics.0)
            .ok_or(HalError::InvalidArgument)?;
        let fragment_library = shader
            .libraries
            .get(graphics.2)
            .ok_or(HalError::InvalidArgument)?;
        let vertex = vertex_library
            .newFunctionWithName(&NSString::from_str(&graphics.1))
            .ok_or(HalError::InvalidArgument)?;
        let fragment = fragment_library
            .newFunctionWithName(&NSString::from_str(&graphics.3))
            .ok_or(HalError::InvalidArgument)?;
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        // SAFETY: Metal defines eight color-attachment slots, so index 0 is valid, and `descriptor` keeps the attachment-array storage allocated through `objectAtIndexedSubscript:`.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        if depth_required {
            descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
        }
        if state.blend == BlendMode::Alpha {
            color.setBlendingEnabled(true);
            color.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
            color.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            color.setSourceAlphaBlendFactor(MTLBlendFactor::One);
            color.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        }
        let pipeline = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|_| HalError::NativeFailure)?;
        // SAFETY: reflection validated the vertex argument-buffer index.
        let vertex_argument_encoder = vertex_texture_heap
            .map(|heap| unsafe { vertex.newArgumentEncoderWithBufferIndex(heap.binding as usize) });
        // SAFETY: reflection validated the fragment argument-buffer index.
        let fragment_argument_encoder = fragment_texture_heap.map(|heap| unsafe {
            fragment.newArgumentEncoderWithBufferIndex(heap.binding as usize)
        });
        Ok(NativePipeline::Graphics {
            state: ThreadBound::new(pipeline),
            vertex_argument_encoder: vertex_argument_encoder.map(ThreadBound::new),
            fragment_argument_encoder: fragment_argument_encoder.map(ThreadBound::new),
        })
    }

    /// Defers pipeline destruction until every referencing frame completes.
    pub fn destroy_pipeline(&mut self, pipeline: NativePipeline) {
        let _ = self.defer_resource(DeferredResource::Pipeline(pipeline));
    }
}

#[cfg(test)]
mod tests {
    use super::{DynamicPipelineState, NativeContext, NativePipeline, ShaderTextureHeapLayout};
    use ez_gfx_artifact::Stage;
    use ez_gfx_compiler::{Target, compile_shader};
    use ez_gfx_core::{Backend, capability::SemanticProfile};
    use ez_gfx_runtime::shader::RuntimeShader;

    #[test]
    fn graphics_pipeline_creates_argument_encoders_for_both_heap_users() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("graphics_texture_heap.slang");
        std::fs::write(
            &source,
            r#"[__AttributeUsage(_AttributeTargets.Var)]
struct BindlessTextureHeapAttribute { int capacity; };
struct TextureEntry { Texture2D<float4> texture; SamplerState sampler; };
struct TextureHeap { TextureEntry entries[1024]; };

[BindlessTextureHeap(1024)]
ParameterBlock<TextureHeap> texture_heap;

struct VertexOutput { float4 position : SV_Position; };

[shader("vertex")]
VertexOutput vertexmain(uint id : SV_VertexID) {
    float sampled = texture_heap.entries[0].texture.SampleLevel(
        texture_heap.entries[0].sampler, float2(0.5, 0.5), 0).r;
    float2 positions[3] = {
        float2(-0.5, -0.5),
        float2(0.5, -0.5),
        float2(0.0, 0.5)
    };
    VertexOutput output;
    output.position = float4(positions[id] + float2(sampled * 0.01, 0.0), 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragmentmain() : SV_Target {
    return texture_heap.entries[0].texture.SampleLevel(
        texture_heap.entries[0].sampler, float2(0.5, 0.5), 0);
}
"#,
        )
        .unwrap();

        let artifact = compile_shader(&source, &[Target::Metal], false).unwrap();
        let runtime = RuntimeShader::load(&artifact, Backend::Metal, SemanticProfile::V1).unwrap();
        let vertex_layout = runtime.pipeline_layout(Stage::Vertex).unwrap();
        let fragment_layout = runtime.pipeline_layout(Stage::Fragment).unwrap();
        let vertex_heap = vertex_layout.texture_heap().unwrap();
        let fragment_heap = fragment_layout.texture_heap().unwrap();
        assert_eq!(vertex_heap, fragment_heap);
        let heap = ShaderTextureHeapLayout::new(
            vertex_heap.space,
            vertex_heap.binding,
            vertex_heap.capacity,
            vertex_heap.argument_stride,
            vertex_heap.texture_argument_offset,
            vertex_heap.sampler_argument_offset,
        )
        .unwrap();
        let products = runtime
            .products()
            .map(|(_, bytes)| bytes)
            .collect::<Vec<_>>();
        let (vertex, fragment) = runtime.graphics_pair().unwrap();
        let graphics = (
            vertex.0,
            vertex.2.to_owned(),
            fragment.0,
            fragment.2.to_owned(),
        );

        let mut context = NativeContext::create_default().unwrap();
        let shader = context.create_shader(&products).unwrap();
        let pipeline = context
            .create_graphics_pipeline(
                &shader,
                &graphics,
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                false,
                Some(heap),
                Some(heap),
            )
            .unwrap();
        let NativePipeline::Graphics {
            vertex_argument_encoder,
            fragment_argument_encoder,
            ..
        } = &pipeline
        else {
            panic!("expected graphics pipeline");
        };
        assert!(vertex_argument_encoder.is_some());
        assert!(fragment_argument_encoder.is_some());

        context.destroy_pipeline(pipeline);
        context.destroy_shader(shader);
        context.wait_idle().unwrap();
    }
}
