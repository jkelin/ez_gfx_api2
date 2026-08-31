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
        Ok(NativePipeline::Compute {
            state: ThreadBound::new(state),
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
        texture_heap: Option<ShaderTextureHeapLayout>,
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
        }
        let pipeline = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|_| HalError::NativeFailure)?;
        // SAFETY: `heap.binding` identifies the texture heap's fragment argument-buffer slot, and `fragment` keeps the `MTLFunction` storage allocated through `newArgumentEncoderWithBufferIndex:`.
        let argument_encoder = texture_heap.map(|heap| unsafe {
            fragment.newArgumentEncoderWithBufferIndex(heap.binding as usize)
        });
        Ok(NativePipeline::Graphics {
            state: ThreadBound::new(pipeline),
            argument_encoder: argument_encoder.map(ThreadBound::new),
        })
    }

    /// Defers pipeline destruction until every referencing frame completes.
    pub fn destroy_pipeline(&mut self, pipeline: NativePipeline) {
        let _ = self.defer_resource(DeferredResource::Pipeline(pipeline));
    }
}
