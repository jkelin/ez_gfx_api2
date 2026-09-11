use super::{
    BlendMode, DeferredResource, DynamicPipelineState, HalError, MTLBlendFactor, MTLDevice,
    MTLFunction, MTLLibrary, MTLMeshRenderPipelineDescriptor, MTLPipelineOption, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, NSString, NativeContext, NativePipeline,
    NativeShader, PrimitiveTopology, ShaderTextureHeapLayout, ThreadBound,
};
use objc2_metal::MTLGPUFamily;

fn validate_mesh_capabilities(
    capabilities: ez_gfx_core::capability::ShaderCapabilities,
    has_task: bool,
) -> Result<(), HalError> {
    if !crate::mesh_pipeline_capabilities_supported(capabilities, has_task) {
        return Err(HalError::Unsupported);
    }

    Ok(())
}

/// Keeps malformed dispatch metadata distinct from a valid shader unsupported by this device.
pub(super) const fn map_mesh_dispatch_error(error: ez_gfx_hal::MeshDispatchError) -> HalError {
    match error {
        ez_gfx_hal::MeshDispatchError::InvalidGroups
        | ez_gfx_hal::MeshDispatchError::InvalidWorkgroup => HalError::InvalidArgument,
        ez_gfx_hal::MeshDispatchError::UnsupportedWorkgroup => HalError::Unsupported,
    }
}

impl NativeContext {
    /// Returns coarse device limits for a direct mesh dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::Unsupported`] when mesh shaders or a requested task stage are unavailable.
    pub fn mesh_dispatch_limits(
        &self,
        has_task: bool,
    ) -> Result<ez_gfx_hal::MeshDispatchLimits, HalError> {
        validate_mesh_capabilities(self.adapter.capabilities().shader_stages, has_task)?;
        let coarse = self.max_threads_per_threadgroup;
        let coarse_total = coarse
            .width
            .checked_mul(coarse.height)
            .and_then(|value| value.checked_mul(coarse.depth))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(HalError::Unsupported)?;

        let max_groups =
            crate::metal_mesh_group_limit(self.device.supportsFamily(MTLGPUFamily::Apple9));
        Ok(crate::metal_mesh_dispatch_limits(
            max_groups,
            coarse_total,
            coarse_total,
        ))
    }
}

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

    // Surface pipelines use BGRA sRGB; depth, compressed, and storage formats are invalid here.
    fn render_pipeline_color_format(
        format: Option<ez_gfx_runtime::target::Format>,
    ) -> Result<MTLPixelFormat, HalError> {
        use ez_gfx_runtime::target::Format;
        match format {
            None => Ok(MTLPixelFormat::BGRA8Unorm_sRGB),
            Some(Format::Rgba8Unorm) => Ok(MTLPixelFormat::RGBA8Unorm),
            Some(Format::Bgra8Srgb) => Ok(MTLPixelFormat::BGRA8Unorm_sRGB),
            Some(Format::Rgba16Float) => Ok(MTLPixelFormat::RGBA16Float),
            Some(_) => Err(HalError::Unsupported),
        }
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
        vertex_shader: &NativeShader,
        fragment_shader: &NativeShader,
        vertex: &(usize, String),
        fragment: &(usize, String),
        state: DynamicPipelineState,
        color_format: Option<ez_gfx_runtime::target::Format>,
        depth_required: bool,
        vertex_texture_heap: Option<ShaderTextureHeapLayout>,
        fragment_texture_heap: Option<ShaderTextureHeapLayout>,
    ) -> Result<NativePipeline, HalError> {
        if vertex.1.is_empty()
            || vertex.1.as_bytes().contains(&0)
            || fragment.1.is_empty()
            || fragment.1.as_bytes().contains(&0)
            || state.topology == PrimitiveTopology::TriangleFan
        {
            return Err(HalError::InvalidArgument);
        }
        let vertex_library = vertex_shader
            .libraries
            .get(vertex.0)
            .ok_or(HalError::InvalidArgument)?;
        let fragment_library = fragment_shader
            .libraries
            .get(fragment.0)
            .ok_or(HalError::InvalidArgument)?;
        let vertex = vertex_library
            .newFunctionWithName(&NSString::from_str(&vertex.1))
            .ok_or(HalError::InvalidArgument)?;
        let fragment = fragment_library
            .newFunctionWithName(&NSString::from_str(&fragment.1))
            .ok_or(HalError::InvalidArgument)?;
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        // SAFETY: Metal defines eight color-attachment slots, so index 0 is valid, and `descriptor` keeps the attachment-array storage allocated through `objectAtIndexedSubscript:`.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setPixelFormat(Self::render_pipeline_color_format(color_format)?);
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

    /// Creates a precompiled task/mesh/fragment render pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid named products, thread limits, or native pipeline rejection.
    #[allow(
        clippy::too_many_arguments,
        reason = "three independently selected shader products and reflected stage layouts form the native PSO boundary"
    )]
    pub fn create_mesh_pipeline(
        &self,
        task_shader: Option<&NativeShader>,
        mesh_shader: &NativeShader,
        fragment_shader: &NativeShader,
        task: Option<&(usize, String)>,
        mesh: &(usize, String),
        fragment: &(usize, String),
        state: ez_gfx_hal::MeshPipelineState,
        color_format: Option<ez_gfx_runtime::target::Format>,
        depth_required: bool,
        task_texture_heap: Option<ShaderTextureHeapLayout>,
        mesh_texture_heap: Option<ShaderTextureHeapLayout>,
        fragment_texture_heap: Option<ShaderTextureHeapLayout>,
        task_buffer_layouts: &[ez_gfx_hal::ShaderBufferLayout],
        mesh_buffer_layouts: &[ez_gfx_hal::ShaderBufferLayout],
        fragment_buffer_layouts: &[ez_gfx_hal::ShaderBufferLayout],
        task_threads: Option<[u32; 3]>,
        mesh_threads: [u32; 3],
    ) -> Result<NativePipeline, HalError> {
        validate_mesh_capabilities(
            self.adapter.capabilities().shader_stages,
            task_shader.is_some(),
        )?;
        if task_shader.is_some() != task.is_some()
            || task.is_some() != task_threads.is_some()
            || mesh.1.is_empty()
            || mesh.1.as_bytes().contains(&0)
            || fragment.1.is_empty()
            || fragment.1.as_bytes().contains(&0)
            || task.is_some_and(|entry| entry.1.is_empty() || entry.1.as_bytes().contains(&0))
        {
            return Err(HalError::InvalidArgument);
        }
        let shared_limits = self.mesh_dispatch_limits(task_shader.is_some())?;
        let coarse = self.max_threads_per_threadgroup;
        for dimensions in task_threads
            .into_iter()
            .chain(core::iter::once(mesh_threads))
        {
            if usize::try_from(dimensions[0]).map_or(true, |value| value > coarse.width)
                || usize::try_from(dimensions[1]).map_or(true, |value| value > coarse.height)
                || usize::try_from(dimensions[2]).map_or(true, |value| value > coarse.depth)
            {
                return Err(HalError::Unsupported);
            }
        }
        ez_gfx_hal::validate_mesh_dispatch([1, 1, 1], mesh_threads, task_threads, shared_limits)
            .map_err(map_mesh_dispatch_error)?;

        let resolve = |shader: &NativeShader, identity: &(usize, String)| {
            shader
                .libraries
                .get(identity.0)
                .ok_or(HalError::InvalidArgument)?
                .newFunctionWithName(&NSString::from_str(&identity.1))
                .ok_or(HalError::InvalidArgument)
        };
        let object = task_shader
            .zip(task)
            .map(|(shader, identity)| resolve(shader, identity))
            .transpose()?;
        let mesh_function = resolve(mesh_shader, mesh)?;
        let fragment_function = resolve(fragment_shader, fragment)?;
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: artifacts were validated for the exact task/mesh/fragment stages and names.
        unsafe {
            descriptor.setObjectFunction(object.as_deref());
            descriptor.setMeshFunction(Some(&mesh_function));
            descriptor.setFragmentFunction(Some(&fragment_function));
        }
        let mesh_total = mesh_threads
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .ok_or(HalError::InvalidArgument)?;
        let task_total = task_threads
            .map(|threads| {
                threads
                    .into_iter()
                    .try_fold(1_u32, u32::checked_mul)
                    .ok_or(HalError::InvalidArgument)
            })
            .transpose()?;
        descriptor.setMaxTotalThreadsPerMeshThreadgroup(mesh_total as usize);
        descriptor.setMaxTotalThreadsPerObjectThreadgroup(task_total.unwrap_or(1) as usize);
        // SAFETY: Metal defines color attachment slot zero for mesh render descriptors.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setPixelFormat(Self::render_pipeline_color_format(color_format)?);
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
            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| {
                // SAFETY: Metal exports this process-lifetime immutable NSError domain.
                let library_error_domain = unsafe { super::MTLLibraryErrorDomain };
                let is_library_error = &*error.domain() == library_error_domain;
                let description = error.localizedDescription().to_string();
                if crate::is_mesh_thread_limit_pipeline_error(
                    is_library_error,
                    error.code(),
                    &description,
                ) {
                    HalError::Unsupported
                } else {
                    HalError::NativeFailure
                }
            })?;
        let max_mesh_threads = u32::try_from(pipeline.maxTotalThreadsPerMeshThreadgroup())
            .map_err(|_| HalError::Unsupported)?;
        let max_task_threads = u32::try_from(pipeline.maxTotalThreadsPerObjectThreadgroup())
            .map_err(|_| HalError::Unsupported)?;
        let dispatch_limits = ez_gfx_hal::MeshDispatchLimits {
            max_mesh_threads,
            max_task_threads,
            ..shared_limits
        };
        ez_gfx_hal::validate_mesh_dispatch([1, 1, 1], mesh_threads, task_threads, dispatch_limits)
            .map_err(map_mesh_dispatch_error)?;

        let object_argument_encoder = object.as_ref().and_then(|function| {
            task_texture_heap.map(|heap| {
                // SAFETY: reflection validated the object argument-buffer index.
                unsafe { function.newArgumentEncoderWithBufferIndex(heap.binding as usize) }
            })
        });
        let mesh_argument_encoder = mesh_texture_heap.map(|heap| {
            // SAFETY: reflection validated the mesh argument-buffer index.
            unsafe { mesh_function.newArgumentEncoderWithBufferIndex(heap.binding as usize) }
        });
        let fragment_argument_encoder = fragment_texture_heap.map(|heap| {
            // SAFETY: reflection validated the fragment argument-buffer index.
            unsafe { fragment_function.newArgumentEncoderWithBufferIndex(heap.binding as usize) }
        });
        Ok(NativePipeline::Mesh {
            state: ThreadBound::new(pipeline),
            object_argument_encoder: object_argument_encoder.map(ThreadBound::new),
            mesh_argument_encoder: mesh_argument_encoder.map(ThreadBound::new),
            fragment_argument_encoder: fragment_argument_encoder.map(ThreadBound::new),
            task_buffer_layouts: task_buffer_layouts.to_vec(),
            mesh_buffer_layouts: mesh_buffer_layouts.to_vec(),
            fragment_buffer_layouts: fragment_buffer_layouts.to_vec(),
            dispatch_limits,
        })
    }

    /// Defers pipeline destruction until every referencing frame completes.
    pub fn destroy_pipeline(&mut self, pipeline: NativePipeline) {
        let _ = self.defer_resource(DeferredResource::Pipeline(pipeline));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DynamicPipelineState, NativeContext, NativePipeline, ShaderTextureHeapLayout,
        map_mesh_dispatch_error,
    };
    use ez_gfx_artifact::Stage;
    use ez_gfx_compiler::{EasyGraphicsCompiler, Target};
    use ez_gfx_core::{Backend, capability::SemanticProfile};
    use ez_gfx_hal::{HalError, MeshDispatchError};
    use ez_gfx_runtime::shader::RuntimeShader;

    #[test]
    fn mesh_dispatch_error_mapping_preserves_argument_and_support_classes() {
        for error in [
            MeshDispatchError::InvalidGroups,
            MeshDispatchError::InvalidWorkgroup,
        ] {
            assert_eq!(map_mesh_dispatch_error(error), HalError::InvalidArgument);
        }
        assert_eq!(
            map_mesh_dispatch_error(MeshDispatchError::UnsupportedWorkgroup),
            HalError::Unsupported
        );
    }

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

        let artifact =
            EasyGraphicsCompiler::compile_shader(&source, &[Target::Metal], false).unwrap();
        let vertex = RuntimeShader::load(
            &artifact,
            Backend::Metal,
            SemanticProfile::V1,
            Stage::Vertex,
            "vertexmain",
        )
        .unwrap();
        let fragment = RuntimeShader::load(
            &artifact,
            Backend::Metal,
            SemanticProfile::V1,
            Stage::Fragment,
            "fragmentmain",
        )
        .unwrap();
        let vertex_layout = vertex.pipeline_layout(Stage::Vertex).unwrap();
        let fragment_layout = fragment.pipeline_layout(Stage::Fragment).unwrap();
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
        let vertex_products = vertex
            .products()
            .map(|(_, bytes)| bytes)
            .collect::<Vec<_>>();
        let fragment_products = fragment
            .products()
            .map(|(_, bytes)| bytes)
            .collect::<Vec<_>>();
        let vertex_product = vertex.shader_product();
        let fragment_product = fragment.shader_product();
        let vertex_identity = (vertex_product.0, vertex_product.2.to_owned());
        let fragment_identity = (fragment_product.0, fragment_product.2.to_owned());

        let mut context = NativeContext::create_default().unwrap();
        let vertex_shader = context.create_shader(&vertex_products).unwrap();
        let fragment_shader = context.create_shader(&fragment_products).unwrap();
        let pipeline = context
            .create_graphics_pipeline(
                &vertex_shader,
                &fragment_shader,
                &vertex_identity,
                &fragment_identity,
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                None,
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
        context.destroy_shader(vertex_shader);
        context.destroy_shader(fragment_shader);
        context.wait_idle().unwrap();
    }
}
