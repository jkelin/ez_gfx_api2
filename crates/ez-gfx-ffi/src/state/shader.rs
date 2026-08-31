use super::{
    EzGfxResult, NativeContext, NativePipeline, NativeShader, PackedHandle, ResourceKind,
    ShaderRecord, map_hal, map_lifecycle, with_context_mut,
};

pub fn load_shader(
    context: u64,
    artifact: &[u8],
    requests: &[ez_gfx_runtime::shader::ShaderRequest],
) -> Result<PackedHandle, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let digest = *blake3::hash(artifact).as_bytes();
        let shader = ez_gfx_runtime::shader::RuntimeShader::load(
            artifact,
            context.options.backend,
            ez_gfx_core::capability::SemanticProfile::V1,
            requests,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        let graphics = shader.graphics_pair().ok().map(|(vertex, fragment)| {
            (
                vertex.0,
                vertex.2.to_owned(),
                fragment.0,
                fragment.2.to_owned(),
            )
        });
        let compute = shader
            .compute_product()
            .ok()
            .map(|product| (product.0, product.2.to_owned()));
        let graphics_layout = graphics
            .as_ref()
            .map(|graphics| {
                ez_gfx_runtime::binding::PipelineLayout::parse(
                    shader.metadata(),
                    context.options.backend,
                    &graphics.3,
                    ez_gfx_artifact::Stage::Fragment,
                )
            })
            .transpose()
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        if graphics.is_none() && compute.is_none() {
            return Err(EzGfxResult::InvalidArgument);
        }
        let products = shader
            .products()
            .map(|(_, bytes)| bytes)
            .collect::<Vec<_>>();
        let native = match &context.native {
            NativeContext::Vulkan(native) => {
                NativeShader::Vulkan(native.create_shader(&products).map_err(map_hal)?)
            }
            #[cfg(windows)]
            NativeContext::Dx12(native) => {
                NativeShader::Dx12(native.create_shader(&products).map_err(map_hal)?)
            }
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => {
                NativeShader::Metal(native.create_shader(&products).map_err(map_hal)?)
            }
        };
        let handle = match context.identity.insert(ResourceKind::Shader) {
            Ok(handle) => handle,
            Err(error) => {
                destroy_native_shader(&mut context.native, native);
                return Err(map_lifecycle(error));
            }
        };
        context.shaders.insert(
            handle.get(),
            ShaderRecord {
                native,
                digest,
                graphics,
                compute,
                runtime: shader,
                graphics_layout,
            },
        );
        Ok(handle)
    })
}

pub fn destroy_shader(context: u64, shader: u64) {
    if shader == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(shader).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        let stale = context
            .pipelines
            .extract_if(|key, _| key.shader() == shader)
            .map(|(_, pipeline)| pipeline)
            .collect::<Vec<_>>();
        for pipeline in stale {
            destroy_native_pipeline(&mut context.native, pipeline);
        }
        let native = context
            .shaders
            .remove(&shader)
            .ok_or(EzGfxResult::InvalidContext)?;
        destroy_native_shader(&mut context.native, native.native);
        Ok(())
    });
}

pub(super) fn destroy_native_shader(context: &mut NativeContext, shader: NativeShader) {
    match (context, shader) {
        (NativeContext::Vulkan(context), NativeShader::Vulkan(shader)) => {
            context.destroy_shader(shader);
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeShader::Dx12(shader)) => {
            context.destroy_shader(shader);
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeShader::Metal(shader)) => {
            context.destroy_shader(shader)
        }
        _ => {}
    }
}

pub(super) fn destroy_native_pipeline(context: &mut NativeContext, pipeline: NativePipeline) {
    match (context, pipeline) {
        (NativeContext::Vulkan(context), NativePipeline::Vulkan(pipeline)) => {
            context.destroy_pipeline(pipeline);
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativePipeline::Dx12(pipeline)) => {
            context.destroy_pipeline(pipeline);
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativePipeline::Metal(pipeline)) => {
            context.destroy_pipeline(pipeline)
        }
        _ => {}
    }
}
