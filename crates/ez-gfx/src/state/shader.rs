use crate::Result;

use super::{
    ContextHandle, Error, NativeContext, NativePipeline, NativeShader, ResourceKind, ShaderHandle,
    ShaderRecord, map_hal, map_lifecycle, with_context_mut,
};

fn validate_stage_capability(
    stage: ez_gfx_artifact::Stage,
    capabilities: ez_gfx_core::capability::ShaderCapabilities,
) -> Result<()> {
    // Mesh is required independently; task additionally requires its own capability bit.
    let supported = match stage {
        ez_gfx_artifact::Stage::Task => capabilities.task,
        ez_gfx_artifact::Stage::Mesh => capabilities.mesh,
        _ => true,
    };
    supported.then_some(()).ok_or(Error::Unsupported)
}

/// Loads one exact stage entry point from a validated shader artifact.
///
/// # Errors
///
/// Returns an error for invalid handles, malformed artifacts, unknown or wrong-stage entry names,
/// missing backend products, or native shader failure.
pub fn load_shader(
    context: ContextHandle,
    artifact: &[u8],
    stage: ez_gfx_artifact::Stage,
    entry_point: &str,
) -> Result<ShaderHandle> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if matches!(
            stage,
            ez_gfx_artifact::Stage::Task | ez_gfx_artifact::Stage::Mesh
        ) {
            let capabilities = super::context::context_shader_capabilities(context)?;
            validate_stage_capability(stage, capabilities)?;
        }
        let shader = ez_gfx_runtime::shader::RuntimeShader::load(
            artifact,
            context.options.backend,
            ez_gfx_core::capability::SemanticProfile::V1,
            stage,
            entry_point,
        )
        .map_err(|_| Error::InvalidArgument)?;
        let digest = shader.execution_digest();
        let product = shader.shader_product();
        let product_index = product.0;
        let entry = product.2.to_owned();
        let products = [shader.product(stage).ok_or(Error::InvalidArgument)?];
        #[cfg(test)]
        {
            context.native_shader_allocation_attempts += 1;
        }
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
        let typed = ShaderHandle::from_packed(handle).map_err(|_| Error::NativeFailure)?;
        context.shaders.insert(
            typed,
            ShaderRecord {
                native,
                digest,
                product: product_index,
                entry,
                stage,
                runtime: shader,
            },
        );
        Ok(typed)
    })
}
/// Validates mesh-stage handles without recording or mutating the active graph.
///
/// # Errors
///
/// Returns an error for stale, foreign-owner, wrong-generation, or wrong-stage handles.
pub(crate) fn validate_mesh_stages(
    context: ContextHandle,
    stages: ez_gfx_hal::MeshStages<ShaderHandle>,
) -> Result<()> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let expected = [
            stages
                .task
                .map(|shader| (shader, ez_gfx_artifact::Stage::Task)),
            Some((stages.mesh, ez_gfx_artifact::Stage::Mesh)),
            Some((stages.fragment, ez_gfx_artifact::Stage::Fragment)),
        ];
        for (shader, stage) in expected.into_iter().flatten() {
            context
                .identity
                .resolve(shader.packed(), ResourceKind::Shader)
                .map_err(map_lifecycle)?;
            if context.shaders.get(&shader).map(|record| record.stage) != Some(stage) {
                return Err(Error::InvalidContext);
            }
        }
        Ok(())
    })
}

/// Requests destruction of a loaded shader.
///
/// A shader referenced by the active raw/FFI frame is invalidated for callers immediately, while
/// its internal record and native objects remain alive until that frame submits or aborts. This
/// mirrors the safe frame's owning `Rc` retention.
pub fn destroy_shader(context: ContextHandle, shader: ShaderHandle) {
    let _ = with_context_mut(context, |context| {
        #[cfg(all(test, not(target_vendor = "apple")))]
        {
            context.shader_destroy_requests += 1;
        }
        context
            .identity
            .remove(shader.packed(), ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        if context.frame_shaders.contains(&shader) {
            context.pending_shader_destroys.insert(shader);
            return Ok(());
        }
        retire_shader_record(context, shader)
    });
}

#[cfg(all(test, not(target_vendor = "apple")))]
pub(crate) fn insert_shader_drop_probe(context: ContextHandle) -> Result<ShaderHandle> {
    with_context_mut(context, |context| {
        let packed = context
            .identity
            .insert(ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        ShaderHandle::from_packed(packed).map_err(|_| Error::NativeFailure)
    })
}

#[cfg(all(test, not(target_vendor = "apple")))]
pub(crate) fn shader_destroy_requests(context: ContextHandle) -> Result<usize> {
    with_context_mut(context, |context| Ok(context.shader_destroy_requests))
}
#[cfg(all(test, windows))]
pub(crate) fn native_shader_allocation_attempts(context: ContextHandle) -> Result<usize> {
    with_context_mut(context, |context| {
        Ok(context.native_shader_allocation_attempts)
    })
}

fn retire_shader_record(context: &mut super::ContextState, shader: ShaderHandle) -> Result<()> {
    let stale = context
        .pipelines
        .extract_if(|key, _| key.involves_shader(shader))
        .map(|(_, pipeline)| pipeline)
        .collect::<Vec<_>>();
    for pipeline in stale {
        destroy_native_pipeline(&mut context.native, pipeline);
    }
    let native = context
        .shaders
        .remove(&shader)
        .ok_or(Error::InvalidContext)?;
    #[cfg(test)]
    {
        context.native_shader_destroys += 1;
    }
    destroy_native_shader(&mut context.native, native.native);
    Ok(())
}

pub(super) fn release_frame_shaders(context: &mut super::ContextState) {
    context.frame_shaders.clear();
    let pending = context.pending_shader_destroys.drain().collect::<Vec<_>>();
    for shader in pending {
        let _ = retire_shader_record(context, shader);
    }
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
            context.destroy_shader(shader);
        }
        #[cfg(any(windows, target_vendor = "apple"))]
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
            context.destroy_pipeline(pipeline);
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => {}
    }
}
