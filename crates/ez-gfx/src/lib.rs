#![doc = include_str!("../README.md")]

mod api;
mod state;

pub use api::*;
pub use ez_gfx_artifact::Stage;
pub use ez_gfx_core::handle::{
    ContextHandle, IndirectBufferHandle, RenderTargetHandle, ShaderHandle, StructuredBufferHandle,
    SurfaceHandle, TextureHandle,
};
pub use ez_gfx_core::{Backend, SemanticId};
pub use ez_gfx_hal::{
    DynamicPipelineState, SamplerAddressMode, SamplerFilter, TextureFormat, TextureRegion,
    TextureSamplerDesc,
};
pub use ez_gfx_runtime::binding::{PublicBinding, ResourceIdentity};
pub use ez_gfx_runtime::indirect::DrawIndexedCommand;
pub use ez_gfx_runtime::texture::{
    DecodedMip, DecodedTexture, TextureDecodeCallback, TextureDestination, TextureError,
    TextureSource, TextureUploadTelemetrySnapshot, register_texture_decoder,
    unregister_texture_decoder,
};
pub use ez_gfx_runtime::{ContextOptions, SurfaceOptions, SurfacePlatform};
pub use state::*;

/// Submits the recorded frame and presents its active surface; a failed submission is never followed by presentation.
pub fn finish_render(context: ContextHandle) -> EzGfxResult {
    submit_then_present(|| frame_submit(context), || present(context))
}

// Submission failures propagate unchanged and must short-circuit presentation.
fn submit_then_present(
    submit: impl FnOnce() -> EzGfxResult,
    present: impl FnOnce() -> EzGfxResult,
) -> EzGfxResult {
    let submitted = submit();
    if submitted != EzGfxResult::Ok {
        return submitted;
    }

    present()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_submission_prevents_presentation() {
        let mut presented = false;
        let status = submit_then_present(
            || EzGfxResult::NativeFailure,
            || {
                presented = true;
                EzGfxResult::Ok
            },
        );

        assert_eq!(status, EzGfxResult::NativeFailure);
        assert!(!presented);
    }
}
