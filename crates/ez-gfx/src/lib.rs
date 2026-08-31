#![doc = include_str!("../README.md")]

mod api;
mod state;

pub use api::*;
pub use ez_gfx_artifact::Stage;
pub use ez_gfx_core::handle::{HandleParts, PackedHandle};
pub use ez_gfx_core::{Backend, SemanticId};
pub use ez_gfx_hal::{DynamicPipelineState, SamplerAddressMode, SamplerFilter, TextureSamplerDesc};
pub use ez_gfx_runtime::binding::{PublicBinding, ResourceIdentity};
pub use ez_gfx_runtime::indirect::DrawIndexedCommand;
pub use ez_gfx_runtime::shader::ShaderRequest;
pub use ez_gfx_runtime::texture::TextureSource;
pub use ez_gfx_runtime::{ContextOptions, SurfaceOptions, SurfacePlatform};
pub use state::*;

/// Submits the recorded frame and presents its active surface; a failed submission is never followed by presentation.
pub fn finish_render(context: u64) -> EzGfxResult {
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
