#![doc = include_str!("../README.md")]

mod api;
mod state;

pub use api::*;
pub use ez_gfx_artifact::Stage;
pub use ez_gfx_core::capability::{AdapterClass, AdapterInfo, CapabilityError};
pub use ez_gfx_core::handle::{
    ContextHandle, IndexAllocationHandle, IndirectBufferHandle, RenderTargetHandle, ShaderHandle,
    StructuredBufferHandle, SurfaceHandle, TextureHandle, VertexAllocationHandle, VertexHeapHandle,
};
pub use ez_gfx_core::{Backend, SemanticId};
pub use ez_gfx_hal::{
    DynamicPipelineState, SamplerAddressMode, SamplerFilter, TextureFormat, TextureRegion,
    TextureSamplerDesc,
};
pub use ez_gfx_runtime::binding::{PublicBinding, ResourceIdentity};
pub use ez_gfx_runtime::indirect::DrawIndexedCommand;
pub use ez_gfx_runtime::target::{ClearValue, Format, TargetDeclaration, TargetError, TargetUsage};
pub use ez_gfx_runtime::texture::{
    DecodedMip, DecodedTexture, TextureDecodeCallback, TextureDestination, TextureError,
    TextureSource, TextureUploadTelemetrySnapshot, register_texture_decoder,
    unregister_texture_decoder,
};
pub use ez_gfx_runtime::upload::{UploadEvent, UploadResource, UploadStatus};
pub use ez_gfx_runtime::{
    AdapterReport, AdapterSelection, ContextOptions, LifecycleError, SurfaceOptions,
    SurfacePlatform, admission_report,
};
pub use state::*;

/// Submits the recorded frame and presents its active surface; a failed submission is never followed by presentation.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn finish_render(context: ContextHandle) -> Result<()> {
    submit_then_present(|| frame_submit(context), || present(context))
}

// Submission failures propagate unchanged and must short-circuit presentation.
fn submit_then_present(
    submit: impl FnOnce() -> Result<()>,
    present: impl FnOnce() -> Result<()>,
) -> Result<()> {
    submit()?;
    present()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_submission_prevents_presentation() {
        let mut presented = false;
        let status = submit_then_present(
            || Err(Error::NativeFailure),
            || {
                presented = true;
                Ok(())
            },
        );

        assert_eq!(status, Err(Error::NativeFailure));
        assert!(!presented);
    }
}
