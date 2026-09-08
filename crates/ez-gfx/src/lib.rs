#![doc = include_str!("../README.md")]

mod api;
mod state;

pub use api::*;
pub use ez_gfx_artifact::Stage;
pub use ez_gfx_core::capability::{AdapterClass, AdapterInfo, CapabilityError};
pub use ez_gfx_core::{Backend, SemanticId};
pub use ez_gfx_hal::{
    DynamicPipelineState, SamplerAddressMode, SamplerFilter, TextureFormat, TextureRegion,
    TextureSamplerDesc,
};
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
pub use state::TextureConfig;

/// Raw handle interface reserved for the C boundary.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub mod raw {
    pub use crate::state::*;
    pub use ez_gfx_core::handle::{
        ContextHandle, IndexAllocationHandle, IndirectBufferHandle, RenderTargetHandle,
        ShaderHandle, StructuredBufferHandle, SurfaceHandle, TextureHandle, VertexAllocationHandle,
        VertexHeapHandle,
    };
    pub use ez_gfx_runtime::binding::{PublicBinding, ResourceIdentity};

    /// Submits and presents the active raw surface frame.
    pub fn finish_render(context: ContextHandle) -> crate::Result<()> {
        frame_submit(context)?;
        present(context)
    }
}
