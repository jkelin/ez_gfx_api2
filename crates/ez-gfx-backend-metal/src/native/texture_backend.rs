//! Manager-trait adapters for the backend texture pipeline.

use super::{
    AllocationError, CompletionToken, ImageMip, NativeContext, NativeTexture, TextureFormat,
    TextureRegion, TextureSamplerDesc,
};

impl ez_gfx_texture_manager::MipTransferValues for NativeTexture {
    /// Delegates to the inherent storage-order accessor; fully-qualified so
    /// this never resolves to the trait method itself.
    fn mip_transfer_values(&self) -> &[u64] {
        NativeTexture::mip_transfer_values(self)
    }
}

impl ez_gfx_texture_manager::TextureBackendTexture for NativeTexture {
    /// Delegates to the inherent latest-transfer accessor.
    fn last_transfer_value(&self) -> u64 {
        NativeTexture::last_transfer_value(self)
    }
}

/// Exposes the Metal texture pipeline behind the backend-neutral manager
/// traits. Every method forwards to the inherent implementation; the manager
/// crate sees only HAL types and the opaque associated texture.
impl ez_gfx_texture_manager::TextureBackendContext for NativeContext {
    type Texture = NativeTexture;

    fn create_texture_with_prefix(
        &mut self,
        format: TextureFormat,
        mips: &[ImageMip<'_>],
        binding: u32,
        sampler: TextureSamplerDesc,
        prefix: u32,
    ) -> Result<(Self::Texture, Vec<CompletionToken>), AllocationError> {
        NativeContext::create_texture_with_prefix(self, format, mips, binding, sampler, prefix)
    }

    fn update_texture_region(
        &mut self,
        texture: &mut Self::Texture,
        region: &TextureRegion<'_>,
    ) -> Result<CompletionToken, AllocationError> {
        NativeContext::update_texture_region(self, texture, region)
    }

    fn reference_texture_prefix(
        &mut self,
        texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), AllocationError> {
        NativeContext::reference_texture_prefix(self, texture, resident_mips)
    }

    fn publish_texture_mips(
        &mut self,
        texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), AllocationError> {
        NativeContext::publish_texture_mips(self, texture, resident_mips)
    }

    fn texture_descriptors_ready(&self) -> Result<bool, AllocationError> {
        // The Metal gate is an infallible in-flight mask read.
        Ok(NativeContext::texture_descriptor_update_ready(self))
    }

    fn texture_retirement_ready(
        &self,
        completion: CompletionToken,
    ) -> Result<bool, AllocationError> {
        NativeContext::texture_retirement_ready(self, completion)
    }

    fn destroy_texture(&mut self, texture: Self::Texture) -> Result<(), AllocationError> {
        NativeContext::destroy_texture(self, texture)
    }

    fn completed_texture_transfer_value(&self) -> Result<u64, AllocationError> {
        NativeContext::completed_texture_transfer_value(self)
    }

    fn cancel_texture_transfers(texture: &Self::Texture) {
        NativeContext::cancel_texture_transfers(texture);
    }
}
