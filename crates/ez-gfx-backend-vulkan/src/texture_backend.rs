//! Manager-trait adapters and staging eviction for the Vulkan backend.

use super::{
    AllocationError, CompletionToken, ImageMip, NativeAllocation, NativeContext, NativeTexture,
    TextureFormat, TextureRegion, TextureSamplerDesc,
};

impl NativeContext {
    /// Returns retained texture-staging bucket capacity for global eviction.
    pub fn retained_texture_staging_bytes(&self) -> u64 {
        self.texture_staging.retained_bytes()
    }

    /// Returns the largest completed texture-staging bucket for global eviction.
    pub fn largest_texture_staging_completed(&self, completed: u64) -> Option<u64> {
        self.texture_staging
            .largest_completed_capacity(ez_gfx_hal::QueueKind::TextureTransfer, completed)
    }

    /// Removes the largest completed texture-staging bucket for global eviction.
    pub fn pop_largest_texture_staging(&mut self, completed: u64) -> Option<NativeAllocation> {
        self.texture_staging
            .pop_largest_completed(ez_gfx_hal::QueueKind::TextureTransfer, completed)
    }
}

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

/// Exposes the Vulkan texture pipeline behind the backend-neutral manager
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
        // The Vulkan gate is an infallible in-flight mask read.
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
