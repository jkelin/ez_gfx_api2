//! Manager-trait links for the selected backend implementations.

#[cfg(windows)]
use super::Dx12Context;
#[cfg(target_vendor = "apple")]
use super::MetalContext;
use super::{
    CompletionToken, MipTransferValues, NativeContext, NativeTexture, TextureBackendContext,
    TextureBackendTexture, TextureFormat, TextureRegion, VulkanContext,
};

/// Links the selected backend implementations behind the manager traits: one
/// exhaustive match per operation, statically dispatched with no `dyn`.
impl MipTransferValues for NativeTexture {
    fn mip_transfer_values(&self) -> &[u64] {
        match self {
            NativeTexture::Vulkan(texture) => texture.mip_transfer_values(),
            #[cfg(windows)]
            NativeTexture::Dx12(texture) => texture.mip_transfer_values(),
            #[cfg(target_vendor = "apple")]
            NativeTexture::Metal(texture) => texture.mip_transfer_values(),
        }
    }
}

impl TextureBackendTexture for NativeTexture {
    fn last_transfer_value(&self) -> u64 {
        match self {
            NativeTexture::Vulkan(texture) => texture.last_transfer_value(),
            #[cfg(windows)]
            NativeTexture::Dx12(texture) => texture.last_transfer_value(),
            #[cfg(target_vendor = "apple")]
            NativeTexture::Metal(texture) => texture.last_transfer_value(),
        }
    }
}

impl TextureBackendContext for NativeContext {
    type Texture = NativeTexture;

    fn create_texture_with_prefix(
        &mut self,
        format: TextureFormat,
        mips: &[ez_gfx_hal::ImageMip<'_>],
        binding: u32,
        sampler: ez_gfx_hal::TextureSamplerDesc,
        prefix: u32,
    ) -> Result<(Self::Texture, Vec<CompletionToken>), ez_gfx_hal::AllocationError> {
        match self {
            NativeContext::Vulkan(context) => context
                .create_texture_with_prefix(format, mips, binding, sampler, prefix)
                .map(|(texture, tokens)| (NativeTexture::Vulkan(texture), tokens)),
            #[cfg(windows)]
            NativeContext::Dx12(context) => context
                .create_texture_with_prefix(format, mips, binding, sampler, prefix)
                .map(|(texture, tokens)| (NativeTexture::Dx12(texture), tokens)),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(context) => context
                .create_texture_with_prefix(format, mips, binding, sampler, prefix)
                .map(|(texture, tokens)| (NativeTexture::Metal(texture), tokens)),
        }
    }

    fn update_texture_region(
        &mut self,
        texture: &mut Self::Texture,
        region: &TextureRegion<'_>,
    ) -> Result<CompletionToken, ez_gfx_hal::AllocationError> {
        // Native methods synchronously copy the borrowed region before returning its token.
        match (self, texture) {
            (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
                context.update_texture_region(texture, region)
            }
            #[cfg(windows)]
            (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
                context.update_texture_region(texture, region)
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
                context.update_texture_region(texture, region)
            }
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
        }
    }

    fn reference_texture_prefix(
        &mut self,
        texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), ez_gfx_hal::AllocationError> {
        // Native methods install the transfer-independent view synchronously; the
        // caller attaches the required GPU wait before any sampling draw executes.
        match (self, texture) {
            (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
                context.reference_texture_prefix(texture, resident_mips)
            }
            #[cfg(windows)]
            (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
                context.reference_texture_prefix(texture, resident_mips)
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
                context.reference_texture_prefix(texture, resident_mips)
            }
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
        }
    }

    fn publish_texture_mips(
        &mut self,
        texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), ez_gfx_hal::AllocationError> {
        // Variant mismatches indicate corrupt context-owned state, never caller input.
        match (self, texture) {
            (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
                context.publish_texture_mips(texture, resident_mips)
            }
            #[cfg(windows)]
            (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
                context.publish_texture_mips(texture, resident_mips)
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
                context.publish_texture_mips(texture, resident_mips)
            }
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
        }
    }

    fn texture_descriptors_ready(&self) -> Result<bool, ez_gfx_hal::AllocationError> {
        match self {
            NativeContext::Vulkan(context) => Ok(context.texture_descriptor_update_ready()),
            #[cfg(windows)]
            NativeContext::Dx12(context) => context.texture_descriptor_update_ready(),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(context) => Ok(context.texture_descriptor_update_ready()),
        }
    }

    fn texture_retirement_ready(
        &self,
        completion: CompletionToken,
    ) -> Result<bool, ez_gfx_hal::AllocationError> {
        match self {
            NativeContext::Vulkan(context) => context.texture_retirement_ready(completion),
            #[cfg(windows)]
            NativeContext::Dx12(context) => context.texture_retirement_ready(completion),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(context) => context.texture_retirement_ready(completion),
        }
    }

    fn destroy_texture(
        &mut self,
        texture: Self::Texture,
    ) -> Result<(), ez_gfx_hal::AllocationError> {
        match (self, texture) {
            (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
                context.destroy_texture(texture)
            }
            #[cfg(windows)]
            (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
                context.destroy_texture(texture)
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
                context.destroy_texture(texture)
            }
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
        }
    }

    fn completed_texture_transfer_value(&self) -> Result<u64, ez_gfx_hal::AllocationError> {
        match self {
            NativeContext::Vulkan(context) => context.completed_texture_transfer_value(),
            #[cfg(windows)]
            NativeContext::Dx12(context) => context.completed_texture_transfer_value(),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(context) => context.completed_texture_transfer_value(),
        }
    }

    fn cancel_texture_transfers(texture: &Self::Texture) {
        match texture {
            NativeTexture::Vulkan(texture) => VulkanContext::cancel_texture_transfers(texture),
            #[cfg(windows)]
            NativeTexture::Dx12(texture) => Dx12Context::cancel_texture_transfers(texture),
            #[cfg(target_vendor = "apple")]
            NativeTexture::Metal(texture) => MetalContext::cancel_texture_transfers(texture),
        }
    }
}
