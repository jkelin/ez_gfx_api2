use super::{
    ContextState, Error, NativeContext, NativeTexture, Result, TextureFormat, map_allocation,
    map_hal, publish_native_texture_mips, wait_native_idle,
};

const FALLBACK_PIXEL: [u8; 4] = [255, 0, 255, 255];

/// Ownership and publication state for the context-wide sampled fallback.
pub(super) enum TextureFallback {
    Uninitialized,
    Uploading(NativeTexture),
    Ready(NativeTexture),
    #[cfg(test)]
    PublicationFailure(NativeTexture),
}

impl TextureFallback {
    pub(super) fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    #[cfg(test)]
    pub(super) fn is_owned(&self) -> bool {
        !matches!(self, Self::Uninitialized)
    }

    pub(super) fn permits_load(&self) -> bool {
        matches!(self, Self::Uninitialized | Self::Ready(_))
    }

    #[cfg(target_vendor = "apple")]
    pub(super) fn ready(&self) -> Option<&NativeTexture> {
        let Self::Ready(texture) = self else {
            return None;
        };
        Some(texture)
    }

    fn ready_mut(&mut self) -> Option<&mut NativeTexture> {
        let Self::Ready(texture) = self else {
            return None;
        };
        Some(texture)
    }

    fn unpublished_mut(&mut self) -> Option<&mut NativeTexture> {
        match self {
            Self::Uploading(texture) => Some(texture),
            #[cfg(test)]
            Self::PublicationFailure(texture) => Some(texture),
            Self::Uninitialized | Self::Ready(_) => None,
        }
    }

    pub(super) fn take(&mut self) -> Option<NativeTexture> {
        match core::mem::replace(self, Self::Uninitialized) {
            Self::Uninitialized => None,
            Self::Uploading(texture) | Self::Ready(texture) => Some(texture),
            #[cfg(test)]
            Self::PublicationFailure(texture) => Some(texture),
        }
    }

    #[cfg(test)]
    pub(super) fn inject_publication_failure(&mut self) -> Result<()> {
        let Self::Ready(_) = self else {
            return Err(Error::NativeFailure);
        };
        let Self::Ready(texture) = core::mem::replace(self, Self::Uninitialized) else {
            unreachable!("the ready variant was checked before replacement")
        };
        *self = Self::PublicationFailure(texture);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn resume_publication(&mut self) -> Result<()> {
        let Self::PublicationFailure(_) = self else {
            return Err(Error::NativeFailure);
        };
        let Self::PublicationFailure(texture) = core::mem::replace(self, Self::Uninitialized)
        else {
            unreachable!("the failure variant was checked before replacement")
        };
        *self = Self::Uploading(texture);
        Ok(())
    }
}

fn fallback_sampler() -> ez_gfx_hal::TextureSamplerDesc {
    ez_gfx_hal::TextureSamplerDesc {
        min_filter: ez_gfx_hal::SamplerFilter::Nearest,
        mag_filter: ez_gfx_hal::SamplerFilter::Nearest,
        max_anisotropy: 1.0,
        address_u: ez_gfx_hal::SamplerAddressMode::Clamp,
        address_v: ez_gfx_hal::SamplerAddressMode::Clamp,
        address_w: ez_gfx_hal::SamplerAddressMode::Clamp,
    }
}

fn create_native_fallback(context: &mut NativeContext) -> Result<NativeTexture> {
    let mip = ez_gfx_hal::ImageMip {
        width: 1,
        height: 1,
        bytes: &FALLBACK_PIXEL,
    };
    match context {
        NativeContext::Vulkan(native) => native
            .create_texture(TextureFormat::Rgba8Unorm, &[mip], 0, fallback_sampler())
            .map(|(texture, _)| NativeTexture::Vulkan(texture)),
        #[cfg(windows)]
        NativeContext::Dx12(native) => native
            .create_texture(TextureFormat::Rgba8Unorm, &[mip], 0, fallback_sampler())
            .map(|(texture, _)| NativeTexture::Dx12(texture)),
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(native) => native
            .create_texture(TextureFormat::Rgba8Unorm, &[mip], 0, fallback_sampler())
            .map(|(texture, _)| NativeTexture::Metal(texture)),
    }
    .map_err(map_allocation)
}

fn publish_native_fallback(
    context: &mut NativeContext,
    fallback: &mut NativeTexture,
    binding: u32,
) -> std::result::Result<(), ez_gfx_hal::AllocationError> {
    match (context, fallback) {
        (NativeContext::Vulkan(context), NativeTexture::Vulkan(fallback)) => {
            context.publish_texture_fallback(fallback, binding)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeTexture::Dx12(fallback)) => {
            context.publish_texture_fallback(fallback, binding)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeTexture::Metal(fallback)) => {
            context.publish_texture_fallback(fallback, binding)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

pub(super) fn publish_reserved_fallback(context: &mut ContextState, binding: u32) -> Result<()> {
    #[cfg(test)]
    if let Some(remaining) = context.texture_fallback_alias_test_failure_after.as_mut() {
        if *remaining == 0 {
            return Err(Error::NativeFailure);
        }
        *remaining -= 1;
    }
    let fallback = context
        .texture_fallback
        .ready_mut()
        .ok_or(Error::NotReady)?;
    publish_native_fallback(&mut context.native, fallback, binding).map_err(map_allocation)
}

/// Creates the context fallback and backfills every pre-device reserved slot.
///
/// # Errors
///
/// Returns an error when native allocation, upload, synchronization, publication, or slot
/// backfill fails. Ownership remains in the explicit uploading state so retries cannot leak or
/// alias an incompletely published texture.
pub(super) fn initialize_texture_fallback(context: &mut ContextState) -> Result<()> {
    if matches!(context.texture_fallback, TextureFallback::Uninitialized) {
        let created = create_native_fallback(&mut context.native)?;
        context.texture_fallback = TextureFallback::Uploading(created);
    }

    if !context.texture_fallback.is_ready() {
        // Device initialization is the only global wait: no frame can submit before it. A failed
        // wait leaves the owned texture uploading so a later initialization can retry safely.
        wait_native_idle(&mut context.native).map_err(map_hal)?;
        #[cfg(test)]
        if matches!(
            context.texture_fallback,
            TextureFallback::PublicationFailure(_)
        ) {
            return Err(Error::NativeFailure);
        }
        let fallback = context
            .texture_fallback
            .unpublished_mut()
            .ok_or(Error::NativeFailure)?;
        publish_native_texture_mips(&mut context.native, fallback, 1).map_err(map_allocation)?;
        let Some(texture) = context.texture_fallback.take() else {
            return Err(Error::NativeFailure);
        };
        context.texture_fallback = TextureFallback::Ready(texture);
    }

    let bindings = context
        .pending_textures
        .iter()
        .filter(|(_, pending)| !pending.fallback_published)
        .map(|(handle, pending)| {
            context
                .texture_registry
                .reserved_binding(pending.id)
                .map(|binding| (*handle, binding))
                .map_err(super::map_texture)
        })
        .collect::<Result<Vec<_>>>()?;
    for (handle, binding) in bindings {
        publish_reserved_fallback(context, binding)?;
        context
            .pending_textures
            .get_mut(&handle)
            .ok_or(Error::InvalidContext)?
            .fallback_published = true;
    }
    Ok(())
}
