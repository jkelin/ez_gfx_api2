//! Standalone Basis Universal decoding and compression admission.

use super::{CompressionSupport, DecodedTexture, TextureDestination, TextureError, TextureFormat};
#[cfg(feature = "basis")]
use super::{DecodedMip, decoded};

pub(super) fn compression_supported(support: CompressionSupport, format: TextureFormat) -> bool {
    match format {
        TextureFormat::Bc1Unorm
        | TextureFormat::Bc1Srgb
        | TextureFormat::Bc3Unorm
        | TextureFormat::Bc3Srgb
        | TextureFormat::Bc7Unorm
        | TextureFormat::Bc7Srgb => support.intersects(CompressionSupport::BC),
        TextureFormat::Astc4x4Unorm | TextureFormat::Astc4x4Srgb => {
            support.intersects(CompressionSupport::ASTC)
        }
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb => true,
    }
}

#[cfg(feature = "basis")]
pub(super) fn select_format(
    destination: TextureDestination,
    compression: CompressionSupport,
    etc1s: bool,
    alpha: bool,
    srgb: bool,
) -> Result<TextureFormat, TextureError> {
    // Explicit requests override source color metadata but never adapter capability admission.
    if let Some(format) = destination.format() {
        return compression_supported(compression, format)
            .then_some(format)
            .ok_or(TextureError::Unsupported);
    }
    let format = if compression.intersects(CompressionSupport::BC) {
        match (etc1s, alpha, srgb) {
            (true, false, false) => TextureFormat::Bc1Unorm,
            (true, false, true) => TextureFormat::Bc1Srgb,
            (true, true, false) => TextureFormat::Bc3Unorm,
            (true, true, true) => TextureFormat::Bc3Srgb,
            (false, _, false) => TextureFormat::Bc7Unorm,
            (false, _, true) => TextureFormat::Bc7Srgb,
        }
    } else if compression.intersects(CompressionSupport::ASTC) {
        if srgb {
            TextureFormat::Astc4x4Srgb
        } else {
            TextureFormat::Astc4x4Unorm
        }
    } else if srgb {
        TextureFormat::Rgba8Srgb
    } else {
        TextureFormat::Rgba8Unorm
    };
    Ok(format)
}

#[cfg(feature = "basis")]
pub(super) fn decode(
    data: &[u8],
    compression: CompressionSupport,
    destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    use ez_gfx_basis::BasisTarget;

    let metadata = ez_gfx_basis::metadata(data).map_err(|_| TextureError::InvalidData)?;
    let format = select_format(
        destination,
        compression,
        metadata.etc1s,
        metadata.alpha,
        metadata.srgb,
    )?;
    let target = match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb => BasisTarget::Rgba8,
        TextureFormat::Bc1Unorm | TextureFormat::Bc1Srgb => BasisTarget::Bc1,
        TextureFormat::Bc3Unorm | TextureFormat::Bc3Srgb => BasisTarget::Bc3,
        TextureFormat::Bc7Unorm | TextureFormat::Bc7Srgb => BasisTarget::Bc7,
        TextureFormat::Astc4x4Unorm | TextureFormat::Astc4x4Srgb => BasisTarget::Astc4x4,
    };

    // The private wrapper shares the native Basis library already used by KTX2 transcoding.
    let mips = ez_gfx_basis::transcode(data, target)
        .map_err(|_| TextureError::InvalidData)?
        .into_iter()
        .map(|mip| DecodedMip {
            width: mip.width,
            height: mip.height,
            bytes: mip.bytes,
        })
        .collect();
    decoded(format, mips)
}

#[cfg(not(feature = "basis"))]
pub(super) fn decode(
    _data: &[u8],
    _compression: CompressionSupport,
    _destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    // The source variant remains available so disabled builds fail predictably at decode time.
    Err(TextureError::Unsupported)
}

#[cfg(all(test, feature = "basis"))]
mod tests {
    use super::*;

    #[test]
    fn auto_preserves_color_and_selects_source_appropriate_blocks() {
        for (etc1s, alpha, srgb, support, expected) in [
            (
                true,
                false,
                true,
                CompressionSupport::BC,
                TextureFormat::Bc1Srgb,
            ),
            (
                true,
                true,
                false,
                CompressionSupport::BC,
                TextureFormat::Bc3Unorm,
            ),
            (
                false,
                true,
                true,
                CompressionSupport::BC,
                TextureFormat::Bc7Srgb,
            ),
            (
                false,
                true,
                false,
                CompressionSupport::ASTC,
                TextureFormat::Astc4x4Unorm,
            ),
            (
                true,
                false,
                true,
                CompressionSupport::NONE,
                TextureFormat::Rgba8Srgb,
            ),
        ] {
            assert_eq!(
                select_format(TextureDestination::Auto, support, etc1s, alpha, srgb),
                Ok(expected)
            );
        }
    }

    #[test]
    fn explicit_destination_overrides_metadata_but_not_capabilities() {
        assert_eq!(
            select_format(
                TextureDestination::Bc3Unorm,
                CompressionSupport::BC,
                false,
                true,
                true
            ),
            Ok(TextureFormat::Bc3Unorm)
        );
        assert_eq!(
            select_format(
                TextureDestination::Bc3Unorm,
                CompressionSupport::NONE,
                false,
                true,
                true
            ),
            Err(TextureError::Unsupported)
        );
    }
}
