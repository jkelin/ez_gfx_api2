//! Standalone Basis Universal decoding and compression admission.

use super::{CompressionSupport, DecodedTexture, TextureDestination, TextureError, TextureFormat};
#[cfg(feature = "basis")]
use super::{DecodedMip, MAX_TEXTURE_BYTES, decoded};

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StandaloneMetadata {
    width: u32,
    height: u32,
    mip_count: u32,
    etc1s: bool,
    alpha: bool,
    srgb: bool,
}

#[cfg(feature = "basis")]
fn inspect_standalone(data: &[u8]) -> Result<StandaloneMetadata, TextureError> {
    const HEADER_BYTES: usize = 77;
    const SLICE_BYTES: usize = 23;
    const MAX_MIPS: usize = 16;
    const MAX_SLICES: u32 = 32;

    if data.len() <= HEADER_BYTES
        || data.get(..2) != Some(&[0x73, 0x42])
        || read_u16(data, 2) != Some(0x13)
        || read_u16(data, 4) != Some(77)
    {
        return Err(TextureError::InvalidData);
    }
    let payload_bytes = read_u32(data, 8).ok_or(TextureError::InvalidData)? as usize;
    if payload_bytes > data.len() - HEADER_BYTES {
        return Err(TextureError::InvalidData);
    }
    let slice_count = read_u24(data, 14).ok_or(TextureError::InvalidData)?;
    let image_count = read_u24(data, 17).ok_or(TextureError::InvalidData)?;
    let source = *data.get(20).ok_or(TextureError::InvalidData)?;
    let flags = read_u16(data, 21).ok_or(TextureError::InvalidData)?;
    let texture_type = *data.get(23).ok_or(TextureError::InvalidData)?;
    if !(1..=MAX_SLICES).contains(&slice_count) {
        return Err(TextureError::InvalidData);
    }
    if image_count != 1 || !matches!(source, 0 | 1) || texture_type != 0 {
        return Err(TextureError::Unsupported);
    }

    let descriptor_offset = read_u32(data, 65).ok_or(TextureError::InvalidData)? as usize;
    let descriptor_bytes = usize::try_from(slice_count)
        .ok()
        .and_then(|count| count.checked_mul(SLICE_BYTES))
        .ok_or(TextureError::InvalidData)?;
    let descriptors = data
        .get(
            descriptor_offset
                ..descriptor_offset
                    .checked_add(descriptor_bytes)
                    .ok_or(TextureError::InvalidData)?,
        )
        .ok_or(TextureError::InvalidData)?;
    let mut alpha = flags & 4 != 0;
    let expected_slices_per_level = if source == 0 && alpha { 2 } else { 1 };
    let mut widths = [0_u32; MAX_MIPS];
    let mut heights = [0_u32; MAX_MIPS];
    let mut slices_per_level = [0_u8; MAX_MIPS];
    let mut max_level = 0_usize;

    for descriptor in descriptors.chunks_exact(SLICE_BYTES) {
        let image = read_u24(descriptor, 0).ok_or(TextureError::InvalidData)?;
        let level = usize::from(descriptor[3]);
        let width = u32::from(read_u16(descriptor, 5).ok_or(TextureError::InvalidData)?);
        let height = u32::from(read_u16(descriptor, 7).ok_or(TextureError::InvalidData)?);
        let blocks_x = u32::from(read_u16(descriptor, 9).ok_or(TextureError::InvalidData)?);
        let blocks_y = u32::from(read_u16(descriptor, 11).ok_or(TextureError::InvalidData)?);
        let payload_offset = read_u32(descriptor, 13).ok_or(TextureError::InvalidData)? as usize;
        let payload_size = read_u32(descriptor, 17).ok_or(TextureError::InvalidData)? as usize;
        if image != 0
            || level >= MAX_MIPS
            || width == 0
            || height == 0
            || blocks_x != width.div_ceil(4)
            || blocks_y != height.div_ceil(4)
            || (source == 1
                && payload_size
                    != usize::try_from(
                        blocks_x
                            .checked_mul(blocks_y)
                            .and_then(|blocks| blocks.checked_mul(16))
                            .ok_or(TextureError::InvalidData)?,
                    )
                    .map_err(|_| TextureError::InvalidData)?)
            || data
                .get(
                    payload_offset
                        ..payload_offset
                            .checked_add(payload_size)
                            .ok_or(TextureError::InvalidData)?,
                )
                .is_none()
        {
            return Err(TextureError::InvalidData);
        }
        if source == 1 && descriptor[4] & 1 != 0 {
            alpha = true;
        }
        if widths[level] != 0 && (widths[level], heights[level]) != (width, height) {
            return Err(TextureError::InvalidData);
        }
        widths[level] = width;
        heights[level] = height;
        slices_per_level[level] = slices_per_level[level]
            .checked_add(1)
            .ok_or(TextureError::InvalidData)?;
        max_level = max_level.max(level);
    }

    let mip_count = max_level + 1;
    let width = widths[0];
    let height = heights[0];
    if width == 0
        || height == 0
        || (0..mip_count).any(|level| {
            widths[level] != (width >> level).max(1)
                || heights[level] != (height >> level).max(1)
                || slices_per_level[level] != expected_slices_per_level
        })
    {
        return Err(TextureError::InvalidData);
    }

    Ok(StandaloneMetadata {
        width,
        height,
        mip_count: u32::try_from(mip_count).map_err(|_| TextureError::TooLarge)?,
        etc1s: source == 0,
        alpha,
        srgb: flags & 16 != 0,
    })
}

#[cfg(feature = "basis")]
fn read_u16(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        data.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

#[cfg(feature = "basis")]
fn read_u24(data: &[u8], offset: usize) -> Option<u32> {
    let bytes = data.get(offset..offset + 3)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]))
}

#[cfg(feature = "basis")]
fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

#[cfg(feature = "basis")]
pub(super) fn transcode(
    data: &[u8],
    format: TextureFormat,
    width: u32,
    height: u32,
    mip_count: u32,
) -> Result<Vec<DecodedMip>, TextureError> {
    use basisu::{DecodeFlags, Error, TargetFormat, Transcoder};

    let target = match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb => TargetFormat::Rgba32,
        TextureFormat::Bc1Unorm | TextureFormat::Bc1Srgb => TargetFormat::Bc1Rgb,
        TextureFormat::Bc3Unorm | TextureFormat::Bc3Srgb => TargetFormat::Bc3Rgba,
        TextureFormat::Bc7Unorm | TextureFormat::Bc7Srgb => TargetFormat::Bc7Rgba,
        TextureFormat::Astc4x4Unorm | TextureFormat::Astc4x4Srgb => TargetFormat::Astc4x4Rgba,
    };
    let mut total = 0_u64;
    for level in 0..mip_count {
        let mip_width = width.checked_shr(level).unwrap_or(0).max(1);
        let mip_height = height.checked_shr(level).unwrap_or(0).max(1);
        let expected = format
            .level_bytes(mip_width, mip_height)
            .ok_or(TextureError::TooLarge)?;
        total = total.checked_add(expected).ok_or(TextureError::TooLarge)?;
        if total > MAX_TEXTURE_BYTES as u64 {
            return Err(TextureError::TooLarge);
        }
    }

    let transcoder = Transcoder::new(data).map_err(|_| TextureError::InvalidData)?;
    if transcoder.base_dimensions() != (width, height)
        || transcoder.level_count() != mip_count
        || transcoder.layer_count() > 1
        || transcoder.face_count() != 1
        || transcoder.is_video()
    {
        return Err(TextureError::InvalidData);
    }
    if !transcoder.supports(target) {
        return Err(TextureError::Unsupported);
    }

    for level in 0..mip_count {
        let mip_width = width.checked_shr(level).unwrap_or(0).max(1);
        let mip_height = height.checked_shr(level).unwrap_or(0).max(1);
        let info = transcoder
            .image_level_info(level)
            .map_err(|_| TextureError::InvalidData)?;
        let expected = format
            .level_bytes(mip_width, mip_height)
            .ok_or(TextureError::TooLarge)?;
        if (info.width, info.height) != (mip_width, mip_height)
            || transcoder.output_size(level, target).ok() != usize::try_from(expected).ok()
        {
            return Err(TextureError::InvalidData);
        }
    }

    let mut mips = Vec::with_capacity(mip_count as usize);
    for level in 0..mip_count {
        let mip_width = width.checked_shr(level).unwrap_or(0).max(1);
        let mip_height = height.checked_shr(level).unwrap_or(0).max(1);
        let byte_count = usize::try_from(
            format
                .level_bytes(mip_width, mip_height)
                .ok_or(TextureError::TooLarge)?,
        )
        .map_err(|_| TextureError::TooLarge)?;
        let mut bytes = vec![0; byte_count];
        transcoder
            .transcode_into(level, target, DecodeFlags::NONE, &mut bytes)
            .map_err(|error| match error {
                Error::Unsupported { .. } | Error::ZstdRequired => TextureError::Unsupported,
                _ => TextureError::InvalidData,
            })?;
        mips.push(DecodedMip {
            width: mip_width,
            height: mip_height,
            bytes,
        });
    }
    Ok(mips)
}

#[cfg(feature = "basis")]
pub(super) fn decode(
    data: &[u8],
    compression: CompressionSupport,
    destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    let metadata = inspect_standalone(data)?;
    let format = select_format(
        destination,
        compression,
        metadata.etc1s,
        metadata.alpha,
        metadata.srgb,
    )?;
    let mips = transcode(
        data,
        format,
        metadata.width,
        metadata.height,
        metadata.mip_count,
    )?;
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
    fn standalone_preflight_bounds_dependency_metadata_work() {
        let source = include_bytes!("../../tests/fixtures/rust-logo-etc.basis");
        let metadata = inspect_standalone(source).unwrap();
        assert_eq!(
            (
                metadata.width,
                metadata.height,
                metadata.mip_count,
                metadata.etc1s,
                metadata.alpha,
                metadata.srgb,
            ),
            (64, 64, 7, true, true, false)
        );

        let mut impossible = source.to_vec();
        impossible[14..17].fill(0xff);
        assert_eq!(
            inspect_standalone(&impossible),
            Err(TextureError::InvalidData)
        );

        let mut oversized_uastc =
            include_bytes!("../../tests/fixtures/cube-uastc-srgb.basis").to_vec();
        let descriptor = read_u32(&oversized_uastc, 65).unwrap() as usize;
        oversized_uastc[descriptor + 5..descriptor + 9].fill(0xff);
        oversized_uastc[descriptor + 9..descriptor + 13]
            .copy_from_slice(&16_384_u16.to_le_bytes().repeat(2));
        assert_eq!(
            inspect_standalone(&oversized_uastc),
            Err(TextureError::InvalidData)
        );

        let mut uastc_alpha = include_bytes!("../../tests/fixtures/cube-uastc-srgb.basis").to_vec();
        let flags = read_u16(&uastc_alpha, 21).unwrap() & !4;
        uastc_alpha[21..23].copy_from_slice(&flags.to_le_bytes());
        assert!(inspect_standalone(&uastc_alpha).unwrap().alpha);
    }

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
