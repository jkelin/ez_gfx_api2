//! KTX2 decoding and Basis Universal transcoding.

use super::{CompressionSupport, DecodedTexture, TextureDestination, TextureError};
#[cfg(feature = "ktx2")]
use super::{DecodedMip, MAX_TEXTURE_BYTES, TextureFormat, basis, decoded};

/// Decodes supported two-dimensional KTX2 payloads.
///
/// # Errors
///
/// Returns an error if KTX2 data is malformed or unsupported by the requested capability set.
#[cfg(feature = "ktx2")]
pub(super) fn decode_ktx2(
    data: &[u8],
    compression: CompressionSupport,
    destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    let reader = ktx2::Reader::new(data).map_err(|_| TextureError::InvalidData)?;
    let header = reader.header();
    if header.pixel_height == 0
        || header.pixel_depth != 0
        || header.layer_count > 1
        || header.face_count != 1
    {
        return Err(TextureError::Unsupported);
    }
    let mip_count = header.level_count.max(1);
    if header.pixel_width == 0
        || mip_count > u32::BITS - header.pixel_width.max(header.pixel_height).leading_zeros()
    {
        return Err(TextureError::InvalidData);
    }
    let direct_format = header.format.and_then(map_ktx2_format);
    if let (Some(format), None) = (direct_format, header.supercompression_scheme) {
        if (format.is_compressed() && !basis::compression_supported(compression, format))
            || destination
                .format()
                .is_some_and(|requested| requested != format)
        {
            return Err(TextureError::Unsupported);
        }
        // Repeated/overlapping source ranges cannot amplify allocations beyond the output budget.
        let mut total = 0_u64;
        for (index, level) in reader.levels().enumerate() {
            let width = (header.pixel_width >> index).max(1);
            let height = (header.pixel_height >> index).max(1);
            let expected = format
                .level_bytes(width, height)
                .ok_or(TextureError::TooLarge)?;
            if level.data.len() as u64 != expected {
                return Err(TextureError::InvalidData);
            }
            total = total.checked_add(expected).ok_or(TextureError::TooLarge)?;
        }
        if total > MAX_TEXTURE_BYTES as u64 {
            return Err(TextureError::TooLarge);
        }
        let mips = reader
            .levels()
            .enumerate()
            .map(|(index, level)| DecodedMip {
                width: (header.pixel_width >> index).max(1),
                height: (header.pixel_height >> index).max(1),
                bytes: level.data.to_vec(),
            })
            .collect();
        return decoded(format, mips);
    }
    match (header.format, header.supercompression_scheme) {
        (None, None) if reader.color_model() == Some(ktx2::ColorModel::UASTC) => decode_ktx2_basis(
            data,
            header.pixel_width,
            header.pixel_height,
            header.level_count.max(1),
            compression,
            destination,
        ),
        (None, Some(ktx2::SupercompressionScheme::BasisLZ))
            if reader.color_model() == Some(ktx2::ColorModel::ETC1S) =>
        {
            decode_ktx2_basis(
                data,
                header.pixel_width,
                header.pixel_height,
                header.level_count.max(1),
                compression,
                destination,
            )
        }
        (None, Some(ktx2::SupercompressionScheme::Zstandard))
            if reader.color_model() == Some(ktx2::ColorModel::UASTC) =>
        {
            decode_ktx2_basis(
                data,
                header.pixel_width,
                header.pixel_height,
                header.level_count.max(1),
                compression,
                destination,
            )
        }
        _ => Err(TextureError::Unsupported),
    }
}

/// The pure-Rust Basis Universal decoder validates UASTC/ETC1S payloads, codebooks, and slices; arrays and cubemaps are rejected by this 2D texture contract.
///
/// # Errors
///
/// Returns an error if Basis transcoding fails, its metadata or output is inconsistent, or the decoded mip data is invalid or oversized.
#[cfg(all(feature = "ktx2", feature = "basis"))]
fn decode_ktx2_basis(
    data: &[u8],
    width: u32,
    height: u32,
    mip_count: u32,
    compression: CompressionSupport,
    destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    // Bound transcoder preparation and output allocations before entering per-level decode.
    if width == 0
        || height == 0
        || mip_count == 0
        || mip_count > u32::BITS - width.max(height).leading_zeros()
    {
        return Err(TextureError::InvalidData);
    }
    let reader = ktx2::Reader::new(data).map_err(|_| TextureError::InvalidData)?;
    let scheme = reader.header().supercompression_scheme;
    if scheme != Some(ktx2::SupercompressionScheme::BasisLZ) {
        let mut source_total = 0_u64;
        for (level, payload) in reader.levels().enumerate() {
            let level = u32::try_from(level).map_err(|_| TextureError::TooLarge)?;
            let mip_width = (width >> level).max(1);
            let mip_height = (height >> level).max(1);
            let expected = u64::from(mip_width.div_ceil(4))
                .checked_mul(u64::from(mip_height.div_ceil(4)))
                .and_then(|blocks| blocks.checked_mul(16))
                .ok_or(TextureError::TooLarge)?;
            let actual = if scheme == Some(ktx2::SupercompressionScheme::Zstandard) {
                payload.uncompressed_byte_length
            } else {
                payload.data.len() as u64
            };
            if actual != expected {
                return Err(TextureError::InvalidData);
            }
            source_total = source_total
                .checked_add(actual)
                .ok_or(TextureError::TooLarge)?;
        }
        if source_total > MAX_TEXTURE_BYTES as u64 {
            return Err(TextureError::TooLarge);
        }
    }
    let dfd = reader.basic_dfd().ok_or(TextureError::InvalidData)?;
    let etc1s =
        reader.header().supercompression_scheme == Some(ktx2::SupercompressionScheme::BasisLZ);
    let first_channel = dfd
        .sample_information
        .first()
        .ok_or(TextureError::InvalidData)?
        .channel_type;
    let second_channel = dfd
        .sample_information
        .get(1)
        .map(|sample| sample.channel_type);
    // ETC1S and UASTC RRRG carry the second data channel in the decoded alpha plane.
    let data_channels = match (etc1s, first_channel, second_channel) {
        (true, 3, Some(4)) | (false, 5, _) => Some(true),
        (true, 3, _) | (false, 4, _) => Some(false),
        _ => None,
    };
    let native_rg = !etc1s && first_channel == 6;
    let source_srgb = reader.transfer_function() == Some(ktx2::TransferFunction::SRGB);
    let destination = if data_channels.is_some() || native_rg {
        match destination {
            TextureDestination::Auto if source_srgb => TextureDestination::Rgba8Srgb,
            TextureDestination::Auto => TextureDestination::Rgba8Unorm,
            TextureDestination::Rgba8Unorm | TextureDestination::Rgba8Srgb => destination,
            _ => return Err(TextureError::Unsupported),
        }
    } else {
        destination
    };
    let alpha = dfd.sample_information.iter().any(|sample| {
        if etc1s {
            sample.channel_type == 15
        } else {
            sample.channel_type == 3
        }
    });
    let format = basis::select_format(destination, compression, etc1s, alpha, source_srgb)?;
    let mut total = 0_u64;
    for level in 0..mip_count {
        let bytes = format
            .level_bytes((width >> level).max(1), (height >> level).max(1))
            .ok_or(TextureError::TooLarge)?;
        total = total.checked_add(bytes).ok_or(TextureError::TooLarge)?;
    }
    if total > MAX_TEXTURE_BYTES as u64 {
        return Err(TextureError::TooLarge);
    }

    let mut mips = basis::transcode(data, format, width, height, mip_count)?;
    if let Some(two_channels) = data_channels {
        for mip in &mut mips {
            for pixel in mip.bytes.chunks_exact_mut(4) {
                pixel[1] = if two_channels { pixel[3] } else { 0 };
                pixel[2] = 0;
                pixel[3] = 255;
            }
        }
    } else if native_rg {
        for mip in &mut mips {
            for pixel in mip.bytes.chunks_exact_mut(4) {
                pixel[2] = 0;
                pixel[3] = 255;
            }
        }
    }
    decoded(format, mips)
}

#[cfg(feature = "ktx2")]
fn map_ktx2_format(format: ktx2::Format) -> Option<TextureFormat> {
    match format {
        ktx2::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        ktx2::Format::R8G8B8A8_SRGB => Some(TextureFormat::Rgba8Srgb),
        ktx2::Format::BC1_RGB_UNORM_BLOCK | ktx2::Format::BC1_RGBA_UNORM_BLOCK => {
            Some(TextureFormat::Bc1Unorm)
        }
        ktx2::Format::BC1_RGB_SRGB_BLOCK | ktx2::Format::BC1_RGBA_SRGB_BLOCK => {
            Some(TextureFormat::Bc1Srgb)
        }
        ktx2::Format::BC3_UNORM_BLOCK => Some(TextureFormat::Bc3Unorm),
        ktx2::Format::BC3_SRGB_BLOCK => Some(TextureFormat::Bc3Srgb),
        ktx2::Format::BC7_UNORM_BLOCK => Some(TextureFormat::Bc7Unorm),
        ktx2::Format::BC7_SRGB_BLOCK => Some(TextureFormat::Bc7Srgb),
        ktx2::Format::ASTC_4x4_UNORM_BLOCK => Some(TextureFormat::Astc4x4Unorm),
        ktx2::Format::ASTC_4x4_SRGB_BLOCK => Some(TextureFormat::Astc4x4Srgb),
        // Other block shapes and numeric models need corresponding backend storage contracts.
        _ => None,
    }
}

#[cfg(not(feature = "ktx2"))]
pub(super) fn decode_ktx2(
    _data: &[u8],
    _compression: CompressionSupport,
    _destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    // Keep source IDs stable when the optional container parser is absent.
    Err(TextureError::Unsupported)
}

#[cfg(all(feature = "ktx2", not(feature = "basis")))]
fn decode_ktx2_basis(
    _data: &[u8],
    _width: u32,
    _height: u32,
    _mip_count: u32,
    _compression: CompressionSupport,
    _destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    // Direct KTX2 loading does not pull in the optional universal transcoder.
    Err(TextureError::Unsupported)
}
