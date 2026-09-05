//! Direct DDS container ingestion for pre-compressed block textures.
//!
//! Only two-dimensional, single-array, non-cubemap images in directly stored
//! native formats are accepted. The whole mip chain is prevalidated on a fixed
//! stack layout before any payload copy.

use super::{
    CompressionSupport, DecodedMip, DecodedTexture, TextureDestination, TextureError,
    TextureFormat, decoded,
};

const MAGIC: &[u8; 4] = b"DDS ";
const HEADER_LEN: usize = 124;
const DX10_LEN: usize = 20;
const MAX_LEVELS: usize = 32;

// Header description flags.
const DDSD_REQUIRED: u32 = 0x1 | 0x2 | 0x4 | 0x1000;
const DDSD_PITCH: u32 = 0x8;
const DDSD_MIPMAPCOUNT: u32 = 0x2_0000;
const DDSD_LINEARSIZE: u32 = 0x8_0000;
const DDSD_DEPTH: u32 = 0x80_0000;
const CAPS_TEXTURE: u32 = 0x1000;
const CAPS2_CUBEMAP: u32 = 0x200;
const CAPS2_CUBEMAP_FACES: u32 = 0x400 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000;
const CAPS2_VOLUME: u32 = 0x20_0000;

// Pixel-format flags.
const DDPF_FOURCC: u32 = 0x4;
const FOURCC_DXT1: u32 = u32::from_le_bytes(*b"DXT1");
const FOURCC_DXT5: u32 = u32::from_le_bytes(*b"DXT5");
const FOURCC_DX10: u32 = u32::from_le_bytes(*b"DX10");

// DXGI formats accepted through the extended header.
const DXGI_R8G8B8A8_UNORM: u32 = 28;
const DXGI_R8G8B8A8_UNORM_SRGB: u32 = 29;
const DXGI_BC1_UNORM: u32 = 71;
const DXGI_BC1_UNORM_SRGB: u32 = 72;
const DXGI_BC3_UNORM: u32 = 77;
const DXGI_BC3_UNORM_SRGB: u32 = 78;
const DXGI_BC7_UNORM: u32 = 98;
const DXGI_BC7_UNORM_SRGB: u32 = 99;
const D3D10_RESOURCE_DIMENSION_TEXTURE2D: u32 = 3;
const D3D11_RESOURCE_MISC_TEXTURECUBE: u32 = 0x4;
const DDS_ALPHA_MODE_MASK: u32 = 0x7;
const DDS_ALPHA_MODE_STRAIGHT: u32 = 1;
const DDS_ALPHA_MODE_PREMULTIPLIED: u32 = 2;
const DDS_ALPHA_MODE_OPAQUE: u32 = 3;
const DDS_ALPHA_MODE_CUSTOM: u32 = 4;

/// One validated mip level: width, height, and tightly packed byte length.
type Level = (u32, u32, u64);

/// Prevalidated chain: format, payload, stack level layout, level count, total.
struct Preflight<'a> {
    format: TextureFormat,
    payload: &'a [u8],
    levels: [Level; MAX_LEVELS],
    count: usize,
    total: u64,
}

pub(super) fn decode(
    data: &[u8],
    compression: CompressionSupport,
    destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    let preflight = parse(data)?;
    if preflight.format.is_compressed()
        && !super::basis::compression_supported(compression, preflight.format)
    {
        return Err(TextureError::Unsupported);
    }
    if destination
        .format()
        .is_some_and(|requested| requested != preflight.format)
    {
        return Err(TextureError::Unsupported);
    }
    if preflight.total != preflight.payload.len() as u64 {
        return Err(TextureError::InvalidData);
    }
    let mut mips = Vec::with_capacity(preflight.count);
    let mut offset = 0_usize;
    for (width, height, length) in &preflight.levels[..preflight.count] {
        let end = offset
            .checked_add(usize::try_from(*length).map_err(|_| TextureError::TooLarge)?)
            .ok_or(TextureError::TooLarge)?;
        let bytes = preflight
            .payload
            .get(offset..end)
            .ok_or(TextureError::InvalidData)?;
        mips.push(DecodedMip {
            width: *width,
            height: *height,
            bytes: bytes.to_vec(),
        });
        offset = end;
    }
    decoded(preflight.format, mips)
}

/// Parses and fully prevalidates the container before any payload copy.
///
/// # Errors
///
/// Returns [`TextureError::InvalidData`] for malformed, inconsistent, or
/// truncated headers, [`TextureError::TooLarge`] for zero dimensions or
/// over-budget chains, and [`TextureError::Unsupported`] for arrays, cubemaps,
/// volumes, 1D/3D images, unpacked layouts, unmapped pixel formats, and
/// premultiplied or custom alpha modes.
fn parse(data: &[u8]) -> Result<Preflight<'_>, TextureError> {
    if data.len() < MAGIC.len() + HEADER_LEN || data[..MAGIC.len()] != *MAGIC {
        return Err(TextureError::InvalidData);
    }
    let header = &data[MAGIC.len()..MAGIC.len() + HEADER_LEN];
    let flags = u32le(header, 4);
    if u32le(header, 0) != u32::try_from(HEADER_LEN).expect("DDS header length fits u32")
        || flags & DDSD_REQUIRED != DDSD_REQUIRED
        || u32le(header, 104) & CAPS_TEXTURE == 0
    {
        return Err(TextureError::InvalidData);
    }
    if u32le(header, 108) & (CAPS2_CUBEMAP | CAPS2_CUBEMAP_FACES | CAPS2_VOLUME) != 0
        || flags & DDSD_DEPTH != 0
        || u32le(header, 20) > 1
    {
        return Err(TextureError::Unsupported);
    }
    let height = u32le(header, 8);
    let width = u32le(header, 12);
    if width == 0 || height == 0 {
        return Err(TextureError::TooLarge);
    }
    // A set flag with a zero count is malformed; a count above one without the
    // flag is malformed. A set flag with a count of one is valid.
    let count = u32le(header, 24);
    if (flags & DDSD_MIPMAPCOUNT != 0 && count == 0) || (count > 1 && flags & DDSD_MIPMAPCOUNT == 0)
    {
        return Err(TextureError::InvalidData);
    }
    let mip_count = count.max(1);
    if u32le(header, 72) != 32 {
        return Err(TextureError::InvalidData);
    }
    let (format, payload_offset) = map_pixel_format(data, header)?;
    // A pitch must name the tightly packed native row stride; a linear size
    // must name exactly the base level byte count.
    if flags & DDSD_PITCH != 0 && u64::from(u32le(header, 16)) != native_pitch(format, width)? {
        return Err(TextureError::InvalidData);
    }
    if flags & DDSD_LINEARSIZE != 0
        && u64::from(u32le(header, 16))
            != format
                .level_bytes(width, height)
                .ok_or(TextureError::TooLarge)?
    {
        return Err(TextureError::InvalidData);
    }
    let (levels, count, total) = chain_levels(format, width, height, mip_count)?;
    let payload = data
        .get(payload_offset..)
        .ok_or(TextureError::InvalidData)?;
    Ok(Preflight {
        format,
        payload,
        levels,
        count,
        total,
    })
}

/// Maps the pixel-format substructure, including the extended DX10 header when
/// present, to a storage format and payload offset.
///
/// # Errors
///
/// Returns [`TextureError::InvalidData`] for truncated headers, a `FourCC` field
/// without its flag, or reserved alpha bits, and [`TextureError::Unsupported`]
/// for non-2D, array, cubemap, unpacked, unmapped, premultiplied-alpha, or
/// custom-alpha formats.
fn map_pixel_format(data: &[u8], header: &[u8]) -> Result<(TextureFormat, usize), TextureError> {
    let four_cc = u32le(header, 80);
    if four_cc == FOURCC_DXT1 {
        require_fourcc_flag(header)?;
        return Ok((TextureFormat::Bc1Unorm, MAGIC.len() + HEADER_LEN));
    }
    if four_cc == FOURCC_DXT5 {
        // DXT3 names explicit-alpha BC2 blocks, which have no native storage
        // enum and fall through to the unsupported FourCC rejection below.
        require_fourcc_flag(header)?;
        return Ok((TextureFormat::Bc3Unorm, MAGIC.len() + HEADER_LEN));
    }
    if four_cc == FOURCC_DX10 {
        require_fourcc_flag(header)?;
        let start = MAGIC.len() + HEADER_LEN;
        let extended = data
            .get(start..start + DX10_LEN)
            .ok_or(TextureError::InvalidData)?;
        check_alpha_mode(u32le(extended, 16))?;
        let format = map_dxgi_format(
            u32le(extended, 0),
            u32le(extended, 4),
            u32le(extended, 8),
            u32le(extended, 12),
        )?;
        return Ok((format, start + DX10_LEN));
    }
    Err(TextureError::Unsupported)
}

/// Requires the `FourCC` present flag before any `FourCC` value is trusted.
///
/// # Errors
///
/// Returns [`TextureError::InvalidData`] when the flag is absent.
fn require_fourcc_flag(header: &[u8]) -> Result<(), TextureError> {
    if u32le(header, 76) & DDPF_FOURCC == 0 {
        return Err(TextureError::InvalidData);
    }
    Ok(())
}

/// Enforces the DX10 alpha-mode policy: unknown, straight, and opaque payloads
/// are retained; premultiplied and custom payloads are rejected rather than
/// normalized; reserved bits are malformed.
///
/// # Errors
///
/// Returns [`TextureError::InvalidData`] for reserved bits or undefined modes
/// and [`TextureError::Unsupported`] for premultiplied or custom modes.
fn check_alpha_mode(misc: u32) -> Result<(), TextureError> {
    if misc & !DDS_ALPHA_MODE_MASK != 0 {
        return Err(TextureError::InvalidData);
    }
    match misc & DDS_ALPHA_MODE_MASK {
        DDS_ALPHA_MODE_PREMULTIPLIED | DDS_ALPHA_MODE_CUSTOM => Err(TextureError::Unsupported),
        DDS_ALPHA_MODE_STRAIGHT | DDS_ALPHA_MODE_OPAQUE | 0 => Ok(()),
        _ => Err(TextureError::InvalidData),
    }
}

/// Maps one DXGI format value plus dimension and misc flags to storage.
///
/// # Errors
///
/// Returns [`TextureError::Unsupported`] for non-2D, array, cubemap, or
/// unmapped DXGI formats.
fn map_dxgi_format(
    dxgi: u32,
    dimension: u32,
    misc: u32,
    array_size: u32,
) -> Result<TextureFormat, TextureError> {
    if dimension != D3D10_RESOURCE_DIMENSION_TEXTURE2D
        || misc & D3D11_RESOURCE_MISC_TEXTURECUBE != 0
        || array_size != 1
    {
        return Err(TextureError::Unsupported);
    }
    match dxgi {
        DXGI_R8G8B8A8_UNORM => Ok(TextureFormat::Rgba8Unorm),
        DXGI_R8G8B8A8_UNORM_SRGB => Ok(TextureFormat::Rgba8Srgb),
        DXGI_BC1_UNORM => Ok(TextureFormat::Bc1Unorm),
        DXGI_BC1_UNORM_SRGB => Ok(TextureFormat::Bc1Srgb),
        DXGI_BC3_UNORM => Ok(TextureFormat::Bc3Unorm),
        DXGI_BC3_UNORM_SRGB => Ok(TextureFormat::Bc3Srgb),
        DXGI_BC7_UNORM => Ok(TextureFormat::Bc7Unorm),
        DXGI_BC7_UNORM_SRGB => Ok(TextureFormat::Bc7Srgb),
        _ => Err(TextureError::Unsupported),
    }
}

/// Computes the tightly packed native row stride for one level row.
///
/// # Errors
///
/// Returns [`TextureError::TooLarge`] when the stride is unrepresentable.
fn native_pitch(format: TextureFormat, width: u32) -> Result<u64, TextureError> {
    match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb => u64::from(width)
            .checked_mul(4)
            .ok_or(TextureError::TooLarge),
        TextureFormat::Bc1Unorm | TextureFormat::Bc1Srgb => u64::from(width.div_ceil(4))
            .checked_mul(8)
            .ok_or(TextureError::TooLarge),
        TextureFormat::Bc3Unorm
        | TextureFormat::Bc3Srgb
        | TextureFormat::Bc7Unorm
        | TextureFormat::Bc7Srgb
        | TextureFormat::Astc4x4Unorm
        | TextureFormat::Astc4x4Srgb => u64::from(width.div_ceil(4))
            .checked_mul(16)
            .ok_or(TextureError::TooLarge),
    }
}

/// Prevalidates the full level geometry, count, and aggregate byte size on a
/// fixed stack layout.
///
/// # Errors
///
/// Returns [`TextureError::InvalidData`] for counts beyond the physical chain
/// and [`TextureError::TooLarge`] for unrepresentable or over-budget sizes.
fn chain_levels(
    format: TextureFormat,
    width: u32,
    height: u32,
    mip_count: u32,
) -> Result<([Level; MAX_LEVELS], usize, u64), TextureError> {
    let physical = u32::BITS - width.max(height).leading_zeros();
    if mip_count > physical {
        return Err(TextureError::InvalidData);
    }
    let count = usize::try_from(mip_count).map_err(|_| TextureError::TooLarge)?;
    if count > MAX_LEVELS {
        return Err(TextureError::InvalidData);
    }
    let mut levels = [(0_u32, 0_u32, 0_u64); MAX_LEVELS];
    let mut total = 0_u64;
    let mut level_width = width;
    let mut level_height = height;
    for slot in levels.iter_mut().take(count) {
        let length = format
            .level_bytes(level_width, level_height)
            .ok_or(TextureError::TooLarge)?;
        total = total
            .checked_add(length)
            .filter(|total| *total <= super::MAX_TEXTURE_BYTES as u64)
            .ok_or(TextureError::TooLarge)?;
        *slot = (level_width, level_height, length);
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }
    Ok((levels, count, total))
}

fn u32le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
