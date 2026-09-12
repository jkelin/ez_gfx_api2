//! Direct ingestion of tightly packed pre-compressed or RGBA mip chains.
//!
//! The caller names the storage format, base dimensions, and an explicit
//! nonzero level count; levels run from the largest to the smallest with no
//! gaps, padding, or container headers. Only two-dimensional images are
//! accepted, which the contiguous level geometry already implies. The whole
//! chain is prevalidated on a fixed stack layout, including total length
//! equality, before any copy.

use super::{
    CompressionSupport, DecodedMip, DecodedTexture, TextureDestination, TextureError,
    TextureFormat, decoded,
};

const MAX_LEVELS: usize = 32;

pub(super) fn decode(
    data: &[u8],
    format: TextureFormat,
    width: u32,
    height: u32,
    mip_count: u32,
    compression: CompressionSupport,
    destination: TextureDestination,
) -> Result<DecodedTexture, TextureError> {
    if mip_count == 0 {
        return Err(TextureError::InvalidData);
    }
    if width == 0 || height == 0 {
        return Err(TextureError::TooLarge);
    }
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
    if format.is_compressed() && !super::basis::compression_supported(compression, format) {
        return Err(TextureError::Unsupported);
    }
    if destination
        .format()
        .is_some_and(|requested| requested != format)
    {
        return Err(TextureError::Unsupported);
    }
    if total != data.len() as u64 {
        return Err(TextureError::InvalidData);
    }
    let mut mips = Vec::with_capacity(count);
    let mut offset = 0_usize;
    for (level_width, level_height, length) in &levels[..count] {
        let end = offset
            .checked_add(usize::try_from(*length).map_err(|_| TextureError::TooLarge)?)
            .ok_or(TextureError::TooLarge)?;
        let bytes = data.get(offset..end).ok_or(TextureError::InvalidData)?;
        mips.push(DecodedMip {
            width: *level_width,
            height: *level_height,
            bytes: bytes.to_vec(),
        });
        offset = end;
    }
    decoded(format, mips)
}
