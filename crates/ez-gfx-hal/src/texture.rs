use crate::ContractError;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Sampled texture storage format shared by runtime and native backends.
pub enum TextureFormat {
    /// Linear normalized RGBA8.
    Rgba8Unorm,
    /// sRGB normalized RGBA8.
    Rgba8Srgb,
    /// Linear BC1 RGB/RGBA blocks.
    Bc1Unorm,
    /// sRGB BC1 RGB/RGBA blocks.
    Bc1Srgb,
    /// Linear BC3 RGBA blocks.
    Bc3Unorm,
    /// sRGB BC3 RGBA blocks.
    Bc3Srgb,
    /// Linear BC7 RGBA blocks.
    Bc7Unorm,
    /// sRGB BC7 RGBA blocks.
    Bc7Srgb,
    /// Linear ASTC 4x4 RGBA blocks.
    Astc4x4Unorm,
    /// sRGB ASTC 4x4 RGBA blocks.
    Astc4x4Srgb,
}

impl TextureFormat {
    /// Returns `[block_width, block_height, bytes_per_block]`.
    pub const fn block(self) -> [u32; 3] {
        match self {
            Self::Rgba8Unorm | Self::Rgba8Srgb => [1, 1, 4],
            Self::Bc1Unorm | Self::Bc1Srgb => [4, 4, 8],
            Self::Bc3Unorm
            | Self::Bc3Srgb
            | Self::Bc7Unorm
            | Self::Bc7Srgb
            | Self::Astc4x4Unorm
            | Self::Astc4x4Srgb => [4, 4, 16],
        }
    }

    /// Returns whether storage uses block compression.
    pub const fn is_compressed(self) -> bool {
        self.block()[0] != 1
    }

    /// Computes tightly packed storage for one mip level.
    pub const fn level_bytes(self, width: u32, height: u32) -> Option<u64> {
        if width == 0 || height == 0 {
            return None;
        }
        let [block_width, block_height, block_bytes] = self.block();
        let blocks_x = width.div_ceil(block_width);
        let blocks_y = height.div_ceil(block_height);
        match (blocks_x as u64).checked_mul(blocks_y as u64) {
            Some(blocks) => blocks.checked_mul(block_bytes as u64),
            None => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Dimensions and tightly packed payload for one mip level.
pub struct ImageMip<'a> {
    /// Mip width in texels.
    pub width: u32,
    /// Mip height in texels.
    pub height: u32,
    /// Borrowed tightly packed texel or block bytes.
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// One tightly packed two-dimensional mip region update.
pub struct TextureRegion<'a> {
    /// Destination mip level.
    pub mip_level: u32,
    /// Destination X offset in texels.
    pub x: u32,
    /// Destination Y offset in texels.
    pub y: u32,
    /// Updated width in texels.
    pub width: u32,
    /// Updated height in texels.
    pub height: u32,
    /// Borrowed tightly packed texel or block bytes.
    pub bytes: &'a [u8],
}

/// Validates a complete mip chain without allocating; dimensions clamp at one.
///
/// # Errors
///
/// Returns [`ContractError::InvalidImage`] for malformed dimensions or byte counts, or
/// [`ContractError::RangeOverflow`] when the required byte count overflows.
pub fn validate_texture_mips(
    format: TextureFormat,
    mips: &[ImageMip<'_>],
) -> Result<(), ContractError> {
    let Some(first) = mips.first() else {
        return Err(ContractError::InvalidImage);
    };
    if first.width == 0 || first.height == 0 {
        return Err(ContractError::InvalidImage);
    }
    // Native mip chains contain the terminal 1x1 level once, never repeated levels after it.
    let max_mips = u32::BITS - first.width.max(first.height).leading_zeros();
    if mips.len() > max_mips as usize {
        return Err(ContractError::InvalidImage);
    }
    let mut width = first.width;
    let mut height = first.height;
    for mip in mips {
        let expected = format
            .level_bytes(width, height)
            .ok_or(ContractError::RangeOverflow)?;
        if mip.width != width || mip.height != height || mip.bytes.len() as u64 != expected {
            return Err(ContractError::InvalidImage);
        }
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    Ok(())
}

/// A validated texture-to-texture copy rectangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextureCopyRegion {
    /// Source mip level and origin.
    pub source_mip: u32,
    /// Destination mip level and origin.
    pub destination_mip: u32,
    /// Source origin in texels.
    pub source_origin: [u32; 2],
    /// Destination origin in texels.
    pub destination_origin: [u32; 2],
    /// Copied extent in texels.
    pub extent: [u32; 2],
}

/// Validates a texture-to-texture copy before backend recording.
///
/// Source and destination must use the same format and dimensions. Overlapping
/// regions are rejected for self-copies because native APIs do not guarantee
/// memmove semantics.
///
/// # Errors
/// Returns [`ContractError::InvalidImage`] for incompatible or overlapping
/// regions and [`ContractError::RangeOverflow`] for checked arithmetic overflow.
pub fn validate_texture_copy(
    format: TextureFormat,
    destination_format: TextureFormat,
    source_extent: (u32, u32),
    destination_extent: (u32, u32),
    same_texture: bool,
    copy: TextureCopyRegion,
) -> Result<(), ContractError> {
    if format != destination_format {
        return Err(ContractError::InvalidImage);
    }
    let [width, height] = copy.extent;
    let [block_width, block_height, block_bytes] = format.block();
    let byte_count = usize::try_from(
        u64::from(width.div_ceil(block_width))
            .checked_mul(u64::from(height.div_ceil(block_height)))
            .and_then(|blocks| blocks.checked_mul(u64::from(block_bytes)))
            .ok_or(ContractError::RangeOverflow)?,
    )
    .map_err(|_| ContractError::RangeOverflow)?;
    let bytes = vec![0_u8; byte_count];
    let source = TextureRegion {
        mip_level: copy.source_mip,
        x: copy.source_origin[0],
        y: copy.source_origin[1],
        width,
        height,
        bytes: &bytes,
    };
    let destination = TextureRegion {
        mip_level: copy.destination_mip,
        x: copy.destination_origin[0],
        y: copy.destination_origin[1],
        width,
        height,
        bytes: &bytes,
    };
    validate_texture_region(format, source_extent.0, source_extent.1, 1, source)?;
    validate_texture_region(
        destination_format,
        destination_extent.0,
        destination_extent.1,
        1,
        destination,
    )?;
    if same_texture
        && copy.source_mip == copy.destination_mip
        && source_extent == destination_extent
        && copy.source_origin[0] < copy.destination_origin[0].saturating_add(width)
        && copy.destination_origin[0] < copy.source_origin[0].saturating_add(width)
        && copy.source_origin[1] < copy.destination_origin[1].saturating_add(height)
        && copy.destination_origin[1] < copy.source_origin[1].saturating_add(height)
    {
        return Err(ContractError::InvalidImage);
    }
    Ok(())
}

/// Validates one mip-region update, including compressed-block edge rules.
///
/// # Errors
///
/// Returns [`ContractError::InvalidImage`] for empty, out-of-range, misaligned, or wrongly sized
/// regions and [`ContractError::RangeOverflow`] for unrepresentable mip arithmetic.
pub fn validate_texture_region(
    format: TextureFormat,
    texture_width: u32,
    texture_height: u32,
    mip_count: u32,
    region: TextureRegion<'_>,
) -> Result<(), ContractError> {
    if texture_width == 0
        || texture_height == 0
        || mip_count == 0
        || region.mip_level >= mip_count
        || region.width == 0
        || region.height == 0
    {
        return Err(ContractError::InvalidImage);
    }
    let mip_width = texture_width
        .checked_shr(region.mip_level)
        .unwrap_or(0)
        .max(1);
    let mip_height = texture_height
        .checked_shr(region.mip_level)
        .unwrap_or(0)
        .max(1);
    let end_x = region
        .x
        .checked_add(region.width)
        .ok_or(ContractError::RangeOverflow)?;
    let end_y = region
        .y
        .checked_add(region.height)
        .ok_or(ContractError::RangeOverflow)?;
    let [block_width, block_height, block_bytes] = format.block();
    if end_x > mip_width
        || end_y > mip_height
        || !region.x.is_multiple_of(block_width)
        || !region.y.is_multiple_of(block_height)
        || (!region.width.is_multiple_of(block_width) && end_x != mip_width)
        || (!region.height.is_multiple_of(block_height) && end_y != mip_height)
    {
        return Err(ContractError::InvalidImage);
    }
    let bytes = u64::from(region.width.div_ceil(block_width))
        .checked_mul(u64::from(region.height.div_ceil(block_height)))
        .and_then(|blocks| blocks.checked_mul(u64::from(block_bytes)))
        .ok_or(ContractError::RangeOverflow)?;
    if region.bytes.len() as u64 != bytes {
        return Err(ContractError::InvalidImage);
    }
    Ok(())
}

/// Validates a complete RGBA8 mip chain.
///
/// # Errors
///
/// Returns the same errors as [`validate_texture_mips`].
pub fn validate_rgba8_mips(mips: &[ImageMip<'_>]) -> Result<(), ContractError> {
    validate_texture_mips(TextureFormat::Rgba8Unorm, mips)
}
