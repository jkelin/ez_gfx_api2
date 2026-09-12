pub use crate::TextureError;
use ez_gfx_core::capability::CompressionSupport;
use ez_gfx_hal::{TextureFormat, TextureSamplerDesc};
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

mod basis;
mod dds;
mod ktx2;
mod raw;
mod telemetry;

pub use telemetry::{TextureUploadTelemetry, TextureUploadTelemetrySnapshot};

/// Maximum total byte size accepted for encoded or decoded texture data.
pub const MAX_TEXTURE_BYTES: usize = 64 * 1024 * 1024;

/// Returns mip indices from the terminal coarse level toward level zero.
pub fn coarse_to_fine_mip_levels(mip_count: u32) -> impl ExactSizeIterator<Item = u32> {
    // Empty chains intentionally produce no submissions.
    (0..mip_count).rev()
}
#[derive(Clone, Copy, Debug)]
/// Source, mip policy, residency requirement, and sampling for a texture.
pub struct TextureConfig {
    /// Encoded source format and dimensions.
    pub source: TextureSource,
    /// Generates the full mip chain from the decoded base level.
    pub generate_mips: bool,
    /// Coarse-prefix mip count gating initial readiness. Zero means optional:
    /// frame recording never waits for the texture and its stable binding
    /// samples fallback until the first real coarse mip is decoded,
    /// submitted, transfer-complete, and descriptor-safe; [`REQUIRED_MIPS_FULL`](crate::REQUIRED_MIPS_FULL)
    /// waits for the decoded total; oversized values fail at submission once
    /// the decoded total is known, never clamp.
    pub required_mips: u32,
    /// Base width in pixels.
    pub width: u32,
    /// Base height in pixels.
    pub height: u32,
    /// Requested mip count, or zero to use the decoded chain.
    pub mip_count: u32,
    /// Requested GPU storage or automatic capability-based selection.
    pub destination: TextureDestination,
    /// Texture sampling configuration.
    pub sampler: TextureSamplerDesc,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Texture source format and dimensions.
pub enum TextureSource {
    /// An RGB8 texture.
    Rgb8 {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// An RGBA8 texture.
    Rgba8 {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// A Windows bitmap.
    Bmp,
    /// A JPEG image.
    Jpeg,
    /// A PNG image.
    Png,
    /// A TGA image.
    Tga,
    /// A KTX2 container.
    Ktx2,
    /// A standalone Basis Universal payload.
    Basis,
    /// A DDS container containing supported native texture blocks.
    Dds,
    /// Tightly packed native mip bytes, ordered largest to smallest.
    Raw {
        /// Native texel or block layout.
        format: TextureFormat,
        /// Base width in texels.
        width: u32,
        /// Base height in texels.
        height: u32,
        /// Number of contiguous mip levels.
        mip_count: u32,
    },
    /// Application-defined bytes handled by a registered decoder.
    Custom(u8),
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// One tightly packed decoded or transcoded mip level.
pub struct DecodedMip {
    /// Width of this mip level in pixels.
    pub width: u32,
    /// Height of this mip level in pixels.
    pub height: u32,
    /// Row-major texel or compressed-block bytes.
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A validated two-dimensional texture with a complete mip chain.
pub struct DecodedTexture {
    /// Width of the base mip level in pixels.
    pub width: u32,
    /// Height of the base mip level in pixels.
    pub height: u32,
    /// Number of decoded mip levels.
    pub mip_count: u32,
    /// GPU storage format for every mip.
    pub format: TextureFormat,
    /// Decoded mip levels ordered from largest to smallest.
    pub mips: Vec<DecodedMip>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Requested GPU texture storage. `Auto` prefers admitted native compression.
pub enum TextureDestination {
    /// Chooses native compression when available.
    Auto,
    /// Linear normalized RGBA8.
    Rgba8Unorm,
    /// sRGB normalized RGBA8.
    Rgba8Srgb,
    /// Linear BC1.
    Bc1Unorm,
    /// sRGB BC1.
    Bc1Srgb,
    /// Linear BC3.
    Bc3Unorm,
    /// sRGB BC3.
    Bc3Srgb,
    /// Linear BC7.
    Bc7Unorm,
    /// sRGB BC7.
    Bc7Srgb,
    /// Linear ASTC 4x4.
    Astc4x4Unorm,
    /// sRGB ASTC 4x4.
    Astc4x4Srgb,
}

impl TextureDestination {
    fn format(self) -> Option<TextureFormat> {
        match self {
            Self::Auto => None,
            Self::Rgba8Unorm => Some(TextureFormat::Rgba8Unorm),
            Self::Rgba8Srgb => Some(TextureFormat::Rgba8Srgb),
            Self::Bc1Unorm => Some(TextureFormat::Bc1Unorm),
            Self::Bc1Srgb => Some(TextureFormat::Bc1Srgb),
            Self::Bc3Unorm => Some(TextureFormat::Bc3Unorm),
            Self::Bc3Srgb => Some(TextureFormat::Bc3Srgb),
            Self::Bc7Unorm => Some(TextureFormat::Bc7Unorm),
            Self::Bc7Srgb => Some(TextureFormat::Bc7Srgb),
            Self::Astc4x4Unorm => Some(TextureFormat::Astc4x4Unorm),
            Self::Astc4x4Srgb => Some(TextureFormat::Astc4x4Srgb),
        }
    }

    fn rgba_format(self) -> Result<TextureFormat, TextureError> {
        match self {
            Self::Auto | Self::Rgba8Unorm => Ok(TextureFormat::Rgba8Unorm),
            Self::Rgba8Srgb => Ok(TextureFormat::Rgba8Srgb),
            _ => Err(TextureError::Unsupported),
        }
    }
}

/// Decodes supported texture sources into validated mip levels.
pub struct TextureDecoder;
/// Thread-safe application decoder invoked on the bounded Rayon pool.
pub type TextureDecodeCallback = Arc<
    dyn Fn(&[u8], CompressionSupport) -> Result<DecodedTexture, TextureError>
        + Send
        + Sync
        + 'static,
>;

/// Decode request with any application callback retained at admission time.
#[derive(Clone)]
pub struct PreparedTextureDecode {
    source: TextureSource,
    compression: CompressionSupport,
    destination: TextureDestination,
    custom: Option<TextureDecodeCallback>,
}

static TEXTURE_DECODERS: LazyLock<RwLock<HashMap<u8, TextureDecodeCallback>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Registers one application source decoder. IDs below 128 are reserved.
///
/// # Errors
///
/// Returns [`TextureError::InvalidData`] for a reserved ID or [`TextureError::InvalidState`] when
/// the ID is already registered or the registry lock was poisoned.
pub fn register_texture_decoder(
    source: u8,
    callback: TextureDecodeCallback,
) -> Result<(), TextureError> {
    if source < 128 {
        return Err(TextureError::InvalidData);
    }
    let mut decoders = TEXTURE_DECODERS
        .write()
        .map_err(|_| TextureError::InvalidState)?;
    if decoders.contains_key(&source) {
        return Err(TextureError::InvalidState);
    }
    decoders.insert(source, callback);
    Ok(())
}

/// Removes one application source decoder.
///
/// # Errors
///
/// Returns [`TextureError::InvalidData`] for a reserved ID, [`TextureError::NotFound`] for an
/// unregistered ID, or [`TextureError::InvalidState`] when the registry lock was poisoned.
pub fn unregister_texture_decoder(source: u8) -> Result<(), TextureError> {
    if source < 128 {
        return Err(TextureError::InvalidData);
    }
    let removed = TEXTURE_DECODERS
        .write()
        .map_err(|_| TextureError::InvalidState)?
        .remove(&source);
    removed.map(|_| ()).ok_or(TextureError::NotFound)
}

impl TextureDecoder {
    /// Captures all decoder state needed by later asynchronous execution.
    ///
    /// # Errors
    ///
    /// Returns an error when a custom decoder is absent or its registry lock is poisoned.
    pub fn prepare(
        source: TextureSource,
        compression: CompressionSupport,
        destination: TextureDestination,
    ) -> Result<PreparedTextureDecode, TextureError> {
        let custom = match source {
            TextureSource::Custom(source) => Some(
                TEXTURE_DECODERS
                    .read()
                    .map_err(|_| TextureError::InvalidState)?
                    .get(&source)
                    .cloned()
                    .ok_or(TextureError::Unsupported)?,
            ),
            _ => None,
        };
        Ok(PreparedTextureDecode {
            source,
            compression,
            destination,
            custom,
        })
    }

    /// Decodes into portable linear RGBA8 storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the source is malformed, unsupported, or exceeds texture limits.
    pub fn decode(source: TextureSource, data: &[u8]) -> Result<DecodedTexture, TextureError> {
        Self::decode_for_destination(
            source,
            data,
            CompressionSupport::NONE,
            TextureDestination::Rgba8Unorm,
        )
    }

    /// Preserves or transcodes KTX2 data to the best format admitted by the backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the source is malformed or cannot target admitted compression.
    pub fn decode_with_support(
        source: TextureSource,
        data: &[u8],
        compression: CompressionSupport,
    ) -> Result<DecodedTexture, TextureError> {
        Self::decode_for_destination(source, data, compression, TextureDestination::Auto)
    }

    /// Decodes to an explicit destination, or selects one from backend capabilities for `Auto`.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed input or a destination unsupported by the source/backend.
    pub fn decode_for_destination(
        source: TextureSource,
        data: &[u8],
        compression: CompressionSupport,
        destination: TextureDestination,
    ) -> Result<DecodedTexture, TextureError> {
        Self::prepare(source, compression, destination)?.decode(data)
    }
}

impl PreparedTextureDecode {
    /// Executes the snapshotted decode request.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed input or output unsupported by the admitted backend.
    pub fn decode(&self, data: &[u8]) -> Result<DecodedTexture, TextureError> {
        if data.is_empty() || data.len() > MAX_TEXTURE_BYTES {
            return Err(TextureError::InvalidData);
        }
        match self.source {
            TextureSource::Rgb8 { width, height } => {
                let pixel_count = (width as usize)
                    .checked_mul(height as usize)
                    .ok_or(TextureError::TooLarge)?;
                if data.len() != pixel_count.checked_mul(3).ok_or(TextureError::TooLarge)? {
                    return Err(TextureError::InvalidData);
                }
                let mut rgba8 =
                    Vec::with_capacity(pixel_count.checked_mul(4).ok_or(TextureError::TooLarge)?);
                for rgb in data.chunks_exact(3) {
                    rgba8.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                }
                decoded(
                    self.destination.rgba_format()?,
                    vec![DecodedMip {
                        width,
                        height,
                        bytes: rgba8,
                    }],
                )
            }
            TextureSource::Rgba8 { width, height } => decoded(
                self.destination.rgba_format()?,
                vec![DecodedMip {
                    width,
                    height,
                    bytes: data.to_vec(),
                }],
            ),
            TextureSource::Bmp | TextureSource::Jpeg | TextureSource::Png | TextureSource::Tga => {
                let image_format = match self.source {
                    TextureSource::Bmp => image::ImageFormat::Bmp,
                    TextureSource::Jpeg => image::ImageFormat::Jpeg,
                    TextureSource::Png => image::ImageFormat::Png,
                    TextureSource::Tga => image::ImageFormat::Tga,
                    _ => unreachable!(),
                };
                let image = image::load_from_memory_with_format(data, image_format)
                    .map_err(|_| TextureError::InvalidData)?
                    .into_rgba8();
                decoded(
                    self.destination.rgba_format()?,
                    vec![DecodedMip {
                        width: image.width(),
                        height: image.height(),
                        bytes: image.into_raw(),
                    }],
                )
            }
            TextureSource::Ktx2 => ktx2::decode_ktx2(data, self.compression, self.destination),
            TextureSource::Basis => basis::decode(data, self.compression, self.destination),
            TextureSource::Dds => dds::decode(data, self.compression, self.destination),
            TextureSource::Raw {
                format,
                width,
                height,
                mip_count,
            } => raw::decode(
                data,
                format,
                width,
                height,
                mip_count,
                self.compression,
                self.destination,
            ),
            TextureSource::Custom(_) => {
                // Preparation guarantees custom requests retain exactly one callback.
                let callback = self.custom.as_ref().ok_or(TextureError::InvalidState)?;
                let texture = callback(data, self.compression)?;
                if self
                    .destination
                    .format()
                    .is_some_and(|format| format != texture.format)
                    || (texture.format.is_compressed()
                        && !basis::compression_supported(self.compression, texture.format))
                {
                    return Err(TextureError::Unsupported);
                }
                decoded(texture.format, texture.mips)
            }
        }
    }
}

/// Existing RGBA8 mip chains are preserved; one valid level is box-filtered to one pixel.
///
/// Each output texel covers its proportional source extent, so trailing odd
/// rows and columns contribute instead of being dropped. sRGB channels are
/// decoded to linear light, averaged, and re-encoded; alpha stays a straight
/// linear mean under the codebase's straight-alpha policy.
///
/// # Errors
///
/// Returns an error if the mip data is invalid, compressed, or exceeds supported limits.
pub fn generate_mips(texture: DecodedTexture) -> Result<DecodedTexture, TextureError> {
    let mut texture = decoded(texture.format, texture.mips)?;
    if texture.mip_count > 1 {
        return Ok(texture);
    }
    let srgb = match texture.format {
        TextureFormat::Rgba8Unorm => false,
        TextureFormat::Rgba8Srgb => true,
        _ => return Err(TextureError::Unsupported),
    };
    // Precompute the generated geometry and validate the full aggregate budget
    // before allocating any level.
    let first = texture.mips.first().ok_or(TextureError::InvalidData)?;
    let mut chain = Vec::with_capacity(16);
    let mut total = texture
        .format
        .level_bytes(first.width, first.height)
        .ok_or(TextureError::TooLarge)?;
    let (mut width, mut height) = (first.width, first.height);
    while width > 1 || height > 1 {
        width = (width / 2).max(1);
        height = (height / 2).max(1);
        let bytes = texture
            .format
            .level_bytes(width, height)
            .ok_or(TextureError::TooLarge)?;
        total = total
            .checked_add(bytes)
            .filter(|total| *total <= MAX_TEXTURE_BYTES as u64)
            .ok_or(TextureError::TooLarge)?;
        chain.push((width, height, bytes));
    }
    // A 256-entry table keeps sRGB decoding exact without per-texel conversion branches.
    let mut linear_table = [0.0; 256];
    for (value, slot) in linear_table.iter_mut().enumerate() {
        let byte = u8::try_from(value).map_err(|_| TextureError::TooLarge)?;
        *slot = srgb_to_linear(byte);
    }
    for (width, height, bytes) in chain {
        let source = texture.mips.last().ok_or(TextureError::InvalidData)?;
        let byte_count = usize::try_from(bytes).map_err(|_| TextureError::TooLarge)?;
        let mut rgba8 = Vec::with_capacity(byte_count);
        for y in 0..height {
            for x in 0..width {
                let (x_start, x_end) = span(x, source.width, width)?;
                let (y_start, y_end) = span(y, source.height, height)?;
                let mut sum = [0_u64; 3];
                let mut linear = [0.0; 3];
                let mut alpha = 0_u32;
                let mut samples = 0_u32;
                for source_y in y_start..y_end {
                    for source_x in x_start..x_end {
                        let offset = usize::try_from(
                            (u64::from(source_y) * u64::from(source.width) + u64::from(source_x))
                                .checked_mul(4)
                                .ok_or(TextureError::TooLarge)?,
                        )
                        .map_err(|_| TextureError::TooLarge)?;
                        for (channel, value) in sum.iter_mut().enumerate() {
                            let byte = source.bytes[offset + channel];
                            *value += u64::from(byte);
                            if srgb {
                                linear[channel] += linear_table[usize::from(byte)];
                            }
                        }
                        alpha += u32::from(source.bytes[offset + 3]);
                        samples += 1;
                    }
                }
                if samples == 0 {
                    return Err(TextureError::InvalidData);
                }
                let count = u64::from(samples);
                for (channel, value) in sum.iter().enumerate() {
                    let encoded = if srgb {
                        linear_to_srgb(linear[channel] / f64::from(samples))
                    } else {
                        // Integer round-half-up matches float round-half-away on
                        // non-negative means; the quotient cannot exceed 255.
                        u8::try_from((2 * value + count) / (2 * count))
                            .map_err(|_| TextureError::TooLarge)?
                    };
                    rgba8.push(encoded);
                }
                rgba8.push(u8::try_from(alpha / samples).map_err(|_| TextureError::TooLarge)?);
            }
        }
        texture.mips.push(DecodedMip {
            width,
            height,
            bytes: rgba8,
        });
    }
    texture.mip_count = u32::try_from(texture.mips.len()).map_err(|_| TextureError::TooLarge)?;

    Ok(texture)
}

/// Sections one output texel's source span; the final texel in the axis
/// absorbs any odd remainder so the full extent contributes.
///
/// # Errors
///
/// Returns [`TextureError::TooLarge`] when a bound exceeds the 32-bit range.
fn span(index: u32, source: u32, out: u32) -> Result<(u32, u32), TextureError> {
    let start = u64::from(index) * u64::from(source) / u64::from(out);
    let end = ((u64::from(index) + 1) * u64::from(source) / u64::from(out)).max(start + 1);
    Ok((
        u32::try_from(start).map_err(|_| TextureError::TooLarge)?,
        u32::try_from(end).map_err(|_| TextureError::TooLarge)?,
    ))
}

/// Decodes one sRGB byte to linear light (IEC 61966-2-1).
fn srgb_to_linear(byte: u8) -> f64 {
    let value = f64::from(byte) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Encodes linear light to one sRGB byte, rounded to nearest.
fn linear_to_srgb(linear: f64) -> u8 {
    let value = if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    let clamped = (value * 255.0).round().clamp(0.0, 255.0);
    // The clamped value lies in [0, 255] by construction, so the float cast
    // saturates within range instead of truncating or losing sign.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] above, so the float cast saturates in range"
    )]
    let byte = clamped as u8;
    byte
}

/// Validates mip dimensions, format-specific byte lengths, and aggregate size.
///
/// # Errors
///
/// Returns an error if the mip chain is empty or inconsistent, or exceeds supported limits.
fn decoded(format: TextureFormat, mips: Vec<DecodedMip>) -> Result<DecodedTexture, TextureError> {
    let Some(first) = mips.first() else {
        return Err(TextureError::InvalidData);
    };
    let width = first.width;
    let height = first.height;
    if width == 0 || height == 0 {
        return Err(TextureError::TooLarge);
    }
    // A native chain contains the terminal 1x1 level once, with no repeated levels after it.
    let max_mips = u32::BITS - width.max(height).leading_zeros();
    if mips.len() > max_mips as usize {
        return Err(TextureError::InvalidData);
    }
    let mut expected_width = width;
    let mut expected_height = height;
    let mut total = 0_u64;
    for mip in &mips {
        if mip.width != expected_width || mip.height != expected_height {
            return Err(TextureError::InvalidData);
        }
        let expected = format
            .level_bytes(mip.width, mip.height)
            .ok_or(TextureError::TooLarge)?;
        if mip.bytes.len() as u64 != expected {
            return Err(TextureError::InvalidData);
        }
        total = total.checked_add(expected).ok_or(TextureError::TooLarge)?;
        expected_width = (expected_width / 2).max(1);
        expected_height = (expected_height / 2).max(1);
    }
    if total > MAX_TEXTURE_BYTES as u64 {
        return Err(TextureError::TooLarge);
    }
    let mip_count = u32::try_from(mips.len()).map_err(|_| TextureError::TooLarge)?;
    Ok(DecodedTexture {
        width,
        height,
        mip_count,
        format,
        mips,
    })
}
