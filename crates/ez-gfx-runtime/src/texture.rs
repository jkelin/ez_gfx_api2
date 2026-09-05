use ez_gfx_core::capability::CompressionSupport;
use ez_gfx_hal::{CompletionToken, QueueKind, TextureFormat};
use std::{
    collections::{HashMap, VecDeque},
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
/// # Errors
///
/// Returns an error if the mip data is invalid, compressed, or exceeds supported limits.
pub fn generate_mips(texture: DecodedTexture) -> Result<DecodedTexture, TextureError> {
    let mut texture = decoded(texture.format, texture.mips)?;
    if texture.mip_count > 1 {
        return Ok(texture);
    }
    if !matches!(
        texture.format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb
    ) {
        return Err(TextureError::Unsupported);
    }
    while texture
        .mips
        .last()
        .is_some_and(|mip| mip.width > 1 || mip.height > 1)
    {
        let source = texture.mips.last().ok_or(TextureError::InvalidData)?;
        let width = (source.width / 2).max(1);
        let height = (source.height / 2).max(1);
        let byte_count = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(TextureError::TooLarge)?;
        let mut rgba8 = Vec::with_capacity(byte_count);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0_u32; 4];
                let mut samples = 0_u32;
                for source_y in y * 2..(y * 2 + 2).min(source.height) {
                    for source_x in x * 2..(x * 2 + 2).min(source.width) {
                        let offset =
                            ((source_y as usize * source.width as usize) + source_x as usize) * 4;
                        for (channel, value) in sum.iter_mut().enumerate() {
                            *value += u32::from(source.bytes[offset + channel]);
                        }
                        samples += 1;
                    }
                }
                for value in sum {
                    rgba8.push(u8::try_from(value / samples).map_err(|_| TextureError::TooLarge)?);
                }
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Generational handle identifying a texture registry slot.
pub struct TextureId {
    /// Index of the registry slot and reserved descriptor binding.
    slot: u32,
    /// Revision used to reject stale texture handles.
    generation: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TextureState {
    Allocated,
    Uploading {
        binding: u32,
        resident_mips: u32,
        pending: VecDeque<(CompletionToken, u32)>,
    },
    Resident {
        binding: u32,
        resident_mips: u32,
    },
}

#[derive(Clone, Debug)]
struct Slot {
    generation: u32,
    state: Option<TextureState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Texture residency event.
pub enum TextureEvent {
    /// A resident texture binding.
    Resident {
        /// Texture handle.
        texture: TextureId,
        /// Descriptor binding.
        binding: u32,
        /// Number of resident mip levels.
        resident_mips: u32,
    },
    /// A texture was unloaded.
    Unloaded {
        /// Texture handle.
        texture: TextureId,
    },
}

/// Tracks texture allocation, upload completion, residency, and descriptor bindings.
pub struct TextureRegistry {
    /// Maximum number of texture slots.
    capacity: u32,
    /// Maximum number of descriptor bindings addressable by texture slots.
    binding_capacity: u32,
    /// Generational storage for texture residency states.
    slots: Vec<Slot>,
    /// Reusable indices of vacant texture slots.
    free: Vec<u32>,
    /// Residency and unload notifications awaiting retrieval.
    events: Vec<TextureEvent>,
}
impl TextureRegistry {
    /// Creates an empty registry with nonzero slot and binding limits.
    ///
    /// # Errors
    ///
    /// Returns an error if either capacity is zero.
    pub fn new(capacity: u32, binding_capacity: u32) -> Result<Self, TextureError> {
        if capacity == 0 || binding_capacity == 0 {
            return Err(TextureError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            binding_capacity,
            slots: Vec::new(),
            free: Vec::new(),
            events: Vec::new(),
        })
    }

    /// Allocates or reuses a texture slot for a pending upload.
    ///
    /// # Errors
    ///
    /// Returns an error if no texture slot remains available.
    ///
    /// # Panics
    ///
    /// The registry capacity bounds every texture slot conversion.
    pub fn begin_upload(&mut self) -> Result<TextureId, TextureError> {
        if let Some(slot) = self.free.pop() {
            let entry = &mut self.slots[slot as usize];
            entry.state = Some(TextureState::Allocated);
            return Ok(TextureId {
                slot,
                generation: entry.generation,
            });
        }
        if self.slots.len() >= self.capacity as usize {
            return Err(TextureError::CapacityExceeded);
        }
        let slot = u32::try_from(self.slots.len()).map_err(|_| TextureError::CapacityExceeded)?;
        self.slots.push(Slot {
            generation: 1,
            state: Some(TextureState::Allocated),
        });
        Ok(TextureId {
            slot,
            generation: 1,
        })
    }

    /// Associates an allocated texture with its initial upload completion token.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid, its binding exceeds the binding capacity, or the texture is not allocated.
    pub fn mark_submitted(
        &mut self,
        texture: TextureId,
        completion: CompletionToken,
    ) -> Result<(), TextureError> {
        let binding = texture.slot;
        if binding >= self.binding_capacity {
            return Err(TextureError::CapacityExceeded);
        }
        let state = self.state_mut(texture)?;
        if *state != TextureState::Allocated {
            return Err(TextureError::InvalidState);
        }
        *state = TextureState::Uploading {
            binding,
            resident_mips: 0,
            pending: VecDeque::from([(completion, 1)]),
        };
        Ok(())
    }

    /// Applies completed uploads for one queue and emits residency notifications.
    ///
    /// # Errors
    ///
    /// Returns an error if a registry slot cannot be represented by a texture handle.
    pub fn poll(&mut self, queue: QueueKind, completed: u64) -> Result<usize, TextureError> {
        let mut changed = 0;
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            let Some(TextureState::Uploading {
                binding,
                resident_mips,
                pending,
            }) = entry.state.as_mut()
            else {
                continue;
            };
            while let Some((token, target_mips)) = pending.front().copied() {
                if token.queue != queue || token.value > completed {
                    break;
                }
                pending.pop_front();
                *resident_mips = target_mips;
                self.events.push(TextureEvent::Resident {
                    texture: TextureId {
                        slot: u32::try_from(slot).map_err(|_| TextureError::CapacityExceeded)?,
                        generation: entry.generation,
                    },
                    binding: *binding,
                    resident_mips: target_mips,
                });
                changed += 1;
            }
            if pending.is_empty() {
                entry.state = Some(TextureState::Resident {
                    binding: *binding,
                    resident_mips: *resident_mips,
                });
            }
        }
        Ok(changed)
    }

    /// Returns the descriptor binding once at least one mip is resident.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or the texture has no resident mip levels.
    pub fn binding_index(&self, texture: TextureId) -> Result<u32, TextureError> {
        match self.state(texture)? {
            TextureState::Uploading {
                binding,
                resident_mips,
                ..
            } if *resident_mips > 0 => Ok(*binding),
            TextureState::Resident { binding, .. } => Ok(*binding),
            _ => Err(TextureError::NotReady),
        }
    }

    /// Returns the number of mip levels currently resident.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or the texture has no resident mip levels.
    pub fn resident_mips(&self, texture: TextureId) -> Result<u32, TextureError> {
        match self.state(texture)? {
            TextureState::Uploading { resident_mips, .. } if *resident_mips > 0 => {
                Ok(*resident_mips)
            }
            TextureState::Resident { resident_mips, .. } => Ok(*resident_mips),
            _ => Err(TextureError::NotReady),
        }
    }

    /// Higher mip counts queue behind prior transfers; duplicate or decreasing targets are rejected.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid, the requested mip count is zero or not increasing, or the texture is in an incompatible state.
    pub fn mark_mips_submitted(
        &mut self,
        texture: TextureId,
        resident_mips: u32,
        completion: CompletionToken,
    ) -> Result<(), TextureError> {
        if resident_mips == 0 {
            return Err(TextureError::InvalidData);
        }
        let state = self.state_mut(texture)?;
        match state {
            TextureState::Uploading {
                resident_mips: current,
                pending,
                ..
            } => {
                let highest = pending.back().map_or(*current, |(_, target)| *target);
                if resident_mips <= highest {
                    return Err(TextureError::InvalidState);
                }
                pending.push_back((completion, resident_mips));
                Ok(())
            }
            TextureState::Resident {
                binding,
                resident_mips: current,
            } if resident_mips > *current => {
                *state = TextureState::Uploading {
                    binding: *binding,
                    resident_mips: *current,
                    pending: VecDeque::from([(completion, resident_mips)]),
                };
                Ok(())
            }
            _ => Err(TextureError::InvalidState),
        }
    }

    /// Rolls back an unexposed allocation/upload without publishing an unload event.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot does not exist, the handle or state is invalid, or the generation counter is exhausted.
    pub fn cancel_upload(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation
            || !matches!(
                entry.state,
                Some(TextureState::Allocated | TextureState::Uploading { .. })
            )
        {
            return Err(TextureError::InvalidState);
        }
        // Exhaustion fails before clearing state so callers never receive an error after destruction.
        let next_generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        entry.state = None;
        entry.generation = next_generation;
        self.free.push(texture.slot);
        Ok(())
    }

    /// Invalidates a submitted texture while withholding its descriptor slot from reuse.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is stale or its generation cannot advance.
    pub fn retire(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation || entry.state.is_none() {
            return Err(TextureError::NotFound);
        }
        let next_generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        entry.state = None;
        entry.generation = next_generation;
        self.events.push(TextureEvent::Unloaded { texture });
        Ok(())
    }

    /// Releases a retired descriptor slot after native transfer and frame dependencies complete.
    ///
    /// # Errors
    ///
    /// Returns an error unless `texture` is the immediately preceding generation of a withheld slot.
    pub fn release_retired(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.state.is_some()
            || entry.generation != texture.generation.saturating_add(1)
            || self.free.contains(&texture.slot)
        {
            return Err(TextureError::InvalidState);
        }
        self.free.push(texture.slot);
        Ok(())
    }

    /// Releases a texture slot, invalidates its handle, and emits an unload notification.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle does not identify a current texture or the generation counter is exhausted.
    pub fn unload(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation || entry.state.is_none() {
            return Err(TextureError::NotFound);
        }
        // Exhaustion fails before clearing state or omitting the corresponding unload event.
        let next_generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        entry.state = None;
        entry.generation = next_generation;
        self.free.push(texture.slot);
        self.events.push(TextureEvent::Unloaded { texture });
        Ok(())
    }

    /// Invalidates every texture slot and discards queued events.
    ///
    /// # Errors
    ///
    /// Returns an error without mutation if an occupied slot cannot advance its generation.
    pub fn clear(&mut self) -> Result<(), TextureError> {
        if self
            .slots
            .iter()
            .any(|entry| entry.state.is_some() && entry.generation == u32::MAX)
        {
            return Err(TextureError::GenerationExhausted);
        }

        self.free.clear();
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            if entry.state.take().is_some() {
                entry.generation += 1;
            }
            if entry.generation != u32::MAX {
                self.free
                    .push(u32::try_from(slot).map_err(|_| TextureError::CapacityExceeded)?);
            }
        }
        self.events.clear();
        Ok(())
    }

    /// Returns the descriptor binding reserved by an existing texture handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or its reserved binding exceeds the binding capacity.
    pub fn reserved_binding(&self, texture: TextureId) -> Result<u32, TextureError> {
        let _ = self.state(texture)?;
        if texture.slot >= self.binding_capacity {
            return Err(TextureError::CapacityExceeded);
        }
        Ok(texture.slot)
    }

    /// Removes and returns all queued texture notifications.
    pub fn drain_events(&mut self) -> Vec<TextureEvent> {
        core::mem::take(&mut self.events)
    }
    /// Resolves a current texture handle to its residency state.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle does not identify a current texture.
    fn state(&self, texture: TextureId) -> Result<&TextureState, TextureError> {
        let entry = self
            .slots
            .get(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation {
            return Err(TextureError::NotFound);
        }
        entry.state.as_ref().ok_or(TextureError::NotFound)
    }
    /// Resolves a current texture handle to mutable residency state.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle does not identify a current texture.
    fn state_mut(&mut self, texture: TextureId) -> Result<&mut TextureState, TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation {
            return Err(TextureError::NotFound);
        }
        entry.state.as_mut().ok_or(TextureError::NotFound)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Error reported while decoding or tracking textures.
/// Texture processing error.
pub enum TextureError {
    /// Encoded bytes or decoded mip data are malformed.
    InvalidData,
    /// The texture format or layout is not supported.
    Unsupported,
    /// Texture dimensions, byte size, or mip count exceed supported limits.
    TooLarge,
    /// A registry limit is zero.
    InvalidCapacity,
    /// No texture slot or descriptor binding remains available.
    CapacityExceeded,
    /// The requested operation is not allowed in the texture's current state.
    InvalidState,
    /// The texture has no resident mip levels yet.
    NotReady,
    /// The texture handle does not identify a current allocation.
    NotFound,
    /// A reused slot can no longer advance its generation counter.
    GenerationExhausted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_exhaustion_does_not_partially_cancel_or_unload() {
        for cancel in [true, false] {
            let mut registry = TextureRegistry::new(1, 1).unwrap();
            registry.slots.push(Slot {
                generation: u32::MAX,
                state: Some(TextureState::Allocated),
            });
            let texture = TextureId {
                slot: 0,
                generation: u32::MAX,
            };

            let result = if cancel {
                registry.cancel_upload(texture)
            } else {
                registry.unload(texture)
            };

            assert_eq!(result, Err(TextureError::GenerationExhausted));
            assert!(registry.state(texture).is_ok());
        }
    }
}
