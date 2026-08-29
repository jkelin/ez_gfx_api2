use ez_gfx_hal::{CompletionToken, QueueKind};
use std::collections::VecDeque;

pub const MAX_TEXTURE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureSource {
    Rgb8 { width: u32, height: u32 },
    Rgba8 { width: u32, height: u32 },
    Bmp,
    Jpeg,
    Png,
    Tga,
    Ktx2,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedMip {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedTexture {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub mips: Vec<DecodedMip>,
}

pub struct TextureDecoder;
impl TextureDecoder {
    /// Inputs, decoded dimensions, and expanded RGBA payloads are bounded before allocation.
    pub fn decode(source: TextureSource, data: &[u8]) -> Result<DecodedTexture, TextureError> {
        if data.is_empty() || data.len() > MAX_TEXTURE_BYTES {
            return Err(TextureError::InvalidData);
        }
        match source {
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
                decoded(vec![DecodedMip {
                    width,
                    height,
                    rgba8,
                }])
            }
            TextureSource::Rgba8 { width, height } => decoded(vec![DecodedMip {
                width,
                height,
                rgba8: data.to_vec(),
            }]),
            TextureSource::Bmp | TextureSource::Jpeg | TextureSource::Png | TextureSource::Tga => {
                let format = match source {
                    TextureSource::Bmp => image::ImageFormat::Bmp,
                    TextureSource::Jpeg => image::ImageFormat::Jpeg,
                    TextureSource::Png => image::ImageFormat::Png,
                    TextureSource::Tga => image::ImageFormat::Tga,
                    _ => unreachable!(),
                };
                let image = image::load_from_memory_with_format(data, format)
                    .map_err(|_| TextureError::InvalidData)?
                    .into_rgba8();
                decoded(vec![DecodedMip {
                    width: image.width(),
                    height: image.height(),
                    rgba8: image.into_raw(),
                }])
            }
            TextureSource::Ktx2 => decode_ktx2(data),
        }
    }
}

/// Existing mip chains are preserved; a single valid RGBA8 level is box-filtered until both dimensions reach one.
pub fn generate_mips(texture: DecodedTexture) -> Result<DecodedTexture, TextureError> {
    let mut texture = decoded(texture.mips)?;
    if texture.mip_count > 1 {
        return Ok(texture);
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
                            *value += u32::from(source.rgba8[offset + channel]);
                        }
                        samples += 1;
                    }
                }
                for value in sum {
                    rgba8.push((value / samples) as u8);
                }
            }
        }
        texture.mips.push(DecodedMip {
            width,
            height,
            rgba8,
        });
    }
    texture.mip_count = texture.mips.len() as u32;
    Ok(texture)
}

fn decode_ktx2(data: &[u8]) -> Result<DecodedTexture, TextureError> {
    let reader = ktx2::Reader::new(data).map_err(|_| TextureError::InvalidData)?;
    let header = reader.header();
    if header.pixel_height == 0
        || header.pixel_depth != 0
        || header.layer_count > 1
        || header.face_count != 1
    {
        return Err(TextureError::Unsupported);
    }
    let levels = reader.levels().collect::<Vec<_>>();
    match (header.format, header.supercompression_scheme) {
        (Some(ktx2::Format::R8G8B8A8_UNORM | ktx2::Format::R8G8B8A8_SRGB), None) => {
            let mips = levels
                .into_iter()
                .enumerate()
                .map(|(index, level)| DecodedMip {
                    width: header
                        .pixel_width
                        .checked_shr(index as u32)
                        .unwrap_or(0)
                        .max(1),
                    height: header
                        .pixel_height
                        .checked_shr(index as u32)
                        .unwrap_or(0)
                        .max(1),
                    rgba8: level.data.to_vec(),
                })
                .collect();
            decoded(mips)
        }
        (None, None) if reader.color_model() == Some(ktx2::ColorModel::UASTC) => decode_ktx2_basis(
            data,
            header.pixel_width,
            header.pixel_height,
            header.level_count.max(1),
        ),
        (None, Some(ktx2::SupercompressionScheme::BasisLZ)) => decode_ktx2_basis(
            data,
            header.pixel_width,
            header.pixel_height,
            header.level_count.max(1),
        ),
        _ => Err(TextureError::Unsupported),
    }
}

/// The maintained Basis Universal C API validates UASTC/ETC1S payloads, codebooks, and slices; arrays and cubemaps are rejected by this 2D texture contract.
fn decode_ktx2_basis(
    data: &[u8],
    width: u32,
    height: u32,
    mip_count: u32,
) -> Result<DecodedTexture, TextureError> {
    use basisu_c_sys::{
        TranscodeTargetFormat,
        extra::{
            BasisuTranscoder, ChannelType, SupportedTextureCompression, basisu_transcoder_init,
        },
    };

    basisu_transcoder_init();
    let transcoder = BasisuTranscoder::new(
        data,
        SupportedTextureCompression::empty(),
        ChannelType::Rgba,
    )
    .map_err(|_| TextureError::InvalidData)?;
    let info = transcoder.get_info();
    if info.width != width
        || info.height != height
        || info.levels != mip_count
        || info.layers > 1
        || info.faces != 1
    {
        return Err(TextureError::InvalidData);
    }
    let image = transcoder
        .transcode(Some(TranscodeTargetFormat::RGBA32), Some(false))
        .map_err(|_| TextureError::InvalidData)?;
    if image.mip_level_count != mip_count {
        return Err(TextureError::InvalidData);
    }

    let mut offset = 0_usize;
    let mut mips = Vec::with_capacity(mip_count as usize);
    for level in 0..mip_count {
        let mip_width = width.checked_shr(level).unwrap_or(0).max(1);
        let mip_height = height.checked_shr(level).unwrap_or(0).max(1);
        let byte_count = (mip_width as usize)
            .checked_mul(mip_height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(TextureError::TooLarge)?;
        let end = offset
            .checked_add(byte_count)
            .ok_or(TextureError::TooLarge)?;
        let rgba8 = image
            .data
            .get(offset..end)
            .ok_or(TextureError::InvalidData)?;
        mips.push(DecodedMip {
            width: mip_width,
            height: mip_height,
            rgba8: rgba8.to_vec(),
        });
        offset = end;
    }
    if offset != image.data.len() {
        return Err(TextureError::InvalidData);
    }
    decoded(mips)
}

fn decoded(mips: Vec<DecodedMip>) -> Result<DecodedTexture, TextureError> {
    let Some(first) = mips.first() else {
        return Err(TextureError::InvalidData);
    };
    let width = first.width;
    let height = first.height;
    let mut expected_width = width;
    let mut expected_height = height;
    let mut total = 0_u64;
    for mip in &mips {
        if mip.width != expected_width || mip.height != expected_height {
            return Err(TextureError::InvalidData);
        }
        let expected = u64::from(mip.width)
            .checked_mul(u64::from(mip.height))
            .and_then(|value| value.checked_mul(4))
            .ok_or(TextureError::TooLarge)?;
        if mip.rgba8.len() as u64 != expected {
            return Err(TextureError::InvalidData);
        }
        total = total.checked_add(expected).ok_or(TextureError::TooLarge)?;
        expected_width = (expected_width / 2).max(1);
        expected_height = (expected_height / 2).max(1);
    }
    if width == 0 || height == 0 || total > MAX_TEXTURE_BYTES as u64 {
        return Err(TextureError::TooLarge);
    }
    let mip_count = u32::try_from(mips.len()).map_err(|_| TextureError::TooLarge)?;
    Ok(DecodedTexture {
        width,
        height,
        mip_count,
        mips,
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureId {
    slot: u32,
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
pub enum TextureEvent {
    Resident {
        texture: TextureId,
        binding: u32,
        resident_mips: u32,
    },
    Unloaded {
        texture: TextureId,
    },
}

pub struct TextureRegistry {
    capacity: u32,
    binding_capacity: u32,
    slots: Vec<Slot>,
    free: Vec<u32>,
    events: Vec<TextureEvent>,
}
impl TextureRegistry {
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
        let slot = self.slots.len() as u32;
        self.slots.push(Slot {
            generation: 1,
            state: Some(TextureState::Allocated),
        });
        Ok(TextureId {
            slot,
            generation: 1,
        })
    }

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
                        slot: slot as u32,
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
        entry.state = None;
        entry.generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        self.free.push(texture.slot);
        Ok(())
    }

    pub fn unload(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation || entry.state.is_none() {
            return Err(TextureError::NotFound);
        }
        entry.state = None;
        entry.generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        self.free.push(texture.slot);
        self.events.push(TextureEvent::Unloaded { texture });
        Ok(())
    }

    pub fn reserved_binding(&self, texture: TextureId) -> Result<u32, TextureError> {
        let _ = self.state(texture)?;
        if texture.slot >= self.binding_capacity {
            return Err(TextureError::CapacityExceeded);
        }
        Ok(texture.slot)
    }

    pub fn drain_events(&mut self) -> Vec<TextureEvent> {
        core::mem::take(&mut self.events)
    }
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
pub enum TextureError {
    InvalidData,
    Unsupported,
    TooLarge,
    InvalidCapacity,
    CapacityExceeded,
    InvalidState,
    NotReady,
    NotFound,
    GenerationExhausted,
}
