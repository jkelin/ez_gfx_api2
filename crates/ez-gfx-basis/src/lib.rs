//! Safe standalone Basis Universal transcoding over the native library shared with `basisu_c_sys`.
#![forbid(unsafe_op_in_unsafe_fn)]

use core::ptr::NonNull;

#[repr(C)]
struct NativeTranscoder {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn ez_basis_metadata(data: *const u8, size: u32) -> u32;
    fn ez_basis_create(data: *const u8, size: u32, target: u32) -> *mut NativeTranscoder;
    fn ez_basis_destroy(transcoder: *mut NativeTranscoder);
    fn ez_basis_level_count(transcoder: *const NativeTranscoder, data: *const u8, size: u32)
    -> u32;
    fn ez_basis_level_description(
        transcoder: *const NativeTranscoder,
        data: *const u8,
        size: u32,
        level: u32,
        width: *mut u32,
        height: *mut u32,
        blocks: *mut u32,
    ) -> bool;
    fn ez_basis_transcode_level(
        transcoder: *const NativeTranscoder,
        data: *const u8,
        size: u32,
        level: u32,
        target: u32,
        output: *mut u8,
        output_elements: u32,
    ) -> bool;
}

/// Supported GPU or raw output layouts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasisTarget {
    /// BC1 RGB blocks.
    Bc1,
    /// BC3 RGBA blocks.
    Bc3,
    /// BC7 RGBA blocks.
    Bc7,
    /// ASTC 4x4 RGBA blocks.
    Astc4x4,
    /// Linear RGBA8 pixels.
    Rgba8,
}

impl BasisTarget {
    const fn native(self) -> u32 {
        match self {
            Self::Bc1 => 2,
            Self::Bc3 => 3,
            Self::Bc7 => 6,
            Self::Astc4x4 => 10,
            Self::Rgba8 => 13,
        }
    }

    const fn bytes_per_element(self) -> u32 {
        match self {
            Self::Bc1 => 8,
            Self::Bc3 | Self::Bc7 | Self::Astc4x4 => 16,
            Self::Rgba8 => 4,
        }
    }
}

/// One transcoded standalone Basis mip.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasisMip {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Tightly packed pixels or blocks.
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Standalone Basis decoding failure.
pub enum BasisError {
    /// Input is empty, oversized, malformed, or unsupported.
    InvalidData,
    /// Valid input could not be transcoded to the requested target.
    Transcode,
}

/// Validated standalone source encoding and color metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BasisMetadata {
    /// ETC1S rather than UASTC encoding.
    pub etc1s: bool,
    /// An encoded alpha plane is present.
    pub alpha: bool,
    /// The source header marks sRGB transfer.
    pub srgb: bool,
}

/// Reads standalone source metadata without starting decompression.
///
/// # Errors
///
/// Rejects malformed, empty, oversized, non-2D, or multi-image sources.
pub fn metadata(data: &[u8]) -> Result<BasisMetadata, BasisError> {
    // Bound native metadata allocation; absent sRGB flags mean linear in the Basis header.
    if data.is_empty() || data.len() > 64 * 1024 * 1024 {
        return Err(BasisError::InvalidData);
    }
    let size = u32::try_from(data.len()).map_err(|_| BasisError::InvalidData)?;
    // SAFETY: the native routine borrows the complete readable source only for this call.
    let flags = unsafe { ez_basis_metadata(data.as_ptr(), size) };
    if flags & 1 == 0 {
        return Err(BasisError::InvalidData);
    }
    Ok(BasisMetadata {
        etc1s: flags & 2 != 0,
        alpha: flags & 4 != 0,
        srgb: flags & 8 != 0,
    })
}

struct Transcoder(NonNull<NativeTranscoder>);

impl Drop for Transcoder {
    fn drop(&mut self) {
        // SAFETY: The pointer was returned by `ez_basis_create` and is destroyed exactly once.
        unsafe { ez_basis_destroy(self.0.as_ptr()) };
    }
}

/// Transcodes one standalone, single-image 2D Basis file.
///
/// # Errors
///
/// Returns [`BasisError::InvalidData`] for malformed metadata, unsupported content, or overflow,
/// and [`BasisError::Transcode`] when native transcoding fails.
pub fn transcode(data: &[u8], target: BasisTarget) -> Result<Vec<BasisMip>, BasisError> {
    let size = u32::try_from(data.len()).map_err(|_| BasisError::InvalidData)?;
    if size == 0 {
        return Err(BasisError::InvalidData);
    }
    // This initializes the one native Basis implementation used by both KTX2 and standalone paths.
    basisu_c_sys::extra::basisu_transcoder_init();
    // SAFETY: The nonempty slice is readable for `size` bytes during every native call below.
    let transcoder = NonNull::new(unsafe { ez_basis_create(data.as_ptr(), size, target.native()) })
        .map(Transcoder)
        .ok_or(BasisError::InvalidData)?;
    // SAFETY: The handle and source slice remain valid for this call.
    let level_count = unsafe { ez_basis_level_count(transcoder.0.as_ptr(), data.as_ptr(), size) };
    if level_count == 0 {
        return Err(BasisError::InvalidData);
    }

    let mut mips = Vec::with_capacity(level_count as usize);
    for level in 0..level_count {
        let mut width = 0;
        let mut height = 0;
        let mut blocks = 0;
        // SAFETY: All outputs are writable and the handle/source remain valid.
        let described = unsafe {
            ez_basis_level_description(
                transcoder.0.as_ptr(),
                data.as_ptr(),
                size,
                level,
                &raw mut width,
                &raw mut height,
                &raw mut blocks,
            )
        };
        if !described || width == 0 || height == 0 || blocks == 0 {
            return Err(BasisError::InvalidData);
        }
        let elements = if target == BasisTarget::Rgba8 {
            width.checked_mul(height).ok_or(BasisError::InvalidData)?
        } else {
            blocks
        };
        let byte_count = elements
            .checked_mul(target.bytes_per_element())
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(BasisError::InvalidData)?;
        let mut bytes = vec![0; byte_count];
        // SAFETY: `bytes` holds exactly `elements` output blocks or pixels for the selected target.
        let succeeded = unsafe {
            ez_basis_transcode_level(
                transcoder.0.as_ptr(),
                data.as_ptr(),
                size,
                level,
                target.native(),
                bytes.as_mut_ptr(),
                elements,
            )
        };
        if !succeeded {
            return Err(BasisError::Transcode);
        }
        mips.push(BasisMip {
            width,
            height,
            bytes,
        });
    }
    Ok(mips)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_rejects_empty_and_truncated_headers() {
        for bytes in [&[][..], &[0_u8; 16][..]] {
            assert_eq!(metadata(bytes), Err(BasisError::InvalidData));
        }
    }
}
