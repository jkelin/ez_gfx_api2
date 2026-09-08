use std::sync::Arc;

use ez_gfx::raw::{ContextHandle, TextureHandle};
use ez_gfx::{SamplerAddressMode, SamplerFilter, TextureSamplerDesc, TextureSource, raw};

use super::{
    EZ_GFX_MAX_BOUNDARY_BYTES, EzGfxContext, EzGfxDecodedTexture, EzGfxResult, EzGfxTexture,
    EzGfxTextureDecoderCallback, EzGfxTextureDecoderReleaseCallback, EzGfxTextureDesc,
    IntoFfiResult, catch_status, catch_void, validate_optional_bounded_string,
};

// Zero/stale/wrong-kind packed handles fail before any context access.
fn context_handle(raw: EzGfxContext) -> Result<ContextHandle, EzGfxResult> {
    ContextHandle::from_raw(raw).map_err(|_| EzGfxResult::InvalidContext)
}

fn texture_handle(raw: EzGfxTexture) -> Result<TextureHandle, EzGfxResult> {
    TextureHandle::from_raw(raw).map_err(|_| EzGfxResult::InvalidContext)
}

// Unknown sampler bytes fail closed instead of defaulting.
pub(crate) fn sampler_address_from_abi(value: u8) -> Result<SamplerAddressMode, EzGfxResult> {
    match value {
        0 => Ok(SamplerAddressMode::Repeat),
        1 => Ok(SamplerAddressMode::Clamp),
        _ => Err(EzGfxResult::InvalidArgument),
    }
}

// Auto is a request policy, never a concrete decoder output format.
fn decoded_format_from_abi(value: u8) -> Option<ez_gfx::TextureFormat> {
    match value {
        0 => Some(ez_gfx::TextureFormat::Rgba8Unorm),
        2 => Some(ez_gfx::TextureFormat::Rgba8Srgb),
        3 => Some(ez_gfx::TextureFormat::Bc1Unorm),
        4 => Some(ez_gfx::TextureFormat::Bc1Srgb),
        5 => Some(ez_gfx::TextureFormat::Bc3Unorm),
        6 => Some(ez_gfx::TextureFormat::Bc3Srgb),
        7 => Some(ez_gfx::TextureFormat::Bc7Unorm),
        8 => Some(ez_gfx::TextureFormat::Bc7Srgb),
        9 => Some(ez_gfx::TextureFormat::Astc4x4Unorm),
        10 => Some(ez_gfx::TextureFormat::Astc4x4Srgb),
        _ => None,
    }
}
// The release guard also runs when successful callback output fails Rust-side validation.

struct DecodedTextureRelease {
    texture: EzGfxDecodedTexture,
    release: unsafe extern "C" fn(*const EzGfxDecodedTexture, *mut core::ffi::c_void),
    user_data: usize,
}

impl Drop for DecodedTextureRelease {
    fn drop(&mut self) {
        // SAFETY: Successful decoder output remains owned by the callback until this paired release.
        unsafe { (self.release)(&raw const self.texture, self.user_data as *mut _) };
    }
}

#[unsafe(no_mangle)]
/// Registers one application decoder for a custom source code in `128..=255`.
///
/// The callback may run concurrently. Successful output is copied and then released exactly once.
///
/// # Safety
///
/// `user_data` and both callbacks must remain valid until every accepted texture request using
/// this source has completed and the decoder has been unregistered.
pub unsafe extern "C" fn ez_gfx_texture_decoder_register(
    source_format: u8,
    callback: EzGfxTextureDecoderCallback,
    release: EzGfxTextureDecoderReleaseCallback,
    user_data: *mut core::ffi::c_void,
) -> EzGfxResult {
    catch_status(|| {
        let (Some(callback), Some(release)) = (callback, release) else {
            return EzGfxResult::InvalidArgument;
        };
        let user_data = user_data as usize;
        let decoder: ez_gfx::TextureDecodeCallback = Arc::new(move |data, compression| {
            let mut output = EzGfxDecodedTexture {
                format: u8::MAX,
                mip_count: 0,
                mips: core::ptr::null(),
            };
            let compression =
                u8::from(compression.intersects(ez_gfx_core::capability::CompressionSupport::BC))
                    | (u8::from(
                        compression.intersects(ez_gfx_core::capability::CompressionSupport::ASTC),
                    ) << 1);
            // SAFETY: Input spans this call; output is writable; callback/user data satisfy the
            // registration contract.
            let status = unsafe {
                callback(
                    data.as_ptr(),
                    data.len(),
                    compression,
                    &raw mut output,
                    user_data as *mut _,
                )
            };
            if status != EzGfxResult::Ok {
                return Err(ez_gfx::TextureError::InvalidData);
            }
            let guard = DecodedTextureRelease {
                texture: output,
                release,
                user_data,
            };
            let format = decoded_format_from_abi(guard.texture.format)
                .ok_or(ez_gfx::TextureError::InvalidData)?;
            let mip_count = usize::try_from(guard.texture.mip_count)
                .map_err(|_| ez_gfx::TextureError::TooLarge)?;
            if mip_count == 0
                || mip_count > 32
                || guard.texture.mips.is_null()
                || !guard.texture.mips.is_aligned()
            {
                return Err(ez_gfx::TextureError::InvalidData);
            }
            // SAFETY: The callback retains this aligned array through the paired release guard.
            let source_mips = unsafe { core::slice::from_raw_parts(guard.texture.mips, mip_count) };
            let mut total = 0_usize;
            let mut mips = Vec::with_capacity(mip_count);
            for mip in source_mips {
                total = total
                    .checked_add(mip.data_size)
                    .ok_or(ez_gfx::TextureError::TooLarge)?;
                if mip.data.is_null()
                    || mip.data_size == 0
                    || mip.data_size > 64 * 1024 * 1024
                    || total > 64 * 1024 * 1024
                {
                    return Err(ez_gfx::TextureError::InvalidData);
                }
                // SAFETY: The callback retains each byte range through the paired release guard.
                let bytes = unsafe { core::slice::from_raw_parts(mip.data, mip.data_size) };
                mips.push(ez_gfx::DecodedMip {
                    width: mip.width,
                    height: mip.height,
                    bytes: bytes.to_vec(),
                });
            }
            let first = mips.first().ok_or(ez_gfx::TextureError::InvalidData)?;
            Ok(ez_gfx::DecodedTexture {
                width: first.width,
                height: first.height,
                mip_count: guard.texture.mip_count,
                format,
                mips,
            })
        });
        match ez_gfx::register_texture_decoder(source_format, decoder) {
            Ok(()) => EzGfxResult::Ok,
            Err(_) => EzGfxResult::InvalidArgument,
        }
    })
}

#[unsafe(no_mangle)]
/// Unregisters one custom source decoder. Accepted requests retain their callback.
pub extern "C" fn ez_gfx_texture_decoder_unregister(source_format: u8) -> EzGfxResult {
    catch_status(|| match ez_gfx::unregister_texture_decoder(source_format) {
        Ok(()) => EzGfxResult::Ok,
        Err(_) => EzGfxResult::InvalidArgument,
    })
}

#[unsafe(no_mangle)]
/// Loads texture bytes, configures sampling and mip residency, and returns a texture handle.
///
/// # Safety
///
/// Non-null `data` must be readable for `data_size` bytes, non-null `desc` readable for one aligned descriptor, and non-null `out_texture` writable for one aligned handle. The optional descriptor label must be either null with zero length or non-null and readable for its exact nonzero UTF-8 byte length without embedded NUL bytes.
pub unsafe extern "C" fn ez_gfx_texture_load(
    data: *const u8,
    data_size: usize,
    desc: *const EzGfxTextureDesc,
    out_texture: *mut EzGfxTexture,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if data.is_null()
            || desc.is_null()
            || out_texture.is_null()
            || data_size == 0
            || data_size > EZ_GFX_MAX_BOUNDARY_BYTES
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `desc` is non-null and readable for this call.
        let desc = unsafe { desc.read() };
        if desc.generate_mips > 1
            || desc.destination_format > 10
            || desc.min_filter > 1
            || desc.mag_filter > 1
            || !desc.max_anisotropy.is_finite()
            || !(1.0..=16.0).contains(&desc.max_anisotropy)
            || validate_optional_bounded_string(desc.debug_label, desc.debug_label_length).is_err()
        {
            return EzGfxResult::InvalidArgument;
        }
        let source = match desc.source_format {
            0 => TextureSource::Rgb8 {
                width: desc.width,
                height: desc.height,
            },
            1 => TextureSource::Rgba8 {
                width: desc.width,
                height: desc.height,
            },
            2 => TextureSource::Bmp,
            3 => TextureSource::Jpeg,
            4 => TextureSource::Png,
            5 => TextureSource::Tga,
            6 => TextureSource::Ktx2,
            7 => TextureSource::Basis,
            8 => TextureSource::Dds,
            9 => match decoded_format_from_abi(desc.destination_format) {
                // Raw ingestion names its storage explicitly; Auto is a request
                // policy, never a concrete stored format.
                Some(format) => TextureSource::Raw {
                    format,
                    width: desc.width,
                    height: desc.height,
                    mip_count: desc.mip_count,
                },
                None => return EzGfxResult::InvalidArgument,
            },
            custom @ 128..=u8::MAX => TextureSource::Custom(custom),
            _ => return EzGfxResult::InvalidArgument,
        };
        // SAFETY: `data_size` is validated and the caller retains the input through this call.
        let bytes = unsafe { core::slice::from_raw_parts(data, data_size) };
        let filter = |value| match value {
            0 => SamplerFilter::Nearest,
            _ => SamplerFilter::Linear,
        };
        let [Ok(address_u), Ok(address_v), Ok(address_w)] = [
            sampler_address_from_abi(desc.address_mode_u),
            sampler_address_from_abi(desc.address_mode_v),
            sampler_address_from_abi(desc.address_mode_w),
        ] else {
            return EzGfxResult::InvalidArgument;
        };
        let destination = match desc.destination_format {
            0 => ez_gfx::TextureDestination::Rgba8Unorm,
            1 => ez_gfx::TextureDestination::Auto,
            2 => ez_gfx::TextureDestination::Rgba8Srgb,
            3 => ez_gfx::TextureDestination::Bc1Unorm,
            4 => ez_gfx::TextureDestination::Bc1Srgb,
            5 => ez_gfx::TextureDestination::Bc3Unorm,
            6 => ez_gfx::TextureDestination::Bc3Srgb,
            7 => ez_gfx::TextureDestination::Bc7Unorm,
            8 => ez_gfx::TextureDestination::Bc7Srgb,
            9 => ez_gfx::TextureDestination::Astc4x4Unorm,
            10 => ez_gfx::TextureDestination::Astc4x4Srgb,
            _ => return EzGfxResult::InvalidArgument,
        };
        let config = ez_gfx::TextureConfig {
            width: desc.width,
            height: desc.height,
            mip_count: desc.mip_count,
            destination,
            sampler: TextureSamplerDesc {
                min_filter: filter(desc.min_filter),
                mag_filter: filter(desc.mag_filter),
                max_anisotropy: desc.max_anisotropy,
                address_u,
                address_v,
                address_w,
            },
        };
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        match raw::load_texture(context, source, bytes, desc.generate_mips != 0, &config) {
            Ok(texture) => {
                // SAFETY: `out_texture` is non-null and writable for this call.
                unsafe { out_texture.write(texture.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Cancels a texture request before native transfer submission.
pub extern "C" fn ez_gfx_texture_cancel(
    texture: EzGfxTexture,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let texture = match texture_handle(texture) {
            Ok(texture) => texture,
            Err(error) => return error,
        };
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        raw::cancel_texture_load(context, texture).into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Queries the binding index assigned to a loaded texture.
///
/// # Safety
///
/// A non-null `out_binding` must address one writable, aligned `u32` for this call.
pub unsafe extern "C" fn ez_gfx_texture_get_binding(
    texture: EzGfxTexture,
    out_binding: *mut u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_binding.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = match context_handle(context) {
            Ok(value) => value,
            Err(error) => return error,
        };
        let texture = match texture_handle(texture) {
            Ok(value) => value,
            Err(error) => return error,
        };
        match raw::texture_binding(context, texture) {
            Ok(binding) => {
                // SAFETY: `out_binding` is non-null and writable for this call.
                unsafe { out_binding.write(binding) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Queries the resident and total mip counts for a loaded texture.
///
/// # Safety
///
/// Each non-null output pointer must address one writable, aligned `u32` for this call.
pub unsafe extern "C" fn ez_gfx_texture_get_residency(
    texture: EzGfxTexture,
    out_resident_mips: *mut u32,
    out_total_mips: *mut u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_resident_mips.is_null() || out_total_mips.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = match context_handle(context) {
            Ok(value) => value,
            Err(error) => return error,
        };
        let texture = match texture_handle(texture) {
            Ok(value) => value,
            Err(error) => return error,
        };
        match raw::texture_residency(context, texture) {
            Ok((resident, total)) => {
                // SAFETY: Both output pointers are non-null and writable for this call.
                unsafe {
                    out_resident_mips.write(resident);
                    out_total_mips.write(total);
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Sets the contiguous coarse mip count exposed through the stable texture binding.
pub extern "C" fn ez_gfx_texture_set_residency(
    texture: EzGfxTexture,
    resident_mips: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let context = match context_handle(context) {
            Ok(value) => value,
            Err(error) => return error,
        };
        let texture = match texture_handle(texture) {
            Ok(value) => value,
            Err(error) => return error,
        };
        raw::set_texture_residency(context, texture, resident_mips).into_ffi_result()
    })
}
#[unsafe(no_mangle)]
/// Copies and asynchronously uploads one validated texture sub-rectangle.
///
/// # Safety
///
/// `desc` must address one readable aligned descriptor. Its non-null `data` must remain readable
/// for exactly `data_size` bytes through this call.
pub unsafe extern "C" fn ez_gfx_update_texture_region(
    texture: EzGfxTexture,
    desc: *const super::EzGfxTextureRegionDesc,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `desc` is non-null and readable for this call.
        let desc = unsafe { desc.read() };
        if desc.data.is_null() || desc.data_size == 0 || desc.data_size > EZ_GFX_MAX_BOUNDARY_BYTES
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: The validated range remains readable through this call.
        let bytes = unsafe { core::slice::from_raw_parts(desc.data, desc.data_size) };
        let context = match context_handle(context) {
            Ok(value) => value,
            Err(error) => return error,
        };
        let texture = match texture_handle(texture) {
            Ok(value) => value,
            Err(error) => return error,
        };
        raw::update_texture_region(
            context,
            texture,
            ez_gfx::TextureRegion {
                mip_level: desc.mip_level,
                x: desc.x,
                y: desc.y,
                width: desc.width,
                height: desc.height,
                bytes,
            },
        )
        .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Returns a snapshot of monotonic context-wide texture upload counters.
///
/// # Safety
///
/// `out_telemetry` must address one writable aligned telemetry structure.
pub unsafe extern "C" fn ez_gfx_texture_get_upload_telemetry(
    out_telemetry: *mut super::EzGfxTextureUploadTelemetry,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_telemetry.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = match context_handle(context) {
            Ok(value) => value,
            Err(error) => return error,
        };
        match raw::texture_upload_telemetry(context) {
            Ok(snapshot) => {
                // SAFETY: The output is non-null and writable for this call.
                unsafe {
                    out_telemetry.write(super::EzGfxTextureUploadTelemetry {
                        decode_microseconds: snapshot.decode_microseconds,
                        staging_bytes: snapshot.staging_bytes,
                        queue_latency_microseconds: snapshot.queue_latency_microseconds,
                        handoff_latency_microseconds: snapshot.handoff_latency_microseconds,
                    });
                }
                EzGfxResult::Ok
            }
            Err(error) => error.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Unloads a texture from the context.
pub extern "C" fn ez_gfx_texture_unload(texture: EzGfxTexture, context: EzGfxContext) {
    catch_void(|| {
        if let (Ok(context), Ok(texture)) = (context_handle(context), texture_handle(texture)) {
            raw::unload_texture(context, texture);
        }
    });
}
