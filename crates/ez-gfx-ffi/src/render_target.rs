use ez_gfx::raw::{ContextHandle, RenderTargetHandle};
use ez_gfx::{ClearValue, Format, TargetDeclaration, TargetError, TargetUsage, raw};

use super::{
    EzGfxContext, EzGfxFrame, EzGfxRenderTarget, EzGfxRenderTargetDesc, EzGfxResult, IntoFfiResult,
    catch_status, catch_void, frame, read_bounded_string,
};

// Zero/stale/wrong-kind packed handles fail before any context access.
fn context_handle(raw: EzGfxContext) -> Result<ContextHandle, EzGfxResult> {
    ContextHandle::from_raw(raw).map_err(|_| EzGfxResult::InvalidContext)
}

fn render_target_handle(raw: EzGfxRenderTarget) -> Result<RenderTargetHandle, EzGfxResult> {
    RenderTargetHandle::from_raw(raw).map_err(|_| EzGfxResult::InvalidContext)
}

// Runtime discriminants are the ABI codes; unknown bytes fail closed.
fn format_from_abi(value: u8) -> Result<Format, EzGfxResult> {
    match value {
        1 => Ok(Format::Rgba8Unorm),
        2 => Ok(Format::Bgra8Srgb),
        3 => Ok(Format::Rgba16Float),
        4 => Ok(Format::Depth32Float),
        5 => Ok(Format::Bc7Unorm),
        6 => Ok(Format::Astc4x4Unorm),
        _ => Err(EzGfxResult::InvalidArgument),
    }
}

fn usage_from_abi(value: u8) -> Result<TargetUsage, EzGfxResult> {
    match value {
        0 => Ok(TargetUsage::Color),
        1 => Ok(TargetUsage::Depth),
        2 => Ok(TargetUsage::Storage),
        3 => Ok(TargetUsage::Sampled),
        _ => Err(EzGfxResult::InvalidArgument),
    }
}

fn map_target_error(error: TargetError) -> EzGfxResult {
    match error {
        TargetError::UnsupportedFormat => EzGfxResult::Unsupported,
        TargetError::DuplicateSupport => EzGfxResult::NativeFailure,
        TargetError::InvalidName
        | TargetError::InvalidScale
        | TargetError::InvalidSamples
        | TargetError::NoCandidates
        | TargetError::DuplicateCandidate
        | TargetError::InvalidClear
        | TargetError::ClearTypeMismatch => EzGfxResult::InvalidArgument,
    }
}

#[unsafe(no_mangle)]
/// Creates a managed render target from a declaration and explicit extents.
///
/// Depth, storage, and sampled-only declarations stay unsupported with explicit
/// errors; multisampled color declarations allocate render storage that
/// resolves into the sampled image. The target leases one heap slot from the
///
/// # Safety
///
/// Non-null `desc` must address one readable, aligned descriptor, non-null
/// `out_target` one writable, aligned handle, and `desc.candidate_formats`
/// exactly `desc.candidate_count` readable format codes, for this call.
pub unsafe extern "C" fn ez_gfx_render_target_create(
    desc: *const EzGfxRenderTargetDesc,
    width: u32,
    height: u32,
    out_target: *mut EzGfxRenderTarget,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_target.is_null() || width == 0 || height == 0 {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `desc` is non-null and readable for this call.
        let desc = unsafe { desc.read() };
        if desc.sampleable > 1 || desc.use_clear > 1 {
            return EzGfxResult::InvalidArgument;
        }
        if desc.candidate_formats.is_null()
            || desc.candidate_count == 0
            || desc.candidate_count > 16
        {
            return EzGfxResult::InvalidArgument;
        }
        let Ok(name) = read_bounded_string(desc.name, desc.name_length) else {
            return EzGfxResult::InvalidArgument;
        };
        let Ok(usage) = usage_from_abi(desc.usage) else {
            return EzGfxResult::InvalidArgument;
        };
        // SAFETY: the candidate range is validated non-empty and bounded; the
        // caller keeps it readable and aligned for exactly this many bytes.
        let raw_candidates = unsafe {
            core::slice::from_raw_parts(desc.candidate_formats, desc.candidate_count as usize)
        };
        let mut candidates = Vec::with_capacity(raw_candidates.len());
        for code in raw_candidates {
            match format_from_abi(*code) {
                Ok(format) => candidates.push(format),
                Err(error) => return error,
            }
        }
        let clear = if desc.use_clear == 0 {
            ClearValue::None
        } else {
            if !desc.clear_color.iter().all(|value| value.is_finite()) {
                return EzGfxResult::InvalidArgument;
            }
            ClearValue::Color(desc.clear_color)
        };
        let declaration = match TargetDeclaration::new(
            name,
            usage,
            desc.relative_scale,
            desc.samples,
            candidates,
            clear,
            desc.sampleable != 0,
        ) {
            Ok(declaration) => declaration,
            Err(error) => return map_target_error(error),
        };
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        match raw::create_render_target(context, &declaration, width, height) {
            Ok(target) => {
                // SAFETY: `out_target` is non-null and writable for this call.
                unsafe { out_target.write(target.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Destroys a render target and clears any bound override; stale handles are ignored.
pub extern "C" fn ez_gfx_render_target_destroy(target: EzGfxRenderTarget, context: EzGfxContext) {
    catch_void(|| {
        if let (Ok(context), Ok(target)) = (context_handle(context), render_target_handle(target)) {
            raw::destroy_render_target(context, target);
        }
    });
}

#[unsafe(no_mangle)]
/// Reports the resolved storage format code of a live render target.
///
/// # Safety
///
/// A non-null `out_format` must address one writable, aligned `u8` for this call.
pub unsafe extern "C" fn ez_gfx_render_target_get_format(
    target: EzGfxRenderTarget,
    out_format: *mut u8,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_format.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let target = match render_target_handle(target) {
            Ok(target) => target,
            Err(error) => return error,
        };
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        match raw::render_target_format(context, target) {
            Ok(format) => {
                // SAFETY: `out_format` is non-null and writable for this call.
                unsafe { out_format.write(format as u8) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Reports the extents of a live render target.
///
/// # Safety
///
/// Each non-null output pointer must address one writable, aligned `u32` for this call.
pub unsafe extern "C" fn ez_gfx_render_target_get_extent(
    target: EzGfxRenderTarget,
    out_width: *mut u32,
    out_height: *mut u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_width.is_null() || out_height.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let target = match render_target_handle(target) {
            Ok(target) => target,
            Err(error) => return error,
        };
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        match raw::render_target_extent(context, target) {
            Ok((width, height)) => {
                // SAFETY: both outputs are non-null and writable for this call.
                unsafe {
                    out_width.write(width);
                    out_height.write(height);
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Reports the stored clear value of a live render target.
///
/// Writes one zero-or-one flag plus four color components (`[0, 0, 0, 0]`
/// when no clear is stored).
///
/// # Safety
///
/// Non-null `out_use_clear` must address one writable, aligned `u8` and
/// non-null `out_color` four writable, aligned `f32` values, for this call.
pub unsafe extern "C" fn ez_gfx_render_target_get_clear(
    target: EzGfxRenderTarget,
    out_use_clear: *mut u8,
    out_color: *mut f32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_use_clear.is_null() || out_color.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let target = match render_target_handle(target) {
            Ok(target) => target,
            Err(error) => return error,
        };
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        match raw::render_target_clear(context, target) {
            Ok(clear) => {
                let (flag, color) = match clear {
                    ClearValue::Color(values) => (1, values),
                    _ => (0, [0.0, 0.0, 0.0, 0.0]),
                };
                // SAFETY: both outputs are non-null and writable for their
                // declared lengths for this call.
                unsafe {
                    out_use_clear.write(flag);
                    core::slice::from_raw_parts_mut(out_color, 4).copy_from_slice(&color);
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Probes whether one format admits a sampled color target at the given sample count.
///
/// The status is the answer: `Ok` when resolvable, `Unsupported` otherwise.
/// Depth, storage, and above-ceiling sample requests stay unsupported with explicit errors.
///
/// # Safety
///
/// This function dereferences no pointers.
pub extern "C" fn ez_gfx_render_target_probe_format(
    format: u8,
    samples: u8,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let format = match format_from_abi(format) {
            Ok(format) => format,
            Err(error) => return error,
        };
        if !matches!(samples, 1 | 2 | 4 | 8) {
            return EzGfxResult::InvalidArgument;
        }
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        raw::probe_render_target_format(context, format, samples).into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Begins one render-target frame and returns its explicit owner handle.
///
/// # Safety
///
/// `out_frame` must address one writable, aligned handle for this call.
pub unsafe extern "C" fn ez_gfx_render_target_frame_begin(
    context: EzGfxContext,
    target: EzGfxRenderTarget,
    out_frame: *mut EzGfxFrame,
) -> EzGfxResult {
    catch_status(|| {
        if out_frame.is_null() || !out_frame.is_aligned() {
            return EzGfxResult::InvalidArgument;
        }
        let target = match render_target_handle(target) {
            Ok(target) => target,
            Err(error) => return error,
        };
        let context = match context_handle(context) {
            Ok(context) => context,
            Err(error) => return error,
        };
        if let Err(status) = raw::begin_render_target(context, target) {
            return status.into();
        }
        let serial = match raw::current_frame_serial(context) {
            Ok(serial) => serial,
            Err(status) => {
                let _ = raw::frame_abort(context);
                return status.into();
            }
        };
        let frame = match frame::insert(context, frame::FrameKind::RenderTarget, serial) {
            Ok(frame) => frame,
            Err(status) => {
                let _ = raw::frame_abort(context);
                return status;
            }
        };
        // SAFETY: `out_frame` was validated and remains caller-owned through this write.
        unsafe { out_frame.write(frame) };
        EzGfxResult::Ok
    })
}
