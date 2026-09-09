use super::{EZ_GFX_MAX_BOUNDARY_BYTES, EzGfxResult, catch_status};

fn error_message(result: u8) -> &'static [u8] {
    match result {
        value if value == EzGfxResult::Ok as u8 => b"ok",
        value if value == EzGfxResult::InvalidArgument as u8 => b"invalid argument",
        value if value == EzGfxResult::InvalidContext as u8 => {
            b"invalid or stale context/resource handle"
        }
        value if value == EzGfxResult::NativeFailure as u8 => b"native graphics backend failure",
        value if value == EzGfxResult::NotReady as u8 => b"operation is not ready",
        value if value == EzGfxResult::Unsupported as u8 => b"unsupported operation or capability",
        value if value == EzGfxResult::DeviceLost as u8 => b"graphics device lost",
        value if value == EzGfxResult::QueueFull as u8 => {
            b"asynchronous scheduling capacity unavailable"
        }
        value if value == EzGfxResult::Cancelled as u8 => b"asynchronous operation cancelled",
        value if value == EzGfxResult::TeardownAbandoned as u8 => {
            b"native teardown abandoned; borrowed host handles must remain alive"
        }
        _ => b"unknown error",
    }
}

#[unsafe(no_mangle)]
/// Writes the stable UTF-8 message for an ABI result code.
///
/// `out_required` receives the byte count including the trailing NUL. Pass a
/// null buffer with zero capacity to query that count. Insufficient capacity
/// returns `InvalidArgument` without modifying the buffer.
///
/// # Safety
///
/// `out_required` must address one writable, aligned `usize`. A nonzero
/// `capacity` requires `buffer` to address that many writable bytes.
pub unsafe extern "C" fn ez_gfx_error_print(
    result: u8,
    buffer: *mut u8,
    capacity: usize,
    out_required: *mut usize,
) -> EzGfxResult {
    catch_status(|| {
        if out_required.is_null()
            || capacity > EZ_GFX_MAX_BOUNDARY_BYTES
            || capacity > isize::MAX as usize
            || (capacity == 0) != buffer.is_null()
        {
            return EzGfxResult::InvalidArgument;
        }

        let message = error_message(result);
        let required = message.len() + 1;
        // SAFETY: `out_required` is validated non-null and the caller keeps one aligned
        // writable `usize` alive through this call.
        unsafe { out_required.write(required) };
        if capacity == 0 {
            return EzGfxResult::Ok;
        }
        if capacity < required {
            return EzGfxResult::InvalidArgument;
        }

        // SAFETY: `buffer` is non-null and writable for at least `required` bytes;
        // the static message is disjoint and the final byte is reserved for NUL.
        unsafe {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer, message.len());
            buffer.add(message.len()).write(0);
        }
        EzGfxResult::Ok
    })
}
