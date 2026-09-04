use crate::{EZ_GFX_MAX_BOUNDARY_BYTES, EzGfxResult};

fn with_bounded_string<T>(
    pointer: *const u8,
    length: usize,
    use_string: impl FnOnce(&str) -> T,
) -> Result<T, EzGfxResult> {
    // Edge cases fail before dereference: null/empty, over-cap/isize ranges, invalid UTF-8,
    // and embedded NUL cannot cross the ABI as a different semantic value.
    if pointer.is_null()
        || length == 0
        || length > EZ_GFX_MAX_BOUNDARY_BYTES
        || length > isize::MAX as usize
    {
        return Err(EzGfxResult::InvalidArgument);
    }
    // SAFETY: `length` is nonzero and bounded by both `isize::MAX` and the ABI cap; the caller guarantees exactly this alignment-1 range is readable for the call.
    let bytes = unsafe { core::slice::from_raw_parts(pointer, length) };
    if bytes.contains(&0) {
        return Err(EzGfxResult::InvalidArgument);
    }
    core::str::from_utf8(bytes)
        .map(use_string)
        .map_err(|_| EzGfxResult::InvalidArgument)
}

pub(crate) fn read_bounded_string(
    pointer: *const u8,
    length: usize,
) -> Result<String, EzGfxResult> {
    with_bounded_string(pointer, length, str::to_owned)
}

pub(crate) fn validate_bounded_string(
    pointer: *const u8,
    length: usize,
) -> Result<(), EzGfxResult> {
    with_bounded_string(pointer, length, |_| ())
}

pub(crate) fn validate_optional_bounded_string(
    pointer: *const u8,
    length: usize,
) -> Result<(), EzGfxResult> {
    // The sole empty optional representation is null+zero; every other pair is required-form.
    if pointer.is_null() && length == 0 {
        return Ok(());
    }
    validate_bounded_string(pointer, length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_string_accepts_exact_non_terminated_utf8_range() {
        let bytes = b"heap";

        assert_eq!(
            read_bounded_string(bytes.as_ptr(), bytes.len()),
            Ok("heap".to_owned())
        );
    }

    #[test]
    fn bounded_string_rejects_invalid_ranges_and_contents() {
        let byte = b"x";
        let invalid_utf8 = [0xff_u8];
        let embedded_nul = b"a\0b";

        for (pointer, length) in [
            (core::ptr::null(), 1),
            (byte.as_ptr(), 0),
            (byte.as_ptr(), EZ_GFX_MAX_BOUNDARY_BYTES.saturating_add(1)),
            (invalid_utf8.as_ptr(), invalid_utf8.len()),
            (embedded_nul.as_ptr(), embedded_nul.len()),
        ] {
            assert_eq!(
                read_bounded_string(pointer, length),
                Err(EzGfxResult::InvalidArgument)
            );
        }
    }
}
