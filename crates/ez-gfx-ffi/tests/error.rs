//! Stable C ABI result-code and diagnostic-string contracts.

use ez_gfx_ffi::{
    EZ_GFX_ABI_VERSION, EzGfxResult, EzGfxTextureError, ez_gfx_abi_version, ez_gfx_error_print,
};

#[test]
fn status_values_and_abi_version_are_stable() {
    assert_eq!(ez_gfx_abi_version(), EZ_GFX_ABI_VERSION);
    assert_eq!(EzGfxResult::Ok as u8, 0);
    assert_eq!(EzGfxResult::InvalidArgument as u8, 1);
    assert_eq!(EzGfxResult::InvalidContext as u8, 2);
    assert_eq!(EzGfxResult::NativeFailure as u8, 3);
    assert_eq!(EzGfxResult::NotReady as u8, 4);
    assert_eq!(EzGfxResult::Unsupported as u8, 5);
    assert_eq!(EzGfxResult::DeviceLost as u8, 6);
    assert_eq!(EzGfxResult::QueueFull as u8, 7);
    assert_eq!(EzGfxResult::Cancelled as u8, 8);
    assert_eq!(EzGfxResult::TeardownAbandoned as u8, 9);
    assert_eq!(
        [
            EzGfxTextureError::None as u8,
            EzGfxTextureError::InvalidContext as u8,
            EzGfxTextureError::InvalidArguments as u8,
            EzGfxTextureError::UnsupportedFormat as u8,
            EzGfxTextureError::OutOfTextureHandles as u8,
            EzGfxTextureError::OutOfMemory as u8,
            EzGfxTextureError::DecodeFailed as u8,
            EzGfxTextureError::VulkanFailed as u8,
            EzGfxTextureError::WorkerUnavailable as u8,
            EzGfxTextureError::NotFound as u8,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(EZ_GFX_ABI_VERSION, 41);
}

#[test]
fn error_printer_validates_and_reports_required_capacity() {
    let mut required = 0;
    // SAFETY: Null+zero is the documented size-query form and `required` is writable.
    assert_eq!(
        // SAFETY: The test provides the documented pointer ranges.
        unsafe {
            ez_gfx_error_print(
                EzGfxResult::DeviceLost as u8,
                core::ptr::null_mut(),
                0,
                &raw mut required,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(required, "graphics device lost".len() + 1);

    let mut exact = vec![0_u8; required];
    // SAFETY: `exact` and `required` remain writable for the call.
    assert_eq!(
        // SAFETY: The test provides the documented pointer ranges.
        unsafe {
            ez_gfx_error_print(
                EzGfxResult::DeviceLost as u8,
                exact.as_mut_ptr(),
                exact.len(),
                &raw mut required,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(&exact[..exact.len() - 1], b"graphics device lost");
    assert_eq!(exact[exact.len() - 1], 0);

    let mut truncated = vec![0xaa_u8; required - 1];
    // SAFETY: `truncated` is writable for exactly its reported capacity.
    assert_eq!(
        // SAFETY: The test provides the documented pointer ranges.
        unsafe {
            ez_gfx_error_print(
                EzGfxResult::DeviceLost as u8,
                truncated.as_mut_ptr(),
                truncated.len(),
                &raw mut required,
            )
        },
        EzGfxResult::InvalidArgument
    );
    assert!(truncated.iter().all(|byte| *byte == 0xaa));

    // SAFETY: Invalid pointer/count pairs intentionally exercise checked rejection.
    assert_eq!(
        // SAFETY: The test provides the documented pointer ranges.
        unsafe {
            ez_gfx_error_print(
                EzGfxResult::DeviceLost as u8,
                core::ptr::null_mut(),
                1,
                &raw mut required,
            )
        },
        EzGfxResult::InvalidArgument
    );
    // SAFETY: Null output intentionally exercises checked rejection.
    assert_eq!(
        // SAFETY: The test provides the documented pointer ranges.
        unsafe { ez_gfx_error_print(255, core::ptr::null_mut(), 0, core::ptr::null_mut()) },
        EzGfxResult::InvalidArgument
    );

    let mut unknown_required = 0;
    // SAFETY: Null+zero is the documented size-query form.
    assert_eq!(
        // SAFETY: The test provides the documented pointer ranges.
        unsafe { ez_gfx_error_print(255, core::ptr::null_mut(), 0, &raw mut unknown_required,) },
        EzGfxResult::Ok
    );
    let mut unknown = vec![0_u8; unknown_required];
    // SAFETY: `unknown` and its size output are writable.
    assert_eq!(
        // SAFETY: The test provides the documented pointer ranges.
        unsafe {
            ez_gfx_error_print(
                255,
                unknown.as_mut_ptr(),
                unknown.len(),
                &raw mut unknown_required,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(&unknown, b"unknown error\0");
}

#[test]
fn error_printer_reports_teardown_abandonment_bytes() {
    const MESSAGE: &[u8] = b"native teardown abandoned; borrowed host handles must remain alive";
    let mut required = 0;
    assert_eq!(
        // SAFETY: Null+zero is the documented size-query form and `required` is writable.
        unsafe {
            ez_gfx_error_print(
                EzGfxResult::TeardownAbandoned as u8,
                core::ptr::null_mut(),
                0,
                &raw mut required,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(required, MESSAGE.len() + 1);

    let mut exact = vec![0_u8; required];
    assert_eq!(
        // SAFETY: `exact` and `required` remain writable for the call.
        unsafe {
            ez_gfx_error_print(
                EzGfxResult::TeardownAbandoned as u8,
                exact.as_mut_ptr(),
                exact.len(),
                &raw mut required,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(&exact[..exact.len() - 1], MESSAGE);
    assert_eq!(exact[exact.len() - 1], 0);
}
