//! Teardown completion contracts for the ABI 40 destroy exports.
//!
//! Abandonment itself is not injectable here: it requires submitted native
//! work that survives device loss, which no validation-level failure can
//! forge. These tests pin every injectable outcome (drained `Ok`, null,
//! stale, and wrong-thread) so only genuine abandonment reports
//! `TeardownAbandoned`.

#[cfg(windows)]
mod common;
#[cfg(not(any(windows, target_vendor = "apple")))]
#[path = "common/headless.rs"]
mod common;

use ez_gfx_ffi::{EzGfxResult, ez_gfx_context_destroy, ez_gfx_surface_destroy};

#[test]
fn destroy_rejects_null_handles_before_native_calls() {
    assert_eq!(ez_gfx_surface_destroy(0, 0), EzGfxResult::InvalidContext);
    assert_eq!(ez_gfx_context_destroy(0), EzGfxResult::InvalidContext);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn destroy_reports_ok_and_rejects_stale_handles() {
    let mut native = common::TestContext::create_with_validation(1, false);
    assert_eq!(
        ez_gfx_surface_destroy(native.context, native.surface),
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_surface_destroy(native.context, native.surface),
        EzGfxResult::InvalidContext
    );
    assert_eq!(ez_gfx_context_destroy(native.context), EzGfxResult::Ok);
    assert_eq!(
        ez_gfx_context_destroy(native.context),
        EzGfxResult::InvalidContext
    );
    native.context = 0;
    native.surface = 0;
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn destroy_rejects_wrong_thread_handles() {
    let native = common::TestContext::create_with_validation(1, false);
    let (context, surface) = (native.context, native.surface);
    let foreign = std::thread::spawn(move || {
        (
            ez_gfx_surface_destroy(context, surface),
            ez_gfx_context_destroy(context),
        )
    })
    .join()
    .expect("foreign destroy thread returns");
    assert_eq!(foreign.0, EzGfxResult::InvalidContext);
    assert_eq!(foreign.1, EzGfxResult::InvalidContext);
}
