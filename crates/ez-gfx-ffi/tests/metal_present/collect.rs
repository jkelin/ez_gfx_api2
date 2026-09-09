//! Callback-scoped readback collection shared by the Metal presentation
//! suite: the libtest launcher's offscreen tests and the hidden main-thread
//! `metal_present` runner each collect presented bytes the same way.

use ez_gfx_ffi::{EzGfxEvent, EzGfxEventKind};

#[derive(Default)]
pub struct Collected {
    pub readback: Option<Vec<u8>>,
}

/// Copies callback-scoped readback bytes into caller-owned storage.
///
/// # Safety
///
/// `user_data` must point to a live `Collected` while registered.
pub unsafe extern "C" fn collect_event(
    event: *const EzGfxEvent,
    user_data: *mut core::ffi::c_void,
) {
    // SAFETY: registration keeps both pointers valid for this callback invocation.
    let (event, collected) = unsafe { (&*event, &mut *user_data.cast::<Collected>()) };
    if matches!(
        event.kind,
        EzGfxEventKind::Readback | EzGfxEventKind::Snapshot
    ) {
        let bytes = if event.readback_byte_count == 0 {
            Vec::new()
        } else {
            // SAFETY: nonempty readback bytes are callback-scoped and copied before returning.
            unsafe {
                core::slice::from_raw_parts(event.readback_bytes, event.readback_byte_count)
                    .to_vec()
            }
        };
        collected.readback = Some(bytes);
    }
}
