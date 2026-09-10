//! Resource-diagnostics C ABI layout and boundary contracts.

use core::mem::{align_of, offset_of, size_of};

use ez_gfx_ffi::{EzGfxResourceDiagnostics, EzGfxResult, ez_gfx_context_get_resource_diagnostics};

#[test]
fn layouts_and_export_signature_are_stable() {
    assert_eq!(
        (
            size_of::<EzGfxResourceDiagnostics>(),
            align_of::<EzGfxResourceDiagnostics>()
        ),
        (80, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxResourceDiagnostics, pending_textures),
            offset_of!(EzGfxResourceDiagnostics, pending_texture_bytes),
            offset_of!(EzGfxResourceDiagnostics, pending_vertex_uploads),
            offset_of!(EzGfxResourceDiagnostics, pending_vertex_bytes),
            offset_of!(EzGfxResourceDiagnostics, pending_index_uploads),
            offset_of!(EzGfxResourceDiagnostics, pending_index_bytes),
            offset_of!(EzGfxResourceDiagnostics, staging_buckets),
            offset_of!(EzGfxResourceDiagnostics, staging_bytes),
            offset_of!(EzGfxResourceDiagnostics, pipeline_entries),
            offset_of!(EzGfxResourceDiagnostics, readback_bytes),
        ],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
    );
    let _: unsafe extern "C" fn(u64, *mut EzGfxResourceDiagnostics) -> EzGfxResult =
        ez_gfx_context_get_resource_diagnostics;
}

#[test]
fn null_or_zero_handles_fail_before_state_access() {
    assert_eq!(
        // SAFETY: Null output intentionally exercises checked boundary rejection.
        unsafe { ez_gfx_context_get_resource_diagnostics(0, core::ptr::null_mut()) },
        EzGfxResult::InvalidArgument
    );
    let mut diagnostics = core::mem::MaybeUninit::<EzGfxResourceDiagnostics>::uninit();
    assert_eq!(
        // SAFETY: Output storage is writable; the zero context fails as stale.
        unsafe { ez_gfx_context_get_resource_diagnostics(0, diagnostics.as_mut_ptr()) },
        EzGfxResult::InvalidContext
    );
}
