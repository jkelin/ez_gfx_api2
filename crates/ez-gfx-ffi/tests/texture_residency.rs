//! Progressive native texture-residency tests through the C ABI.
#![cfg(windows)]

use std::ffi::CString;

use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxResult, EzGfxTextureDesc, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_wait_idle, ez_gfx_texture_get_residency,
    ez_gfx_texture_load, ez_gfx_texture_unload,
};

#[cfg(windows)]
#[test]
fn vulkan_reports_completed_progressive_mip_residency() {
    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend: 1,
    };
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const context_desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );

    let bytes = [128_u8; 4 * 4 * 4];
    let label = CString::new("residency-test").unwrap();
    let desc = EzGfxTextureDesc {
        source_format: 1,
        destination_format: 0,
        width: 4,
        height: 4,
        mip_count: 0,
        generate_mips: 1,
        min_filter: 1,
        mag_filter: 1,
        max_anisotropy: 1.0,
        address_mode_u: 0,
        address_mode_v: 0,
        address_mode_w: 0,
        debug_label: label.as_ptr(),
    };
    let mut texture = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_texture_load(
                    bytes.as_ptr(),
                    bytes.len(),
                    &raw const desc,
                    &raw mut texture,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );

    let mut resident = 0;
    let mut total = 0;
    let first = {
        // SAFETY: Non-null outputs point to live, aligned u32 storage; null pointers intentionally exercise checked rejection.
        unsafe { ez_gfx_texture_get_residency(texture, &raw mut resident, &raw mut total, context) }
    };
    assert!(matches!(first, EzGfxResult::Ok | EzGfxResult::NotReady));
    assert!(resident <= total);

    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_texture_get_residency(texture, &raw mut resident, &raw mut total, context)
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(resident, total);
    assert_eq!(total, 3);
    ez_gfx_texture_unload(texture, context);
    ez_gfx_context_destroy(context);
}
