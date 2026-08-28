use std::ffi::CString;

use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxResult, EzGfxTextureDesc, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_wait_idle, ez_gfx_texture_get_residency,
    ez_gfx_texture_load, ez_gfx_texture_unload,
};

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
        ez_gfx_context_create_backend(&context_desc, &mut context),
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
        ez_gfx_texture_load(bytes.as_ptr(), bytes.len(), &desc, &mut texture, context),
        EzGfxResult::Ok
    );

    let mut resident = 0;
    let mut total = 0;
    let first = ez_gfx_texture_get_residency(texture, &mut resident, &mut total, context);
    assert!(matches!(first, EzGfxResult::Ok | EzGfxResult::NotReady));
    assert!(resident <= total);

    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    assert_eq!(
        ez_gfx_texture_get_residency(texture, &mut resident, &mut total, context),
        EzGfxResult::Ok
    );
    assert_eq!(resident, total);
    assert_eq!(total, 3);
    ez_gfx_texture_unload(texture, context);
    ez_gfx_context_destroy(context);
}
