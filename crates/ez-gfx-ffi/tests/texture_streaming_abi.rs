//! Texture streaming C ABI layout and boundary contracts.

use core::mem::{align_of, offset_of, size_of};

use ez_gfx_ffi::{
    EzGfxResult, EzGfxTextureRegionDesc, EzGfxTextureUploadTelemetry,
    ez_gfx_texture_get_upload_telemetry, ez_gfx_update_texture_region,
};

#[test]
fn layouts_and_export_signatures_are_stable() {
    assert_eq!(
        (
            size_of::<EzGfxTextureRegionDesc>(),
            align_of::<EzGfxTextureRegionDesc>()
        ),
        (40, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxTextureRegionDesc, mip_level),
            offset_of!(EzGfxTextureRegionDesc, x),
            offset_of!(EzGfxTextureRegionDesc, y),
            offset_of!(EzGfxTextureRegionDesc, width),
            offset_of!(EzGfxTextureRegionDesc, height),
            offset_of!(EzGfxTextureRegionDesc, data),
            offset_of!(EzGfxTextureRegionDesc, data_size),
        ],
        [0, 4, 8, 12, 16, 24, 32]
    );
    assert_eq!(
        (
            size_of::<EzGfxTextureUploadTelemetry>(),
            align_of::<EzGfxTextureUploadTelemetry>()
        ),
        (32, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxTextureUploadTelemetry, decode_microseconds),
            offset_of!(EzGfxTextureUploadTelemetry, staging_bytes),
            offset_of!(EzGfxTextureUploadTelemetry, queue_latency_microseconds),
            offset_of!(EzGfxTextureUploadTelemetry, handoff_latency_microseconds),
        ],
        [0, 8, 16, 24]
    );
    let _: unsafe extern "C" fn(u64, *const EzGfxTextureRegionDesc, u64) -> EzGfxResult =
        ez_gfx_update_texture_region;
    let _: unsafe extern "C" fn(*mut EzGfxTextureUploadTelemetry, u64) -> EzGfxResult =
        ez_gfx_texture_get_upload_telemetry;
}

#[test]
fn null_streaming_descriptors_fail_before_context_access() {
    assert_eq!(
        // SAFETY: Null descriptor intentionally exercises checked boundary rejection.
        unsafe { ez_gfx_update_texture_region(1, core::ptr::null(), 0) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null output intentionally exercises checked boundary rejection.
        unsafe { ez_gfx_texture_get_upload_telemetry(core::ptr::null_mut(), 0) },
        EzGfxResult::InvalidArgument
    );
}

#[test]
fn dds_and_raw_source_codes_pass_mapping_before_context_access() {
    use ez_gfx_ffi::{EzGfxTexture, EzGfxTextureDesc, ez_gfx_texture_load};

    fn desc(source: u8, destination: u8) -> EzGfxTextureDesc {
        EzGfxTextureDesc {
            source_format: source,
            destination_format: destination,
            width: 4,
            height: 4,
            mip_count: 1,
            generate_mips: 0,
            min_filter: 1,
            mag_filter: 1,
            max_anisotropy: 1.0,
            address_mode_u: 1,
            address_mode_v: 1,
            address_mode_w: 1,
            debug_label: core::ptr::null(),
            debug_label_length: 0,
        }
    }

    // A null context fails only after source mapping succeeds; unmapped codes
    // fail closed with InvalidArgument before any context access.
    static DATA: [u8; 16] = [0; 16];
    for (source, destination, expected) in [
        (6_u8, 1_u8, EzGfxResult::InvalidContext),
        (8_u8, 1_u8, EzGfxResult::InvalidContext),
        (9_u8, 7_u8, EzGfxResult::InvalidContext),
        (9_u8, 1_u8, EzGfxResult::InvalidArgument),
        (10_u8, 1_u8, EzGfxResult::InvalidArgument),
    ] {
        let desc = desc(source, destination);
        let mut texture: EzGfxTexture = 0;
        // SAFETY: The static input stays readable; the descriptor and output
        // stay valid through this boundary-rejection call.
        let status = unsafe {
            ez_gfx_texture_load(
                DATA.as_ptr(),
                DATA.len(),
                &raw const desc,
                &raw mut texture,
                0,
            )
        };
        assert_eq!(
            status, expected,
            "source {source} destination {destination}"
        );
        assert_eq!(texture, 0);
    }
}
