//! Texture streaming C ABI layout and boundary contracts.

use core::mem::{align_of, offset_of, size_of};

#[cfg(windows)]
use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxTextureDesc, EzGfxUploadEvent, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_poll_upload_event, ez_gfx_texture_get_binding,
    ez_gfx_texture_load, ez_gfx_texture_unload,
};
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
#[cfg(windows)]
#[test]
fn context_decode_workers_flow_from_c_descriptor_to_creation() {
    // Null pairs reject before any worker-count handling.
    let mut context = 99;
    assert_eq!(
        {
            // SAFETY: Null descriptor intentionally exercises checked rejection.
            unsafe { ez_gfx_context_create_backend(core::ptr::null(), &raw mut context) }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(context, 99);

    // Default topology (0) and an explicit count both create working DX12 contexts:
    // texture admission exercises the decode pool the count sizes. Surfaceless
    // Vulkan upload is outside the proven contract (Vulkan flows init a device
    // through a surface), so Vulkan only pins descriptor parsing and creation.
    let pixels = [1_u8, 2, 3, 4];
    let texture_desc = EzGfxTextureDesc {
        source_format: 1,
        destination_format: 0,
        width: 1,
        height: 1,
        mip_count: 1,
        generate_mips: 0,
        min_filter: 0,
        mag_filter: 0,
        max_anisotropy: 1.0,
        address_mode_u: 0,
        address_mode_v: 0,
        address_mode_w: 0,
        debug_label: core::ptr::null(),
        debug_label_length: 0,
    };
    for workers in [0, 2] {
        let context_desc = EzGfxBackendContextDesc {
            enable_debug: 0,
            enable_validation: 0,
            surface_platform: 0,
            backend: 2,
            texture_decode_workers: workers,
            adapter_count: 0,
            adapter: core::ptr::null(),
        };
        let mut context = 0;
        let mut texture = 0;
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access.
                unsafe { ez_gfx_context_create_backend(&raw const context_desc, &raw mut context) }
            },
            EzGfxResult::Ok,
            "workers={workers}"
        );
        // The Rust-side read-back proves the C count arrived: default resolves to
        // the creator topology, an explicit count pins the pool exactly.
        let expected = if workers == 0 {
            u32::try_from(
                std::thread::available_parallelism()
                    .map_or(2, usize::from)
                    .saturating_sub(1)
                    .max(1),
            )
            .unwrap()
        } else {
            workers
        };
        let handle = ez_gfx::raw::ContextHandle::from_raw(context).unwrap();
        assert_eq!(
            ez_gfx::raw::texture_decode_worker_count(handle).unwrap(),
            expected,
            "workers={workers}"
        );
        // No oversized-count case: rayon grinds instead of failing fast on huge
        // pools, which stalls the runner rather than proving rejection.
        assert_eq!(
            {
                // SAFETY: Pixel, descriptor, and output storage stay live through the admitted load.
                unsafe {
                    ez_gfx_texture_load(
                        pixels.as_ptr(),
                        pixels.len(),
                        &raw const texture_desc,
                        &raw mut texture,
                        context,
                    )
                }
            },
            EzGfxResult::Ok,
            "workers={workers}"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            // SAFETY: all-zero is the documented initialization for this plain C record.
            let mut event = unsafe { core::mem::zeroed::<EzGfxUploadEvent>() };
            let mut present = 0;
            assert_eq!(
                // SAFETY: event and presence outputs remain writable for this call.
                unsafe { ez_gfx_poll_upload_event(&raw mut event, &raw mut present, context) },
                EzGfxResult::Ok
            );
            let mut binding = 0;
            let status =
                // SAFETY: binding remains writable and both handles are live.
                unsafe { ez_gfx_texture_get_binding(texture, &raw mut binding, context) };
            assert_ne!(status, EzGfxResult::InvalidArgument, "workers={workers}");
            if status != EzGfxResult::NotReady || std::time::Instant::now() >= deadline {
                assert_eq!(status, EzGfxResult::Ok, "workers={workers}");
                break;
            }
        }
        ez_gfx_texture_unload(texture, context);
        ez_gfx_context_destroy(context);
    }

    // The Vulkan descriptor path parses the same trailing field.
    let vulkan_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend: 1,
        texture_decode_workers: 2,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
    let mut vulkan = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access.
            unsafe { ez_gfx_context_create_backend(&raw const vulkan_desc, &raw mut vulkan) }
        },
        EzGfxResult::Ok
    );
    ez_gfx_context_destroy(vulkan);
}
