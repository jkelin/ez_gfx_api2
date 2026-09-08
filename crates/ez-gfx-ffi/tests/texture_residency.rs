//! Progressive native texture-residency tests through the C ABI.
#![cfg(not(target_vendor = "apple"))]

#[cfg(windows)]
mod common;
#[cfg(not(any(windows, target_vendor = "apple")))]
#[path = "common/headless.rs"]
mod common;

use common::TestContext;
#[cfg(all(feature = "ktx2", feature = "basis"))]
use ez_gfx_core::capability::CompressionSupport;
#[cfg(all(feature = "ktx2", feature = "basis"))]
use ez_gfx_runtime::texture::{TextureDecoder, TextureSource};

use ez_gfx_ffi::{
    EzGfxResult, EzGfxTextureDesc, EzGfxTextureRegionDesc, EzGfxTextureUploadTelemetry,
    EzGfxUploadEvent, ez_gfx_poll_upload_event, ez_gfx_texture_cancel, ez_gfx_texture_get_binding,
    ez_gfx_texture_get_residency, ez_gfx_texture_get_upload_telemetry, ez_gfx_texture_load,
    ez_gfx_texture_set_residency, ez_gfx_texture_unload, ez_gfx_update_texture_region,
};
#[cfg(all(feature = "ktx2", feature = "basis"))]
use ez_gfx_ffi::{
    ez_gfx_frame_begin, ez_gfx_frame_end, ez_gfx_frame_readback,
    ez_gfx_graph_enqueue_texture_readback,
};
fn poll_texture_ready(context: u64, texture: u64) -> EzGfxResult {
    // SAFETY: all-zero is the documented initialization for this plain C record.
    let mut event = unsafe { core::mem::zeroed::<EzGfxUploadEvent>() };
    let mut present = 0;
    let progress =
        // SAFETY: event and presence outputs remain writable for this call.
        unsafe { ez_gfx_poll_upload_event(&raw mut event, &raw mut present, context) };
    if progress != EzGfxResult::Ok {
        return progress;
    }
    let mut binding = 0;
    // SAFETY: binding remains writable and both handles are supplied by this test.
    unsafe { ez_gfx_texture_get_binding(texture, &raw mut binding, context) }
}

fn cancel_after_native_admission(context: u64, bytes: &[u8], desc: &EzGfxTextureDesc) {
    let mut texture = 0;
    assert_eq!(
        // SAFETY: all byte, descriptor, and output storage remains live through this call.
        unsafe {
            ez_gfx_texture_load(bytes.as_ptr(), bytes.len(), desc, &raw mut texture, context)
        },
        EzGfxResult::Ok
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let mut resident = 0;
        let mut total = 0;
        let status = {
            // SAFETY: both outputs remain live writable u32 storage.
            unsafe {
                ez_gfx_texture_get_residency(texture, &raw mut resident, &raw mut total, context)
            }
        };
        if status == EzGfxResult::Ok {
            break;
        }
        assert_eq!(status, EzGfxResult::NotReady);
        assert!(
            std::time::Instant::now() < deadline,
            "native texture admission timed out"
        );
        std::thread::yield_now();
    }
    assert_eq!(ez_gfx_texture_cancel(texture, context), EzGfxResult::Ok);
    assert_eq!(
        poll_texture_ready(context, texture),
        EzGfxResult::InvalidContext
    );
}

#[cfg(not(target_vendor = "apple"))]
#[expect(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "one hardware scenario keeps batch, update, compressed upload, and telemetry lifetime ordered"
)]
fn exercises_async_texture_batches(backend: u8) {
    let native = TestContext::create(backend);
    let context = native.context;
    let bytes = [128_u8; 4 * 4 * 4];
    let label = b"residency-test";
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
        debug_label_length: label.len(),
    };

    for wave in 0_u8..4 {
        let mut textures = [0_u64; 8];
        for texture in &mut textures {
            assert_eq!(
                {
                    // SAFETY: all byte, descriptor, and output storage remains live through this call.
                    unsafe {
                        ez_gfx_texture_load(
                            bytes.as_ptr(),
                            bytes.len(),
                            &raw const desc,
                            texture,
                            context,
                        )
                    }
                },
                EzGfxResult::Ok,
                "wave {wave}"
            );
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let mut ready = true;
            for &texture in &textures {
                match poll_texture_ready(context, texture) {
                    EzGfxResult::Ok => {}
                    EzGfxResult::NotReady => ready = false,
                    error => panic!("wave {wave} texture failed: {error:?}"),
                }
            }
            if ready {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "wave {wave} timed out"
            );
            std::thread::yield_now();
        }

        let update = [wave, 1, 2, 255];
        let region = EzGfxTextureRegionDesc {
            mip_level: 0,
            x: 1,
            y: 1,
            width: 1,
            height: 1,
            data: update.as_ptr(),
            data_size: update.len(),
        };
        assert_eq!(
            // SAFETY: The region descriptor and bytes remain live through admission.
            unsafe { ez_gfx_update_texture_region(textures[0], &raw const region, context,) },
            EzGfxResult::Ok
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match poll_texture_ready(context, textures[0]) {
                EzGfxResult::Ok => break,
                EzGfxResult::NotReady => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "region update timed out"
                    );
                    std::thread::yield_now();
                }
                error => panic!("region update failed: {error:?}"),
            }
        }

        for texture in textures {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let (mut resident, mut total) = loop {
                match poll_texture_ready(context, texture) {
                    EzGfxResult::Ok | EzGfxResult::NotReady => {}
                    error => panic!("updated texture failed: {error:?}"),
                }
                let mut resident = 0;
                let mut total = 0;
                assert_eq!(
                    {
                        // SAFETY: both outputs are live writable u32 storage and the handles remain live.
                        unsafe {
                            ez_gfx_texture_get_residency(
                                texture,
                                &raw mut resident,
                                &raw mut total,
                                context,
                            )
                        }
                    },
                    EzGfxResult::Ok
                );
                if (resident, total) == (3, 3) {
                    break (resident, total);
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "updated texture did not restore full residency: {resident}/{total}"
                );
                std::thread::yield_now();
            };
            assert_eq!(
                ez_gfx_texture_set_residency(texture, 0, context),
                EzGfxResult::InvalidArgument
            );
            assert_eq!(
                ez_gfx_texture_set_residency(texture, total + 1, context),
                EzGfxResult::InvalidArgument
            );
            assert_eq!(
                ez_gfx_texture_set_residency(texture, 1, context),
                EzGfxResult::Ok
            );
            assert_eq!(
                {
                    // SAFETY: both outputs remain live writable u32 storage.
                    unsafe {
                        ez_gfx_texture_get_residency(
                            texture,
                            &raw mut resident,
                            &raw mut total,
                            context,
                        )
                    }
                },
                EzGfxResult::Ok
            );
            assert_eq!((resident, total), (1, 3));
            assert_eq!(
                ez_gfx_texture_set_residency(texture, total, context),
                EzGfxResult::Ok
            );
            let mut binding = u32::MAX;
            assert_eq!(
                {
                    // SAFETY: `binding` is live writable u32 storage and the handles remain live.
                    unsafe { ez_gfx_texture_get_binding(texture, &raw mut binding, context) }
                },
                EzGfxResult::Ok
            );
            assert_ne!(binding, u32::MAX);
            ez_gfx_texture_unload(texture, context);
        }
    }

    cancel_after_native_admission(context, &bytes, &desc);
    #[cfg(all(feature = "ktx2", feature = "basis"))]
    {
        let basis = include_bytes!("../../ez-gfx-runtime/tests/fixtures/alpha_simple_basis.ktx2");
        let decoded =
            TextureDecoder::decode_with_support(TextureSource::Ktx2, basis, CompressionSupport::BC)
                .unwrap();
        assert!(decoded.format.is_compressed());
        let compressed_desc = EzGfxTextureDesc {
            source_format: 6,
            destination_format: 1,
            width: 0,
            height: 0,
            mip_count: 0,
            generate_mips: 0,
            ..desc
        };
        let mut compressed = 0;
        assert_eq!(
            // SAFETY: Fixture, descriptor, and output storage remain live through this call.
            unsafe {
                ez_gfx_texture_load(
                    basis.as_ptr(),
                    basis.len(),
                    &raw const compressed_desc,
                    &raw mut compressed,
                    context,
                )
            },
            EzGfxResult::Ok
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match poll_texture_ready(context, compressed) {
                EzGfxResult::Ok => break,
                EzGfxResult::NotReady => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "Basis texture timed out"
                    );
                    std::thread::yield_now();
                }
                error => panic!("Basis texture failed: {error:?}"),
            }
        }
        let mut resident = 0;
        let mut total = 0;
        assert_eq!(
            // SAFETY: Outputs and handles remain live through this call.
            unsafe {
                ez_gfx_texture_get_residency(compressed, &raw mut resident, &raw mut total, context)
            },
            EzGfxResult::Ok
        );
        assert_eq!((resident, total), (1, 1));
        let base = &decoded.mips[0];
        let region = EzGfxTextureRegionDesc {
            mip_level: 0,
            x: 0,
            y: 0,
            width: base.width,
            height: base.height,
            data: base.bytes.as_ptr(),
            data_size: base.bytes.len(),
        };
        assert_eq!(
            // SAFETY: The descriptor and compressed block bytes remain live through admission.
            unsafe { ez_gfx_update_texture_region(compressed, &raw const region, context) },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_ffi::ez_gfx_context_wait_idle(context),
            EzGfxResult::Ok
        );
        let mut frame = 0;
        assert_eq!(
            // SAFETY: frame output storage is live and aligned.
            unsafe { ez_gfx_frame_begin(context, native.surface, &raw mut frame) },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_graph_enqueue_texture_readback(compressed, frame),
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_frame_end(frame),
            if cfg!(windows) {
                EzGfxResult::Ok
            } else {
                EzGfxResult::Unsupported
            }
        );
        let mut size = 0;
        assert_eq!(
            // SAFETY: The size output remains live and writable through the query.
            unsafe { ez_gfx_frame_readback(core::ptr::null_mut(), 0, &raw mut size, context) },
            EzGfxResult::Ok
        );
        let mut actual = vec![0; size];
        assert_eq!(
            // SAFETY: `actual` exposes exactly its writable initialized allocation.
            unsafe {
                ez_gfx_frame_readback(actual.as_mut_ptr(), actual.len(), &raw mut size, context)
            },
            EzGfxResult::Ok
        );
        assert_eq!(&actual[..base.bytes.len()], base.bytes);
        assert!(actual[base.bytes.len()..].iter().all(|byte| *byte == 0));
        ez_gfx_texture_unload(compressed, context);
    }
    let mut telemetry = EzGfxTextureUploadTelemetry {
        decode_microseconds: 0,
        staging_bytes: 0,
        queue_latency_microseconds: 0,
        handoff_latency_microseconds: 0,
    };
    assert_eq!(
        // SAFETY: Output storage remains writable through this call.
        unsafe { ez_gfx_texture_get_upload_telemetry(&raw mut telemetry, context) },
        EzGfxResult::Ok
    );
    assert!(telemetry.staging_bytes > 0);
    drop(native);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_async_texture_batches_reach_full_residency() {
    exercises_async_texture_batches(1);
}

#[cfg(windows)]
#[test]
fn dx12_async_texture_batches_reach_full_residency() {
    exercises_async_texture_batches(2);
}
