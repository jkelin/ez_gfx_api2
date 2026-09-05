//! Progressive native texture-residency tests through the C ABI.
#![cfg(windows)]

mod common;

use common::TestContext;

use ez_gfx_ffi::{
    EzGfxResult, EzGfxTextureDesc, ez_gfx_texture_get_binding, ez_gfx_texture_get_residency,
    ez_gfx_texture_load, ez_gfx_texture_poll, ez_gfx_texture_unload,
};

#[cfg(windows)]
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

    for wave in 0..4 {
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
                match ez_gfx_texture_poll(texture, context) {
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

        for texture in textures {
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
            assert_eq!((resident, total), (3, 3));
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
    drop(native);
}

#[cfg(windows)]
#[test]
fn vulkan_async_texture_batches_reach_full_residency() {
    exercises_async_texture_batches(1);
}

#[cfg(windows)]
#[test]
fn dx12_async_texture_batches_reach_full_residency() {
    exercises_async_texture_batches(2);
}
