//! ABI GPU upload and frame contract tests (hidden devices).

use super::*;

#[cfg(not(target_vendor = "apple"))]
fn geometry_uploads_use_real_device_buffers_and_transfer_fence(backend: u8) {
    let native = common::TestContext::create(backend);
    let context = native.context;
    let heap = b"position";
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_vertex_heap_create(heap.as_ptr(), heap.len(), 8192, 16, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_index_heap_create(256, heap.as_ptr(), heap.len(), context) }
        },
        EzGfxResult::Ok
    );
    let vertices = [1_u8; 64];
    for upload in 0..96 {
        let mut first = u32::MAX;
        assert_eq!(
            {
                // SAFETY: the vertex slice is readable for four 16-byte elements for this call.
                unsafe {
                    ez_gfx_vertex_upload(
                        heap.as_ptr(),
                        heap.len(),
                        vertices.as_ptr().cast(),
                        4,
                        16,
                        &raw mut first,
                        context,
                    )
                }
            },
            EzGfxResult::Ok,
            "upload {upload}"
        );
        assert_eq!(first, upload * 4);
        if upload % 8 == 7 {
            assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
        }
    }
    let mut first = u32::MAX;
    let indices = [0_u32, 1, 2];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_vertex_upload_indices(indices.as_ptr().cast(), 3, &raw mut first, context)
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(first, 0);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    // SAFETY: `heap` is readable for exactly `heap.len()` UTF-8 bytes for this call.
    unsafe { ez_gfx_vertex_heap_destroy(heap.as_ptr(), heap.len(), context) };
    ez_gfx_index_heap_destroy(context);
    drop(native);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_geometry_uploads_use_real_device_buffers_and_transfer_timeline() {
    geometry_uploads_use_real_device_buffers_and_transfer_fence(1);
}

#[cfg(windows)]
#[test]
fn dx12_geometry_uploads_use_real_device_buffers_and_transfer_fence() {
    geometry_uploads_use_real_device_buffers_and_transfer_fence(2);
}

#[cfg(windows)]
#[test]
fn dx12_texture_upload_becomes_resident_and_unload_invalidates_handle() {
    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend: 2,
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
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
    let mut context = 0;
    let mut texture = 0;
    let pixels = [1_u8, 2, 3, 4];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const context_desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
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
        EzGfxResult::Ok
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let completion = loop {
        let status = ez_gfx_texture_poll(texture, context);
        if status != EzGfxResult::NotReady || std::time::Instant::now() >= deadline {
            break status;
        }
        std::thread::yield_now();
    };
    assert_eq!(completion, EzGfxResult::Ok);
    let mut binding = u32::MAX;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_texture_get_binding(texture, &raw mut binding, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(binding, 0);
    ez_gfx_texture_unload(texture, context);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_texture_get_binding(texture, &raw mut binding, context) }
        },
        EzGfxResult::InvalidContext
    );
    ez_gfx_context_destroy(context);
}

#[cfg(not(target_vendor = "apple"))]
fn frame_uploads_indirect_compiles_graph_and_reads_back_texture(backend: u8) {
    let native = common::TestContext::create(backend);
    let context = native.context;
    let texture_desc = EzGfxTextureDesc {
        source_format: 1,
        destination_format: 0,
        width: 2,
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
    let pixels = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let command = EzGfxDrawIndexedCommand {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
    };
    let mut texture = 0;
    let mut indirect = 0;
    let debug_name = b"frame";
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
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
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    let mut binding = u32::MAX;
    assert_eq!(
        {
            // SAFETY: `binding` is writable u32 storage and both handles remain live.
            unsafe { ez_gfx_texture_get_binding(texture, &raw mut binding, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(binding, 0);
    assert_eq!(ez_gfx_frame_begin(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_acquire_indirect(
                    1,
                    debug_name.as_ptr(),
                    debug_name.len(),
                    &raw mut indirect,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_indirect_write_draw(indirect, 0, &raw const command, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_indirect_set_draw_count(indirect, 1, context),
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_graph_enqueue_texture_readback(texture, context),
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_submit(context), EzGfxResult::Ok);
    let mut size = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_frame_readback(core::ptr::null_mut(), 0, &raw mut size, context) }
        },
        EzGfxResult::Ok
    );
    let mut actual = vec![0; size];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_frame_readback(actual.as_mut_ptr(), actual.len(), &raw mut size, context)
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(actual, pixels);
    ez_gfx_indirect_release(indirect, context);
    ez_gfx_texture_unload(texture, context);
    drop(native);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_frame_uploads_indirect_compiles_graph_and_reads_back_texture() {
    frame_uploads_indirect_compiles_graph_and_reads_back_texture(1);
}

#[cfg(windows)]
#[test]
fn dx12_frame_uploads_indirect_compiles_graph_and_reads_back_texture() {
    frame_uploads_indirect_compiles_graph_and_reads_back_texture(2);
}
