//! ABI GPU upload and frame contract tests (hidden devices).

use super::*;

#[derive(Debug, Eq, PartialEq)]
struct ObservedReadback {
    request_id: u64,
    texture: u64,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct Collected {
    uploads: Vec<(u64, u8)>,
    readbacks: Vec<ObservedReadback>,
}

/// Records callback-delivered events into test-owned storage.
///
/// # Safety
///
/// `user_data` must address a live `Collected` for the registration lifetime.
unsafe extern "C" fn collect_event(event: *const EzGfxEvent, user_data: *mut core::ffi::c_void) {
    // SAFETY: registration keeps this test-owned allocation alive through each delivery.
    let out = unsafe { &mut *user_data.cast::<Collected>() };
    // SAFETY: the event borrows its payload for this invocation only.
    let event = unsafe { &*event };
    match event.kind {
        EzGfxEventKind::Upload => {
            out.uploads
                .push((event.upload.resource, event.upload.status));
        }
        EzGfxEventKind::Readback => {
            let bytes = if event.readback_bytes.is_null() || event.readback_byte_count == 0 {
                Vec::new()
            } else {
                // SAFETY: the borrowed byte range is live for this invocation.
                unsafe {
                    core::slice::from_raw_parts(event.readback_bytes, event.readback_byte_count)
                }
                .to_vec()
            };
            out.readbacks.push(ObservedReadback {
                request_id: event.readback_request_id,
                texture: event.readback_texture,
                width: event.readback_width,
                height: event.readback_height,
                bytes,
            });
        }
        _ => {}
    }
}
fn begin_offscreen_frame(context: u64) -> (u64, u64) {
    let name = b"readback-target";
    let format = 1_u8;
    let desc = EzGfxRenderTargetDesc {
        name: name.as_ptr(),
        name_length: name.len(),
        usage: 0,
        relative_scale: 1.0,
        samples: 1,
        candidate_formats: &raw const format,
        candidate_count: 1,
        sampleable: 0,
        use_clear: 0,
        clear_color: [0.0; 4],
    };
    let mut target = 0;
    assert_eq!(
        // SAFETY: descriptor, format, and output storage remain live through the call.
        unsafe { ez_gfx_render_target_create(&raw const desc, 1, 1, &raw mut target, context) },
        EzGfxResult::Ok
    );
    let mut frame = 0;
    assert_eq!(
        // SAFETY: frame output storage is live and aligned.
        unsafe { ez_gfx_render_target_frame_begin(context, target, &raw mut frame) },
        EzGfxResult::Ok
    );
    (frame, target)
}

#[cfg(not(target_vendor = "apple"))]
fn geometry_uploads_use_real_device_buffers_and_transfer_fence(backend: u8) {
    let native = common::TestContext::create(backend);
    let context = native.context;
    let heap_name = b"position";
    let mut heap = 0;
    assert_eq!(
        {
            // SAFETY: the heap name and output are valid for this call.
            unsafe {
                ez_gfx_vertex_heap_create(
                    heap_name.as_ptr(),
                    heap_name.len(),
                    16,
                    &raw mut heap,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let vertices = [1_u8; 64];
    let mut vertex_allocations = Vec::new();
    for upload in 0..96 {
        let mut allocation = 0_u64;
        assert_eq!(
            {
                // SAFETY: the vertex slice is readable for four 16-byte elements for this call.
                unsafe {
                    ez_gfx_vertex_upload(
                        heap,
                        vertices.as_ptr().cast(),
                        4,
                        16,
                        &raw mut allocation,
                        context,
                    )
                }
            },
            EzGfxResult::Ok,
            "upload {upload}"
        );
        let mut first = u32::MAX;
        let mut count = 0;
        assert_eq!(
            // SAFETY: both outputs are writable and the allocation/context handles are live.
            unsafe {
                ez_gfx_vertex_allocation_get_range(
                    allocation,
                    &raw mut first,
                    &raw mut count,
                    context,
                )
            },
            EzGfxResult::Ok
        );
        assert_eq!((first, count), (upload * 4, 4));
        vertex_allocations.push(allocation);
        if upload % 8 == 7 {
            assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
        }
    }
    let mut index_allocation = 0_u64;
    let indices = [0_u32, 1, 2];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_vertex_upload_indices(
                    indices.as_ptr().cast(),
                    3,
                    &raw mut index_allocation,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let mut first = u32::MAX;
    let mut count = 0;
    assert_eq!(
        // SAFETY: both outputs are writable and the allocation/context handles are live.
        unsafe {
            ez_gfx_index_allocation_get_range(
                index_allocation,
                &raw mut first,
                &raw mut count,
                context,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!((first, count), (0, 3));
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    for allocation in vertex_allocations {
        assert_eq!(
            ez_gfx_vertex_allocation_remove(allocation, context),
            EzGfxResult::Ok
        );
    }
    assert_eq!(
        ez_gfx_index_allocation_remove(index_allocation, context),
        EzGfxResult::Ok
    );
    ez_gfx_vertex_heap_destroy(heap, context);
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
    let mut collected = Collected::default();
    assert_eq!(
        // SAFETY: `collected` outlives the registration below through explicit clearing.
        unsafe {
            ez_gfx_callback_register(context, Some(collect_event), (&raw mut collected).cast())
        },
        EzGfxResult::Ok
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let completion = loop {
        let mut binding = u32::MAX;
        let status =
            // SAFETY: binding remains writable and both handles are live.
            unsafe { ez_gfx_texture_get_binding(texture, &raw mut binding, context) };
        if status != EzGfxResult::NotReady || std::time::Instant::now() >= deadline {
            break status;
        }
        // Owner-thread idle admits transfers and dispatches their events.
        assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
        std::thread::yield_now();
    };
    assert_eq!(completion, EzGfxResult::Ok);
    assert!(
        collected
            .uploads
            .iter()
            .any(|&(resource, status)| resource == texture && status == 2)
    );
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
    let mut collected = Collected::default();
    assert_eq!(
        // SAFETY: `collected` outlives the registration below through explicit clearing.
        unsafe {
            ez_gfx_callback_register(context, Some(collect_event), (&raw mut collected).cast())
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
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    let (frame, target) = begin_offscreen_frame(context);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_counter_buffer_acquire(
                    u32::try_from(core::mem::size_of::<EzGfxDrawIndexedCommand>())
                        .expect("draw command size fits u32"),
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
            unsafe {
                ez_gfx_counter_buffer_write_draws(indirect, 0, &raw const command, 1, context)
            }
        },
        EzGfxResult::Ok
    );
    let mut first_request = 0;
    let mut second_request = 0;
    assert_eq!(
        // SAFETY: each output points to writable aligned test-owned storage.
        unsafe { ez_gfx_graph_enqueue_texture_readback(texture, frame, &raw mut first_request) },
        EzGfxResult::Ok
    );
    assert_eq!(
        // SAFETY: repeated same-source requests remain separately correlated.
        unsafe { ez_gfx_graph_enqueue_texture_readback(texture, frame, &raw mut second_request) },
        EzGfxResult::Ok
    );
    assert_ne!(first_request, second_request);
    assert_eq!(ez_gfx_frame_end(frame), EzGfxResult::Ok);
    assert_eq!(collected.readbacks.len(), 2);
    assert_eq!(collected.readbacks[0].request_id, first_request);
    assert_eq!(collected.readbacks[1].request_id, second_request);
    assert!(collected.readbacks.iter().all(|readback| {
        readback.texture == texture
            && readback.width == texture_desc.width
            && readback.height == texture_desc.height
            && readback.bytes == pixels
    }));
    assert_eq!(
        // SAFETY: clearing a live registration needs no user data.
        unsafe { ez_gfx_callback_register(context, None, core::ptr::null_mut()) },
        EzGfxResult::Ok
    );
    ez_gfx_render_target_destroy(target, context);
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
