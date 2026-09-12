//! ABI GPU upload and frame contract tests (hidden devices).

use super::*;

#[derive(Debug, Eq, PartialEq)]
struct ObservedReadback {
    request_id: u64,
    source: u64,
    source_kind: EzGfxReadbackSourceKind,
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
                source: event.readback_source,
                source_kind: event.readback_source_kind,
                width: event.readback_width,
                height: event.readback_height,
                bytes,
            });
        }
        _ => {}
    }
}
fn create_offscreen_target(context: u64, width: u32, height: u32) -> u64 {
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
        unsafe {
            ez_gfx_render_target_create(context, &raw const desc, width, height, &raw mut target)
        },
        EzGfxResult::Ok
    );
    target
}

fn begin_offscreen_frame(context: u64) -> (u64, u64) {
    let target = create_offscreen_target(context, 1, 1);
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
    let native = common::TestContext::create_with_validation(backend, false);
    let context = native.context;
    let heap_name = b"position";
    let mut heap = 0;
    assert_eq!(
        {
            // SAFETY: the heap name and output are valid for this call.
            unsafe {
                ez_gfx_vertex_heap_create(
                    context,
                    heap_name.as_ptr(),
                    heap_name.len(),
                    16,
                    &raw mut heap,
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
                    ez_gfx_vertex_heap_upload(
                        context,
                        heap,
                        vertices.as_ptr().cast(),
                        4,
                        16,
                        &raw mut allocation,
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
                    context,
                    allocation,
                    &raw mut first,
                    &raw mut count,
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
                ez_gfx_index_allocation_create(
                    context,
                    indices.as_ptr().cast(),
                    3,
                    &raw mut index_allocation,
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
                context,
                index_allocation,
                &raw mut first,
                &raw mut count,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!((first, count), (0, 3));
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    for allocation in vertex_allocations {
        assert_eq!(
            ez_gfx_vertex_allocation_remove(context, allocation),
            EzGfxResult::Ok
        );
    }
    assert_eq!(
        ez_gfx_index_allocation_remove(context, index_allocation),
        EzGfxResult::Ok
    );
    ez_gfx_vertex_heap_destroy(context, heap);
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
        required_mips: 0,
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
                    context,
                    pixels.as_ptr(),
                    pixels.len(),
                    &raw const texture_desc,
                    &raw mut texture,
                )
            }
        },
        EzGfxResult::Ok
    );
    let mut collected = Collected::default();
    assert_eq!(
        // SAFETY: `collected` outlives the registration below through explicit clearing.
        unsafe {
            ez_gfx_context_register_callback(
                context,
                Some(collect_event),
                (&raw mut collected).cast(),
            )
        },
        EzGfxResult::Ok
    );
    let mut binding = u32::MAX;
    assert_eq!(
        // SAFETY: binding remains writable and both handles are live.
        unsafe { ez_gfx_texture_get_binding(context, texture, &raw mut binding) },
        EzGfxResult::Ok
    );
    assert_eq!(binding, 0);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
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
            unsafe { ez_gfx_texture_get_binding(context, texture, &raw mut binding) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(binding, 0);
    ez_gfx_texture_unload(context, texture);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_texture_get_binding(context, texture, &raw mut binding) }
        },
        EzGfxResult::InvalidContext
    );
    assert_eq!(ez_gfx_context_destroy(context), EzGfxResult::Ok);
}

#[cfg(not(target_vendor = "apple"))]
fn frame_uploads_indirect_compiles_graph_and_reads_back_texture(backend: u8) {
    let native = common::TestContext::create_with_validation(backend, false);
    let context = native.context;
    let texture_desc = EzGfxTextureDesc {
        source_format: 1,
        destination_format: 0,
        width: 2,
        height: 1,
        mip_count: 1,
        generate_mips: 0,
        required_mips: 0,
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
            ez_gfx_context_register_callback(
                context,
                Some(collect_event),
                (&raw mut collected).cast(),
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_texture_load(
                    context,
                    pixels.as_ptr(),
                    pixels.len(),
                    &raw const texture_desc,
                    &raw mut texture,
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
                    context,
                    u32::try_from(core::mem::size_of::<EzGfxDrawIndexedCommand>())
                        .expect("draw command size fits u32"),
                    1,
                    debug_name.as_ptr(),
                    debug_name.len(),
                    &raw mut indirect,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_counter_buffer_write_draws(context, indirect, 0, &raw const command, 1)
            }
        },
        EzGfxResult::Ok
    );
    let mut first_request = 0;
    let mut second_request = 0;
    assert_eq!(
        // SAFETY: each output points to writable aligned test-owned storage.
        unsafe {
            ez_gfx_frame_enqueue_texture_readback(context, frame, texture, &raw mut first_request)
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        // SAFETY: repeated same-source requests remain separately correlated.
        unsafe {
            ez_gfx_frame_enqueue_texture_readback(context, frame, texture, &raw mut second_request)
        },
        EzGfxResult::Ok
    );
    assert_ne!(first_request, second_request);
    assert_eq!(ez_gfx_frame_end(context, frame), EzGfxResult::Ok);
    assert_eq!(collected.readbacks.len(), 2);
    assert_eq!(collected.readbacks[0].request_id, first_request);
    assert_eq!(collected.readbacks[1].request_id, second_request);
    assert!(collected.readbacks.iter().all(|readback| {
        readback.source == texture
            && readback.source_kind == EzGfxReadbackSourceKind::Texture
            && readback.width == texture_desc.width
            && readback.height == texture_desc.height
            && readback.bytes == pixels
    }));
    assert_eq!(
        // SAFETY: clearing a live registration needs no user data.
        unsafe { ez_gfx_context_register_callback(context, None, core::ptr::null_mut()) },
        EzGfxResult::Ok
    );
    ez_gfx_render_target_destroy(context, target);
    ez_gfx_texture_unload(context, texture);
    drop(native);
}

#[cfg(not(target_vendor = "apple"))]
fn render_target_readback_validates_handles_and_reports_dimensions(backend: u8) {
    let native = common::TestContext::create_with_validation(backend, false);
    let foreign = common::TestContext::create_with_validation(backend, false);
    let context = native.context;
    let target = create_offscreen_target(context, 3, 2);
    let stale = create_offscreen_target(context, 1, 1);
    let foreign_target = create_offscreen_target(foreign.context, 1, 1);
    ez_gfx_render_target_destroy(context, stale);

    let mut collected = Collected::default();
    assert_eq!(
        // SAFETY: the boxed callback state remains live until registration is cleared.
        unsafe {
            ez_gfx_context_register_callback(
                context,
                Some(collect_event),
                (&raw mut collected).cast(),
            )
        },
        EzGfxResult::Ok
    );
    let mut frame = 0;
    assert_eq!(
        // SAFETY: frame output storage is live and aligned.
        unsafe { ez_gfx_render_target_frame_begin(context, target, &raw mut frame) },
        EzGfxResult::Ok
    );

    let mut request_id = 0xA5_A5_u64;
    for invalid_target in [stale, foreign_target] {
        assert_eq!(
            // SAFETY: output storage is live and aligned; the target handle is intentionally invalid for this context.
            unsafe {
                ffi::ez_gfx_frame_enqueue_render_target_readback(
                    context,
                    frame,
                    invalid_target,
                    &raw mut request_id,
                )
            },
            EzGfxResult::InvalidContext
        );
        assert_eq!(request_id, 0xA5_A5);
    }

    assert_eq!(
        // SAFETY: frame, target, and output storage remain live through the call.
        unsafe {
            ffi::ez_gfx_frame_enqueue_render_target_readback(
                context,
                frame,
                target,
                &raw mut request_id,
            )
        },
        EzGfxResult::Ok
    );
    assert_ne!(request_id, 0xA5_A5);
    assert_eq!(ez_gfx_frame_end(context, frame), EzGfxResult::Ok);
    assert_eq!(collected.readbacks.len(), 1);
    let readback = &collected.readbacks[0];
    assert_eq!(readback.request_id, request_id);
    assert_eq!(readback.source, target);
    assert_eq!(readback.source_kind, EzGfxReadbackSourceKind::RenderTarget);
    assert_eq!((readback.width, readback.height), (3, 2));
    assert_eq!(readback.bytes.len(), 3 * 2 * 4);

    assert_eq!(
        // SAFETY: clearing a live registration needs no user data.
        unsafe { ez_gfx_context_register_callback(context, None, core::ptr::null_mut()) },
        EzGfxResult::Ok
    );
    ez_gfx_render_target_destroy(context, target);
    ez_gfx_render_target_destroy(foreign.context, foreign_target);
    drop(foreign);
    drop(native);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_render_target_readback_validates_handles_and_reports_dimensions() {
    render_target_readback_validates_handles_and_reports_dimensions(1);
}

#[cfg(windows)]
#[test]
fn dx12_render_target_readback_validates_handles_and_reports_dimensions() {
    render_target_readback_validates_handles_and_reports_dimensions(2);
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
