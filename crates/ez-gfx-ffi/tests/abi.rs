//! ABI layout, validation, lifecycle, and observability contract tests.

use core::mem::{align_of, size_of};
#[allow(
    unused_imports,
    reason = "platform-specific tests consume different ABI symbols"
)]
use ez_gfx_ffi::{
    EZ_GFX_ABI_VERSION, EzGfxBackendContextDesc, EzGfxContextDesc, EzGfxDiagnostic,
    EzGfxDrawIndexedCommand, EzGfxDynamicState, EzGfxHandleParts, EzGfxResult, EzGfxRuntimeRecord,
    EzGfxSurfaceDesc, EzGfxTextureDesc, ez_gfx_abi_version, ez_gfx_acquire_indirect,
    ez_gfx_context_create, ez_gfx_context_create_backend, ez_gfx_context_destroy,
    ez_gfx_context_wait_idle, ez_gfx_finish_render, ez_gfx_frame_begin, ez_gfx_frame_readback,
    ez_gfx_frame_submit, ez_gfx_graph_enqueue_texture_readback, ez_gfx_handle_inspect,
    ez_gfx_index_heap_create, ez_gfx_index_heap_destroy, ez_gfx_indirect_release,
    ez_gfx_indirect_set_draw_count, ez_gfx_indirect_write_draw, ez_gfx_poll_diagnostic,
    ez_gfx_poll_runtime_event, ez_gfx_semantic_id, ez_gfx_structured_acquire,
    ez_gfx_structured_release, ez_gfx_structured_write, ez_gfx_texture_get_binding,
    ez_gfx_texture_get_residency, ez_gfx_texture_load, ez_gfx_texture_unload,
    ez_gfx_vertex_heap_create, ez_gfx_vertex_heap_destroy, ez_gfx_vertex_upload,
    ez_gfx_vertex_upload_indices,
};

#[test]
fn layouts_and_status_values_are_stable() {
    assert_eq!(ez_gfx_abi_version(), EZ_GFX_ABI_VERSION);
    assert_eq!(EzGfxResult::Ok as u8, 0);
    assert_eq!(EzGfxResult::InvalidArgument as u8, 1);
    assert_eq!(EzGfxResult::InvalidContext as u8, 2);
    assert_eq!(EzGfxResult::NativeFailure as u8, 3);
    assert_eq!(EzGfxResult::NotReady as u8, 4);
    assert_eq!(EzGfxResult::Unsupported as u8, 5);
    assert_eq!(EZ_GFX_ABI_VERSION, 17);
    assert_eq!(size_of::<EzGfxContextDesc>(), 3);
    assert_eq!(size_of::<EzGfxBackendContextDesc>(), 4);
    assert_eq!(size_of::<EzGfxDynamicState>(), 4);
    assert_eq!(size_of::<EzGfxDrawIndexedCommand>(), 20);
    assert!(size_of::<EzGfxSurfaceDesc>() >= 29);
    assert_eq!(size_of::<EzGfxHandleParts>(), 20);
    assert_eq!(align_of::<EzGfxHandleParts>(), 4);
    assert_eq!(size_of::<EzGfxRuntimeRecord>(), 24);
    assert_eq!(align_of::<EzGfxRuntimeRecord>(), 8);
    assert_eq!(size_of::<EzGfxDiagnostic>(), 32);
}

#[test]
fn finish_render_rejects_a_null_context_before_native_dispatch() {
    assert_eq!(ez_gfx_finish_render(0), EzGfxResult::InvalidContext);
}

#[test]
fn observability_polls_validate_all_output_pointers() {
    let mut record = EzGfxRuntimeRecord {
        correlation_id: 0,
        resource: 0,
        backend: 0,
        phase: 0,
        status: 0,
        _padding: [0; 5],
    };
    let mut diagnostic = EzGfxDiagnostic {
        record,
        level: 0,
        _padding: [0; 7],
    };
    let mut present = 0;
    let mut dropped = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_poll_runtime_event(
                    core::ptr::null_mut(),
                    &raw mut present,
                    &raw mut dropped,
                    0,
                )
            }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_poll_runtime_event(
                    &raw mut record,
                    core::ptr::null_mut(),
                    &raw mut dropped,
                    0,
                )
            }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_poll_diagnostic(
                    &raw mut diagnostic,
                    &raw mut present,
                    core::ptr::null_mut(),
                    0,
                )
            }
        },
        EzGfxResult::InvalidArgument
    );
}

#[test]
fn texture_residency_validates_both_output_pointers() {
    let mut value = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_texture_get_residency(0, core::ptr::null_mut(), &raw mut value, 0) }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_texture_get_residency(0, &raw mut value, core::ptr::null_mut(), 0) }
        },
        EzGfxResult::InvalidArgument
    );
}

#[test]
fn context_creation_rejects_boundary_inputs_before_native_calls() {
    let mut context = 99;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create(core::ptr::null(), &raw mut context) }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(context, 99);

    let invalid = EzGfxContextDesc {
        enable_debug: 2,
        enable_validation: 0,
        surface_platform: 0,
    };
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create(&raw const invalid, &raw mut context) }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(context, 99);
}

#[cfg(windows)]
#[test]
fn context_lifecycle_admits_a_real_vulkan_device_and_invalidates_destroyed_handle() {
    let desc = EzGfxContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
    };
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create(&raw const desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );
    assert_ne!(context, 0);
    std::thread::spawn(move || ez_gfx_context_destroy(context))
        .join()
        .unwrap();
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    ez_gfx_context_destroy(context);
    assert_eq!(
        ez_gfx_context_wait_idle(context),
        EzGfxResult::InvalidContext
    );
}

#[cfg(windows)]
#[test]
fn explicit_dx12_context_allocates_writes_and_releases_structured_memory() {
    let desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend: 2,
    };
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );
    let name = std::ffi::CString::new("vertices").unwrap();
    let mut structured = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_structured_acquire(16, 4, name.as_ptr(), &raw mut structured, context) }
        },
        EzGfxResult::Ok
    );
    let bytes = [7_u8; 64];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_structured_write(
                    structured,
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_structured_write(structured, bytes.as_ptr().cast(), 65, context) }
        },
        EzGfxResult::InvalidArgument
    );
    ez_gfx_structured_release(structured, context);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_structured_write(
                    structured,
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    context,
                )
            }
        },
        EzGfxResult::InvalidContext
    );
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    ez_gfx_context_destroy(context);
    assert_eq!(
        ez_gfx_context_wait_idle(context),
        EzGfxResult::InvalidContext
    );
}

#[cfg(windows)]
#[test]
fn dx12_geometry_uploads_use_real_device_buffers_and_transfer_fence() {
    let desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend: 2,
    };
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );
    let heap = std::ffi::CString::new("position").unwrap();
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_vertex_heap_create(heap.as_ptr(), 256, 16, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_index_heap_create(256, heap.as_ptr(), context) }
        },
        EzGfxResult::Ok
    );
    let vertices = [1_u8; 64];
    let mut first = u32::MAX;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_vertex_upload(
                    heap.as_ptr(),
                    vertices.as_ptr().cast(),
                    4,
                    16,
                    &raw mut first,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(first, 0);
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
    // SAFETY: `heap` is a live NUL-terminated string for this call.
    unsafe { ez_gfx_vertex_heap_destroy(heap.as_ptr(), context) };
    ez_gfx_index_heap_destroy(context);
    ez_gfx_context_destroy(context);
}

#[test]
fn texture_descriptor_rejects_unsupported_pipeline_state_before_context_access() {
    let pixels = [0_u8; 4];
    let mut texture = 0;
    let valid = EzGfxTextureDesc {
        source_format: 1,
        destination_format: 0,
        width: 1,
        height: 1,
        mip_count: 1,
        generate_mips: 0,
        min_filter: 0,
        mag_filter: 1,
        max_anisotropy: 1.0,
        address_mode_u: 0,
        address_mode_v: 1,
        address_mode_w: 0,
        debug_label: core::ptr::null(),
    };
    for desc in [
        EzGfxTextureDesc {
            destination_format: 1,
            ..valid
        },
        EzGfxTextureDesc {
            min_filter: 2,
            ..valid
        },
        EzGfxTextureDesc {
            address_mode_u: 2,
            ..valid
        },
        EzGfxTextureDesc {
            max_anisotropy: 0.0,
            ..valid
        },
        EzGfxTextureDesc {
            max_anisotropy: f32::NAN,
            ..valid
        },
    ] {
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe {
                    ez_gfx_texture_load(
                        pixels.as_ptr(),
                        pixels.len(),
                        &raw const desc,
                        &raw mut texture,
                        0,
                    )
                }
            },
            EzGfxResult::InvalidArgument
        );
    }
}

#[cfg(windows)]
#[test]
fn dx12_texture_upload_becomes_resident_and_unload_invalidates_handle() {
    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend: 2,
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
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
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

#[cfg(windows)]
#[test]
fn dx12_frame_uploads_indirect_compiles_graph_and_reads_back_texture() {
    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend: 2,
    };
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
    };
    let pixels = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let command = EzGfxDrawIndexedCommand {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
    };
    let mut context = 0;
    let mut texture = 0;
    let mut indirect = 0;
    let debug_name = std::ffi::CString::new("frame").unwrap();
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
    assert_eq!(ez_gfx_frame_begin(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_acquire_indirect(1, debug_name.as_ptr(), &raw mut indirect, context) }
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
    ez_gfx_context_destroy(context);
}
#[test]
fn pointer_count_and_utf8_are_validated() {
    let mut output = [0_u8; 16];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_semantic_id(core::ptr::null(), 1, output.as_mut_ptr()) }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_semantic_id([0xff_u8].as_ptr(), 1, output.as_mut_ptr()) }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_semantic_id(b"material.albedo".as_ptr(), 15, output.as_mut_ptr()) }
        },
        EzGfxResult::Ok
    );
    assert_ne!(output, [0; 16]);
}

#[test]
fn malformed_handles_fail_without_touching_output() {
    let mut parts = EzGfxHandleParts {
        context_slot: 99,
        context_generation: 99,
        child_slot: 99,
        child_generation: 99,
        is_context: 99,
        _padding: [99; 3],
    };
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_handle_inspect(1, &raw mut parts) }
        },
        EzGfxResult::InvalidContext
    );
    assert_eq!(parts.context_slot, 99);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_handle_inspect(0, core::ptr::null_mut()) }
        },
        EzGfxResult::InvalidArgument
    );
}
