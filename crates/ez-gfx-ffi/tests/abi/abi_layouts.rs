//! ABI layout, signature, and string contract tests.

use super::*;

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one contiguous test keeps every public C layout and field-offset assertion visible as a single ABI contract"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "sequential layout assertions share one ABI contract; splitting would hide drift"
)]
fn layouts_are_stable() {
    // ABI 26 appends the adapter selector pair to both creation descriptors.
    // Earlier fields keep their offsets; the trailing pointer raises size and
    // alignment to the pointer width.
    assert_eq!(
        (
            size_of::<EzGfxContextDesc>(),
            align_of::<EzGfxContextDesc>()
        ),
        (24, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxContextDesc, enable_debug),
            offset_of!(EzGfxContextDesc, enable_validation),
            offset_of!(EzGfxContextDesc, surface_platform),
            offset_of!(EzGfxContextDesc, texture_decode_workers),
            offset_of!(EzGfxContextDesc, adapter_count),
            offset_of!(EzGfxContextDesc, adapter)
        ],
        [0, 1, 2, 4, 8, 16]
    );
    assert_eq!(
        (
            size_of::<EzGfxBackendContextDesc>(),
            align_of::<EzGfxBackendContextDesc>()
        ),
        (24, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxBackendContextDesc, enable_debug),
            offset_of!(EzGfxBackendContextDesc, enable_validation),
            offset_of!(EzGfxBackendContextDesc, surface_platform),
            offset_of!(EzGfxBackendContextDesc, backend),
            offset_of!(EzGfxBackendContextDesc, texture_decode_workers),
            offset_of!(EzGfxBackendContextDesc, adapter_count),
            offset_of!(EzGfxBackendContextDesc, adapter)
        ],
        [0, 1, 2, 3, 4, 8, 16]
    );
    assert_eq!(
        (
            size_of::<EzGfxAdapterDesc>(),
            align_of::<EzGfxAdapterDesc>()
        ),
        (17, 1)
    );
    assert_eq!(
        [
            offset_of!(EzGfxAdapterDesc, stable_id),
            offset_of!(EzGfxAdapterDesc, allow_software)
        ],
        [0, 16]
    );
    assert_eq!(
        (
            size_of::<EzGfxAdapterInfo>(),
            align_of::<EzGfxAdapterInfo>()
        ),
        (24, 4)
    );
    assert_eq!(
        [
            offset_of!(EzGfxAdapterInfo, stable_id),
            offset_of!(EzGfxAdapterInfo, backend),
            offset_of!(EzGfxAdapterInfo, adapter_class),
            offset_of!(EzGfxAdapterInfo, admitted),
            offset_of!(EzGfxAdapterInfo, software_rejected),
            offset_of!(EzGfxAdapterInfo, error_count)
        ],
        [0, 16, 17, 18, 19, 20]
    );
    assert_eq!(
        (
            size_of::<EzGfxSurfaceDesc>(),
            align_of::<EzGfxSurfaceDesc>()
        ),
        (32, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxSurfaceDesc, window),
            offset_of!(EzGfxSurfaceDesc, display),
            offset_of!(EzGfxSurfaceDesc, platform),
            offset_of!(EzGfxSurfaceDesc, width),
            offset_of!(EzGfxSurfaceDesc, height),
            offset_of!(EzGfxSurfaceDesc, cache_presented_snapshots)
        ],
        [0, 8, 16, 20, 24, 28]
    );
    assert_eq!(
        (size_of::<EzGfxShaderDesc>(), align_of::<EzGfxShaderDesc>()),
        (72, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxShaderDesc, path),
            offset_of!(EzGfxShaderDesc, path_length),
            offset_of!(EzGfxShaderDesc, vertex_entry),
            offset_of!(EzGfxShaderDesc, vertex_entry_length),
            offset_of!(EzGfxShaderDesc, fragment_entry),
            offset_of!(EzGfxShaderDesc, fragment_entry_length),
            offset_of!(EzGfxShaderDesc, compute_entry),
            offset_of!(EzGfxShaderDesc, compute_entry_length),
            offset_of!(EzGfxShaderDesc, kind)
        ],
        [0, 8, 16, 24, 32, 40, 48, 56, 64]
    );
    assert_eq!(
        (
            size_of::<EzGfxTextureDesc>(),
            align_of::<EzGfxTextureDesc>()
        ),
        (48, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxTextureDesc, source_format),
            offset_of!(EzGfxTextureDesc, destination_format),
            offset_of!(EzGfxTextureDesc, width),
            offset_of!(EzGfxTextureDesc, height),
            offset_of!(EzGfxTextureDesc, mip_count),
            offset_of!(EzGfxTextureDesc, generate_mips),
            offset_of!(EzGfxTextureDesc, min_filter),
            offset_of!(EzGfxTextureDesc, mag_filter),
            offset_of!(EzGfxTextureDesc, max_anisotropy),
            offset_of!(EzGfxTextureDesc, address_mode_u),
            offset_of!(EzGfxTextureDesc, address_mode_v),
            offset_of!(EzGfxTextureDesc, address_mode_w),
            offset_of!(EzGfxTextureDesc, debug_label),
            offset_of!(EzGfxTextureDesc, debug_label_length)
        ],
        [0, 1, 4, 8, 12, 16, 17, 18, 20, 24, 25, 26, 32, 40]
    );
    // ABI 25 adds the render-target declaration struct. Discriminant-carrying
    // enums stay one byte; the descriptor borrows its name and candidate ranges.
    assert_eq!(
        (
            size_of::<EzGfxRenderTargetDesc>(),
            align_of::<EzGfxRenderTargetDesc>()
        ),
        (64, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxRenderTargetDesc, name),
            offset_of!(EzGfxRenderTargetDesc, name_length),
            offset_of!(EzGfxRenderTargetDesc, usage),
            offset_of!(EzGfxRenderTargetDesc, relative_scale),
            offset_of!(EzGfxRenderTargetDesc, samples),
            offset_of!(EzGfxRenderTargetDesc, candidate_formats),
            offset_of!(EzGfxRenderTargetDesc, candidate_count),
            offset_of!(EzGfxRenderTargetDesc, sampleable),
            offset_of!(EzGfxRenderTargetDesc, use_clear),
            offset_of!(EzGfxRenderTargetDesc, clear_color)
        ],
        [0, 8, 16, 20, 24, 32, 40, 44, 45, 48]
    );
    assert_eq!(
        (size_of::<EzGfxBinding>(), align_of::<EzGfxBinding>()),
        (40, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxBinding, name),
            offset_of!(EzGfxBinding, name_length),
            offset_of!(EzGfxBinding, buffer),
            offset_of!(EzGfxBinding, counter_buffer),
            offset_of!(EzGfxBinding, render_target)
        ],
        [0, 8, 16, 24, 32]
    );
    assert_eq!(
        (
            size_of::<EzGfxDynamicState>(),
            align_of::<EzGfxDynamicState>()
        ),
        (4, 1)
    );
    assert_eq!(
        [
            offset_of!(EzGfxDynamicState, cull_mode),
            offset_of!(EzGfxDynamicState, front_face),
            offset_of!(EzGfxDynamicState, primitive_type),
            offset_of!(EzGfxDynamicState, blend_mode)
        ],
        [0, 1, 2, 3]
    );
    assert_eq!(
        (
            size_of::<EzGfxDrawIndexedCommand>(),
            align_of::<EzGfxDrawIndexedCommand>()
        ),
        (20, 4)
    );
    assert_eq!(
        [
            offset_of!(EzGfxDrawIndexedCommand, index_count),
            offset_of!(EzGfxDrawIndexedCommand, instance_count),
            offset_of!(EzGfxDrawIndexedCommand, first_index),
            offset_of!(EzGfxDrawIndexedCommand, vertex_offset),
            offset_of!(EzGfxDrawIndexedCommand, first_instance)
        ],
        [0, 4, 8, 12, 16]
    );
    assert_eq!(
        (size_of::<EzGfxByteBuffer>(), align_of::<EzGfxByteBuffer>()),
        (16, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxByteBuffer, length),
            offset_of!(EzGfxByteBuffer, data)
        ],
        [0, 8]
    );
    assert_eq!(
        (
            size_of::<EzGfxRuntimeRecord>(),
            align_of::<EzGfxRuntimeRecord>()
        ),
        (24, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxRuntimeRecord, correlation_id),
            offset_of!(EzGfxRuntimeRecord, resource),
            offset_of!(EzGfxRuntimeRecord, backend),
            offset_of!(EzGfxRuntimeRecord, phase),
            offset_of!(EzGfxRuntimeRecord, status),
            offset_of!(EzGfxRuntimeRecord, _padding)
        ],
        [0, 8, 16, 17, 18, 19]
    );
    assert_eq!(
        (
            size_of::<EzGfxUploadEvent>(),
            align_of::<EzGfxUploadEvent>()
        ),
        (16, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxUploadEvent, resource),
            offset_of!(EzGfxUploadEvent, resource_kind),
            offset_of!(EzGfxUploadEvent, status),
            offset_of!(EzGfxUploadEvent, error),
            offset_of!(EzGfxUploadEvent, _padding)
        ],
        [0, 8, 9, 10, 11]
    );
    assert_eq!(
        (size_of::<EzGfxDiagnostic>(), align_of::<EzGfxDiagnostic>()),
        (32, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxDiagnostic, record),
            offset_of!(EzGfxDiagnostic, level),
            offset_of!(EzGfxDiagnostic, _padding)
        ],
        [0, 24, 25]
    );
    assert_eq!(EzGfxEventKind::Upload as u8, 1);
    assert_eq!(EzGfxEventKind::Runtime as u8, 2);
    assert_eq!(EzGfxEventKind::Diagnostic as u8, 3);
    assert_eq!(EzGfxEventKind::ObservationsDropped as u8, 4);
    assert_eq!(EzGfxEventKind::Readback as u8, 5);
    assert_eq!(EzGfxEventKind::Snapshot as u8, 6);
    assert_eq!(
        (size_of::<EzGfxEvent>(), align_of::<EzGfxEvent>()),
        (104, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxEvent, kind),
            offset_of!(EzGfxEvent, upload),
            offset_of!(EzGfxEvent, record),
            offset_of!(EzGfxEvent, level),
            offset_of!(EzGfxEvent, dropped),
            offset_of!(EzGfxEvent, readback_request_id),
            offset_of!(EzGfxEvent, readback_texture),
            offset_of!(EzGfxEvent, readback_width),
            offset_of!(EzGfxEvent, readback_height),
            offset_of!(EzGfxEvent, readback_byte_count),
            offset_of!(EzGfxEvent, readback_bytes)
        ],
        [0, 8, 24, 48, 56, 64, 72, 80, 84, 88, 96]
    );
    assert_eq!(
        (
            size_of::<EzGfxHandleParts>(),
            align_of::<EzGfxHandleParts>()
        ),
        (20, 4)
    );
    assert_eq!(
        [
            offset_of!(EzGfxHandleParts, context_slot),
            offset_of!(EzGfxHandleParts, context_generation),
            offset_of!(EzGfxHandleParts, child_slot),
            offset_of!(EzGfxHandleParts, child_generation),
            offset_of!(EzGfxHandleParts, is_context),
            offset_of!(EzGfxHandleParts, _padding)
        ],
        [0, 4, 8, 12, 16, 17]
    );
}

#[test]
fn all_public_export_signatures_are_stable() {
    type Status = EzGfxResult;
    type Handle = u64;

    let _: extern "C" fn() -> u32 = ffi::ez_gfx_abi_version;
    let _: unsafe extern "C" fn(u8, *mut u8, usize, *mut usize) -> Status = ffi::ez_gfx_print_error;
    let _: unsafe extern "C" fn(*const EzGfxContextDesc, *mut Handle) -> Status =
        ffi::ez_gfx_context_create;
    let _: unsafe extern "C" fn(*const EzGfxBackendContextDesc, *mut Handle) -> Status =
        ffi::ez_gfx_context_create_backend;
    let _: unsafe extern "C" fn(*mut u32) -> Status = ffi::ez_gfx_adapter_count;
    let _: unsafe extern "C" fn(u8, *mut EzGfxAdapterInfo, u32, *mut u32) -> Status =
        ffi::ez_gfx_adapters_query;
    let _: extern "C" fn(Handle) -> Status = ffi::ez_gfx_context_wait_idle;
    let _: extern "C" fn(Handle) = ffi::ez_gfx_context_destroy;
    let _: unsafe extern "C" fn(*const EzGfxSurfaceDesc, *mut Handle, Handle) -> Status =
        ffi::ez_gfx_surface_create;
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_context_init_device;
    let _: extern "C" fn(Handle, u32, u32, Handle) -> Status = ffi::ez_gfx_surface_resize;
    let _: unsafe extern "C" fn(Handle, *mut u32, *mut u32, Handle) -> Status =
        ffi::ez_gfx_surface_get_extent;
    let _: unsafe extern "C" fn(Handle, *mut i32, Handle) -> Status =
        ffi::ez_gfx_surface_resize_pending;
    let _: extern "C" fn(Handle, i32, Handle) -> Status = ffi::ez_gfx_surface_set_snapshot_cache;
    let _: unsafe extern "C" fn(*const u8, usize, *mut Handle, Handle) -> Status =
        ffi::ez_gfx_shader_load_artifact;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_shader_destroy;
    let _: unsafe extern "C" fn(
        *const u8,
        usize,
        *const EzGfxTextureDesc,
        *mut Handle,
        Handle,
    ) -> Status = ffi::ez_gfx_texture_load;
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_texture_cancel;
    let _: unsafe extern "C" fn(Handle, *mut u32, Handle) -> Status =
        ffi::ez_gfx_texture_get_binding;
    let _: unsafe extern "C" fn(Handle, *mut u32, *mut u32, Handle) -> Status =
        ffi::ez_gfx_texture_get_residency;
    let _: extern "C" fn(Handle, u32, Handle) -> Status = ffi::ez_gfx_texture_set_residency;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_texture_unload;
    let _: unsafe extern "C" fn(
        *const EzGfxRenderTargetDesc,
        u32,
        u32,
        *mut Handle,
        Handle,
    ) -> Status = ffi::ez_gfx_render_target_create;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_render_target_destroy;
    let _: unsafe extern "C" fn(Handle, *mut u8, Handle) -> Status =
        ffi::ez_gfx_render_target_get_format;
    let _: unsafe extern "C" fn(Handle, *mut u32, *mut u32, Handle) -> Status =
        ffi::ez_gfx_render_target_get_extent;
    let _: unsafe extern "C" fn(Handle, *mut u8, *mut f32, Handle) -> Status =
        ffi::ez_gfx_render_target_get_clear;
    let _: extern "C" fn(u8, u8, Handle) -> Status = ffi::ez_gfx_render_target_probe_format;
    let _: unsafe extern "C" fn(Handle, Handle, *mut Handle) -> Status =
        ffi::ez_gfx_render_target_frame_begin;
    let _: unsafe extern "C" fn(Handle, Handle, *mut Handle) -> Status = ffi::ez_gfx_frame_begin;
    let _: unsafe extern "C" fn(u32, u32, *const u8, usize, *mut Handle, Handle) -> Status =
        ffi::ez_gfx_counter_buffer_acquire;
    let _: unsafe extern "C" fn(
        Handle,
        u32,
        *const EzGfxDrawIndexedCommand,
        u32,
        Handle,
    ) -> Status = ffi::ez_gfx_counter_buffer_write_draws;
    let _: extern "C" fn(Handle, u32, Handle) -> Status = ffi::ez_gfx_counter_buffer_publish_count;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_counter_buffer_release;
    let _: unsafe extern "C" fn(
        Handle,
        Handle,
        *const EzGfxBinding,
        u32,
        *const EzGfxDynamicState,
        *const c_void,
        u32,
        Handle,
    ) -> Status = ffi::ez_gfx_render_add_vertex_pipeline;
    let _: unsafe extern "C" fn(
        Handle,
        u32,
        u32,
        u32,
        *const EzGfxBinding,
        u32,
        *const c_void,
        u32,
        Handle,
    ) -> Status = ffi::ez_gfx_render_add_compute_pipeline;
    let _: unsafe extern "C" fn(Handle, Handle, *mut u64) -> Status =
        ffi::ez_gfx_graph_enqueue_texture_readback;
    let _: extern "C" fn(Handle) -> Status = ffi::ez_gfx_frame_end;
    let _: extern "C" fn(Handle) -> Status = ffi::ez_gfx_frame_abort;
    let _: unsafe extern "C" fn(Handle, EzGfxEventCallback, *mut c_void) -> Status =
        ffi::ez_gfx_callback_register;
    let _: unsafe extern "C" fn(*const u8, usize, u64, *mut Handle, Handle) -> Status =
        ffi::ez_gfx_vertex_heap_create;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_vertex_heap_destroy;
    let _: unsafe extern "C" fn(*const c_void, u32, *mut u64, Handle) -> Status =
        ffi::ez_gfx_vertex_upload_indices;
    let _: unsafe extern "C" fn(Handle, *const c_void, u32, u64, *mut u64, Handle) -> Status =
        ffi::ez_gfx_vertex_upload;
    let _: unsafe extern "C" fn(Handle, *mut u32, *mut u32, Handle) -> Status =
        ffi::ez_gfx_vertex_allocation_get_range;
    let _: unsafe extern "C" fn(Handle, *mut u32, *mut u32, Handle) -> Status =
        ffi::ez_gfx_index_allocation_get_range;
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_vertex_allocation_remove;
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_index_allocation_remove;
    let _: unsafe extern "C" fn(u32, u32, *const u8, usize, *mut Handle, Handle) -> Status =
        ffi::ez_gfx_buffer_acquire;
    let _: unsafe extern "C" fn(Handle, u32, *const c_void, u32, u32, Handle) -> Status =
        ffi::ez_gfx_buffer_write;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_buffer_release;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_surface_destroy;
    let _: unsafe extern "C" fn(u64, *mut EzGfxHandleParts) -> Status = ffi::ez_gfx_handle_inspect;
    let _: unsafe extern "C" fn(*const u8, usize, *mut u8) -> Status = ffi::ez_gfx_semantic_id;
}

#[test]
fn shader_load_v19_signature_and_boundary_validation_are_stable() {
    let _: unsafe extern "C" fn(*const u8, usize, *mut u64, u64) -> EzGfxResult =
        ez_gfx_shader_load_artifact;
    let mut shader = 99;
    assert_eq!(
        // SAFETY: Null data intentionally exercises checked rejection; output storage is live and aligned.
        unsafe { ez_gfx_shader_load_artifact(core::ptr::null(), 1, &raw mut shader, 0) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(shader, 99);
}

#[test]
fn counted_strings_reject_invalid_ranges_before_reading_or_delegating() {
    let valid = b"frame";
    let invalid_utf8 = [0xff_u8];
    let embedded_nul = b"a\0b";
    let mut indirect = 7;

    assert_eq!(
        // SAFETY: `valid` is readable for exactly its nonzero byte length and intentionally has no terminator.
        unsafe {
            ez_gfx_counter_buffer_acquire(20, 1, valid.as_ptr(), valid.len(), &raw mut indirect, 0)
        },
        EzGfxResult::InvalidContext
    );

    for (pointer, length) in [
        (core::ptr::null(), 1),
        (valid.as_ptr(), 0),
        (
            valid.as_ptr(),
            ffi::EZ_GFX_MAX_BOUNDARY_BYTES.saturating_add(1),
        ),
        (invalid_utf8.as_ptr(), invalid_utf8.len()),
        (embedded_nul.as_ptr(), embedded_nul.len()),
    ] {
        assert_eq!(
            // SAFETY: Valid pointers name the declared test-owned ranges; oversized and null ranges are rejected before dereference.
            unsafe { ez_gfx_counter_buffer_acquire(20, 1, pointer, length, &raw mut indirect, 0) },
            EzGfxResult::InvalidArgument
        );
    }
}

#[test]
fn optional_and_nested_counted_strings_enforce_the_same_contract() {
    let pixels = [0_u8; 4];
    let label = b"label";
    let base = EzGfxTextureDesc {
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
    let mut texture = 0;

    for desc in [
        EzGfxTextureDesc {
            debug_label: core::ptr::null(),
            debug_label_length: 1,
            ..base
        },
        EzGfxTextureDesc {
            debug_label: label.as_ptr(),
            debug_label_length: 0,
            ..base
        },
    ] {
        assert_eq!(
            // SAFETY: Descriptor and pixel storage are live; invalid string pairs are rejected before any string read.
            unsafe {
                ez_gfx_texture_load(
                    pixels.as_ptr(),
                    pixels.len(),
                    &raw const desc,
                    &raw mut texture,
                    0,
                )
            },
            EzGfxResult::InvalidArgument
        );
    }

    let valid_desc = EzGfxTextureDesc {
        debug_label: label.as_ptr(),
        debug_label_length: label.len(),
        ..base
    };
    assert_eq!(
        // SAFETY: `label` is a readable, non-terminated exact UTF-8 range.
        unsafe {
            ez_gfx_texture_load(
                pixels.as_ptr(),
                pixels.len(),
                &raw const valid_desc,
                &raw mut texture,
                0,
            )
        },
        EzGfxResult::InvalidContext
    );

    let binding_name = b"values";
    let child_handle = 1_u64 | (1_u64 << 20) | (1_u64 << 40) | (1_u64 << 52);
    let binding = EzGfxBinding {
        name: binding_name.as_ptr(),
        name_length: binding_name.len(),
        buffer: child_handle,
        counter_buffer: 0,
        render_target: 0,
    };
    assert_eq!(
        // SAFETY: The binding and its non-terminated exact name range remain readable for the call.
        unsafe {
            ffi::ez_gfx_render_add_compute_pipeline(
                child_handle,
                1,
                1,
                1,
                &raw const binding,
                1,
                core::ptr::null(),
                0,
                0,
            )
        },
        EzGfxResult::InvalidContext
    );

    let invalid_utf8 = [0xff_u8];
    let embedded_nul = b"a\0b";
    for (name, name_length) in [
        (core::ptr::null(), 1),
        (binding_name.as_ptr(), 0),
        (
            binding_name.as_ptr(),
            ffi::EZ_GFX_MAX_BOUNDARY_BYTES.saturating_add(1),
        ),
        (invalid_utf8.as_ptr(), invalid_utf8.len()),
        (embedded_nul.as_ptr(), embedded_nul.len()),
    ] {
        let invalid_binding = EzGfxBinding {
            name,
            name_length,
            ..binding
        };
        assert_eq!(
            // SAFETY: The binding is readable; valid name pointers own their declared ranges, while null and oversized ranges are rejected before dereference.
            unsafe {
                ffi::ez_gfx_render_add_compute_pipeline(
                    child_handle,
                    1,
                    1,
                    1,
                    &raw const invalid_binding,
                    1,
                    core::ptr::null(),
                    0,
                    0,
                )
            },
            EzGfxResult::InvalidArgument
        );
    }
}
