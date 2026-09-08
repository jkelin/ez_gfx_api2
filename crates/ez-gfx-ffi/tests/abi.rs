//! ABI layout, validation, lifecycle, and observability contract tests.

#[path = "abi/abi_layouts.rs"]
mod abi_layouts;
#[cfg(not(target_vendor = "apple"))]
#[path = "abi/abi_uploads.rs"]
mod abi_uploads;
#[cfg(windows)]
mod common;
#[cfg(not(any(windows, target_vendor = "apple")))]
#[path = "common/headless.rs"]
mod common;

use core::{
    ffi::c_void,
    mem::{align_of, offset_of, size_of},
};
use ez_gfx_ffi as ffi;
#[allow(
    unused_imports,
    reason = "platform-specific tests consume different ABI symbols"
)]
use ez_gfx_ffi::{
    EzGfxAdapterClass, EzGfxAdapterDesc, EzGfxAdapterInfo, EzGfxBackendContextDesc, EzGfxBinding,
    EzGfxByteBuffer, EzGfxContextDesc, EzGfxDiagnostic, EzGfxDrawIndexedCommand, EzGfxDynamicState,
    EzGfxHandleParts, EzGfxRenderTargetDesc, EzGfxRenderTargetFormat, EzGfxRenderTargetUsage,
    EzGfxResult, EzGfxRuntimeRecord, EzGfxShaderDesc, EzGfxSurfaceDesc, EzGfxTextureDesc,
    EzGfxUploadEvent, ez_gfx_acquire_indirect, ez_gfx_adapter_count, ez_gfx_adapters_query,
    ez_gfx_begin_render_target, ez_gfx_context_create, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_wait_idle, ez_gfx_finish_render, ez_gfx_frame_begin,
    ez_gfx_frame_readback, ez_gfx_frame_submit, ez_gfx_graph_enqueue_texture_readback,
    ez_gfx_handle_inspect, ez_gfx_index_allocation_get_range, ez_gfx_index_allocation_remove,
    ez_gfx_index_heap_create, ez_gfx_index_heap_destroy, ez_gfx_indirect_publish_compute_count,
    ez_gfx_indirect_release, ez_gfx_indirect_write_draws, ez_gfx_poll_diagnostic,
    ez_gfx_poll_runtime_event, ez_gfx_poll_upload_event, ez_gfx_render_target_create,
    ez_gfx_render_target_destroy, ez_gfx_render_target_get_clear, ez_gfx_render_target_get_extent,
    ez_gfx_render_target_get_format, ez_gfx_render_target_probe_format, ez_gfx_semantic_id,
    ez_gfx_shader_load_artifact, ez_gfx_structured_acquire, ez_gfx_structured_release,
    ez_gfx_structured_write, ez_gfx_texture_get_binding, ez_gfx_texture_get_residency,
    ez_gfx_texture_load, ez_gfx_texture_unload, ez_gfx_vertex_allocation_get_range,
    ez_gfx_vertex_allocation_remove, ez_gfx_vertex_heap_create, ez_gfx_vertex_heap_destroy,
    ez_gfx_vertex_upload, ez_gfx_vertex_upload_indices,
};

#[test]
fn render_target_codes_are_stable() {
    assert_eq!(
        [
            EzGfxRenderTargetFormat::Rgba8Unorm as u8,
            EzGfxRenderTargetFormat::Bgra8Srgb as u8,
            EzGfxRenderTargetFormat::Rgba16Float as u8,
            EzGfxRenderTargetFormat::Depth32Float as u8,
            EzGfxRenderTargetFormat::Bc7Unorm as u8,
            EzGfxRenderTargetFormat::Astc4x4Unorm as u8,
        ],
        [1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        [
            EzGfxRenderTargetUsage::Color as u8,
            EzGfxRenderTargetUsage::Depth as u8,
            EzGfxRenderTargetUsage::Storage as u8,
            EzGfxRenderTargetUsage::Sampled as u8,
        ],
        [0, 1, 2, 3]
    );
}
#[test]
fn adapter_codes_are_stable() {
    assert_eq!(
        [
            EzGfxAdapterClass::Software as u8,
            EzGfxAdapterClass::Other as u8,
            EzGfxAdapterClass::Integrated as u8,
            EzGfxAdapterClass::Discrete as u8,
        ],
        [0, 1, 2, 3]
    );
}

#[test]
fn headless_platform_rejects_non_vulkan_backends_before_native_calls() {
    // Platform code 3 is headless (ABI 27): Vulkan-only with no native handles.
    // Rejection happens in descriptor validation before any native call, on every host.
    for backend in [2, 3] {
        let desc = EzGfxBackendContextDesc {
            enable_debug: 0,
            enable_validation: 0,
            surface_platform: 3,
            backend,
            texture_decode_workers: 0,
            adapter_count: 0,
            adapter: core::ptr::null(),
        };
        let mut context = 0;
        assert_eq!(
            {
                // SAFETY: descriptor and output storage are live and aligned through the call.
                unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
            },
            EzGfxResult::InvalidArgument,
            "backend={backend}"
        );
        assert_eq!(context, 0);
    }
    // Unknown platform codes fail closed the same way.
    let unknown = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 9,
        backend: 1,
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: descriptor and output storage are live and aligned through the call.
            unsafe { ez_gfx_context_create_backend(&raw const unknown, &raw mut context) }
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(context, 0);
}

#[test]
fn adapter_selectors_reject_mismatched_pairs_before_delegating() {
    let valid = EzGfxAdapterDesc {
        stable_id: [0xA5; 16],
        allow_software: 0,
    };
    let base = EzGfxContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
    let mut context = 99;

    // A nonzero count with a null selector fails before any native call.
    let missing = EzGfxContextDesc {
        adapter_count: 1,
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor names live test-owned storage; the null selector intentionally exercises checked rejection.
        unsafe { ez_gfx_context_create(&raw const missing, &raw mut context) },
        EzGfxResult::InvalidArgument
    );
    // A null count with a non-null selector fails the same way.
    let dangling = EzGfxContextDesc {
        adapter: &raw const valid,
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor and selector name live test-owned storage through the call.
        unsafe { ez_gfx_context_create(&raw const dangling, &raw mut context) },
        EzGfxResult::InvalidArgument
    );
    // Counts above one, non-boolean software policy, and the all-zero
    // identity (which can never enter a catalog) fail without delegating.
    let too_many = EzGfxContextDesc {
        adapter_count: 2,
        adapter: &raw const valid,
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor and selector name live test-owned storage through the call.
        unsafe { ez_gfx_context_create(&raw const too_many, &raw mut context) },
        EzGfxResult::InvalidArgument
    );
    let permissive = EzGfxAdapterDesc {
        allow_software: 2,
        ..valid
    };
    let bad_policy = EzGfxContextDesc {
        adapter_count: 1,
        adapter: &raw const permissive,
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor and selector name live test-owned storage through the call.
        unsafe { ez_gfx_context_create(&raw const bad_policy, &raw mut context) },
        EzGfxResult::InvalidArgument
    );
    let zero = EzGfxAdapterDesc {
        stable_id: [0; 16],
        ..valid
    };
    let zero_id = EzGfxContextDesc {
        adapter_count: 1,
        adapter: &raw const zero,
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor and selector name live test-owned storage through the call.
        unsafe { ez_gfx_context_create(&raw const zero_id, &raw mut context) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(context, 99);
}

#[test]
fn adapter_enumeration_reports_stable_diagnostics() {
    let mut total = 0;
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe { ez_gfx_adapter_count(core::ptr::null_mut()) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe { ez_gfx_adapter_count(&raw mut total) },
        EzGfxResult::Ok
    );

    // Policy bytes outside zero-or-one fail; the written-output pointer is
    // always required.
    let mut written = 0;
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe { ez_gfx_adapters_query(2, core::ptr::null_mut(), 0, &raw mut written) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null outputs intentionally exercise checked rejection.
        unsafe { ez_gfx_adapters_query(0, core::ptr::null_mut(), 0, core::ptr::null_mut()) },
        EzGfxResult::InvalidArgument
    );
    // A nonzero capacity with a null buffer fails; a null buffer with a
    // zero capacity queries the total instead.
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe { ez_gfx_adapters_query(0, core::ptr::null_mut(), 1, &raw mut written) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe { ez_gfx_adapters_query(0, core::ptr::null_mut(), 0, &raw mut written) },
        EzGfxResult::Ok
    );
    assert_eq!(written, total);

    let blank = EzGfxAdapterInfo {
        stable_id: [0; 16],
        backend: 0,
        adapter_class: 0,
        admitted: 0,
        software_rejected: 0,
        error_count: 0,
    };
    let mut infos = vec![blank; total as usize];
    assert_eq!(
        // SAFETY: The buffer names exactly `capacity` writable aligned entries; the written output is live and aligned.
        unsafe { ez_gfx_adapters_query(0, infos.as_mut_ptr(), total, &raw mut written,) },
        EzGfxResult::Ok
    );
    assert_eq!(written, total);
    let mut identities = std::collections::HashSet::new();
    for info in &infos {
        // Stable identities are nonzero catalog keys and unique per backend.
        assert_ne!(info.stable_id, [0; 16]);
        assert!(identities.insert((info.backend, info.stable_id)));
        assert!((1..=3).contains(&info.backend));
        assert!(info.adapter_class <= 3);
        assert!(info.admitted <= 1);
        assert!(info.software_rejected <= 1);
        if info.admitted == 1 {
            assert_eq!(info.software_rejected, 0);
            assert_eq!(info.error_count, 0);
        }
    }
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "canary arrays verify untouched outputs with exact sentinel equality"
)]
fn render_target_create_rejects_invalid_ranges_before_delegating() {
    let name = b"target";
    let candidates = [1_u8];
    let base = EzGfxRenderTargetDesc {
        name: name.as_ptr(),
        name_length: name.len(),
        usage: 0,
        relative_scale: 1.0,
        samples: 1,
        candidate_formats: candidates.as_ptr(),
        candidate_count: u32::try_from(candidates.len()).expect("test candidate count fits u32"),
        sampleable: 1,
        use_clear: 1,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let mut target = 7;

    // Null descriptors and outputs fail before any read; outputs stay untouched.
    assert_eq!(
        // SAFETY: Null descriptor intentionally exercises checked rejection; output storage is live and aligned.
        unsafe { ez_gfx_render_target_create(core::ptr::null(), 64, 64, &raw mut target, 0) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null output intentionally exercises checked rejection; the descriptor names live test-owned ranges.
        unsafe { ez_gfx_render_target_create(&raw const base, 64, 64, core::ptr::null_mut(), 0) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(target, 7);

    // Zero extents fail before delegation.
    for (width_in, height_in) in [(0, 64), (64, 0)] {
        assert_eq!(
            // SAFETY: The descriptor names live test-owned ranges; zero extents are rejected before any native call.
            unsafe {
                ez_gfx_render_target_create(
                    &raw const base,
                    width_in,
                    height_in,
                    &raw mut target,
                    0,
                )
            },
            EzGfxResult::InvalidArgument
        );
    }

    // Unknown enum codes, non-boolean flags, and malformed count pairs fail closed.
    for desc in [
        EzGfxRenderTargetDesc { usage: 4, ..base },
        EzGfxRenderTargetDesc { samples: 3, ..base },
        EzGfxRenderTargetDesc {
            sampleable: 2,
            ..base
        },
        EzGfxRenderTargetDesc {
            use_clear: 2,
            ..base
        },
        EzGfxRenderTargetDesc {
            relative_scale: f32::NAN,
            ..base
        },
        EzGfxRenderTargetDesc {
            candidate_formats: core::ptr::null(),
            ..base
        },
        EzGfxRenderTargetDesc {
            candidate_count: 0,
            ..base
        },
        EzGfxRenderTargetDesc {
            candidate_count: 17,
            ..base
        },
        EzGfxRenderTargetDesc {
            name: core::ptr::null(),
            name_length: 0,
            ..base
        },
    ] {
        assert_eq!(
            // SAFETY: Live pointers name the declared test-owned ranges; invalid fields are rejected before delegation.
            unsafe { ez_gfx_render_target_create(&raw const desc, 64, 64, &raw mut target, 0) },
            EzGfxResult::InvalidArgument
        );
    }
    assert_eq!(target, 7);

    // An unknown candidate code fails without touching the output.
    let bad_code = [7_u8];
    let bad_candidate = EzGfxRenderTargetDesc {
        candidate_formats: bad_code.as_ptr(),
        ..base
    };
    assert_eq!(
        // SAFETY: The candidate range is live; the unknown code is rejected before delegation.
        unsafe {
            ez_gfx_render_target_create(&raw const bad_candidate, 64, 64, &raw mut target, 0)
        },
        EzGfxResult::InvalidArgument
    );

    // A non-finite stored clear fails before delegation.
    let bad_clear = EzGfxRenderTargetDesc {
        clear_color: [f32::INFINITY, 0.0, 0.0, 1.0],
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor names live test-owned ranges; the clear is rejected before delegation.
        unsafe { ez_gfx_render_target_create(&raw const bad_clear, 64, 64, &raw mut target, 0) },
        EzGfxResult::InvalidArgument
    );

    // A well-formed descriptor reaches the safe layer, which rejects the null context.
    assert_eq!(
        // SAFETY: The descriptor names live test-owned ranges with valid fields.
        unsafe { ez_gfx_render_target_create(&raw const base, 64, 64, &raw mut target, 0) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(target, 7);

    // Non-color usage passes FFI validation; the null context fails first.
    // `Unsupported` mapping is pinned safe-side and in the hidden-GPU test below.
    let depth = EzGfxRenderTargetDesc {
        usage: 1,
        use_clear: 0,
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor names live test-owned ranges; the null context fails delegation.
        unsafe { ez_gfx_render_target_create(&raw const depth, 64, 64, &raw mut target, 0) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(target, 7);
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "canary arrays verify untouched outputs with exact sentinel equality"
)]
fn render_target_queries_probe_and_begin_validate_handles() {
    let mut format = 9;
    let mut width = 11;
    let mut height = 13;
    let mut use_clear = 15;
    let mut color = [17.0, 19.0, 23.0, 29.0];

    assert_eq!(
        // SAFETY: Null format output intentionally exercises checked rejection.
        unsafe { ez_gfx_render_target_get_format(0, core::ptr::null_mut(), 0) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null extent outputs intentionally exercise checked rejection.
        unsafe { ez_gfx_render_target_get_extent(0, core::ptr::null_mut(), &raw mut height, 0) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null clear outputs intentionally exercise checked rejection.
        unsafe { ez_gfx_render_target_get_clear(0, &raw mut use_clear, core::ptr::null_mut(), 0) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(format, 9);
    assert_eq!(height, 13);
    assert_eq!(use_clear, 15);
    assert_eq!(color, [17.0, 19.0, 23.0, 29.0]);

    // Malformed handles fail before any context access.
    assert_eq!(
        // SAFETY: Outputs are live and aligned; the zero handles fail validation first.
        unsafe { ez_gfx_render_target_get_format(0, &raw mut format, 0) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        // SAFETY: Outputs are live and aligned; the zero handles fail validation first.
        unsafe { ez_gfx_render_target_get_extent(0, &raw mut width, &raw mut height, 0) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        // SAFETY: Outputs are live and aligned; the zero handles fail validation first.
        unsafe { ez_gfx_render_target_get_clear(0, &raw mut use_clear, color.as_mut_ptr(), 0) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        ez_gfx_render_target_probe_format(7, 1, 0),
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        ez_gfx_render_target_probe_format(1, 3, 0),
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        ez_gfx_render_target_probe_format(1, 1, 0),
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        ez_gfx_begin_render_target(0, 0),
        EzGfxResult::InvalidContext
    );
    // Destroy stays infallible over garbage handles.
    ez_gfx_render_target_destroy(0, 0);
    assert_eq!(format, 9);
    assert_eq!((width, height), (11, 13));
    assert_eq!(use_clear, 15);
    assert_eq!(color, [17.0, 19.0, 23.0, 29.0]);
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
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
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

#[cfg(not(target_vendor = "apple"))]
#[test]
fn context_lifecycle_rejects_cross_thread_destroy_and_invalidates_destroyed_handle() {
    let desc = EzGfxContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        // Win32 contexts need a Win32 host; every other non-Apple host runs headless.
        surface_platform: if cfg!(windows) { 0 } else { 3 },
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
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
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::NotReady);
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
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_begin(context), EzGfxResult::Ok);
    let name = b"vertices";
    let mut structured = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_structured_acquire(
                    16,
                    4,
                    name.as_ptr(),
                    name.len(),
                    &raw mut structured,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let bytes = [7_u8; 64];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_structured_write(structured, 0, bytes.as_ptr().cast(), 4, 16, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_structured_write(structured, 0, bytes.as_ptr().cast(), 1, 65, context) }
        },
        EzGfxResult::InvalidArgument
    );
    ez_gfx_structured_release(structured, context);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_structured_write(structured, 0, bytes.as_ptr().cast(), 4, 16, context) }
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
        debug_label_length: 0,
    };
    for desc in [
        EzGfxTextureDesc {
            destination_format: 11,
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

#[test]
fn semantic_id_enforces_canonical_name_bytes_and_boundaries() {
    let mut output = [0xa5_u8; 16];
    let invalid_utf8 = [0xff_u8];
    let embedded_nul = b"material\0albedo";
    let too_long = [b'a'; 256];

    for (name, length) in [
        (core::ptr::null(), 1),
        (b"a".as_ptr(), 0),
        (invalid_utf8.as_ptr(), invalid_utf8.len()),
        (embedded_nul.as_ptr(), embedded_nul.len()),
        (b".leading".as_ptr(), b".leading".len()),
        (b"trailing.".as_ptr(), b"trailing.".len()),
        (b"double..dot".as_ptr(), b"double..dot".len()),
        (b"1starts_with_digit".as_ptr(), b"1starts_with_digit".len()),
        (b"bad-dash".as_ptr(), b"bad-dash".len()),
        (b"caf\xc3\xa9".as_ptr(), b"caf\xc3\xa9".len()),
        (too_long.as_ptr(), too_long.len()),
    ] {
        assert_eq!(
            // SAFETY: Valid pointers own exactly the declared range; null and zero deliberately exercise validation before dereference.
            unsafe { ez_gfx_semantic_id(name, length, output.as_mut_ptr()) },
            EzGfxResult::InvalidArgument
        );
        assert_eq!(output, [0xa5; 16]);
    }

    let max_length_name = [b'a'; 255];
    for name in [b"material.albedo_2".as_slice(), max_length_name.as_slice()] {
        assert_eq!(
            // SAFETY: `name` is a live exact canonical semantic-name range and output has 16 writable bytes.
            unsafe { ez_gfx_semantic_id(name.as_ptr(), name.len(), output.as_mut_ptr()) },
            EzGfxResult::Ok
        );
        assert_ne!(output, [0xa5; 16]);
        output.fill(0xa5);
    }
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

/// Creates, queries, and destroys a ceiling-admitted multisampled target.
///
/// Adapters below 4-sample support skip the resolve path they cannot exercise;
/// no frame begins here since the caller owns the test's frame lifecycle.
#[cfg(not(target_vendor = "apple"))]
fn msaa_target_lifecycle(context: ffi::EzGfxContext, base: EzGfxRenderTargetDesc) {
    // A 4-sample RGBA8 probe follows the device ceiling; the RTX runners admit
    // it, and a multisampled target then passes the full lifecycle below.
    if ez_gfx_render_target_probe_format(1, 4, context) == EzGfxResult::Ok {
        let msaa_desc = EzGfxRenderTargetDesc { samples: 4, ..base };
        let mut msaa = 0;
        assert_eq!(
            // SAFETY: The descriptor names live test-owned ranges; outputs are live and aligned.
            unsafe {
                ez_gfx_render_target_create(&raw const msaa_desc, 64, 64, &raw mut msaa, context)
            },
            EzGfxResult::Ok
        );
        assert_ne!(msaa, 0);
        let mut msaa_format = 0;
        assert_eq!(
            // SAFETY: Output storage is live and aligned through the call.
            unsafe { ez_gfx_render_target_get_format(msaa, &raw mut msaa_format, context) },
            EzGfxResult::Ok
        );
        assert_eq!(msaa_format, 1);
        // No frame begin here: the single begun frame below owns the test's
        // frame lifecycle, and a second begin would fail it.
        ez_gfx_render_target_destroy(msaa, context);
        assert_eq!(
            // SAFETY: Output storage is live and aligned; the destroyed handle fails first.
            unsafe { ez_gfx_render_target_get_format(msaa, &raw mut msaa_format, context) },
            EzGfxResult::InvalidArgument
        );
    }
}
#[allow(
    clippy::float_cmp,
    reason = "stored clear values round-trip exactly; no arithmetic is compared"
)]
#[cfg(not(target_vendor = "apple"))]
fn render_target_lifecycle_queries_probe_and_begin_on_hidden_context(backend: u8) {
    use common::TestContext;

    let native = TestContext::create(backend);
    let name = b"rt";
    let candidates = [1_u8];
    let base = EzGfxRenderTargetDesc {
        name: name.as_ptr(),
        name_length: name.len(),
        usage: 0,
        relative_scale: 1.0,
        samples: 1,
        candidate_formats: candidates.as_ptr(),
        candidate_count: u32::try_from(candidates.len()).expect("test candidate count fits u32"),
        sampleable: 1,
        use_clear: 1,
        clear_color: [0.25, 0.5, 0.75, 1.0],
    };
    let mut target = 0;
    assert_eq!(
        // SAFETY: The descriptor names live test-owned ranges; outputs are live and aligned.
        unsafe {
            ez_gfx_render_target_create(&raw const base, 64, 64, &raw mut target, native.context)
        },
        EzGfxResult::Ok
    );
    assert_ne!(target, 0);

    let mut format = 0;
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe { ez_gfx_render_target_get_format(target, &raw mut format, native.context) },
        EzGfxResult::Ok
    );
    assert_eq!(format, 1);
    let (mut width, mut height) = (0, 0);
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe {
            ez_gfx_render_target_get_extent(target, &raw mut width, &raw mut height, native.context)
        },
        EzGfxResult::Ok
    );
    assert_eq!((width, height), (64, 64));
    let (mut use_clear, mut color) = (0, [0.0; 4]);
    assert_eq!(
        // SAFETY: `out_color` names four writable aligned floats through the call.
        unsafe {
            ez_gfx_render_target_get_clear(
                target,
                &raw mut use_clear,
                color.as_mut_ptr(),
                native.context,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(use_clear, 1);
    assert_eq!(color, [0.25, 0.5, 0.75, 1.0]);

    // Targets without a stored clear report zeros.
    let mut plain = 0;
    let no_clear = EzGfxRenderTargetDesc {
        use_clear: 0,
        ..base
    };
    assert_eq!(
        // SAFETY: The descriptor names live test-owned ranges; outputs are live and aligned.
        unsafe {
            ez_gfx_render_target_create(&raw const no_clear, 32, 32, &raw mut plain, native.context)
        },
        EzGfxResult::Ok
    );
    let (mut plain_flag, mut plain_color) = (9, [9.0; 4]);
    assert_eq!(
        // SAFETY: `out_color` names four writable aligned floats through the call.
        unsafe {
            ez_gfx_render_target_get_clear(
                plain,
                &raw mut plain_flag,
                plain_color.as_mut_ptr(),
                native.context,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(plain_flag, 0);
    assert_eq!(plain_color, [0.0; 4]);
    ez_gfx_render_target_destroy(plain, native.context);

    // Probing resolves the same declarations creation would admit.
    assert_eq!(
        ez_gfx_render_target_probe_format(1, 1, native.context),
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_render_target_probe_format(4, 1, native.context),
        EzGfxResult::Unsupported
    );

    // A 4-sample RGBA8 probe follows the device ceiling; the RTX runners admit
    // it, and a multisampled target then passes the full lifecycle below.
    msaa_target_lifecycle(native.context, base);

    // Depth and storage creation stay unsupported with explicit errors.
    let mut rejected = 0;
    for desc in [
        EzGfxRenderTargetDesc {
            usage: 1,
            use_clear: 0,
            ..base
        },
        EzGfxRenderTargetDesc {
            usage: 2,
            use_clear: 0,
            ..base
        },
    ] {
        assert_eq!(
            // SAFETY: The descriptor names live test-owned ranges; outputs are live and aligned.
            unsafe {
                ez_gfx_render_target_create(
                    &raw const desc,
                    64,
                    64,
                    &raw mut rejected,
                    native.context,
                )
            },
            EzGfxResult::Unsupported
        );
    }
    assert_eq!(rejected, 0);

    // Binding the target for a frame succeeds; destroying it invalidates queries.
    assert_eq!(
        ez_gfx_begin_render_target(target, native.context),
        EzGfxResult::Ok
    );
    ez_gfx_render_target_destroy(target, native.context);
    assert_eq!(
        // SAFETY: Output storage is live and aligned; the destroyed handle fails first.
        unsafe { ez_gfx_render_target_get_format(target, &raw mut format, native.context) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        ez_gfx_begin_render_target(target, native.context),
        EzGfxResult::InvalidContext
    );
    drop(native);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_render_target_lifecycle_queries_probe_and_begin() {
    render_target_lifecycle_queries_probe_and_begin_on_hidden_context(1);
}

#[cfg(windows)]
#[test]
fn dx12_render_target_lifecycle_queries_probe_and_begin() {
    render_target_lifecycle_queries_probe_and_begin_on_hidden_context(2);
}
#[cfg(not(target_vendor = "apple"))]
#[test]
fn explicit_adapter_selection_creates_and_rejects_hidden_contexts() {
    // No surface is created, shown, or activated by this test.
    let mut total = 0;
    assert_eq!(
        // SAFETY: Output storage is live and aligned through the call.
        unsafe { ez_gfx_adapter_count(&raw mut total) },
        EzGfxResult::Ok
    );
    let blank = EzGfxAdapterInfo {
        stable_id: [0; 16],
        backend: 0,
        adapter_class: 0,
        admitted: 0,
        software_rejected: 0,
        error_count: 0,
    };
    let mut infos = vec![blank; total as usize];
    let mut written = 0;
    assert_eq!(
        // SAFETY: The buffer names exactly `capacity` writable aligned entries; the written output is live and aligned.
        unsafe { ez_gfx_adapters_query(1, infos.as_mut_ptr(), total, &raw mut written) },
        EzGfxResult::Ok
    );
    assert_eq!(written, total);
    let wanted = infos
        .iter()
        .find(|info| info.backend == 1 && info.admitted == 1)
        .expect("at least one admissible Vulkan adapter")
        .stable_id;

    let request = EzGfxAdapterDesc {
        stable_id: wanted,
        allow_software: 1,
    };
    let desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        // Win32 contexts need a Win32 host; every other non-Apple host runs headless.
        surface_platform: if cfg!(windows) { 0 } else { 3 },
        backend: 1,
        texture_decode_workers: 0,
        adapter_count: 1,
        adapter: &raw const request,
    };
    let mut context = 0;
    assert_eq!(
        // SAFETY: The descriptor and selector name live test-owned storage; the output is live and aligned.
        unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) },
        EzGfxResult::Ok
    );
    assert_ne!(context, 0);
    ez_gfx_context_destroy(context);

    // An unknown stable identity fails as a caller error without a device.
    let unknown = EzGfxAdapterDesc {
        stable_id: [0xA5; 16],
        allow_software: 0,
    };
    let rejected = EzGfxBackendContextDesc {
        adapter_count: 1,
        adapter: &raw const unknown,
        ..desc
    };
    let mut missing = 0;
    assert_eq!(
        // SAFETY: The descriptor and selector name live test-owned storage; the output is live and aligned.
        unsafe { ez_gfx_context_create_backend(&raw const rejected, &raw mut missing) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(missing, 0);
}
