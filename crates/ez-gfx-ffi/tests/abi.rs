//! ABI layout, validation, lifecycle, and observability contract tests.

#[cfg(windows)]
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
    EZ_GFX_ABI_VERSION, EzGfxAdapterClass, EzGfxAdapterDesc, EzGfxAdapterInfo,
    EzGfxBackendContextDesc, EzGfxBinding, EzGfxByteBuffer, EzGfxContextDesc, EzGfxDiagnostic,
    EzGfxDrawIndexedCommand, EzGfxDynamicState, EzGfxHandleParts, EzGfxRenderTargetDesc,
    EzGfxRenderTargetFormat, EzGfxRenderTargetUsage, EzGfxResult, EzGfxRuntimeRecord,
    EzGfxShaderDesc, EzGfxSurfaceDesc, EzGfxTextureDesc, EzGfxTextureError, ez_gfx_abi_version,
    ez_gfx_acquire_indirect, ez_gfx_adapter_count, ez_gfx_adapters_query,
    ez_gfx_begin_render_target, ez_gfx_context_create, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_wait_idle, ez_gfx_finish_render, ez_gfx_frame_begin,
    ez_gfx_frame_readback, ez_gfx_frame_submit, ez_gfx_graph_enqueue_texture_readback,
    ez_gfx_handle_inspect, ez_gfx_index_heap_create, ez_gfx_index_heap_destroy,
    ez_gfx_indirect_release, ez_gfx_indirect_set_draw_count, ez_gfx_indirect_write_draw,
    ez_gfx_poll_diagnostic, ez_gfx_poll_runtime_event, ez_gfx_render_target_create,
    ez_gfx_render_target_destroy, ez_gfx_render_target_get_clear, ez_gfx_render_target_get_extent,
    ez_gfx_render_target_get_format, ez_gfx_render_target_probe_format, ez_gfx_semantic_id,
    ez_gfx_shader_load_artifact, ez_gfx_structured_acquire, ez_gfx_structured_release,
    ez_gfx_structured_write, ez_gfx_texture_get_binding, ez_gfx_texture_get_residency,
    ez_gfx_texture_load, ez_gfx_texture_poll, ez_gfx_texture_unload, ez_gfx_vertex_heap_create,
    ez_gfx_vertex_heap_destroy, ez_gfx_vertex_upload, ez_gfx_vertex_upload_indices,
};

#[test]
fn status_values_and_abi_version_are_stable() {
    assert_eq!(ez_gfx_abi_version(), EZ_GFX_ABI_VERSION);
    assert_eq!(EzGfxResult::Ok as u8, 0);
    assert_eq!(EzGfxResult::InvalidArgument as u8, 1);
    assert_eq!(EzGfxResult::InvalidContext as u8, 2);
    assert_eq!(EzGfxResult::NativeFailure as u8, 3);
    assert_eq!(EzGfxResult::NotReady as u8, 4);
    assert_eq!(EzGfxResult::Unsupported as u8, 5);
    assert_eq!(EzGfxResult::DeviceLost as u8, 6);
    assert_eq!(EzGfxResult::QueueFull as u8, 7);
    assert_eq!(EzGfxResult::Cancelled as u8, 8);
    assert_eq!(
        [
            EzGfxTextureError::None as u8,
            EzGfxTextureError::InvalidContext as u8,
            EzGfxTextureError::InvalidArguments as u8,
            EzGfxTextureError::UnsupportedFormat as u8,
            EzGfxTextureError::OutOfTextureHandles as u8,
            EzGfxTextureError::OutOfMemory as u8,
            EzGfxTextureError::DecodeFailed as u8,
            EzGfxTextureError::VulkanFailed as u8,
            EzGfxTextureError::WorkerUnavailable as u8,
            EzGfxTextureError::NotFound as u8,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(EZ_GFX_ABI_VERSION, 26);
}

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
            offset_of!(EzGfxBinding, structured),
            offset_of!(EzGfxBinding, indirect),
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
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_texture_poll;
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
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_begin_render_target;
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_begin_render;
    let _: extern "C" fn(Handle) -> Status = ffi::ez_gfx_frame_begin;
    let _: unsafe extern "C" fn(u32, *const u8, usize, *mut Handle, Handle) -> Status =
        ffi::ez_gfx_acquire_indirect;
    let _: unsafe extern "C" fn(Handle, u32, *const EzGfxDrawIndexedCommand, Handle) -> Status =
        ffi::ez_gfx_indirect_write_draw;
    let _: extern "C" fn(Handle, u32, Handle) -> Status = ffi::ez_gfx_indirect_set_draw_count;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_indirect_release;
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
    let _: extern "C" fn(Handle, Handle) -> Status = ffi::ez_gfx_graph_enqueue_texture_readback;
    let _: extern "C" fn(Handle) -> Status = ffi::ez_gfx_frame_submit;
    let _: unsafe extern "C" fn(*mut EzGfxRuntimeRecord, *mut u8, *mut u64, Handle) -> Status =
        ffi::ez_gfx_poll_runtime_event;
    let _: unsafe extern "C" fn(*mut EzGfxDiagnostic, *mut u8, *mut u64, Handle) -> Status =
        ffi::ez_gfx_poll_diagnostic;
    let _: extern "C" fn(Handle) -> Status = ffi::ez_gfx_finish_render;
    let _: unsafe extern "C" fn(*mut u8, usize, *mut usize, Handle) -> Status =
        ffi::ez_gfx_frame_readback;
    let _: unsafe extern "C" fn(*const u8, usize, u64, u64, Handle) -> Status =
        ffi::ez_gfx_vertex_heap_create;
    let _: unsafe extern "C" fn(*const u8, usize, Handle) = ffi::ez_gfx_vertex_heap_destroy;
    let _: unsafe extern "C" fn(u64, *const u8, usize, Handle) -> Status =
        ffi::ez_gfx_index_heap_create;
    let _: extern "C" fn(Handle) = ffi::ez_gfx_index_heap_destroy;
    let _: unsafe extern "C" fn(*const c_void, u32, *mut u32, Handle) -> Status =
        ffi::ez_gfx_vertex_upload_indices;
    let _: unsafe extern "C" fn(
        *const u8,
        usize,
        *const c_void,
        u32,
        u64,
        *mut u32,
        Handle,
    ) -> Status = ffi::ez_gfx_vertex_upload;
    let _: unsafe extern "C" fn(u32, u32, *const u8, usize, *mut Handle, Handle) -> Status =
        ffi::ez_gfx_structured_acquire;
    let _: unsafe extern "C" fn(Handle, *const c_void, u64, Handle) -> Status =
        ffi::ez_gfx_structured_write;
    let _: extern "C" fn(Handle, Handle) = ffi::ez_gfx_structured_release;
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
        unsafe { ez_gfx_acquire_indirect(1, valid.as_ptr(), valid.len(), &raw mut indirect, 0) },
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
            unsafe { ez_gfx_acquire_indirect(1, pointer, length, &raw mut indirect, 0) },
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
        structured: child_handle,
        indirect: 0,
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

#[cfg(windows)]
#[test]
fn context_lifecycle_rejects_cross_thread_destroy_and_invalidates_destroyed_handle() {
    let desc = EzGfxContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
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

#[cfg(windows)]
#[test]
fn vulkan_geometry_uploads_use_real_device_buffers_and_transfer_timeline() {
    geometry_uploads_use_real_device_buffers_and_transfer_fence(1);
}

#[cfg(windows)]
#[test]
fn dx12_geometry_uploads_use_real_device_buffers_and_transfer_fence() {
    geometry_uploads_use_real_device_buffers_and_transfer_fence(2);
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

#[cfg(windows)]
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

#[cfg(windows)]
#[test]
fn vulkan_frame_uploads_indirect_compiles_graph_and_reads_back_texture() {
    frame_uploads_indirect_compiles_graph_and_reads_back_texture(1);
}

#[cfg(windows)]
#[test]
fn dx12_frame_uploads_indirect_compiles_graph_and_reads_back_texture() {
    frame_uploads_indirect_compiles_graph_and_reads_back_texture(2);
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
#[cfg(windows)]
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
#[cfg(windows)]
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

#[cfg(windows)]
#[test]
fn vulkan_render_target_lifecycle_queries_probe_and_begin() {
    render_target_lifecycle_queries_probe_and_begin_on_hidden_context(1);
}

#[cfg(windows)]
#[test]
fn dx12_render_target_lifecycle_queries_probe_and_begin() {
    render_target_lifecycle_queries_probe_and_begin_on_hidden_context(2);
}
#[cfg(windows)]
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
        unsafe { ez_gfx_adapters_query(0, infos.as_mut_ptr(), total, &raw mut written) },
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
        allow_software: 0,
    };
    let desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
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
