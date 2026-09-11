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
        unsafe { ez_gfx_render_target_create(0, core::ptr::null(), 64, 64, &raw mut target) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null output intentionally exercises checked rejection; the descriptor names live test-owned ranges.
        unsafe { ez_gfx_render_target_create(0, &raw const base, 64, 64, core::ptr::null_mut()) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(target, 7);

    // Zero extents fail before delegation.
    for (width_in, height_in) in [(0, 64), (64, 0)] {
        assert_eq!(
            // SAFETY: The descriptor names live test-owned ranges; zero extents are rejected before any native call.
            unsafe {
                ez_gfx_render_target_create(
                    0,
                    &raw const base,
                    width_in,
                    height_in,
                    &raw mut target,
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
            unsafe { ez_gfx_render_target_create(0, &raw const desc, 64, 64, &raw mut target) },
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
            ez_gfx_render_target_create(0, &raw const bad_candidate, 64, 64, &raw mut target)
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
        unsafe { ez_gfx_render_target_create(0, &raw const bad_clear, 64, 64, &raw mut target) },
        EzGfxResult::InvalidArgument
    );

    // A well-formed descriptor reaches the safe layer, which rejects the null context.
    assert_eq!(
        // SAFETY: The descriptor names live test-owned ranges with valid fields.
        unsafe { ez_gfx_render_target_create(0, &raw const base, 64, 64, &raw mut target) },
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
        unsafe { ez_gfx_render_target_create(0, &raw const depth, 64, 64, &raw mut target) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(target, 7);
}

#[test]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "the test intentionally constructs misaligned FFI pointers"
)]
fn render_target_boundaries_reject_misaligned_typed_pointers() {
    fn offset_for_misalignment(address: usize, alignment: usize) -> usize {
        (alignment - address % alignment) % alignment + 1
    }

    let mut descriptor_bytes =
        vec![0_u8; size_of::<EzGfxRenderTargetDesc>() + align_of::<EzGfxRenderTargetDesc>()];
    let descriptor_offset = offset_for_misalignment(
        descriptor_bytes.as_ptr().addr(),
        align_of::<EzGfxRenderTargetDesc>(),
    );
    // SAFETY: The computed pointer remains in the allocation and is rejected before reading.
    let misaligned_descriptor = unsafe {
        descriptor_bytes
            .as_mut_ptr()
            .add(descriptor_offset)
            .cast::<EzGfxRenderTargetDesc>()
    };
    assert!(!misaligned_descriptor.is_aligned());
    let mut target = 7_u64;
    assert_eq!(
        // SAFETY: The deliberately misaligned descriptor is rejected before dereference.
        unsafe { ez_gfx_render_target_create(0, misaligned_descriptor, 1, 1, &raw mut target,) },
        EzGfxResult::InvalidArgument
    );

    let name = b"target";
    let candidates = [1_u8];
    let descriptor = EzGfxRenderTargetDesc {
        name: name.as_ptr(),
        name_length: name.len(),
        usage: 0,
        relative_scale: 1.0,
        samples: 1,
        candidate_formats: candidates.as_ptr(),
        candidate_count: 1,
        sampleable: 1,
        use_clear: 0,
        clear_color: [0.0; 4],
    };
    let mut target_bytes = vec![0_u8; size_of::<u64>() + align_of::<u64>()];
    let target_offset = offset_for_misalignment(target_bytes.as_ptr().addr(), align_of::<u64>());
    // SAFETY: The computed pointer remains in the allocation and is rejected before writing.
    let misaligned_target = unsafe { target_bytes.as_mut_ptr().add(target_offset).cast::<u64>() };
    assert!(!misaligned_target.is_aligned());
    assert_eq!(
        // SAFETY: The deliberately misaligned output is rejected before writing.
        unsafe { ez_gfx_render_target_create(0, &raw const descriptor, 1, 1, misaligned_target) },
        EzGfxResult::InvalidArgument
    );

    let mut u32_bytes = vec![0_u8; size_of::<u32>() + align_of::<u32>()];
    let u32_offset = offset_for_misalignment(u32_bytes.as_ptr().addr(), align_of::<u32>());
    // SAFETY: The computed pointer remains in the allocation and is rejected before writing.
    let misaligned_u32 = unsafe { u32_bytes.as_mut_ptr().add(u32_offset).cast::<u32>() };
    assert!(!misaligned_u32.is_aligned());
    let mut aligned_u32 = 0_u32;
    for (width, height) in [
        (misaligned_u32, &raw mut aligned_u32),
        (&raw mut aligned_u32, misaligned_u32),
    ] {
        assert_eq!(
            // SAFETY: The deliberately misaligned output is rejected before writing.
            unsafe { ez_gfx_render_target_get_extent(0, 0, width, height) },
            EzGfxResult::InvalidArgument
        );
    }

    let mut color_bytes = vec![0_u8; 4 * size_of::<f32>() + align_of::<f32>()];
    let color_offset = offset_for_misalignment(color_bytes.as_ptr().addr(), align_of::<f32>());
    // SAFETY: The computed range remains in the allocation and is rejected before writing.
    let misaligned_color = unsafe { color_bytes.as_mut_ptr().add(color_offset).cast::<f32>() };
    assert!(!misaligned_color.is_aligned());
    let mut use_clear = 0_u8;
    assert_eq!(
        // SAFETY: The deliberately misaligned color output is rejected before writing.
        unsafe { ez_gfx_render_target_get_clear(0, 0, &raw mut use_clear, misaligned_color) },
        EzGfxResult::InvalidArgument
    );
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
        unsafe { ez_gfx_render_target_get_format(0, 0, core::ptr::null_mut()) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null extent outputs intentionally exercise checked rejection.
        unsafe { ez_gfx_render_target_get_extent(0, 0, core::ptr::null_mut(), &raw mut height) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Null clear outputs intentionally exercise checked rejection.
        unsafe { ez_gfx_render_target_get_clear(0, 0, &raw mut use_clear, core::ptr::null_mut()) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(format, 9);
    assert_eq!(height, 13);
    assert_eq!(use_clear, 15);
    assert_eq!(color, [17.0, 19.0, 23.0, 29.0]);

    // Malformed handles fail before any context access.
    assert_eq!(
        // SAFETY: Outputs are live and aligned; the zero handles fail validation first.
        unsafe { ez_gfx_render_target_get_format(0, 0, &raw mut format) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        // SAFETY: Outputs are live and aligned; the zero handles fail validation first.
        unsafe { ez_gfx_render_target_get_extent(0, 0, &raw mut width, &raw mut height) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        // SAFETY: Outputs are live and aligned; the zero handles fail validation first.
        unsafe { ez_gfx_render_target_get_clear(0, 0, &raw mut use_clear, color.as_mut_ptr()) },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        ez_gfx_render_target_probe_format(0, 7, 1),
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        ez_gfx_render_target_probe_format(0, 1, 3),
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        ez_gfx_render_target_probe_format(0, 1, 1),
        EzGfxResult::InvalidContext
    );
    let mut frame = 0;
    assert_eq!(
        // SAFETY: The frame output is live and aligned; the zero handles fail validation.
        unsafe { ez_gfx_render_target_frame_begin(0, 0, &raw mut frame) },
        EzGfxResult::InvalidContext
    );
    // Destroy stays infallible over garbage handles.
    ez_gfx_render_target_destroy(0, 0);
    assert_eq!(format, 9);
    assert_eq!((width, height), (11, 13));
    assert_eq!(use_clear, 15);
    assert_eq!(color, [17.0, 19.0, 23.0, 29.0]);
}
