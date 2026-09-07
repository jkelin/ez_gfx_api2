use super::*;
use std::collections::HashSet;

#[cfg(not(target_vendor = "apple"))]
fn vulkan_options() -> Result<ContextOptions, ez_gfx_runtime::PublicApiError> {
    // Win32 contexts need a Win32 host; every other non-Apple host runs headless.
    ContextOptions::new_for_backend(0, 0, if cfg!(windows) { 0 } else { 3 }, Backend::Vulkan)
}

fn shader() -> ShaderHandle {
    ShaderHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(7, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn pipeline_layout_keys_ignore_reflection_order() {
    let first = ez_gfx_hal::ShaderBufferLayout::new(0, 3, 1, false).unwrap();
    let second = ez_gfx_hal::ShaderBufferLayout::new(0, 1, 2, true).unwrap();

    assert_eq!(
        pipeline_layout_key(&[first, second]),
        pipeline_layout_key(&[second, first])
    );
}

#[test]
fn graphics_pipeline_keys_include_state_attachment_and_texture_interface() {
    let state = DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap();
    let key = PipelineKey::Graphics {
        backend: Backend::Vulkan,
        shader: shader(),
        shader_digest: [1; 32],
        vertex_product: 0,
        vertex_entry: "vertexmain".to_owned(),
        fragment_product: 1,
        fragment_entry: "fragmentmain".to_owned(),
        texture_heap: None,
        layouts: Vec::new(),
        state,
        depth_required: false,
        color_format: 44,
        depth_format: 0,
        sample_count: 1,
    };
    let mut changed_state = key.clone();
    let PipelineKey::Graphics { state, .. } = &mut changed_state else {
        unreachable!()
    };
    state.blend = ez_gfx_hal::BlendMode::Alpha;
    let mut changed_format = key.clone();
    let PipelineKey::Graphics { color_format, .. } = &mut changed_format else {
        unreachable!()
    };
    *color_format = 50;
    let mut changed_heap = key.clone();
    let PipelineKey::Graphics { texture_heap, .. } = &mut changed_heap else {
        unreachable!()
    };
    *texture_heap = Some(ez_gfx_hal::ShaderTextureHeapLayout::new(0, 4, 16, 2, 0, 1).unwrap());

    assert_eq!(
        HashSet::from([key, changed_state, changed_format, changed_heap]).len(),
        4
    );
}

#[cfg(not(target_vendor = "apple"))]
fn texture_config() -> TextureConfig {
    TextureConfig {
        width: 1,
        height: 1,
        mip_count: 1,
        destination: TextureDestination::Rgba8Unorm,
        sampler: ez_gfx_hal::TextureSamplerDesc {
            min_filter: ez_gfx_hal::SamplerFilter::Linear,
            mag_filter: ez_gfx_hal::SamplerFilter::Linear,
            max_anisotropy: 1.0,
            address_u: ez_gfx_hal::SamplerAddressMode::Clamp,
            address_v: ez_gfx_hal::SamplerAddressMode::Clamp,
            address_w: ez_gfx_hal::SamplerAddressMode::Clamp,
        },
    }
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn texture_admission_is_nonblocking_and_pending_cancellation_invalidates_the_handle() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let gate = Arc::new(std::sync::Barrier::new(2));
    with_context_mut(context, |state| {
        state.async_textures.decode_gate = Some(gate.clone());
        Ok(())
    })
    .unwrap();

    let texture = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1, 2, 3, 4],
        false,
        &texture_config(),
    )
    .unwrap();
    assert_eq!(poll_texture_load(context, texture), EzGfxResult::NotReady);

    gate.wait();
    assert_eq!(cancel_texture_load(context, texture), EzGfxResult::Ok);
    assert_eq!(
        poll_texture_load(context, texture),
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        cancel_texture_load(context, texture),
        EzGfxResult::InvalidArgument
    );
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
}

#[test]
fn texture_region_validation_and_update_backpressure_are_stable() {
    let bytes = [0_u8; 16];
    let valid = TextureRegion {
        mip_level: 0,
        x: 1,
        y: 1,
        width: 2,
        height: 2,
        bytes: &bytes,
    };
    assert_eq!(
        validate_texture_update(TextureFormat::Rgba8Unorm, 4, 4, 3, valid),
        Ok(())
    );
    let invalid = TextureRegion { width: 3, ..valid };
    assert_eq!(
        validate_texture_update(TextureFormat::Rgba8Unorm, 4, 4, 3, invalid),
        Err(EzGfxResult::InvalidArgument)
    );
    assert_eq!(
        map_texture_update_error(ez_gfx_hal::AllocationError::OutOfMemory),
        EzGfxResult::QueueFull
    );
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn first_coarse_publication_records_handoff_telemetry_once() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let texture = with_context_mut(context, |state| {
        let packed = state
            .identity
            .insert(ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let texture = TextureHandle::from_packed(packed).map_err(|_| EzGfxResult::NativeFailure)?;
        state.texture_ready.insert(
            texture,
            CompletionToken::new(QueueKind::TextureTransfer, 3)
                .map_err(|_| EzGfxResult::NativeFailure)?,
        );
        state.texture_handoffs.insert(
            texture,
            Instant::now()
                .checked_sub(std::time::Duration::from_micros(10))
                .expect("ten microseconds fits in the monotonic clock"),
        );

        record_texture_ready(state, texture, 2);
        assert!(state.texture_ready.contains_key(&texture));
        record_texture_ready(state, texture, 3);
        // GPU completion alone is insufficient while a frame still prevents publication.
        assert!(state.texture_ready.contains_key(&texture));
        assert_eq!(
            state
                .texture_telemetry
                .snapshot()
                .handoff_latency_microseconds,
            0
        );
        state.texture_published_mips.insert(texture, 1);
        record_texture_ready(state, texture, 3);
        assert!(!state.texture_ready.contains_key(&texture));
        Ok(texture)
    })
    .unwrap();

    let snapshot = texture_upload_telemetry(context).unwrap();
    assert!(snapshot.handoff_latency_microseconds >= 10);
    with_context_mut(context, |state| {
        record_texture_ready(state, texture, u64::MAX);
        assert_eq!(
            state
                .texture_telemetry
                .snapshot()
                .handoff_latency_microseconds,
            snapshot.handoff_latency_microseconds
        );
        Ok(())
    })
    .unwrap();
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
}

#[cfg(windows)]
fn dx12_context() -> ContextHandle {
    create_context(ContextOptions::new_for_backend(0, 0, 0, Backend::Dx12).unwrap()).unwrap()
}
#[cfg(not(target_vendor = "apple"))]
#[test]
fn destroy_context_before_device_initialization_is_successful_and_terminal() {
    let context = create_context(vulkan_options().unwrap()).unwrap();

    assert_eq!(destroy_context(context), EzGfxResult::Ok);
    assert_eq!(wait_idle(context), EzGfxResult::InvalidContext);
    assert_eq!(destroy_context(context), EzGfxResult::InvalidContext);
}

#[test]
fn decode_worker_topology_defaults_and_honors_explicit_counts() {
    // Zero preserves `available_parallelism - 1`; an explicit count pins the pool.
    // This exercises pool construction directly so every host covers the topology.
    let expected_default = std::thread::available_parallelism()
        .map_or(2, usize::from)
        .saturating_sub(1)
        .max(1);
    let default_state = AsyncTextureState::new_with_workers(0).unwrap();
    assert_eq!(default_state.worker_count(), expected_default);
    let explicit_state = AsyncTextureState::new_with_workers(2).unwrap();
    assert_eq!(explicit_state.worker_count(), 2);
}

#[test]
fn decode_worker_topology_rejects_absurd_counts_before_spawning() {
    // The admission cap precedes pool construction, so no threads are spawned.
    assert_eq!(
        AsyncTextureState::new_with_workers(u32::MAX).map(|_| ()),
        Err(EzGfxResult::InvalidArgument)
    );
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn context_decode_worker_count_reaches_pool_construction() {
    // The context option must reach the async texture pool on a real backend.
    let explicit_options = vulkan_options().unwrap().with_texture_decode_workers(2);
    let explicit_context = create_context(explicit_options).unwrap();
    let explicit_workers = with_context_mut(explicit_context, |state| {
        Ok(state.async_textures.worker_count())
    })
    .unwrap();
    assert_eq!(explicit_workers, 2);
    assert_eq!(destroy_context(explicit_context), EzGfxResult::Ok);
}

#[cfg(windows)]
#[test]
fn destroy_context_reclaims_populated_state_and_invalidates_handles() {
    let context = dx12_context();
    let structured = acquire_structured(context, 64).unwrap();
    let indirect = acquire_indirect(context, 2).unwrap();
    assert_eq!(
        create_vertex_heap(context, "vertices", 256, 16),
        EzGfxResult::Ok
    );
    assert_eq!(create_index_heap(context, 256), EzGfxResult::Ok);

    assert_eq!(destroy_context(context), EzGfxResult::Ok);
    assert_eq!(wait_idle(context), EzGfxResult::InvalidContext);
    assert_eq!(
        write_structured(context, structured, &[1; 16]),
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        set_indirect_count(context, indirect, 1),
        EzGfxResult::InvalidContext
    );
    assert_eq!(destroy_context(context), EzGfxResult::InvalidContext);
}

#[cfg(windows)]
#[test]
fn destroy_context_rejects_wrong_thread_without_consuming_context() {
    let context = dx12_context();

    assert_eq!(
        std::thread::spawn(move || destroy_context(context))
            .join()
            .unwrap(),
        EzGfxResult::InvalidContext
    );
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
}

#[cfg(windows)]
fn thread_exit_context() -> ContextHandle {
    dx12_context()
}

#[cfg(not(any(windows, target_vendor = "apple")))]
fn thread_exit_context() -> ContextHandle {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let surface = create_surface(
        context,
        SurfaceOptions::new(0, 0, SurfacePlatform::Headless, 1, 1, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(init_device(context, surface), EzGfxResult::Ok);
    context
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn recursive_context_access_returns_native_failure_without_panicking() {
    let context = thread_exit_context();

    let nested = with_context_mut(context, |_| {
        with_context_mut(context, |_| Ok::<_, EzGfxResult>(()))
    });

    assert_eq!(nested, Err(EzGfxResult::NativeFailure));
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn thread_exit_invalidates_populated_context_handle() {
    let context = thread_exit_context();
    let _structured = acquire_structured(context, 64).unwrap();
    let cleanup = CONTEXTS.with(|contexts| contexts.borrow_mut().cleanup_for_thread_exit());

    assert_eq!(cleanup, EzGfxResult::Ok);
    assert_eq!(wait_idle(context), EzGfxResult::InvalidContext);
}
#[cfg(not(target_vendor = "apple"))]
#[test]
fn creator_thread_exit_returns_and_invalidates_context_handle() {
    let stale = std::thread::spawn(thread_exit_context).join().unwrap();
    // Join completes thread-local abandonment before stale lookup and possible slot reuse.

    assert_eq!(wait_idle(stale), EzGfxResult::InvalidContext);
    let current = thread_exit_context();
    assert_ne!(current, stale);
    assert_eq!(destroy_context(current), EzGfxResult::Ok);
}

#[cfg(windows)]
#[test]
fn destroyed_resource_handles_are_rejected_by_other_owners() {
    let first = dx12_context();
    let stale = acquire_structured(first, 64).unwrap();
    let second = dx12_context();

    assert_eq!(destroy_context(first), EzGfxResult::Ok);
    assert_eq!(
        write_structured(second, stale, &[1; 16]),
        EzGfxResult::InvalidContext
    );
    assert_eq!(destroy_context(second), EzGfxResult::Ok);
}

#[cfg(windows)]
#[test]
fn lost_context_can_still_be_destroyed_terminally() {
    let context = dx12_context();
    with_context_mut(context, |owned| {
        owned.identity.mark_lost().map_err(map_lifecycle)
    })
    .unwrap();

    assert_eq!(wait_idle(context), EzGfxResult::DeviceLost);
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
    assert_eq!(wait_idle(context), EzGfxResult::InvalidContext);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn device_loss_sweeps_pending_decodes_to_fast_device_lost() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let gate = Arc::new(std::sync::Barrier::new(2));
    with_context_mut(context, |state| {
        state.async_textures.decode_gate = Some(gate.clone());
        Ok(())
    })
    .unwrap();

    let texture = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1, 2, 3, 4],
        false,
        &texture_config(),
    )
    .unwrap();
    assert_eq!(poll_texture_load(context, texture), EzGfxResult::NotReady);

    // First observed loss marks the context; the poll itself must sweep the
    // still-gated decode so no later poll can report NotReady.
    with_context_mut(context, |owned| {
        owned.identity.mark_lost().map_err(map_lifecycle)
    })
    .unwrap();
    assert_eq!(poll_texture_load(context, texture), EzGfxResult::DeviceLost);
    assert_eq!(poll_texture_load(context, texture), EzGfxResult::DeviceLost);
    with_context_mut(context, |state| {
        assert!(state.pending_textures.is_empty());
        Ok(())
    })
    .unwrap();

    // Releasing the gate must not revive the swept request.
    gate.wait();
    assert_eq!(poll_texture_load(context, texture), EzGfxResult::DeviceLost);
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
}

#[test]
fn coarse_range_completion_ignores_hidden_fine_updates() {
    for (values, resident, expected) in [
        (&[7, 2, 1][..], 1, Some(1)),
        (&[7, 9, 1][..], 2, Some(9)),
        (&[7, 9, 1][..], 3, Some(9)),
        (&[0, 2, 1][..], 2, Some(2)),
        (&[0, 2, 1][..], 3, None),
        (&[1][..], 0, None),
        (&[1][..], 2, None),
        (&[][..], 1, None),
    ] {
        assert_eq!(
            super::texture::mip_range_completion(values, resident),
            expected
        );
    }
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn render_target_lifecycle_rejects_misuse_before_native_work() {
    use ez_gfx_runtime::target::{ClearValue, TargetDeclaration, TargetUsage};
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let declaration = TargetDeclaration::new(
        "rt-proof",
        TargetUsage::Color,
        1.0,
        1,
        vec![Format::Rgba8Unorm],
        ClearValue::Color([1.0, 0.0, 0.0, 1.0]),
        true,
    )
    .unwrap();
    // Empty extents fail before leasing allocator state; no device is needed.
    assert_eq!(
        create_render_target(context, &declaration, 0, 64),
        Err(EzGfxResult::InvalidArgument)
    );
    // Depth usage is deferred to the pass-attachment slice.
    let depth = TargetDeclaration::new(
        "rt-depth",
        TargetUsage::Depth,
        1.0,
        1,
        vec![Format::Depth32Float],
        ClearValue::DepthStencil {
            depth: 1.0,
            stencil: 0,
        },
        false,
    )
    .unwrap();
    assert_eq!(
        create_render_target(context, &depth, 64, 64),
        Err(EzGfxResult::Unsupported)
    );
    // Unknown handles never reach native code. Live-target creation, format,
    // extent, clear, and destroy need an initialized device, which requires a
    // real surface; that path is proven by the native allocation tests on
    // Vulkan, DX12, and Metal instead of here.
    let phantom = RenderTargetHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(7, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        render_target_format(context, phantom),
        Err(EzGfxResult::InvalidArgument)
    );
    assert_eq!(
        render_target_extent(context, phantom),
        Err(EzGfxResult::InvalidArgument)
    );
    assert_eq!(
        render_target_clear(context, phantom),
        Err(EzGfxResult::InvalidArgument)
    );
    destroy_render_target(context, phantom);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn probe_render_target_format_rejects_misuse_before_native_work() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    // Sample counts outside the closed set fail before probing any device.
    assert_eq!(
        probe_render_target_format(context, Format::Rgba8Unorm, 3),
        EzGfxResult::InvalidArgument
    );
    // Probing without an initialized device cannot query adapter capabilities.
    // Live-device resolution is proven by the native allocation tests on
    // Vulkan, DX12, and Metal instead of here.
    assert_eq!(
        probe_render_target_format(context, Format::Rgba8Unorm, 1),
        EzGfxResult::NativeFailure
    );
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn begin_render_target_rejects_foreign_handles() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    // A forged handle resolves to nothing.
    let phantom = RenderTargetHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(7, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        begin_render_target(context, phantom),
        EzGfxResult::InvalidContext
    );
    // A live texture handle is the wrong kind, never an alias.
    let texture = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1, 2, 3, 4],
        false,
        &texture_config(),
    )
    .unwrap();
    let mistaken = RenderTargetHandle::from_packed(texture.packed()).unwrap();
    assert_eq!(
        begin_render_target(context, mistaken),
        EzGfxResult::InvalidContext
    );
}

#[cfg(not(target_vendor = "apple"))]
fn stale_target() -> RenderTargetHandle {
    RenderTargetHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(7, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn frame_begin_clears_stale_render_target_override() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    with_context_mut(context, |context| {
        context.frame_render_target = Some(stale_target());
        Ok(())
    })
    .unwrap();
    assert_eq!(frame_begin(context), EzGfxResult::Ok);
    with_context_mut(context, |context| {
        assert_eq!(context.frame_render_target, None);
        Ok(())
    })
    .unwrap();
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn destroy_render_target_clears_bound_override() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    with_context_mut(context, |context| {
        context.frame_render_target = Some(stale_target());
        Ok(())
    })
    .unwrap();
    // Unknown handles stay infallible, but a matching stale binding is dropped.
    destroy_render_target(context, stale_target());
    with_context_mut(context, |context| {
        assert_eq!(context.frame_render_target, None);
        Ok(())
    })
    .unwrap();
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn heap_slots_unify_textures_and_render_targets_without_collision() {
    use std::collections::HashSet;
    let context = create_context(vulkan_options().unwrap()).unwrap();
    with_context_mut(context, |context| {
        // Textures (`begin_upload`) and render targets (same call in
        // `create_render_target`) draw from one free-list, so interleaved
        // leases must never share a binding.
        let mut leased = Vec::new();
        for _ in 0..3 {
            let texture = context.texture_registry.begin_upload().unwrap();
            let target = context.texture_registry.begin_upload().unwrap();
            leased.push(texture);
            leased.push(target);
        }
        let bindings: HashSet<u32> = leased
            .iter()
            .map(|id| context.texture_registry.reserved_binding(*id).unwrap())
            .collect();
        assert_eq!(bindings.len(), leased.len());
        for id in leased {
            context.texture_registry.cancel_upload(id).unwrap();
        }
        Ok(())
    })
    .unwrap();
}
#[cfg(not(target_vendor = "apple"))]
#[test]
fn heap_slot_release_reuses_the_freed_binding() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    with_context_mut(context, |context| {
        // `destroy_render_target` releases via `cancel_upload`; the next lease
        // must reuse the freed slot instead of growing the heap.
        let first = context.texture_registry.begin_upload().unwrap();
        let binding = context.texture_registry.reserved_binding(first).unwrap();
        context.texture_registry.cancel_upload(first).unwrap();
        let second = context.texture_registry.begin_upload().unwrap();
        assert_eq!(
            context.texture_registry.reserved_binding(second).unwrap(),
            binding
        );
        context.texture_registry.cancel_upload(second).unwrap();
        Ok(())
    })
    .unwrap();
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn heap_slot_exhaustion_is_shared_and_fail_fast() {
    let capacity = ez_gfx_runtime::binding::MAX_TEXTURE_HEAP_CAPACITY as usize;
    let context = create_context(vulkan_options().unwrap()).unwrap();
    with_context_mut(context, |context| {
        // Textures and targets share one cap: filling it with texture leases
        // leaves no room for a target lease, mapping to `NativeFailure` like
        // the old top-down range exhaustion did.
        let mut leased = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            leased.push(context.texture_registry.begin_upload().unwrap());
        }
        assert_eq!(
            context.texture_registry.begin_upload().map(|_| ()),
            Err(ez_gfx_runtime::texture::TextureError::CapacityExceeded)
        );
        for id in leased {
            context.texture_registry.cancel_upload(id).unwrap();
        }
        // The heap is whole again after release.
        context.texture_registry.begin_upload().unwrap();
        Ok(())
    })
    .unwrap();
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn rejected_render_target_admissions_leave_no_allocator_residue() {
    use ez_gfx_runtime::target::{ClearValue, TargetDeclaration, TargetUsage};
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let color = TargetDeclaration::new(
        "rt-residue",
        TargetUsage::Color,
        1.0,
        1,
        vec![Format::Rgba8Unorm],
        ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
        true,
    )
    .unwrap();
    assert_eq!(
        create_render_target(context, &color, 0, 64),
        Err(EzGfxResult::InvalidArgument)
    );
    let depth = TargetDeclaration::new(
        "rt-residue-depth",
        TargetUsage::Depth,
        1.0,
        1,
        vec![Format::Depth32Float],
        ClearValue::DepthStencil {
            depth: 1.0,
            stencil: 0,
        },
        false,
    )
    .unwrap();
    assert_eq!(
        create_render_target(context, &depth, 64, 64),
        Err(EzGfxResult::Unsupported)
    );
    // A later texture admission takes slot zero, proving the rejections leased
    // nothing from the shared heap.
    let texture = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1, 2, 3, 4],
        false,
        &texture_config(),
    )
    .unwrap();
    with_context_mut(context, |context| {
        let pending = context.pending_textures.get(&texture).unwrap();
        let binding = context
            .texture_registry
            .reserved_binding(pending.id)
            .unwrap();
        assert_eq!(binding, 0);
        Ok(())
    })
    .unwrap();
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn explicit_selection_rejects_unknown_identity_before_native_calls() {
    // No surface is created, shown, or activated by this test.
    let options = vulkan_options().unwrap().with_adapter([0xA5; 16], false);
    assert_eq!(create_context(options), Err(EzGfxResult::InvalidArgument));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn explicit_selection_creates_context_for_enumerated_adapter() {
    // No surface is created, shown, or activated by this test.
    let wanted = query_adapter_report(true)
        .into_iter()
        .filter(|report| report.adapter().backend() == Backend::Vulkan)
        .find(ez_gfx_runtime::AdapterReport::admitted)
        .expect("at least one profile-admitted Vulkan adapter")
        .adapter()
        .stable_id();
    let options = vulkan_options().unwrap().with_adapter(wanted, true);
    let context = create_context(options).expect("enumerated adapter creates a context");
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
}

#[test]
fn adapter_report_names_every_enumerated_adapter() {
    // No surface is created, shown, or activated by this test.
    let adapters = enumerate_adapters();
    assert!(!adapters.is_empty());
    let mut identities = HashSet::new();
    for adapter in &adapters {
        assert!(identities.insert((adapter.backend(), adapter.stable_id())));
    }
    let strict = query_adapter_report(false);
    let permissive = query_adapter_report(true);
    assert_eq!(strict.len(), adapters.len());
    assert_eq!(permissive.len(), adapters.len());
    for (info, report) in adapters.iter().zip(&strict) {
        assert_eq!(report.adapter().stable_id(), info.stable_id());
        assert_eq!(
            report.admitted(),
            report.errors().is_empty() && !report.software_rejected()
        );
    }
    // Opting into software never un-admits an adapter.
    for (strict_report, permissive_report) in strict.iter().zip(&permissive) {
        if strict_report.admitted() {
            assert!(permissive_report.admitted());
        }
    }
}
