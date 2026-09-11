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
        Err(Error::InvalidArgument)
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
        Err(Error::Unsupported)
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
        Err(Error::InvalidArgument)
    );
    assert_eq!(
        render_target_extent(context, phantom),
        Err(Error::InvalidArgument)
    );
    assert_eq!(
        render_target_clear(context, phantom),
        Err(Error::InvalidArgument)
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
        Err(Error::InvalidArgument)
    );
    // Probing without an initialized device cannot query adapter capabilities.
    // Live-device resolution is proven by the native allocation tests on
    // Vulkan, DX12, and Metal instead of here.
    assert_eq!(
        probe_render_target_format(context, Format::Rgba8Unorm, 1),
        Err(Error::NativeFailure)
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
        Err(Error::Lifecycle(LifecycleError::WrongOwner))
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
        Err(Error::Lifecycle(LifecycleError::WrongResourceKind))
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
    assert_eq!(frame_begin(context), Ok(()));
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
        Err(Error::InvalidArgument)
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
        Err(Error::Unsupported)
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
    assert_eq!(create_context(options), Err(Error::InvalidArgument));
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
    assert_eq!(destroy_context(context), Ok(()));
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
