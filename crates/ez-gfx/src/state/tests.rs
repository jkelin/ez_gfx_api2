use super::*;
use std::collections::HashSet;

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

fn texture_config() -> TextureConfig {
    TextureConfig {
        width: 1,
        height: 1,
        mip_count: 1,
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

#[test]
fn texture_admission_is_nonblocking_and_pending_cancellation_invalidates_the_handle() {
    let context =
        create_context(ContextOptions::new_for_backend(0, 0, 0, Backend::Vulkan).unwrap()).unwrap();
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

#[cfg(windows)]
fn dx12_context() -> ContextHandle {
    create_context(ContextOptions::new_for_backend(0, 0, 0, Backend::Dx12).unwrap()).unwrap()
}
#[cfg(windows)]
#[test]
fn destroy_context_before_device_initialization_is_successful_and_terminal() {
    let context =
        create_context(ContextOptions::new_for_backend(0, 0, 0, Backend::Vulkan).unwrap()).unwrap();

    assert_eq!(destroy_context(context), EzGfxResult::Ok);
    assert_eq!(wait_idle(context), EzGfxResult::InvalidContext);
    assert_eq!(destroy_context(context), EzGfxResult::InvalidContext);
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

#[cfg(target_vendor = "apple")]
fn thread_exit_context() -> ContextHandle {
    create_context(ContextOptions::new_for_backend(0, 0, 2, Backend::Metal).unwrap()).unwrap()
}

#[cfg(any(windows, target_vendor = "apple"))]
#[test]
fn recursive_context_access_returns_native_failure_without_panicking() {
    let context = thread_exit_context();

    let nested = with_context_mut(context, |_| {
        with_context_mut(context, |_| Ok::<_, EzGfxResult>(()))
    });

    assert_eq!(nested, Err(EzGfxResult::NativeFailure));
    assert_eq!(destroy_context(context), EzGfxResult::Ok);
}

#[cfg(any(windows, target_vendor = "apple"))]
#[test]
fn thread_exit_invalidates_populated_context_handle() {
    let context = thread_exit_context();
    let _structured = acquire_structured(context, 64).unwrap();
    let cleanup = CONTEXTS.with(|contexts| contexts.borrow_mut().cleanup_for_thread_exit());

    assert_eq!(cleanup, EzGfxResult::Ok);
    assert_eq!(wait_idle(context), EzGfxResult::InvalidContext);
}
#[cfg(any(windows, target_vendor = "apple"))]
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
