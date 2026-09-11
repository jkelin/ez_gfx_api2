use super::*;

#[cfg(not(target_vendor = "apple"))]
#[test]
fn queued_texture_cancellation_is_terminal_without_admission_retry() {
    let context = create_context(vulkan_options().unwrap().with_texture_decode_workers(4)).unwrap();
    let gate = Arc::new(std::sync::Barrier::new(5));
    with_context_mut(context, |state| {
        state.async_textures.decode_gate = Some(gate.clone());
        Ok(())
    })
    .unwrap();
    let mut textures = Vec::new();
    for color in 0_u8..5 {
        textures.push(
            load_texture(
                context,
                TextureSource::Rgba8 {
                    width: 1,
                    height: 1,
                },
                &[color, color, color, 255],
                false,
                &texture_config(),
            )
            .unwrap(),
        );
    }

    let queued = textures[4];
    assert_eq!(cancel_texture_load(context, queued), Ok(()));
    assert_eq!(
        texture_binding(context, queued),
        Err(Error::Lifecycle(LifecycleError::StaleHandle))
    );

    gate.wait();
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn failed_decode_binding_returns_the_terminal_error() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let texture = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1],
        false,
        &texture_config(),
    )
    .unwrap();
    assert_eq!(texture_binding(context, texture), Ok(0));
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    assert_eq!(wait_idle(context), Ok(()));
    assert_eq!(
        texture_binding(context, texture),
        Err(Error::InvalidArgument)
    );
    destroy_surface(context, surface).unwrap();
    destroy_context(context).unwrap();
}

#[cfg(windows)]
#[test]
fn wait_idle_publishes_every_generated_mip() {
    let context = dx12_context();
    let config = TextureConfig {
        width: 4,
        height: 4,
        mip_count: 0,
        destination: TextureDestination::Rgba8Unorm,
        sampler: texture_config().sampler,
    };
    let texture = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 4,
            height: 4,
        },
        &[255; 4 * 4 * 4],
        true,
        &config,
    )
    .unwrap();

    assert_eq!(wait_idle(context), Ok(()));
    assert_eq!(texture_residency(context, texture), Ok((3, 3)));
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(windows)]
#[test]
fn region_update_remains_pending_in_resource_diagnostics() {
    let context = dx12_context();
    let texture = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[0, 0, 0, 255],
        false,
        &texture_config(),
    )
    .unwrap();
    assert_eq!(wait_idle(context), Ok(()));

    assert_eq!(
        update_texture_region(
            context,
            texture,
            TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                bytes: &[255, 255, 255, 255],
            },
        ),
        Ok(())
    );
    let pending = resource_diagnostics(context).unwrap();
    assert_eq!(
        (pending.pending_textures, pending.pending_texture_bytes),
        (1, 4)
    );

    assert_eq!(wait_idle(context), Ok(()));
    assert_eq!(resource_diagnostics(context).unwrap().pending_textures, 0);
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(windows)]
#[test]
fn transfer_stage_cancellation_releases_the_full_manager_window() {
    let context = create_context(
        ContextOptions::new_for_backend(0, 0, Backend::Dx12)
            .unwrap()
            .with_texture_decode_workers(4),
    )
    .unwrap();
    let gate = Arc::new(std::sync::Barrier::new(5));
    with_context_mut(context, |state| {
        state.async_textures.decode_gate = Some(gate.clone());
        Ok(())
    })
    .unwrap();
    let textures = (0_u8..4)
        .map(|color| {
            load_texture(
                context,
                TextureSource::Rgba8 {
                    width: 1,
                    height: 1,
                },
                &[color, color, color, 255],
                false,
                &texture_config(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    gate.wait();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let decoded = with_context_mut(context, |state| {
            super::super::texture_manager::collect_decode_results(state);
            Ok(state.async_textures.decoded.len())
        })
        .unwrap();
        if decoded == textures.len() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "decode workers did not fill the manager window"
        );
        std::thread::yield_now();
    }

    with_context_mut(context, |state| {
        state.async_textures.decode_gate = None;
        pump_async_textures(state)?;
        assert_eq!(state.texture_transfer_work.len(), textures.len());
        assert_eq!(
            state.async_textures.working_bytes,
            TEXTURE_WORKING_SET_BUDGET
        );
        Ok(())
    })
    .unwrap();
    for texture in textures {
        assert_eq!(cancel_texture_load(context, texture), Ok(()));
    }
    with_context_mut(context, |state| {
        assert!(state.texture_transfer_work.is_empty());
        assert_eq!(state.async_textures.working_bytes, 0);
        Ok(())
    })
    .unwrap();

    let replacement = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[255, 255, 255, 255],
        false,
        &texture_config(),
    )
    .unwrap();
    assert_eq!(wait_idle(context), Ok(()));
    assert!(texture_binding(context, replacement).is_ok());
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn fallback_publication_failure_remains_owned_and_retries() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    with_context_mut(context, |state| {
        assert!(state.texture_fallback.is_owned());
        assert!(state.texture_fallback.is_ready());

        state.texture_fallback.inject_publication_failure()?;
        assert_eq!(
            initialize_texture_fallback(state),
            Err(Error::NativeFailure)
        );
        assert!(state.texture_fallback.is_owned());
        assert!(!state.texture_fallback.is_ready());

        state.texture_fallback.resume_publication()?;
        initialize_texture_fallback(state)?;
        assert!(state.texture_fallback.is_ready());
        Ok(())
    })
    .unwrap();
    destroy_surface(context, surface).unwrap();
    destroy_context(context).unwrap();
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn pre_device_fallback_backfill_retries_only_unpublished_aliases() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let first = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1, 2, 3, 255],
        false,
        &texture_config(),
    )
    .unwrap();
    let second = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[4, 5, 6, 255],
        false,
        &texture_config(),
    )
    .unwrap();
    assert_eq!(texture_binding(context, first), Ok(0));
    assert_eq!(texture_binding(context, second), Ok(1));
    with_context_mut(context, |state| {
        assert!(!state.texture_fallback.is_ready());
        assert!(
            state
                .pending_textures
                .values()
                .all(|pending| !pending.fallback_published)
        );
        state.texture_fallback_alias_test_failure_after = Some(1);
        Ok(())
    })
    .unwrap();

    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    assert_eq!(init_device(context, surface), Err(Error::NativeFailure));
    with_context_mut(context, |state| {
        assert!(state.texture_fallback.is_ready());
        assert_eq!(
            state
                .pending_textures
                .values()
                .filter(|pending| pending.fallback_published)
                .count(),
            1
        );
        state.texture_fallback_alias_test_failure_after = None;
        Ok(())
    })
    .unwrap();

    init_device(context, surface).unwrap();
    with_context_mut(context, |state| {
        assert!(
            state
                .pending_textures
                .values()
                .all(|pending| pending.fallback_published)
        );
        // Any redundant repeated-init alias write would trigger this injection.
        state.texture_fallback_alias_test_failure_after = Some(0);
        Ok(())
    })
    .unwrap();
    init_device(context, surface).unwrap();
    assert_eq!(texture_binding(context, first), Ok(0));
    assert_eq!(texture_binding(context, second), Ok(1));

    cancel_texture_load(context, first).unwrap();
    cancel_texture_load(context, second).unwrap();
    destroy_surface(context, surface).unwrap();
    destroy_context(context).unwrap();
}
