use super::*;

#[cfg(not(target_vendor = "apple"))]
#[test]
fn queued_texture_cancellation_is_terminal_without_admission_retry() {
    let context = create_context(vulkan_options().unwrap().with_texture_decode_workers(4)).unwrap();
    let gate = Arc::new(std::sync::Barrier::new(5));
    with_context_mut(context, |state| {
        state.decode_textures.set_decode_gate(Some(gate.clone()));
        Ok(())
    })
    .unwrap();
    let mut textures = Vec::new();
    for color in 0_u8..5 {
        textures
            .push(load_texture(context, &[color, color, color, 255], &texture_config()).unwrap());
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
fn decoded_pending_cancellation_releases_decode_admission() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let texture = load_texture(context, &[1, 2, 3, 255], &texture_config()).unwrap();

    // Native submission waits for device admission, so the completed decode
    // stays pending and holds its reservation until cancellation.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let decoded = with_context_mut(context, |state| {
            super::super::texture_manager::collect_decode_results(state);
            Ok(state.texture_pipeline.decoded().len())
        })
        .unwrap();
        if decoded == 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "decode worker did not complete before cancellation"
        );
        std::thread::yield_now();
    }
    with_context_mut(context, |state| {
        assert_eq!(state.transfer_pool.in_use(), DECODE_RESERVATION_BYTES);
        Ok(())
    })
    .unwrap();

    assert_eq!(cancel_texture_load(context, texture), Ok(()));
    with_context_mut(context, |state| {
        assert_eq!(state.transfer_pool.in_use(), 0);
        Ok(())
    })
    .unwrap();
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn failed_decode_binding_returns_the_terminal_error() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let texture = load_texture(context, &[1], &texture_config()).unwrap();
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

#[cfg(not(target_vendor = "apple"))]
#[test]
fn oversized_required_mips_fails_terminally_without_clamping() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let config = TextureConfig {
        required_mips: 4,
        ..texture_config()
    };
    let texture = load_texture(context, &[1, 2, 3, 4], &config).unwrap();
    assert_eq!(texture_binding(context, texture), Ok(0));
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    assert_eq!(wait_idle(context), Ok(()));
    // One decoded mip cannot satisfy four required: the load fails with
    // InvalidArgument instead of weakening the wait to the coarse mip.
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
        source: TextureSource::Rgba8 {
            width: 4,
            height: 4,
        },
        generate_mips: true,
        required_mips: 1,
        width: 4,
        height: 4,
        mip_count: 0,
        destination: TextureDestination::Rgba8Unorm,
        sampler: texture_config().sampler,
    };
    let texture = load_texture(context, &[255; 4 * 4 * 4], &config).unwrap();

    assert_eq!(wait_idle(context), Ok(()));
    assert_eq!(texture_residency(context, texture), Ok((3, 3)));
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(windows)]
#[test]
fn event_pumping_publishes_every_generated_mip_without_wait_idle() {
    let context = dx12_context();
    let config = TextureConfig {
        source: TextureSource::Rgba8 {
            width: 4,
            height: 4,
        },
        generate_mips: true,
        required_mips: 1,
        width: 4,
        height: 4,
        mip_count: 0,
        destination: TextureDestination::Rgba8Unorm,
        sampler: texture_config().sampler,
    };
    let texture = load_texture(context, &[255; 4 * 4 * 4], &config).unwrap();

    // Ordinary event polling must advance residency after required readiness
    // leaves the ready map empty. Querying residency directly would advance it
    // as a side effect, so inspect the published count without advancing.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        poll_upload_event(context).unwrap();
        let published = with_context_mut(context, |state| {
            Ok(state.texture_pipeline.published().get(&texture).copied())
        })
        .unwrap();
        if published == Some(3) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "event pumping stalled at {published:?} published mips"
        );
        std::thread::yield_now();
    }
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(windows)]
#[test]
fn region_update_remains_pending_in_resource_diagnostics() {
    let context = dx12_context();
    let texture = load_texture(context, &[0, 0, 0, 255], &texture_config()).unwrap();
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
        state.decode_textures.set_decode_gate(Some(gate.clone()));
        Ok(())
    })
    .unwrap();
    let textures = (0_u8..4)
        .map(|color| load_texture(context, &[color, color, color, 255], &texture_config()).unwrap())
        .collect::<Vec<_>>();

    gate.wait();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let decoded = with_context_mut(context, |state| {
            super::super::texture_manager::collect_decode_results(state);
            Ok(state.texture_pipeline.decoded().len())
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
        state.decode_textures.set_decode_gate(None);
        pump_async_textures(state)?;
        assert_eq!(state.texture_pipeline.work().len(), textures.len());
        assert_eq!(state.transfer_pool.in_use(), WORKING_SET_BUDGET_BYTES);
        Ok(())
    })
    .unwrap();
    for texture in textures {
        assert_eq!(cancel_texture_load(context, texture), Ok(()));
    }
    with_context_mut(context, |state| {
        assert!(state.texture_pipeline.work().is_empty());
        assert_eq!(state.transfer_pool.in_use(), 0);
        Ok(())
    })
    .unwrap();

    let replacement = load_texture(context, &[255, 255, 255, 255], &texture_config()).unwrap();
    assert_eq!(wait_idle(context), Ok(()));
    assert!(texture_binding(context, replacement).is_ok());
    assert_eq!(destroy_context(context), Ok(()));
}

#[test]
fn completed_required_prefix_releases_fine_mip_budget() {
    let required = CompletionToken::new(QueueKind::TextureTransfer, 7).unwrap();
    let fine = CompletionToken::new(QueueKind::TextureTransfer, 9).unwrap();
    let mut pool = SharedTransferPool::new(128);
    pool.acquire_required(128).unwrap();
    let mut work = ez_gfx_texture_manager::PrefixTransferWork::new(required, 128);

    assert_eq!(
        pool.acquire_background(1),
        Err(ez_gfx_texture_manager::TextureScheduleError::BudgetOverflow)
    );
    let (required_bytes, fine_bytes) = work.release_completed(7);
    pool.release_texture(required_bytes);
    pool.release_background(fine_bytes);
    assert_eq!(pool.texture_ledger(), 0);

    pool.acquire_background(64).unwrap();
    work.note_fine(fine, 64).unwrap();
    assert_eq!(work.release_completed(8), (0, 0));
    let (_, fine_bytes) = work.release_completed(9);
    pool.release_background(fine_bytes);
    assert_eq!(pool.in_use(), 0);
    assert!(work.is_clear());
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
    let first = load_texture(context, &[1, 2, 3, 255], &texture_config()).unwrap();
    let second = load_texture(context, &[4, 5, 6, 255], &texture_config()).unwrap();
    assert_eq!(texture_binding(context, first), Ok(0));
    assert_eq!(texture_binding(context, second), Ok(1));
    with_context_mut(context, |state| {
        assert!(!state.texture_fallback.is_ready());
        assert!(
            state
                .texture_pipeline
                .pending()
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
                .texture_pipeline
                .pending()
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
                .texture_pipeline
                .pending()
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

#[cfg(not(target_vendor = "apple"))]
#[test]
fn gate_skips_decode_wait_without_heap_demand() {
    let context = create_context(vulkan_options().unwrap().with_texture_decode_workers(4)).unwrap();
    let gate = Arc::new(std::sync::Barrier::new(2));
    with_context_mut(context, |state| {
        state.decode_textures.set_decode_gate(Some(gate.clone()));
        Ok(())
    })
    .unwrap();
    let texture = load_texture(context, &[1, 2, 3, 255], &texture_config()).unwrap();
    // Workers block at the decode gate, so no decode can complete; without heap
    // demand the submission gate must return immediately instead of driving.
    with_context_mut(context, |state| {
        super::super::texture_manager::gate_required_textures_for_submit(state, false)?;
        assert!(!state.texture_pipeline.pending().is_empty());
        Ok(())
    })
    .unwrap();
    gate.wait();
    assert_eq!(cancel_texture_load(context, texture), Ok(()));
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn gate_drives_pending_decode_to_referenced_prefix() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    let texture = load_texture(context, &[1, 2, 3, 255], &texture_config()).unwrap();
    // No polling: the gate alone must drive decode to a referenced required
    // prefix with its GPU-wait token retained.
    with_context_mut(context, |state| {
        super::super::texture_manager::gate_required_textures_for_submit(state, true)?;
        assert!(state.texture_pipeline.pending().is_empty());
        assert!(state.texture_pipeline.ready().contains_key(&texture));
        assert_eq!(
            state.texture_pipeline.published().get(&texture).copied(),
            Some(1)
        );
        Ok(())
    })
    .unwrap();
    destroy_surface(context, surface).unwrap();
    destroy_context(context).unwrap();
}

#[cfg(windows)]
#[test]
fn failed_submitted_texture_leaves_no_pipeline_residue() {
    let context = dx12_context();
    let texture = load_texture(context, &[0, 0, 0, 255], &texture_config()).unwrap();
    assert_eq!(wait_idle(context), Ok(()));
    with_context_mut(context, |state| {
        assert!(state.texture_pipeline.submitted().contains_key(&texture));
        assert!(state.texture_pipeline.published().contains_key(&texture));
        super::super::texture::record_texture_failure(state, texture, Error::NativeFailure);
        assert!(state.texture_pipeline.submitted().get(&texture).is_none());
        assert!(state.texture_pipeline.published().get(&texture).is_none());
        assert!(state.texture_pipeline.targets().get(&texture).is_none());
        assert!(
            state
                .texture_pipeline
                .last_transfer()
                .get(&texture)
                .is_none()
        );
        assert!(state.texture_pipeline.fine().get(&texture).is_none());
        assert!(state.texture_pipeline.decoded().get(&texture).is_none());
        assert!(state.texture_pipeline.ready().get(&texture).is_none());
        assert!(state.texture_pipeline.work().get(&texture).is_none());
        assert!(
            state
                .texture_pipeline
                .transfer_bytes()
                .get(&texture)
                .is_none()
        );
        assert_eq!(state.transfer_pool.in_use(), 0);
        Ok(())
    })
    .unwrap();
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(windows)]
#[test]
fn region_update_completion_does_not_reemit_device_ready() {
    use crate::{UploadResource, UploadStatus};
    let context = dx12_context();
    let texture = load_texture(context, &[0, 0, 0, 255], &texture_config()).unwrap();
    assert_eq!(wait_idle(context), Ok(()));
    let mut ready = 0;
    while let Some(event) = poll_upload_event(context).unwrap() {
        if matches!(
            event,
            crate::UploadEvent {
                resource: UploadResource::Texture(event_texture),
                status: UploadStatus::DeviceReady,
            } if event_texture == texture
        ) {
            ready += 1;
        }
    }
    assert_eq!(ready, 1);
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
    assert_eq!(wait_idle(context), Ok(()));
    let mut reemitted = 0;
    while let Some(event) = poll_upload_event(context).unwrap() {
        if matches!(
            event,
            crate::UploadEvent {
                resource: UploadResource::Texture(event_texture),
                status: UploadStatus::DeviceReady,
            } if event_texture == texture
        ) {
            reemitted += 1;
        }
    }
    assert_eq!(reemitted, 0);
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn optional_zero_texture_does_not_block_graphics_gate() {
    use crate::{UploadResource, UploadStatus};
    let context = create_context(vulkan_options().unwrap().with_texture_decode_workers(4)).unwrap();
    let gate = Arc::new(std::sync::Barrier::new(2));
    with_context_mut(context, |state| {
        state.decode_textures.set_decode_gate(Some(gate.clone()));
        Ok(())
    })
    .unwrap();
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    let config = TextureConfig {
        required_mips: 0,
        ..texture_config()
    };
    let texture = load_texture(context, &[1, 2, 3, 255], &config).unwrap();
    // The stable binding aliases fallback from admission.
    assert_eq!(texture_binding(context, texture), Ok(0));
    // The worker blocks at the decode gate, so no decode can complete; the
    // heap-demanding gate must still return immediately for optional work.
    with_context_mut(context, |state| {
        super::super::texture_manager::gate_required_textures_for_submit(state, true)?;
        assert!(!state.texture_pipeline.pending().is_empty());
        assert!(!super::super::texture_manager::has_required_pending(
            &state.texture_pipeline
        ));
        Ok(())
    })
    .unwrap();
    gate.wait();
    assert_eq!(cancel_texture_load(context, texture), Ok(()));
    // Terminal cancellation emits exactly one Cancelled event and releases
    // the decode reservation without residue.
    let mut cancelled = 0;
    let mut staged = 0;
    while let Some(event) = poll_upload_event(context).unwrap() {
        match event.status {
            UploadStatus::SourceStaged => staged += 1,
            UploadStatus::Cancelled => {
                assert_eq!(event.resource, UploadResource::Texture(texture));
                cancelled += 1;
            }
            _ => {}
        }
    }
    assert_eq!((staged, cancelled), (1, 1));
    with_context_mut(context, |state| {
        assert_eq!(state.transfer_pool.in_use(), 0);
        Ok(())
    })
    .unwrap();
    destroy_surface(context, surface).unwrap();
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn gate_wait_condition_distinguishes_required_from_optional() {
    let context = create_context(vulkan_options().unwrap().with_texture_decode_workers(4)).unwrap();
    // Two textures dispatch two workers, so the rendezvous needs both plus
    // the test thread; a count of two would let the workers release each
    // other and strand the final wait.
    let gate = Arc::new(std::sync::Barrier::new(3));
    with_context_mut(context, |state| {
        state.decode_textures.set_decode_gate(Some(gate.clone()));
        Ok(())
    })
    .unwrap();
    let required = load_texture(context, &[1, 2, 3, 255], &texture_config()).unwrap();
    with_context_mut(context, |state| {
        assert!(super::super::texture_manager::has_required_pending(
            &state.texture_pipeline
        ));
        Ok(())
    })
    .unwrap();
    let config = TextureConfig {
        required_mips: 0,
        ..texture_config()
    };
    let optional = load_texture(context, &[4, 5, 6, 255], &config).unwrap();
    // Cancelling the required load leaves only optional work behind, which
    // the frame gate must never wait on.
    assert_eq!(cancel_texture_load(context, required), Ok(()));
    with_context_mut(context, |state| {
        assert!(state.texture_pipeline.pending().contains_key(&optional));
        assert!(!super::super::texture_manager::has_required_pending(
            &state.texture_pipeline
        ));
        Ok(())
    })
    .unwrap();
    gate.wait();
    assert_eq!(cancel_texture_load(context, optional), Ok(()));
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn optional_zero_progresses_to_device_ready_without_gate_wait() {
    use crate::{UploadResource, UploadStatus};
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let config = TextureConfig {
        required_mips: 0,
        ..texture_config()
    };
    let texture = load_texture(context, &[1, 2, 3, 255], &config).unwrap();
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    // Ordinary event polling alone drives the optional load: no frame gate
    // wait and no explicit idle call precede readiness here.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut ready = 0;
    loop {
        while let Some(event) = poll_upload_event(context).unwrap() {
            if matches!(
                event,
                crate::UploadEvent {
                    resource: UploadResource::Texture(event_texture),
                    status: UploadStatus::DeviceReady,
                } if event_texture == texture
            ) {
                ready += 1;
            }
        }
        if ready == 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "optional texture never reached DeviceReady"
        );
        std::thread::yield_now();
    }
    // The first real coarse mip is resident and exposed; fallback no longer
    // samples once publication completes.
    assert_eq!(texture_residency(context, texture), Ok((1, 1)));
    while poll_upload_event(context).unwrap().is_some() {}
    let mut reemitted = 0;
    while let Some(event) = poll_upload_event(context).unwrap() {
        if matches!(
            event,
            crate::UploadEvent {
                resource: UploadResource::Texture(event_texture),
                status: UploadStatus::DeviceReady,
            } if event_texture == texture
        ) {
            reemitted += 1;
        }
    }
    assert_eq!(reemitted, 0);
    with_context_mut(context, |state| {
        assert_eq!(state.transfer_pool.in_use(), 0);
        Ok(())
    })
    .unwrap();
    destroy_surface(context, surface).unwrap();
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn mixed_optional_and_required_textures_reach_device_ready() {
    use crate::{UploadResource, UploadStatus};
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    let optional_config = TextureConfig {
        required_mips: 0,
        ..texture_config()
    };
    // The optional load admits first so any head-of-line block would stall
    // the required load behind it.
    let optional = load_texture(context, &[1, 2, 3, 255], &optional_config).unwrap();
    let required = load_texture(context, &[4, 5, 6, 255], &texture_config()).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut ready = [0, 0];
    loop {
        while let Some(event) = poll_upload_event(context).unwrap() {
            if let crate::UploadEvent {
                resource: UploadResource::Texture(event_texture),
                status: UploadStatus::DeviceReady,
            } = event
            {
                if event_texture == optional {
                    ready[0] += 1;
                } else if event_texture == required {
                    ready[1] += 1;
                }
            }
        }
        if ready == [1, 1] {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "mixed loads stalled at {ready:?}"
        );
        std::thread::yield_now();
    }
    assert_eq!(texture_residency(context, optional), Ok((1, 1)));
    assert_eq!(texture_residency(context, required), Ok((1, 1)));
    destroy_surface(context, surface).unwrap();
    assert_eq!(destroy_context(context), Ok(()));
}
