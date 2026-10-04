#[cfg(not(target_vendor = "apple"))]
fn texture_config() -> TextureConfig {
    TextureConfig {
        source: TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        generate_mips: false,
        required_mips: 1,
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
        state.decode_textures.set_decode_gate(Some(gate.clone()));
        Ok(())
    })
    .unwrap();

    let texture = load_texture(context, &[1, 2, 3, 4], &texture_config()).unwrap();
    assert_eq!(
        poll_upload_event(context),
        Ok(Some(crate::UploadEvent {
            resource: crate::UploadResource::Texture(texture),
            status: crate::UploadStatus::SourceStaged,
        }))
    );
    assert_eq!(texture_binding(context, texture), Ok(0));

    assert_eq!(cancel_texture_load(context, texture), Ok(()));
    assert_eq!(
        poll_upload_event(context),
        Ok(Some(crate::UploadEvent {
            resource: crate::UploadResource::Texture(texture),
            status: crate::UploadStatus::Cancelled,
        }))
    );
    assert_eq!(poll_upload_event(context), Ok(None));
    assert_eq!(
        texture_binding(context, texture),
        Err(Error::Lifecycle(LifecycleError::StaleHandle))
    );
    assert_eq!(
        cancel_texture_load(context, texture),
        Err(Error::InvalidArgument)
    );

    gate.wait();
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(windows)]
#[test]
fn natural_texture_submissions_need_one_context_wait_and_keep_lossless_events() {
    let context = dx12_context();
    let mut textures = Vec::new();
    for color in 0_u8..9 {
        textures
            .push(load_texture(context, &[color, color, color, 255], &texture_config()).unwrap());
    }

    assert_eq!(wait_idle(context), Ok(()));
    let bindings = textures
        .iter()
        .map(|texture| texture_binding(context, *texture).unwrap())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(bindings.len(), textures.len());
    assert_eq!(resource_diagnostics(context).unwrap().pending_textures, 0);

    let mut staged = 0;
    let mut ready = 0;
    while let Some(event) = poll_upload_event(context).unwrap() {
        match event.status {
            crate::UploadStatus::SourceStaged => staged += 1,
            crate::UploadStatus::DeviceReady => ready += 1,
            status => panic!("unexpected terminal upload status: {status:?}"),
        }
    }
    assert_eq!((staged, ready), (textures.len(), textures.len()));
    assert_eq!(destroy_context(context), Ok(()));
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
        Err(Error::InvalidArgument)
    );
    assert_eq!(
        map_texture_update_error(ez_gfx_hal::AllocationError::OutOfMemory),
        Error::QueueFull
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
        let texture = TextureHandle::from_packed(packed).map_err(|_| Error::NativeFailure)?;
        state.texture_pipeline.ready_mut().insert(
            texture,
            CompletionToken::new(QueueKind::TextureTransfer, 3)
                .map_err(|_| Error::NativeFailure)?,
        );
        state.texture_pipeline.handoffs_mut().insert(
            texture,
            Instant::now()
                .checked_sub(std::time::Duration::from_micros(10))
                .expect("ten microseconds fits in the monotonic clock"),
        );

        record_texture_ready(state, texture, 2);
        assert!(state.texture_pipeline.ready().contains_key(&texture));
        record_texture_ready(state, texture, 3);
        // GPU completion alone is insufficient while a frame still prevents publication.
        assert!(state.texture_pipeline.ready().contains_key(&texture));
        assert_eq!(
            state
                .texture_pipeline
                .telemetry()
                .snapshot()
                .handoff_latency_microseconds,
            0
        );
        state.texture_pipeline.published_mut().insert(texture, 1);
        record_texture_ready(state, texture, 3);
        assert!(!state.texture_pipeline.ready().contains_key(&texture));
        Ok(texture)
    })
    .unwrap();

    let snapshot = texture_upload_telemetry(context).unwrap();
    assert!(snapshot.handoff_latency_microseconds >= 10);
    with_context_mut(context, |state| {
        record_texture_ready(state, texture, u64::MAX);
        assert_eq!(
            state
                .texture_pipeline
                .telemetry()
                .snapshot()
                .handoff_latency_microseconds,
            snapshot.handoff_latency_microseconds
        );
        Ok(())
    })
    .unwrap();
    assert_eq!(destroy_context(context), Ok(()));
}
