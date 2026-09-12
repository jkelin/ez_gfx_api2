use super::*;
#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_parked_optional_decode_records_fallback_frame() {
    // Distinct process-global source IDs keep parallel backend variants
    // from colliding on the shared decoder registry.
    parked_optional_decode_records_fallback_frame(1, 200);
}

#[cfg(windows)]
#[test]
fn dx12_parked_optional_decode_records_fallback_frame() {
    parked_optional_decode_records_fallback_frame(2, 201);
}

/// Releases a parked decode worker exactly once without ever blocking: the
/// one-shot send pairs with the worker's receive whether the worker starts
/// before or after a timeout or panic.
struct ReleaseGuard(Option<std::sync::mpsc::Sender<()>>);

impl ReleaseGuard {
    fn release(mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

impl Drop for ReleaseGuard {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

/// Unregisters a process-global custom source on scope exit so a failure
/// cannot leak registry state into later tests.
struct DecoderRegistration(u8);

impl Drop for DecoderRegistration {
    fn drop(&mut self) {
        let _ = Context::unregister_texture_decoder(self.0);
    }
}

fn parked_optional_decode_records_fallback_frame(backend: u8, source: u8) {
    use ez_gfx::{UploadEvent, UploadResource, UploadStatus};

    // Two workers: one parks on the custom decode while the owner thread
    // drives a genuine heap-demanding frame.
    let native = TestContext::create_with_validation_and_workers(backend, backend == 1, 2);
    let context = ContextHandle::from_raw(native.context).unwrap();
    let surface = SurfaceHandle::from_raw(native.surface).unwrap();
    let quad = Quad::create(context, surface, cube_artifact());
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    // The receiver crosses into the decoder closure, which must be `Sync`.
    let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
    // Armed before registration: the send never blocks, so cleanup cannot
    // hang whether the worker starts, parks late, or never runs.
    let guard = ReleaseGuard(Some(release_tx));
    // Declared after the release guard so it drops first: the registry entry
    // is removed before the worker is freed. Admitted loads retain their
    // decoder, so a freed worker still decodes safely.
    let _registration = DecoderRegistration(source);
    Context::register_texture_decoder(
        source,
        Arc::new(move |bytes: &[u8], _| {
            // Signal arrival, then park until the test releases the worker
            // after recording.
            let _ = started_tx.send(());
            let _ = release_rx.lock().unwrap().recv();
            assert_eq!(bytes.len(), 16);
            Ok(DecodedTexture {
                width: 4,
                height: 4,
                mip_count: 1,
                format: TextureFormat::Rgba8Unorm,
                mips: vec![DecodedMip {
                    width: 4,
                    height: 4,
                    bytes: [0_u8, 255, 0, 255].repeat(16),
                }],
            })
        }),
    )
    .unwrap();
    let config = TextureConfig {
        source: TextureSource::Custom(source),
        generate_mips: false,
        required_mips: 0,
        width: 0,
        height: 0,
        mip_count: 0,
        destination: TextureDestination::Rgba8Unorm,
        sampler: TextureSamplerDesc {
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            max_anisotropy: 1.0,
            address_u: SamplerAddressMode::Clamp,
            address_v: SamplerAddressMode::Clamp,
            address_w: SamplerAddressMode::Clamp,
        },
    };
    let texture = load_texture(context, &[7_u8; 16], &config).unwrap();
    // Prove the worker parked before recording: the release guard is already
    // armed, so a timeout here still frees a late worker through Drop.
    started_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("decode worker never parked");
    // The stable alias is valid from admission, but nothing real is resident
    // while the worker is parked.
    let _ = texture_binding(context, texture).unwrap();
    assert_eq!(texture_residency(context, texture), Err(Error::NotReady));
    // A genuine heap-demanding frame records and finishes while parked: the
    // gate never waits for optional work.
    let parked_pixels = match quad.draw_status(texture, true) {
        Ok(()) => Some(frame_readback(context).unwrap()),
        Err(Error::Unsupported) => {
            // Headless Linux: recording passed the submission gate while
            // parked; only presentation is unsupported.
            None
        }
        Err(status) => panic!("parked frame failed: {status:?}"),
    };
    assert_eq!(texture_residency(context, texture), Err(Error::NotReady));
    // Release the worker, then observe exactly one DeviceReady and real
    // coarse residency through ordinary event polling.
    guard.release();
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut ready = 0;
    loop {
        while let Some(event) = poll_upload_event(context).unwrap() {
            if matches!(
                event,
                UploadEvent {
                    resource: UploadResource::Texture(event_texture),
                    status: UploadStatus::DeviceReady,
                } if event_texture == texture
            ) {
                ready += 1;
            }
        }
        if texture_residency(context, texture) == Ok((1, 1)) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "optional texture never reached DeviceReady"
        );
        std::thread::yield_now();
    }
    while let Some(event) = poll_upload_event(context).unwrap() {
        if matches!(
            event,
            UploadEvent {
                resource: UploadResource::Texture(event_texture),
                status: UploadStatus::DeviceReady,
            } if event_texture == texture
        ) {
            ready += 1;
        }
    }
    assert_eq!(ready, 1, "DeviceReady must fire exactly once");
    if let Some(parked) = parked_pixels {
        quad.draw(texture, true);
        let real = frame_readback(context).unwrap();
        assert_ne!(
            parked, real,
            "parked frame must sample fallback, not real texels"
        );
    }
    unload_texture(context, texture);
    assert_eq!(wait_idle(context), Ok(()));
}
