//! Decode-driver lifecycle and pipeline-integration contracts.

use ez_gfx_core::{
    capability::CompressionSupport,
    handle::{LocalHandle, PackedHandle, TextureHandle},
};
use ez_gfx_hal::{SamplerAddressMode, SamplerFilter, TextureSamplerDesc};
use ez_gfx_texture_manager::{
    DecodeDriver, DecodeDriverError, PendingUpload, QueuedDecode, SharedTransferPool,
    TextureConfig, TextureDecoder, TextureDestination, TextureError, TexturePipeline,
    TextureRegistry, TextureSource, WORKING_SET_BUDGET_BYTES, register_texture_decoder,
    unregister_texture_decoder,
};
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

fn handle(slot: u32) -> TextureHandle {
    let owner = LocalHandle::new(0, 1).unwrap();
    let child = LocalHandle::new(slot, 1).unwrap();
    TextureHandle::from_packed(PackedHandle::child(owner, child).unwrap()).unwrap()
}

fn config(source: TextureSource) -> TextureConfig {
    TextureConfig {
        source,
        generate_mips: false,
        required_mips: 1,
        width: 1,
        height: 1,
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
    }
}

fn queue(
    pipe: &mut TexturePipeline,
    registry: &mut TextureRegistry,
    slot: u32,
    cancelled: Arc<AtomicBool>,
) -> TextureHandle {
    let handle = handle(slot);
    let source = TextureSource::Rgba8 {
        width: 1,
        height: 1,
    };
    pipe.admit(
        handle,
        PendingUpload {
            id: registry.begin_upload().unwrap(),
            cancelled: cancelled.clone(),
            config: config(source),
            fallback_published: false,
            source_bytes: 4,
            decoded_bytes: None,
            admitted_at: Instant::now(),
        },
        QueuedDecode {
            handle,
            prepared: TextureDecoder::prepare(
                source,
                CompressionSupport::NONE,
                TextureDestination::Rgba8Unorm,
            )
            .unwrap(),
            bytes: vec![u8::try_from(slot).unwrap(), 2, 3, 255].into_boxed_slice(),
            generate: false,
            cancelled,
        },
    );
    handle
}

fn collect_until(
    driver: &mut DecodeDriver,
    pipe: &mut TexturePipeline,
    pool: &mut SharedTransferPool,
    count: usize,
) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while pipe.decoded().len() < count {
        driver.collect(pipe, pool);
        assert!(Instant::now() < deadline, "decode result timed out");
        std::thread::yield_now();
    }
}

#[test]
fn validates_worker_policy_without_eager_pool_construction() {
    let expected = std::thread::available_parallelism()
        .map_or(2, usize::from)
        .saturating_sub(1)
        .max(1);
    let mut default_driver = DecodeDriver::new(0).unwrap();
    assert_eq!(default_driver.worker_count(), expected);
    assert!(!default_driver.is_started());
    default_driver.ensure_started().unwrap();
    assert!(default_driver.is_started());

    let explicit = DecodeDriver::new(2).unwrap();
    assert_eq!(explicit.worker_count(), 2);
    assert!(!explicit.is_started());
    assert_eq!(
        DecodeDriver::new(u32::MAX).map(|_| ()),
        Err(DecodeDriverError::InvalidWorkerCount)
    );
}

#[test]
fn dispatch_collects_results_and_tracks_active_jobs() {
    let mut driver = DecodeDriver::new(2).unwrap();
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(8, 8).unwrap();
    let mut pool = SharedTransferPool::new(WORKING_SET_BUDGET_BYTES);
    let first = queue(
        &mut pipe,
        &mut registry,
        1,
        Arc::new(AtomicBool::new(false)),
    );
    let second = queue(
        &mut pipe,
        &mut registry,
        2,
        Arc::new(AtomicBool::new(false)),
    );

    let report = driver.dispatch(&mut pipe, &mut pool).unwrap();
    assert_eq!(report.admitted, 2);
    assert!(report.failed.is_empty());
    assert_eq!(driver.active_jobs(), 2);

    collect_until(&mut driver, &mut pipe, &mut pool, 2);
    assert!(pipe.decoded().contains_key(&first));
    assert!(pipe.decoded().contains_key(&second));
    assert_eq!(driver.active_jobs(), 0);
}

#[test]
fn cancellation_is_delivered_and_reservation_remains_for_terminal_handling() {
    let mut driver = DecodeDriver::new(1).unwrap();
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(8, 8).unwrap();
    let mut pool = SharedTransferPool::new(WORKING_SET_BUDGET_BYTES);
    let handle = queue(&mut pipe, &mut registry, 3, Arc::new(AtomicBool::new(true)));

    let report = driver.dispatch(&mut pipe, &mut pool).unwrap();
    assert_eq!(report.admitted, 1);
    collect_until(&mut driver, &mut pipe, &mut pool, 1);
    assert_eq!(
        pipe.decoded().get(&handle),
        Some(&Err(TextureError::NotFound))
    );
    // The terminal cancellation path owns the reservation release.
    assert_eq!(
        pool.texture_ledger(),
        ez_gfx_texture_manager::DECODE_RESERVATION_BYTES
    );
}

#[test]
fn worker_panic_becomes_invalid_data_and_does_not_poison_driver() {
    const SOURCE: u8 = 250;
    register_texture_decoder(SOURCE, Arc::new(|_, _| panic!("decode panic"))).unwrap();
    let prepared = TextureDecoder::prepare(
        TextureSource::Custom(SOURCE),
        CompressionSupport::NONE,
        TextureDestination::Rgba8Unorm,
    )
    .unwrap();
    unregister_texture_decoder(SOURCE).unwrap();

    let mut driver = DecodeDriver::new(1).unwrap();
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(8, 8).unwrap();
    let mut pool = SharedTransferPool::new(WORKING_SET_BUDGET_BYTES);
    let handle_value = handle(4);
    let cancelled = Arc::new(AtomicBool::new(false));
    pipe.admit(
        handle_value,
        PendingUpload {
            id: registry.begin_upload().unwrap(),
            cancelled: cancelled.clone(),
            config: config(TextureSource::Custom(SOURCE)),
            fallback_published: false,
            source_bytes: 1,
            decoded_bytes: None,
            admitted_at: Instant::now(),
        },
        QueuedDecode {
            handle: handle_value,
            prepared,
            bytes: vec![1].into_boxed_slice(),
            generate: false,
            cancelled,
        },
    );

    let report = driver.dispatch(&mut pipe, &mut pool).unwrap();
    assert_eq!(report.admitted, 1);
    collect_until(&mut driver, &mut pipe, &mut pool, 1);
    assert_eq!(
        pipe.decoded().get(&handle_value),
        Some(&Err(TextureError::InvalidData))
    );
    assert_eq!(driver.active_jobs(), 0);
}

#[test]
fn shutdown_prevents_new_dispatch_without_consuming_queue() {
    let mut driver = DecodeDriver::new(1).unwrap();
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(8, 8).unwrap();
    let mut pool = SharedTransferPool::new(WORKING_SET_BUDGET_BYTES);
    queue(
        &mut pipe,
        &mut registry,
        5,
        Arc::new(AtomicBool::new(false)),
    );
    driver.shutdown();
    assert_eq!(
        driver.dispatch(&mut pipe, &mut pool).unwrap_err(),
        DecodeDriverError::Shutdown
    );
    assert_eq!(pipe.queue().len(), 1);
    assert_eq!(driver.active_jobs(), 0);
    assert_eq!(pool.texture_ledger(), 0);
}
