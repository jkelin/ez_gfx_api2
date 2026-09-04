//! Asset format, residency, and worker queue contract tests.
use ez_gfx_assets::{
    AssetError, AssetEvent, BlockFormat, EventOutcome, EventPhase, EventQueue, MipChain, Region,
    TextureId, TextureState, parse_ktx2, validate_block_payload,
};

#[test]
fn validates_blocks_and_regions() {
    assert!(validate_block_payload(BlockFormat::Bc1, 8, 8, 0, &[0; 32]).is_ok());
    assert!(validate_block_payload(BlockFormat::Bc1, 7, 8, 0, &[0; 16]).is_err());
    assert!(Region::new(0, 0, 4, 4, 8, 8, BlockFormat::Bc1).is_ok());
    assert!(Region::new(1, 0, 4, 4, 8, 8, BlockFormat::Bc1).is_err());
}
#[test]
fn block_payload_rejects_zero_dimensions() {
    assert_eq!(
        validate_block_payload(BlockFormat::Bc1, 0, 8, 0, &[0; 16]),
        Err(AssetError::InvalidDimensions)
    );
    assert_eq!(
        validate_block_payload(BlockFormat::Bc1, 8, 0, 0, &[0; 16]),
        Err(AssetError::InvalidDimensions)
    );
}

#[test]
fn mip_progression_starts_coarsest() {
    let mut c = MipChain::new(TextureId::try_new(7).unwrap(), 16, 8, 3, BlockFormat::Bc7).unwrap();
    assert_eq!(c.state(), TextureState::Empty);
    c.upload_mip(2, vec![0; 16]).unwrap();
    assert_eq!(c.state(), TextureState::SampleReady);
    c.upload_mip(0, vec![0; 128]).unwrap();
    c.upload_mip(1, vec![0; 32]).unwrap();
    assert_eq!(c.state(), TextureState::FullyResident);
}
#[test]
fn queue_structured_events_and_overflow() {
    let q = EventQueue::new(1).unwrap();
    let e = AssetEvent {
        correlation_id: 9,
        job_id: 1,
        texture: TextureId::try_new(7).unwrap(),
        phase: EventPhase::Decode,
        bytes: 1,
        outcome: EventOutcome::Completed,
        error: None,
    };
    q.push(e).unwrap();
    assert!(q.push(e).is_err());
    assert_eq!(q.pop(), Some(e));
    q.cancel();
    assert!(matches!(q.push(e), Err(AssetError::Cancelled)));
}
#[test]
fn ids_and_zero_queue_rejected() {
    assert!(EventQueue::new(0).is_err());
    assert!(TextureId::try_new(0).is_err());
}
#[test]
fn parses_direct_ktx2_and_rejects_layers_or_scheme() {
    let mut d = vec![0u8; 136];
    d[..12].copy_from_slice(b"\xabKTX 20\xbb\r\n\x1a\n");
    d[12..16].copy_from_slice(&131u32.to_le_bytes());
    d[20..24].copy_from_slice(&8u32.to_le_bytes());
    d[24..28].copy_from_slice(&8u32.to_le_bytes());
    d[36..40].copy_from_slice(&1u32.to_le_bytes());
    d[40..44].copy_from_slice(&1u32.to_le_bytes());
    d[80..88].copy_from_slice(&104u64.to_le_bytes());
    d[88..96].copy_from_slice(&32u64.to_le_bytes());
    assert!(matches!(
        parse_ktx2(&d),
        Ok(ez_gfx_assets::Ktx2Payload::Direct { .. })
    ));
    d[32..36].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(parse_ktx2(&d), Err(AssetError::InvalidKtx2)));
    d[32..36].copy_from_slice(&0u32.to_le_bytes());
    d[12..16].copy_from_slice(&0u32.to_le_bytes());
    d[44..48].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(parse_ktx2(&d), Err(AssetError::UnsupportedKtx2)));
}
#[test]
fn cpu_pool_emits_completion_and_error_once() {
    use std::{
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };
    let p = ez_gfx_assets::CpuPool::new(2, 2, 1024).unwrap();
    let q = Arc::new(EventQueue::new(2).unwrap());
    let e = AssetEvent {
        correlation_id: 4,
        job_id: 8,
        texture: TextureId::try_new(9).unwrap(),
        phase: EventPhase::Transcode,
        bytes: 0,
        outcome: EventOutcome::Failed,
        error: None,
    };
    p.submit_event(q.clone(), e, || Ok(12)).unwrap();
    p.submit_event(q.clone(), e, || Err(AssetError::InvalidPayload))
        .unwrap();
    let start = Instant::now();
    let mut got = Vec::new();
    while got.len() < 2 && start.elapsed() < Duration::from_secs(1) {
        if let Some(x) = q.pop() {
            got.push(x);
        } else {
            thread::yield_now();
        }
    }
    assert_eq!(got.len(), 2);
}
#[test]
fn accepted_shutdown_emits_cancelled() {
    use std::{
        sync::{Arc, mpsc},
        thread,
        time::{Duration, Instant},
    };
    let p = ez_gfx_assets::CpuPool::new(1, 1, 64).unwrap();
    let q = Arc::new(EventQueue::new(1).unwrap());
    let e = AssetEvent {
        correlation_id: 1,
        job_id: 1,
        texture: TextureId::try_new(1).unwrap(),
        phase: EventPhase::Decode,
        bytes: 1,
        outcome: EventOutcome::Failed,
        error: None,
    };
    let (tx, rx) = mpsc::channel();
    p.submit_event(q.clone(), e, move || {
        let _ = rx.recv();
        Ok(1)
    })
    .unwrap();
    p.shutdown();
    assert!(matches!(p.submit(|| {}), Err(AssetError::Cancelled)));
    q.cancel();
    tx.send(()).unwrap();
    let start = Instant::now();
    let mut got = None;
    while got.is_none() && start.elapsed() < Duration::from_secs(1) {
        got = q.pop();
        thread::yield_now();
    }
    assert_eq!(got.unwrap().outcome, EventOutcome::Cancelled);
}
#[test]
fn panic_releases_job_and_byte_capacity() {
    use std::{
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };
    let p = ez_gfx_assets::CpuPool::new(1, 1, 8).unwrap();
    let q = Arc::new(EventQueue::new(2).unwrap());
    let e = AssetEvent {
        correlation_id: 2,
        job_id: 2,
        texture: TextureId::try_new(2).unwrap(),
        phase: EventPhase::Decode,
        bytes: 8,
        outcome: EventOutcome::Failed,
        error: None,
    };
    p.submit_event(q.clone(), e, || -> Result<usize, AssetError> {
        panic!("test")
    })
    .unwrap();
    let start = Instant::now();
    let mut got = None;
    while got.is_none() && start.elapsed() < Duration::from_secs(1) {
        got = q.pop();
        thread::yield_now();
    }
    assert_eq!(got.unwrap().outcome, EventOutcome::Failed);
    let e2 = AssetEvent {
        correlation_id: 3,
        job_id: 3,
        texture: TextureId::try_new(3).unwrap(),
        phase: EventPhase::Decode,
        bytes: 8,
        outcome: EventOutcome::Failed,
        error: None,
    };
    assert!(p.submit_event(q, e2, || Ok(1)).is_ok());
}

#[test]
fn unpolled_completion_reservation_rejects_without_blocking() {
    let p = ez_gfx_assets::CpuPool::new(1, 2, 8).unwrap();
    let q = std::sync::Arc::new(EventQueue::new(1).unwrap());
    let event = AssetEvent {
        correlation_id: 7,
        job_id: 7,
        texture: TextureId::try_new(7).unwrap(),
        phase: EventPhase::Decode,
        bytes: 1,
        outcome: EventOutcome::Failed,
        error: None,
    };
    p.submit_event(q.clone(), event, || Ok(1)).unwrap();
    assert!(matches!(
        p.submit_event(q, event, || Ok(1)),
        Err(AssetError::QueueFull)
    ));
}
