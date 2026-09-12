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

#[cfg(feature = "basis")]
#[test]
fn standalone_basis_transcodes_base_level_to_exact_blocks() {
    let data = include_bytes!("../../ez-gfx-texture-manager/tests/fixtures/rust-logo-etc.basis");
    for (format, bytes) in [
        (BlockFormat::Bc1, 64 * 64 / 2),
        (BlockFormat::Bc3, 64 * 64),
        (BlockFormat::Bc7, 64 * 64),
        (BlockFormat::Astc4x4, 64 * 64),
    ] {
        assert_eq!(
            ez_gfx_assets::transcode_basis(data, format).unwrap().len(),
            bytes
        );
    }
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
