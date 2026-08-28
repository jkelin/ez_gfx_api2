use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_hal::{CompletionToken, QueueKind};
use ez_gfx_runtime::{cache::*, descriptor::*};

#[test]
fn bindless_registry_rejects_stale_generation_and_capacity_overflow() {
    let mut registry = BindlessRegistry::new(2).unwrap();
    let first = registry.insert("texture-a").unwrap();
    let second = registry.insert("texture-b").unwrap();
    assert_eq!(
        registry.insert("texture-c"),
        Err(DescriptorError::CapacityExhausted)
    );
    assert_eq!(registry.remove(first).unwrap(), "texture-a");
    let reused = registry.insert("texture-c").unwrap();
    assert_eq!(first.slot(), reused.slot());
    assert_ne!(first.generation(), reused.generation());
    assert_eq!(registry.get(first), Err(DescriptorError::StaleHandle));
    assert_eq!(registry.get(second), Ok(&"texture-b"));
}

#[test]
fn frame_arena_resets_only_after_its_completion_token() {
    let mut arena = FrameDescriptorArena::new(8).unwrap();
    assert_eq!(
        arena.allocate(3).unwrap(),
        DescriptorRange { start: 0, count: 3 }
    );
    arena
        .retire(CompletionToken::new(QueueKind::Graphics, 7).unwrap())
        .unwrap();
    assert_eq!(
        arena.retire(CompletionToken::new(QueueKind::Graphics, 5).unwrap()),
        Err(DescriptorError::AlreadyRetired)
    );
    assert_eq!(
        arena.reset(QueueKind::Graphics, 5),
        Err(DescriptorError::GpuWorkPending)
    );
    assert_eq!(
        arena.reset(QueueKind::Graphics, 6),
        Err(DescriptorError::GpuWorkPending)
    );
    arena.reset(QueueKind::Graphics, 7).unwrap();
    assert_eq!(
        arena.allocate(8).unwrap(),
        DescriptorRange { start: 0, count: 8 }
    );
}

#[test]
fn cache_envelope_round_trips_and_rejects_identity_corruption_and_size() {
    let identity = CacheIdentity::new(
        Backend::Vulkan,
        [1; 16],
        "driver-551",
        SemanticProfile::V1,
        3,
    )
    .unwrap();
    let envelope = PipelineCacheEnvelope::new(identity.clone(), vec![1, 2, 3, 4]).unwrap();
    let bytes = envelope.encode().unwrap();
    assert_eq!(
        PipelineCacheEnvelope::decode(&bytes, &identity)
            .unwrap()
            .payload(),
        &[1, 2, 3, 4]
    );

    let other = CacheIdentity::new(
        Backend::Vulkan,
        [2; 16],
        "driver-551",
        SemanticProfile::V1,
        3,
    )
    .unwrap();
    assert_eq!(
        PipelineCacheEnvelope::decode(&bytes, &other),
        Err(CacheError::IdentityMismatch)
    );
    let mut corrupt = bytes;
    *corrupt.last_mut().unwrap() ^= 1;
    assert_eq!(
        PipelineCacheEnvelope::decode(&corrupt, &identity),
        Err(CacheError::ChecksumMismatch)
    );
    assert_eq!(
        PipelineCacheEnvelope::new(identity, vec![0; MAX_CACHE_PAYLOAD_BYTES + 1]),
        Err(CacheError::TooLarge)
    );
}

#[test]
fn pipeline_keys_are_value_identity_not_descriptor_ownership() {
    let key = PipelineKey::new([7; 32], [8; 32], Backend::Dx12, 3, 1);
    let same = PipelineKey::new([7; 32], [8; 32], Backend::Dx12, 3, 1);
    assert_eq!(key, same);
}
