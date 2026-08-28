use ez_gfx_core::handle::{GenerationalArena, HandleError, HandleParts, LocalHandle, PackedHandle};

#[test]
fn packed_handles_round_trip_and_encode_ownership() {
    let owner = LocalHandle::new(3, 7).unwrap();
    let child = LocalHandle::new(4, 5).unwrap();

    let context = PackedHandle::context(owner).unwrap();
    let resource = PackedHandle::child(owner, child).unwrap();

    assert_eq!(context.parts().unwrap(), HandleParts::Context(owner));
    assert_eq!(
        resource.parts().unwrap(),
        HandleParts::Child { owner, child }
    );
    assert_ne!(
        resource,
        PackedHandle::child(LocalHandle::new(8, 7).unwrap(), child).unwrap()
    );
}

#[test]
fn packed_handles_reject_zero_inconsistent_and_out_of_range_fields() {
    assert_eq!(PackedHandle::from_raw(0), Err(HandleError::Null));
    assert_eq!(PackedHandle::from_raw(1), Err(HandleError::ZeroGeneration));
    assert_eq!(
        PackedHandle::from_raw((1_u64 << 52) | 1 | (1 << 20)),
        Err(HandleError::Malformed)
    );
    assert_eq!(
        PackedHandle::context(LocalHandle::new((1 << 20) - 1, 1).unwrap()),
        Err(HandleError::SlotOutOfRange)
    );
    assert_eq!(
        PackedHandle::child(
            LocalHandle::new(0, 1).unwrap(),
            LocalHandle::new((1 << 12) - 1, 1).unwrap()
        ),
        Err(HandleError::SlotOutOfRange)
    );
}

#[test]
fn arena_reuse_invalidates_stale_handles_and_clear_keeps_them_invalid() {
    let mut arena = GenerationalArena::new();
    let first = arena.insert(10).unwrap();
    assert_eq!(arena.remove(first).unwrap(), 10);

    let second = arena.insert(20).unwrap();
    assert_eq!(first.slot(), second.slot());
    assert_ne!(first.generation(), second.generation());
    assert_eq!(arena.get(first), Err(HandleError::Stale));

    arena.clear().unwrap();
    assert_eq!(arena.get(second), Err(HandleError::Stale));
    assert!(arena.is_empty());
}

#[test]
fn local_handles_reject_zero_generation() {
    assert_eq!(LocalHandle::new(0, 0), Err(HandleError::ZeroGeneration));
}
