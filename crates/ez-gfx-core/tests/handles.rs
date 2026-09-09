//! Contract tests for packed handles and generational arenas.
use core::mem::{align_of, size_of};

use ez_gfx_core::handle::{
    BufferHandle, ContextHandle, CounterBufferHandle, GenerationalArena, HandleError, HandleParts,
    LocalHandle, PackedHandle, RenderTargetHandle, ShaderHandle, SurfaceHandle, TextureHandle,
    VertexHeapHandle,
};

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
fn typed_handles_preserve_wire_layout_and_raw_values() {
    let owner = LocalHandle::new(3, 7).unwrap();
    let child = LocalHandle::new(4, 5).unwrap();
    let context_raw = PackedHandle::context(owner).unwrap().get();
    let resource_raw = PackedHandle::child(owner, child).unwrap().get();

    assert_eq!(
        ContextHandle::from_raw(context_raw).unwrap().into_raw(),
        context_raw
    );
    for (size, alignment, raw) in [
        (
            size_of::<SurfaceHandle>(),
            align_of::<SurfaceHandle>(),
            SurfaceHandle::from_raw(resource_raw).unwrap().into_raw(),
        ),
        (
            size_of::<ShaderHandle>(),
            align_of::<ShaderHandle>(),
            ShaderHandle::from_raw(resource_raw).unwrap().into_raw(),
        ),
        (
            size_of::<CounterBufferHandle>(),
            align_of::<CounterBufferHandle>(),
            CounterBufferHandle::from_raw(resource_raw)
                .unwrap()
                .into_raw(),
        ),
        (
            size_of::<BufferHandle>(),
            align_of::<BufferHandle>(),
            BufferHandle::from_raw(resource_raw).unwrap().into_raw(),
        ),
        (
            size_of::<VertexHeapHandle>(),
            align_of::<VertexHeapHandle>(),
            VertexHeapHandle::from_raw(resource_raw).unwrap().into_raw(),
        ),
        (
            size_of::<TextureHandle>(),
            align_of::<TextureHandle>(),
            TextureHandle::from_raw(resource_raw).unwrap().into_raw(),
        ),
        (
            size_of::<RenderTargetHandle>(),
            align_of::<RenderTargetHandle>(),
            RenderTargetHandle::from_raw(resource_raw)
                .unwrap()
                .into_raw(),
        ),
    ] {
        assert_eq!(
            (size, alignment, raw),
            (size_of::<u64>(), align_of::<u64>(), resource_raw)
        );
    }
    assert_eq!(
        (size_of::<ContextHandle>(), align_of::<ContextHandle>()),
        (size_of::<u64>(), align_of::<u64>())
    );
}

#[test]
fn typed_handles_reject_zero_and_the_wrong_handle_shape() {
    let owner = LocalHandle::new(3, 7).unwrap();
    let child = LocalHandle::new(4, 5).unwrap();
    let context_raw = PackedHandle::context(owner).unwrap().get();
    let resource_raw = PackedHandle::child(owner, child).unwrap().get();

    assert_eq!(ContextHandle::from_raw(0), Err(HandleError::Null));
    assert_eq!(SurfaceHandle::from_raw(0), Err(HandleError::Null));
    assert_eq!(
        ContextHandle::from_raw(resource_raw),
        Err(HandleError::ExpectedContext)
    );
    assert_eq!(
        SurfaceHandle::from_raw(context_raw),
        Err(HandleError::ExpectedResource)
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
