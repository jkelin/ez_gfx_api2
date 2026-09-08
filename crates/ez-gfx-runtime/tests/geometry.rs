//! Runtime geometry allocation contracts.

use ez_gfx_core::handle::{LocalHandle, PackedHandle};
use ez_gfx_hal::{CompletionToken, QueueKind};
use ez_gfx_runtime::geometry::{GeometryError, GeometryManager, StagingPool};

fn handle(slot: u32, generation: u32) -> PackedHandle {
    PackedHandle::child(
        LocalHandle::new(1, 1).unwrap(),
        LocalHandle::new(slot, generation).unwrap(),
    )
    .unwrap()
}

#[test]
fn free_ranges_coalesce_and_reuse_without_fragmentation() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("mesh", 64, 16).unwrap();
    let first = geometry
        .reserve_vertices("mesh", 1, 16, handle(0, 1))
        .unwrap();
    let second = geometry
        .reserve_vertices("mesh", 2, 16, handle(1, 1))
        .unwrap();
    geometry.free_vertices("mesh", first.handle).unwrap();
    geometry.free_vertices("mesh", second.handle).unwrap();

    let reused = geometry
        .reserve_vertices("mesh", 4, 16, handle(2, 1))
        .unwrap();
    assert_eq!((reused.first_element, reused.element_count), (0, 4));
}

#[test]
fn stale_double_free_and_wrong_heap_are_rejected() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("a", 64, 16).unwrap();
    geometry.create_vertex_heap("b", 64, 16).unwrap();
    let allocation = geometry.reserve_vertices("a", 1, 16, handle(0, 1)).unwrap();

    assert_eq!(
        geometry.free_vertices("b", allocation.handle),
        Err(GeometryError::WrongHeap)
    );
    geometry.free_vertices("a", allocation.handle).unwrap();
    assert_eq!(
        geometry.free_vertices("a", allocation.handle),
        Err(GeometryError::UnknownAllocation)
    );
    assert_eq!(
        geometry.allocation(handle(0, 2)),
        Err(GeometryError::UnknownAllocation)
    );
}

#[test]
fn index_heap_is_single_and_uses_u32_elements() {
    let mut geometry = GeometryManager::new();
    geometry.create_index_heap(32).unwrap();
    assert_eq!(
        geometry.create_index_heap(32),
        Err(GeometryError::DuplicateHeap)
    );
    let upload = geometry.reserve_indices(3, handle(0, 1)).unwrap();
    assert_eq!((upload.first_element, upload.byte_size), (0, 12));
}

#[test]
fn readiness_tokens_are_per_allocation_and_heap_monotonic() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("mesh", 64, 16).unwrap();
    let upload = geometry
        .reserve_vertices("mesh", 1, 16, handle(0, 1))
        .unwrap();
    let seven = CompletionToken::new(QueueKind::Transfer, 7).unwrap();
    let six = CompletionToken::new(QueueKind::Transfer, 6).unwrap();
    geometry.mark_ready(upload.handle, seven).unwrap();
    assert_eq!(geometry.vertex_ready("mesh"), Some(seven));
    assert_eq!(
        geometry.mark_ready(upload.handle, six),
        Err(GeometryError::TimelineRegression)
    );
}

#[test]
fn staging_pool_grows_without_fixed_slot_admission() {
    let mut pool = StagingPool::new();
    let slots: Vec<_> = (0..1_000).map(|_| pool.checkout(64, 0).unwrap()).collect();
    assert_eq!(slots.len(), 1_000);
}

#[test]
fn staging_pool_reuses_only_completed_compatible_slots() {
    let mut pool = StagingPool::new();
    let first = pool.checkout(64, 0).unwrap();
    pool.retire(first, CompletionToken::new(QueueKind::Transfer, 3).unwrap())
        .unwrap();
    let second = pool.checkout(64, 2).unwrap();
    assert_ne!(first, second);
    pool.release_unsubmitted(second).unwrap();
    assert_eq!(pool.checkout(64, 3).unwrap(), first);
}
