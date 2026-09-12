//! Geometry manager allocation contracts behind the extraction boundary.

use ez_gfx_core::handle::{LocalHandle, PackedHandle};
use ez_gfx_geometry_manager::{GeometryError, GeometryManager};
use ez_gfx_hal::{CompletionToken, QueueKind};

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
fn growing_heaps_preserves_live_offsets_and_extends_free_space() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("mesh", 32, 16).unwrap();
    geometry.create_index_heap(8).unwrap();
    let vertices = geometry
        .reserve_vertices("mesh", 2, 16, handle(0, 1))
        .unwrap();
    let indices = geometry.reserve_indices(2, handle(1, 1)).unwrap();

    geometry.grow_vertex_heap("mesh", 64).unwrap();
    geometry.grow_index_heap(32).unwrap();

    assert_eq!(geometry.allocation(vertices.handle).unwrap(), vertices);
    assert_eq!(geometry.allocation(indices.handle).unwrap(), indices);
    assert_eq!(
        geometry
            .reserve_vertices("mesh", 2, 16, handle(2, 1))
            .unwrap()
            .first_element,
        2
    );
    assert_eq!(
        geometry
            .reserve_indices(6, handle(3, 1))
            .unwrap()
            .first_element,
        2
    );
}

#[test]
fn heap_growth_rejects_nonincreasing_capacity() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("mesh", 32, 16).unwrap();
    geometry.create_index_heap(16).unwrap();

    assert_eq!(
        geometry.grow_vertex_heap("mesh", 32),
        Err(GeometryError::InvalidCapacity)
    );
    assert_eq!(
        geometry.grow_index_heap(8),
        Err(GeometryError::InvalidCapacity)
    );
}
