//! Runtime integration and contract tests.

use ez_gfx_hal::{CompletionToken, QueueKind};
use ez_gfx_runtime::geometry::{GeometryError, GeometryManager, StagingPool};

#[test]
fn heaps_validate_names_stride_uniqueness_and_capacity() {
    let mut geometry = GeometryManager::new();
    assert_eq!(
        geometry.create_vertex_heap("", 64, 16),
        Err(GeometryError::InvalidName)
    );
    assert_eq!(
        geometry.create_vertex_heap("mesh", 64, 0),
        Err(GeometryError::InvalidStride)
    );
    geometry.create_vertex_heap("mesh", 64, 16).unwrap();
    assert_eq!(
        geometry.create_vertex_heap("mesh", 64, 16),
        Err(GeometryError::DuplicateHeap)
    );
    assert_eq!(
        geometry.reserve_vertices("mesh", 2, 12),
        Err(GeometryError::StrideMismatch)
    );
    assert_eq!(
        geometry.reserve_vertices("mesh", 5, 16),
        Err(GeometryError::CapacityExceeded)
    );
    let first = geometry.reserve_vertices("mesh", 2, 16).unwrap();
    let second = geometry.reserve_vertices("mesh", 2, 16).unwrap();
    assert_eq!(
        (first.first_element, first.byte_offset, first.byte_size),
        (0, 0, 32)
    );
    assert_eq!((second.first_element, second.byte_offset), (2, 32));
}

#[test]
fn index_heap_is_single_and_uses_u32_elements() {
    let mut geometry = GeometryManager::new();
    geometry.create_index_heap(32).unwrap();
    assert_eq!(
        geometry.create_index_heap(32),
        Err(GeometryError::DuplicateHeap)
    );
    let upload = geometry.reserve_indices(3).unwrap();
    assert_eq!((upload.first_element, upload.byte_size), (0, 12));
    assert_eq!(
        geometry.reserve_indices(6),
        Err(GeometryError::CapacityExceeded)
    );
}

#[test]
fn readiness_tokens_are_per_resource_and_monotonic() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("a", 64, 16).unwrap();
    geometry.create_vertex_heap("b", 64, 16).unwrap();
    let seven = CompletionToken::new(QueueKind::Transfer, 7).unwrap();
    let six = CompletionToken::new(QueueKind::Transfer, 6).unwrap();
    geometry.mark_vertex_ready("a", seven).unwrap();
    assert_eq!(geometry.vertex_ready("a"), Some(seven));
    assert_eq!(geometry.vertex_ready("b"), None);
    assert_eq!(
        geometry.mark_vertex_ready("a", six),
        Err(GeometryError::TimelineRegression)
    );
}

#[test]
fn staging_pool_reuses_only_completed_compatible_slots() {
    let mut pool = StagingPool::new(2).unwrap();
    let first = pool.checkout(64, 0).unwrap();
    pool.retire(first, CompletionToken::new(QueueKind::Transfer, 3).unwrap())
        .unwrap();
    let second = pool.checkout(32, 2).unwrap();
    assert_ne!(first, second);
    assert_eq!(
        pool.checkout(8, 2),
        Err(GeometryError::StagingPoolExhausted)
    );
    pool.release_unsubmitted(second).unwrap();
    assert_eq!(pool.checkout(32, 3).unwrap(), first);
}

#[test]
fn staging_pool_allocates_power_of_two_buckets() {
    let mut pool = StagingPool::new(2).unwrap();
    let small = pool.checkout(1, 0).unwrap();
    let large = pool.checkout(65 * 1024, 0).unwrap();

    assert_eq!(pool.slot_capacity(small).unwrap(), 64 * 1024);
    assert_eq!(pool.slot_capacity(large).unwrap(), 128 * 1024);
}

#[test]
fn staging_pool_rejects_non_transfer_retirement_tokens() {
    let mut pool = StagingPool::new(1).unwrap();
    let slot = pool.checkout(64, 0).unwrap();

    assert!(
        pool.retire(slot, CompletionToken::new(QueueKind::Compute, 1).unwrap())
            .is_err()
    );
    pool.release_unsubmitted(slot).unwrap();
}

#[test]
fn rollback_rejects_a_forged_tail_subrange() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("mesh", 64, 16).unwrap();
    let upload = geometry.reserve_vertices("mesh", 2, 16).unwrap();
    let forged = ez_gfx_runtime::geometry::GeometryUpload {
        first_element: 1,
        byte_offset: 16,
        byte_size: 16,
    };

    assert_eq!(
        geometry.rollback_vertices("mesh", forged),
        Err(GeometryError::InvalidRollback)
    );
    assert_eq!(geometry.rollback_vertices("mesh", upload), Ok(()));
}

#[test]
fn rollback_does_not_restore_an_older_reservation() {
    let mut geometry = GeometryManager::new();
    geometry.create_vertex_heap("mesh", 64, 16).unwrap();
    let older = geometry.reserve_vertices("mesh", 1, 16).unwrap();
    let latest = geometry.reserve_vertices("mesh", 1, 16).unwrap();

    assert_eq!(geometry.rollback_vertices("mesh", latest), Ok(()));
    assert_eq!(
        geometry.rollback_vertices("mesh", older),
        Err(GeometryError::InvalidRollback)
    );
}
