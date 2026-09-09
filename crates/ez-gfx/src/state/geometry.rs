//! Named vertex and global index heap lifecycle.

use crate::Result;

use super::{
    AllocationRequest, CompletionToken, ContextHandle, ContextState, DEFAULT_STAGING_POLICY, Error,
    GeometryAllocation, GeometryError, IndexAllocationHandle, MemoryClass, NativeAllocation,
    NativeContext, ResourceKind, RetiredGeometry, RetiredGeometryRange, RetiredRangeGraphics,
    RetiredRangeKind, RetiredVertexHeap, UploadEvent, UploadResource, UploadStatus,
    VertexAllocationHandle, VertexHeapHandle, allocate_native, completed_native_frame_value,
    completed_transfer_native, copy_native, free_native_allocation, last_native_frame_completion,
    map_allocation, map_geometry, map_lifecycle, result_status, staging_bucket_size,
    with_context_mut, write_native,
};

const INITIAL_GEOMETRY_HEAP_BYTES: u64 = 64 * 1024;

/// Creates an auto-growing named vertex heap and returns its typed identity.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or native allocation fails.
pub fn create_vertex_heap(
    context: ContextHandle,
    name: &str,
    stride: u64,
) -> Result<VertexHeapHandle> {
    let capacity = initial_heap_capacity(stride)?;
    create_vertex_heap_with_capacity(context, name, capacity, stride)
}

/// Creates a fixed-initial-capacity heap for the C raw seam.
///
/// # Errors
/// Returns an error when validation, capacity, ownership, or allocation fails.
#[allow(
    dead_code,
    reason = "the C raw seam preserves caller-selected initial capacity"
)]
pub fn create_vertex_heap_with_capacity(
    context: ContextHandle,
    name: &str,
    capacity: u64,
    stride: u64,
) -> Result<VertexHeapHandle> {
    if capacity == 0 || stride == 0 {
        return Err(Error::InvalidArgument);
    }
    create_vertex_heap_impl(context, name, capacity, stride)
}

fn create_vertex_heap_impl(
    context: ContextHandle,
    name: &str,
    capacity: u64,
    stride: u64,
) -> Result<VertexHeapHandle> {
    with_context_mut(context, |context| {
        reclaim_retired_geometry(context)?;
        let heap_id = context.next_vertex_heap_id;
        let next_heap_id = heap_id.checked_add(1).ok_or(Error::NativeFailure)?;
        context
            .geometry
            .create_vertex_heap(name, capacity, stride)
            .map_err(map_geometry)?;
        let request = AllocationRequest::new(capacity, 16, MemoryClass::Device, false, None)
            .map_err(map_allocation)?;
        let allocation = match allocate_native(&mut context.native, request) {
            Ok(allocation) => allocation,
            Err(error) => {
                let _ = context.geometry.remove_vertex_heap(name);
                return Err(map_allocation(error));
            }
        };
        let packed = match context.identity.insert(ResourceKind::VertexHeap) {
            Ok(packed) => packed,
            Err(error) => {
                let _ = context.geometry.remove_vertex_heap(name);
                let _ = free_native_allocation(&mut context.native, allocation);
                return Err(map_lifecycle(error));
            }
        };
        let handle = VertexHeapHandle::from_packed(packed).map_err(|_| Error::NativeFailure)?;
        context.vertex_heaps.insert(
            name.to_owned(),
            GeometryAllocation {
                allocation,
                ready: None,
                size: capacity,
                heap_id: Some(heap_id),
            },
        );
        context.vertex_heap_handles.insert(handle, name.to_owned());
        context.next_vertex_heap_id = next_heap_id;
        Ok(handle)
    })
}

/// Retires a vertex heap identified by its owner-validated handle.
pub fn destroy_vertex_heap(context: ContextHandle, handle: VertexHeapHandle) {
    let _ = with_context_mut(context, |context| {
        reclaim_retired_geometry(context)?;
        context
            .identity
            .resolve(handle.packed(), ResourceKind::VertexHeap)
            .map_err(map_lifecycle)?;
        let name = context
            .vertex_heap_handles
            .get(&handle)
            .cloned()
            .ok_or(Error::InvalidContext)?;
        let heap_id = context
            .vertex_heaps
            .get(&name)
            .and_then(|heap| heap.heap_id)
            .ok_or(Error::InvalidContext)?;
        if context.frame_vertex_heaps.contains_key(&heap_id) {
            return Err(Error::NotReady);
        }
        let logical_removed = match context.geometry.remove_vertex_heap(&name) {
            Ok(()) => true,
            Err(GeometryError::HeapNotEmpty) => false,
            Err(error) => return Err(map_geometry(error)),
        };
        context.vertex_heap_handles.remove(&handle);
        let heap = context
            .vertex_heaps
            .remove(&name)
            .ok_or(Error::InvalidContext)?;
        context
            .identity
            .remove(handle.packed(), ResourceKind::VertexHeap)
            .map_err(map_lifecycle)?;
        if logical_removed {
            free_native_allocation(&mut context.native, heap.allocation).map_err(map_allocation)
        } else {
            context.retired_vertex_heaps.push(RetiredVertexHeap {
                name,
                allocation: heap.allocation,
            });
            Ok(())
        }
    });
}

/// Creates the context index heap.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
#[allow(
    dead_code,
    reason = "the C raw seam retains explicit index-heap creation while safe Rust creates it lazily"
)]
pub fn create_index_heap(context: ContextHandle, capacity: u64) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .geometry
            .create_index_heap(capacity)
            .map_err(map_geometry)?;
        let request = AllocationRequest::new(capacity, 16, MemoryClass::Device, false, None)
            .map_err(map_allocation)?;
        match allocate_native(&mut context.native, request) {
            Ok(allocation) => {
                context.index_heap = Some(GeometryAllocation {
                    allocation,
                    ready: None,
                    size: capacity,
                    heap_id: None,
                });
                Ok(())
            }
            Err(error) => {
                let _ = context.geometry.remove_index_heap();
                Err(map_allocation(error))
            }
        }
    }))
}

/// Destroys the context index heap.
#[allow(
    dead_code,
    reason = "the C raw seam explicitly destroys the singleton heap"
)]
pub fn destroy_index_heap(context: ContextHandle) {
    let _ = with_context_mut(context, |context| {
        if context.frame_index.is_some() {
            return Err(Error::NotReady);
        }
        context.geometry.remove_index_heap().map_err(map_geometry)?;
        let heap = context.index_heap.take().ok_or(Error::InvalidArgument)?;
        free_native_allocation(&mut context.native, heap.allocation).map_err(map_allocation)
    });
}

fn initial_heap_capacity(stride: u64) -> Result<u64> {
    if stride == 0 || stride > u64::from(u32::MAX) {
        return Err(Error::InvalidArgument);
    }
    Ok(INITIAL_GEOMETRY_HEAP_BYTES.max(stride))
}

fn grown_capacity(current: u64, appended_bytes: u64) -> Result<u64> {
    let required = current
        .checked_add(appended_bytes)
        .ok_or(Error::InvalidArgument)?;
    Ok(current.saturating_mul(2).max(required))
}

fn ensure_index_heap(context: &mut ContextState, required_bytes: u64) -> Result<()> {
    if context.index_heap.is_some() {
        return Ok(());
    }
    let capacity = initial_heap_capacity(4)?.max(required_bytes);
    context
        .geometry
        .create_index_heap(capacity)
        .map_err(map_geometry)?;
    let request = AllocationRequest::new(capacity, 16, MemoryClass::Device, false, None)
        .map_err(map_allocation)?;
    match allocate_native(&mut context.native, request) {
        Ok(allocation) => {
            context.index_heap = Some(GeometryAllocation {
                allocation,
                ready: None,
                size: capacity,
                heap_id: None,
            });
            Ok(())
        }
        Err(error) => {
            let _ = context.geometry.remove_index_heap();
            Err(map_allocation(error))
        }
    }
}

fn grow_vertex_storage(context: &mut ContextState, name: &str, appended_bytes: u64) -> Result<()> {
    let heap = context
        .vertex_heaps
        .get(name)
        .ok_or(Error::InvalidContext)?;
    let heap_id = heap.heap_id.ok_or(Error::NativeFailure)?;
    if context.frame_vertex_heaps.contains_key(&heap_id) {
        return Err(Error::NotReady);
    }
    let capacity = grown_capacity(heap.size, appended_bytes)?;
    grow_storage(context, Some(name), capacity)
}

fn grow_index_storage(context: &mut ContextState, appended_bytes: u64) -> Result<()> {
    if context.frame_index.is_some() {
        return Err(Error::NotReady);
    }
    let heap = context.index_heap.as_ref().ok_or(Error::InvalidContext)?;
    let capacity = grown_capacity(heap.size, appended_bytes)?;
    grow_storage(context, None, capacity)
}

fn grow_storage(
    context: &mut ContextState,
    vertex_name: Option<&str>,
    capacity: u64,
) -> Result<()> {
    reclaim_retired_geometry(context)?;
    let current = match vertex_name {
        Some(name) => context.vertex_heaps.get(name),
        None => context.index_heap.as_ref(),
    }
    .ok_or(Error::InvalidContext)?;
    let request = AllocationRequest::new(capacity, 16, MemoryClass::Device, false, None)
        .map_err(map_allocation)?;
    let replacement = allocate_native(&mut context.native, request).map_err(map_allocation)?;
    let transfer = match copy_native(
        &mut context.native,
        &current.allocation,
        &replacement,
        0,
        0,
        current.size,
    ) {
        Ok(token) => token,
        Err(error) => {
            let _ = free_native_allocation(&mut context.native, replacement);
            return Err(map_allocation(error));
        }
    };
    let graphics = last_native_frame_completion(&context.native).ok();
    let old = if let Some(name) = vertex_name {
        context
            .geometry
            .grow_vertex_heap(name, capacity)
            .map_err(map_geometry)?;
        context.vertex_heaps.get_mut(name)
    } else {
        context
            .geometry
            .grow_index_heap(capacity)
            .map_err(map_geometry)?;
        context.index_heap.as_mut()
    }
    .ok_or(Error::InvalidContext)?;
    old.ready = Some(transfer);
    old.size = capacity;
    let allocation = core::mem::replace(&mut old.allocation, replacement);
    context.retired_geometry.push(RetiredGeometry {
        allocation,
        transfer,
        graphics,
    });
    Ok(())
}

fn reclaim_retired_geometry(context: &mut ContextState) -> Result<()> {
    if !context.retired_geometry.is_empty() || !context.retired_geometry_ranges.is_empty() {
        let transfer = completed_transfer_native(&mut context.native).map_err(map_allocation)?;
        let graphics = completed_native_frame_value(&mut context.native)?;
        let mut index = 0;
        while index < context.retired_geometry.len() {
            let retired = &context.retired_geometry[index];
            let graphics_ready = retired
                .graphics
                .is_none_or(|completion| completion.value <= graphics);
            if retired.transfer.value > transfer || !graphics_ready {
                index += 1;
                continue;
            }
            let retired = context.retired_geometry.swap_remove(index);
            free_native_allocation(&mut context.native, retired.allocation)
                .map_err(map_allocation)?;
        }

        let mut index = 0;
        while index < context.retired_geometry_ranges.len() {
            let retired = &context.retired_geometry_ranges[index];
            let graphics_ready = match retired.graphics {
                RetiredRangeGraphics::Prior(completion) => {
                    completion.is_none_or(|completion| completion.value <= graphics)
                }
                RetiredRangeGraphics::Recording(_) => false,
            };
            if retired.transfer.value > transfer || !graphics_ready {
                index += 1;
                continue;
            }
            let retired = context.retired_geometry_ranges.swap_remove(index);
            match retired.kind {
                RetiredRangeKind::Vertex => context.geometry.free_vertex(retired.handle),
                RetiredRangeKind::Index => context.geometry.free_indices(retired.handle),
            }
            .map_err(map_geometry)?;
        }
    }

    let mut index = 0;
    while index < context.retired_vertex_heaps.len() {
        let name = &context.retired_vertex_heaps[index].name;
        match context.geometry.remove_vertex_heap(name) {
            Ok(()) => {
                let retired = context.retired_vertex_heaps.swap_remove(index);
                free_native_allocation(&mut context.native, retired.allocation)
                    .map_err(map_allocation)?;
            }
            Err(GeometryError::HeapNotEmpty) => index += 1,
            Err(error) => return Err(map_geometry(error)),
        }
    }
    Ok(())
}

/// Uploads a typed POD slice to a vertex heap.
///
/// Count, stride, and byte size are derived from `vertices`.
///
/// # Errors
///
/// Returns an error for a stale or foreign heap, an empty or zero-sized slice,
/// stride/count overflow, exhausted heap capacity or handles, or native upload failure.
pub fn upload_vertices<T: bytemuck::Pod>(
    context: ContextHandle,
    heap: VertexHeapHandle,
    vertices: &[T],
) -> Result<VertexAllocationHandle> {
    let count = u32::try_from(vertices.len()).map_err(|_| Error::InvalidArgument)?;
    let element_size =
        u64::try_from(core::mem::size_of::<T>()).map_err(|_| Error::InvalidArgument)?;
    if count == 0 || element_size == 0 {
        // Empty and zero-sized slices cannot describe a heap allocation.
        return Err(Error::InvalidArgument);
    }

    upload_vertices_raw_impl(
        context,
        heap,
        count,
        element_size,
        bytemuck::cast_slice(vertices),
    )
}

/// Uploads caller-validated raw vertex bytes for the C ABI.
///
/// # Errors
///
/// Returns an error for an invalid heap handle, sizes, capacity, handle exhaustion,
/// or native upload failure.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub fn upload_vertices_raw(
    context: ContextHandle,
    heap: VertexHeapHandle,
    count: u32,
    element_size: u64,
    bytes: &[u8],
) -> Result<VertexAllocationHandle> {
    upload_vertices_raw_impl(context, heap, count, element_size, bytes)
}

fn upload_vertices_raw_impl(
    context: ContextHandle,
    heap: VertexHeapHandle,
    count: u32,
    element_size: u64,
    bytes: &[u8],
) -> Result<VertexAllocationHandle> {
    with_context_mut(context, |context| {
        context
            .identity
            .resolve(heap.packed(), ResourceKind::VertexHeap)
            .map_err(map_lifecycle)?;
        let name = context
            .vertex_heap_handles
            .get(&heap)
            .cloned()
            .ok_or(Error::InvalidContext)?;
        let byte_size = u64::from(count)
            .checked_mul(element_size)
            .ok_or(Error::InvalidArgument)?;
        if bytes.len() as u64 != byte_size {
            return Err(Error::InvalidArgument);
        }
        let packed = context
            .identity
            .insert(ResourceKind::VertexAllocation)
            .map_err(map_lifecycle)?;
        let handle =
            VertexAllocationHandle::from_packed(packed).map_err(|_| Error::NativeFailure)?;
        let upload = match context
            .geometry
            .reserve_vertices(&name, count, element_size, packed)
        {
            Ok(upload) => upload,
            Err(GeometryError::CapacityExceeded) => {
                if let Err(error) = grow_vertex_storage(context, &name, byte_size) {
                    let _ = context
                        .identity
                        .remove(packed, ResourceKind::VertexAllocation);
                    return Err(error);
                }
                context
                    .geometry
                    .reserve_vertices(&name, count, element_size, packed)
                    .map_err(map_geometry)?
            }
            Err(error) => {
                let _ = context
                    .identity
                    .remove(packed, ResourceKind::VertexAllocation);
                return Err(map_geometry(error));
            }
        };
        let heap = context
            .vertex_heaps
            .get_mut(&name)
            .ok_or(Error::InvalidArgument)?;
        match stage_upload(
            &mut context.native,
            &mut context.staging,
            &heap.allocation,
            upload.byte_offset,
            bytes,
        ) {
            Ok(token) => {
                heap.ready = Some(token);
                context
                    .geometry
                    .mark_ready(packed, token)
                    .map_err(map_geometry)?;
                context.geometry_uploads.insert(packed, token);
                context.geometry_last_transfer.insert(packed, token);
                context.upload_events.push(UploadEvent {
                    resource: UploadResource::Vertex(handle),
                    status: UploadStatus::SourceStaged,
                });
                Ok(handle)
            }
            Err(error) => {
                let _ = context.geometry.free_vertices(&name, packed);
                let _ = context
                    .identity
                    .remove(packed, ResourceKind::VertexAllocation);
                Err(map_allocation(error))
            }
        }
    })
}

/// Uploads packed `u32` indices to the global heap.
///
/// # Errors
///
/// Returns an error for an empty or oversized slice, exhausted heap capacity
/// or handles, or native upload failure.
pub fn upload_indices(context: ContextHandle, indices: &[u32]) -> Result<IndexAllocationHandle> {
    let count = u32::try_from(indices.len()).map_err(|_| Error::InvalidArgument)?;
    if count == 0 {
        // Empty slices cannot reserve a meaningful index allocation.
        return Err(Error::InvalidArgument);
    }

    upload_indices_raw_impl(context, count, bytemuck::cast_slice(indices))
}

/// Uploads caller-validated packed-u32 bytes for the C ABI.
///
/// # Errors
///
/// Returns an error for invalid sizes, capacity, handle exhaustion, or native upload failure.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub fn upload_indices_raw(
    context: ContextHandle,
    count: u32,
    bytes: &[u8],
) -> Result<IndexAllocationHandle> {
    upload_indices_raw_impl(context, count, bytes)
}

fn upload_indices_raw_impl(
    context: ContextHandle,
    count: u32,
    bytes: &[u8],
) -> Result<IndexAllocationHandle> {
    with_context_mut(context, |context| {
        let byte_size = u64::from(count)
            .checked_mul(4)
            .ok_or(Error::InvalidArgument)?;
        if bytes.len() as u64 != byte_size {
            return Err(Error::InvalidArgument);
        }
        ensure_index_heap(context, byte_size)?;
        let packed = context
            .identity
            .insert(ResourceKind::IndexAllocation)
            .map_err(map_lifecycle)?;
        let handle =
            IndexAllocationHandle::from_packed(packed).map_err(|_| Error::NativeFailure)?;
        let upload = match context.geometry.reserve_indices(count, packed) {
            Ok(upload) => upload,
            Err(GeometryError::CapacityExceeded) => {
                if let Err(error) = grow_index_storage(context, byte_size) {
                    let _ = context
                        .identity
                        .remove(packed, ResourceKind::IndexAllocation);
                    return Err(error);
                }
                context
                    .geometry
                    .reserve_indices(count, packed)
                    .map_err(map_geometry)?
            }
            Err(error) => {
                let _ = context
                    .identity
                    .remove(packed, ResourceKind::IndexAllocation);
                return Err(map_geometry(error));
            }
        };
        let heap = context.index_heap.as_mut().ok_or(Error::InvalidArgument)?;
        match stage_upload(
            &mut context.native,
            &mut context.staging,
            &heap.allocation,
            upload.byte_offset,
            bytes,
        ) {
            Ok(token) => {
                heap.ready = Some(token);
                context
                    .geometry
                    .mark_ready(packed, token)
                    .map_err(map_geometry)?;
                context.geometry_uploads.insert(packed, token);
                context.geometry_last_transfer.insert(packed, token);
                context.upload_events.push(UploadEvent {
                    resource: UploadResource::Index(handle),
                    status: UploadStatus::SourceStaged,
                });
                Ok(handle)
            }
            Err(error) => {
                let _ = context.geometry.free_indices(packed);
                let _ = context
                    .identity
                    .remove(packed, ResourceKind::IndexAllocation);
                Err(map_allocation(error))
            }
        }
    })
}

/// Returns `(first_element, element_count)` for a live vertex allocation.
///
/// # Errors
///
/// Returns an error for an invalid, stale, foreign, or wrong-kind allocation handle.
pub fn vertex_allocation_range(
    context: ContextHandle,
    handle: VertexAllocationHandle,
) -> Result<(u32, u32)> {
    with_context_mut(context, |context| {
        context
            .identity
            .resolve(handle.packed(), ResourceKind::VertexAllocation)
            .map_err(map_lifecycle)?;
        let upload = context
            .geometry
            .allocation(handle.packed())
            .map_err(map_geometry)?;
        Ok((upload.first_element, upload.element_count))
    })
}

/// Returns `(first_index, index_count)` for a live index allocation.
///
/// # Errors
///
/// Returns an error for an invalid, stale, foreign, or wrong-kind allocation handle.
pub fn index_allocation_range(
    context: ContextHandle,
    handle: IndexAllocationHandle,
) -> Result<(u32, u32)> {
    with_context_mut(context, |context| {
        context
            .identity
            .resolve(handle.packed(), ResourceKind::IndexAllocation)
            .map_err(map_lifecycle)?;
        let upload = context
            .geometry
            .allocation(handle.packed())
            .map_err(map_geometry)?;
        Ok((upload.first_element, upload.element_count))
    })
}

fn retired_range_graphics(context: &ContextState) -> RetiredRangeGraphics {
    // A drop during recording may precede the heap's lazy import, so every such
    // range waits for that transaction rather than guessing whether shaders use it.
    if context.frame.state() == ez_gfx_runtime::frame::FrameState::Recording {
        RetiredRangeGraphics::Recording(context.frame_serial)
    } else {
        RetiredRangeGraphics::Prior(last_native_frame_completion(&context.native).ok())
    }
}

pub(super) fn finalize_recording_range_drops(
    context: &mut ContextState,
    serial: u64,
    completion: Option<CompletionToken>,
) -> Result<()> {
    for retired in &mut context.retired_geometry_ranges {
        if matches!(retired.graphics, RetiredRangeGraphics::Recording(candidate) if candidate == serial)
        {
            retired.graphics = RetiredRangeGraphics::Prior(completion);
        }
    }
    reclaim_retired_geometry(context)
}

/// Retires a vertex range without blocking resource `Drop`.
///
/// # Errors
/// Returns an error for a stale identity or unavailable completion state.
pub fn remove_vertices(context: ContextHandle, handle: VertexAllocationHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        reclaim_retired_geometry(context)?;
        context
            .identity
            .resolve(handle.packed(), ResourceKind::VertexAllocation)
            .map_err(map_lifecycle)?;
        // Completed uploads leave the pending-event map but retain this fence until allocation drop.
        let transfer = context
            .geometry_last_transfer
            .remove(&handle.packed())
            .ok_or(Error::InvalidContext)?;
        let graphics = retired_range_graphics(context);
        context
            .identity
            .remove(handle.packed(), ResourceKind::VertexAllocation)
            .map_err(map_lifecycle)?;
        context.retired_geometry_ranges.push(RetiredGeometryRange {
            handle: handle.packed(),
            kind: RetiredRangeKind::Vertex,
            transfer,
            graphics,
        });
        reclaim_retired_geometry(context)
    }))
}

/// Retires an index range without blocking resource `Drop`.
///
/// # Errors
/// Returns an error for a stale identity or unavailable completion state.
pub fn remove_indices(context: ContextHandle, handle: IndexAllocationHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        reclaim_retired_geometry(context)?;
        context
            .identity
            .resolve(handle.packed(), ResourceKind::IndexAllocation)
            .map_err(map_lifecycle)?;
        // Completed uploads leave the pending-event map but retain this fence until allocation drop.
        let transfer = context
            .geometry_last_transfer
            .remove(&handle.packed())
            .ok_or(Error::InvalidContext)?;
        let graphics = retired_range_graphics(context);
        context
            .identity
            .remove(handle.packed(), ResourceKind::IndexAllocation)
            .map_err(map_lifecycle)?;
        context.retired_geometry_ranges.push(RetiredGeometryRange {
            handle: handle.packed(),
            kind: RetiredRangeKind::Index,
            transfer,
            graphics,
        });
        reclaim_retired_geometry(context)
    }))
}

/// Public uploads copy caller slices here, so no mapped lease exists today. A P-011
/// zero-copy staging lease hooking in at this seam MUST bind the `ContextIdentity`
/// epoch and fail `DeviceLost`/`InvalidContext` on use-after-loss, never fallback bytes.
pub(super) fn stage_upload(
    context: &mut NativeContext,
    pool: &mut ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    destination: &NativeAllocation,
    destination_offset: u64,
    bytes: &[u8],
) -> std::result::Result<CompletionToken, ez_gfx_hal::AllocationError> {
    let completed = completed_transfer_native(context)?;
    for stale in pool.trim(completed) {
        free_native_allocation(context, stale)?;
    }
    let requested = bytes.len() as u64;
    let (capacity, mut allocation) = if let Some(entry) = pool.take(requested, completed) {
        entry
    } else {
        let bucket = staging_bucket_size(requested, DEFAULT_STAGING_POLICY)
            .map_err(|_| ez_gfx_hal::AllocationError::OutOfMemory)?;
        let request = AllocationRequest::new(bucket, 16, MemoryClass::Upload, true, None)?;
        (bucket, allocate_native(context, request)?)
    };
    if let Err(error) = write_native(context, &mut allocation, bytes) {
        pool.put(capacity, allocation, None);
        return Err(error);
    }
    let token = match copy_native(
        context,
        &allocation,
        destination,
        0,
        destination_offset,
        requested,
    ) {
        Ok(token) => token,
        Err(error) => {
            pool.put(capacity, allocation, None);
            return Err(error);
        }
    };
    pool.put(capacity, allocation, Some(token));
    Ok(token)
}
