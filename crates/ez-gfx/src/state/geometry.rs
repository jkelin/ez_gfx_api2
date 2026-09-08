//! Named vertex and global index heap lifecycle.

use crate::Result;

use super::{
    AllocationRequest, CompletionToken, ContextHandle, DEFAULT_STAGING_POLICY, Error,
    GeometryAllocation, IndexAllocationHandle, MemoryClass, NativeAllocation, NativeContext,
    ResourceKind, UploadEvent, UploadResource, UploadStatus, VertexAllocationHandle,
    VertexHeapHandle, allocate_native, completed_transfer_native, copy_native,
    free_native_allocation, map_allocation, map_geometry, map_hal, map_lifecycle, result_status,
    staging_bucket_size, wait_native_idle, with_context_mut, write_native,
};

/// Creates a named vertex heap and returns its typed identity.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn create_vertex_heap(
    context: ContextHandle,
    name: &str,
    capacity: u64,
    stride: u64,
) -> Result<VertexHeapHandle> {
    with_context_mut(context, |context| {
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

/// Destroys a vertex heap identified by its owner-validated handle.
pub fn destroy_vertex_heap(context: ContextHandle, handle: VertexHeapHandle) {
    let _ = with_context_mut(context, |context| {
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
        context
            .geometry
            .remove_vertex_heap(&name)
            .map_err(map_geometry)?;
        context.vertex_heap_handles.remove(&handle);
        let heap = context
            .vertex_heaps
            .remove(&name)
            .ok_or(Error::InvalidContext)?;
        context
            .identity
            .remove(handle.packed(), ResourceKind::VertexHeap)
            .map_err(map_lifecycle)?;
        free_native_allocation(&mut context.native, heap.allocation).map_err(map_allocation)
    });
}

/// Creates the context index heap.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
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
            Err(error) => {
                let _ = context
                    .identity
                    .remove(packed, ResourceKind::VertexAllocation);
                return Err(map_geometry(error));
            }
        };
        if bytes.len() as u64 != upload.byte_size {
            let _ = context.geometry.free_vertices(&name, packed);
            let _ = context
                .identity
                .remove(packed, ResourceKind::VertexAllocation);
            return Err(Error::InvalidArgument);
        }
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
        let packed = context
            .identity
            .insert(ResourceKind::IndexAllocation)
            .map_err(map_lifecycle)?;
        let handle =
            IndexAllocationHandle::from_packed(packed).map_err(|_| Error::NativeFailure)?;
        let upload = match context.geometry.reserve_indices(count, packed) {
            Ok(upload) => upload,
            Err(error) => {
                let _ = context
                    .identity
                    .remove(packed, ResourceKind::IndexAllocation);
                return Err(map_geometry(error));
            }
        };
        if bytes.len() as u64 != upload.byte_size {
            let _ = context.geometry.free_indices(packed);
            let _ = context
                .identity
                .remove(packed, ResourceKind::IndexAllocation);
            return Err(Error::InvalidArgument);
        }
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

/// Removes a vertex range after waiting for transfer and graphics safety.
///
/// # Errors
pub fn remove_vertices(context: ContextHandle, handle: VertexAllocationHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .resolve(handle.packed(), ResourceKind::VertexAllocation)
            .map_err(map_lifecycle)?;
        wait_native_idle(&mut context.native).map_err(map_hal)?;
        context
            .geometry
            .free_vertex(handle.packed())
            .map_err(map_geometry)?;
        context.geometry_uploads.remove(&handle.packed());
        context
            .identity
            .remove(handle.packed(), ResourceKind::VertexAllocation)
            .map_err(map_lifecycle)
    }))
}

/// Removes an index range after waiting for transfer and graphics safety.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn remove_indices(context: ContextHandle, handle: IndexAllocationHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .resolve(handle.packed(), ResourceKind::IndexAllocation)
            .map_err(map_lifecycle)?;
        wait_native_idle(&mut context.native).map_err(map_hal)?;
        context
            .geometry
            .free_indices(handle.packed())
            .map_err(map_geometry)?;
        context.geometry_uploads.remove(&handle.packed());
        context
            .identity
            .remove(handle.packed(), ResourceKind::IndexAllocation)
            .map_err(map_lifecycle)
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
