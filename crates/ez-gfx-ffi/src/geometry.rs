use ez_gfx::raw::{
    self, ContextHandle, IndexAllocationHandle, VertexAllocationHandle, VertexHeapHandle,
};

use super::{
    EZ_GFX_MAX_BOUNDARY_BYTES, EzGfxBuffer, EzGfxContext, EzGfxIndexAllocation, EzGfxResult,
    EzGfxVertexAllocation, EzGfxVertexHeap, IntoFfiResult, buffer, catch_status, catch_void,
    read_bounded_string, validate_bounded_string,
};

#[unsafe(no_mangle)]
/// Creates a named vertex heap and returns its typed handle.
///
/// # Safety
///
/// `name` must cover `name_length` readable bytes and `out_heap` one writable handle.
pub unsafe extern "C" fn ez_gfx_vertex_heap_create(
    context: EzGfxContext,
    name: *const u8,
    name_length: usize,
    stride: u64,
    out_heap: *mut EzGfxVertexHeap,
) -> EzGfxResult {
    catch_status(|| {
        let name = match read_bounded_string(name, name_length) {
            Ok(name) => name,
            Err(status) => return status,
        };
        if out_heap.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match raw::create_vertex_heap(context, &name, stride) {
            Ok(handle) => {
                // SAFETY: the validated caller-owned output remains writable for this call.
                unsafe { out_heap.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Destroys the vertex heap selected by its typed handle.
pub extern "C" fn ez_gfx_vertex_heap_destroy(context: EzGfxContext, heap: EzGfxVertexHeap) {
    catch_void(|| {
        if let (Ok(context), Ok(heap)) = (
            ContextHandle::from_raw(context),
            VertexHeapHandle::from_raw(heap),
        ) {
            raw::destroy_vertex_heap(context, heap);
        }
    });
}

#[unsafe(no_mangle)]
/// Uploads packed-u32 indices and returns a typed allocation handle.
///
/// # Safety
///
/// `data` must cover `count * 4` bytes and `out_allocation` one writable handle.
pub unsafe extern "C" fn ez_gfx_index_allocation_create(
    context: EzGfxContext,
    data: *const std::ffi::c_void,
    count: u32,
    out_allocation: *mut EzGfxIndexAllocation,
) -> EzGfxResult {
    catch_status(|| {
        let size = match usize::try_from(u64::from(count) * 4) {
            Ok(size) if count != 0 => size,
            _ => return EzGfxResult::InvalidArgument,
        };
        if data.is_null() || out_allocation.is_null() || size > EZ_GFX_MAX_BOUNDARY_BYTES {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: the checked nonzero caller range remains readable through this call.
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), size) };
        let context = try_handle!(ContextHandle, context);
        match raw::upload_indices_raw(context, count, bytes) {
            Ok(handle) => {
                // SAFETY: the validated out pointer remains writable through this call.
                unsafe { out_allocation.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Uploads elements to a typed vertex heap and returns an allocation handle.
///
/// # Safety
///
/// `data` and `out_allocation` must cover their documented ranges for this call.
pub unsafe extern "C" fn ez_gfx_vertex_heap_upload(
    context: EzGfxContext,
    heap: EzGfxVertexHeap,
    data: *const std::ffi::c_void,
    element_count: u32,
    element_size: u64,
    out_allocation: *mut EzGfxVertexAllocation,
) -> EzGfxResult {
    catch_status(|| {
        let size = match u64::from(element_count)
            .checked_mul(element_size)
            .and_then(|size| usize::try_from(size).ok())
        {
            Some(size)
                if element_count != 0 && element_size != 0 && size <= EZ_GFX_MAX_BOUNDARY_BYTES =>
            {
                size
            }
            _ => return EzGfxResult::InvalidArgument,
        };
        if data.is_null() || out_allocation.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: the checked nonzero caller range remains readable through this call.
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), size) };
        let context = try_handle!(ContextHandle, context);
        let heap = try_handle!(VertexHeapHandle, heap);
        match raw::upload_vertices_raw(context, heap, element_count, element_size, bytes) {
            Ok(handle) => {
                // SAFETY: the validated out pointer remains writable through this call.
                unsafe { out_allocation.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Queries a live vertex allocation range.
///
/// # Safety
///
/// Both output pointers must address writable aligned `u32` values for this call.
pub unsafe extern "C" fn ez_gfx_vertex_allocation_get_range(
    context: EzGfxContext,
    allocation: EzGfxVertexAllocation,
    out_first: *mut u32,
    out_count: *mut u32,
) -> EzGfxResult {
    catch_status(|| {
        if out_first.is_null() || out_count.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        let allocation = try_handle!(VertexAllocationHandle, allocation);
        match raw::vertex_allocation_range(context, allocation) {
            Ok((first, count)) => {
                // SAFETY: both output pointers were validated for this call.
                unsafe {
                    out_first.write(first);
                    out_count.write(count);
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Queries a live index allocation range.
///
/// # Safety
///
/// Both output pointers must address writable aligned `u32` values for this call.
pub unsafe extern "C" fn ez_gfx_index_allocation_get_range(
    context: EzGfxContext,
    allocation: EzGfxIndexAllocation,
    out_first: *mut u32,
    out_count: *mut u32,
) -> EzGfxResult {
    catch_status(|| {
        if out_first.is_null() || out_count.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        let allocation = try_handle!(IndexAllocationHandle, allocation);
        match raw::index_allocation_range(context, allocation) {
            Ok((first, count)) => {
                // SAFETY: both output pointers were validated for this call.
                unsafe {
                    out_first.write(first);
                    out_count.write(count);
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Removes a live vertex allocation after establishing GPU safety.
pub extern "C" fn ez_gfx_vertex_allocation_remove(
    context: EzGfxContext,
    allocation: EzGfxVertexAllocation,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        let allocation = try_handle!(VertexAllocationHandle, allocation);
        raw::remove_vertices(context, allocation).into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Removes a live index allocation after establishing GPU safety.
pub extern "C" fn ez_gfx_index_allocation_remove(
    context: EzGfxContext,
    allocation: EzGfxIndexAllocation,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        let allocation = try_handle!(IndexAllocationHandle, allocation);
        raw::remove_indices(context, allocation).into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Acquires a context-owned buffer sized for the requested elements.
///
/// # Safety
///
/// `debug_name` must cover its non-empty UTF-8 range and `out_buffer` one
/// writable handle.
pub unsafe extern "C" fn ez_gfx_buffer_acquire(
    context: EzGfxContext,
    element_size: u32,
    element_count: u32,
    debug_name: *const u8,
    debug_name_length: usize,
    out_buffer: *mut EzGfxBuffer,
) -> EzGfxResult {
    catch_status(|| {
        let byte_size = u64::from(element_size) * u64::from(element_count);
        if element_size == 0
            || element_count == 0
            || usize::try_from(byte_size)
                .ok()
                .is_none_or(|size| size > EZ_GFX_MAX_BOUNDARY_BYTES)
            || out_buffer.is_null()
            || !out_buffer.is_aligned()
            || validate_bounded_string(debug_name, debug_name_length).is_err()
        {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match buffer::insert(
            context,
            buffer::Kind::Structured,
            element_size,
            element_count,
        ) {
            Ok(handle) => {
                // SAFETY: the validated caller-owned output remains writable for this call.
                unsafe { out_buffer.write(handle) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Copies a typed element range into a buffer.
///
/// # Safety
///
/// `data` must cover `element_count * element_size` readable bytes, or may be
/// null when `element_count` is zero.
pub unsafe extern "C" fn ez_gfx_buffer_write(
    context: EzGfxContext,
    buffer: EzGfxBuffer,
    start_index: u32,
    data: *const std::ffi::c_void,
    element_count: u32,
    element_size: u32,
) -> EzGfxResult {
    catch_status(|| {
        let byte_size = match u64::from(element_count)
            .checked_mul(u64::from(element_size))
            .and_then(|size| usize::try_from(size).ok())
        {
            Some(size) if element_size != 0 && size <= EZ_GFX_MAX_BOUNDARY_BYTES => size,
            _ => return EzGfxResult::InvalidArgument,
        };
        if element_count != 0 && data.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let bytes = if byte_size == 0 {
            &[]
        } else {
            // SAFETY: the checked non-null caller range remains readable for this call.
            unsafe { core::slice::from_raw_parts(data.cast::<u8>(), byte_size) }
        };
        let context = try_handle!(ContextHandle, context);
        buffer::write(
            buffer,
            context,
            buffer::Kind::Structured,
            start_index,
            element_size,
            bytes,
        )
        .map_or_else(|status| status, |()| EzGfxResult::Ok)
    })
}

#[unsafe(no_mangle)]
/// Releases a context-owned buffer handle; zero and stale values are ignored.
pub extern "C" fn ez_gfx_buffer_release(context: EzGfxContext, buffer: EzGfxBuffer) {
    catch_void(|| {
        if let Ok(context) = ContextHandle::from_raw(context) {
            let _ = buffer::remove(buffer, context, buffer::Kind::Structured);
        }
    });
}
