use crate::Result;

use super::{
    AllocationRequest, BufferHandle, COUNTER_BUFFER_ELEMENT_OFFSET, ContextHandle, ContextState,
    CounterBufferHandle, DrawIndexedCommand, Error, IndexedIndirectBuffer, MemoryClass,
    NativeAllocation, PackedHandle, ResourceKind, TransientBuffer, TransientUse, allocate_native,
    completed_native_frame_value, free_native_allocation, map_allocation, map_lifecycle,
    result_status, retire_native_allocation, stage_upload, with_context_mut,
};

/// Acquires a runtime-typed structured buffer for the C ABI.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub fn acquire_buffer_raw(
    context: ContextHandle,
    element_size: u32,
    element_count: u32,
) -> Result<BufferHandle> {
    acquire_buffer_raw_impl(context, element_size, element_count)
}

pub(crate) fn acquire_buffer_sized(
    context: ContextHandle,
    element_size: u32,
    element_count: u32,
) -> Result<BufferHandle> {
    acquire_buffer_raw_impl(context, element_size, element_count)
}

fn acquire_buffer_raw_impl(
    context: ContextHandle,
    element_size: u32,
    element_count: u32,
) -> Result<BufferHandle> {
    let (_, size) = checked_element_range(element_size, element_count, 0, element_count)?;
    with_context_mut(context, |context| {
        require_recording(context)?;
        let completed = completed_native_frame_value(&mut context.native)?;
        let (reused, stale) = {
            let pool = context
                .buffer_pool
                .entry(element_size)
                .or_insert_with(|| ez_gfx_hal::ReusableStagingPool::new(256));
            let reused = pool.take(size, completed);
            let stale = pool.trim(completed);
            (reused, stale)
        };
        for allocation in stale {
            free_native_allocation(&mut context.native, allocation).map_err(map_allocation)?;
        }
        let (byte_capacity, allocation) = if let Some(reused) = reused {
            reused
        } else {
            let request = AllocationRequest::new(size, 16, MemoryClass::Device, false, None)
                .map_err(|_| Error::InvalidArgument)?;
            (
                size,
                allocate_native(&mut context.native, request).map_err(map_allocation)?,
            )
        };
        insert_transient(
            context,
            ResourceKind::Buffer,
            size,
            allocation,
            TransientBuffer {
                element_size,
                element_count,
                byte_capacity,
                usage: TransientUse::Available,
            },
        )
        .and_then(|packed| BufferHandle::from_packed(packed).map_err(|_| Error::NativeFailure))
    })
}

/// Allocates a per-frame counter and indexed-command buffer.
///
/// The count occupies the first four bytes; commands begin at the shared aligned element offset.
///
/// # Errors
///
/// Returns an error outside frame recording, for invalid capacity, exhausted
/// handles, overflow, or native allocation failure.
pub fn acquire_counter(context: ContextHandle, capacity: u32) -> Result<CounterBufferHandle> {
    with_context_mut(context, |context| {
        require_recording(context)?;
        let buffer = IndexedIndirectBuffer::new(capacity).map_err(|_| Error::InvalidArgument)?;
        let (_, command_size) = checked_element_range(20, capacity, 0, capacity)?;
        let size = command_size
            .checked_add(COUNTER_BUFFER_ELEMENT_OFFSET)
            .ok_or(Error::InvalidArgument)?;
        let completed = completed_native_frame_value(&mut context.native)?;
        let reused = context.counter_pool.take(size, completed);
        for allocation in context.counter_pool.trim(completed) {
            free_native_allocation(&mut context.native, allocation).map_err(map_allocation)?;
        }
        let (byte_capacity, allocation) = if let Some(reused) = reused {
            reused
        } else {
            let request = AllocationRequest::new(size, 4, MemoryClass::Device, false, None)
                .map_err(|_| Error::InvalidArgument)?;
            (
                size,
                allocate_native(&mut context.native, request).map_err(map_allocation)?,
            )
        };
        let packed = insert_transient(
            context,
            ResourceKind::CounterBuffer,
            size,
            allocation,
            TransientBuffer {
                element_size: 20,
                element_count: capacity,
                byte_capacity,
                usage: TransientUse::Available,
            },
        )?;
        let typed = CounterBufferHandle::from_packed(packed).map_err(|_| Error::NativeFailure)?;
        context.indirects.insert(typed, buffer);
        Ok(typed)
    })
}

/// Writes and publishes a contiguous indexed-draw command range.
///
/// Active count becomes `max(previous_count, start_index + commands.len())`.
///
/// # Errors
///
/// Returns an error for a stale, consumed, foreign, or out-of-range handle,
/// checked arithmetic failure, or failed upload.
#[cfg_attr(
    not(any(test, feature = "ffi")),
    allow(
        dead_code,
        reason = "only raw FFI and state tests write typed commands"
    )
)]
pub fn write_counter_commands(
    context: ContextHandle,
    indirect: CounterBufferHandle,
    start_index: u32,
    commands: &[DrawIndexedCommand],
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = indirect.packed();
        validate_writable_transient(context, handle, ResourceKind::CounterBuffer)?;
        let count = u32::try_from(commands.len()).map_err(|_| Error::InvalidArgument)?;
        context
            .indirects
            .get_mut(&indirect)
            .ok_or(Error::InvalidContext)?
            .write_batch(start_index, commands)
            .map_err(|_| Error::InvalidArgument)?;
        let visible_count = context
            .indirects
            .get(&indirect)
            .ok_or(Error::InvalidContext)?
            .draw_count();
        if commands.is_empty() {
            return Ok(());
        }
        let offset = u64::from(start_index)
            .checked_mul(20)
            .and_then(|offset| offset.checked_add(COUNTER_BUFFER_ELEMENT_OFFSET))
            .ok_or(Error::InvalidArgument)?;
        let byte_size = u64::from(count)
            .checked_mul(20)
            .ok_or(Error::InvalidArgument)?;
        if byte_size == 0 {
            return Ok(());
        }
        let byte_size = usize::try_from(byte_size).map_err(|_| Error::InvalidArgument)?;
        let mut bytes = Vec::with_capacity(byte_size);
        for command in commands {
            bytes.extend_from_slice(&command.index_count.to_le_bytes());
            bytes.extend_from_slice(&command.instance_count.to_le_bytes());
            bytes.extend_from_slice(&command.first_index.to_le_bytes());
            bytes.extend_from_slice(&command.vertex_offset.to_le_bytes());
            bytes.extend_from_slice(&command.first_instance.to_le_bytes());
        }
        let ContextState {
            native,
            staging,
            allocations,
            allocation_ready,
            ..
        } = context;
        let (_, allocation) = allocations.get(&handle).ok_or(Error::InvalidContext)?;
        stage_upload(native, staging, allocation, 0, &visible_count.to_le_bytes())
            .map_err(map_allocation)?;
        let token =
            stage_upload(native, staging, allocation, offset, &bytes).map_err(map_allocation)?;
        allocation_ready.insert(handle, token);
        Ok(())
    }))
}
fn counter_payload(bytes: &[u8], initial_count: u32) -> Result<Vec<u8>> {
    // Padding is explicitly zeroed so no stale pooled bytes exist between the count and elements.
    let element_offset =
        usize::try_from(COUNTER_BUFFER_ELEMENT_OFFSET).map_err(|_| Error::InvalidArgument)?;
    let payload_size = bytes
        .len()
        .checked_add(element_offset)
        .ok_or(Error::InvalidArgument)?;
    let mut payload = Vec::with_capacity(payload_size);
    payload.extend_from_slice(&initial_count.to_le_bytes());
    payload.resize(element_offset, 0);
    payload.extend_from_slice(bytes);
    Ok(payload)
}

/// Stages a complete counter-buffer payload without an intermediate command copy.
///
/// # Errors
///
/// Returns an error for stale, consumed, foreign, incorrectly sized data, or an
/// initial count exceeding capacity.
#[doc(hidden)]
pub fn write_counter_bytes(
    context: ContextHandle,
    counter: CounterBufferHandle,
    bytes: &[u8],
    initial_count: u32,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = counter.packed();
        validate_writable_transient(context, handle, ResourceKind::CounterBuffer)?;
        let metadata = context
            .transient_buffers
            .get(&handle)
            .copied()
            .ok_or(Error::InvalidContext)?;
        let expected = usize::try_from(metadata.element_count)
            .ok()
            .and_then(|count| count.checked_mul(metadata.element_size as usize))
            .ok_or(Error::InvalidArgument)?;
        // Partial command initialization would leave unspecified command bytes visible.
        if metadata.element_size != 20
            || bytes.len() != expected
            || initial_count > metadata.element_count
        {
            return Err(Error::InvalidArgument);
        }
        let payload = counter_payload(bytes, initial_count)?;
        let ContextState {
            native,
            staging,
            allocations,
            allocation_ready,
            ..
        } = context;
        let (_, allocation) = allocations.get(&handle).ok_or(Error::InvalidContext)?;
        let token =
            stage_upload(native, staging, allocation, 0, &payload).map_err(map_allocation)?;
        allocation_ready.insert(handle, token);
        Ok(())
    }))
}

/// Writes runtime-typed buffer elements for the C ABI.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub fn write_buffer_raw(
    context: ContextHandle,
    buffer: BufferHandle,
    start_index: u32,
    element_count: u32,
    element_size: u32,
    bytes: &[u8],
) -> Result<()> {
    let value_count = usize::try_from(element_count).map_err(|_| Error::InvalidArgument)?;
    write_buffer_raw_impl(
        context,
        buffer,
        start_index,
        element_size,
        bytes,
        value_count,
    )
}

pub(crate) fn write_buffer_bytes(
    context: ContextHandle,
    buffer: BufferHandle,
    element_size: u32,
    bytes: &[u8],
) -> Result<()> {
    let value_count = bytes
        .len()
        .checked_div(element_size as usize)
        .ok_or(Error::InvalidArgument)?;
    write_buffer_raw_impl(context, buffer, 0, element_size, bytes, value_count)
}

fn write_buffer_raw_impl(
    context: ContextHandle,
    buffer: BufferHandle,
    start_index: u32,
    element_size: u32,
    bytes: &[u8],
    value_count: usize,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = buffer.packed();
        validate_writable_transient(context, handle, ResourceKind::Buffer)?;
        let metadata = *context
            .transient_buffers
            .get(&handle)
            .ok_or(Error::InvalidContext)?;
        if element_size == 0 || metadata.element_size != element_size {
            return Err(Error::InvalidArgument);
        }
        let count = u32::try_from(value_count).map_err(|_| Error::InvalidArgument)?;
        let (offset, byte_size) = checked_element_range(
            metadata.element_size,
            metadata.element_count,
            start_index,
            count,
        )?;
        if usize::try_from(byte_size).ok() != Some(bytes.len()) {
            return Err(Error::InvalidArgument);
        }
        if byte_size == 0 {
            return Ok(());
        }
        let ContextState {
            native,
            staging,
            allocations,
            allocation_ready,
            ..
        } = context;
        let (_, allocation) = allocations.get(&handle).ok_or(Error::InvalidContext)?;
        let token =
            stage_upload(native, staging, allocation, offset, bytes).map_err(map_allocation)?;
        allocation_ready.insert(handle, token);
        Ok(())
    }))
}

/// Releases a counter buffer that was not consumed by a recorded frame.
pub fn release_counter(context: ContextHandle, counter: CounterBufferHandle) {
    let _ = release_transient(
        context,
        counter.packed(),
        ResourceKind::CounterBuffer,
        Some(counter),
    );
}

/// Releases a buffer that was not consumed by a recorded frame.
pub fn release_buffer(context: ContextHandle, buffer: BufferHandle) {
    let _ = release_transient(context, buffer.packed(), ResourceKind::Buffer, None);
}

fn insert_transient(
    context: &mut ContextState,
    kind: ResourceKind,
    size: u64,
    allocation: NativeAllocation,
    metadata: TransientBuffer,
) -> Result<PackedHandle> {
    let handle = match context.identity.insert(kind) {
        Ok(handle) => handle,
        Err(error) => {
            let _ = free_native_allocation(&mut context.native, allocation);
            return Err(map_lifecycle(error));
        }
    };
    context.allocations.insert(handle, (size, allocation));
    context.transient_buffers.insert(handle, metadata);
    Ok(handle)
}

fn release_transient(
    context_handle: ContextHandle,
    handle: PackedHandle,
    kind: ResourceKind,
    indirect: Option<CounterBufferHandle>,
) -> Result<()> {
    with_context_mut(context_handle, |context| {
        context
            .identity
            .resolve(handle, kind)
            .map_err(map_lifecycle)?;
        let metadata = context
            .transient_buffers
            .get(&handle)
            .copied()
            .ok_or(Error::InvalidContext)?;
        if metadata.usage != TransientUse::Available {
            return Err(Error::NotReady);
        }
        context
            .identity
            .remove(handle, kind)
            .map_err(map_lifecycle)?;
        context.transient_buffers.remove(&handle);
        if let Some(indirect) = indirect {
            context.indirects.remove(&indirect);
        }
        let ready = context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        match ready {
            Some(completion) => {
                retire_native_allocation(&mut context.native, allocation, completion)
            }
            None => free_native_allocation(&mut context.native, allocation),
        }
        .map_err(map_allocation)
    })
}

// Unrecorded writes may still be pending on transfer; retire those allocations
// against their upload token instead of admitting them to a graphics-completion pool.
pub(super) fn reclaim_available_transients(context: &mut ContextState) -> Result<()> {
    let handles = context
        .transient_buffers
        .iter()
        .filter_map(|(handle, metadata)| {
            (metadata.usage == TransientUse::Available).then_some(*handle)
        })
        .collect::<Vec<_>>();

    for handle in handles {
        let kind = context
            .identity
            .resource_kind(handle)
            .map_err(map_lifecycle)?;
        context
            .identity
            .remove(handle, kind)
            .map_err(map_lifecycle)?;
        context.transient_buffers.remove(&handle);
        if kind == ResourceKind::CounterBuffer
            && let Ok(indirect) = CounterBufferHandle::from_packed(handle)
        {
            context.indirects.remove(&indirect);
        }
        let ready = context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        match ready {
            Some(completion) => {
                retire_native_allocation(&mut context.native, allocation, completion)
            }
            None => free_native_allocation(&mut context.native, allocation),
        }
        .map_err(map_allocation)?;
    }

    Ok(())
}

fn require_recording(context: &ContextState) -> Result<()> {
    context
        .identity
        .check_thread_and_health()
        .map_err(map_lifecycle)?;
    if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording {
        return Err(Error::NotReady);
    }
    Ok(())
}

fn validate_writable_transient(
    context: &ContextState,
    handle: PackedHandle,
    kind: ResourceKind,
) -> Result<()> {
    context
        .identity
        .resolve(handle, kind)
        .map_err(map_lifecycle)?;
    let metadata = context
        .transient_buffers
        .get(&handle)
        .ok_or(Error::InvalidContext)?;
    if metadata.usage != TransientUse::Available {
        return Err(Error::NotReady);
    }
    Ok(())
}

fn checked_element_range(
    element_size: u32,
    capacity: u32,
    start_index: u32,
    count: u32,
) -> Result<(u64, u64)> {
    if element_size == 0 {
        return Err(Error::InvalidArgument);
    }
    let end = start_index
        .checked_add(count)
        .ok_or(Error::InvalidArgument)?;
    if end > capacity {
        return Err(Error::InvalidArgument);
    }
    let offset = u64::from(start_index)
        .checked_mul(u64::from(element_size))
        .ok_or(Error::InvalidArgument)?;
    let size = u64::from(count)
        .checked_mul(u64::from(element_size))
        .ok_or(Error::InvalidArgument)?;
    Ok((offset, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_element_ranges_accept_boundaries_and_reject_overflow() {
        assert_eq!(checked_element_range(4, 8, 2, 3), Ok((8, 12)));
        assert_eq!(checked_element_range(4, 8, 8, 0), Ok((32, 0)));
        assert_eq!(
            checked_element_range(4, 8, 7, 2),
            Err(Error::InvalidArgument)
        );
        assert_eq!(
            checked_element_range(u32::MAX, u32::MAX, u32::MAX, 2),
            Err(Error::InvalidArgument)
        );
    }

    #[test]
    fn counter_payload_places_elements_at_shared_aligned_offset() {
        let command = [0x5a; 20];
        let payload = counter_payload(&command, 7).unwrap();
        let offset = usize::try_from(COUNTER_BUFFER_ELEMENT_OFFSET).unwrap();

        assert_eq!(&payload[..4], &7_u32.to_le_bytes());
        assert!(payload[4..offset].iter().all(|byte| *byte == 0));
        assert_eq!(&payload[offset..], &command);
        assert_eq!(payload.len(), offset + command.len());
    }
}
