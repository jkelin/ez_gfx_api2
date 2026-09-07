use super::{
    AllocationRequest, ContextHandle, ContextState, DrawIndexedCommand, EzGfxResult,
    IndexedIndirectBuffer, IndirectBufferHandle, MemoryClass, ResourceKind, StructuredBufferHandle,
    allocate_native, free_native_allocation, map_allocation, map_lifecycle, result_status,
    retire_native_allocation, stage_upload, with_context_mut,
};

/// Allocates a structured upload buffer.
///
/// # Errors
///
/// Returns an error for an invalid context, zero or excessive size, or native allocation failure.
pub fn acquire_structured(
    context: ContextHandle,
    size: u64,
) -> Result<StructuredBufferHandle, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let request = AllocationRequest::new(size, 16, MemoryClass::Device, false, None)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let allocation = allocate_native(&mut context.native, request).map_err(map_allocation)?;
        let handle = match context.identity.insert(ResourceKind::Structured) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = free_native_allocation(&mut context.native, allocation);
                return Err(map_lifecycle(error));
            }
        };
        context.allocations.insert(handle, (size, allocation));
        StructuredBufferHandle::from_packed(handle).map_err(|_| EzGfxResult::NativeFailure)
    })
}

/// Allocates an indirect draw buffer.
///
/// # Errors
///
/// Returns an error for an invalid context or capacity, exhausted handles, or native allocation failure.
pub fn acquire_indirect(
    context: ContextHandle,
    capacity: u32,
) -> Result<IndirectBufferHandle, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let buffer =
            IndexedIndirectBuffer::new(capacity).map_err(|_| EzGfxResult::InvalidArgument)?;
        let size = u64::from(capacity)
            .checked_mul(20)
            .ok_or(EzGfxResult::InvalidArgument)?;
        let request = AllocationRequest::new(size, 4, MemoryClass::Device, false, None)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let allocation = allocate_native(&mut context.native, request).map_err(map_allocation)?;
        let handle = match context.identity.insert(ResourceKind::Indirect) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = free_native_allocation(&mut context.native, allocation);
                return Err(map_lifecycle(error));
            }
        };
        context.allocations.insert(handle, (size, allocation));
        let typed =
            IndirectBufferHandle::from_packed(handle).map_err(|_| EzGfxResult::NativeFailure)?;
        context.indirects.insert(typed, buffer);
        Ok(typed)
    })
}

/// Writes one indexed draw command.
pub fn write_indirect(
    context: ContextHandle,
    indirect: IndirectBufferHandle,
    index: u32,
    command: DrawIndexedCommand,
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = indirect.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context
            .indirects
            .get_mut(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?
            .write(index, command)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&command.index_count.to_le_bytes());
        bytes.extend_from_slice(&command.instance_count.to_le_bytes());
        bytes.extend_from_slice(&command.first_index.to_le_bytes());
        bytes.extend_from_slice(&command.vertex_offset.to_le_bytes());
        bytes.extend_from_slice(&command.first_instance.to_le_bytes());
        let ContextState {
            native,
            staging,
            allocations,
            allocation_ready,
            ..
        } = context;
        let (_, allocation) = allocations
            .get(&handle)
            .ok_or(EzGfxResult::InvalidContext)?;
        let token = stage_upload(native, staging, allocation, u64::from(index) * 20, &bytes)
            .map_err(map_allocation)?;
        allocation_ready.insert(handle, token);
        Ok(())
    }))
}

/// Publishes the active indirect draw count.
pub fn set_indirect_count(
    context: ContextHandle,
    indirect: IndirectBufferHandle,
    count: u32,
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = indirect.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context
            .indirects
            .get_mut(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?
            .set_draw_count(count)
            .map_err(|_| EzGfxResult::InvalidArgument)
    }))
}

/// Releases an indirect draw buffer.
pub fn release_indirect(context: ContextHandle, indirect: IndirectBufferHandle) {
    let _ = with_context_mut(context, |context| {
        let handle = indirect.packed();
        context
            .identity
            .remove(handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context
            .indirects
            .remove(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?;
        let ready = context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(EzGfxResult::InvalidContext)?;
        match ready {
            Some(completion) => {
                retire_native_allocation(&mut context.native, allocation, completion)
            }
            None => free_native_allocation(&mut context.native, allocation),
        }
        .map_err(map_allocation)
    });
}
/// Uploads bytes to a structured buffer.
pub fn write_structured(
    context: ContextHandle,
    structured: StructuredBufferHandle,
    bytes: &[u8],
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = structured.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Structured)
            .map_err(map_lifecycle)?;
        let ContextState {
            native,
            staging,
            allocations,
            allocation_ready,
            ..
        } = context;
        let (capacity, allocation) = allocations
            .get(&handle)
            .ok_or(EzGfxResult::InvalidContext)?;
        if bytes.len() as u64 > *capacity {
            return Err(EzGfxResult::InvalidArgument);
        }
        let token = stage_upload(native, staging, allocation, 0, bytes).map_err(map_allocation)?;
        allocation_ready.insert(handle, token);
        Ok(())
    }))
}

/// Releases a structured buffer.
pub fn release_structured(context: ContextHandle, structured: StructuredBufferHandle) {
    let _ = with_context_mut(context, |context| {
        let handle = structured.packed();
        context
            .identity
            .remove(handle, ResourceKind::Structured)
            .map_err(map_lifecycle)?;
        let ready = context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(EzGfxResult::InvalidContext)?;
        match ready {
            Some(completion) => {
                retire_native_allocation(&mut context.native, allocation, completion)
            }
            None => free_native_allocation(&mut context.native, allocation),
        }
        .map_err(map_allocation)
    });
}
