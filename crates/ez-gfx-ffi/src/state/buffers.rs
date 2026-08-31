use super::{
    AllocationRequest, DrawIndexedCommand, EzGfxResult, FfiContext, IndexedIndirectBuffer,
    MemoryClass, PackedHandle, ResourceKind, allocate_native, free_native_allocation,
    map_allocation, map_lifecycle, result_status, stage_upload, with_context_mut,
};

pub fn acquire_structured(context: u64, size: u64) -> Result<PackedHandle, EzGfxResult> {
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
        context.allocations.insert(handle.get(), (size, allocation));
        Ok(handle)
    })
}

pub fn acquire_indirect(context: u64, capacity: u32) -> Result<PackedHandle, EzGfxResult> {
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
        context.allocations.insert(handle.get(), (size, allocation));
        context.indirects.insert(handle.get(), buffer);
        Ok(handle)
    })
}

pub fn write_indirect(
    context: u64,
    indirect: u64,
    index: u32,
    command: DrawIndexedCommand,
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(indirect).map_err(|_| EzGfxResult::InvalidContext)?;
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
        let (_, allocation) = context
            .allocations
            .get(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?;
        stage_upload(
            &mut context.native,
            &mut context.staging,
            allocation,
            u64::from(index) * 20,
            &bytes,
        )
        .map(|_| ())
        .map_err(map_allocation)
    }))
}

pub fn set_indirect_count(context: u64, indirect: u64, count: u32) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(indirect).map_err(|_| EzGfxResult::InvalidContext)?;
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

pub fn release_indirect(context: u64, indirect: u64) {
    if indirect == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(indirect).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        context
            .indirects
            .remove(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?;
        let (_, allocation) = context
            .allocations
            .remove(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?;
        free_native_allocation(&mut context.native, allocation).map_err(map_allocation)
    });
}
pub fn write_structured(context: u64, structured: u64, bytes: &[u8]) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(structured).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Structured)
            .map_err(map_lifecycle)?;
        let FfiContext {
            native,
            staging,
            allocations,
            ..
        } = context;
        let (capacity, allocation) = allocations
            .get(&structured)
            .ok_or(EzGfxResult::InvalidContext)?;
        if bytes.len() as u64 > *capacity {
            return Err(EzGfxResult::InvalidArgument);
        }
        stage_upload(native, staging, allocation, 0, bytes)
            .map(|_| ())
            .map_err(map_allocation)
    }))
}

pub fn release_structured(context: u64, structured: u64) {
    if structured == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        let handle = PackedHandle::from_raw(structured).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Structured)
            .map_err(map_lifecycle)?;
        let (_, allocation) = context
            .allocations
            .remove(&structured)
            .ok_or(EzGfxResult::InvalidContext)?;
        free_native_allocation(&mut context.native, allocation).map_err(map_allocation)
    });
}
