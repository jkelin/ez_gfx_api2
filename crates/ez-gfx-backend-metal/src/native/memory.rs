use super::{
    AllocationCreateDesc, AllocationError, AllocationRequest, BufferTransfer, CompletionToken,
    DeferredResource, HalError, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLHeap, MemoryAllocator,
    MemoryClass, MemoryLocation, NativeAllocation, NativeContext, QueueKind, RetiredAllocation,
    ThreadBound,
};

impl MemoryAllocator for NativeContext {
    type Allocation = NativeAllocation;

    fn allocate(
        &mut self,
        request: AllocationRequest,
    ) -> Result<Self::Allocation, AllocationError> {
        let location = match request.memory_class {
            MemoryClass::Device | MemoryClass::Transient => MemoryLocation::GpuOnly,
            MemoryClass::Upload => MemoryLocation::CpuToGpu,
            MemoryClass::Readback => MemoryLocation::GpuToCpu,
        };
        let desc =
            AllocationCreateDesc::buffer(&self.device, "ez-gfx-buffer", request.size, location);
        if desc.alignment < request.alignment {
            return Err(AllocationError::InvalidAlignment);
        }
        let allocation = self
            .allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .allocate(&desc)
            .map_err(map_allocator)?;
        // SAFETY: `allocation` was just returned by `self.allocator` and remains allocated, so `allocation.heap()` refers to its backing heap until the allocation is freed.
        let heap = unsafe { allocation.heap() };
        let allocation_size =
            usize::try_from(allocation.size()).map_err(|_| AllocationError::NativeFailure)?;
        let allocation_offset =
            usize::try_from(allocation.offset()).map_err(|_| AllocationError::NativeFailure)?;
        // SAFETY: the allocation belongs to this heap and the checked size and offset describe it.
        let buffer = unsafe {
            heap.newBufferWithLength_options_offset(
                allocation_size,
                heap.resourceOptions(),
                allocation_offset,
            )
        }
        .ok_or(AllocationError::OutOfMemory)?;
        let mapped_address = if request.mapped {
            let address = buffer.contents().as_ptr() as usize;
            if address == 0 {
                return Err(AllocationError::NotHostVisible);
            }
            address
        } else {
            0
        };
        Ok(NativeAllocation {
            buffer: ThreadBound::new(buffer),
            allocation: ThreadBound::new(allocation),
            mapped_address,
        })
    }

    fn mapped_slice<'a>(
        &self,
        allocation: &'a Self::Allocation,
    ) -> Result<&'a [u8], AllocationError> {
        if allocation.mapped_address == 0 || allocation.allocation.size() > isize::MAX as u64 {
            return Err(AllocationError::NotHostVisible);
        }
        // SAFETY: `mapped_address` is non-null, and `allocation.buffer` retains at least `allocation.size()` readable bytes at that address for `'a`; the size is checked to fit `isize`.
        Ok(unsafe {
            core::slice::from_raw_parts(
                allocation.mapped_address as *const u8,
                usize::try_from(allocation.allocation.size())
                    .map_err(|_| AllocationError::NativeFailure)?,
            )
        })
    }

    fn mapped_slice_mut<'a>(
        &mut self,
        allocation: &'a mut Self::Allocation,
    ) -> Result<&'a mut [u8], AllocationError> {
        if allocation.mapped_address == 0 || allocation.allocation.size() > isize::MAX as u64 {
            return Err(AllocationError::NotHostVisible);
        }
        // SAFETY: `mapped_address` is non-null, and the exclusive borrow of `allocation.buffer` retains at least `allocation.size()` uniquely accessible bytes there for `'a`; the size fits `isize`.
        Ok(unsafe {
            core::slice::from_raw_parts_mut(
                allocation.mapped_address as *mut u8,
                usize::try_from(allocation.allocation.size())
                    .map_err(|_| AllocationError::NativeFailure)?,
            )
        })
    }

    fn flush(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError> {
        validate_range(allocation.allocation.size(), offset, size)?;
        if allocation.mapped_address == 0 {
            return Err(AllocationError::NotHostVisible);
        }
        Ok(())
    }

    fn invalidate(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError> {
        validate_range(allocation.allocation.size(), offset, size)?;
        if allocation.mapped_address == 0 {
            return Err(AllocationError::NotHostVisible);
        }
        Ok(())
    }

    fn free(&mut self, allocation: Self::Allocation) -> Result<(), AllocationError> {
        self.defer_resource(DeferredResource::Allocation(allocation))
    }

    fn retire(
        &mut self,
        allocation: Self::Allocation,
        completion: CompletionToken,
    ) -> Result<(), AllocationError> {
        self.retired.push(RetiredAllocation {
            allocation,
            completion,
        });
        Ok(())
    }

    fn reclaim(&mut self, queue: QueueKind, completed: u64) -> Result<usize, AllocationError> {
        let mut count = 0;
        let mut index = 0;
        while index < self.retired.len() {
            if self.retired[index].completion.queue == queue
                && self.retired[index].completion.value <= completed
            {
                let retired = self.retired.swap_remove(index);
                self.free(retired.allocation)?;
                count += 1;
            } else {
                index += 1;
            }
        }
        Ok(count)
    }
}

impl BufferTransfer for NativeContext {
    fn copy_buffer(
        &mut self,
        source: &Self::Allocation,
        destination: &Self::Allocation,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    ) -> Result<CompletionToken, AllocationError> {
        let completed = self.completed_transfer_value()?;
        self.completed_transfer_value = completed;
        self.pending_transfers
            .retain(|pending| pending.value > completed);
        validate_range(source.allocation.size(), source_offset, size)?;
        validate_range(destination.allocation.size(), destination_offset, size)?;
        let command = self
            .transfer_queue
            .commandBuffer()
            .ok_or(AllocationError::NativeFailure)?;
        let blit = command
            .blitCommandEncoder()
            .ok_or(AllocationError::NativeFailure)?;
        // SAFETY: the validated ranges fit the source and destination buffer storage, and those buffers and `blit` remain retained through the copy and `endEncoding` message sends.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &source.buffer,
                usize::try_from(source_offset).map_err(|_| AllocationError::NativeFailure)?,
                &destination.buffer,
                usize::try_from(destination_offset).map_err(|_| AllocationError::NativeFailure)?,
                usize::try_from(size).map_err(|_| AllocationError::NativeFailure)?,
            );
            blit.endEncoding();
        }
        let value = self.next_transfer_value;
        let next = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        let pending_command = command.clone();
        self.transfer_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .submit(super::transfer::MetalTransferJob {
                value,
                bytes: size,
                command: super::transfer::TransferCommand::new(command),
            })
            .map_err(|error| match error {
                ez_gfx_hal::TransferWorkerError::Full => AllocationError::OutOfMemory,
                ez_gfx_hal::TransferWorkerError::Failed => AllocationError::NativeFailure,
            })?;
        self.drain_complete = false;
        // Rejected admission must not create an unsignaled queue highwater.
        self.next_transfer_value = next;
        self.pending_transfers.push(super::PendingTransfer {
            value,
            command: super::ThreadBound::new(pending_command),
        });
        CompletionToken::new(QueueKind::Transfer, value).map_err(|_| AllocationError::NativeFailure)
    }

    fn completed_transfer_value(&self) -> Result<u64, AllocationError> {
        if self
            .transfer_worker
            .as_ref()
            .is_some_and(ez_gfx_hal::TransferWorker::failed)
        {
            return Err(AllocationError::NativeFailure);
        }
        let mut completed = self.completed_transfer_value;
        for pending in &self.pending_transfers {
            match pending.command.status() {
                MTLCommandBufferStatus::Completed if pending.command.error().is_none() => {
                    completed = completed.max(pending.value);
                }
                MTLCommandBufferStatus::Error => return Err(AllocationError::NativeFailure),
                _ => break,
            }
        }
        Ok(completed)
    }
}
impl Drop for NativeContext {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        if let Some(mut worker) = self.transfer_worker.take() {
            worker.shutdown();
        }
        if let Some(mut worker) = self.texture_worker.take() {
            worker.shutdown();
        }
        if !self.is_drained() {
            // A failed live-device drain must retain heap backing and every GPU-owned resource.
            core::mem::forget(self.allocator.take());
            core::mem::forget(core::mem::take(&mut self.retired));
            core::mem::forget(core::mem::take(&mut self.deferred));
            core::mem::forget(core::mem::take(&mut self.frame_slots));
            core::mem::forget(core::mem::take(&mut self.pending_transfers));
            core::mem::forget(core::mem::take(&mut self.pending_texture_transfers));
            for allocation in self.texture_staging.drain() {
                core::mem::forget(allocation);
            }
            return;
        }
        for allocation in self.texture_staging.drain() {
            let _ = self.free(allocation);
        }
        while let Some(retired) = self.retired.pop() {
            let _ = self.free(retired.allocation);
        }
        drop(self.allocator.take());
    }
}

pub(super) fn validate_range(length: u64, offset: u64, size: u64) -> Result<(), AllocationError> {
    if size == 0 || offset.checked_add(size).is_none_or(|end| end > length) {
        return Err(AllocationError::NativeFailure);
    }
    Ok(())
}
pub(super) fn map_allocation_hal(error: AllocationError) -> HalError {
    match error {
        AllocationError::OutOfMemory => HalError::OutOfMemory,
        AllocationError::ZeroSize
        | AllocationError::InvalidAlignment
        | AllocationError::InvalidAliasClass => HalError::InvalidArgument,
        AllocationError::DeviceLost => HalError::DeviceLost,
        AllocationError::Unsupported => HalError::Unsupported,
        AllocationError::NotHostVisible | AllocationError::NativeFailure => HalError::NativeFailure,
    }
}

pub(super) fn map_allocator_hal(error: gpu_allocator::AllocationError) -> HalError {
    let mapped = match &error {
        gpu_allocator::AllocationError::OutOfMemory => HalError::OutOfMemory,
        _ => HalError::NativeFailure,
    };
    drop(error);
    mapped
}
pub(super) fn map_allocator(error: gpu_allocator::AllocationError) -> AllocationError {
    let mapped = match &error {
        gpu_allocator::AllocationError::OutOfMemory => AllocationError::OutOfMemory,
        _ => AllocationError::NativeFailure,
    };
    drop(error);
    mapped
}
