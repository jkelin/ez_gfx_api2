use super::{
    AllocationCreateDesc, AllocationError, AllocationRequest, BufferTransfer, CloseHandle,
    CompletionToken, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT, D3D12_RANGE, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, DXGI_FORMAT_UNKNOWN,
    DXGI_SAMPLE_DESC, DeferredResource, HalError, ID3D12Resource, MemoryAllocator, MemoryClass,
    MemoryLocation, NativeAllocation, NativeContext, QueueKind, RetiredAllocation, c_void, ptr,
    transfer::{Dx12TransferCopy, Dx12TransferJob},
};

impl MemoryAllocator for NativeContext {
    type Allocation = NativeAllocation;

    fn allocate(
        &mut self,
        request: AllocationRequest,
    ) -> Result<Self::Allocation, AllocationError> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: u64::from(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT),
            Width: request.size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if matches!(
                request.memory_class,
                MemoryClass::Device | MemoryClass::Transient
            ) {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        };
        let location = match request.memory_class {
            MemoryClass::Device | MemoryClass::Transient => MemoryLocation::GpuOnly,
            MemoryClass::Upload => MemoryLocation::CpuToGpu,
            MemoryClass::Readback => MemoryLocation::GpuToCpu,
        };
        let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
            self.allocator
                .as_ref()
                .ok_or(AllocationError::NativeFailure)?
                .device(),
            &desc,
            "ez-gfx-buffer",
            location,
        );
        if allocation_desc.alignment < request.alignment {
            return Err(AllocationError::InvalidAlignment);
        }
        let allocation = self
            .allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .allocate(&allocation_desc)
            .map_err(|error| map_allocator(&error))?;
        let initial_state = match request.memory_class {
            MemoryClass::Upload => D3D12_RESOURCE_STATE_GENERIC_READ,
            MemoryClass::Readback => D3D12_RESOURCE_STATE_COPY_DEST,
            _ => D3D12_RESOURCE_STATE_COMMON,
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: CreatePlacedResource receives the heap and aligned offset allocated for `desc`; `desc` and the `resource` out-slot remain initialized and addressable for the call.
        let created = unsafe {
            self.device.CreatePlacedResource(
                allocation.heap(),
                allocation.offset(),
                &raw const desc,
                initial_state,
                None,
                &raw mut resource,
            )
        };
        if let Err(error) = created {
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
            return Err(map_allocation_windows(&error));
        }
        let resource = resource.ok_or(AllocationError::NativeFailure)?;
        let mut mapped_address = 0;
        if request.mapped {
            let mut mapped: *mut c_void = ptr::null_mut();
            // SAFETY: `resource` is a buffer with subresource 0, and `mapped` is writable pointer-result storage that remains addressable for the `Map` call.
            unsafe { resource.Map(0, None, Some(&raw mut mapped)) }
                .map_err(|error| map_allocation_windows(&error))?;
            if mapped.is_null() {
                return Err(AllocationError::NotHostVisible);
            }
            mapped_address = mapped as usize;
        }
        Ok(NativeAllocation {
            resource,
            allocation,
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
        // SAFETY: `mapped_address` is Map's non-null base for this allocation's still-mapped storage, and the checked byte length fits `isize` and remains valid for the allocation borrow.
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
        // SAFETY: `mapped_address` is Map's non-null base for this allocation's still-mapped storage, the checked length fits `isize`, and `&mut allocation` provides exclusive access for the returned slice lifetime.
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
        let written = D3D12_RANGE {
            Begin: usize::try_from(offset).map_err(|_| AllocationError::NativeFailure)?,
            End: usize::try_from(
                offset
                    .checked_add(size)
                    .ok_or(AllocationError::NativeFailure)?,
            )
            .map_err(|_| AllocationError::NativeFailure)?,
        };
        // SAFETY: nonzero `mapped_address` records that resource subresource 0 is mapped, and `validate_range` bounds the initialized `written` range whose storage lasts through `Unmap`.
        unsafe {
            allocation.resource.Unmap(0, Some(&raw const written));
        }
        let mut mapped: *mut c_void = ptr::null_mut();
        // SAFETY: the preceding `Unmap` ended the subresource-0 mapping, and `mapped` is writable pointer-result storage that remains addressable for this buffer's `Map` call.
        unsafe { allocation.resource.Map(0, None, Some(&raw mut mapped)) }
            .map_err(|error| map_allocation_windows(&error))?;
        allocation.mapped_address = mapped as usize;
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
        let no_write = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: nonzero `mapped_address` records that resource subresource 0 is mapped, and initialized `no_write` storage remains addressable through `Unmap`.
        unsafe {
            allocation.resource.Unmap(0, Some(&raw const no_write));
        }
        let read = D3D12_RANGE {
            Begin: usize::try_from(offset).map_err(|_| AllocationError::NativeFailure)?,
            End: usize::try_from(
                offset
                    .checked_add(size)
                    .ok_or(AllocationError::NativeFailure)?,
            )
            .map_err(|_| AllocationError::NativeFailure)?,
        };
        let mut mapped: *mut c_void = ptr::null_mut();
        // SAFETY: `validate_range` bounds the initialized `read` range to subresource 0, and `read` plus writable `mapped` output storage remain addressable through `Map`.
        unsafe {
            allocation
                .resource
                .Map(0, Some(&raw const read), Some(&raw mut mapped))
        }
        .map_err(|error| map_allocation_windows(&error))?;
        allocation.mapped_address = mapped as usize;
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
        validate_range(source.allocation.size(), source_offset, size)?;
        validate_range(destination.allocation.size(), destination_offset, size)?;
        let value = self.next_transfer_fence;
        let next = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        self.transfer_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .submit(Dx12TransferJob {
                value,
                bytes: size,
                cancelled: None,
                copy: Dx12TransferCopy::Buffer {
                    source: source.resource.clone(),
                    destination: destination.resource.clone(),
                    source_offset,
                    destination_offset,
                    size,
                },
            })
            .map_err(ez_gfx_hal::TransferWorkerError::to_allocation_error)?;
        // Rejected work must not leave an unsignalable idle-wait target.
        self.next_transfer_fence = next;
        CompletionToken::new(QueueKind::Transfer, value).map_err(|_| AllocationError::NativeFailure)
    }

    fn completed_transfer_value(&self) -> Result<u64, AllocationError> {
        if let Some(error) = self
            .transfer_worker
            .as_ref()
            .and_then(ez_gfx_hal::TransferWorker::terminal_error)
        {
            return Err(error.to_allocation_error());
        }
        // SAFETY: the transfer fence is retained by this context.
        let value = unsafe { self.transfer_fence.GetCompletedValue() };
        if value == u64::MAX {
            return Err(AllocationError::DeviceLost);
        }
        Ok(value)
    }
}
impl Drop for NativeContext {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        if let Some(mut worker) = self.texture_worker.take() {
            worker.shutdown();
        }
        if let Some(mut worker) = self.transfer_worker.take() {
            worker.shutdown();
        }
        if !self.is_drained() {
            // Retain at most this failed context's GPU owners. Releasing COM
            // references or allocator heaps before an actual drain is unsafe.
            core::mem::forget((
                self.adapter.clone(),
                self.device.clone(),
                self.queue.clone(),
                self.fence.clone(),
                self.transfer_fence.clone(),
                self.texture_fence.clone(),
                self.descriptors.clone(),
                self.samplers.clone(),
                self.allocator.take(),
                core::mem::take(&mut self.retired),
                core::mem::take(&mut self.deferred),
                core::mem::take(&mut self.frame_slots),
                core::mem::replace(
                    &mut self.texture_staging,
                    ez_gfx_hal::ReusableStagingPool::new(256),
                ),
            ));
            // The event may still be registered with a live native fence.
            return;
        }
        for allocation in self.texture_staging.drain() {
            let _ = self.free(allocation);
        }
        while let Some(retired) = self.retired.pop() {
            let _ = self.free(retired.allocation);
        }
        drop(self.allocator.take());
        // SAFETY: `fence_event` is the open event handle stored by `NativeContext` after creation and is closed exactly once here by `CloseHandle` during `Drop`.
        unsafe {
            let _ = CloseHandle(self.fence_event);
        }
    }
}

pub(super) fn adapter_id(desc: &windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_DESC3) -> [u8; 16] {
    let mut id = [0_u8; 16];
    id[0..4].copy_from_slice(&desc.VendorId.to_le_bytes());
    id[4..8].copy_from_slice(&desc.DeviceId.to_le_bytes());
    id[8..12].copy_from_slice(&desc.AdapterLuid.LowPart.to_le_bytes());
    id[12..16].copy_from_slice(&desc.AdapterLuid.HighPart.to_le_bytes());
    id
}

///
/// # Errors
///
/// Returns `AllocationError::NativeFailure` if `size` is zero, `offset + size` overflows, or the range exceeds `length`.
pub(super) fn validate_range(length: u64, offset: u64, size: u64) -> Result<(), AllocationError> {
    if size == 0 || offset.checked_add(size).is_none_or(|end| end > length) {
        return Err(AllocationError::NativeFailure);
    }
    Ok(())
}

pub(super) fn map_windows(_: windows::core::Error) -> HalError {
    HalError::NativeFailure
}
pub(super) fn map_allocation_windows(error: &windows::core::Error) -> AllocationError {
    if error.code() == windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED {
        AllocationError::DeviceLost
    } else {
        AllocationError::NativeFailure
    }
}
pub(super) fn map_allocator(error: &gpu_allocator::AllocationError) -> AllocationError {
    match error {
        gpu_allocator::AllocationError::OutOfMemory => AllocationError::OutOfMemory,
        _ => AllocationError::NativeFailure,
    }
}

pub(super) fn map_allocator_hal(error: &gpu_allocator::AllocationError) -> HalError {
    match error {
        gpu_allocator::AllocationError::OutOfMemory => HalError::OutOfMemory,
        _ => HalError::NativeFailure,
    }
}
