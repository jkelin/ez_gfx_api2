use super::{
    AllocationError, CloseHandle, CreateEventW, D3D12_COMMAND_LIST_TYPE_COPY,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FENCE_FLAG_NONE, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, HANDLE,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12Resource, INFINITE, Interface, WaitForSingleObject,
    map_allocation_windows,
};
struct WorkerEvent(HANDLE);

// SAFETY: Win32 event handles are process-wide synchronization objects.
unsafe impl Send for WorkerEvent {}
impl WorkerEvent {
    const fn handle(&self) -> HANDLE {
        self.0
    }
}

impl Drop for WorkerEvent {
    fn drop(&mut self) {
        // SAFETY: this wrapper owns the event returned by `CreateEventW`.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

pub(super) enum Dx12TransferCopy {
    Buffer {
        source: ID3D12Resource,
        destination: ID3D12Resource,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    },
    Texture {
        source: ID3D12Resource,
        destination: ID3D12Resource,
        footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    },
}

pub(super) struct Dx12TransferJob {
    pub value: u64,
    pub bytes: u64,
    pub copy: Dx12TransferCopy,
}
pub(super) fn job_group(job: &Dx12TransferJob) -> u64 {
    u64::from(matches!(job.copy, Dx12TransferCopy::Texture { .. }))
}

pub(super) fn job_bytes(job: &Dx12TransferJob) -> u64 {
    job.bytes
}
struct BatchResources {
    copy_allocator: ID3D12CommandAllocator,
    copy_list: ID3D12GraphicsCommandList,
    transition_allocator: ID3D12CommandAllocator,
    transition_list: ID3D12GraphicsCommandList,
    event: WorkerEvent,
    last_completion: u64,
}

fn create_batch_resources(device: &ID3D12Device) -> Result<BatchResources, AllocationError> {
    // SAFETY: command allocators and lists are created from the same live device and retained together.
    let copy_allocator = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY) }
        .map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: the copy list and allocator share this live device.
    let copy_list: ID3D12GraphicsCommandList =
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &copy_allocator, None) }
            .map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: the newly created copy list is idle and valid to close before its first reset.
    unsafe { copy_list.Close() }.map_err(|error| map_allocation_windows(&error))?;
    let transition_allocator: ID3D12CommandAllocator =
    // SAFETY: the direct allocator is created from the same live device for ownership transitions.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: the transition list and allocator share this live device.
    let transition_list: ID3D12GraphicsCommandList = unsafe {
        device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &transition_allocator,
            None,
        )
    }
    .map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: the newly created transition list is idle and valid to close before its first reset.
    unsafe { transition_list.Close() }.map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: an unnamed auto-reset event needs no security attributes.
    let event = WorkerEvent(
        unsafe { CreateEventW(None, false, false, None) }
            .map_err(|error| map_allocation_windows(&error))?,
    );
    Ok(BatchResources {
        copy_allocator,
        copy_list,
        transition_allocator,
        transition_list,
        event,
        last_completion: 0,
    })
}

pub(super) fn start_worker(
    device: &ID3D12Device,
    transfer_queue: ID3D12CommandQueue,
    graphics_queue: ID3D12CommandQueue,
    completion_fence: ID3D12Fence,
) -> Result<ez_gfx_hal::TransferWorker<Dx12TransferJob>, AllocationError> {
    // Four reusable command-resource slots allow batches to overlap without resetting in-flight lists.
    let mut resources = (0..4)
        .map(|_| create_batch_resources(device))
        .collect::<Result<Vec<_>, _>>()?;
    // SAFETY: the worker-local fence is created from the live device and retained by its closure.
    let copy_fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: an unnamed auto-reset event needs no security attributes.
    let shutdown_event = WorkerEvent(
        unsafe { CreateEventW(None, false, false, None) }
            .map_err(|error| map_allocation_windows(&error))?,
    );
    let shutdown_fence = completion_fence.clone();
    let final_completion = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let submitted_completion = final_completion.clone();
    let mut slot = 0_usize;
    let mut copy_value = 0_u64;
    ez_gfx_hal::TransferWorker::new_grouped_with_shutdown(
        64,
        ez_gfx_hal::DEFAULT_STAGING_POLICY,
        job_bytes,
        job_group,
        move |jobs| {
            let resource = &mut resources[slot];
            submit_batch(
                &transfer_queue,
                &graphics_queue,
                &completion_fence,
                &copy_fence,
                resource.event.handle(),
                &resource.copy_allocator,
                &resource.copy_list,
                &resource.transition_allocator,
                &resource.transition_list,
                &mut resource.last_completion,
                &mut copy_value,
                &jobs,
            )
            .map_err(|_| ez_gfx_hal::TransferWorkerError::Failed)?;
            submitted_completion.store(
                resource.last_completion,
                std::sync::atomic::Ordering::Release,
            );
            slot = (slot + 1) % resources.len();
            Ok(())
        },
        move || {
            let value = final_completion.load(std::sync::atomic::Ordering::Acquire);
            // SAFETY: the shutdown closure retains both the fence and event until the wait completes.
            if value != 0 && unsafe { shutdown_fence.GetCompletedValue() } < value {
                // SAFETY: the event is live and owned by this closure.
                unsafe { shutdown_fence.SetEventOnCompletion(value, shutdown_event.handle()) }
                    .map_err(|_| ez_gfx_hal::TransferWorkerError::Failed)?;
                // SAFETY: the event remains live until this blocking wait returns.
                unsafe { WaitForSingleObject(shutdown_event.handle(), INFINITE) };
            }
            // SAFETY: the retained fence remains live throughout worker shutdown.
            let completed = unsafe { shutdown_fence.GetCompletedValue() };
            if completed == u64::MAX || completed < value {
                return Err(ez_gfx_hal::TransferWorkerError::Failed);
            }
            Ok(())
        },
    )
    .map_err(|_| AllocationError::NativeFailure)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the worker keeps reusable COPY/DIRECT command state and paired queue synchronization local"
)]
fn submit_batch(
    transfer_queue: &ID3D12CommandQueue,
    graphics_queue: &ID3D12CommandQueue,
    completion_fence: &ID3D12Fence,
    copy_fence: &ID3D12Fence,
    event: HANDLE,
    copy_allocator: &ID3D12CommandAllocator,
    copy_list: &ID3D12GraphicsCommandList,
    transition_allocator: &ID3D12CommandAllocator,
    transition_list: &ID3D12GraphicsCommandList,
    last_completion: &mut u64,
    copy_value: &mut u64,
    jobs: &[Dx12TransferJob],
) -> windows::core::Result<()> {
    // SAFETY: the completion fence remains live while retained by the worker.
    let completed = unsafe { completion_fence.GetCompletedValue() };
    if *last_completion != 0 && completed < *last_completion {
        // SAFETY: `event` is a live waitable handle retained by the batch resource.
        unsafe { completion_fence.SetEventOnCompletion(*last_completion, event)? };
        // SAFETY: `event` remains live until this blocking wait returns.
        unsafe { WaitForSingleObject(event, INFINITE) };
    }
    // SAFETY: the slot is idle after the preceding completion wait, so its allocator and list may reset.
    unsafe {
        copy_allocator.Reset()?;
        copy_list.Reset(copy_allocator, None)?;
    }
    let mut textures = Vec::new();
    for job in jobs {
        match &job.copy {
            Dx12TransferCopy::Buffer {
                source,
                destination,
                source_offset,
                destination_offset,
                size,
            } => {
                // SAFETY: validated resource ranges stay live through execution of this copy list.
                unsafe {
                    copy_list.CopyBufferRegion(
                        destination,
                        *destination_offset,
                        source,
                        *source_offset,
                        *size,
                    );
                }
            }
            Dx12TransferCopy::Texture {
                source,
                destination,
                footprints,
            } => {
                for (level, footprint) in footprints.iter().enumerate() {
                    let level = u32::try_from(level).map_err(|_| {
                        windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG)
                    })?;
                    let source_location = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: core::mem::ManuallyDrop::new(Some(source.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: *footprint,
                        },
                    };
                    let destination_location = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: core::mem::ManuallyDrop::new(Some(destination.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: level,
                        },
                    };
                    // SAFETY: source and destination locations describe retained resources and validated footprints.
                    unsafe {
                        copy_list.CopyTextureRegion(
                            &raw const destination_location,
                            0,
                            0,
                            0,
                            &raw const source_location,
                            None,
                        );
                    }
                }
                textures.push(destination.clone());
            }
        }
    }
    // SAFETY: recording is complete and the list remains retained for submission.
    unsafe { copy_list.Close()? };
    let copy_command: ID3D12CommandList = copy_list.cast()?;
    // SAFETY: the closed copy list and queue share the live device.
    unsafe { transfer_queue.ExecuteCommandLists(&[Some(copy_command)]) };
    let final_value = jobs.iter().map(|job| job.value).max().unwrap_or(0);

    if textures.is_empty() {
        // SAFETY: the completion fence and transfer queue share the live device.
        unsafe { transfer_queue.Signal(completion_fence, final_value)? };
        *last_completion = final_value;
        return Ok(());
    }

    *copy_value = copy_value
        .checked_add(1)
        .ok_or_else(|| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))?;
    // SAFETY: the copy fence and transfer queue share the live device.
    unsafe { transfer_queue.Signal(copy_fence, *copy_value)? };
    // SAFETY: the direct slot is idle after the completion wait and may be reset.
    unsafe {
        transition_allocator.Reset()?;
        transition_list.Reset(transition_allocator, None)?;
    }
    for resource in textures {
        let transition = D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: core::mem::ManuallyDrop::new(Some(resource)),
            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
            StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        };
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: core::mem::ManuallyDrop::new(transition),
            },
        };
        // SAFETY: the barrier references a retained texture and all subresources.
        unsafe { transition_list.ResourceBarrier(&[barrier]) };
    }
    // SAFETY: transition recording is complete and the list remains retained for submission.
    unsafe { transition_list.Close()? };
    let transition_command: ID3D12CommandList = transition_list.cast()?;
    // SAFETY: both queues, fences, and the closed list share the live device.
    unsafe {
        graphics_queue.Wait(copy_fence, *copy_value)?;
        graphics_queue.ExecuteCommandLists(&[Some(transition_command)]);
        graphics_queue.Signal(completion_fence, final_value)?;
    }
    *last_completion = final_value;
    Ok(())
}
