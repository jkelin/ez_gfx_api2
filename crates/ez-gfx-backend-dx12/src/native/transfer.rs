use super::{
    AllocationError, CloseHandle, CreateEventW, D3D12_COMMAND_LIST_TYPE_COPY,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FENCE_FLAG_NONE, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_TRANSITION_BARRIER,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
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
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        subresource: u32,
        destination_x: u32,
        destination_y: u32,
        transition_from_shader: bool,
        stream_stage: u32,
    },
}

pub(super) struct Dx12TransferJob {
    pub value: u64,
    pub bytes: u64,
    pub copy: Dx12TransferCopy,
    pub cancelled: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}
pub(super) fn job_group(job: &Dx12TransferJob) -> u64 {
    match &job.copy {
        // Only adjacent jobs coalesce. Updates remain separate from initial stream stages.
        Dx12TransferCopy::Texture {
            transition_from_shader: true,
            ..
        } => u64::MAX,
        Dx12TransferCopy::Texture { stream_stage, .. } => u64::from(*stream_stage) + 1,
        Dx12TransferCopy::Buffer { .. } => 0,
    }
}

pub(super) fn job_bytes(job: &Dx12TransferJob) -> u64 {
    job.bytes
}

fn job_cancelled(job: &Dx12TransferJob) -> bool {
    job.cancelled
        .as_ref()
        .is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::Acquire))
}
struct BatchResources {
    copy_allocator: ID3D12CommandAllocator,
    copy_list: ID3D12GraphicsCommandList,
    pre_transition_allocator: ID3D12CommandAllocator,
    pre_transition_list: ID3D12GraphicsCommandList,
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
    let pre_transition_allocator: ID3D12CommandAllocator =
        // SAFETY: the direct allocator is created from the same live device for the optional shader-to-copy transition.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: the pre-transition list and allocator share this live device.
    let pre_transition_list: ID3D12GraphicsCommandList = unsafe {
        device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &pre_transition_allocator,
            None,
        )
    }
    .map_err(|error| map_allocation_windows(&error))?;
    // SAFETY: the newly created direct list is idle and valid to close before its first reset.
    unsafe { pre_transition_list.Close() }.map_err(|error| map_allocation_windows(&error))?;
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
        pre_transition_allocator,
        pre_transition_list,
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
    // A private shutdown marker covers partially submitted batches without publishing
    // readiness on the application's completion fence.
    // SAFETY: the private fence shares the live device with both retained queues.
    let shutdown_fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|error| map_allocation_windows(&error))?;
    let shutdown_queues = [transfer_queue.clone(), graphics_queue.clone()];
    let mut slot = 0_usize;
    let mut copy_value = 0_u64;
    ez_gfx_hal::TransferWorker::new_ordered_with_shutdown(
        64,
        ez_gfx_hal::DEFAULT_STAGING_POLICY,
        job_bytes,
        job_group,
        |job| job.value,
        move |jobs| {
            let mut start = 0;
            while start < jobs.len() {
                let mut end = start + 1;
                while end < jobs.len() {
                    // Separate repeated-image writes into ordered native submissions; distinct
                    // textures still share one copy list and one graphics handoff.
                    if jobs[start..end].iter().any(|previous| {
                        matches!(
                            (&previous.copy, &jobs[end].copy),
                            (Dx12TransferCopy::Texture { destination: left, .. },
                             Dx12TransferCopy::Texture { destination: right, .. }) if left == right
                        )
                    }) {
                        break;
                    }
                    end += 1;
                }
                let resource = &mut resources[slot];
                submit_batch(
                    &transfer_queue,
                    &graphics_queue,
                    &completion_fence,
                    &copy_fence,
                    resource.event.handle(),
                    &resource.copy_allocator,
                    &resource.copy_list,
                    &resource.pre_transition_allocator,
                    &resource.pre_transition_list,
                    &resource.transition_allocator,
                    &resource.transition_list,
                    &mut resource.last_completion,
                    &mut copy_value,
                    &jobs[start..end],
                )
                .map_err(|_| ez_gfx_hal::TransferWorkerError::Failed)?;
                slot = (slot + 1) % resources.len();
                start = end;
            }
            Ok(())
        },
        move || {
            for (index, queue) in shutdown_queues.iter().enumerate() {
                let value = index as u64 + 1;
                // SAFETY: the marker follows every actual submission, including the
                // current batch's copy or pre-transition when a later call failed.
                unsafe { queue.Signal(&shutdown_fence, value) }
                    .map_err(|_| ez_gfx_hal::TransferWorkerError::Failed)?;
                loop {
                    // SAFETY: the closure retains this private fence until both queues drain.
                    let completed = unsafe { shutdown_fence.GetCompletedValue() };
                    if completed == u64::MAX {
                        return Err(ez_gfx_hal::TransferWorkerError::Failed);
                    }
                    if completed >= value {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
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
    pre_transition_allocator: &ID3D12CommandAllocator,
    pre_transition_list: &ID3D12GraphicsCommandList,
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
    // Cancellation is sampled once so every recorded release retains its copy and acquire.
    let live = jobs
        .iter()
        .filter(|job| !job_cancelled(job))
        .collect::<Vec<_>>();
    let texture_batch = jobs.first().is_some_and(|job| job_group(job) != 0);
    let mut textures = Vec::with_capacity(live.len());
    for job in &live {
        if let Dx12TransferCopy::Texture {
            destination,
            subresource,
            transition_from_shader,
            ..
        } = &job.copy
        {
            textures.push((destination, *subresource, *transition_from_shader));
        }
    }
    if textures.iter().any(|(_, _, initialized)| *initialized) {
        // SAFETY: this slot's previous completion retired before reset.
        unsafe {
            pre_transition_allocator.Reset()?;
            pre_transition_list.Reset(pre_transition_allocator, None)?;
        }
        for &(destination, subresource, initialized) in &textures {
            if !initialized {
                continue;
            }
            let transition = D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: core::mem::ManuallyDrop::new(Some(destination.clone())),
                Subresource: subresource,
                StateBefore: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
            };
            let mut barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: core::mem::ManuallyDrop::new(transition),
                },
            };
            // SAFETY: recording copies the barrier; jobs retain the resource until submission.
            unsafe {
                pre_transition_list.ResourceBarrier(core::slice::from_ref(&barrier));
                core::mem::ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
            }
        }
        // SAFETY: all pre-copy transitions have been recorded.
        unsafe { pre_transition_list.Close()? };
        let transition_command: ID3D12CommandList = pre_transition_list.cast()?;
        *copy_value = copy_value.checked_add(1).ok_or_else(|| {
            windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL)
        })?;
        // SAFETY: the COPY queue waits for every graphics release in this batch.
        unsafe {
            graphics_queue.ExecuteCommandLists(&[Some(transition_command)]);
            graphics_queue.Signal(copy_fence, *copy_value)?;
            transfer_queue.Wait(copy_fence, *copy_value)?;
        }
    }
    for job in &live {
        record_copy(copy_list, &job.copy);
    }
    // SAFETY: recording is complete and the list remains retained for submission.
    unsafe { copy_list.Close()? };
    let copy_command: ID3D12CommandList = copy_list.cast()?;
    // SAFETY: the closed copy list and queue share the live device.
    unsafe { transfer_queue.ExecuteCommandLists(&[Some(copy_command)]) };
    #[cfg(test)]
    super::texture_tests::copy_submitted(transfer_queue)?;
    let final_value = jobs.iter().map(|job| job.value).max().unwrap_or(0);

    // Cancelled texture batches must signal on graphics too; signaling on COPY could overtake
    // an earlier live batch's graphics transition and falsely retire its completion token.
    if !texture_batch {
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
    for &(resource, subresource, _) in &textures {
        let transition = D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
            Subresource: subresource,
            StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
            StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        };
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: core::mem::ManuallyDrop::new(transition),
            },
        };
        // SAFETY: the barrier references the retained texture's selected copied subresource.
        unsafe {
            transition_list.ResourceBarrier(core::slice::from_ref(&barrier));
            core::mem::ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
        }
    }
    // SAFETY: transition recording is complete and the list remains retained for submission.
    unsafe { transition_list.Close()? };
    let transition_command: ID3D12CommandList = transition_list.cast()?;
    {
        // Fine updates outside a published view must not block coarse frames on graphics.
        // The dedicated owner drains submitted storage even after cancellation/shutdown.
        // SAFETY: the worker retains this fence and event until the copy has completed.
        unsafe { copy_fence.SetEventOnCompletion(*copy_value, event)? };
        loop {
            // SAFETY: the worker owns the live copy fence.
            let completed = unsafe { copy_fence.GetCompletedValue() };
            if completed == u64::MAX {
                return Err(windows::core::Error::from_hresult(
                    windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED,
                ));
            }
            if completed >= *copy_value {
                break;
            }
            // SAFETY: the event remains live across each bounded wait.
            if unsafe { WaitForSingleObject(event, 100) } == windows::Win32::Foundation::WAIT_FAILED
            {
                return Err(windows::core::Error::from_thread());
            }
        }
    }
    // SAFETY: both queues, fences, and the closed list share the live device.
    unsafe {
        graphics_queue.Wait(copy_fence, *copy_value)?;
        graphics_queue.ExecuteCommandLists(&[Some(transition_command)]);
        graphics_queue.Signal(completion_fence, final_value)?;
    }
    *last_completion = final_value;
    Ok(())
}

fn record_copy(copy_list: &ID3D12GraphicsCommandList, copy: &Dx12TransferCopy) {
    // Locations describe validated extents; clipped BC mip edges retain complete source blocks.
    match copy {
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
            footprint,
            subresource,
            destination_x,
            destination_y,
            ..
        } => {
            let mut source_location = D3D12_TEXTURE_COPY_LOCATION {
                pResource: core::mem::ManuallyDrop::new(Some(source.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            let mut destination_location = D3D12_TEXTURE_COPY_LOCATION {
                pResource: core::mem::ManuallyDrop::new(Some(destination.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: *subresource,
                },
            };
            // SAFETY: validated locations retain both resources and the footprint describes
            // the staging allocation copied before this job was admitted.
            unsafe {
                copy_list.CopyTextureRegion(
                    &raw const destination_location,
                    *destination_x,
                    *destination_y,
                    0,
                    &raw const source_location,
                    None,
                );
                core::mem::ManuallyDrop::drop(&mut source_location.pResource);
                core::mem::ManuallyDrop::drop(&mut destination_location.pResource);
            }
        }
    }
}

#[cfg(test)]
pub(super) fn submit_test_batch(
    device: &ID3D12Device,
    transfer_queue: &ID3D12CommandQueue,
    graphics_queue: &ID3D12CommandQueue,
    completion_fence: &ID3D12Fence,
    jobs: &[Dx12TransferJob],
) -> u64 {
    // Keep the real worker command slot alive until its final native submission retires.
    assert!(!jobs.is_empty());
    let mut resources = create_batch_resources(device).unwrap();
    // SAFETY: the live device owns the private fence.
    let copy_fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
    let mut copy_value = 0;
    submit_batch(
        transfer_queue,
        graphics_queue,
        completion_fence,
        &copy_fence,
        resources.event.handle(),
        &resources.copy_allocator,
        &resources.copy_list,
        &resources.pre_transition_allocator,
        &resources.pre_transition_list,
        &resources.transition_allocator,
        &resources.transition_list,
        &mut resources.last_completion,
        &mut copy_value,
        jobs,
    )
    .unwrap();
    // SAFETY: resources and event remain alive until all lists have finished.
    unsafe {
        completion_fence
            .SetEventOnCompletion(resources.last_completion, resources.event.handle())
            .unwrap();
        WaitForSingleObject(resources.event.handle(), INFINITE);
    }
    copy_value
}
