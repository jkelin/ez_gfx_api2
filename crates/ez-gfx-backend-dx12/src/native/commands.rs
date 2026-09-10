use super::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_TYPE_UAV,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE,
    D3D12_RESOURCE_STATE_INDEX_BUFFER, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATES,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER, D3D12_TEXTURE_COPY_LOCATION,
    D3D12_TEXTURE_COPY_LOCATION_0, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
    D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, FRAMES_IN_FLIGHT, FrameSlot, HalError, ID3D12Device,
    ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12Resource, NativeBufferBinding,
    NativePipeline, ResourceAccess,
};

pub(super) fn dx12_resource_state(access: ResourceAccess) -> D3D12_RESOURCE_STATES {
    match access {
        ResourceAccess::SampledRead
        | ResourceAccess::StorageRead
        | ResourceAccess::IndirectStorageRead => {
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        }
        ResourceAccess::StorageWrite | ResourceAccess::StorageReadWrite => {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        }
        ResourceAccess::IndexRead => D3D12_RESOURCE_STATE_INDEX_BUFFER,
        ResourceAccess::IndirectRead => D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ResourceAccess::IndirectStorageReadWrite => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ResourceAccess::ColorAttachmentWrite => D3D12_RESOURCE_STATE_RENDER_TARGET,
        ResourceAccess::DepthStencilRead => D3D12_RESOURCE_STATE_DEPTH_READ,
        ResourceAccess::DepthStencilWrite => D3D12_RESOURCE_STATE_DEPTH_WRITE,
        ResourceAccess::TransferRead => D3D12_RESOURCE_STATE_COPY_SOURCE,
        ResourceAccess::TransferWrite => D3D12_RESOURCE_STATE_COPY_DEST,
        ResourceAccess::Present => D3D12_RESOURCE_STATE_PRESENT,
    }
}

pub(super) fn transition_barrier(
    resource: ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: core::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: core::mem::ManuallyDrop::new(Some(resource)),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

pub(super) fn uav_barrier(resource: ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: core::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: core::mem::ManuallyDrop::new(Some(resource)),
            }),
        },
    }
}

pub(super) fn record_resource_barriers<const N: usize>(
    list: &ID3D12GraphicsCommandList,
    mut barriers: [D3D12_RESOURCE_BARRIER; N],
) {
    // SAFETY: ResourceBarrier copies the initialized array during the call. Each Type selects the
    // initialized union arm whose cloned COM resource is released exactly once afterward.
    unsafe {
        list.ResourceBarrier(&barriers);
        for barrier in &mut barriers {
            if barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION {
                core::mem::ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
            } else if barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV {
                core::mem::ManuallyDrop::drop(&mut (*barrier.Anonymous.UAV).pResource);
            } else {
                unreachable!("resource barrier helper accepts only transition or UAV barriers");
            }
        }
    }
}

pub(super) unsafe fn copy_texture_to_readback(
    list: &ID3D12GraphicsCommandList,
    source: &ID3D12Resource,
    destination: &ID3D12Resource,
    footprint: windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let mut source = D3D12_TEXTURE_COPY_LOCATION {
        pResource: core::mem::ManuallyDrop::new(Some(source.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let mut destination = D3D12_TEXTURE_COPY_LOCATION {
        pResource: core::mem::ManuallyDrop::new(Some(destination.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    // SAFETY: the locations retain both resources through recording; the bindings require their
    // ManuallyDrop fields to be released explicitly after the native call has copied the values.
    unsafe {
        list.CopyTextureRegion(&raw const destination, 0, 0, 0, &raw const source, None);
        core::mem::ManuallyDrop::drop(&mut source.pResource);
        core::mem::ManuallyDrop::drop(&mut destination.pResource);
    }
}

///
/// # Errors
///
/// Returns `HalError::InvalidArgument` if a binding's writability does not match the pipeline, its offset is out of bounds, its root index cannot be represented, or its GPU virtual address overflows.
pub(super) unsafe fn bind_dx12_compute_buffers(
    list: &ID3D12GraphicsCommandList,
    pipeline: &NativePipeline,
    bindings: &[NativeBufferBinding<'_>],
) -> Result<(), HalError> {
    for (index, (binding, writable)) in bindings.iter().zip(&pipeline.buffer_writable).enumerate() {
        if binding.writable != *writable || binding.offset >= binding.allocation.allocation.size() {
            return Err(HalError::InvalidArgument);
        }
        let root_index = u32::try_from(index).map_err(|_| HalError::InvalidArgument)?;
        // SAFETY: the allocation resource is live and the validated offset stays within it.
        let address = unsafe {
            binding
                .allocation
                .resource
                .GetGPUVirtualAddress()
                .checked_add(binding.offset)
        }
        .ok_or(HalError::InvalidArgument)?;
        // SAFETY: SetComputeRootUnorderedAccessView/SetComputeRootShaderResourceView receives this pipeline binding's checked root index and the allocation resource's GPU base address plus an in-bounds offset.
        unsafe {
            if *writable {
                list.SetComputeRootUnorderedAccessView(root_index, address);
            } else {
                list.SetComputeRootShaderResourceView(root_index, address);
            }
        }
    }
    Ok(())
}

///
/// # Errors
///
/// Returns `HalError::InvalidArgument` if a binding's writability does not match the pipeline, its offset is out of bounds, its root index cannot be represented, or its GPU virtual address overflows.
pub(super) unsafe fn bind_dx12_graphics_buffers(
    list: &ID3D12GraphicsCommandList,
    pipeline: &NativePipeline,
    bindings: &[NativeBufferBinding<'_>],
) -> Result<(), HalError> {
    for (index, (binding, writable)) in bindings.iter().zip(&pipeline.buffer_writable).enumerate() {
        if binding.writable != *writable || binding.offset >= binding.allocation.allocation.size() {
            return Err(HalError::InvalidArgument);
        }
        let root_index = u32::try_from(index).map_err(|_| HalError::InvalidArgument)?;
        // SAFETY: the allocation resource is live and the validated offset stays within it.
        let address = unsafe {
            binding
                .allocation
                .resource
                .GetGPUVirtualAddress()
                .checked_add(binding.offset)
        }
        .ok_or(HalError::InvalidArgument)?;
        // SAFETY: SetGraphicsRootUnorderedAccessView/SetGraphicsRootShaderResourceView receives this pipeline binding's checked root index and the allocation resource's GPU base address plus an in-bounds offset.
        unsafe {
            if *writable {
                list.SetGraphicsRootUnorderedAccessView(root_index, address);
            } else {
                list.SetGraphicsRootShaderResourceView(root_index, address);
            }
        }
    }
    Ok(())
}

///
/// # Errors
///
/// Returns an error if creating a command allocator or command list, or closing a command list, fails.
pub(super) fn create_frame_slots(device: &ID3D12Device) -> windows::core::Result<Vec<FrameSlot>> {
    let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        let allocator =
            // SAFETY: CreateCommandAllocator is called through the referenced device interface with the defined DIRECT list type, and the Windows binding owns the returned interface's output storage.
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;
        // SAFETY: CreateCommandList uses the referenced device and same-device DIRECT allocator with matching list type; the allocator remains alive for the call and the Windows binding owns the returned interface's output storage.
        let list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None::<&ID3D12PipelineState>,
            )
        }?;
        // SAFETY: Close is called on the newly created ID3D12GraphicsCommandList while it is still in its initial recording state.
        unsafe { list.Close() }?;
        slots.push(FrameSlot {
            allocator,
            list,
            fence_value: 0,
            garbage: Vec::new(),
        });
    }
    Ok(slots)
}
