use super::{
    AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, CompletionToken,
    D3D12_CLEAR_FLAG_DEPTH, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_INDEX_BUFFER_VIEW,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_VIEWPORT, DXGI_FORMAT_R32_UINT, DXGI_PRESENT,
    FRAMES_IN_FLIGHT, HalError, ID3D12CommandList, ID3D12PipelineState, INFINITE, Interface,
    MemoryAllocator, MemoryClass, NativeAllocation, NativeContext, NativeFrameAction,
    NativeFrameResource, NativeSurface, QueueKind, RECT, WaitForSingleObject,
    bind_dx12_compute_buffers, bind_dx12_graphics_buffers, copy_texture_to_readback,
    dx12_resource_state, map_windows, transition_barrier, uav_barrier,
};

const DRAW_INDEXED_ARGUMENT_BYTES: u64 = core::mem::size_of::<
    windows::Win32::Graphics::Direct3D12::D3D12_DRAW_INDEXED_ARGUMENTS,
>() as u64;

struct DxFramePlan {
    uses_surface: bool,
    presents: bool,
    external_waits: Vec<CompletionToken>,
}

// D3D12 copy commands are invalid between BeginRenderPass and EndRenderPass.
fn validate_indirect_copy_phase(pass_active: bool) -> Result<(), HalError> {
    if pass_active {
        Err(HalError::InvalidArgument)
    } else {
        Ok(())
    }
}
fn indirect_command_bytes(draw_count: u32) -> u64 {
    u64::from(draw_count) * DRAW_INDEXED_ARGUMENT_BYTES
}

fn validate_frame_plan(
    actions: &[NativeFrameAction<'_>],
    extent: (u32, u32),
    surface_available: bool,
    capture_presented: bool,
) -> Result<DxFramePlan, HalError> {
    let present_count = actions
        .iter()
        .filter(|action| matches!(action, NativeFrameAction::Present))
        .count();
    let presents = present_count == 1;
    // Surface use is attachment-precise: render-target-only passes, barriers,
    // and draws never acquire the swapchain. Draws inherit their pass target,
    // so only surface-attached passes, presents, and surface/depth barriers
    // require a surface.
    let uses_surface = actions.iter().any(|action| match action {
        NativeFrameAction::BeginPass { colors, .. } => colors.iter().any(|attachment| {
            matches!(attachment.resource, NativeFrameResource::Surface)
        }),
        NativeFrameAction::Present => true,
        NativeFrameAction::Barrier {
            resource: NativeFrameResource::Surface | NativeFrameResource::Depth,
            ..
        } => true,
        _ => false,
    });
    if present_count > 1
        || uses_surface && !surface_available
        || (uses_surface || capture_presented) && !presents
    {
        return Err(HalError::InvalidArgument);
    }
    let external_waits = actions
        .iter()
        .filter_map(|action| match action {
            NativeFrameAction::Wait(token)
                if matches!(
                    token.queue,
                    QueueKind::Transfer | QueueKind::TextureTransfer
                ) =>
            {
                Some(Ok(*token))
            }
            NativeFrameAction::Wait(_) => Some(Err(HalError::InvalidArgument)),
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut pass_active = false;
    let mut saw_present = false;
    for action in actions {
        if saw_present {
            return Err(HalError::InvalidArgument);
        }
        match action {
            NativeFrameAction::Wait(_) | NativeFrameAction::Barrier { .. } => {}
            NativeFrameAction::BeginPass { pass, colors } => {
                // Textures, buffers, and depth images are never color
                // attachments; depth with a render target stays unsupported.
                let mut target_extent = None;
                let mut valid = !pass_active
                    && pass.colors.len() == 1
                    && colors.len() == 1
                    && pass.samples == 1;
                if let Some(attachment) = colors.first() {
                    target_extent = match attachment.resource {
                        NativeFrameResource::Surface => Some(extent),
                        NativeFrameResource::RenderTarget(texture) => {
                            valid &= pass.depth.is_none();
                            Some((texture.width, texture.height))
                        }
                        _ => None,
                    };
                }
                let Some((target_width, target_height)) = target_extent else {
                    return Err(HalError::InvalidArgument);
                };
                if !valid
                    || pass.area[0]
                        .checked_add(pass.area[2])
                        .is_none_or(|end| end > target_width)
                    || pass.area[1]
                        .checked_add(pass.area[3])
                        .is_none_or(|end| end > target_height)
                {
                    return Err(HalError::InvalidArgument);
                }
                pass_active = true;
            }
            NativeFrameAction::Compute(dispatch) => {
                if pass_active
                    || dispatch.groups.contains(&0)
                    || dispatch.push_constants.len() > 128
                    || !dispatch.push_constants.len().is_multiple_of(4)
                    || dispatch.bindings.len() != dispatch.pipeline.buffer_writable.len()
                    || dispatch
                        .bindings
                        .iter()
                        .zip(&dispatch.pipeline.buffer_writable)
                        .any(|(binding, writable)| {
                            binding.writable != *writable
                                || binding.offset >= binding.allocation.allocation.size()
                        })
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::Graphics(draw) => {
                let indirect_size = indirect_command_bytes(draw.draw_count);
                if !pass_active
                    || draw.draw_count == 0
                    || draw.push_constants.len() > 128
                    || !draw.push_constants.len().is_multiple_of(4)
                    || draw.bindings.len() != draw.pipeline.buffer_writable.len()
                    || draw.pipeline.topology.is_none()
                    || draw.pipeline.signature.is_none()
                    || draw.index_size == 0
                    || draw.index_size > u64::from(u32::MAX)
                    || draw.indirect_size < indirect_size
                    || draw
                        .bindings
                        .iter()
                        .zip(&draw.pipeline.buffer_writable)
                        .any(|(binding, writable)| {
                            binding.writable != *writable
                                || binding.offset >= binding.allocation.allocation.size()
                        })
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::TextureReadback { width, height, .. } => {
                if pass_active || *width == 0 || *height == 0 {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::EndPass => {
                if !pass_active {
                    return Err(HalError::InvalidArgument);
                }
                pass_active = false;
            }
            NativeFrameAction::Present => {
                if pass_active {
                    return Err(HalError::InvalidArgument);
                }
                saw_present = true;
            }
        }
    }
    if pass_active {
        return Err(HalError::InvalidArgument);
    }
    Ok(DxFramePlan {
        uses_surface,
        presents,
        external_waits,
    })
}

struct DxPreparedFrame {
    swapchain: Option<super::IDXGISwapChain4>,
    back_buffer: Option<super::ID3D12Resource>,
    rtv: Option<windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
    dsv: Option<windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
    slot_index: usize,
    list: super::ID3D12GraphicsCommandList,
}

type DxReadback = (
    u32,
    u32,
    windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    u64,
    NativeAllocation,
);

type DxFrameResources = (Vec<DxReadback>, Vec<Option<NativeAllocation>>);

impl NativeContext {
    fn collect_frame_readbacks(
        &mut self,
        readbacks: Vec<DxReadback>,
        capture_presented: bool,
        surface: Option<&mut NativeSurface>,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        let mut outputs = Vec::with_capacity(readbacks.len());
        let mut remaining = readbacks.into_iter();
        while let Some((width, height, footprint, total, mut readback)) = remaining.next() {
            let output = (|| {
                self.invalidate(&mut readback, 0, total)
                    .map_err(|_| HalError::NativeFailure)?;
                let source = self
                    .mapped_slice(&readback)
                    .map_err(|_| HalError::NativeFailure)?;
                let row_bytes = (width as usize)
                    .checked_mul(4)
                    .ok_or(HalError::InvalidArgument)?;
                let pixel_bytes = row_bytes
                    .checked_mul(height as usize)
                    .ok_or(HalError::InvalidArgument)?;
                let mut pixels = vec![0; pixel_bytes];
                for row in 0..height as usize {
                    let source_start = row
                        .checked_mul(footprint.Footprint.RowPitch as usize)
                        .ok_or(HalError::InvalidArgument)?;
                    let source_end = source_start
                        .checked_add(row_bytes)
                        .ok_or(HalError::InvalidArgument)?;
                    let destination_start = row
                        .checked_mul(row_bytes)
                        .ok_or(HalError::InvalidArgument)?;
                    let destination_end = destination_start
                        .checked_add(row_bytes)
                        .ok_or(HalError::InvalidArgument)?;
                    pixels
                        .get_mut(destination_start..destination_end)
                        .ok_or(HalError::NativeFailure)?
                        .copy_from_slice(
                            source
                                .get(source_start..source_end)
                                .ok_or(HalError::NativeFailure)?,
                        );
                }
                Ok(pixels)
            })();
            let freed = self.free(readback).map_err(|_| HalError::NativeFailure);
            match (output, freed) {
                (Ok(pixels), Ok(())) => outputs.push(pixels),
                (Err(error), _) | (_, Err(error)) => {
                    for (_, _, _, _, allocation) in remaining {
                        let _ = self.free(allocation);
                    }
                    return Err(error);
                }
            }
        }
        if capture_presented && let Some(pixels) = outputs.last() {
            surface
                .ok_or(HalError::InvalidArgument)?
                .presented
                .clone_from(pixels);
        }
        Ok(outputs)
    }
}

impl NativeContext {
    fn allocate_frame_resources(
        &mut self,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
        extent: (u32, u32),
        back_buffer: Option<&super::ID3D12Resource>,
    ) -> Result<DxFrameResources, HalError> {
        let mut readbacks = Vec::new();
        for action in actions {
            let resource = match action {
                NativeFrameAction::TextureReadback { texture, .. } => {
                    Some(texture.resource.clone())
                }
                NativeFrameAction::Present if capture_presented => back_buffer.cloned(),
                _ => None,
            };
            let Some(resource) = resource else {
                continue;
            };
            // SAFETY: `resource` retains the `ID3D12Resource` object and vtable storage for the duration of `GetDesc`.
            let desc = unsafe { resource.GetDesc() };
            let mut footprint =
                windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut rows = 0;
            let mut row_size = 0;
            let mut total = 0;
            // SAFETY: `self` retains the `ID3D12Device` during `GetCopyableFootprints`; `desc` is initialized, and the distinct output locals provide writable storage through the call.
            unsafe {
                self.device.GetCopyableFootprints(
                    &raw const desc,
                    0,
                    1,
                    0,
                    Some(&raw mut footprint),
                    Some(&raw mut rows),
                    Some(&raw mut row_size),
                    Some(&raw mut total),
                );
            }
            let Ok(request) = AllocationRequest::new(total, 256, MemoryClass::Readback, true, None)
            else {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                return Err(HalError::InvalidArgument);
            };
            let Ok(allocation) = self.allocate(request) else {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                return Err(HalError::NativeFailure);
            };
            let dimensions = match action {
                NativeFrameAction::TextureReadback { width, height, .. } => (*width, *height),
                NativeFrameAction::Present => extent,
                _ => unreachable!(),
            };
            readbacks.push((dimensions.0, dimensions.1, footprint, total, allocation));
        }
        let mut indirect_copies: Vec<Option<NativeAllocation>> =
            (0..actions.len()).map(|_| None).collect();
        for (action_index, action) in actions.iter().enumerate() {
            let NativeFrameAction::Graphics(draw) = action else {
                continue;
            };
            let indirect_raw = windows::core::Interface::as_raw(&draw.indirect_buffer.resource);
            if !draw.bindings.iter().any(|binding| {
                windows::core::Interface::as_raw(&binding.allocation.resource) == indirect_raw
            }) {
                continue;
            }
            let Ok(request) = AllocationRequest::new(
                indirect_command_bytes(draw.draw_count),
                16,
                MemoryClass::Device,
                false,
                None,
            ) else {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                for allocation in indirect_copies.into_iter().flatten() {
                    let _ = self.free(allocation);
                }
                return Err(HalError::InvalidArgument);
            };
            indirect_copies[action_index] = if let Ok(allocation) = self.allocate(request) {
                Some(allocation)
            } else {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                for allocation in indirect_copies.into_iter().flatten() {
                    let _ = self.free(allocation);
                }
                return Err(HalError::NativeFailure);
            };
        }
        Ok((readbacks, indirect_copies))
    }
}

impl NativeContext {
    fn prepare_frame(
        &mut self,
        surface: &mut Option<&mut NativeSurface>,
        extent: (u32, u32),
        actions: &[NativeFrameAction<'_>],
        uses_surface: bool,
    ) -> Result<DxPreparedFrame, HalError> {
        // Queue the worker's DIRECT release/acquire before a frame waits on its completion.
        // Waiting first would stall the same queue that must execute the handoff signal.
        for queue in [QueueKind::Transfer, QueueKind::TextureTransfer] {
            if let Some(required) = actions
                .iter()
                .filter_map(|action| match action {
                    NativeFrameAction::Wait(token) if token.queue == queue => Some(token.value),
                    _ => None,
                })
                .max()
            {
                let worker = if queue == QueueKind::Transfer {
                    self.transfer_worker.as_ref()
                } else {
                    self.texture_worker.as_ref()
                };
                worker
                    .ok_or(HalError::NotReady)?
                    .flush_through(required)
                    .map_err(|_| HalError::NativeFailure)?;
            }
        }
        if uses_surface {
            self.ensure_swapchain(
                surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                extent.0,
                extent.1,
            )?;
        }
        if actions.iter().any(|action| {
            matches!(
                action,
                NativeFrameAction::BeginPass { pass, .. } if pass.depth.is_some()
            )
        }) {
            self.ensure_surface_depth(surface.as_deref_mut().ok_or(HalError::InvalidArgument)?)?;
        }

        let surface_resources = surface
            .as_ref()
            .map(|surface| {
                let swapchain = surface
                    .swapchain
                    .as_ref()
                    .ok_or(HalError::NotReady)?
                    .clone();
                // SAFETY: the cloned `IDXGISwapChain4` retains its COM object and vtable storage for `GetCurrentBackBufferIndex`.
                let image_index = unsafe { swapchain.GetCurrentBackBufferIndex() } as usize;
                let back_buffer = surface
                    .buffers
                    .get(image_index)
                    .ok_or(HalError::NativeFailure)?
                    .clone();
                let heap = surface.rtv_heap.as_ref().ok_or(HalError::NativeFailure)?;
                // SAFETY: `self` retains the `ID3D12Device` object and vtable storage during `GetDescriptorHandleIncrementSize`.
                let stride = unsafe {
                    self.device
                        .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
                } as usize;
                // SAFETY: `heap` is borrowed from `surface.rtv_heap`, so the `ID3D12DescriptorHeap` object and its CPU descriptor storage remain allocated during `GetCPUDescriptorHandleForHeapStart`.
                let mut rtv = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
                rtv.ptr += image_index * stride;
                let dsv = surface.depth.as_ref().map(|depth| {
                    // SAFETY: `depth` retains its descriptor heap, whose first CPU handle remains valid while the surface depth target is alive.
                    unsafe { depth.heap.GetCPUDescriptorHandleForHeapStart() }
                });
                Ok((swapchain, back_buffer, rtv, dsv))
            })
            .transpose()?;
        let (swapchain, back_buffer, rtv, dsv) = match surface_resources {
            Some((swapchain, back_buffer, rtv, dsv)) => {
                (Some(swapchain), Some(back_buffer), Some(rtv), dsv)
            }
            None => (None, None, None, None),
        };

        let slot_index = self.frame_cursor;
        self.frame_cursor = (self.frame_cursor + 1) % FRAMES_IN_FLIGHT;
        let (allocator, list, fence_value, garbage) = {
            let slot = self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NotReady)?;
            (
                slot.allocator.clone(),
                slot.list.clone(),
                slot.fence_value,
                core::mem::take(&mut slot.garbage),
            )
        };
        // SAFETY: `self.fence` is retained by the context while its completed value is queried.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if fence_value != 0 && completed < fence_value {
            // SAFETY: `self` retains the `ID3D12Fence`, and `self.fence_event` is the event handle created for this context and kept open through `SetEventOnCompletion`.
            unsafe {
                self.fence
                    .SetEventOnCompletion(fence_value, self.fence_event)
            }
            .map_err(map_windows)?;
            // SAFETY: `self.fence_event` is the context's open event handle and remains open for the entire `WaitForSingleObject` call.
            unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
        }
        self.reclaim_deferred()
            .map_err(|_| HalError::NativeFailure)?;
        for allocation in garbage {
            self.free(allocation).map_err(|_| HalError::NativeFailure)?;
        }
        // SAFETY: the fence wait above has completed this slot's prior commands, so `allocator` is no longer in GPU use; `list` and `allocator` remain retained through both resets.
        unsafe {
            allocator.Reset().map_err(map_windows)?;
            list.Reset(&allocator, None::<&ID3D12PipelineState>)
                .map_err(map_windows)?;
        }
        Ok(DxPreparedFrame {
            swapchain,
            back_buffer,
            rtv,
            dsv,
            slot_index,
            list,
        })
    }
}

struct DxFrameEncoder<'a> {
    list: &'a super::ID3D12GraphicsCommandList,
    descriptor_heap: &'a super::ID3D12DescriptorHeap,
    sampler_heap: &'a super::ID3D12DescriptorHeap,
    surface: Option<&'a NativeSurface>,
    back_buffer: Option<&'a super::ID3D12Resource>,
    rtv: Option<windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
    dsv: Option<windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
    extent: (u32, u32),
    readbacks: &'a [DxReadback],
    indirect_copies: &'a [Option<NativeAllocation>],
    pass_active: bool,
    pass_target: Option<super::ID3D12Resource>,
    discard_store: bool,
    discard_depth: bool,
    readback_index: usize,
}

impl DxFrameEncoder<'_> {
    fn encode_barrier(
        &mut self,
        barrier: &super::ExecutionBarrier,
        resource: &super::NativeFrameResource<'_>,
    ) -> Result<(), HalError> {
        let native = match resource {
            NativeFrameResource::Buffer(allocation) => allocation.resource.clone(),
            NativeFrameResource::Texture(texture) | NativeFrameResource::RenderTarget(texture) => {
                texture.resource.clone()
            }
            NativeFrameResource::Surface => {
                self.back_buffer.cloned().ok_or(HalError::InvalidArgument)?
            }
            NativeFrameResource::Depth => self
                .surface
                .as_ref()
                .and_then(|surface| surface.depth.as_ref())
                .ok_or(HalError::NotReady)?
                .resource
                .clone(),
        };
        let before = barrier.before.map_or(D3D12_RESOURCE_STATE_COMMON, |state| {
            dx12_resource_state(state.access)
        });
        let after = dx12_resource_state(barrier.after.access);
        // SAFETY: each barrier owns an `ID3D12Resource` clone, retaining its COM object until `ResourceBarrier` copies the barrier array during the call.
        unsafe {
            if before == after && after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
                self.list.ResourceBarrier(&[uav_barrier(native)]);
            } else if before != after {
                self.list
                    .ResourceBarrier(&[transition_barrier(native, before, after)]);
            }
        }

        Ok(())
    }
    fn begin_pass(
        &mut self,
        pass: &&super::ExecutionPass,
        colors: &[super::PassAttachment<'_>],
    ) -> Result<(), HalError> {
        if self.pass_active || pass.colors.len() != 1 || colors.len() != 1 || pass.samples != 1 {
            return Err(HalError::InvalidArgument);
        }
        let attachment = colors.first().ok_or(HalError::InvalidArgument)?;
        // Textures, buffers, and depth images are never color attachments.
        let (rtv, target, target_extent) = match attachment.resource {
            super::NativeFrameResource::Surface => (
                self.rtv.ok_or(HalError::InvalidArgument)?,
                self.back_buffer.cloned().ok_or(HalError::InvalidArgument)?,
                self.extent,
            ),
            super::NativeFrameResource::RenderTarget(texture) => {
                if pass.depth.is_some() {
                    return Err(HalError::InvalidArgument);
                }
                let (_, rtv) = texture.rtv.as_ref().ok_or(HalError::InvalidArgument)?;
                (
                    *rtv,
                    texture.resource.clone(),
                    (texture.width, texture.height),
                )
            }
            _ => return Err(HalError::InvalidArgument),
        };
        if pass.area[0]
            .checked_add(pass.area[2])
            .is_none_or(|end| end > target_extent.0)
            || pass.area[1]
                .checked_add(pass.area[3])
                .is_none_or(|end| end > target_extent.1)
        {
            return Err(HalError::InvalidArgument);
        }
        // SAFETY: `rtv` and the optional `dsv` are descriptor handles from retained
        // heaps, and their pointer storage remains readable through `OMSetRenderTargets`.
        unsafe {
            self.list.OMSetRenderTargets(
                1,
                Some(&raw const rtv),
                false,
                pass.depth.and(self.dsv).as_ref().map(std::ptr::from_ref),
            );
            match pass.load {
                AttachmentLoadOp::Load => {}
                AttachmentLoadOp::Clear => {
                    self.list.ClearRenderTargetView(rtv, &attachment.clear, None);
                    if let Some(dsv) = pass.depth.and(self.dsv) {
                        self.list
                            .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                    }
                }
                AttachmentLoadOp::Discard => {
                    self.list.DiscardResource(&target, None);
                    if pass.depth.is_some()
                        && let Some(depth) = self
                            .surface
                            .as_ref()
                            .and_then(|surface| surface.depth.as_ref())
                    {
                        self.list.DiscardResource(&depth.resource, None);
                    }
                }
            }
        }
        self.pass_target = Some(target);
        self.discard_store = pass.store == AttachmentStoreOp::Discard;
        self.discard_depth = pass.depth.is_some();
        self.pass_active = true;

        Ok(())
    }
    fn compute(&mut self, dispatch: &super::NativeComputeDispatch<'_>) -> Result<(), HalError> {
        if self.pass_active {
            return Err(HalError::InvalidArgument);
        }
        let pipeline = dispatch.pipeline;
        // SAFETY: the encoder and `pipeline` references retain every command-list, pipeline, root-signature, heap, binding-resource, and push-constant pointer used by these recording calls until each call returns.
        unsafe {
            self.list.SetPipelineState(&pipeline.state);
            self.list.SetComputeRootSignature(&pipeline.root);
            let table = u32::try_from(pipeline.buffer_writable.len())
                .ok()
                .and_then(|count| count.checked_add(1))
                .ok_or(HalError::InvalidArgument)?;
            self.list.SetComputeRootDescriptorTable(
                table,
                self.descriptor_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            self.list.SetComputeRootDescriptorTable(
                table + 1,
                self.sampler_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            bind_dx12_compute_buffers(self.list, pipeline, dispatch.bindings)?;
            if !dispatch.push_constants.is_empty() {
                self.list.SetComputeRoot32BitConstants(
                    0,
                    u32::try_from(dispatch.push_constants.len())
                        .map_err(|_| HalError::InvalidArgument)?
                        / 4,
                    dispatch.push_constants.as_ptr().cast(),
                    0,
                );
            }
            self.list
                .Dispatch(dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
        }

        Ok(())
    }
    fn copy_indirect_before_pass(
        &mut self,
        action_index: usize,
        draw: &super::NativeDrawIndexed<'_>,
    ) -> Result<(), HalError> {
        validate_indirect_copy_phase(self.pass_active)?;
        let Some(copy) = self.indirect_copies[action_index].as_ref() else {
            return Ok(());
        };
        let indirect_raw = windows::core::Interface::as_raw(&draw.indirect_buffer.resource);
        let binding = draw
            .bindings
            .iter()
            .find(|binding| {
                windows::core::Interface::as_raw(&binding.allocation.resource) == indirect_raw
            })
            .ok_or(HalError::InvalidArgument)?;
        let shader_state = if binding.writable {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        };
        // SAFETY: no render pass is active; the draw and frame allocations retain both resources, and each barrier array remains readable until its recording call returns.
        unsafe {
            self.list.ResourceBarrier(&[
                transition_barrier(
                    draw.indirect_buffer.resource.clone(),
                    shader_state,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
                transition_barrier(
                    copy.resource.clone(),
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            self.list.CopyBufferRegion(
                &copy.resource,
                0,
                &draw.indirect_buffer.resource,
                0,
                indirect_command_bytes(draw.draw_count),
            );
            self.list.ResourceBarrier(&[
                transition_barrier(
                    draw.indirect_buffer.resource.clone(),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    shader_state,
                ),
                transition_barrier(
                    copy.resource.clone(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ),
            ]);
        }
        Ok(())
    }
    fn graphics(
        &mut self,
        action_index: usize,
        draw: &super::NativeDrawIndexed<'_>,
    ) -> Result<(), HalError> {
        if !self.pass_active {
            return Err(HalError::InvalidArgument);
        }
        let pipeline = draw.pipeline;
        let topology = pipeline.topology.ok_or(HalError::InvalidArgument)?;
        let signature = pipeline
            .signature
            .as_ref()
            .ok_or(HalError::InvalidArgument)?;
        let index_size = u32::try_from(draw.index_size).map_err(|_| HalError::InvalidArgument)?;
        let index_view = D3D12_INDEX_BUFFER_VIEW {
            // SAFETY: `draw.index_buffer.resource` retains the `ID3D12Resource` object and vtable storage during `GetGPUVirtualAddress`.
            BufferLocation: unsafe { draw.index_buffer.resource.GetGPUVirtualAddress() },
            SizeInBytes: index_size,
            Format: DXGI_FORMAT_R32_UINT,
        };
        let indirect_resource = self.indirect_copies[action_index]
            .as_ref()
            .map_or(&draw.indirect_buffer.resource, |copy| &copy.resource);
        // SAFETY: the encoder, `pipeline`, and `draw` references retain every command-list, state object, heap, resource, index-view, and push-constant pointer consumed by these recording calls until each call returns.
        unsafe {
            self.list.SetPipelineState(&pipeline.state);
            self.list.SetGraphicsRootSignature(&pipeline.root);
            let table = u32::try_from(pipeline.buffer_writable.len())
                .ok()
                .and_then(|count| count.checked_add(1))
                .ok_or(HalError::InvalidArgument)?;
            self.list.SetGraphicsRootDescriptorTable(
                table,
                self.descriptor_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            self.list.SetGraphicsRootDescriptorTable(
                table + 1,
                self.sampler_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            bind_dx12_graphics_buffers(self.list, pipeline, draw.bindings)?;
            if !draw.push_constants.is_empty() {
                self.list.SetGraphicsRoot32BitConstants(
                    0,
                    u32::try_from(draw.push_constants.len())
                        .map_err(|_| HalError::InvalidArgument)?
                        / 4,
                    draw.push_constants.as_ptr().cast(),
                    0,
                );
            }
            self.list.IASetPrimitiveTopology(topology);
            self.list.IASetIndexBuffer(Some(&raw const index_view));
            self.list
                .ExecuteIndirect(signature, draw.draw_count, indirect_resource, 0, None, 0);
        }

        Ok(())
    }
    fn texture_readback(&mut self, texture: &super::NativeTexture) -> Result<(), HalError> {
        if self.pass_active {
            return Err(HalError::InvalidArgument);
        }
        let (_, _, footprint, _, readback) = self
            .readbacks
            .get(self.readback_index)
            .ok_or(HalError::InvalidArgument)?;
        // SAFETY: `self.list`, `texture.resource`, and `readback.resource` are retained for the call, and `footprint` describes the readback allocation selected for this action.
        unsafe {
            copy_texture_to_readback(self.list, &texture.resource, &readback.resource, *footprint);
        }
        self.readback_index += 1;

        Ok(())
    }
    fn record(
        &mut self,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
        viewport: &D3D12_VIEWPORT,
        scissor: &RECT,
    ) -> Result<(), HalError> {
        // SAFETY: the encoder retains the command list and both descriptor heaps, while `viewport` and `scissor` provide initialized slice storage through the recording calls.
        unsafe {
            self.list.SetDescriptorHeaps(&[
                Some(self.descriptor_heap.clone()),
                Some(self.sampler_heap.clone()),
            ]);
            self.list.RSSetViewports(core::slice::from_ref(viewport));
            self.list.RSSetScissorRects(core::slice::from_ref(scissor));
        }
        for (action_index, action) in actions.iter().enumerate() {
            match action {
                NativeFrameAction::Wait(_) => {}
                NativeFrameAction::Barrier { barrier, resource } => {
                    self.encode_barrier(barrier, resource)?;
                }
                NativeFrameAction::BeginPass { pass, colors } => {
                    for (draw_index, candidate) in actions.iter().enumerate().skip(action_index + 1)
                    {
                        match candidate {
                            NativeFrameAction::Graphics(draw) => {
                                self.copy_indirect_before_pass(draw_index, draw)?;
                            }
                            NativeFrameAction::EndPass => break,
                            _ => {}
                        }
                    }
                    self.begin_pass(pass, colors)?;
                }
                NativeFrameAction::Compute(dispatch) => self.compute(dispatch)?,
                NativeFrameAction::Graphics(draw) => self.graphics(action_index, draw)?,
                NativeFrameAction::TextureReadback { texture, .. } => {
                    self.texture_readback(texture)?;
                }
                NativeFrameAction::EndPass => {
                    if !self.pass_active {
                        return Err(HalError::InvalidArgument);
                    }
                    if self.discard_store {
                        // SAFETY: the encoder retains the command list, pass target, and optional depth resource through each `DiscardResource` call; no region pointer is supplied.
                        unsafe {
                            self.list.DiscardResource(
                                self.pass_target.as_ref().ok_or(HalError::InvalidArgument)?,
                                None,
                            );
                            if self.discard_depth
                                && let Some(depth) = self
                                    .surface
                                    .as_ref()
                                    .and_then(|surface| surface.depth.as_ref())
                            {
                                self.list.DiscardResource(&depth.resource, None);
                            }
                        }
                    }
                    self.pass_target = None;
                    self.pass_active = false;
                    self.discard_store = false;
                    self.discard_depth = false;
                }
                NativeFrameAction::Present => {
                    if self.pass_active {
                        return Err(HalError::InvalidArgument);
                    }
                    if capture_presented {
                        let (_, _, footprint, _, readback) = self
                            .readbacks
                            .get(self.readback_index)
                            .ok_or(HalError::InvalidArgument)?;
                        // SAFETY: the encoder retains the command list, back buffer, and readback resource, and each temporary barrier array plus `footprint` remains readable through its recording call.
                        unsafe {
                            let back_buffer = self.back_buffer.ok_or(HalError::InvalidArgument)?;
                            self.list.ResourceBarrier(&[transition_barrier(
                                back_buffer.clone(),
                                D3D12_RESOURCE_STATE_PRESENT,
                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                            )]);
                            copy_texture_to_readback(
                                self.list,
                                back_buffer,
                                &readback.resource,
                                *footprint,
                            );
                            self.list.ResourceBarrier(&[transition_barrier(
                                back_buffer.clone(),
                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                                D3D12_RESOURCE_STATE_PRESENT,
                            )]);
                        }
                        self.readback_index += 1;
                    }
                }
            }
        }
        if self.pass_active {
            return Err(HalError::InvalidArgument);
        }
        // SAFETY: `self.list` retains the command-list COM object and vtable storage for `Close`, and `pass_active == false` ensures no render pass remains open.
        unsafe { self.list.Close().map_err(map_windows) }
    }
}

impl NativeContext {
    /// Records one immutable graph plan into one command list and presents after every action.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame plan or extent is invalid, required surface or frame resources are unavailable, resource allocation or readback fails, or a Direct3D/DXGI operation fails.
    pub fn execute_frame(
        &mut self,
        surface: Option<(&mut NativeSurface, (u32, u32))>,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        if actions.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        let (mut surface, extent) = match surface {
            Some((surface, extent)) if extent.0 != 0 && extent.1 != 0 => (Some(surface), extent),
            Some(_) => return Err(HalError::InvalidArgument),
            None => (None, (0, 0)),
        };
        let DxFramePlan {
            uses_surface,
            presents,
            external_waits,
        } = validate_frame_plan(actions, extent, surface.is_some(), capture_presented)?;
        let DxPreparedFrame {
            swapchain,
            back_buffer,
            rtv,
            dsv,
            slot_index,
            list,
        } = self.prepare_frame(&mut surface, extent, actions, uses_surface)?;
        let (readbacks, indirect_copies) = self.allocate_frame_resources(
            actions,
            capture_presented,
            extent,
            back_buffer.as_ref(),
        )?;

        let descriptor_heap = self.descriptors.clone();
        let sampler_heap = self.samplers.clone();
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: f32::from(u16::try_from(extent.0).map_err(|_| HalError::InvalidArgument)?),
            Height: f32::from(u16::try_from(extent.1).map_err(|_| HalError::InvalidArgument)?),
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: i32::try_from(extent.0).map_err(|_| HalError::InvalidArgument)?,
            bottom: i32::try_from(extent.1).map_err(|_| HalError::InvalidArgument)?,
        };
        let mut encoder = DxFrameEncoder {
            list: &list,
            descriptor_heap: &descriptor_heap,
            sampler_heap: &sampler_heap,
            surface: surface.as_deref(),
            back_buffer: back_buffer.as_ref(),
            rtv,
            dsv,
            extent,
            readbacks: &readbacks,
            indirect_copies: &indirect_copies,
            pass_active: false,
            pass_target: None,
            discard_store: false,
            discard_depth: false,
            readback_index: 0,
        };
        let record_result = encoder.record(actions, capture_presented, &viewport, &scissor);
        if let Err(error) = record_result {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            for allocation in indirect_copies.into_iter().flatten() {
                let _ = self.free(allocation);
            }
            return Err(error);
        }
        let command: ID3D12CommandList = match list.cast() {
            Ok(command) => command,
            Err(error) => {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                for allocation in indirect_copies.into_iter().flatten() {
                    let _ = self.free(allocation);
                }
                return Err(map_windows(error));
            }
        };
        for token in external_waits {
            let fence = match token.queue {
                QueueKind::Transfer => &self.transfer_fence,
                QueueKind::TextureTransfer => &self.texture_fence,
                _ => unreachable!("frame plan rejected unsupported wait queue"),
            };
            // SAFETY: the selected completion fence shares this device and signals `token.value`.
            if let Err(error) = unsafe { self.queue.Wait(fence, token.value) } {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                for allocation in indirect_copies.into_iter().flatten() {
                    let _ = self.free(allocation);
                }
                return Err(map_windows(error));
            }
        }
        let mut frame_garbage = indirect_copies.into_iter().flatten().collect::<Vec<_>>();
        // SAFETY: `command` retains the closed command-list interface through `ExecuteCommandLists`, and the one-element interface array remains readable for the call.
        unsafe { self.queue.ExecuteCommandLists(&[Some(command)]) };
        let value = self.next_fence;
        self.next_fence = value.checked_add(1).ok_or(HalError::NativeFailure)?;
        // SAFETY: `self` retains the command queue and fence through `Signal`, and `value` is the newly reserved monotonically increasing fence value.
        unsafe { self.queue.Signal(&self.fence, value) }.map_err(map_windows)?;
        self.frame_slots
            .get_mut(slot_index)
            .ok_or(HalError::NativeFailure)?
            .fence_value = value;
        let present_error = if presents {
            // SAFETY: `swapchain` retains the `IDXGISwapChain4` object and vtable storage through `Present`, after the recorded back-buffer transition to `PRESENT`.
            unsafe {
                swapchain
                    .as_ref()
                    .ok_or(HalError::InvalidArgument)?
                    .Present(1, DXGI_PRESENT(0))
                    .ok()
                    .err()
            }
        } else {
            None
        };
        if present_error.is_some() || !readbacks.is_empty() {
            // SAFETY: `self.fence` is retained by the context while its completed value is queried.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if completed < value {
                // SAFETY: `self` retains the fence, and `self.fence_event` is the context's event handle kept open through `SetEventOnCompletion`.
                unsafe { self.fence.SetEventOnCompletion(value, self.fence_event) }
                    .map_err(map_windows)?;
                // SAFETY: `self.fence_event` is the context's open event handle and remains open for the entire `WaitForSingleObject` call.
                unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
            }
            for allocation in frame_garbage.drain(..) {
                self.free(allocation).map_err(|_| HalError::NativeFailure)?;
            }
        } else {
            self.frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NativeFailure)?
                .garbage
                .append(&mut frame_garbage);
        }
        if let Some(error) = present_error {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(map_windows(error));
        }

        self.collect_frame_readbacks(readbacks, capture_presented, surface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_written_indirect_copy_uses_logical_extent_before_render_pass() {
        assert_eq!(indirect_command_bytes(1), 20);
        assert_eq!(indirect_command_bytes(3), 60);
        assert_eq!(validate_indirect_copy_phase(false), Ok(()));
        assert_eq!(
            validate_indirect_copy_phase(true),
            Err(HalError::InvalidArgument)
        );
    }
}
