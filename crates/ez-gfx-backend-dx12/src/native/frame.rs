use super::{
    AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, CompletionToken,
    D3D12_CLEAR_FLAG_DEPTH, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_INDEX_BUFFER_VIEW,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_VIEWPORT, DXGI_FORMAT_R32_UINT, FRAMES_IN_FLIGHT,
    HalError, ID3D12CommandList, ID3D12PipelineState, INFINITE, Interface, MemoryAllocator,
    MemoryClass, NativeAllocation, NativeContext, NativeFrameAction, NativeFrameActionSource,
    NativeFrameResource, NativeSurface, PresentationMode, QueueKind, RECT, WaitForSingleObject,
    bind_dx12_compute_buffers, bind_dx12_graphics_buffers, copy_texture_to_readback,
    dx12_resource_state, map_windows, presentation_parameters, record_resource_barriers,
    transition_barrier, uav_barrier,
};
use arrayvec::ArrayVec;
use ez_gfx_hal::COUNTER_BUFFER_ELEMENT_OFFSET;

type FrameSurface<'a> = (&'a mut NativeSurface, (u32, u32), PresentationMode);
type ResolvedFrameSurface<'a> = (Option<&'a mut NativeSurface>, (u32, u32), PresentationMode);

fn validate_surface_request(
    surface: Option<FrameSurface<'_>>,
) -> Result<ResolvedFrameSurface<'_>, HalError> {
    match surface {
        Some((surface, extent, mode)) if extent.0 != 0 && extent.1 != 0 => {
            if !surface.presentation_modes().contains(mode) {
                return Err(HalError::Unsupported);
            }
            Ok((Some(surface), extent, mode))
        }
        Some(_) => Err(HalError::InvalidArgument),
        None => Ok((None, (0, 0), PresentationMode::Fifo)),
    }
}
const DRAW_INDEXED_ARGUMENT_BYTES: u64 = core::mem::size_of::<
    windows::Win32::Graphics::Direct3D12::D3D12_DRAW_INDEXED_ARGUMENTS,
>() as u64;

struct DxFramePlan {
    uses_surface: bool,
    presents: bool,
    external_waits: ArrayVec<CompletionToken, 2>,
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

#[allow(
    clippy::too_many_lines,
    reason = "one validation pass keeps cross-action frame invariants local"
)]
fn bindings_match_pipeline(
    bindings: &dyn super::NativeBufferBindingSource,
    writable: &[bool],
) -> Result<bool, HalError> {
    if bindings.len() != writable.len() {
        return Ok(false);
    }
    let mut valid = true;
    bindings.visit(&mut |index, binding| {
        valid &= writable.get(index).is_some_and(|expected| {
            binding.writable == *expected
                && binding.offset < binding.allocation.allocation.size()
        });
        Ok(())
    })?;
    Ok(valid)
}

fn indirect_binding_writable(
    bindings: &dyn super::NativeBufferBindingSource,
    indirect: &super::ID3D12Resource,
) -> Result<Option<bool>, HalError> {
    let indirect = windows::core::Interface::as_raw(indirect);
    let mut found = None;
    bindings.visit(&mut |_, binding| {
        if windows::core::Interface::as_raw(&binding.allocation.resource) == indirect {
            found = Some(binding.writable);
        }
        Ok(())
    })?;
    Ok(found)
}

fn validate_frame_plan(
    actions: &(impl NativeFrameActionSource + ?Sized),
    extent: (u32, u32),
    surface_available: bool,
    capture_presented: bool,
) -> Result<DxFramePlan, HalError> {
    let mut present_count = 0_usize;
    let mut uses_surface = false;
    let mut external_waits = ArrayVec::<CompletionToken, 2>::new();
    let mut pass_active = false;
    let mut saw_present = false;
    actions.visit(&mut |_, action| {
        if saw_present {
            return Err(HalError::InvalidArgument);
        }
        match action {
            NativeFrameAction::Wait(token) => {
                if !matches!(
                    token.queue,
                    QueueKind::Transfer | QueueKind::TextureTransfer
                ) {
                    return Err(HalError::InvalidArgument);
                }
                if let Some(existing) = external_waits
                    .iter_mut()
                    .find(|existing| existing.queue == token.queue)
                {
                    if token.value > existing.value {
                        *existing = *token;
                    }
                } else {
                    external_waits
                        .try_push(*token)
                        .map_err(|_| HalError::InvalidArgument)?;
                }
            }
            NativeFrameAction::Barrier { resource, .. } => {
                uses_surface |= matches!(
                    resource,
                    NativeFrameResource::Surface | NativeFrameResource::Depth
                );
            }
            NativeFrameAction::BeginPass { pass, colors } => {
                uses_surface |= colors
                    .iter()
                    .any(|attachment| matches!(attachment.resource, NativeFrameResource::Surface));
                let mut target_extent = None;
                let mut valid = !pass_active
                    && pass.colors.len() == 1
                    && colors.len() == 1
                    && matches!(pass.samples, 1 | 2 | 4 | 8);
                if let Some(attachment) = colors.first() {
                    target_extent = match attachment.resource {
                        NativeFrameResource::Surface => {
                            valid &= pass.samples == 1;
                            Some(extent)
                        }
                        NativeFrameResource::RenderTarget(texture) => {
                            valid &= pass.depth.is_none();
                            valid &= texture.msaa.as_ref().map_or(1, |msaa| msaa.samples)
                                == pass.samples;
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
                    || !bindings_match_pipeline(
                        dispatch.bindings,
                        &dispatch.pipeline.buffer_writable,
                    )?
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::Graphics(draw) => {
                let indirect_size = indirect_command_bytes(draw.draw_count)
                    .checked_add(COUNTER_BUFFER_ELEMENT_OFFSET)
                    .ok_or(HalError::InvalidArgument)?;
                if !pass_active
                    || draw.draw_count == 0
                    || !bindings_match_pipeline(
                        draw.bindings,
                        &draw.pipeline.buffer_writable,
                    )?
                    || draw.pipeline.topology.is_none()
                    || draw.pipeline.signature.is_none()
                    || draw.index_size == 0
                    || draw.index_size > u64::from(u32::MAX)
                    || draw.indirect_size < indirect_size
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
                present_count += 1;
                saw_present = true;
                uses_surface = true;
            }
        }
        Ok(())
    })?;
    let presents = present_count == 1;
    if pass_active
        || present_count > 1
        || uses_surface && !surface_available
        || (uses_surface || capture_presented) && !presents
    {
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
    garbage: Vec<NativeAllocation>,
}

type DxReadback = (
    u32,
    u32,
    windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    u64,
    NativeAllocation,
);

/// Indirect-copy staging for one aliased graphics action, in ascending action order.
///
/// Only draws whose indirect buffer is also shader-bound need a staging copy;
/// every other action needs no entry, so this replaces the action-sized optional
/// list without changing encoder semantics.
pub(super) struct IndirectCopyEntry {
    action: usize,
    allocation: NativeAllocation,
}

type DxFrameResources = (Vec<DxReadback>, Vec<IndirectCopyEntry>);

/// Returns the staging copy for one graphics action, if the action is aliased.
///
/// Entries are pushed in ascending action order, so binary search is valid.
fn indirect_copy(entries: &[IndirectCopyEntry], action: usize) -> Option<&NativeAllocation> {
    entries
        .binary_search_by_key(&action, |entry| entry.action)
        .ok()
        .map(|index| &entries[index].allocation)
}

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
    /// Returns an emptied indirect-copy shell to its frame slot, keeping capacity.
    ///
    /// Callers drain or free every entry allocation first; the shell itself owns
    /// no GPU work. A missing slot is unreachable after preparation, so falling
    /// back to a drop preserves behavior and only loses retained capacity.
    fn restore_indirect_scratch(&mut self, slot_index: usize, entries: Vec<IndirectCopyEntry>) {
        if let Some(slot) = self.frame_slots.get_mut(slot_index) {
            slot.indirect_scratch = entries;
        }
    }

    fn allocate_frame_resources(
        &mut self,
        slot_index: usize,
        actions: &(impl NativeFrameActionSource + ?Sized),
        capture_presented: bool,
        extent: (u32, u32),
        back_buffer: Option<&super::ID3D12Resource>,
    ) -> Result<DxFrameResources, HalError> {
        let mut readbacks = Vec::new();
        let readback_result = actions.visit(&mut |_, action| {
            let resource = match action {
                NativeFrameAction::TextureReadback { texture, .. } => {
                    Some(texture.resource.clone())
                }
                NativeFrameAction::Present if capture_presented => back_buffer.cloned(),
                _ => None,
            };
            let Some(resource) = resource else {
                return Ok(());
            };
            // SAFETY: the cloned resource remains live throughout this query.
            let desc = unsafe { resource.GetDesc() };
            let mut footprint =
                windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut rows = 0;
            let mut row_size = 0;
            let mut total = 0;
            // SAFETY: distinct output locals remain writable for the native call.
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
            let request = AllocationRequest::new(total, 256, MemoryClass::Readback, true, None)
                .map_err(|_| HalError::InvalidArgument)?;
            let allocation = self.allocate(request).map_err(|_| HalError::NativeFailure)?;
            let dimensions = match action {
                NativeFrameAction::TextureReadback { width, height, .. } => (*width, *height),
                NativeFrameAction::Present => extent,
                _ => unreachable!(),
            };
            readbacks.push((dimensions.0, dimensions.1, footprint, total, allocation));
            Ok(())
        });
        if let Err(error) = readback_result {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            return Err(error);
        }
        // The slot is retired: `prepare_frame` waited its fence before returning
        // it, and this frame has not submitted yet. Only the emptied shell is
        // retained; entry allocations join frame garbage on success.
        let mut indirect_copies = core::mem::take(
            &mut self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NotReady)?
                .indirect_scratch,
        );
        indirect_copies.clear();
        let indirect_result = actions.visit(&mut |action_index, action| {
            let NativeFrameAction::Graphics(draw) = action else {
                return Ok(());
            };
            if indirect_binding_writable(draw.bindings, &draw.indirect_buffer.resource)?.is_none() {
                return Ok(());
            }
            let request = AllocationRequest::new(
                indirect_command_bytes(draw.draw_count) + COUNTER_BUFFER_ELEMENT_OFFSET,
                16,
                MemoryClass::Device,
                false,
                None,
            )
            .map_err(|_| HalError::InvalidArgument)?;
            let allocation = self.allocate(request).map_err(|_| HalError::NativeFailure)?;
            indirect_copies.push(IndirectCopyEntry {
                action: action_index,
                allocation,
            });
            Ok(())
        });
        if let Err(error) = indirect_result {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            for entry in indirect_copies.drain(..) {
                let _ = self.free(entry.allocation);
            }
            self.restore_indirect_scratch(slot_index, indirect_copies);
            return Err(error);
        }
        Ok((readbacks, indirect_copies))
    }
}

impl NativeContext {
    fn prepare_frame(
        &mut self,
        surface: &mut Option<&mut NativeSurface>,
        extent: (u32, u32),
        actions: &(impl NativeFrameActionSource + ?Sized),
        uses_surface: bool,
    ) -> Result<DxPreparedFrame, HalError> {
        // Queue the worker's DIRECT release/acquire before a frame waits on its completion.
        // Waiting first would stall the same queue that must execute the handoff signal.
        for queue in [QueueKind::Transfer, QueueKind::TextureTransfer] {
            let mut required = None::<u64>;
            actions.visit(&mut |_, action| {
                if let NativeFrameAction::Wait(token) = action
                    && token.queue == queue
                {
                    required = Some(required.map_or(token.value, |value| value.max(token.value)));
                }
                Ok(())
            })?;
            if let Some(required) = required {
                let worker = if queue == QueueKind::Transfer {
                    self.transfer_worker.as_ref()
                } else {
                    self.texture_worker.as_ref()
                };
                worker
                    .ok_or(HalError::NotReady)?
                    .flush_through(required)
                    .map_err(ez_gfx_hal::TransferWorkerError::to_hal_error)?;
            }
        }
        if uses_surface {
            self.ensure_swapchain(
                surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                extent.0,
                extent.1,
            )?;
        }
        let mut requires_depth = false;
        actions.visit(&mut |_, action| {
            requires_depth |= matches!(
                action,
                NativeFrameAction::BeginPass { pass, .. } if pass.depth.is_some()
            );
            Ok(())
        })?;
        if requires_depth {
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
        let (allocator, list, fence_value, mut garbage) = {
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
        for allocation in garbage.drain(..) {
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
            garbage,
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
    indirect_copies: &'a [IndirectCopyEntry],
    pass_active: bool,
    pass_target: Option<super::ID3D12Resource>,
    /// Pending multisampled resolve `(render, destination, format)` recorded at
    /// begin of pass; applied at end of pass unless the store is discarded.
    pass_resolve: Option<(
        super::ID3D12Resource,
        super::ID3D12Resource,
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    )>,
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
        // A multisampled target transitions its render storage alongside the
        // sampled resolve image so both stay in the compiler-derived states;
        // the MSAA image is never sampled.
        let mut natives = [None, None];
        let native_count = match resource {
            NativeFrameResource::Buffer(allocation) => {
                natives[0] = Some(allocation.resource.clone());
                1
            }
            NativeFrameResource::Texture(texture) | NativeFrameResource::RenderTarget(texture) => {
                natives[0] = Some(texture.resource.clone());
                if let Some(msaa) = texture.msaa.as_ref() {
                    natives[1] = Some(msaa.resource.clone());
                    2
                } else {
                    1
                }
            }
            NativeFrameResource::Surface => {
                natives[0] = Some(self.back_buffer.cloned().ok_or(HalError::InvalidArgument)?);
                1
            }
            NativeFrameResource::Depth => {
                natives[0] = Some(
                    self.surface
                        .as_ref()
                        .and_then(|surface| surface.depth.as_ref())
                        .ok_or(HalError::NotReady)?
                        .resource
                        .clone(),
                );
                1
            }
        };
        let before = barrier.before.map_or(D3D12_RESOURCE_STATE_COMMON, |state| {
            dx12_resource_state(state.access)
        });
        let after = dx12_resource_state(barrier.after.access);
        if before == after && after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            for native in natives.iter().take(native_count).flatten() {
                record_resource_barriers(self.list, [uav_barrier(native.clone())]);
            }
        } else if before != after {
            for native in natives.iter().take(native_count).flatten() {
                record_resource_barriers(
                    self.list,
                    [transition_barrier(native.clone(), before, after)],
                );
            }
        }

        Ok(())
    }
    fn begin_pass(
        &mut self,
        pass: &&super::ExecutionPass,
        colors: &[super::PassAttachment<'_>],
    ) -> Result<(), HalError> {
        if self.pass_active
            || pass.colors.len() != 1
            || colors.len() != 1
            || !matches!(pass.samples, 1 | 2 | 4 | 8)
        {
            return Err(HalError::InvalidArgument);
        }
        let attachment = colors.first().ok_or(HalError::InvalidArgument)?;
        // Textures, buffers, and depth images are never color attachments. A
        // multisampled target renders into its MSAA storage; the resolve into
        // the sampled resource is stashed for end of pass.
        let (rtv, target, target_extent, resolve) = match attachment.resource {
            super::NativeFrameResource::Surface => {
                if pass.samples != 1 {
                    return Err(HalError::InvalidArgument);
                }
                (
                    self.rtv.ok_or(HalError::InvalidArgument)?,
                    self.back_buffer.cloned().ok_or(HalError::InvalidArgument)?,
                    self.extent,
                    None,
                )
            }
            super::NativeFrameResource::RenderTarget(texture) => {
                if pass.depth.is_some() {
                    return Err(HalError::InvalidArgument);
                }
                if let Some(msaa) = texture.msaa.as_ref() {
                    if msaa.samples != pass.samples {
                        return Err(HalError::InvalidArgument);
                    }
                    (
                        msaa.rtv,
                        msaa.resource.clone(),
                        (texture.width, texture.height),
                        Some((msaa.resource.clone(), texture.resource.clone(), msaa.format)),
                    )
                } else {
                    if pass.samples != 1 {
                        return Err(HalError::InvalidArgument);
                    }
                    let (_, rtv) = texture.rtv.as_ref().ok_or(HalError::InvalidArgument)?;
                    (
                        *rtv,
                        texture.resource.clone(),
                        (texture.width, texture.height),
                        None,
                    )
                }
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
                    self.list
                        .ClearRenderTargetView(rtv, &attachment.clear, None);
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
        self.pass_resolve = resolve;
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
                .map_err(|_| HalError::InvalidArgument)?;
            self.list.SetComputeRootDescriptorTable(
                table,
                self.descriptor_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            self.list.SetComputeRootDescriptorTable(
                table + 1,
                self.sampler_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            bind_dx12_compute_buffers(self.list, pipeline, dispatch.bindings)?;
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
        let Some(copy) = indirect_copy(self.indirect_copies, action_index) else {
            return Ok(());
        };
        let writable = indirect_binding_writable(draw.bindings, &draw.indirect_buffer.resource)?
            .ok_or(HalError::InvalidArgument)?;
        let shader_state = if writable {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        };
        // SAFETY: no render pass is active; the draw and frame allocations retain both resources, and each barrier array remains readable until its recording call returns.
        unsafe {
            record_resource_barriers(
                self.list,
                [
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
                ],
            );
            self.list.CopyBufferRegion(
                &copy.resource,
                0,
                &draw.indirect_buffer.resource,
                0,
                indirect_command_bytes(draw.draw_count) + COUNTER_BUFFER_ELEMENT_OFFSET,
            );
            record_resource_barriers(
                self.list,
                [
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
                ],
            );
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
        let indirect_resource = indirect_copy(self.indirect_copies, action_index)
            .map_or(&draw.indirect_buffer.resource, |copy| &copy.resource);
        // SAFETY: the encoder, `pipeline`, and `draw` references retain every command-list, state object, heap, resource, index-view, and binding pointer consumed by these recording calls until each call returns.
        unsafe {
            self.list.SetPipelineState(&pipeline.state);
            self.list.SetGraphicsRootSignature(&pipeline.root);
            let table = u32::try_from(pipeline.buffer_writable.len())
                .map_err(|_| HalError::InvalidArgument)?;
            self.list.SetGraphicsRootDescriptorTable(
                table,
                self.descriptor_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            self.list.SetGraphicsRootDescriptorTable(
                table + 1,
                self.sampler_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            bind_dx12_graphics_buffers(self.list, pipeline, draw.bindings)?;
            self.list.IASetPrimitiveTopology(topology);
            self.list.IASetIndexBuffer(Some(&raw const index_view));
            self.list.ExecuteIndirect(
                signature,
                draw.draw_count,
                indirect_resource,
                COUNTER_BUFFER_ELEMENT_OFFSET,
                Some(indirect_resource),
                0,
            );
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
}

#[path = "frame_encode.rs"]
mod encode;

impl NativeContext {
    /// Records one immutable graph plan into one command list and presents after every action.
    ///
    /// # Errors
    ///
    /// Returns an error if validation, allocation, encoding, or submission fails.
    pub fn execute_frame(
        &mut self,
        surface: Option<FrameSurface<'_>>,
        actions: &[NativeFrameAction<'_>],
        capture_presented: bool,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        self.execute_frame_source(surface, &actions, capture_presented)
    }

    /// Records a synchronous source without retaining borrowed native views.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::execute_frame`].
    #[allow(
        clippy::too_many_lines,
        reason = "one transaction keeps frame resource cleanup and submission ordering local"
    )]
    pub fn execute_frame_source(
        &mut self,
        surface: Option<FrameSurface<'_>>,
        actions: &impl NativeFrameActionSource,
        capture_presented: bool,
    ) -> Result<Vec<Vec<u8>>, HalError> {
        if actions.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        let (mut surface, extent, presentation_mode) = validate_surface_request(surface)?;
        let DxFramePlan {
            uses_surface,
            presents,
            external_waits,
        } = validate_frame_plan(actions, extent, surface.is_some(), capture_presented)?;
        let (present_interval, present_flags) =
            presentation_parameters(presentation_mode).ok_or(HalError::Unsupported)?;
        let DxPreparedFrame {
            swapchain,
            back_buffer,
            rtv,
            dsv,
            slot_index,
            list,
            garbage: mut frame_garbage,
        } = self.prepare_frame(&mut surface, extent, actions, uses_surface)?;
        let (readbacks, mut indirect_copies) = self.allocate_frame_resources(
            slot_index,
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
            pass_resolve: None,
            discard_store: false,
            discard_depth: false,
            readback_index: 0,
        };
        let record_result = encoder.record(actions, capture_presented, &viewport, &scissor);
        if let Err(error) = record_result {
            for (_, _, _, _, allocation) in readbacks {
                let _ = self.free(allocation);
            }
            for entry in indirect_copies.drain(..) {
                let _ = self.free(entry.allocation);
            }
            self.restore_indirect_scratch(slot_index, indirect_copies);
            return Err(error);
        }
        let command: ID3D12CommandList = match list.cast() {
            Ok(command) => command,
            Err(error) => {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                for entry in indirect_copies.drain(..) {
                    let _ = self.free(entry.allocation);
                }
                self.restore_indirect_scratch(slot_index, indirect_copies);
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
                for entry in indirect_copies.drain(..) {
                    let _ = self.free(entry.allocation);
                }
                self.restore_indirect_scratch(slot_index, indirect_copies);
                return Err(map_windows(error));
            }
        }
        frame_garbage.extend(indirect_copies.drain(..).map(|entry| entry.allocation));
        // SAFETY: `command` retains the closed command-list interface through `ExecuteCommandLists`, and the one-element interface array remains readable for the call.
        unsafe { self.queue.ExecuteCommandLists(&[Some(command)]) };
        let value = self.next_fence;
        self.next_fence = value.checked_add(1).ok_or(HalError::NativeFailure)?;
        // SAFETY: `self` retains the command queue and fence through `Signal`, and `value` is the newly reserved monotonically increasing fence value.
        unsafe { self.queue.Signal(&self.fence, value) }.map_err(map_windows)?;
        {
            let slot = self
                .frame_slots
                .get_mut(slot_index)
                .ok_or(HalError::NativeFailure)?;
            slot.fence_value = value;
            // Entries drained into garbage above; the emptied shell returns to the
            // retired slot for the next frame.
            slot.indirect_scratch = indirect_copies;
        }
        let present_error = if presents {
            // SAFETY: `swapchain` retains the `IDXGISwapChain4` object and vtable storage through `Present`, after the recorded back-buffer transition to `PRESENT`.
            unsafe {
                swapchain
                    .as_ref()
                    .ok_or(HalError::InvalidArgument)?
                    .Present(present_interval, present_flags)
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
#[path = "frame_tests.rs"]
mod plan_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_written_indirect_copy_uses_logical_extent_before_render_pass() {
        assert_eq!(indirect_command_bytes(1), 20);
        assert_eq!(indirect_command_bytes(3), 60);
        assert_eq!(
            indirect_command_bytes(3) + COUNTER_BUFFER_ELEMENT_OFFSET,
            316
        );
        assert_eq!(validate_indirect_copy_phase(false), Ok(()));
        assert_eq!(
            validate_indirect_copy_phase(true),
            Err(HalError::InvalidArgument)
        );
    }
}
