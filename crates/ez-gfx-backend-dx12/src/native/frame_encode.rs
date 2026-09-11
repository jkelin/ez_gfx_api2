//! Direct3D 12 frame action recording into one graphics command list.

use super::{
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT, D3D12_VIEWPORT, DxFrameEncoder,
    HalError, NativeFrameAction, NativeFrameActionSource, RECT, copy_texture_to_readback,
    map_windows, record_resource_barriers, transition_barrier,
};

impl DxFrameEncoder<'_> {
    pub(super) fn record(
        &mut self,
        actions: &(impl NativeFrameActionSource + ?Sized),
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
        actions.visit(&mut |action_index, action| {
            match action {
                NativeFrameAction::Wait(_) => {}
                NativeFrameAction::Barrier { barrier, resource } => {
                    self.encode_barrier(barrier, resource)?;
                }
                NativeFrameAction::BeginPass { pass, colors } => {
                    let mut ended = false;
                    actions.visit(&mut |draw_index, candidate| {
                        if draw_index <= action_index || ended {
                            return Ok(());
                        }
                        match candidate {
                            NativeFrameAction::Graphics(draw) => {
                                self.copy_indirect_before_pass(draw_index, draw)?;
                            }
                            NativeFrameAction::EndPass => ended = true,
                            _ => {}
                        }
                        Ok(())
                    })?;
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
                    } else if let Some((render, destination, format)) = self.pass_resolve.take() {
                        // Resolve the multisampled render into the sampled
                        // image, then return both to render-target state so the
                        // compiler-derived barriers keep matching.
                        // SAFETY: the encoder retains the command list and both
                        // resources; each barrier array and the resolve call
                        // complete before their temporaries drop.
                        unsafe {
                            use windows::Win32::Graphics::Direct3D12::{
                                D3D12_RESOURCE_STATE_RENDER_TARGET,
                                D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                            };
                            record_resource_barriers(
                                self.list,
                                [transition_barrier(
                                    render.clone(),
                                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                                )],
                            );
                            record_resource_barriers(
                                self.list,
                                [transition_barrier(
                                    destination.clone(),
                                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                )],
                            );
                            self.list
                                .ResolveSubresource(&destination, 0, &render, 0, format);
                            record_resource_barriers(
                                self.list,
                                [transition_barrier(
                                    render,
                                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                                )],
                            );
                            record_resource_barriers(
                                self.list,
                                [transition_barrier(
                                    destination,
                                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                                )],
                            );
                        }
                    }
                    self.pass_target = None;
                    self.pass_resolve = None;
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
                            record_resource_barriers(
                                self.list,
                                [transition_barrier(
                                    back_buffer.clone(),
                                    D3D12_RESOURCE_STATE_PRESENT,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                                )],
                            );
                            copy_texture_to_readback(
                                self.list,
                                back_buffer,
                                &readback.resource,
                                *footprint,
                            );
                            record_resource_barriers(
                                self.list,
                                [transition_barrier(
                                    back_buffer.clone(),
                                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                                    D3D12_RESOURCE_STATE_PRESENT,
                                )],
                            );
                        }
                        self.readback_index += 1;
                    }
                }
            }
            Ok(())
        })?;
        if self.pass_active {
            return Err(HalError::InvalidArgument);
        }
        // SAFETY: `self.list` retains the command-list COM object and vtable storage for `Close`, and `pass_active == false` ensures no render pass remains open.
        unsafe { self.list.Close().map_err(map_windows) }
    }
}
