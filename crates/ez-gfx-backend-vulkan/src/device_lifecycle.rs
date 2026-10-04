//! Vulkan frame completion and deferred resource reclamation.

use super::{
    AllocationError, DeferredNativeResource, DeferredResource, NativeContext, NativeSurface,
    map_allocation_vk, map_allocator, map_vk, vk,
};

impl NativeContext {
    pub(super) fn in_flight_mask(&self) -> u8 {
        self.frame_slots
            .iter()
            .enumerate()
            .fold(0_u8, |mask, (index, slot)| {
                if slot.in_flight {
                    mask | (1 << index)
                } else {
                    mask
                }
            })
    }
    /// Reaps frame slots whose fences already signaled without blocking.
    ///
    /// Polling texture paths call this before consulting the descriptor gate so
    /// completed submissions unblock publication during sustained rendering.
    /// Slot reuse already reclaims the same state on wrap-around; this only
    /// observes fence status and never waits.
    ///
    /// # Errors
    ///
    /// Returns an error when the device is missing or fence status cannot be queried.
    pub fn poll_frame_completion(&mut self) -> Result<(), AllocationError> {
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        for slot_index in 0..self.frame_slots.len() {
            if !self.frame_slots[slot_index].in_flight {
                continue;
            }
            // SAFETY: the fence belongs to this slot on the retained live device,
            // and a status query performs no wait and mutates no command state.
            let signaled = unsafe { device.get_fence_status(self.frame_slots[slot_index].fence) }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
            if signaled {
                self.frame_slots[slot_index].in_flight = false;
                self.complete_frame_slot(slot_index)?;
            }
        }
        Ok(())
    }

    /// Returns the most recently submitted graphics-frame token.
    pub fn last_frame_completion(&self) -> Option<ez_gfx_hal::CompletionToken> {
        ez_gfx_hal::CompletionToken::new(ez_gfx_hal::QueueKind::Graphics, self.last_frame_value)
            .ok()
    }

    /// Polls frame fences and returns the completed graphics-frame prefix.
    ///
    /// # Errors
    ///
    /// Returns an allocation error when a fence status cannot be queried.
    pub fn completed_frame_value(&mut self) -> Result<u64, AllocationError> {
        self.poll_frame_completion()?;
        Ok(self.completed_frame_value)
    }
    ///
    /// # Errors
    ///
    /// Returns an error if immediate destruction requires a missing device or allocator, or freeing the allocation fails.
    pub(super) fn defer_resource(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        let pending_slots = self.in_flight_mask();
        if pending_slots == 0 {
            return self.destroy_deferred_now(resource);
        }
        self.deferred.push(DeferredNativeResource {
            pending_slots,
            resource,
        });
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if `slot_index` cannot select a bit in the `u8` slot mask or destroying a newly unblocked resource fails.
    pub(super) fn complete_frame_slot(&mut self, slot_index: usize) -> Result<(), AllocationError> {
        let bit = 1_u8
            .checked_shl(u32::try_from(slot_index).map_err(|_| AllocationError::NativeFailure)?)
            .ok_or(AllocationError::NativeFailure)?;
        let submission_value = self
            .frame_slots
            .get(slot_index)
            .ok_or(AllocationError::NativeFailure)?
            .submission_value;
        self.completed_frame_value = self.completed_frame_value.max(submission_value);
        let mut ready = Vec::new();
        let mut index = 0;
        while index < self.deferred.len() {
            self.deferred[index].pending_slots &= !bit;
            if self.deferred[index].pending_slots == 0 {
                ready.push(self.deferred.swap_remove(index).resource);
            } else {
                index += 1;
            }
        }
        for resource in ready {
            self.destroy_deferred_now(resource)?;
        }
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if the device or a required allocator is missing, or freeing an allocation fails.
    pub(super) fn destroy_deferred_now(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        match resource {
            DeferredResource::Allocation(allocation) => {
                // SAFETY: deferred reclamation consumes `allocation.buffer` only after its pending frame-slot mask clears, before freeing its paired allocation storage.
                unsafe { device.destroy_buffer(allocation.buffer, None) };
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(allocation.allocation)
                    .map_err(|error| map_allocator(&error))
            }
            DeferredResource::Pipeline(pipeline) => {
                // SAFETY: the deferred `pipeline` resource is consumed once after its pending frame-slot mask clears, so its pipeline and layout handles have no remaining submitted uses.
                unsafe {
                    device.destroy_pipeline(pipeline.pipeline, None);
                    device.destroy_pipeline_layout(pipeline.layout, None);
                    device.destroy_descriptor_set_layout(pipeline.public_descriptor_layout, None);
                }
                Ok(())
            }
            DeferredResource::Shader(shader) => {
                for module in shader.modules {
                    // SAFETY: each `module` is consumed once from deferred shader storage, and Vulkan pipelines retain no `VkShaderModule` storage after pipeline creation.
                    unsafe { device.destroy_shader_module(module, None) };
                }
                Ok(())
            }
            DeferredResource::TextureView(view) => {
                // SAFETY: the replaced view is destroyed only after every frame slot that could
                // have observed its descriptor has completed.
                unsafe { device.destroy_image_view(view, None) };
                Ok(())
            }
            DeferredResource::Texture(texture) => {
                let texture = *texture;
                // The MSAA render storage is never sampled or described, so
                // only its view and image retire alongside the sampled image.
                let msaa = texture.msaa;
                let depth = texture.depth;
                // SAFETY: the deferred texture is consumed after its pending frame-slot mask clears; its view and sampler are destroyed before its image and allocation storage.
                unsafe {
                    device.destroy_image_view(texture.view, None);
                    device.destroy_sampler(texture.sampler, None);
                    device.destroy_image(texture.image, None);
                    if let Some(storage) = msaa.as_ref() {
                        device.destroy_image_view(storage.view, None);
                        device.destroy_image(storage.image, None);
                    }
                    if let Some(depth) = depth.as_ref() {
                        device.destroy_image_view(depth.view, None);
                        device.destroy_image(depth.image, None);
                    }
                }
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(texture.allocation)
                    .map_err(|error| map_allocator(&error))?;
                if let Some(storage) = msaa {
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(storage.allocation)
                        .map_err(|error| map_allocator(&error))?;
                }
                if let Some(depth) = depth {
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(depth.allocation)
                        .map_err(|error| map_allocator(&error))?;
                }
                Ok(())
            }
        }
    }

    /// Destroys backend-owned state associated with a borrowed host surface.
    ///
    /// Returns `false` only when native work could not drain and the surface was abandoned.
    pub fn destroy_surface(&mut self, surface: NativeSurface) -> bool {
        let _ = self.wait_idle();
        if !self.is_drained() {
            // Keep the surface and swapchain alive when submitted uses cannot retire.
            core::mem::forget(surface);
            return false;
        }
        let _ = self.destroy_depth_target();
        if let Some(device) = self.device.as_ref() {
            for view in self.swapchain_views.drain(..) {
                // SAFETY: native idle proved all uses complete; each view is destroyed once before its swapchain.
                unsafe { device.destroy_image_view(view, None) };
            }
        }
        if let (Some(loader), Some(swapchain)) =
            (self.swapchain_loader.as_ref(), self.swapchain.take())
        {
            // SAFETY: native idle proved all uses complete and the swapchain's views were destroyed first.
            unsafe { loader.destroy_swapchain(swapchain, None) };
        }
        self.swapchain_format = vk::Format::UNDEFINED;
        // Images die with their swapchain; dropping the cache here keeps stale
        // handles from surviving past destruction.
        self.swapchain_images.clear();
        self.swapchain_initialized.clear();
        // Images, views, and the swapchain are gone; the extent must die with
        // them or telemetry keeps reporting a resolution for a dead surface.
        self.swapchain_extent = vk::Extent2D::default();
        if surface.handle != vk::SurfaceKHR::null() {
            // SAFETY: the host window is still live and no swapchain references this surface.
            unsafe { self.surface_loader.destroy_surface(surface.handle, None) };
        }
        drop(surface);
        true
    }
}
