use super::{
    AdapterCapabilities, AdapterClass, AdapterInfo, AllocationError, AllocationSizes, Allocator,
    AllocatorCreateDesc, BACKEND, CompressionSupport, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DeferredNativeResource, DeferredResource, FRAMES_IN_FLIGHT, FrameSlot, FrameSlotTracker,
    HalError, MTLArgumentBuffersTier, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCreateSystemDefaultDevice, MTLDevice, MemoryAllocator, NativeContext, NativeSurface,
    QueueKind, SemanticProfile, TEXTURE_DESCRIPTOR_CAPACITY, complete_deferred_slot,
    map_allocation_hal, map_allocator, map_allocator_hal,
};

impl NativeContext {
    /// Creates a context on the highest-ranked hardware adapter satisfying the semantic profile.
    ///
    /// # Errors
    ///
    /// Returns an error if no default Metal device or command queue is available, the adapter does not satisfy the semantic profile, adapter metadata is invalid, or allocator creation fails.
    pub fn create_default() -> Result<Self, HalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(HalError::Unsupported)?;
        let queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
        let transfer_queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
        let texture_queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
        let texture_graphics_event = device.newEvent().ok_or(HalError::NativeFailure)?;
        let texture_completion_event = device.newEvent().ok_or(HalError::NativeFailure)?;
        let tier_two = device.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2;
        let sampler_capacity =
            u32::try_from(device.maxArgumentBufferSamplerCount()).unwrap_or(u32::MAX);
        // BC and ASTC are independent: Apple GPUs can support both families.
        let mut compression = CompressionSupport::NONE;
        if device.supportsBCTextureCompression() {
            compression = compression.union(CompressionSupport::BC);
        }
        if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple2) {
            compression = compression.union(CompressionSupport::ASTC);
        }
        let capabilities = AdapterCapabilities {
            bindless_sampled_textures: if tier_two {
                TEXTURE_DESCRIPTOR_CAPACITY.min(sampler_capacity)
            } else {
                0
            },
            bindless_storage_resources: if tier_two { 1024 } else { 0 },
            bindless_samplers: sampler_capacity,
            max_indirect_draw_count: u32::MAX,
            shader_model: 0x0605,
            timeline_synchronization: true,
            resource_aliasing: true,
            dynamic_rendering: true,
            presentation: true,
            compression,
        };
        SemanticProfile::V1
            .admit(&capabilities)
            .map_err(|_| HalError::Unsupported)?;
        let registry = device.registryID();
        let mut stable_id = [0_u8; 16];
        stable_id[..8].copy_from_slice(&registry.to_le_bytes());
        stable_id[8..].copy_from_slice(&device.maxBufferLength().to_le_bytes());
        let name = device.name().to_string();
        let adapter = AdapterInfo::new(
            BACKEND,
            stable_id,
            name,
            format!("registry-{registry:016x}"),
            AdapterClass::Integrated,
            capabilities,
        )
        .map_err(|_| HalError::Unsupported)?;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            device: device.clone(),
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            allocation_sizes: AllocationSizes::new(
                DEFAULT_ALLOCATION_BLOCK_POLICY.initial_device,
                DEFAULT_ALLOCATION_BLOCK_POLICY.initial_host,
            )
            .with_max_device_memblock_size(DEFAULT_ALLOCATION_BLOCK_POLICY.maximum_device)
            .with_max_host_memblock_size(DEFAULT_ALLOCATION_BLOCK_POLICY.maximum_host),
            create_residency_set: false,
        })
        .map_err(map_allocator_hal)?;
        let transfer_worker =
            super::transfer::start_worker().map_err(|_| HalError::NativeFailure)?;
        let texture_worker = super::transfer::start_texture_worker(
            texture_queue,
            texture_graphics_event.clone(),
            texture_completion_event.clone(),
        )
        .map_err(|_| HalError::NativeFailure)?;
        Ok(Self {
            device,
            queue,
            transfer_queue,
            texture_graphics_event,
            texture_completion_event,
            allocator: Some(allocator),
            retired: Vec::new(),
            frame_slots: (0..FRAMES_IN_FLIGHT)
                .map(|_| FrameSlot {
                    command: None,
                    argument_buffers: Vec::new(),
                })
                .collect(),
            frame_tracker: FrameSlotTracker::default(),
            deferred: Vec::new(),
            adapter,
            drain_complete: true,
            next_transfer_value: 1,
            #[cfg(test)]
            buffer_wait_observer: None,
            completed_transfer_value: 0,
            pending_transfers: Vec::new(),
            transfer_worker: Some(transfer_worker),
            next_texture_value: 1,
            completed_texture_value: 0,
            pending_texture_transfers: Vec::new(),
            texture_worker: Some(texture_worker),
            texture_staging: ez_gfx_hal::ReusableStagingPool::new(256),
        })
    }

    /// Returns the immutable identity and capabilities of the admitted adapter.
    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }

    /// Admits the selected adapter and initializes all device-owned queues and descriptor state.
    ///
    /// # Errors
    ///
    /// Returns an error if the surface has a null Metal layer.
    pub fn init_device(&self, surface: &NativeSurface) -> Result<AdapterInfo, HalError> {
        if surface.layer == 0 {
            return Err(HalError::InvalidArgument);
        }
        Ok(self.adapter.clone())
    }

    /// Waits for submitted native work and reclaims completed deferred resources.
    ///
    /// # Errors
    ///
    /// Returns an error if submitted work fails or reclaiming a deferred allocation fails.
    pub fn wait_idle(&mut self) -> Result<(), HalError> {
        self.drain_complete = false;
        let transfer_failed = self
            .transfer_worker
            .as_ref()
            .is_none_or(|worker| worker.flush().is_err());
        let texture_failed = self
            .texture_worker
            .as_ref()
            .is_none_or(|worker| worker.flush().is_err());
        // Failed flush can precede worker shutdown; join its actual-command drain before reclaiming.
        if transfer_failed && let Some(worker) = self.transfer_worker.as_mut() {
            worker.shutdown();
        }
        if texture_failed && let Some(worker) = self.texture_worker.as_mut() {
            worker.shutdown();
        }
        let mut failed = transfer_failed || texture_failed;
        for pending in &self.pending_transfers {
            // Rejected/failed submission may never commit this command.
            if matches!(
                pending.command.status(),
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled
            ) {
                pending.command.waitUntilCompleted();
            }
            failed |= pending.command.status() == MTLCommandBufferStatus::Error
                || pending.command.error().is_some();
        }
        // Successful texture flush already waited its committed batches, including graphics releases.
        for slot in &self.frame_slots {
            if let Some(command) = &slot.command {
                command.waitUntilCompleted();
                failed |= command.status() != MTLCommandBufferStatus::Completed
                    || command.error().is_some();
            }
        }
        self.drain_complete = (!transfer_failed
            || self
                .transfer_worker
                .as_ref()
                .is_some_and(ez_gfx_hal::TransferWorker::drained))
            && (!texture_failed
                || self
                    .texture_worker
                    .as_ref()
                    .is_some_and(ez_gfx_hal::TransferWorker::drained))
            && self.pending_transfers.iter().all(|pending| {
                !matches!(
                    pending.command.status(),
                    MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled
                )
            })
            && self.frame_slots.iter().all(|slot| {
                slot.command.as_ref().is_none_or(|command| {
                    matches!(
                        command.status(),
                        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                    )
                })
            });
        failed |= !self.drain_complete;
        if failed {
            // Drained failures permit destruction, not publication of texture readiness.
            return Err(HalError::NativeFailure);
        }
        self.completed_transfer_value = self.next_transfer_value.saturating_sub(1);
        self.reclaim(QueueKind::Transfer, self.completed_transfer_value)
            .map_err(map_allocation_hal)?;
        self.pending_transfers.clear();
        self.completed_texture_value = self.next_texture_value.saturating_sub(1);
        self.reclaim(QueueKind::TextureTransfer, self.completed_texture_value)
            .map_err(map_allocation_hal)?;
        self.pending_texture_transfers.clear();
        for slot in 0..self.frame_slots.len() {
            self.frame_slots[slot].command = None;
            self.frame_tracker.mark_completed(slot);
        }
        let deferred = self
            .deferred
            .drain(..)
            .map(|item| item.resource)
            .collect::<Vec<_>>();
        for resource in deferred {
            self.destroy_deferred_now(resource)
                .map_err(map_allocation_hal)?;
        }
        Ok(())
    }

    /// Whether the last idle drain proved that no submitted command still owns GPU work.
    pub fn is_drained(&self) -> bool {
        self.drain_complete
    }

    pub(super) fn complete_frame_slot(&mut self, slot: usize) -> Result<(), HalError> {
        let failed = if let Some(command) = self.frame_slots[slot].command.take() {
            command.waitUntilCompleted();
            command.status() != MTLCommandBufferStatus::Completed || command.error().is_some()
        } else {
            false
        };
        self.frame_tracker.mark_completed(slot);

        let mut index = 0;
        while index < self.deferred.len() {
            if complete_deferred_slot(&mut self.deferred[index].pending_slots, slot) {
                let resource = self.deferred.swap_remove(index).resource;
                self.destroy_deferred_now(resource)
                    .map_err(map_allocation_hal)?;
            } else {
                index += 1;
            }
        }
        if failed {
            return Err(HalError::NativeFailure);
        }
        Ok(())
    }

    /// Reaps frame slots whose commands already settled without blocking.
    ///
    /// Polling texture paths call this before consulting the descriptor gate so
    /// completed submissions unblock publication during sustained rendering.
    /// Settled failures still release the slot: the GPU owns no more work, so
    /// a dead frame must not pin texture readiness. The slot error itself
    /// surfaces through the normal frame-submission path, not here.
    pub fn poll_frame_completion(&mut self) {
        for slot in 0..self.frame_slots.len() {
            let settled = self.frame_slots[slot]
                .command
                .as_ref()
                .is_some_and(|command| {
                    matches!(
                        command.status(),
                        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                    )
                });
            if settled {
                // `waitUntilCompleted` returns immediately on a settled buffer.
                let _ = self.complete_frame_slot(slot);
            }
        }
    }

    pub(super) fn defer_resource(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        let pending_slots = self.frame_tracker.in_flight_mask();
        if pending_slots == 0 {
            self.destroy_deferred_now(resource)
        } else {
            self.deferred.push(DeferredNativeResource {
                pending_slots,
                resource,
            });
            Ok(())
        }
    }

    pub(super) fn destroy_deferred_now(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        match resource {
            DeferredResource::Allocation(allocation) => {
                drop(allocation.buffer);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&allocation.allocation)
                    .map_err(map_allocator)
            }
            DeferredResource::Texture(texture) => {
                drop(texture.texture);
                drop(texture.sampler);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&texture.allocation)
                    .map_err(map_allocator)
            }
            DeferredResource::Pipeline(pipeline) => {
                drop(pipeline);
                Ok(())
            }
            DeferredResource::Depth(depth) => {
                drop(depth);
                Ok(())
            }
            DeferredResource::Shader(shader) => {
                drop(shader.libraries);
                Ok(())
            }
            DeferredResource::Surface(surface) => {
                drop(surface);
                Ok(())
            }
        }
    }
}
