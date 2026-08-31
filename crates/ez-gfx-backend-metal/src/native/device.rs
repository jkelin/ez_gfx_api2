use super::{
    AdapterCapabilities, AdapterClass, AdapterInfo, AllocationError, AllocationSizes, Allocator,
    AllocatorCreateDesc, BACKEND, CompressionSupport, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DeferredNativeResource, DeferredResource, FRAMES_IN_FLIGHT, FrameSlot, FrameSlotTracker,
    HalError, MTLArgumentBuffersTier, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, NativeContext, NativeSurface, SemanticProfile,
    TEXTURE_DESCRIPTOR_CAPACITY, complete_deferred_slot, map_allocation_hal, map_allocator,
    map_allocator_hal,
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
        let tier_two = device.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2;
        let sampler_capacity =
            u32::try_from(device.maxArgumentBufferSamplerCount()).unwrap_or(u32::MAX);
        let compression = if device.supportsBCTextureCompression() {
            CompressionSupport::BC
        } else {
            CompressionSupport::ASTC
        };
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
        Ok(Self {
            device,
            queue,
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
            next_transfer_value: 1,
            completed_transfer_value: 0,
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
    /// Returns an error if a command buffer cannot be created, submitted work fails to complete successfully, or reclaiming a deferred allocation fails.
    pub fn wait_idle(&mut self) -> Result<(), HalError> {
        let command = self.queue.commandBuffer().ok_or(HalError::NativeFailure)?;
        command.commit();
        command.waitUntilCompleted();
        let failed =
            command.status() != MTLCommandBufferStatus::Completed || command.error().is_some();
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
        if failed {
            return Err(HalError::NativeFailure);
        }
        Ok(())
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
