use super::{
    AdapterCapabilities, AdapterClass, AdapterInfo, AllocationError, AllocationSizes, Allocator,
    AllocatorCreateDesc, BACKEND, CompletionToken, CompressionSupport,
    DEFAULT_ALLOCATION_BLOCK_POLICY, DeferredNativeResource, DeferredResource, FRAMES_IN_FLIGHT,
    FrameSlot, FrameSlotTracker, HalError, MTLArgumentBuffersTier, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice,
    MemoryAllocator, NativeContext, NativeSurface, ProtocolObject, QueueKind, Retained,
    SemanticProfile, TEXTURE_DESCRIPTOR_CAPACITY, complete_deferred_slot, map_allocation_hal,
    map_allocator, map_allocator_hal,
};

/// Architecture-guaranteed render-target roles per format.
///
/// Metal exposes no runtime renderability query beyond family checks, so these
/// roles follow the API guarantee rather than a device probe: RGBA8 and BGRA
/// sRGB render and sample on every Metal device, RGBA16F renders and samples
/// on Apple GPUs, and D32 float attaches depth everywhere. A device-specific
fn target_format_support(max_color_samples: u8) -> Vec<ez_gfx_runtime::target::FormatSupport> {
    use ez_gfx_core::capability::CompressionSupport;
    use ez_gfx_runtime::target::{Format, FormatSupport};
    let color = |format, storage| {
        FormatSupport::new(
            format,
            true,
            true,
            storage,
            max_color_samples,
            CompressionSupport::NONE,
        )
        .expect("probed sample counts are always valid")
    };
    let single = |format, color_role, sampled, storage| {
        FormatSupport::new(
            format,
            color_role,
            sampled,
            storage,
            1,
            CompressionSupport::NONE,
        )
        .expect("single-sample support is always valid")
    };
    vec![
        color(Format::Rgba8Unorm, true),
        // Every surface graph resource is BGRA8 sRGB and the native
        // constructor lowers it, so the probe must advertise the family.
        color(Format::Bgra8Srgb, true),
        color(Format::Rgba16Float, true),
        single(Format::Depth32Float, false, true, false),
    ]
}

/// Describes one Metal device for identity and capabilities without creating queues.
///
/// The stable identity derives from the registry ID and maximum buffer length,
/// matching context creation so explicit selection resolves to the same bytes.
///
/// # Errors
///
/// Returns an error if adapter metadata construction fails.
fn describe_device(device: &ProtocolObject<dyn MTLDevice>) -> Result<AdapterInfo, HalError> {
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
    let registry = device.registryID();
    let mut stable_id = [0_u8; 16];
    stable_id[..8].copy_from_slice(&registry.to_le_bytes());
    stable_id[8..].copy_from_slice(&device.maxBufferLength().to_le_bytes());
    let name = device.name().to_string();
    AdapterInfo::new(
        BACKEND,
        stable_id,
        name,
        format!("registry-{registry:016x}"),
        AdapterClass::Integrated,
        capabilities,
    )
    .map_err(|_| HalError::Unsupported)
}

impl NativeContext {
    /// Creates a context on the highest-ranked hardware adapter satisfying the semantic profile.
    ///
    /// # Errors
    ///
    /// Returns an error if no default Metal device or command queue is available, the adapter does not satisfy the semantic profile, adapter metadata is invalid, or allocator creation fails.
    pub fn create_default() -> Result<Self, HalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(HalError::Unsupported)?;
        let adapter = describe_device(&device)?;
        SemanticProfile::V1
            .admit(adapter.capabilities())
            .map_err(|_| HalError::Unsupported)?;
        Self::initialize(device, adapter)
    }

    /// Enumerates every describable Metal device, admitted or not.
    ///
    /// Rejected adapters stay listed so rejection diagnostics can name them;
    /// an empty array is a valid empty enumeration, not a failure. Metal
    /// leaves the power/device decision to the caller, so enumeration never
    /// switches devices the way the system default can.
    ///
    /// # Errors
    ///
    /// This query cannot fail today; the `Result` keeps the shared dispatch
    /// uniform across backends.
    pub fn enumerate_adapters() -> Result<Vec<AdapterInfo>, HalError> {
        let mut adapters = Vec::new();
        for device in MTLCopyAllDevices().to_vec() {
            // One undescribable device skips itself, never the enumeration.
            if let Ok(adapter) = describe_device(&device) {
                adapters.push(adapter);
            }
        }
        Ok(adapters)
    }

    /// Creates a context for one explicitly selected adapter.
    ///
    /// Ranking is bypassed but admission never is: an unknown identity fails
    /// `InvalidArgument` and a matched but inadmissible adapter fails
    /// `Unsupported`. Metal exposes no software adapters (every enumerated
    /// device is hardware), so no software policy applies here.
    ///
    /// # Errors
    ///
    /// Returns an error if no enumerated adapter matches the requested
    /// identity, the adapter fails the semantic profile, adapter metadata is
    /// invalid, or native context initialization fails.
    pub fn create_for_adapter(stable_id: [u8; 16]) -> Result<Self, HalError> {
        for device in MTLCopyAllDevices().to_vec() {
            // One undescribable device skips itself, never the selection.
            let Ok(adapter) = describe_device(&device) else {
                continue;
            };
            if adapter.stable_id() != stable_id {
                continue;
            }
            // Explicit selection bypasses ranking but never bypasses admission.
            SemanticProfile::V1
                .admit(adapter.capabilities())
                .map_err(|_| HalError::Unsupported)?;
            return Self::initialize(device, adapter);
        }
        Err(HalError::InvalidArgument)
    }
    /// Initializes device-owned queues, allocator, and workers for an admitted adapter.
    ///
    /// Shared by default construction and explicit adapter selection so both
    /// entry points build the identical context.
    ///
    fn initialize(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        adapter: AdapterInfo,
    ) -> Result<Self, HalError> {
        let queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
        let transfer_queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
        let texture_queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
        let texture_graphics_event = device.newEvent().ok_or(HalError::NativeFailure)?;
        let texture_completion_event = device.newEvent().ok_or(HalError::NativeFailure)?;
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
                    submission_value: 0,
                })
                .collect(),
            frame_tracker: FrameSlotTracker::default(),
            next_frame_value: 1,
            last_frame_value: 0,
            completed_frame_value: 0,
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

    /// Queries render-target format support for the admitted adapter.
    ///
    /// Metal reports no per-format renderability query beyond family checks, so
    /// this returns the architecture-guaranteed table: RGBA8/BGRA-sRGB color
    /// render plus sampling on every device, RGBA16F color render plus sampling
    /// on Apple GPUs, and D32 float depth attachment everywhere. The color
    /// multisample ceiling follows the device-wide sample-count query and is
    /// eye-reviewed only until a Mac run confirms it.
    pub fn probe_target_formats(
        &self,
    ) -> Result<ez_gfx_runtime::target::FormatCapabilities, AllocationError> {
        // Device-wide support gates every color format equally; 8-sample
        // render targets stay out of scope on Metal hardware.
        let ceiling = if self.device.supportsTextureSampleCount(4) {
            4
        } else if self.device.supportsTextureSampleCount(2) {
            2
        } else {
            1
        };
        ez_gfx_runtime::target::FormatCapabilities::new(target_format_support(ceiling))
            .map_err(|_| AllocationError::NativeFailure)
    }

    /// Admits the selected adapter and initializes device-owned state.
    ///
    /// # Errors
    ///
    /// This implementation has no fallible surface-specific initialization.
    pub fn init_device(&self, _surface: &NativeSurface) -> Result<AdapterInfo, HalError> {
        Ok(self.adapter.clone())
    }

    /// Waits for submitted native work and reclaims completed deferred resources.
    ///
    /// # Errors
    ///
    /// Returns an error if submitted work fails or reclaiming a deferred allocation fails.
    pub fn wait_idle(&mut self) -> Result<(), HalError> {
        self.drain_complete = false;
        let transfer_flush = self.transfer_worker.as_ref().map(|worker| worker.flush());
        let transfer_failed = transfer_flush.as_ref().is_none_or(|result| result.is_err());
        let texture_flush = self.texture_worker.as_ref().map(|worker| worker.flush());
        let texture_failed = texture_flush.as_ref().is_none_or(|result| result.is_err());
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
            // A loss-poisoned worker names device loss instead of a generic failure.
            let lost = transfer_flush.is_some_and(|result| {
                matches!(result, Err(ez_gfx_hal::TransferWorkerError::DeviceLost))
            }) || texture_flush.is_some_and(|result| {
                matches!(result, Err(ez_gfx_hal::TransferWorkerError::DeviceLost))
            }) || self
                .transfer_worker
                .as_ref()
                .is_some_and(|worker| worker.device_lost())
                || self
                    .texture_worker
                    .as_ref()
                    .is_some_and(|worker| worker.device_lost());
            return Err(if lost {
                HalError::DeviceLost
            } else {
                HalError::NativeFailure
            });
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
        self.completed_frame_value = self.last_frame_value;
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
        self.completed_frame_value = self
            .completed_frame_value
            .max(self.frame_slots[slot].submission_value);

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
    /// Returns the most recently submitted graphics-frame token.
    pub fn last_frame_completion(&self) -> Option<CompletionToken> {
        CompletionToken::new(QueueKind::Graphics, self.last_frame_value).ok()
    }

    /// Polls command buffers and returns the completed graphics-frame prefix.
    pub fn completed_frame_value(&mut self) -> Result<u64, AllocationError> {
        self.poll_frame_completion();
        Ok(self.completed_frame_value)
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
                // The MSAA render storage retires with the sampled texture.
                if let Some(msaa) = texture.msaa {
                    drop(msaa.texture);
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(&msaa.allocation)
                        .map_err(map_allocator)?;
                }
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

#[cfg(test)]
mod target_tests {
    use super::target_format_support;
    use ez_gfx_runtime::target::{Format, FormatSupport};

    #[test]
    fn static_table_covers_probed_families_once() {
        let supports = target_format_support(4);
        let formats: Vec<Format> = supports.iter().map(|support| support.format).collect();
        assert_eq!(
            formats,
            [
                Format::Rgba8Unorm,
                Format::Bgra8Srgb,
                Format::Rgba16Float,
                Format::Depth32Float
            ]
        );
        assert!(
            FormatSupport::new(
                Format::Rgba8Unorm,
                true,
                true,
                true,
                4,
                ez_gfx_core::capability::CompressionSupport::NONE
            )
            .is_ok_and(|expected| supports.contains(&expected))
        );
        // Depth stays single-sample while color formats share the ceiling.
        assert!(
            FormatSupport::new(
                Format::Depth32Float,
                false,
                true,
                false,
                1,
                ez_gfx_core::capability::CompressionSupport::NONE
            )
            .is_ok_and(|expected| supports.contains(&expected))
        );
    }
}
