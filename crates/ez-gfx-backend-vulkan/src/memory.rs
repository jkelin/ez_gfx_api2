use super::{
    AllocationCreateDesc, AllocationError, AllocationRequest, AllocationScheme, BufferTransfer,
    CompletionToken, DeferredResource, FRAME_DESCRIPTOR_SET_CAPACITY, FRAMES_IN_FLIGHT, FrameSlot,
    HalError, MemoryAllocator, MemoryClass, MemoryLocation, NativeAllocation, NativeContext,
    QueueKind, ResourceAccess, ResourceState, RetiredAllocation, ShaderStage,
    transfer::{VulkanTransferCopy, VulkanTransferJob},
    vk,
};

impl MemoryAllocator for NativeContext {
    type Allocation = NativeAllocation;

    fn allocate(
        &mut self,
        request: AllocationRequest,
    ) -> Result<Self::Allocation, AllocationError> {
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let queue_families = [
            self.graphics_queue_family
                .ok_or(AllocationError::NativeFailure)?,
            self.transfer_queue_family
                .ok_or(AllocationError::NativeFailure)?,
        ];
        let create = vk::BufferCreateInfo::default().size(request.size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let create = if queue_families[0] == queue_families[1] {
            create.sharing_mode(vk::SharingMode::EXCLUSIVE)
        } else {
            create
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_families)
        };
        // SAFETY: the device is live and the descriptor contains no borrowed arrays.
        let buffer = unsafe { device.create_buffer(&create, None) }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
        // SAFETY: the buffer was created by this device and remains live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        if requirements.alignment < request.alignment {
            // SAFETY: destroy_buffer receives the unbound buffer created with allocator None, exactly once on this allocation-error path.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(AllocationError::InvalidAlignment);
        }
        let location = match request.memory_class {
            MemoryClass::Device | MemoryClass::Transient => MemoryLocation::GpuOnly,
            MemoryClass::Upload => MemoryLocation::CpuToGpu,
            MemoryClass::Readback => MemoryLocation::GpuToCpu,
        };
        let allocator = self
            .allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?;
        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name: "ez-gfx-buffer",
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: destroy_buffer receives the still-unbound buffer created with allocator None, exactly once after allocation failed.
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(map_allocator(&error));
            }
        };
        if request.mapped && allocation.mapped_ptr().is_none() {
            let _ = allocator.free(allocation);
            // SAFETY: destroy_buffer receives the unbound buffer created with allocator None, after its separate allocation was freed.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(AllocationError::NotHostVisible);
        }
        // SAFETY: the allocation is live, compatible with these queried requirements, and outlives the buffer.
        if let Err(error) =
            // SAFETY: bind_buffer_memory uses the DeviceMemory and offset allocated for this buffer's queried requirements, whose backing storage the allocator retains through the call.
            unsafe {
                device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            }
        {
            let _ = allocator.free(allocation);
            // SAFETY: bind_buffer_memory did not succeed, so destroy_buffer may destroy this buffer once using the allocator None chosen at creation.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(map_allocation_vk(map_vk(error)));
        }
        Ok(NativeAllocation { buffer, allocation })
    }

    fn mapped_slice<'a>(
        &self,
        allocation: &'a Self::Allocation,
    ) -> Result<&'a [u8], AllocationError> {
        allocation
            .allocation
            .mapped_slice()
            .ok_or(AllocationError::NotHostVisible)
    }

    fn mapped_slice_mut<'a>(
        &mut self,
        allocation: &'a mut Self::Allocation,
    ) -> Result<&'a mut [u8], AllocationError> {
        allocation
            .allocation
            .mapped_slice_mut()
            .ok_or(AllocationError::NotHostVisible)
    }

    fn flush(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError> {
        validate_allocation_range(allocation.allocation.size(), offset, size)?;
        if allocation
            .allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(());
        }
        if allocation.allocation.mapped_ptr().is_none() {
            return Err(AllocationError::NotHostVisible);
        }
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let range = vk::MappedMemoryRange::default()
            // SAFETY: Allocation::memory reads the DeviceMemory handle from allocator backing storage that remains allocated for allocation's lifetime.
            .memory(unsafe { allocation.allocation.memory() })
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: gpu-allocator keeps this DeviceMemory mapped, and 0..WHOLE_SIZE covers its mapped storage with noncoherent-atom-aligned bounds for flush_mapped_memory_ranges.
        unsafe { device.flush_mapped_memory_ranges(core::slice::from_ref(&range)) }
            .map_err(|error| map_allocation_vk(map_vk(error)))
    }

    fn invalidate(
        &mut self,
        allocation: &mut Self::Allocation,
        offset: u64,
        size: u64,
    ) -> Result<(), AllocationError> {
        validate_allocation_range(allocation.allocation.size(), offset, size)?;
        if allocation
            .allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(());
        }
        if allocation.allocation.mapped_ptr().is_none() {
            return Err(AllocationError::NotHostVisible);
        }
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let range = vk::MappedMemoryRange::default()
            // SAFETY: Allocation::memory reads the DeviceMemory handle from allocator backing storage that remains allocated for allocation's lifetime.
            .memory(unsafe { allocation.allocation.memory() })
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: gpu-allocator keeps this DeviceMemory mapped, and 0..WHOLE_SIZE covers its mapped storage with noncoherent-atom-aligned bounds for invalidate_mapped_memory_ranges.
        unsafe { device.invalidate_mapped_memory_ranges(core::slice::from_ref(&range)) }
            .map_err(|error| map_allocation_vk(map_vk(error)))
    }

    fn free(&mut self, allocation: Self::Allocation) -> Result<(), AllocationError> {
        self.defer_resource(DeferredResource::Allocation(allocation))
    }

    fn retire(
        &mut self,
        allocation: Self::Allocation,
        completion: CompletionToken,
    ) -> Result<(), AllocationError> {
        self.retired.push(RetiredAllocation {
            allocation,
            completion,
        });
        Ok(())
    }

    fn reclaim(&mut self, queue: QueueKind, completed: u64) -> Result<usize, AllocationError> {
        let mut reclaimed = 0;
        let mut index = 0;
        while index < self.retired.len() {
            if self.retired[index].completion.queue == queue
                && self.retired[index].completion.value <= completed
            {
                let retired = self.retired.swap_remove(index);
                self.free(retired.allocation)?;
                reclaimed += 1;
            } else {
                index += 1;
            }
        }
        Ok(reclaimed)
    }
}

impl BufferTransfer for NativeContext {
    fn copy_buffer(
        &mut self,
        source: &Self::Allocation,
        destination: &Self::Allocation,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    ) -> Result<CompletionToken, AllocationError> {
        validate_allocation_range(source.allocation.size(), source_offset, size)?;
        validate_allocation_range(destination.allocation.size(), destination_offset, size)?;
        let value = self.next_transfer_value;
        let next = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        let worker = self
            .transfer_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?;
        worker
            .submit(VulkanTransferJob {
                value,
                bytes: size,
                cancelled: None,
                copy: VulkanTransferCopy::Buffer {
                    source: source.buffer,
                    destination: destination.buffer,
                    region: vk::BufferCopy {
                        src_offset: source_offset,
                        dst_offset: destination_offset,
                        size,
                    },
                },
            })
            .map_err(ez_gfx_hal::TransferWorkerError::to_allocation_error)?;
        // Rejected work does not consume a completion value.
        self.next_transfer_value = next;
        CompletionToken::new(QueueKind::Transfer, value).map_err(|_| AllocationError::NativeFailure)
    }

    fn completed_transfer_value(&self) -> Result<u64, AllocationError> {
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        // SAFETY: transfer_timeline is a timeline semaphore created for this device, and its handle remains stored in self throughout get_semaphore_counter_value.
        unsafe {
            device.get_semaphore_counter_value(
                self.transfer_timeline
                    .ok_or(AllocationError::NativeFailure)?,
            )
        }
        .map_err(|error| map_allocation_vk(map_vk(error)))
    }
}
pub(super) fn vulkan_state(
    state: ResourceState,
) -> (vk::PipelineStageFlags, vk::AccessFlags, vk::ImageLayout) {
    let shader_stage = match state.stage {
        ShaderStage::None => vk::PipelineStageFlags::ALL_COMMANDS,
        ShaderStage::Vertex => vk::PipelineStageFlags::VERTEX_SHADER,
        ShaderStage::Fragment => vk::PipelineStageFlags::FRAGMENT_SHADER,
        ShaderStage::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
        ShaderStage::AllGraphics => vk::PipelineStageFlags::ALL_GRAPHICS,
    };
    let pipeline_stage = match state.access {
        ResourceAccess::IndexRead => vk::PipelineStageFlags::VERTEX_INPUT,
        ResourceAccess::IndirectRead => vk::PipelineStageFlags::DRAW_INDIRECT,
        ResourceAccess::IndirectStorageRead | ResourceAccess::IndirectStorageReadWrite => {
            vk::PipelineStageFlags::DRAW_INDIRECT | shader_stage
        }
        ResourceAccess::ColorAttachmentWrite => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ResourceAccess::DepthStencilRead | ResourceAccess::DepthStencilWrite => {
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
        }
        ResourceAccess::TransferRead | ResourceAccess::TransferWrite => {
            vk::PipelineStageFlags::TRANSFER
        }
        ResourceAccess::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        ResourceAccess::SampledRead
        | ResourceAccess::StorageRead
        | ResourceAccess::StorageWrite
        | ResourceAccess::StorageReadWrite => shader_stage,
    };
    let (access, layout) = match state.access {
        ResourceAccess::SampledRead => (
            vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        ResourceAccess::StorageRead => (vk::AccessFlags::SHADER_READ, vk::ImageLayout::GENERAL),
        ResourceAccess::StorageWrite => (vk::AccessFlags::SHADER_WRITE, vk::ImageLayout::GENERAL),
        ResourceAccess::StorageReadWrite => (
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        ResourceAccess::IndexRead => (vk::AccessFlags::INDEX_READ, vk::ImageLayout::UNDEFINED),
        ResourceAccess::IndirectRead => (
            vk::AccessFlags::INDIRECT_COMMAND_READ,
            vk::ImageLayout::UNDEFINED,
        ),
        ResourceAccess::IndirectStorageRead => (
            vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::UNDEFINED,
        ),
        ResourceAccess::IndirectStorageReadWrite => (
            vk::AccessFlags::INDIRECT_COMMAND_READ
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE,
            vk::ImageLayout::UNDEFINED,
        ),
        ResourceAccess::ColorAttachmentWrite => (
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        ResourceAccess::DepthStencilRead => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ),
        ResourceAccess::DepthStencilWrite => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
        ResourceAccess::TransferRead => (
            vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        ResourceAccess::TransferWrite => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        ResourceAccess::Present => (vk::AccessFlags::empty(), vk::ImageLayout::PRESENT_SRC_KHR),
    };
    (pipeline_stage, access, layout)
}
///
/// # Errors
///
/// Returns `AllocationError::NativeFailure` if the size is zero, the range overflows, or the range exceeds the allocation length.
pub(super) fn validate_allocation_range(
    length: u64,
    offset: u64,
    size: u64,
) -> Result<(), AllocationError> {
    if size == 0 || offset.checked_add(size).is_none_or(|end| end > length) {
        return Err(AllocationError::NativeFailure);
    }
    Ok(())
}

pub(super) fn map_allocation_hal(error: AllocationError) -> HalError {
    match error {
        AllocationError::DeviceLost => HalError::DeviceLost,
        AllocationError::OutOfMemory => HalError::OutOfMemory,
        _ => HalError::NativeFailure,
    }
}

pub(super) fn map_allocation_vk(error: HalError) -> AllocationError {
    match error {
        HalError::DeviceLost => AllocationError::DeviceLost,
        HalError::OutOfMemory => AllocationError::OutOfMemory,
        _ => AllocationError::NativeFailure,
    }
}

pub(super) fn map_allocator_hal(error: &gpu_allocator::AllocationError) -> HalError {
    match error {
        gpu_allocator::AllocationError::OutOfMemory => HalError::OutOfMemory,
        _ => HalError::NativeFailure,
    }
}

pub(super) fn map_allocator(error: &gpu_allocator::AllocationError) -> AllocationError {
    match error {
        gpu_allocator::AllocationError::OutOfMemory => AllocationError::OutOfMemory,
        _ => AllocationError::NativeFailure,
    }
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        if let Some(mut worker) = self.transfer_worker.take() {
            worker.shutdown();
        }
        if let Some(mut worker) = self.texture_worker.take() {
            worker.shutdown();
        }
        if !self.is_drained() {
            // Quarantine this failed context's owners: a live queue may still use
            // their storage. Raw Vulkan handles are intentionally not destroyed.
            core::mem::forget((
                self.entry_loader.clone(),
                self.allocator.take(),
                core::mem::take(&mut self.retired),
                core::mem::take(&mut self.deferred),
                core::mem::take(&mut self.frame_slots),
                self.depth_target.take(),
                core::mem::replace(
                    &mut self.texture_staging,
                    ez_gfx_hal::ReusableStagingPool::new(256),
                ),
            ));
            return;
        }
        for slot in &mut self.frame_slots {
            slot.in_flight = false;
        }
        let deferred = self
            .deferred
            .drain(..)
            .map(|item| item.resource)
            .collect::<Vec<_>>();
        for resource in deferred {
            let _ = self.destroy_deferred_now(resource);
        }
        let _ = self.destroy_depth_target();
        for allocation in self.texture_staging.drain() {
            let _ = self.free(allocation);
        }
        while let Some(retired) = self.retired.pop() {
            let _ = self.free(retired.allocation);
        }
        drop(self.allocator.take());
        if let Some(device) = self.device.take() {
            for slot in self.frame_slots.drain(..) {
                destroy_frame_slot(&device, &slot);
            }
            for view in self.swapchain_views.drain(..) {
                // SAFETY: device_wait_idle was issued before teardown, and each drained image view is passed once to destroy_image_view with the allocator None used at creation.
                unsafe { device.destroy_image_view(view, None) };
            }
            for semaphore in self.swapchain_finished.drain(..) {
                // SAFETY: device idle completed presentation waits before swapchain teardown.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            if let (Some(loader), Some(swapchain)) =
                (self.swapchain_loader.as_ref(), self.swapchain.take())
            {
                // SAFETY: device_wait_idle was issued and all swapchain image views were destroyed before destroy_swapchain consumes the taken swapchain handle.
                unsafe { loader.destroy_swapchain(swapchain, None) };
            }
            if let Some(semaphore) = self.image_available.take() {
                // SAFETY: device_wait_idle was issued before the taken image_available semaphore is passed once to destroy_semaphore with its creation allocator None.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            if let Some(pool) = self.texture_descriptor_pool.take() {
                // SAFETY: device_wait_idle was issued before the taken texture descriptor pool is passed once to destroy_descriptor_pool with its creation allocator None.
                unsafe { device.destroy_descriptor_pool(pool, None) };
            }
            if let Some(layout) = self.texture_descriptor_layout.take() {
                // SAFETY: the texture descriptor pool was destroyed first, so no sets from layout remain when destroy_descriptor_set_layout consumes it.
                unsafe { device.destroy_descriptor_set_layout(layout, None) };
            }
            if let Some(pool) = self.transfer_command_pool.take() {
                // SAFETY: device_wait_idle was issued, so no transfer command buffer is pending when destroy_command_pool releases the pool and its command buffers.
                unsafe { device.destroy_command_pool(pool, None) };
            }
            if let Some(pool) = self.transfer_acquire_pool.take() {
                // SAFETY: transfer worker shutdown and device_wait_idle completed before pool destruction.
                unsafe { device.destroy_command_pool(pool, None) };
            }
            if let Some(semaphore) = self.transfer_timeline.take() {
                // SAFETY: device_wait_idle was issued before the taken transfer timeline semaphore is passed once to destroy_semaphore with its creation allocator None.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            if let Some(semaphore) = self.transfer_ownership_timeline.take() {
                // SAFETY: transfer worker shutdown and device_wait_idle completed before semaphore destruction.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            if let Some(pool) = self.texture_command_pool.take() {
                // SAFETY: texture worker shutdown and device idle completed before pool destruction.
                unsafe { device.destroy_command_pool(pool, None) };
            }
            if let Some(pool) = self.texture_acquire_pool.take() {
                // SAFETY: texture worker shutdown and device idle completed before pool destruction.
                unsafe { device.destroy_command_pool(pool, None) };
            }
            if let Some(semaphore) = self.texture_timeline.take() {
                // SAFETY: texture worker shutdown and device idle completed before semaphore destruction.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            if let Some(semaphore) = self.texture_ownership_timeline.take() {
                // SAFETY: texture worker shutdown and device idle completed before semaphore destruction.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            // SAFETY: all allocator-owned memory and child resources were released above.
            unsafe {
                device.destroy_device(None);
            }
        }
        // SAFETY: every child owned by the context is destroyed before the instance.
        unsafe { self.instance.destroy_instance(None) };
    }
}

///
/// # Errors
///
/// Returns the error from creating any frame slot.
pub(super) fn create_frame_slots(
    device: &ash::Device,
    queue_family: u32,
) -> Result<Vec<FrameSlot>, HalError> {
    let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        match create_frame_slot(device, queue_family) {
            Ok(slot) => slots.push(slot),
            Err(error) => {
                for slot in slots.drain(..) {
                    destroy_frame_slot(device, &slot);
                }
                return Err(error);
            }
        }
    }
    Ok(slots)
}

///
/// # Errors
///
/// Returns an error if Vulkan fails to create the command pool, allocate the command buffer, or create a semaphore, fence, or descriptor pool.
pub(super) fn create_frame_slot(
    device: &ash::Device,
    queue_family: u32,
) -> Result<FrameSlot, HalError> {
    // SAFETY: queue_family is the selected family for device, and the CommandPoolCreateInfo temporary with an empty pNext chain spans create_command_pool.
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .map_err(map_vk)?;
    let created = (|| {
        // SAFETY: pool was just created and is not concurrently accessed, and the CommandBufferAllocateInfo storage spans allocate_command_buffers.
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(map_vk)?[0];
        let available =
            // SAFETY: the default SemaphoreCreateInfo temporary has an empty pNext chain and remains allocated throughout create_semaphore.
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(map_vk)?;
        // SAFETY: the FenceCreateInfo temporary contains no borrowed arrays and remains allocated throughout create_fence.
        let fence = match unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        } {
            Ok(value) => value,
            Err(error) => {
                // SAFETY: available was created with allocator None and never submitted.
                unsafe {
                    device.destroy_semaphore(available, None);
                }
                return Err(map_vk(error));
            }
        };
        // SAFETY: the DescriptorPoolSize and create-info storage remain allocated throughout create_descriptor_pool, and their declared counts are nonzero.
        let descriptor_pool = match unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAME_DESCRIPTOR_SET_CAPACITY)
                    .pool_sizes(core::slice::from_ref(
                        &vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(FRAME_DESCRIPTOR_SET_CAPACITY * 4),
                    )),
                None,
            )
        } {
            Ok(value) => value,
            Err(error) => {
                // SAFETY: fence and available were created with allocator None and never submitted.
                unsafe {
                    device.destroy_fence(fence, None);
                    device.destroy_semaphore(available, None);
                }
                return Err(map_vk(error));
            }
        };
        Ok(FrameSlot {
            command_pool: pool,
            command_buffer: command,
            image_available: available,
            fence,
            descriptor_pool,
            in_flight: false,
            submission_value: 0,
        })
    })();
    if created.is_err() {
        // SAFETY: no command buffer from pool was submitted, and destroy_command_pool releases any allocated command buffer using the creation allocator None.
        unsafe { device.destroy_command_pool(pool, None) };
    }
    created
}

pub(super) fn destroy_frame_slot(device: &ash::Device, slot: &FrameSlot) {
    let FrameSlot {
        command_pool,
        image_available,
        fence,
        descriptor_pool,
        ..
    } = slot;
    // SAFETY: callers use destroy_frame_slot only before submission or after device idle; destroying the pool releases command_buffer, and every listed handle uses allocator None.
    unsafe {
        device.destroy_descriptor_pool(*descriptor_pool, None);
        device.destroy_fence(*fence, None);
        device.destroy_semaphore(*image_available, None);
        device.destroy_command_pool(*command_pool, None);
    }
}

pub(super) fn map_vk(error: vk::Result) -> HalError {
    match error {
        vk::Result::ERROR_DEVICE_LOST => HalError::DeviceLost,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
            HalError::OutOfMemory
        }
        vk::Result::NOT_READY | vk::Result::TIMEOUT => HalError::NotReady,
        vk::Result::ERROR_EXTENSION_NOT_PRESENT
        | vk::Result::ERROR_FEATURE_NOT_PRESENT
        | vk::Result::ERROR_INCOMPATIBLE_DRIVER => HalError::Unsupported,
        _ => HalError::NativeFailure,
    }
}
