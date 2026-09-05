use super::{AllocationError, vk};
use ez_gfx_hal::{DEFAULT_STAGING_POLICY, TransferWorker, TransferWorkerError};

pub(super) enum VulkanTransferCopy {
    Buffer {
        source: vk::Buffer,
        destination: vk::Buffer,
        region: vk::BufferCopy,
    },
    Texture {
        source: vk::Buffer,
        destination: vk::Image,
        regions: Vec<vk::BufferImageCopy>,
    },
}

pub(super) struct VulkanTransferJob {
    pub value: u64,
    pub bytes: u64,
    pub copy: VulkanTransferCopy,
}

pub(super) fn job_bytes(job: &VulkanTransferJob) -> u64 {
    job.bytes
}
pub(super) fn job_group(job: &VulkanTransferJob) -> u64 {
    u64::from(matches!(job.copy, VulkanTransferCopy::Texture { .. }))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the owner receives the exact immutable Vulkan queue contract it exclusively drives"
)]
pub(super) fn start_worker(
    device: ash::Device,
    transfer_queue: vk::Queue,
    graphics_queue: vk::Queue,
    transfer_family: u32,
    graphics_family: u32,
    transfer_pool: vk::CommandPool,
    graphics_pool: Option<vk::CommandPool>,
    completion: vk::Semaphore,
    ownership: Option<vk::Semaphore>,
    queue_lock: std::sync::Arc<parking_lot::Mutex<()>>,
    graphics_lock: std::sync::Arc<parking_lot::Mutex<()>>,
) -> Result<TransferWorker<VulkanTransferJob>, AllocationError> {
    const COMMAND_SLOTS: usize = 4;
    const COMMAND_BUFFER_COUNT: u32 = 4;
    // SAFETY: the pool belongs to this device and remains live for every allocated slot.
    let transfers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(transfer_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(COMMAND_BUFFER_COUNT),
        )
    }
    .map_err(|_| AllocationError::NativeFailure)?;
    let acquires = if let Some(pool) = graphics_pool {
        Some(
            // SAFETY: the optional graphics pool belongs to this device and remains live.
            unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(COMMAND_BUFFER_COUNT),
                )
            }
            .map_err(|_| AllocationError::NativeFailure)?,
        )
    } else {
        None
    };
    let final_value = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let submitted_value = final_value.clone();
    let shutdown_device = device.clone();
    let mut ownership_value = 0_u64;
    let mut slot_values = [0_u64; COMMAND_SLOTS];
    let mut slot = 0_usize;
    TransferWorker::new_grouped_with_shutdown(
        64,
        DEFAULT_STAGING_POLICY,
        job_bytes,
        job_group,
        move |jobs| {
            let _queue_guard = queue_lock.lock();
            let _graphics_guard = (transfer_family != graphics_family
                && jobs.first().is_some_and(|job| job_group(job) == 1))
            .then(|| graphics_lock.lock());
            let previous = slot_values[slot];
            if previous != 0 {
                let wait = vk::SemaphoreWaitInfo::default()
                    .semaphores(core::slice::from_ref(&completion))
                    .values(core::slice::from_ref(&previous));
                // SAFETY: the completion timeline belongs to this device and the wait storage spans the call.
                unsafe { device.wait_semaphores(&wait, u64::MAX) }
                    .map_err(|_| TransferWorkerError::Failed)?;
            }
            // SAFETY: the completed slot's command buffers belong to their retained pools.
            unsafe {
                device
                    .reset_command_buffer(transfers[slot], vk::CommandBufferResetFlags::empty())
                    .map_err(|_| TransferWorkerError::Failed)?;
                if let Some(acquires) = &acquires {
                    device
                        .reset_command_buffer(acquires[slot], vk::CommandBufferResetFlags::empty())
                        .map_err(|_| TransferWorkerError::Failed)?;
                }
            }
            submit_batch(
                &device,
                transfer_queue,
                graphics_queue,
                transfer_family,
                graphics_family,
                transfers[slot],
                acquires.as_ref().map(|buffers| buffers[slot]),
                completion,
                ownership,
                &mut ownership_value,
                &jobs,
            )
            .map_err(|_| TransferWorkerError::Failed)?;
            let value = jobs.iter().map(|job| job.value).max().unwrap_or(0);
            slot_values[slot] = value;
            submitted_value.store(value, std::sync::atomic::Ordering::Release);
            slot = (slot + 1) % COMMAND_SLOTS;
            Ok(())
        },
        move || {
            let value = final_value.load(std::sync::atomic::Ordering::Acquire);
            if value != 0 {
                let wait = vk::SemaphoreWaitInfo::default()
                    .semaphores(core::slice::from_ref(&completion))
                    .values(core::slice::from_ref(&value));
                // SAFETY: the shutdown closure retains the device and timeline through this wait.
                unsafe { shutdown_device.wait_semaphores(&wait, u64::MAX) }
                    .map_err(|_| TransferWorkerError::Failed)?;
            }
            Ok(())
        },
    )
    .map_err(|_| AllocationError::NativeFailure)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one function keeps paired Vulkan release/acquire barriers and rollback-visible submission local"
)]
fn submit_batch(
    device: &ash::Device,
    transfer_queue: vk::Queue,
    graphics_queue: vk::Queue,
    transfer_family: u32,
    graphics_family: u32,
    transfer: vk::CommandBuffer,
    acquire: Option<vk::CommandBuffer>,
    completion: vk::Semaphore,
    ownership: Option<vk::Semaphore>,
    ownership_value: &mut u64,
    jobs: &[VulkanTransferJob],
) -> Result<(), vk::Result> {
    // SAFETY: this command buffer is idle, belongs to the transfer pool, and is retained through submission.
    unsafe {
        device.begin_command_buffer(
            transfer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }

    let separate = transfer_family != graphics_family;
    let mut acquire_images = Vec::new();
    for job in jobs {
        match &job.copy {
            VulkanTransferCopy::Buffer {
                source,
                destination,
                region,
            } => {
                // SAFETY: validated source and destination ranges remain live through submission.
                unsafe {
                    device.cmd_copy_buffer(transfer, *source, *destination, &[*region]);
                }
            }
            VulkanTransferCopy::Texture {
                source,
                destination,
                regions,
            } => {
                let level_count =
                    u32::try_from(regions.len()).map_err(|_| vk::Result::ERROR_UNKNOWN)?;
                // SAFETY: validated image ranges and retained resources remain live through submission.
                unsafe {
                    let range = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(level_count)
                        .layer_count(1);
                    let to_copy = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(*destination)
                        .subresource_range(range);
                    device.cmd_pipeline_barrier(
                        transfer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_copy],
                    );
                    device.cmd_copy_buffer_to_image(
                        transfer,
                        *source,
                        *destination,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        regions,
                    );
                    let mut release = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(*destination)
                        .subresource_range(range);
                    let destination_stage = if separate {
                        release = release
                            .src_queue_family_index(transfer_family)
                            .dst_queue_family_index(graphics_family);
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE
                    } else {
                        release = release.dst_access_mask(vk::AccessFlags::SHADER_READ);
                        vk::PipelineStageFlags::ALL_COMMANDS
                    };
                    device.cmd_pipeline_barrier(
                        transfer,
                        vk::PipelineStageFlags::TRANSFER,
                        destination_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[release],
                    );
                    if separate {
                        acquire_images.push((*destination, range));
                    }
                }
            }
        }
    }
    // SAFETY: transfer recording is complete and the buffer remains retained for submission.
    unsafe { device.end_command_buffer(transfer)? };
    let final_value = jobs.iter().map(|job| job.value).max().unwrap_or(0);

    if !separate || acquire_images.is_empty() {
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(core::slice::from_ref(&final_value));
        let submit = vk::SubmitInfo::default()
            .command_buffers(core::slice::from_ref(&transfer))
            .signal_semaphores(core::slice::from_ref(&completion))
            .push_next(&mut timeline);
        // SAFETY: queue, command buffer, and timeline belong to this live device.
        unsafe { device.queue_submit(transfer_queue, &[submit], vk::Fence::null())? };
        return Ok(());
    }

    *ownership_value = ownership_value
        .checked_add(1)
        .ok_or(vk::Result::ERROR_UNKNOWN)?;
    let ownership_value_now = *ownership_value;
    let ownership = ownership.ok_or(vk::Result::ERROR_UNKNOWN)?;
    let mut transfer_timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .signal_semaphore_values(core::slice::from_ref(&ownership_value_now));
    let transfer_submit = vk::SubmitInfo::default()
        .command_buffers(core::slice::from_ref(&transfer))
        .signal_semaphores(core::slice::from_ref(&ownership))
        .push_next(&mut transfer_timeline);
    // SAFETY: queue, command buffer, and ownership timeline belong to this live device.
    unsafe { device.queue_submit(transfer_queue, &[transfer_submit], vk::Fence::null())? };

    let acquire = acquire.ok_or(vk::Result::ERROR_UNKNOWN)?;
    // SAFETY: the acquire command buffer is idle and belongs to the graphics pool.
    unsafe {
        device.begin_command_buffer(
            acquire,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        for (image, range) in acquire_images {
            let barrier = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(transfer_family)
                .dst_queue_family_index(graphics_family)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                acquire,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        device.end_command_buffer(acquire)?;
    }
    let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
    let mut graphics_timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(core::slice::from_ref(&ownership_value_now))
        .signal_semaphore_values(core::slice::from_ref(&final_value));
    let graphics_submit = vk::SubmitInfo::default()
        .wait_semaphores(core::slice::from_ref(&ownership))
        .wait_dst_stage_mask(core::slice::from_ref(&wait_stage))
        .command_buffers(core::slice::from_ref(&acquire))
        .signal_semaphores(core::slice::from_ref(&completion))
        .push_next(&mut graphics_timeline);
    // SAFETY: the graphics queue and ownership/completion timelines belong to this live device.
    unsafe { device.queue_submit(graphics_queue, &[graphics_submit], vk::Fence::null())? };
    Ok(())
}
