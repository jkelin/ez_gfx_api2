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
        region: vk::BufferImageCopy,
        initialized: bool,
        stream_stage: u32,
    },
}

pub(super) struct VulkanTransferJob {
    pub value: u64,
    pub bytes: u64,
    pub copy: VulkanTransferCopy,
    pub cancelled: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

pub(super) fn job_bytes(job: &VulkanTransferJob) -> u64 {
    job.bytes
}

fn job_cancelled(job: &VulkanTransferJob) -> bool {
    job.cancelled
        .as_ref()
        .is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::Acquire))
}
pub(super) fn job_group(job: &VulkanTransferJob) -> u64 {
    match job.copy {
        VulkanTransferCopy::Buffer { .. } => 0,
        // Only adjacent jobs coalesce. A later stage can never complete with an earlier one.
        VulkanTransferCopy::Texture {
            initialized: true, ..
        } => u64::MAX,
        VulkanTransferCopy::Texture { stream_stage, .. } => u64::from(stream_stage) + 1,
    }
}

/// Maps a Vulkan submission result onto the worker poison reason without collapsing loss.
fn map_submit(error: vk::Result) -> TransferWorkerError {
    if error == vk::Result::ERROR_DEVICE_LOST {
        TransferWorkerError::DeviceLost
    } else {
        TransferWorkerError::Failed
    }
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
    const GRAPHICS_COMMAND_BUFFER_COUNT: u32 = COMMAND_BUFFER_COUNT * 2;
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
    let graphics = if let Some(pool) = graphics_pool {
        Some(
            // SAFETY: the optional graphics pool belongs to this device and remains live.
            unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(GRAPHICS_COMMAND_BUFFER_COUNT),
                )
            }
            .map_err(|_| AllocationError::NativeFailure)?,
        )
    } else {
        None
    };
    let shutdown_queue_lock = queue_lock.clone();
    let shutdown_graphics_lock = graphics_lock.clone();
    let shutdown_device = device.clone();
    let mut ownership_value = 0_u64;
    let mut slot_values = [0_u64; COMMAND_SLOTS];
    let mut slot = 0_usize;
    TransferWorker::new_ordered_with_shutdown(
        64,
        DEFAULT_STAGING_POLICY,
        job_bytes,
        job_group,
        |job| job.value,
        move |jobs| {
            let _queue_guard = queue_lock.lock();
            let previous = slot_values[slot];
            if previous != 0 {
                let wait = vk::SemaphoreWaitInfo::default()
                    .semaphores(core::slice::from_ref(&completion))
                    .values(core::slice::from_ref(&previous));
                // SAFETY: the completion timeline belongs to this device and the wait storage spans the call.
                unsafe { device.wait_semaphores(&wait, u64::MAX) }.map_err(map_submit)?;
            }
            // SAFETY: the completed slot's command buffers belong to their retained pools.
            unsafe {
                device
                    .reset_command_buffer(transfers[slot], vk::CommandBufferResetFlags::empty())
                    .map_err(map_submit)?;
                if let Some(graphics) = &graphics {
                    device
                        .reset_command_buffer(
                            graphics[slot * 2],
                            vk::CommandBufferResetFlags::empty(),
                        )
                        .map_err(map_submit)?;
                    device
                        .reset_command_buffer(
                            graphics[slot * 2 + 1],
                            vk::CommandBufferResetFlags::empty(),
                        )
                        .map_err(map_submit)?;
                }
            }
            submit_batch(
                &device,
                transfer_queue,
                graphics_queue,
                transfer_family,
                graphics_family,
                transfers[slot],
                graphics.as_ref().map(|buffers| buffers[slot * 2]),
                graphics.as_ref().map(|buffers| buffers[slot * 2 + 1]),
                completion,
                ownership,
                &mut ownership_value,
                &graphics_lock,
                &jobs,
            )
            .map_err(map_submit)?;
            let value = jobs.iter().map(|job| job.value).max().unwrap_or(0);
            slot_values[slot] = value;
            slot = (slot + 1) % COMMAND_SLOTS;
            Ok(())
        },
        move || {
            // A partial batch may never signal its application completion. Drain the
            // actual queues before their command pools or staging storage can retire.
            let transfer_result = {
                let _guard = shutdown_queue_lock.lock();
                let _graphics_guard =
                    (transfer_queue == graphics_queue).then(|| shutdown_graphics_lock.lock());
                // SAFETY: the queue is retained and its host access is serialized.
                unsafe { shutdown_device.queue_wait_idle(transfer_queue) }
            };
            let graphics_result = if graphics_queue == transfer_queue {
                Ok(())
            } else {
                let _guard = shutdown_graphics_lock.lock();
                // SAFETY: the graphics queue is retained and locked independently.
                unsafe { shutdown_device.queue_wait_idle(graphics_queue) }
            };
            transfer_result.and(graphics_result).map_err(map_submit)
        },
    )
    .map_err(|_| AllocationError::NativeFailure)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one function keeps paired Vulkan release/acquire barriers and rollback-visible submission local"
)]
#[expect(
    clippy::too_many_lines,
    reason = "one native queue transaction keeps ownership barriers and timeline submission ordered"
)]
fn submit_batch(
    device: &ash::Device,
    transfer_queue: vk::Queue,
    graphics_queue: vk::Queue,
    transfer_family: u32,
    graphics_family: u32,
    transfer: vk::CommandBuffer,
    release_command: Option<vk::CommandBuffer>,
    acquire_command: Option<vk::CommandBuffer>,
    completion: vk::Semaphore,
    ownership: Option<vk::Semaphore>,
    ownership_value: &mut u64,
    graphics_lock: &parking_lot::Mutex<()>,
    jobs: &[VulkanTransferJob],
) -> Result<(), vk::Result> {
    let separate = transfer_family != graphics_family;
    // Snapshot cancellation once: a release must retain its matching copy and acquire even if
    // cancellation races recording. Skipped jobs still retire in FIFO completion order.
    let live = jobs
        .iter()
        .filter(|job| !job_cancelled(job))
        .collect::<Vec<_>>();
    let texture_batch = jobs.first().is_some_and(|job| job_group(job) != 0);
    let mut textures = Vec::with_capacity(live.len());
    for job in &live {
        if let VulkanTransferCopy::Texture {
            destination,
            region,
            initialized,
            ..
        } = &job.copy
        {
            let mip = region.image_subresource.mip_level;
            // Several region writes can target one mip; ownership changes only once per batch.
            if !textures
                .iter()
                .any(|(image, level, _)| image == destination && *level == mip)
            {
                textures.push((*destination, mip, *initialized));
            }
        }
    }

    let mut transfer_wait = None;
    if textures.iter().any(|(_, _, initialized)| *initialized) {
        let release = release_command.ok_or(vk::Result::ERROR_UNKNOWN)?;
        // SAFETY: Both command buffers belong to live queues, are idle, and remain retained through
        // submission. Updates release an initialized graphics-owned mip before transfer writes it.
        unsafe {
            device.begin_command_buffer(
                release,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            for &(image, mip_level, initialized) in &textures {
                if !initialized {
                    continue;
                }
                let range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip_level)
                    .level_count(1)
                    .layer_count(1);
                let mut barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range);
                if separate {
                    barrier = barrier
                        .src_queue_family_index(graphics_family)
                        .dst_queue_family_index(transfer_family);
                }
                device.cmd_pipeline_barrier(
                    release,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
            device.end_command_buffer(release)?;
        }
        *ownership_value = ownership_value
            .checked_add(1)
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        let value = *ownership_value;
        let ownership = ownership.ok_or(vk::Result::ERROR_UNKNOWN)?;
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(core::slice::from_ref(&value));
        let submit = vk::SubmitInfo::default()
            .command_buffers(core::slice::from_ref(&release))
            .signal_semaphores(core::slice::from_ref(&ownership))
            .push_next(&mut timeline);
        let _graphics_guard = graphics_lock.lock();
        // SAFETY: The recorded release buffer and timeline storage remain live through submission.
        unsafe { device.queue_submit(graphics_queue, &[submit], vk::Fence::null())? };
        transfer_wait = Some((ownership, value));
    }

    // SAFETY: this command buffer is idle, belongs to the transfer pool, and remains retained.
    unsafe {
        device.begin_command_buffer(
            transfer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        for (index, job) in live.iter().enumerate() {
            match &job.copy {
                VulkanTransferCopy::Buffer {
                    source,
                    destination,
                    region,
                } => device.cmd_copy_buffer(transfer, *source, *destination, &[*region]),
                VulkanTransferCopy::Texture {
                    source,
                    destination,
                    region,
                    initialized,
                    ..
                } => {
                    let range = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(region.image_subresource.mip_level)
                        .level_count(1)
                        .layer_count(1);
                    let repeated = live[..index].iter().any(|previous| matches!(
                        &previous.copy,
                        VulkanTransferCopy::Texture { destination: image, region: copy, .. }
                            if image == destination
                                && copy.image_subresource.mip_level == region.image_subresource.mip_level
                    ));
                    if repeated {
                        // Overlapping region writes must preserve admission order while the mip
                        // stays transfer-owned between its first and last copy.
                        let barrier = vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                        device.cmd_pipeline_barrier(
                            transfer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[barrier],
                            &[],
                            &[],
                        );
                    } else {
                        let mut to_copy = vk::ImageMemoryBarrier::default()
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .old_layout(if *initialized {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            } else {
                                vk::ImageLayout::UNDEFINED
                            })
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(*destination)
                            .subresource_range(range);
                        if *initialized {
                            // Ownership acquires must repeat the release's exact old/new layouts.
                            // Same-family copies instead see the release's completed transition.
                            if !separate {
                                to_copy = to_copy.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                            }
                            if separate {
                                to_copy = to_copy
                                    .src_queue_family_index(graphics_family)
                                    .dst_queue_family_index(transfer_family);
                            }
                        }
                        device.cmd_pipeline_barrier(
                            transfer,
                            if *initialized {
                                vk::PipelineStageFlags::ALL_COMMANDS
                            } else {
                                vk::PipelineStageFlags::TOP_OF_PIPE
                            },
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[to_copy],
                        );
                    }
                    device.cmd_copy_buffer_to_image(
                        transfer,
                        *source,
                        *destination,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        core::slice::from_ref(region),
                    );
                    let copied_again = live[index + 1..].iter().any(|next| matches!(
                        &next.copy,
                        VulkanTransferCopy::Texture { destination: image, region: copy, .. }
                            if image == destination
                                && copy.image_subresource.mip_level == region.image_subresource.mip_level
                    ));
                    if copied_again {
                        continue;
                    }
                    let mut to_shader = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(*destination)
                        .subresource_range(range);
                    let destination_stage = if separate {
                        to_shader = to_shader
                            .src_queue_family_index(transfer_family)
                            .dst_queue_family_index(graphics_family);
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE
                    } else {
                        to_shader = to_shader.dst_access_mask(vk::AccessFlags::SHADER_READ);
                        vk::PipelineStageFlags::ALL_COMMANDS
                    };
                    device.cmd_pipeline_barrier(
                        transfer,
                        vk::PipelineStageFlags::TRANSFER,
                        destination_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_shader],
                    );
                }
            }
        }
        device.end_command_buffer(transfer)?;
    }

    let final_value = jobs.iter().map(|job| job.value).max().unwrap_or(0);
    // Even an entirely cancelled texture batch follows the graphics completion path, so its
    // signal cannot overtake an earlier live batch's pending graphics ownership acquire.
    if !separate || !texture_batch {
        let wait_values = transfer_wait.map(|(_, value)| [value]);
        let wait_semaphores = transfer_wait.map(|(semaphore, _)| [semaphore]);
        let signal_values = [final_value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(wait_values.as_ref().map_or(&[], |values| values))
            .signal_semaphore_values(&signal_values);
        let wait_stage = [vk::PipelineStageFlags::TRANSFER];
        let mut submit = vk::SubmitInfo::default()
            .command_buffers(core::slice::from_ref(&transfer))
            .signal_semaphores(core::slice::from_ref(&completion));
        if let Some(semaphores) = wait_semaphores.as_ref() {
            submit = submit
                .wait_semaphores(semaphores)
                .wait_dst_stage_mask(&wait_stage);
        }
        submit = submit.push_next(&mut timeline);
        let _graphics_guard = (transfer_queue == graphics_queue).then(|| graphics_lock.lock());
        // SAFETY: The recorded transfer buffer and semaphore storage remain live through submission.
        unsafe { device.queue_submit(transfer_queue, &[submit], vk::Fence::null())? };
        #[cfg(test)]
        submission_observation::submitted(device, transfer_queue, final_value)?;
        return Ok(());
    }

    *ownership_value = ownership_value
        .checked_add(1)
        .ok_or(vk::Result::ERROR_UNKNOWN)?;
    let transfer_done = *ownership_value;
    let ownership = ownership.ok_or(vk::Result::ERROR_UNKNOWN)?;
    let wait_values = transfer_wait.map(|(_, value)| [value]);
    let signal_values = [transfer_done];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(wait_values.as_ref().map_or(&[], |values| values))
        .signal_semaphore_values(&signal_values);
    let wait_stage = [vk::PipelineStageFlags::TRANSFER];
    let mut submit = vk::SubmitInfo::default()
        .command_buffers(core::slice::from_ref(&transfer))
        .signal_semaphores(core::slice::from_ref(&ownership));
    if transfer_wait.is_some() {
        submit = submit
            .wait_semaphores(core::slice::from_ref(&ownership))
            .wait_dst_stage_mask(&wait_stage);
    }
    submit = submit.push_next(&mut timeline);
    // SAFETY: The recorded transfer buffer and timeline storage remain live through submission.
    unsafe { device.queue_submit(transfer_queue, &[submit], vk::Fence::null())? };
    #[cfg(test)]
    submission_observation::submitted(device, transfer_queue, final_value)?;

    let acquire = acquire_command.ok_or(vk::Result::ERROR_UNKNOWN)?;
    // SAFETY: The acquire buffer is idle, retained, and allocated from the graphics-family pool.
    unsafe {
        device.begin_command_buffer(
            acquire,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        for &(image, mip_level, _) in &textures {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(mip_level)
                .level_count(1)
                .layer_count(1);
            let barrier = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
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
    {
        // Keep pending fine copies, including updates outside a coarse view, off graphics.
        // Submitted storage still drains after cancellation; only completion or native failure
        // permits retirement. This wait belongs to the dedicated transfer owner, not the caller.
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(core::slice::from_ref(&ownership))
            .values(core::slice::from_ref(&transfer_done));
        loop {
            // SAFETY: the worker retains this device and timeline until the handoff finishes.
            match unsafe { device.wait_semaphores(&wait, 100_000_000) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => {}
                Err(error) => return Err(error),
            }
        }
    }
    let wait_values = [transfer_done];
    let signal_values = [final_value];
    let wait_stage = [vk::PipelineStageFlags::ALL_COMMANDS];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values)
        .signal_semaphore_values(&signal_values);
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(core::slice::from_ref(&ownership))
        .wait_dst_stage_mask(&wait_stage)
        .command_buffers(core::slice::from_ref(&acquire))
        .signal_semaphores(core::slice::from_ref(&completion))
        .push_next(&mut timeline);
    let _graphics_guard = graphics_lock.lock();
    // SAFETY: The acquire buffer and both timeline semaphores remain live through submission.
    unsafe { device.queue_submit(graphics_queue, &[submit], vk::Fence::null())? };
    Ok(())
}

#[cfg(test)]
pub(super) mod submission_observation {
    use ash::vk::{self, Handle};
    use parking_lot::Mutex;
    use std::sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };

    type QueueKey = (u64, u64);
    static OBSERVERS: LazyLock<Mutex<std::collections::HashMap<QueueKey, Arc<Observation>>>> =
        LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

    struct Observation {
        value: AtomicU64,
        fail_next: AtomicBool,
    }

    pub(crate) struct Observer {
        key: QueueKey,
        observation: Arc<Observation>,
    }

    impl Observer {
        pub(crate) fn new(device: &ash::Device, queue: vk::Queue) -> Self {
            // Queue handles are scoped by device; parallel tests cannot observe each other's work.
            let key = (device.handle().as_raw(), queue.as_raw());
            let observation = Arc::new(Observation {
                value: AtomicU64::new(0),
                fail_next: AtomicBool::new(false),
            });
            let mut observers = OBSERVERS.lock();
            assert!(!observers.contains_key(&key), "queue already observed");
            observers.insert(key, observation.clone());
            Self { key, observation }
        }

        pub(crate) fn value(&self) -> u64 {
            self.observation.value.load(Ordering::Acquire)
        }

        pub(crate) fn fail_next_submission(&self) {
            self.observation.fail_next.store(true, Ordering::Release);
        }
    }

    impl Drop for Observer {
        fn drop(&mut self) {
            // Remove registration before device destruction permits native handle reuse.
            OBSERVERS.lock().remove(&self.key);
        }
    }

    pub(super) fn submitted(
        device: &ash::Device,
        queue: vk::Queue,
        value: u64,
    ) -> Result<(), vk::Result> {
        if let Some(observer) = OBSERVERS
            .lock()
            .get(&(device.handle().as_raw(), queue.as_raw()))
        {
            observer.value.store(value, Ordering::Release);
            // Inject only after a real native submission; the GPU may still own all its storage.
            if observer.fail_next.swap(false, Ordering::AcqRel) {
                return Err(vk::Result::ERROR_UNKNOWN);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn native_texture_stages_coalesce_without_skipping_fine_mips() {
        let (sent, received) = std::sync::mpsc::channel();
        let mut worker = TransferWorker::new_ordered_with_shutdown(
            8,
            ez_gfx_hal::StagingPolicy::new(1, 8, 8, 8).unwrap(),
            job_bytes,
            job_group,
            |job| job.value,
            move |jobs| {
                sent.send(jobs.iter().map(|job| job.value).collect::<Vec<_>>())
                    .unwrap();
                Ok(())
            },
            || Ok(()),
        )
        .unwrap();
        // The final coarse mip cannot jump ahead of the preceding fine stage; updates must
        // neither merge with that coarse stage nor split merely because texture IDs differ.
        let jobs = [
            (1, 0, false),
            (2, 0, false),
            (1, 1, false),
            (3, 0, false),
            (1, 0, true),
            (2, 0, true),
        ]
        .into_iter()
        .enumerate()
        .map(
            |(index, (image, stream_stage, initialized))| VulkanTransferJob {
                value: index as u64 + 1,
                bytes: 1,
                cancelled: None,
                copy: VulkanTransferCopy::Texture {
                    source: vk::Buffer::null(),
                    destination: vk::Image::from_raw(image),
                    region: vk::BufferImageCopy::default(),
                    initialized,
                    stream_stage,
                },
            },
        )
        .collect();
        worker.submit_batch(jobs).unwrap();
        worker.shutdown();

        assert_eq!(
            received.into_iter().collect::<Vec<_>>(),
            [vec![1, 2], vec![3], vec![4], vec![5, 6]]
        );
    }
}
