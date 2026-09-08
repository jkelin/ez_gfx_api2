use super::{
    AllocationError, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLEvent, MTLOrigin, MTLSize, MTLTexture, ProtocolObject,
    Retained,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// An encoded buffer command transferred once to its dedicated submission owner.
pub(super) struct TransferCommand {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl TransferCommand {
    pub(super) fn new(command: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self { command }
    }

    fn commit(&self) {
        self.command.commit();
    }
}

// SAFETY: Metal permits command-buffer commit/status/release across threads; encoding is
// finished before transfer. Shared texture status access below is serialized by its mutex.
unsafe impl Send for TransferCommand {}

#[derive(Default)]
struct SubmittedCommands {
    commands: Mutex<Vec<TransferCommand>>,
}

impl SubmittedCommands {
    fn retain_before_commit(&self, command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        // Poison cannot discard submitted buffers: shutdown must still own their storage.
        let mut commands = self
            .commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        commands.retain(|entry| {
            !matches!(
                entry.command.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        });
        commands.push(TransferCommand::new(command.clone()));
    }

    /// Reports whether any retained buffer already reached error status.
    ///
    /// Status queries never block; unsettled buffers simply report no error yet.
    /// Late failures still surface through per-submission `completed()` polling.
    fn has_error(&self) -> bool {
        self.commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|entry| entry.command.status() == MTLCommandBufferStatus::Error)
    }

    fn drain(&self) -> Result<(), ez_gfx_hal::TransferWorkerError> {
        let mut commands = self
            .commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut failed = false;
        for entry in commands.iter() {
            // A failed encoder/commit may leave an unsubmitted buffer; never wait for it.
            if matches!(
                entry.command.status(),
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled
            ) {
                entry.command.waitUntilCompleted();
            }
            failed |= matches!(
                entry.command.status(),
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled
            );
        }
        if failed {
            return Err(ez_gfx_hal::TransferWorkerError::Failed);
        }
        commands.clear();
        Ok(())
    }
}

impl Drop for SubmittedCommands {
    fn drop(&mut self) {
        // Retention must remain safe even while a worker callback unwinds.
        if self.drain().is_err() {
            let commands = self
                .commands
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            core::mem::forget(core::mem::take(commands));
        }
    }
}

pub(super) struct TransferCancellation {
    cancelled: AtomicBool,
}

impl TransferCancellation {
    pub(super) const fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
        }
    }

    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

pub(super) struct MetalTransferJob {
    pub value: u64,
    pub bytes: u64,
    pub command: TransferCommand,
}

fn job_bytes(job: &MetalTransferJob) -> u64 {
    job.bytes
}

pub(super) fn start_worker() -> Result<ez_gfx_hal::TransferWorker<MetalTransferJob>, AllocationError>
{
    let mut last_value = 0_u64;
    let submitted = Arc::new(SubmittedCommands::default());
    let shutdown = submitted.clone();
    ez_gfx_hal::TransferWorker::new_ordered_with_shutdown(
        ez_gfx_hal::DEFAULT_STAGING_POLICY,
        job_bytes,
        |_| 0,
        |job| job.value,
        move |jobs| {
            for job in jobs {
                if job.value <= last_value {
                    return Err(ez_gfx_hal::TransferWorkerError::Failed);
                }
                last_value = job.value;
                submitted.retain_before_commit(&job.command.command);
                job.command.commit();
            }
            Ok(())
        },
        move || shutdown.drain(),
    )
    .map_err(|_| AllocationError::NativeFailure)
}

/// Validated immutable copy description; only the texture owner creates native encoders.
pub(super) struct TextureCopy {
    pub source: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub destination: Retained<ProtocolObject<dyn MTLTexture>>,
    pub source_offset: usize,
    pub row_bytes: usize,
    pub image_bytes: usize,
    pub size: MTLSize,
    pub level: usize,
    pub origin: MTLOrigin,
}

// SAFETY: Metal resources permit retained ownership across threads. Staging bytes are flushed
// before admission and cannot be reused until completion; only this worker encodes these copies.
unsafe impl Send for TextureCopy {}

#[derive(Default)]
pub(super) struct TextureSubmission {
    command: Mutex<Option<TransferCommand>>,
    skipped: AtomicBool,
}

impl TextureSubmission {
    #[cfg(test)]
    pub(super) fn submitted(&self) -> bool {
        // The command is retained before commit; status distinguishes actual GPU admission.
        self.command.lock().is_ok_and(|command| {
            command.as_ref().is_some_and(|command| {
                matches!(
                    command.command.status(),
                    MTLCommandBufferStatus::Committed
                        | MTLCommandBufferStatus::Scheduled
                        | MTLCommandBufferStatus::Completed
                        | MTLCommandBufferStatus::Error
                )
            })
        })
    }

    pub(super) fn completed(&self) -> Result<bool, AllocationError> {
        if self.skipped.load(Ordering::Acquire) {
            return Ok(true);
        }
        // A queued job has no native command yet and must not advance CPU completion.
        let command = self
            .command
            .lock()
            .map_err(|_| AllocationError::NativeFailure)?;
        let Some(command) = command.as_ref() else {
            return Ok(false);
        };
        match command.command.status() {
            MTLCommandBufferStatus::Completed if command.command.error().is_none() => Ok(true),
            MTLCommandBufferStatus::Error => Err(AllocationError::NativeFailure),
            _ => Ok(false),
        }
    }
}

pub(super) struct TextureTransferJob {
    pub value: u64,
    pub bytes: u64,
    /// Coarse-first ordinal; updates use `u64::MAX`, never a texture-specific token.
    pub stage: u64,
    pub graphics_wait: bool,
    pub copy: TextureCopy,
    pub cancellation: Arc<TransferCancellation>,
    pub submission: Arc<TextureSubmission>,
}

pub(super) struct PendingTextureTransfer {
    pub value: u64,
    pub submission: Arc<TextureSubmission>,
}

pub(super) fn start_texture_worker(
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    graphics_event: Retained<ProtocolObject<dyn MTLEvent>>,
    completion_event: Retained<ProtocolObject<dyn MTLEvent>>,
) -> Result<ez_gfx_hal::TransferWorker<TextureTransferJob>, AllocationError> {
    let mut last_value = 0;
    let submitted = Arc::new(SubmittedCommands::default());
    let shutdown = submitted.clone();
    ez_gfx_hal::TransferWorker::new_ordered_with_shutdown(ez_gfx_hal::DEFAULT_STAGING_POLICY, |job: &TextureTransferJob| job.bytes, |job| job.stage, |job| job.value, move |mut jobs| {
        // Admission order is immutable: a later signal may cover only copied or cancelled jobs.
        for job in &jobs {
            if job.value <= last_value {
                return Err(ez_gfx_hal::TransferWorkerError::Failed);
            }
            last_value = job.value;
        }
        // Snapshot cancellation once. Cancellation after this point retires submitted work.
        jobs.retain(|job| {
            if job.cancellation.cancelled.load(Ordering::Acquire) {
                job.submission.skipped.store(true, Ordering::Release);
                false
            } else {
                true
            }
        });
        // An all-cancelled batch still needs a real FIFO marker for accepted event waits.
        let command = queue.commandBuffer().ok_or(ez_gfx_hal::TransferWorkerError::Failed)?;
        // Wait outside the encoder for every accepted overwrite's graphics release.
        for job in &jobs {
            if job.graphics_wait {
                command.encodeWaitForEvent_value(&graphics_event, job.value);
            }
        }
        if !jobs.is_empty() {
        let blit = command.blitCommandEncoder().ok_or(ez_gfx_hal::TransferWorkerError::Failed)?;
        for job in &jobs {
            let copy = &job.copy;
            // SAFETY: producer validation bounds the source strides and destination extent.
            // The job retains both resources; the default command buffer retains encoded resources.
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &copy.source, copy.source_offset, copy.row_bytes, copy.image_bytes,
                    copy.size, &copy.destination, 0, copy.level, copy.origin,
                );
            }
        }
        blit.endEncoding();
        }
        command.encodeSignalEvent_value(&completion_event, last_value);
        // Fail promptly without blocking when an earlier batch errored: retained
        // buffers are pruned below, so observe their status before pruning.
        if submitted.has_error() {
            return Err(ez_gfx_hal::TransferWorkerError::Failed);
        }
        // Publish every observer before commit: a poisoned observer cannot orphan live work.
        for job in &jobs {
            *job.submission.command.lock().map_err(|_| ez_gfx_hal::TransferWorkerError::Failed)? =
                Some(TransferCommand::new(command.clone()));
        }
        submitted.retain_before_commit(&command);
        // Ordering and failure observation are GPU-side from here: the completion
        // event orders graphics waits and `completed()` polls buffer status lazily.
        // Shutdown and wait-idle drains still join actual GPU work terminally.
        command.commit();
        Ok(())
    }, move || shutdown.drain()).map_err(|_| AllocationError::NativeFailure)
}

#[cfg(test)]
mod tests {
    use super::super::{
        MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat, MTLResourceOptions,
        MTLStorageMode, MTLTextureDescriptor,
    };
    use super::*;
    use objc2_metal::MTLSharedEvent;

    /// Encodes a host-signalable gate wait into a fresh command buffer.
    #[allow(
        clippy::semicolon_if_nothing_returned,
        reason = "a trailing semicolon breaks msg_send return-type inference"
    )]
    fn encode_gate_wait(
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        event: &ProtocolObject<dyn MTLSharedEvent>,
    ) {
        // SAFETY: `encodeWaitForEvent:value:` is implemented by every Metal command
        // buffer, and the shared event conforms to `MTLEvent` and outlives the call.
        let () = unsafe { objc2::msg_send![buffer, encodeWaitForEvent: event, value: 1u64] };
    }

    #[test]
    fn shutdown_drains_partial_commit_after_submission_failure() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let queue = device.newCommandQueue().unwrap();
        let command = queue.commandBuffer().unwrap();
        let mut worker = start_worker().unwrap();
        worker
            .submit_batch(vec![
                MetalTransferJob {
                    value: 1,
                    bytes: 1,
                    command: TransferCommand::new(command.clone()),
                },
                MetalTransferJob {
                    value: 0,
                    bytes: 1,
                    command: TransferCommand::new(queue.commandBuffer().unwrap()),
                },
            ])
            .unwrap();
        assert_eq!(worker.flush(), Err(ez_gfx_hal::TransferWorkerError::Failed));
        worker.shutdown();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }
    #[test]
    fn shared_texture_batch_copies_independent_destinations_and_skips_cancelled_copy() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let queue = device.newCommandQueue().unwrap();
        let completion_event = device.newEvent().unwrap();
        let mut worker = start_texture_worker(
            queue.clone(),
            device.newEvent().unwrap(),
            completion_event.clone(),
        )
        .unwrap();
        let mut jobs = Vec::new();
        let mut textures = Vec::new();
        let mut submissions = Vec::new();
        for (index, pixel) in [[17_u8, 33, 65, 255], [99, 88, 77, 255], [0, 0, 0, 0]]
            .into_iter()
            .enumerate()
        {
            let source = device
                .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
                .unwrap();
            // SAFETY: the shared buffer is four writable bytes and no work references it yet.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pixel.as_ptr(),
                    source.contents().as_ptr().cast(),
                    4,
                );
            };
            // SAFETY: one RGBA8 pixel and one mip form a valid 2D descriptor.
            let desc = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    MTLPixelFormat::RGBA8Unorm,
                    1,
                    1,
                    false,
                )
            };
            desc.setStorageMode(MTLStorageMode::Shared);
            let texture = device.newTextureWithDescriptor(&desc).unwrap();
            let cancellation = Arc::new(TransferCancellation::new());
            if index == 2 {
                cancellation.cancel();
            }
            let submission = Arc::new(TextureSubmission::default());
            assert!(!submission.completed().unwrap());
            jobs.push(TextureTransferJob {
                value: index as u64 + 1,
                bytes: 4,
                stage: u64::from(index == 2),
                graphics_wait: false,
                copy: TextureCopy {
                    source,
                    destination: texture.clone(),
                    source_offset: 0,
                    row_bytes: 4,
                    image_bytes: 4,
                    size: MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    level: 0,
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                },
                cancellation,
                submission: submission.clone(),
            });
            textures.push(texture);
            submissions.push(submission);
        }
        worker.submit_batch(jobs).unwrap();
        worker.flush_through(3).unwrap();
        // Flush proves driver submission, not GPU completion: the completion event
        // orders the real graphics wait below, which still proves GPU execution.
        // The cancelled-only final batch must unblock a real graphics event wait.
        let graphics = device.newCommandQueue().unwrap();
        let waiter = graphics.commandBuffer().unwrap();
        waiter.encodeWaitForEvent_value(&completion_event, 3);
        waiter.commit();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !matches!(
            waiter.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        ) && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(waiter.status(), MTLCommandBufferStatus::Completed);
        for submission in &submissions {
            assert!(submission.completed().unwrap());
        }
        assert!(submissions[2].skipped.load(Ordering::Acquire));
        {
            // Native batching is part of the contract, not merely adjacent independent commits.
            let first = submissions[0].command.lock().unwrap();
            let second = submissions[1].command.lock().unwrap();
            assert!(core::ptr::eq(
                &raw const *first.as_ref().unwrap().command,
                &raw const *second.as_ref().unwrap().command,
            ));
        }
        for (texture, expected) in textures
            .iter()
            .zip([[17_u8, 33, 65, 255], [99, 88, 77, 255]])
        {
            let mut actual = [0_u8; 4];
            // SAFETY: the destination holds one RGBA8 pixel; the queue has finished all writes.
            unsafe {
                texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    core::ptr::NonNull::new(actual.as_mut_ptr().cast()).unwrap(),
                    4,
                    objc2_metal::MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                    },
                    0,
                );
            }
            assert_eq!(actual, expected);
        }
        worker.shutdown();
    }

    /// Holds GPU execution behind a host-signaled event; dropping releases it so
    /// a panic cannot wedge the queue and hang worker shutdown.
    struct EventGate {
        event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    }

    impl EventGate {
        fn release(&self) {
            self.event.setSignaledValue(1);
        }
    }

    impl Drop for EventGate {
        fn drop(&mut self) {
            self.release();
        }
    }

    #[test]
    fn consecutive_texture_batches_submit_without_waiting_for_gpu_completion() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let queue = device.newCommandQueue().unwrap();
        let completion_event = device.newEvent().unwrap();
        let mut worker =
            start_texture_worker(queue.clone(), device.newEvent().unwrap(), completion_event)
                .unwrap();
        // Gate all GPU execution behind a host-signaled event: the gate buffer is
        // committed first, so every worker submission queues behind it and cannot
        // execute until release. Host submission flow is fully deterministic.
        let gate_event = device.newSharedEvent().unwrap();
        let gate_buffer = queue.commandBuffer().unwrap();
        encode_gate_wait(&gate_buffer, &gate_event);
        gate_buffer.commit();
        let gate = EventGate { event: gate_event };
        // A 64 MiB first batch keeps the GPU busy after release for good measure.
        let big_bytes = 4096_usize * 4096 * 4;
        let big_source = device
            .newBufferWithLength_options(big_bytes, MTLResourceOptions::StorageModeShared)
            .unwrap();
        // SAFETY: a 4096x4096 RGBA8 shared texture forms a valid 2D descriptor.
        let big_desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                4096,
                4096,
                false,
            )
        };
        big_desc.setStorageMode(MTLStorageMode::Shared);
        let big = device.newTextureWithDescriptor(&big_desc).unwrap();
        let small_source = device
            .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        // SAFETY: one RGBA8 pixel and one mip form a valid 2D descriptor.
        let small_desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                1,
                1,
                false,
            )
        };
        small_desc.setStorageMode(MTLStorageMode::Shared);
        let small = device.newTextureWithDescriptor(&small_desc).unwrap();
        let big_sub = Arc::new(TextureSubmission::default());
        let small_sub = Arc::new(TextureSubmission::default());
        let copy = |source,
                    destination: &Retained<ProtocolObject<dyn MTLTexture>>,
                    row_bytes: usize,
                    image_bytes: usize,
                    size: MTLSize| TextureCopy {
            source,
            destination: destination.clone(),
            source_offset: 0,
            row_bytes,
            image_bytes,
            size,
            level: 0,
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
        };
        worker
            .submit_batch(vec![TextureTransferJob {
                value: 1,
                bytes: u64::try_from(big_bytes).unwrap(),
                stage: 1,
                graphics_wait: false,
                copy: copy(
                    big_source,
                    &big,
                    4096 * 4,
                    4096 * 4096 * 4,
                    MTLSize {
                        width: 4096,
                        height: 4096,
                        depth: 1,
                    },
                ),
                cancellation: Arc::new(TransferCancellation::new()),
                submission: big_sub.clone(),
            }])
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !big_sub.submitted() {
            assert!(
                std::time::Instant::now() < deadline,
                "first native copy submission timed out"
            );
            std::thread::yield_now();
        }
        // Observed submission proves the worker finished the first callback, so the
        // second batch must not wait for still-executing GPU work.
        worker
            .submit_batch(vec![TextureTransferJob {
                value: 2,
                bytes: 4,
                stage: 2,
                graphics_wait: false,
                copy: copy(
                    small_source,
                    &small,
                    4,
                    4,
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                ),
                cancellation: Arc::new(TransferCancellation::new()),
                submission: small_sub.clone(),
            }])
            .unwrap();
        while !small_sub.submitted() {
            assert!(
                std::time::Instant::now() < deadline,
                "second batch submission waited for GPU completion"
            );
            std::thread::yield_now();
        }

        // Both commands reached the driver while GPU execution is still gated: that
        // is the overlap the removed host wait serialized away. Deterministic, not
        // timing-dependent, because the gate holds every blit until release.
        assert!(!big_sub.completed().unwrap());
        gate.release();
        while !big_sub.completed().unwrap() || !small_sub.completed().unwrap() {
            assert!(
                std::time::Instant::now() < deadline,
                "transfers never completed after submission"
            );
            std::thread::yield_now();
        }
        worker.shutdown();
    }

    #[test]
    fn event_gate_holds_gpu_execution_until_host_signal() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let queue = device.newCommandQueue().unwrap();
        let gate_event = device.newSharedEvent().unwrap();
        let gate_buffer = queue.commandBuffer().unwrap();
        encode_gate_wait(&gate_buffer, &gate_event);
        gate_buffer.commit();
        let gate = EventGate { event: gate_event };
        // A blit queued behind the gate must stay scheduled, never completed.
        let source = device
            .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        // SAFETY: one RGBA8 pixel and one mip form a valid 2D descriptor.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                1,
                1,
                false,
            )
        };
        desc.setStorageMode(MTLStorageMode::Shared);
        let destination = device.newTextureWithDescriptor(&desc).unwrap();
        let blit = queue.commandBuffer().unwrap();
        let encoder = blit.blitCommandEncoder().unwrap();
        // SAFETY: the four shared source bytes and the 1x1 destination are live
        // through commit, with tight strides and a zero origin.
        unsafe {
            encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &source, 0, 4, 4,
                MTLSize { width: 1, height: 1, depth: 1 },
                &destination, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        encoder.endEncoding();
        blit.commit();
        std::thread::sleep(std::time::Duration::from_millis(500));
        assert_ne!(
            blit.status(),
            MTLCommandBufferStatus::Completed,
            "unsignaled event wait did not hold later work"
        );
        gate.release();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while blit.status() != MTLCommandBufferStatus::Completed {
            assert!(
                std::time::Instant::now() < deadline,
                "released gate never drained"
            );
            std::thread::yield_now();
        }
    }
}
