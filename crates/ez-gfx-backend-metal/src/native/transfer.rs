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
    ez_gfx_hal::TransferWorker::new_grouped_with_shutdown(
        64,
        ez_gfx_hal::DEFAULT_STAGING_POLICY,
        job_bytes,
        |_| 0,
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
    ez_gfx_hal::TransferWorker::new_ordered_with_shutdown(
        64,
        ez_gfx_hal::DEFAULT_STAGING_POLICY,
        |job: &TextureTransferJob| job.bytes,
        |job| job.stage,
        |job| job.value,
        move |mut jobs| {
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
            // Publish every observer before commit: a poisoned observer cannot orphan live work.
            for job in &jobs {
                *job.submission.command.lock().map_err(|_| ez_gfx_hal::TransferWorkerError::Failed)? =
                    Some(TransferCommand::new(command.clone()));
            }
            submitted.retain_before_commit(&command);
            command.commit();
            // GPU failure must reach flush_through before graphics waits on its event.
            // Waiting on this dedicated owner preserves independent graphics progress.
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(ez_gfx_hal::TransferWorkerError::Failed);
            }
            Ok(())
        },
        move || shutdown.drain(),
    ).map_err(|_| AllocationError::NativeFailure)
}

#[cfg(test)]
mod tests {
    use super::super::{
        MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat, MTLResourceOptions,
        MTLStorageMode, MTLTextureDescriptor,
    };
    use super::*;

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
                    value: 1,
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
                core::ptr::copy_nonoverlapping(pixel.as_ptr(), source.contents().as_ptr().cast(), 4)
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
        // Flush must prove GPU completion, not merely commit, before graphics can wait.
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
                &*first.as_ref().unwrap().command,
                &*second.as_ref().unwrap().command,
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
}
