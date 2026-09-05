use super::{AllocationError, MTLCommandBuffer, ProtocolObject, Retained};

/// A command buffer transferred exactly once to the dedicated submission thread.
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

// SAFETY: Metal command buffers support encoding on one thread and ordered commit/release on
// another; this wrapper exposes the object only to the dedicated transfer owner.
unsafe impl Send for TransferCommand {}

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
    ez_gfx_hal::TransferWorker::new(
        64,
        ez_gfx_hal::DEFAULT_STAGING_POLICY,
        job_bytes,
        move |jobs| {
            for job in jobs {
                if job.value <= last_value {
                    return Err(ez_gfx_hal::TransferWorkerError::Failed);
                }
                job.command.commit();
                last_value = job.value;
            }
            Ok(())
        },
    )
    .map_err(|_| AllocationError::NativeFailure)
}
