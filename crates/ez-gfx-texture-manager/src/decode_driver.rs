//! Manager-owned CPU decode worker queue and Rayon lifecycle.
//!
//! [`DecodeDriver`] owns one context's decode execution: the validated worker
//! policy, the lazily built Rayon pool, the unbounded result channel, the
//! active-job count, and shutdown. [`TexturePipeline`](super::pipeline::TexturePipeline)
//! keeps owning every queue datum; the driver only pops admitted work,
//! reserves shared-transfer bytes atomically with the spawn, and collects
//! terminal results back into the pipeline. Worker closures capture decoded
//! bytes and decoder state only, never runtime thread-local state.

use super::pipeline::{TexturePipeline, collect_result};
use super::texture::{TextureError, generate_mips};
use super::{SharedTransferPool, TextureId};
use crossbeam_channel::{Receiver, Sender};
use ez_gfx_core::handle::TextureHandle;
use rayon::ThreadPool;
use std::panic::AssertUnwindSafe;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Instant;

/// Maximum worker threads admitted to one decode pool.
///
/// Rayon spawns one OS thread per worker eagerly at pool construction, so an
/// unbounded count grinds thread creation (stack reservation, scheduler load)
/// instead of failing fast. 256 threads already exceed twice the logical CPUs
/// of large commodity workstations, beyond which extra workers add only
/// overhead; any larger request is configuration garbage.
pub const MAX_CPU_POOL_THREADS: usize = 256;

/// Failures produced by the decode driver lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeDriverError {
    /// The requested worker count is zero after topology resolution, exceeds
    /// [`MAX_CPU_POOL_THREADS`], or overflows the platform thread count.
    InvalidWorkerCount,
    /// The lazily built Rayon pool could not spawn its OS threads.
    PoolUnavailable,
    /// The driver shut down; queued work is untouched and dispatch is refused.
    Shutdown,
    /// The in-flight job counter overflowed instead of admitting more work.
    QueueFull,
}

impl core::fmt::Display for DecodeDriverError {
    /// Formats the driver error using its debug name.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DecodeDriverError {}

/// One finished CPU decode traveling from a worker to the pipeline.
#[derive(Debug)]
pub struct DecodedTextureJob {
    /// Texture the result belongs to.
    pub handle: TextureHandle,
    /// Decoded payload or the terminal decode failure.
    pub decoded: Result<super::texture::DecodedTexture, TextureError>,
}

/// Outcome of one [`DecodeDriver::dispatch`] sweep.
#[derive(Debug, Default)]
pub struct DispatchReport {
    /// Jobs admitted to workers; each holds one transfer reservation.
    pub admitted: usize,
    /// Handles whose spawn failed after reservation; the caller fails each
    /// through its own terminal path and releases nothing further.
    pub failed: Vec<(TextureHandle, TextureId)>,
}

/// Tracks one accepted job until its worker closure finishes.
struct JobPermit {
    jobs: Arc<AtomicUsize>,
}

impl Drop for JobPermit {
    fn drop(&mut self) {
        self.jobs.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Decode-only Rayon pool with cancellation and in-flight accounting.
struct CpuPool {
    pool: ThreadPool,
    cancelled: Arc<AtomicBool>,
    jobs: Arc<AtomicUsize>,
}

impl CpuPool {
    /// Creates a worker pool with unbounded job admission.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeDriverError::InvalidWorkerCount`] when `threads` is
    /// zero or exceeds [`MAX_CPU_POOL_THREADS`], or
    /// [`DecodeDriverError::PoolUnavailable`] when the OS refuses threads.
    fn new(threads: usize) -> Result<Self, DecodeDriverError> {
        if threads == 0 || threads > MAX_CPU_POOL_THREADS {
            return Err(DecodeDriverError::InvalidWorkerCount);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|_| DecodeDriverError::PoolUnavailable)?;
        Ok(Self {
            pool,
            cancelled: Arc::new(AtomicBool::new(false)),
            jobs: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Returns the worker thread count backing this pool.
    fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Returns true once [`CpuPool::shutdown`] ran.
    fn is_shutdown(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn permit(&self) -> Result<JobPermit, DecodeDriverError> {
        self.jobs
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |jobs| {
                jobs.checked_add(1)
            })
            .map_err(|_| DecodeDriverError::QueueFull)?;
        Ok(JobPermit {
            jobs: self.jobs.clone(),
        })
    }

    /// Schedules a CPU job without an artificial count or byte limit.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeDriverError::Shutdown`] after shutdown or
    /// [`DecodeDriverError::QueueFull`] only if the in-flight counter overflows.
    fn submit<F>(&self, job: F) -> Result<(), DecodeDriverError>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.is_shutdown() {
            return Err(DecodeDriverError::Shutdown);
        }
        let permit = self.permit()?;
        let cancelled = self.cancelled.clone();
        self.pool.spawn(move || {
            let _permit = permit;
            if !cancelled.load(Ordering::Acquire) {
                job();
            }
        });
        Ok(())
    }

    /// Returns the number of accepted jobs that have not finished.
    #[cfg(test)]
    fn in_flight_jobs(&self) -> usize {
        self.jobs.load(Ordering::Acquire)
    }

    /// Cancels queued and future CPU work.
    fn shutdown(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// Owns the decode worker queue, Rayon lifecycle, and result channel.
pub struct DecodeDriver {
    /// Validated worker policy; the pool is built on first dispatch.
    threads: usize,
    /// Rayon pool built lazily so textureless contexts never spawn threads.
    pool: Option<CpuPool>,
    ready_tx: Sender<DecodedTextureJob>,
    ready_rx: Receiver<DecodedTextureJob>,
    active: usize,
    /// Sticky shutdown: set before the pool exists, dispatch refuses after it.
    shutdown_requested: bool,
    /// Test-only rendezvous held by every worker before decoding.
    decode_gate: Option<Arc<std::sync::Barrier>>,
}

impl DecodeDriver {
    /// Creates an unstarted driver from a validated worker request.
    ///
    /// Zero preserves the historical default topology
    /// (`available_parallelism - 1`, at least one); an explicit count is
    /// honored verbatim so embedders can pin decode concurrency. No threads
    /// exist until the first dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeDriverError::InvalidWorkerCount`] when an explicit
    /// count is zero-sized after conversion or exceeds
    /// [`MAX_CPU_POOL_THREADS`].
    pub fn new(workers: u32) -> Result<Self, DecodeDriverError> {
        let threads = if workers == 0 {
            std::thread::available_parallelism()
                .map_or(2, usize::from)
                .saturating_sub(1)
                .max(1)
        } else {
            let threads =
                usize::try_from(workers).map_err(|_| DecodeDriverError::InvalidWorkerCount)?;
            if threads > MAX_CPU_POOL_THREADS {
                return Err(DecodeDriverError::InvalidWorkerCount);
            }
            threads
        };
        let (ready_tx, ready_rx) = crossbeam_channel::unbounded();
        Ok(Self {
            threads,
            pool: None,
            ready_tx,
            ready_rx,
            active: 0,
            shutdown_requested: false,
            decode_gate: None,
        })
    }

    /// Builds the lazy pool before any admission that would need rollback.
    ///
    /// Callers invoke this before registering registry, identity, or pending
    /// state: late OS thread refusal then fails with nothing to unwind. The
    /// creator-thread model means no other thread can drop the pool between
    /// this call and submission.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeDriverError::PoolUnavailable`] when the OS refuses threads.
    pub fn ensure_started(&mut self) -> Result<(), DecodeDriverError> {
        if self.shutdown_requested {
            return Err(DecodeDriverError::Shutdown);
        }
        if self.pool.is_none() {
            self.pool = Some(CpuPool::new(self.threads)?);
        }
        Ok(())
    }

    /// Configured worker count; reports policy before the pool exists.
    pub fn worker_count(&self) -> usize {
        self.pool
            .as_ref()
            .map_or(self.threads, CpuPool::thread_count)
    }

    /// Returns true once the lazy pool has been built.
    pub fn is_started(&self) -> bool {
        self.pool.is_some()
    }

    /// Returns jobs admitted to workers without a collected result.
    pub const fn active_jobs(&self) -> usize {
        self.active
    }

    /// Installs the test-only worker rendezvous; `None` restores direct dispatch.
    #[doc(hidden)]
    pub fn set_decode_gate(&mut self, gate: Option<Arc<std::sync::Barrier>>) {
        self.decode_gate = gate;
    }

    /// Admits queued pipeline decodes to workers under the shared budget.
    ///
    /// Required-positive jobs dispatch first in queue order so an optional
    /// head never occupies the last worker while required work waits; each
    /// class keeps its own admission order. Preemption is intentionally
    /// absent: an already-running optional decode cannot be preempted and may
    /// delay a later required dispatch until that worker completes. Active
    /// decodes are collected at every pump, which the frame gate drives in a
    /// yield loop bounded by the 30s submission timeout, so the delay ends at
    /// completion or that timeout rather than growing without bound. A failed
    /// reservation requeues the job instead of dropping it, and a failed
    /// spawn releases the reservation and reports the handle for the caller's
    /// terminal path. Nothing is popped before the pool exists, so OS thread
    /// refusal leaves the queue and ledgers untouched.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeDriverError::PoolUnavailable`] when lazy pool
    /// construction fails, or [`DecodeDriverError::Shutdown`] after shutdown.
    pub fn dispatch(
        &mut self,
        pipe: &mut TexturePipeline,
        transfer: &mut SharedTransferPool,
    ) -> Result<DispatchReport, DecodeDriverError> {
        self.ensure_started()?;
        if self.shutdown_requested || self.pool.as_ref().is_some_and(CpuPool::is_shutdown) {
            return Err(DecodeDriverError::Shutdown);
        }
        let mut report = DispatchReport::default();
        let plan = TexturePipeline::decode_plan();
        while plan.admits(self.active, self.threads, transfer) && !pipe.queue().is_empty() {
            let Some(job) = pipe.pop_queued_prioritized() else {
                break;
            };
            // The shared pool counts geometry and buffer pressure too; a
            // failed admission requeues the job instead of dropping it.
            if transfer
                .acquire_required(super::DECODE_RESERVATION_BYTES)
                .is_err()
            {
                pipe.push_queued_front(job);
                break;
            }
            self.active = self.active.saturating_add(1);
            let handle = job.handle;
            if self.spawn(pipe, job).is_err() {
                self.active = self.active.saturating_sub(1);
                transfer.release_texture(super::DECODE_RESERVATION_BYTES);
                // Matches queue-full handling: the reservation unwinds and the
                // caller fails the pending slot through its terminal path.
                if let Some(pending) = pipe.remove_pending(handle) {
                    report.failed.push((handle, pending.id));
                }
            } else {
                report.admitted += 1;
            }
        }
        Ok(report)
    }

    /// Collects every ready result; stale jobs release their reservation credit.
    pub fn collect(
        &mut self,
        pipe: &mut TexturePipeline,
        transfer: &mut SharedTransferPool,
    ) -> usize {
        let mut collected = 0;
        while let Ok(job) = self.ready_rx.try_recv() {
            // Every collected result matches one prior admission; a zero here
            // means double collection, which stays saturating in release.
            debug_assert!(self.active > 0, "decode result without active job");
            self.active = self.active.saturating_sub(1);
            let release = collect_result(pipe, job.handle, job.decoded);
            if release > 0 {
                transfer.release_texture(release);
            }
            collected += 1;
        }
        collected
    }

    /// Cancels queued and future CPU work; sticky before the pool exists.
    pub fn shutdown(&mut self) {
        // An unbuilt pool owns no threads or jobs, so the flag alone refuses
        // later dispatch instead of constructing a pool only to cancel it.
        self.shutdown_requested = true;
        if let Some(pool) = self.pool.as_ref() {
            pool.shutdown();
        }
    }

    /// Spawns one reserved decode; the caller unwinds reservation and pending
    /// state when this fails.
    fn spawn(
        &self,
        pipe: &TexturePipeline,
        job: super::pipeline::QueuedDecode,
    ) -> Result<(), DecodeDriverError> {
        let Some(pool) = self.pool.as_ref() else {
            return Err(DecodeDriverError::PoolUnavailable);
        };
        let handle = job.handle;
        let cancelled = job.cancelled.clone();
        let telemetry = pipe.telemetry().clone();
        let ready = self.ready_tx.clone();
        let decode_gate = self.decode_gate.clone();
        pool.submit(move || {
            if let Some(gate) = decode_gate {
                gate.wait();
            }
            let started = Instant::now();
            let decoded = if cancelled.load(Ordering::Acquire) {
                Err(TextureError::NotFound)
            } else {
                std::panic::catch_unwind(AssertUnwindSafe(|| {
                    job.prepared.decode(&job.bytes).and_then(|texture| {
                        if job.generate {
                            generate_mips(texture)
                        } else {
                            Ok(texture)
                        }
                    })
                }))
                .unwrap_or(Err(TextureError::InvalidData))
            };
            telemetry
                .record_decode(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
            let _ = ready.send(DecodedTextureJob { handle, decoded });
        })
    }
}

impl Drop for DecodeDriver {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn pool_admission_returns_before_the_admitted_work_finishes() {
        let pool = CpuPool::new(1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        pool.submit(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
        started_rx.recv().unwrap();
        assert_eq!(pool.in_flight_jobs(), 1);
        release_tx.send(()).unwrap();
    }

    #[test]
    fn pool_rejects_absurd_thread_counts_without_spawning() {
        // The admission cap precedes ThreadPoolBuilder, so neither rejected
        // call creates threads.
        assert_eq!(
            CpuPool::new(usize::MAX).map(|_| ()),
            Err(DecodeDriverError::InvalidWorkerCount)
        );
        assert_eq!(
            CpuPool::new(MAX_CPU_POOL_THREADS + 1).map(|_| ()),
            Err(DecodeDriverError::InvalidWorkerCount)
        );
        assert_eq!(
            CpuPool::new(MAX_CPU_POOL_THREADS).map(|pool| pool.thread_count()),
            Ok(MAX_CPU_POOL_THREADS)
        );
    }

    #[test]
    fn pool_admits_more_jobs_than_worker_threads() {
        let pool = CpuPool::new(1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        pool.submit(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
        started_rx.recv().unwrap();
        for _ in 0..1_000 {
            pool.submit(|| {}).unwrap();
        }
        assert_eq!(pool.in_flight_jobs(), 1_001);
        release_tx.send(()).unwrap();
    }

    #[test]
    fn shutdown_rejects_submit_and_skips_unstarted_pool() {
        let idle = CpuPool::new(1).unwrap();
        idle.shutdown();
        assert_eq!(idle.submit(|| {}), Err(DecodeDriverError::Shutdown));
        // An unbuilt driver owns nothing to cancel.
        let mut unstarted = DecodeDriver::new(1).unwrap();
        unstarted.shutdown();
        assert!(!unstarted.is_started());
    }

    #[test]
    fn decode_gate_holds_workers_inside_the_worker_bound() {
        use super::super::pipeline::{PendingUpload, QueuedDecode};
        use super::super::texture::{
            TextureConfig, TextureDecoder, TextureDestination, TextureSource,
        };
        use ez_gfx_hal::{SamplerAddressMode, SamplerFilter, TextureSamplerDesc};

        fn admitted(
            pipe: &mut TexturePipeline,
            registry: &mut super::super::TextureRegistry,
            slot: u32,
        ) {
            let owner = ez_gfx_core::handle::LocalHandle::new(0, 1).unwrap();
            let child = ez_gfx_core::handle::LocalHandle::new(slot, 1).unwrap();
            let handle = TextureHandle::from_packed(
                ez_gfx_core::handle::PackedHandle::child(owner, child).unwrap(),
            )
            .unwrap();
            let cancelled = Arc::new(AtomicBool::new(false));
            let source = TextureSource::Rgba8 {
                width: 1,
                height: 1,
            };
            pipe.admit(
                handle,
                PendingUpload {
                    id: registry.begin_upload().unwrap(),
                    cancelled: cancelled.clone(),
                    config: TextureConfig {
                        source,
                        generate_mips: false,
                        required_mips: 1,
                        width: 1,
                        height: 1,
                        mip_count: 0,
                        destination: TextureDestination::Rgba8Unorm,
                        sampler: TextureSamplerDesc {
                            min_filter: SamplerFilter::Nearest,
                            mag_filter: SamplerFilter::Nearest,
                            max_anisotropy: 1.0,
                            address_u: SamplerAddressMode::Clamp,
                            address_v: SamplerAddressMode::Clamp,
                            address_w: SamplerAddressMode::Clamp,
                        },
                    },
                    fallback_published: false,
                    source_bytes: 4,
                    decoded_bytes: None,
                    admitted_at: Instant::now(),
                },
                QueuedDecode {
                    handle,
                    prepared: TextureDecoder::prepare(
                        source,
                        ez_gfx_core::capability::CompressionSupport::NONE,
                        TextureDestination::Rgba8Unorm,
                    )
                    .unwrap(),
                    bytes: vec![1, 2, 3, 255].into_boxed_slice(),
                    generate: false,
                    cancelled,
                },
            );
        }

        let mut driver = DecodeDriver::new(1).unwrap();
        let gate = Arc::new(std::sync::Barrier::new(2));
        driver.set_decode_gate(Some(gate.clone()));
        let mut pipe = TexturePipeline::new();
        let mut registry = super::super::TextureRegistry::new(8, 8).unwrap();
        let mut transfer = SharedTransferPool::new(super::super::WORKING_SET_BUDGET_BYTES);
        admitted(&mut pipe, &mut registry, 1);
        admitted(&mut pipe, &mut registry, 2);
        // One worker holds the first decode at the gate; the bound admits no
        // second job while it is active.
        let report = driver.dispatch(&mut pipe, &mut transfer).unwrap();
        assert_eq!((report.admitted, report.failed.len()), (1, 0));
        assert_eq!(driver.active_jobs(), 1);
        assert_eq!(pipe.queue().len(), 1);
        gate.wait();
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while pipe.decoded().is_empty() {
            driver.collect(&mut pipe, &mut transfer);
            assert!(Instant::now() < deadline, "gated decode timed out");
            std::thread::yield_now();
        }
        assert_eq!(driver.active_jobs(), 0);
    }
}
