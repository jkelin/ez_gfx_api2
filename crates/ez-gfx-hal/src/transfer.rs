use crate::CompletionToken;
use core::fmt;

mod worker;
pub use worker::{TransferWorker, TransferWorkerError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Failures produced while allocating, mapping, transferring, or retiring backend memory.
pub enum AllocationError {
    /// The requested allocation size is zero.
    ZeroSize,
    /// The requested alignment is not a nonzero power of two.
    InvalidAlignment,
    /// Mapping was requested for memory that is not host-visible.
    NotHostVisible,
    /// The alias class is missing, zero, or assigned to non-transient memory.
    InvalidAliasClass,
    /// No suitable memory remains for the allocation.
    OutOfMemory,
    /// The device became unavailable during allocation.
    DeviceLost,
    /// The native allocator reported an unclassified failure.
    NativeFailure,
    /// The backend cannot support the requested resource configuration.
    Unsupported,
}

impl fmt::Display for AllocationError {
    /// Formats the allocation error using its debug name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for AllocationError {}

/// One reusable host-visible staging allocation.
pub struct StagingEntry<T> {
    /// Bucket capacity in bytes.
    pub capacity: u64,
    /// Backend allocation retained by the pool.
    pub allocation: T,
    retirement: Option<CompletionToken>,
    last_used: u64,
}

/// Backend-neutral state for best-fit staging reuse and idle trimming.
pub struct ReusableStagingPool<T> {
    entries: Vec<StagingEntry<T>>,
    epoch: u64,
    idle_epochs: u64,
}

impl<T> ReusableStagingPool<T> {
    /// Creates an empty pool. `idle_epochs` is clamped to one.
    pub const fn new(idle_epochs: u64) -> Self {
        Self {
            entries: Vec::new(),
            epoch: 0,
            idle_epochs: if idle_epochs == 0 { 1 } else { idle_epochs },
        }
    }

    /// Removes and returns the smallest completed bucket containing `size`.
    pub fn take(&mut self, size: u64, completed: u64) -> Option<(u64, T)> {
        self.epoch = self.epoch.saturating_add(1);
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.capacity >= size
                    && entry
                        .retirement
                        .is_none_or(|token| token.value <= completed)
            })
            .min_by_key(|(_, entry)| entry.capacity)
            .map(|(index, _)| index)?;
        let entry = self.entries.swap_remove(index);
        Some((entry.capacity, entry.allocation))
    }

    /// Returns an allocation to the pool, optionally pending timeline completion.
    pub fn put(&mut self, capacity: u64, allocation: T, retirement: Option<CompletionToken>) {
        self.epoch = self.epoch.saturating_add(1);
        self.entries.push(StagingEntry {
            capacity,
            allocation,
            retirement,
            last_used: self.epoch,
        });
    }

    /// Removes completed buckets unused for the configured number of epochs.
    pub fn trim(&mut self, completed: u64) -> Vec<T> {
        let epoch = self.epoch;
        let idle_epochs = self.idle_epochs;
        let mut removed = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            let entry = &self.entries[index];
            let completed = entry
                .retirement
                .is_none_or(|token| token.value <= completed);
            if completed && epoch.saturating_sub(entry.last_used) >= idle_epochs {
                removed.push(self.entries.swap_remove(index).allocation);
            } else {
                index += 1;
            }
        }
        removed
    }

    /// Number of retained buckets.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Total bucket capacity retained across every staging entry.
    ///
    /// Capacities accumulate with saturation: a diagnostic total must never wrap.
    pub fn retained_bytes(&self) -> u64 {
        // Buckets are reused whole, so retained capacity (not live occupancy) is the
        // honest cache size; per-entry capacities cannot overflow `u64` when saturated.
        self.entries
            .iter()
            .fold(0_u64, |total, entry| total.saturating_add(entry.capacity))
    }

    /// Reports whether no bucket is retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Removes every retained allocation.
    pub fn drain(&mut self) -> Vec<T> {
        self.entries
            .drain(..)
            .map(|entry| entry.allocation)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QueueKind;
    use crate::StagingPolicy;
    use std::{
        sync::{Arc, mpsc::channel},
        thread,
        time::Duration,
    };

    struct ReleaseOnDrop(std::sync::mpsc::Sender<()>);

    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    #[test]
    fn staging_pool_uses_best_completed_fit_and_trims_idle_buckets() {
        let pending = CompletionToken::new(QueueKind::Transfer, 3).unwrap();
        let mut pool = ReusableStagingPool::new(2);
        pool.put(64, "small", Some(pending));
        pool.put(128, "large", None);

        assert_eq!(pool.take(32, 2), Some((128, "large")));
        pool.put(128, "large", None);
        assert_eq!(pool.take(32, 3), Some((64, "small")));
        pool.put(64, "small", None);

        assert_eq!(pool.take(1024, 3), None);
        assert_eq!(pool.take(1024, 3), None);
        let mut trimmed = pool.trim(3);
        trimmed.sort_unstable();
        assert_eq!(trimmed, ["large", "small"]);
        assert!(pool.is_empty());
    }

    #[test]
    fn transfer_worker_admits_without_a_fixed_channel_limit() {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut first = true;
        let mut worker = TransferWorker::new(
            StagingPolicy::new(1, 8, 1, 1).unwrap(),
            |_| 1,
            move |_| {
                if first {
                    first = false;
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                }
                Ok(())
            },
        )
        .unwrap();

        worker.submit(1).unwrap();
        started_rx.recv().unwrap();
        for job in 2..=1_000 {
            worker.submit(job).unwrap();
        }
        release_tx.send(()).unwrap();
        worker.shutdown();
    }

    #[test]
    fn transfer_worker_coalesces_ready_jobs() {
        let (batch_tx, batch_rx) = std::sync::mpsc::channel();
        let policy = StagingPolicy::new(1, 8, 8, 8).unwrap();
        let mut worker = TransferWorker::new(
            policy,
            |_| 1,
            move |jobs| {
                batch_tx.send(jobs).unwrap();
                Ok(())
            },
        )
        .unwrap();

        worker.submit(1).unwrap();
        worker.submit(2).unwrap();
        worker.shutdown();
        assert_eq!(batch_rx.recv().unwrap(), vec![1, 2]);
        assert!(batch_rx.try_recv().is_err());
    }

    #[test]
    fn transfer_worker_separates_adjacent_groups_and_runs_shutdown_hook() {
        let (batch_tx, batch_rx) = std::sync::mpsc::channel();
        let shutdown_tx = batch_tx.clone();
        let policy = StagingPolicy::new(1, 8, 8, 8).unwrap();
        let mut worker = TransferWorker::new_grouped_with_shutdown(
            policy,
            |_| 1,
            |job: &u64| job % 2,
            move |jobs| {
                batch_tx.send(jobs).unwrap();
                Ok(())
            },
            move || {
                shutdown_tx.send(vec![99]).unwrap();
                Ok(())
            },
        )
        .unwrap();

        worker.submit(2).unwrap();
        worker.submit(4).unwrap();
        worker.submit(3).unwrap();
        worker.flush().unwrap();
        assert_eq!(batch_rx.recv().unwrap(), vec![2, 4]);
        assert_eq!(batch_rx.recv().unwrap(), vec![3]);
        worker.shutdown();
        assert_eq!(batch_rx.recv().unwrap(), vec![99]);
    }

    #[test]
    fn atomic_texture_bundle_batches_equal_stages_without_reordering() {
        let (sent, received) = channel();
        let mut worker = TransferWorker::new_grouped_with_shutdown(
            StagingPolicy::new(1, 8, 8, 8).unwrap(),
            |_| 1,
            |job: &(u64, u64)| job.1,
            move |jobs| {
                sent.send(jobs).unwrap();
                Ok(())
            },
            || Ok(()),
        )
        .unwrap();
        worker
            .submit_batch(vec![(1, 0), (2, 0), (1, 1), (3, 0)])
            .unwrap();
        worker.shutdown();
        assert_eq!(
            received.into_iter().collect::<Vec<_>>(),
            [vec![(1, 0), (2, 0)], vec![(1, 1)], vec![(3, 0)],]
        );
    }

    #[test]
    fn targeted_flush_wakes_a_partial_batch_before_following_work() {
        let (sent, received) = channel();
        let mut worker = TransferWorker::new_ordered_with_shutdown(
            StagingPolicy::new(1, 8, 8, 8).unwrap(),
            |_| 1,
            |_| 1,
            |value: &u64| *value,
            move |jobs| {
                sent.send(jobs).unwrap();
                Ok(())
            },
            || Ok(()),
        )
        .unwrap();

        worker.submit(1).unwrap();
        worker.flush_through(1).unwrap();
        worker.submit(2).unwrap();
        worker.shutdown();

        assert_eq!(received.into_iter().collect::<Vec<_>>(), [vec![1], vec![2]]);
    }

    #[test]
    fn ordered_flush_does_not_drain_a_later_blocked_stage() {
        let (started, reached) = channel();
        let (release, gate) = channel();
        let worker = Arc::new(
            TransferWorker::new_ordered_with_shutdown(
                StagingPolicy::new(1, 8, 1, 1).unwrap(),
                |_| 1,
                |value: &u64| *value,
                |value| *value,
                move |jobs| {
                    if jobs == [2] {
                        started.send(()).unwrap();
                        gate.recv().unwrap();
                    }
                    Ok(())
                },
                || Ok(()),
            )
            .unwrap(),
        );
        let release = ReleaseOnDrop(release);
        worker.submit_batch(vec![1, 2]).unwrap();
        reached.recv_timeout(Duration::from_secs(2)).unwrap();
        let (sent, received) = channel();
        let flushing = worker.clone();
        let join = thread::spawn(move || sent.send(flushing.flush_through(1)).unwrap());
        let result = received.recv_timeout(Duration::from_secs(2));
        // Release before asserting: a failure must never strand the transfer-owner thread.
        drop(release);
        join.join().unwrap();
        assert_eq!(result.unwrap(), Ok(()));
        assert_eq!(worker.flush_through(3), Err(TransferWorkerError::Failed));
        worker.flush_through(2).unwrap();
    }

    #[test]
    fn ordered_admission_rejects_invalid_bundles_without_advancing_watermark() {
        let (sent, received) = channel();
        let mut worker = TransferWorker::new_ordered_with_shutdown(
            StagingPolicy::new(1, 8, 8, 8).unwrap(),
            |_| 1,
            |_| 1,
            |value: &u64| *value,
            move |jobs| {
                sent.send(jobs).unwrap();
                Ok(())
            },
            || Ok(()),
        )
        .unwrap();
        assert_eq!(
            worker.submit_batch(Vec::new()),
            Err(TransferWorkerError::Failed)
        );
        // Admission has no artificial bundle-size limit.
        assert_eq!(
            worker.submit_batch(vec![3, 2]),
            Err(TransferWorkerError::Failed)
        );
        assert_eq!(worker.flush_through(1), Err(TransferWorkerError::Failed));
        worker.submit_batch(vec![1, 3]).unwrap();
        assert_eq!(worker.submit(2), Err(TransferWorkerError::Failed));
        worker.flush_through(2).unwrap();
        worker.flush_through(3).unwrap();
        worker.shutdown();
        assert_eq!(received.into_iter().flatten().collect::<Vec<_>>(), [1, 3]);
    }

    #[test]
    fn ordered_flush_observes_submission_failure_and_callback_panic() {
        for panic in [false, true] {
            let (drained, cleanup) = channel();
            let worker = TransferWorker::new_ordered_with_shutdown(
                StagingPolicy::new(1, 8, 1, 1).unwrap(),
                |_| 1,
                |_| 1,
                |value: &u64| *value,
                move |_| {
                    assert!(!panic, "injected native callback panic");
                    Err(TransferWorkerError::Failed)
                },
                move || {
                    drained.send(()).unwrap();
                    Ok(())
                },
            )
            .unwrap();
            worker.submit(1).unwrap();
            assert_eq!(worker.flush_through(1), Err(TransferWorkerError::Failed));
            cleanup
                .recv_timeout(Duration::from_secs(2))
                .expect("native shutdown must run after failed submission");
        }
    }
    #[test]
    fn device_loss_poison_reports_device_lost_not_generic_failure() {
        let worker = TransferWorker::new_ordered_with_shutdown(
            StagingPolicy::new(1, 8, 1, 1).unwrap(),
            |_| 1,
            |_| 1,
            |value: &u64| *value,
            move |_| Err(TransferWorkerError::DeviceLost),
            || Ok(()),
        )
        .unwrap();
        worker.submit(1).unwrap();
        assert_eq!(
            worker.flush_through(1),
            Err(TransferWorkerError::DeviceLost)
        );
        assert!(worker.device_lost());
        assert!(worker.failed());
        assert_eq!(worker.submit(2), Err(TransferWorkerError::DeviceLost));
        assert_eq!(worker.flush(), Err(TransferWorkerError::DeviceLost));
    }

    #[test]
    fn bundles_have_no_artificial_job_credit_limit() {
        let (started, reached) = channel();
        let (release, gate) = channel();
        let (sent, received) = channel();
        let mut worker = TransferWorker::new_ordered_with_shutdown(
            StagingPolicy::new(1, 8, 1, 1).unwrap(),
            |_| 1,
            |_| 1,
            |value: &u64| *value,
            move |jobs| {
                if jobs == [1] {
                    started.send(()).unwrap();
                    gate.recv().unwrap();
                }
                sent.send(jobs).unwrap();
                Ok(())
            },
            || Ok(()),
        )
        .unwrap();
        let release = ReleaseOnDrop(release);
        worker.submit(1).unwrap();
        reached.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.submit_batch(vec![2, 3, 4]).unwrap();
        worker.submit_batch(vec![5, 6]).unwrap();
        worker.submit(7).unwrap();
        drop(release);
        worker.shutdown();
        assert_eq!(
            received.into_iter().flatten().collect::<Vec<_>>(),
            [1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn trailing_unordered_job_preserves_ordered_batch_completion() {
        let mut worker = TransferWorker::new_ordered_with_shutdown(
            StagingPolicy::new(1, 8, 8, 8).unwrap(),
            |_| 1,
            |_| 1,
            |value: &u64| *value,
            |_| Ok(()),
            || Ok(()),
        )
        .unwrap();
        worker.submit_batch(vec![3, 0]).unwrap();
        worker.shutdown();
        assert_eq!(worker.flush_through(3), Ok(()));
    }

    struct DropProbe(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    #[test]
    fn failed_native_drain_retains_captures_but_normal_shutdown_releases_them() {
        for failure in [0, 1, 2] {
            let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let resource = DropProbe(dropped.clone());
            let mut worker = TransferWorker::new_ordered_with_shutdown(
                StagingPolicy::new(1, 8, 1, 1).unwrap(),
                |_| 1,
                |_| 1,
                |value: &u64| *value,
                move |_| {
                    let _keep_alive = &resource;
                    Ok(())
                },
                move || match failure {
                    0 => Ok(()),
                    1 => Err(TransferWorkerError::Failed),
                    _ => panic!("injected native cleanup panic"),
                },
            )
            .unwrap();
            worker.submit(1).unwrap();
            worker.shutdown();
            assert_eq!(worker.drained(), failure == 0);
            assert_eq!(worker.failed(), failure != 0);
            assert_eq!(
                dropped.load(std::sync::atomic::Ordering::Acquire),
                failure == 0
            );
        }
    }
}
