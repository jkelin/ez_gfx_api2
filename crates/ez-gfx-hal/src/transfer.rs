use crate::{CompletionToken, StagingPolicy};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{RecvTimeoutError, SyncSender, TryRecvError, TrySendError, channel, sync_channel},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Error returned by a bounded transfer worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferWorkerError {
    /// The bounded request channel has no free slot.
    Full,
    /// The worker stopped or its submission handler failed.
    Failed,
}

enum TransferWorkerMessage<J> {
    Job(J),
    Flush(std::sync::mpsc::Sender<()>),
}

/// Dedicated owner thread that drains bounded transfer requests into adaptive batches.
pub struct TransferWorker<J> {
    sender: Option<SyncSender<TransferWorkerMessage<J>>>,
    failed: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl<J: Send + 'static> TransferWorker<J> {
    /// Starts one transfer-owner thread.
    ///
    /// # Errors
    ///
    /// Returns [`TransferWorkerError::Failed`] when `capacity` is zero or the thread cannot start.
    pub fn new(
        capacity: usize,
        policy: StagingPolicy,
        bytes: fn(&J) -> u64,
        submit: impl FnMut(Vec<J>) -> Result<(), TransferWorkerError> + Send + 'static,
    ) -> Result<Self, TransferWorkerError> {
        Self::new_grouped_with_shutdown(capacity, policy, bytes, |_| 0, submit, || Ok(()))
    }

    /// Starts one grouped transfer-owner thread with an explicit drain-completion hook.
    ///
    /// Adjacent jobs with different group keys are submitted in separate batches.
    ///
    /// # Errors
    ///
    /// Returns [`TransferWorkerError::Failed`] when `capacity` is zero or the thread cannot start.
    pub fn new_grouped_with_shutdown(
        capacity: usize,
        policy: StagingPolicy,
        bytes: fn(&J) -> u64,
        group: fn(&J) -> u64,
        mut submit: impl FnMut(Vec<J>) -> Result<(), TransferWorkerError> + Send + 'static,
        mut shutdown: impl FnMut() -> Result<(), TransferWorkerError> + Send + 'static,
    ) -> Result<Self, TransferWorkerError> {
        if capacity == 0 {
            return Err(TransferWorkerError::Failed);
        }
        let (sender, receiver) = sync_channel(capacity);
        let failed = Arc::new(AtomicBool::new(false));
        let worker_failed = failed.clone();
        let thread = thread::Builder::new()
            .name("ez-gfx-transfer".into())
            .spawn(move || {
                let mut carry = None;
                'worker: loop {
                    let first = match carry.take() {
                        Some(job) => job,
                        None => loop {
                            match receiver.recv() {
                                Ok(TransferWorkerMessage::Job(job)) => break job,
                                Ok(TransferWorkerMessage::Flush(ready)) => {
                                    let _ = ready.send(());
                                }
                                Err(_) => break 'worker,
                            }
                        },
                    };
                    let batch_group = group(&first);
                    let mut batch = vec![first];
                    let mut total = bytes(&batch[0]);
                    let deadline = Instant::now() + Duration::from_micros(200);
                    let mut flush = None;
                    while !policy.should_flush(batch.len(), total) {
                        let message = match receiver.try_recv() {
                            Ok(job) => Ok(job),
                            Err(TryRecvError::Disconnected) => break,
                            Err(TryRecvError::Empty) => {
                                let Some(remaining) =
                                    deadline.checked_duration_since(Instant::now())
                                else {
                                    break;
                                };
                                receiver
                                    .recv_timeout(remaining)
                                    .map_err(|error| match error {
                                        RecvTimeoutError::Timeout => TryRecvError::Empty,
                                        RecvTimeoutError::Disconnected => {
                                            TryRecvError::Disconnected
                                        }
                                    })
                            }
                        };
                        match message {
                            Ok(TransferWorkerMessage::Job(job)) if group(&job) == batch_group => {
                                total = total.saturating_add(bytes(&job));
                                batch.push(job);
                            }
                            Ok(TransferWorkerMessage::Job(job)) => {
                                carry = Some(job);
                                break;
                            }
                            Ok(TransferWorkerMessage::Flush(ready)) => {
                                flush = Some(ready);
                                break;
                            }
                            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                        }
                    }
                    if submit(batch).is_err() {
                        worker_failed.store(true, Ordering::Release);
                        return;
                    }
                    if let Some(ready) = flush {
                        let _ = ready.send(());
                    }
                }
                if shutdown().is_err() {
                    worker_failed.store(true, Ordering::Release);
                }
            })
            .map_err(|_| TransferWorkerError::Failed)?;
        Ok(Self {
            sender: Some(sender),
            failed,
            thread: Some(thread),
        })
    }

    /// Enqueues one request without waiting for worker progress.
    ///
    /// # Errors
    ///
    /// Returns `Full` for backpressure or `Failed` after worker failure/shutdown.
    pub fn submit(&self, job: J) -> Result<(), TransferWorkerError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(TransferWorkerError::Failed);
        }
        self.sender
            .as_ref()
            .ok_or(TransferWorkerError::Failed)?
            .try_send(TransferWorkerMessage::Job(job))
            .map_err(|error| match error {
                TrySendError::Full(_) => TransferWorkerError::Full,
                TrySendError::Disconnected(_) => TransferWorkerError::Failed,
            })
    }

    /// Blocks until all requests accepted before this call have been submitted.
    ///
    /// # Errors
    ///
    /// Returns [`TransferWorkerError::Failed`] if the owner has stopped.
    pub fn flush(&self) -> Result<(), TransferWorkerError> {
        if self.failed() {
            return Err(TransferWorkerError::Failed);
        }
        let (ready_tx, ready_rx) = channel();
        self.sender
            .as_ref()
            .ok_or(TransferWorkerError::Failed)?
            .send(TransferWorkerMessage::Flush(ready_tx))
            .map_err(|_| TransferWorkerError::Failed)?;
        ready_rx.recv().map_err(|_| TransferWorkerError::Failed)
    }

    /// Reports whether the submission handler failed.
    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Closes admission, drains accepted jobs, and joins the owner thread.
    pub fn shutdown(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            self.failed.store(true, Ordering::Release);
        }
    }
}

impl<J> Drop for TransferWorker<J> {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

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
    fn transfer_worker_reports_backpressure_without_waiting() {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut first = true;
        let mut worker = TransferWorker::new(
            1,
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
        worker.submit(2).unwrap();
        assert_eq!(worker.submit(3), Err(TransferWorkerError::Full));
        release_tx.send(()).unwrap();
        worker.shutdown();
    }

    #[test]
    fn transfer_worker_coalesces_ready_jobs() {
        let (batch_tx, batch_rx) = std::sync::mpsc::channel();
        let policy = StagingPolicy::new(1, 8, 8, 8).unwrap();
        let mut worker = TransferWorker::new(
            4,
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
            4,
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
}
