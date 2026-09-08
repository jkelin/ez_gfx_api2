use crate::StagingPolicy;
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Error returned by a bounded transfer worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferWorkerError {
    /// The bounded request channel has no free slot.
    Full,
    /// The worker stopped, the request is invalid, or submission failed.
    Failed,
    /// Native submission reported device loss; the worker is poisoned with the loss reason.
    DeviceLost,
}

impl TransferWorkerError {
    /// Maps a worker rejection onto the portable backend failure without collapsing loss.
    pub fn to_hal_error(self) -> crate::HalError {
        match self {
            TransferWorkerError::Full => crate::HalError::OutOfMemory,
            TransferWorkerError::Failed => crate::HalError::NativeFailure,
            TransferWorkerError::DeviceLost => crate::HalError::DeviceLost,
        }
    }

    /// Maps a worker rejection onto the allocation failure without collapsing loss.
    pub fn to_allocation_error(self) -> super::AllocationError {
        match self {
            TransferWorkerError::Full => super::AllocationError::OutOfMemory,
            TransferWorkerError::Failed => super::AllocationError::NativeFailure,
            TransferWorkerError::DeviceLost => super::AllocationError::DeviceLost,
        }
    }
}

enum Message<J> {
    Job(J),
    Batch(Vec<J>),
    Flush(std::sync::mpsc::Sender<()>),
}

#[derive(Default)]
struct SubmissionState {
    submitted: u64,
    stopped: bool,
    drained: bool,
}

#[derive(Default)]
struct SubmissionProgress {
    state: Mutex<SubmissionState>,
    changed: Condvar,
}

struct WorkerExit {
    progress: Arc<SubmissionProgress>,
    failed: Arc<AtomicBool>,
    lost: Arc<AtomicBool>,
    clean: bool,
}

impl Drop for WorkerExit {
    fn drop(&mut self) {
        // A panicking native callback must wake targeted flushes, not strand their callers.
        if !self.clean {
            self.failed.store(true, Ordering::Release);
        }
        let mut state = self
            .progress
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.stopped = true;
        self.progress.changed.notify_all();
    }
}

struct Inbox<J> {
    receiver: Receiver<Message<J>>,
    bundle: std::vec::IntoIter<J>,
    queued: Arc<AtomicUsize>,
}

impl<J> Inbox<J> {
    fn next(&mut self, deadline: Option<Instant>) -> Result<Message<J>, RecvTimeoutError> {
        // A bundle is admitted atomically but keeps its original FIFO and stage boundaries.
        loop {
            if let Some(job) = self.bundle.next() {
                self.queued.fetch_sub(1, Ordering::Release);
                return Ok(Message::Job(job));
            }
            let message = if let Some(deadline) = deadline {
                match self.receiver.try_recv() {
                    Ok(message) => message,
                    Err(TryRecvError::Disconnected) => return Err(RecvTimeoutError::Disconnected),
                    Err(TryRecvError::Empty) => {
                        let remaining = deadline
                            .checked_duration_since(Instant::now())
                            .ok_or(RecvTimeoutError::Timeout)?;
                        self.receiver.recv_timeout(remaining)?
                    }
                }
            } else {
                self.receiver
                    .recv()
                    .map_err(|_| RecvTimeoutError::Disconnected)?
            };
            match message {
                Message::Batch(jobs) => self.bundle = jobs.into_iter(),
                Message::Job(job) => {
                    self.queued.fetch_sub(1, Ordering::Release);
                    return Ok(Message::Job(job));
                }
                other @ Message::Flush(_) => return Ok(other),
            }
        }
    }
}

/// Dedicated owner thread that drains an unbounded request stream into adaptive native batches.
pub struct TransferWorker<J> {
    sender: Option<Sender<Message<J>>>,
    failed: Arc<AtomicBool>,
    lost: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    queued: Arc<AtomicUsize>,
    completion: fn(&J) -> u64,
    accepted: Mutex<u64>,
    progress: Arc<SubmissionProgress>,
}

impl<J: Send + 'static> TransferWorker<J> {
    /// Starts one transfer-owner thread.
    ///
    /// # Errors
    /// Returns `Failed` for thread creation failure.
    pub fn new(
        policy: StagingPolicy,
        bytes: fn(&J) -> u64,
        submit: impl FnMut(Vec<J>) -> Result<(), TransferWorkerError> + Send + 'static,
    ) -> Result<Self, TransferWorkerError> {
        Self::new_grouped_with_shutdown(policy, bytes, |_| 0, submit, || Ok(()))
    }

    /// Starts an owner that separates adjacent group keys and drains on shutdown.
    ///
    /// # Errors
    /// Returns `Failed` for thread creation failure.
    pub fn new_grouped_with_shutdown(
        policy: StagingPolicy,
        bytes: fn(&J) -> u64,
        group: fn(&J) -> u64,
        submit: impl FnMut(Vec<J>) -> Result<(), TransferWorkerError> + Send + 'static,
        shutdown: impl FnMut() -> Result<(), TransferWorkerError> + Send + 'static,
    ) -> Result<Self, TransferWorkerError> {
        Self::new_ordered_with_shutdown(policy, bytes, group, |_| 0, submit, shutdown)
    }

    /// Starts an owner with monotonically increasing submission tokens.
    ///
    /// `completion` returns zero for unordered work; ordered jobs must arrive in token order.
    /// The callback must return only after all batch jobs have been submitted or safely skipped.
    ///
    /// # Errors
    /// Returns `Failed` for thread creation failure.
    pub fn new_ordered_with_shutdown(
        policy: StagingPolicy,
        bytes: fn(&J) -> u64,
        group: fn(&J) -> u64,
        completion: fn(&J) -> u64,
        mut submit: impl FnMut(Vec<J>) -> Result<(), TransferWorkerError> + Send + 'static,
        mut shutdown: impl FnMut() -> Result<(), TransferWorkerError> + Send + 'static,
    ) -> Result<Self, TransferWorkerError> {
        let (sender, receiver) = channel();
        let failed = Arc::new(AtomicBool::new(false));
        let lost = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(SubmissionProgress::default());
        let worker_progress = progress.clone();
        let worker_failed = failed.clone();
        let worker_lost = lost.clone();
        let queued = Arc::new(AtomicUsize::new(0));
        let worker_queued = queued.clone();
        let thread = thread::Builder::new()
            .name("ez-gfx-transfer".into())
            .spawn(move || {
                let mut exit = WorkerExit {
                    progress: worker_progress,
                    failed: worker_failed,
                    lost: worker_lost,
                    clean: false,
                };
                let mut inbox = Inbox {
                    receiver,
                    bundle: Vec::new().into_iter(),
                    queued: worker_queued,
                };
                let mut carry = None;
                let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    'worker: loop {
                        let first = match carry.take() {
                            Some(job) => job,
                            None => loop {
                                match inbox.next(None) {
                                    Ok(Message::Job(job)) => break job,
                                    Ok(Message::Flush(ready)) => {
                                        let _ = ready.send(());
                                    }
                                    Ok(Message::Batch(_)) => unreachable!("inbox expands bundles"),
                                    Err(_) => break 'worker,
                                }
                            },
                        };
                        let batch_group = group(&first);
                        let mut total = bytes(&first);
                        let mut final_value = completion(&first);
                        let mut batch = vec![first];
                        let deadline = Instant::now() + Duration::from_micros(200);
                        let mut flush = None;
                        while !policy.should_flush(batch.len(), total) {
                            match inbox.next(Some(deadline)) {
                                Ok(Message::Job(job)) if group(&job) == batch_group => {
                                    total = total.saturating_add(bytes(&job));
                                    final_value = final_value.max(completion(&job));
                                    batch.push(job);
                                }
                                Ok(Message::Job(job)) => {
                                    carry = Some(job);
                                    break;
                                }
                                Ok(Message::Flush(ready)) => {
                                    flush = Some(ready);
                                    break;
                                }
                                Ok(Message::Batch(_)) => unreachable!("inbox expands bundles"),
                                Err(_) => break,
                            }
                        }
                        submit(batch)?;
                        {
                            let mut state = exit
                                .progress
                                .state
                                .lock()
                                .map_err(|_| TransferWorkerError::Failed)?;
                            state.submitted = state.submitted.max(final_value);
                            exit.progress.changed.notify_all();
                        }
                        if let Some(ready) = flush {
                            let _ = ready.send(());
                        }
                    }
                    Ok::<(), TransferWorkerError>(())
                }));
                let submission_failed = !matches!(submitted, Ok(Ok(())));
                // Only a native `DeviceLost` return poisons the latch with loss; a callback
                // panic stays a generic failure (its payload carries no loss reason).
                let submission_lost = matches!(submitted, Ok(Err(TransferWorkerError::DeviceLost)));
                if submission_lost {
                    // Sticky loss reason: later admissions report DeviceLost, not NativeFailure.
                    exit.lost.store(true, Ordering::Release);
                }
                if submission_failed {
                    exit.failed.store(true, Ordering::Release);
                    let _state = exit
                        .progress
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    exit.progress.changed.notify_all();
                }
                // Callback captures retain native allocators until cleanup drains even partial work.
                let drained = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut shutdown));
                let drain_failed = !matches!(drained, Ok(Ok(())));
                // A loss-reporting drain poisons the latch like a loss-reporting submit.
                if matches!(drained, Ok(Err(TransferWorkerError::DeviceLost))) {
                    exit.lost.store(true, Ordering::Release);
                }
                {
                    let mut state = exit
                        .progress
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.drained = !drain_failed;
                }
                if drain_failed {
                    // An undrained live queue may still consume captured command/storage resources.
                    // Retain only this failed owner's captures; normal shutdown releases everything.
                    std::mem::forget(submit);
                    std::mem::forget(shutdown);
                }
                exit.clean = !submission_failed && !drain_failed;
            })
            .map_err(|_| TransferWorkerError::Failed)?;
        Ok(Self {
            sender: Some(sender),
            failed,
            lost,
            thread: Some(thread),
            queued,
            completion,
            accepted: Mutex::new(0),
            progress,
        })
    }

    /// Reports the sticky admission failure: loss outranks generic failure.
    fn admission_error(&self) -> TransferWorkerError {
        if self.device_lost() {
            return TransferWorkerError::DeviceLost;
        }
        TransferWorkerError::Failed
    }

    fn send(
        &self,
        message: Message<J>,
        highwater: u64,
        accepted: &mut u64,
        count: usize,
    ) -> Result<(), TransferWorkerError> {
        // Rejected admission never advances the accepted watermark. The queue
        // itself is unbounded; only integer exhaustion or allocation failure can
        // prevent ownership from being retained after this point.
        if self.failed() {
            return Err(self.admission_error());
        }
        let sender = self.sender.as_ref().ok_or(TransferWorkerError::Failed)?;
        self.queued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                queued.checked_add(count)
            })
            .map_err(|_| TransferWorkerError::Full)?;
        if sender.send(message).is_err() {
            self.queued.fetch_sub(count, Ordering::Release);
            return Err(self.admission_error());
        }
        *accepted = highwater;
        Ok(())
    }

    /// Enqueues one request without waiting for worker progress.
    ///
    /// # Errors
    /// Returns `Failed` for stopped/invalid ordered admission, `Full` only for
    /// counter exhaustion, or `DeviceLost` after native device loss.
    pub fn submit(&self, job: J) -> Result<(), TransferWorkerError> {
        // Serialize watermark validation with channel admission when callers share the worker.
        let mut accepted = self
            .accepted
            .lock()
            .map_err(|_| TransferWorkerError::Failed)?;
        let value = (self.completion)(&job);
        if value != 0 && value <= *accepted {
            return Err(TransferWorkerError::Failed);
        }
        self.send(Message::Job(job), value.max(*accepted), &mut accepted, 1)
    }

    /// Atomically admits a FIFO bundle; native batches still split at stage and policy boundaries.
    ///
    /// # Errors
    /// Empty or out-of-order bundles return `Failed`; counter exhaustion returns
    /// `Full`; a loss-poisoned worker returns `DeviceLost`.
    pub fn submit_batch(&self, jobs: Vec<J>) -> Result<(), TransferWorkerError> {
        if jobs.is_empty() {
            return Err(TransferWorkerError::Failed);
        }
        let mut accepted = self
            .accepted
            .lock()
            .map_err(|_| TransferWorkerError::Failed)?;
        let mut highwater = *accepted;
        for job in &jobs {
            let value = (self.completion)(job);
            if value != 0 {
                if value <= highwater {
                    return Err(TransferWorkerError::Failed);
                }
                highwater = value;
            }
        }
        let count = jobs.len();
        self.send(Message::Batch(jobs), highwater, &mut accepted, count)
    }

    /// Waits until the callback finishes through an accepted token, without draining later jobs.
    /// Native callbacks may wait for a specific GPU copy before completing their handoff.
    ///
    /// # Errors
    /// Returns `Failed` for a token beyond accepted work, owner failure, or premature shutdown.
    /// A loss-poisoned worker returns `DeviceLost`.
    pub fn flush_through(&self, value: u64) -> Result<(), TransferWorkerError> {
        if value
            > *self
                .accepted
                .lock()
                .map_err(|_| TransferWorkerError::Failed)?
        {
            return Err(TransferWorkerError::Failed);
        }
        let mut state = self
            .progress
            .state
            .lock()
            .map_err(|_| TransferWorkerError::Failed)?;
        loop {
            if self.failed() {
                return Err(self.admission_error());
            }
            // Zero and already-submitted tokens require no queue message or drain.
            if state.submitted >= value {
                return Ok(());
            }
            if state.stopped {
                return Err(self.admission_error());
            }
            state = self
                .progress
                .changed
                .wait(state)
                .map_err(|_| TransferWorkerError::Failed)?;
        }
    }

    /// Blocks until all requests accepted before this call have been submitted.
    ///
    /// # Errors
    /// Returns `Failed` if the owner has stopped, or `DeviceLost` once loss has poisoned it.
    pub fn flush(&self) -> Result<(), TransferWorkerError> {
        if self.failed() {
            return Err(self.admission_error());
        }
        let (ready_tx, ready_rx) = channel();
        self.sender
            .as_ref()
            .ok_or(TransferWorkerError::Failed)?
            .send(Message::Flush(ready_tx))
            .map_err(|_| TransferWorkerError::Failed)?;
        ready_rx.recv().map_err(|_| self.admission_error())
    }

    /// Reports whether the submission handler failed for any reason, including device loss.
    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire) || self.device_lost()
    }

    /// Reports whether native submission failed with device loss.
    pub fn device_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Sticky terminal rejection for a failed worker, or `None` while healthy.
    /// Loss outranks generic failure so diagnostics name the cause.
    pub fn terminal_error(&self) -> Option<TransferWorkerError> {
        if self.device_lost() {
            Some(TransferWorkerError::DeviceLost)
        } else if self.failed.load(Ordering::Acquire) {
            Some(TransferWorkerError::Failed)
        } else {
            None
        }
    }

    /// Reports whether the stopped owner successfully drained all actual native submissions.
    /// A failed callback may still be drained; failed cleanup never implies safe reclamation.
    pub fn drained(&self) -> bool {
        self.progress
            .state
            .lock()
            .is_ok_and(|state| state.stopped && state.drained)
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
