use ez_gfx_core::Backend;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// Identifies the stage of runtime processing recorded by an observation.
pub enum RuntimePhase {
    /// The request is being validated before processing.
    Admission = 1,
    /// Encoded input is being decoded.
    Decode = 2,
    /// Resource data is being transferred to the device.
    Upload = 3,
    /// Resources are being bound for execution.
    Bind = 4,
    /// Work is being submitted to the backend.
    Submit = 5,
    /// Rendered output is being presented.
    Present = 6,
    /// Device data is being copied back to the host.
    Readback = 7,
    /// Device-wide processing or state is involved.
    Device = 8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// Reports the outcome associated with a runtime observation.
pub enum RuntimeStatus {
    /// The operation completed successfully.
    Ok = 0,
    /// An argument failed validation.
    InvalidArgument = 1,
    /// Required data or state is not yet available.
    NotReady = 2,
    /// The requested operation is not supported.
    Unsupported = 3,
    /// The native graphics API reported a failure.
    NativeFailure = 4,
    /// The graphics device became unavailable.
    DeviceLost = 5,
    /// The asynchronous operation was cancelled before completion.
    Cancelled = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// Classifies the severity of a runtime diagnostic.
pub enum DiagnosticLevel {
    /// Informational runtime activity.
    Info = 1,
    /// A recoverable or potentially problematic condition.
    Warning = 2,
    /// A runtime failure requiring attention.
    Error = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Captures the context and outcome of one observed runtime operation.
pub struct RuntimeRecord {
    /// Correlates records produced by the same request.
    pub correlation_id: u64,
    /// Identifies the resource associated with the operation.
    pub resource: u64,
    /// Graphics backend that processed the operation.
    pub backend: Backend,
    /// Processing stage where the observation occurred.
    pub phase: RuntimePhase,
    /// Outcome reported for the operation.
    pub status: RuntimeStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Reports invalid observability configuration.
pub enum ObservabilityError {
    /// An event or diagnostic queue capacity was zero.
    InvalidCapacity,
}

/// Buffers runtime events and diagnostics in bounded FIFO queues.
pub struct Observability {
    /// Pending runtime records in arrival order.
    events: VecDeque<RuntimeRecord>,
    /// Pending diagnostics with their severity in arrival order.
    diagnostics: VecDeque<(DiagnosticLevel, RuntimeRecord)>,
    /// Maximum number of pending runtime records.
    event_capacity: usize,
    /// Maximum number of pending diagnostics.
    diagnostic_capacity: usize,
    /// Runtime records discarded since the previous event poll.
    event_dropped: u64,
    /// Diagnostics discarded since the previous diagnostic poll.
    diagnostic_dropped: u64,
    /// Correlation identifier to issue on the next request.
    next_correlation: u64,
}

impl Observability {
    /// Zero capacities are rejected; correlation wrap skips zero, which is reserved for host input without correlation.
    ///
    /// # Errors
    ///
    /// Returns `ObservabilityError::InvalidCapacity` if either queue capacity is zero.
    pub fn new(
        event_capacity: usize,
        diagnostic_capacity: usize,
    ) -> Result<Self, ObservabilityError> {
        if event_capacity == 0 || diagnostic_capacity == 0 {
            return Err(ObservabilityError::InvalidCapacity);
        }
        Ok(Self {
            events: VecDeque::with_capacity(event_capacity),
            diagnostics: VecDeque::with_capacity(diagnostic_capacity),
            event_capacity,
            diagnostic_capacity,
            event_dropped: 0,
            diagnostic_dropped: 0,
            next_correlation: 1,
        })
    }

    /// Returns a nonzero correlation identifier and advances the sequence with wrapping.
    pub fn next_correlation(&mut self) -> u64 {
        let current = self.next_correlation;
        self.next_correlation = self.next_correlation.wrapping_add(1).max(1);
        current
    }

    /// Overflow drops the newest record and saturates the loss counter rather than evicting causal history.
    pub fn push_event(&mut self, record: RuntimeRecord) {
        if self.events.len() == self.event_capacity {
            self.event_dropped = self.event_dropped.saturating_add(1);
        } else {
            self.events.push_back(record);
        }
    }

    /// Diagnostic overflow is independent from runtime-event overflow.
    pub fn push_diagnostic(&mut self, level: DiagnosticLevel, record: RuntimeRecord) {
        if self.diagnostics.len() == self.diagnostic_capacity {
            self.diagnostic_dropped = self.diagnostic_dropped.saturating_add(1);
        } else {
            self.diagnostics.push_back((level, record));
        }
    }

    /// The accumulated dropped count is returned once with the next poll, including an empty poll.
    pub fn poll_event(&mut self) -> (Option<RuntimeRecord>, u64) {
        (
            self.events.pop_front(),
            core::mem::take(&mut self.event_dropped),
        )
    }

    /// The accumulated dropped count is returned once with the next poll, including an empty poll.
    pub fn poll_diagnostic(&mut self) -> (Option<(DiagnosticLevel, RuntimeRecord)>, u64) {
        (
            self.diagnostics.pop_front(),
            core::mem::take(&mut self.diagnostic_dropped),
        )
    }
}
