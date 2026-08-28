use ez_gfx_core::Backend;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RuntimePhase {
    Admission = 1,
    Decode = 2,
    Upload = 3,
    Bind = 4,
    Submit = 5,
    Present = 6,
    Readback = 7,
    Device = 8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RuntimeStatus {
    Ok = 0,
    InvalidArgument = 1,
    NotReady = 2,
    Unsupported = 3,
    NativeFailure = 4,
    DeviceLost = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DiagnosticLevel {
    Info = 1,
    Warning = 2,
    Error = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeRecord {
    pub correlation_id: u64,
    pub resource: u64,
    pub backend: Backend,
    pub phase: RuntimePhase,
    pub status: RuntimeStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservabilityError {
    InvalidCapacity,
}

pub struct Observability {
    events: VecDeque<RuntimeRecord>,
    diagnostics: VecDeque<(DiagnosticLevel, RuntimeRecord)>,
    event_capacity: usize,
    diagnostic_capacity: usize,
    event_dropped: u64,
    diagnostic_dropped: u64,
    next_correlation: u64,
}

impl Observability {
    /// Zero capacities are rejected; correlation wrap skips zero, which is reserved for host input without correlation.
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
