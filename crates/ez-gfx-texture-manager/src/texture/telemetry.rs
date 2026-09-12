//! Lock-free texture upload telemetry.

use std::sync::atomic::{AtomicU64, Ordering};

/// Cumulative lock-free texture upload counters.
#[derive(Debug, Default)]
pub struct TextureUploadTelemetry {
    decode_microseconds: AtomicU64,
    staging_bytes: AtomicU64,
    queue_latency_microseconds: AtomicU64,
    handoff_latency_microseconds: AtomicU64,
}

/// One point-in-time texture upload telemetry sample.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextureUploadTelemetrySnapshot {
    /// Total CPU decode time.
    pub decode_microseconds: u64,
    /// Total bytes copied into upload staging.
    pub staging_bytes: u64,
    /// Total latency from admission to native submission.
    pub queue_latency_microseconds: u64,
    /// Total latency from native submission to first sample-ready handoff.
    pub handoff_latency_microseconds: u64,
}

impl TextureUploadTelemetry {
    /// Adds one decode duration. Counter overflow saturates rather than wrapping.
    pub fn record_decode(&self, microseconds: u64) {
        saturating_fetch_add(&self.decode_microseconds, microseconds);
    }

    /// Adds bytes copied into upload staging. Counter overflow saturates rather than wrapping.
    pub fn record_staging_bytes(&self, bytes: u64) {
        saturating_fetch_add(&self.staging_bytes, bytes);
    }

    /// Adds latency between queue admission and native submission.
    pub fn record_queue_latency(&self, microseconds: u64) {
        saturating_fetch_add(&self.queue_latency_microseconds, microseconds);
    }

    /// Adds latency between native submission and graphics sample readiness.
    pub fn record_handoff_latency(&self, microseconds: u64) {
        saturating_fetch_add(&self.handoff_latency_microseconds, microseconds);
    }

    /// Reads every counter without locks.
    pub fn snapshot(&self) -> TextureUploadTelemetrySnapshot {
        TextureUploadTelemetrySnapshot {
            decode_microseconds: self.decode_microseconds.load(Ordering::Relaxed),
            staging_bytes: self.staging_bytes.load(Ordering::Relaxed),
            queue_latency_microseconds: self.queue_latency_microseconds.load(Ordering::Relaxed),
            handoff_latency_microseconds: self.handoff_latency_microseconds.load(Ordering::Relaxed),
        }
    }
}

fn saturating_fetch_add(counter: &AtomicU64, value: u64) {
    // Concurrent overflow is folded into the same terminal saturated value.
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}
