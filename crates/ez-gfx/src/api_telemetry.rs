use super::{Context, ReadbackId, state};
use crate::Result;

/// Creator-thread event delivered by `Context::register_callback`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Event<'a> {
    /// One asynchronous upload state transition.
    Upload(ez_gfx_runtime::upload::UploadEvent),
    /// One runtime observation.
    Runtime(ez_gfx_runtime::observability::RuntimeRecord),
    /// One diagnostic observation.
    Diagnostic {
        /// Diagnostic severity.
        level: ez_gfx_runtime::observability::DiagnosticLevel,
        /// Runtime operation that produced the diagnostic.
        record: ez_gfx_runtime::observability::RuntimeRecord,
    },
    /// Bounded observability storage discarded records.
    ObservationsDropped(u64),
    /// Completed requested readback; bytes are valid only for this callback.
    Readback {
        /// Request identity.
        request: ReadbackId,
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
        /// Tightly packed RGBA8 pixels.
        bytes: &'a [u8],
    },
    /// Completed implicit terminal-frame snapshot.
    Snapshot(&'a [u8]),
}

/// Point-in-time pending-upload counts with retained bytes plus retained cache sizes.
///
/// Unlike [`Context::texture_upload_telemetry`], which reports monotonic pipeline
/// counters, this snapshot describes what is outstanding right now. Counts are
/// outstanding upload allocations and bytes accumulate with saturation, never
/// wrapping. Vertex and index uploads each carry the byte size reserved by the
/// original upload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceDiagnostics {
    /// Texture uploads queued for decode, holding decoded output, or awaiting final transfer.
    pub pending_textures: u32,
    /// Owned source bytes before decode plus decoded bytes after decode.
    pub pending_texture_bytes: u64,
    /// Vertex uploads awaiting transfer completion.
    pub pending_vertex_uploads: u32,
    /// Vertex bytes awaiting transfer completion.
    pub pending_vertex_bytes: u64,
    /// Index uploads awaiting transfer completion.
    pub pending_index_uploads: u32,
    /// Index bytes awaiting transfer completion.
    pub pending_index_bytes: u64,
    /// Staging buckets retained across the shared, buffer, and counter pools.
    pub staging_buckets: u32,
    /// Staging bucket capacity retained across those pools.
    pub staging_bytes: u64,
    /// Compiled pipeline entries retained in the context cache.
    pub pipeline_entries: u32,
    /// Bytes retained across completed readback frames.
    pub readback_bytes: u64,
}

/// On-demand allocator and memory telemetry; zero means unknown at context level.
///
/// Backend allocators report through `generate_report` exactly once per query,
/// so query explicitly and never per frame. This is a safe-Rust observation
/// only and is not part of the C ABI.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryTelemetryReport {
    /// Backend allocator, swapchain, depth, and slot telemetry.
    pub backend: ez_gfx_hal::BackendMemoryTelemetry,
    /// Staging buckets retained across the shared, buffer, and counter pools.
    pub staging_buckets: u32,
    /// Staging bucket capacity retained across those pools.
    pub staging_bytes: u64,
    /// Peak staging capacity retained across those pools, recorded at staging
    /// mutation boundaries and read here without side effects.
    pub staging_high_water: u64,
    /// Counter serialization capacity retained outside the staging pools.
    pub counter_scratch_bytes: u64,
    /// Async texture decode workers sized at creation.
    pub decode_workers: u32,
}

impl Context {
    /// Returns on-demand allocator and memory telemetry without dispatching events.
    ///
    /// Backend allocators report through `generate_report` exactly once per
    /// query, so query explicitly and never per frame. Unlike
    /// [`Context::resource_diagnostics`], this never dispatches callbacks.
    ///
    /// # Errors
    /// Returns [`crate::Error`] when the context is stale, called from the wrong
    /// thread, or reentered from a callback.
    pub fn memory_telemetry(&self) -> Result<MemoryTelemetryReport> {
        self.check_entry()?;
        state::memory_telemetry(self.raw())
    }

    /// Releases retained staging caches down to their finite budgets.
    ///
    /// Trims the shared, buffer, and counter staging pools plus excess counter
    /// serialization capacity, freeing evicted buckets natively. Buckets owned
    /// by in-flight GPU work stay retained. Call on memory pressure or after
    /// large streaming bursts — never per frame.
    /// # Errors
    /// Returns [`crate::Error`] when the context is stale, called from the wrong
    /// thread, or reentered from a callback.
    pub fn release_staging_memory(&self) -> Result<()> {
        self.check_entry()?;
        state::release_staging_memory(self.raw())
    }
}
