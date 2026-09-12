//! Backend-neutral texture upload pipeline: admission state and generic transitions.
//!
//! [`TexturePipeline`] owns every backend-neutral upload datum: the tissue of
//! pending decodes, FIFO order, decoded payloads, submitted residency,
//! retained fine mips, publication progress, transfer ledgers, and telemetry.
//! Native textures stay outside in a caller map keyed by the same handle, and
//! every transition that touches them is a generic function over
//! [`TextureBackendContext`], so backends implement the traits once and `ez-gfx`
//! links the selected implementation with static dispatch. The shared
//! [`SharedTransferPool`] is never owned here; callers pass their single
//! instance down so texture, geometry, and buffer uploads share one domain.
//!
//! Fallible transitions report [`UploadFailure`] and structured outcomes; the
//! caller maps failures to its own error type and emits its own events. This
//! crate never names devices, frames, observers, or upload queues.

mod submit;
pub use submit::{Rollback, SubmitOutcome, submit_ready_uploads};

use super::texture::{
    DecodedMip, DecodedTexture, PreparedTextureDecode, TextureConfig, TextureUploadTelemetry,
};
use super::{
    BatchPlan, MipTransferValues, PrefixTransferWork, SharedTransferPool, TextureBackendContext,
    TextureError, TextureId, TextureRegistry, TextureScheduleError, mip_range_completion,
};
use ez_gfx_core::handle::TextureHandle;
use ez_gfx_hal::{AllocationError, CompletionToken, TextureFormat, TextureRegion};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

/// Maximum total byte size reserved per active decode while the true output
/// size is unknown. Mirrors the ingestion limit so one admission domain
/// covers encoded, decoded, and native-transfer bytes alike.
pub const DECODE_RESERVATION_BYTES: u64 = super::texture::MAX_TEXTURE_BYTES as u64;

/// Admission window: four worst-case requests preserve useful decode
/// parallelism while bounding active payloads.
pub const WORKING_SET_BUDGET_BYTES: u64 = 4 * DECODE_RESERVATION_BYTES;

/// One admitted but not yet submitted texture load.
#[derive(Clone, Debug)]
pub struct PendingUpload {
    /// Registry slot reserved at admission.
    pub id: TextureId,
    /// Cooperative cancellation shared with the decode worker.
    pub cancelled: Arc<AtomicBool>,
    /// Admission contract; the decoded total validates it at submission.
    pub config: TextureConfig,
    /// True after this reserved slot aliases the ready context fallback.
    pub fallback_published: bool,
    /// Source bytes owned by queued storage or an active decode closure.
    pub source_bytes: u64,
    /// Decoded payload bytes once a decode result arrives.
    pub decoded_bytes: Option<u64>,
    /// Admission instant for queue-latency telemetry.
    pub admitted_at: Instant,
}

/// One queued CPU decode with its snapshotted decoder state.
#[derive(Clone)]
pub struct QueuedDecode {
    /// Texture the result belongs to.
    pub handle: TextureHandle,
    /// Decoder state captured at admission.
    pub prepared: PreparedTextureDecode,
    /// Owned source bytes moved into the worker.
    pub bytes: Box<[u8]>,
    /// Whether to generate the full mip chain from the decoded base level.
    pub generate: bool,
    /// Cooperative cancellation shared with the admission record.
    pub cancelled: Arc<AtomicBool>,
}

/// One retained fine mip awaiting phase-two upload, coarsest last.
#[derive(Clone, Debug)]
pub struct FineUpload {
    /// Contiguous coarse residency count once this level submits.
    pub target: u32,
    /// Storage mip level, zero being the largest.
    pub level: u32,
    /// Owned mip payload.
    pub mip: DecodedMip,
}

/// Backend-neutral record of a submitted texture.
#[derive(Clone, Copy, Debug)]
pub struct SubmittedInfo {
    /// Registry slot submitted for this handle.
    pub id: TextureId,
    /// Base width in texels.
    pub width: u32,
    /// Base height in texels.
    pub height: u32,
    /// Total decoded mip count, the residency ceiling.
    pub total: u32,
    /// GPU storage format for every mip.
    pub format: TextureFormat,
}

/// One retired native texture awaiting frame-safe destruction.
#[derive(Debug)]
pub struct RetiredEntry<T> {
    /// Registry slot withheld from reuse until destruction.
    pub id: TextureId,
    /// Descriptor binding reset to fallback before destruction.
    pub binding: u32,
    /// Native texture owned by the caller until destruction.
    pub texture: T,
    /// Transfer completion gating destruction.
    pub completion: CompletionToken,
}

/// Backend-neutral texture upload state owned by the pipeline.
pub struct TexturePipeline {
    pending: HashMap<TextureHandle, PendingUpload>,
    queue: VecDeque<QueuedDecode>,
    order: VecDeque<TextureHandle>,
    decoded: HashMap<TextureHandle, Result<DecodedTexture, TextureError>>,
    submitted: HashMap<TextureHandle, SubmittedInfo>,
    fine: HashMap<TextureHandle, Vec<FineUpload>>,
    published: HashMap<TextureHandle, u32>,
    ready: HashMap<TextureHandle, CompletionToken>,
    /// Handles whose initial readiness already fired its notification.
    /// Region updates re-arm `ready` for transfer waits without re-emitting.
    notified: HashSet<TextureHandle>,
    work: HashMap<TextureHandle, PrefixTransferWork>,
    last_transfer: HashMap<TextureHandle, CompletionToken>,
    transfer_bytes: HashMap<TextureHandle, u64>,
    targets: HashMap<TextureHandle, u32>,
    handoffs: HashMap<TextureHandle, Instant>,
    telemetry: Arc<TextureUploadTelemetry>,
}

impl TexturePipeline {
    /// Creates an empty pipeline. Slot and binding limits live in the
    /// caller-owned [`TextureRegistry`], which render targets share.
    pub fn new() -> Self {
        Self::default()
    }
}
impl Default for TexturePipeline {
    /// Creates an empty pipeline; see [`TexturePipeline::new`].
    fn default() -> Self {
        Self {
            pending: HashMap::new(),
            queue: VecDeque::new(),
            order: VecDeque::new(),
            decoded: HashMap::new(),
            submitted: HashMap::new(),
            fine: HashMap::new(),
            published: HashMap::new(),
            ready: HashMap::new(),
            notified: HashSet::new(),
            work: HashMap::new(),
            last_transfer: HashMap::new(),
            transfer_bytes: HashMap::new(),
            targets: HashMap::new(),
            handoffs: HashMap::new(),
            telemetry: Arc::new(TextureUploadTelemetry::default()),
        }
    }
}

impl TexturePipeline {
    /// Returns pending uploads awaiting decode or submission.
    pub const fn pending(&self) -> &HashMap<TextureHandle, PendingUpload> {
        &self.pending
    }

    /// Returns queued decodes awaiting worker admission.
    pub const fn queue(&self) -> &VecDeque<QueuedDecode> {
        &self.queue
    }

    /// Returns FIFO submission order for pending uploads.
    pub const fn order(&self) -> &VecDeque<TextureHandle> {
        &self.order
    }

    /// Returns decoded payloads awaiting FIFO submission.
    pub const fn decoded(&self) -> &HashMap<TextureHandle, Result<DecodedTexture, TextureError>> {
        &self.decoded
    }

    /// Returns backend-neutral submitted records.
    pub const fn submitted(&self) -> &HashMap<TextureHandle, SubmittedInfo> {
        &self.submitted
    }

    /// Returns retained fine-mip queues awaiting phase-two upload.
    pub const fn fine(&self) -> &HashMap<TextureHandle, Vec<FineUpload>> {
        &self.fine
    }

    /// Returns published coarse-prefix counts per submitted texture.
    pub const fn published(&self) -> &HashMap<TextureHandle, u32> {
        &self.published
    }

    /// Returns required-prefix completion tokens awaiting observation.
    pub const fn ready(&self) -> &HashMap<TextureHandle, CompletionToken> {
        &self.ready
    }

    /// Returns per-texture transfer ledger trackers.
    pub const fn work(&self) -> &HashMap<TextureHandle, PrefixTransferWork> {
        &self.work
    }

    /// Returns latest transfer tokens per submitted texture.
    pub const fn last_transfer(&self) -> &HashMap<TextureHandle, CompletionToken> {
        &self.last_transfer
    }

    /// Returns decoded staging bytes per transfer-pending texture.
    pub const fn transfer_bytes(&self) -> &HashMap<TextureHandle, u64> {
        &self.transfer_bytes
    }

    /// Returns residency targets per submitted texture.
    pub const fn targets(&self) -> &HashMap<TextureHandle, u32> {
        &self.targets
    }

    /// Returns cumulative upload telemetry.
    pub fn telemetry(&self) -> &Arc<TextureUploadTelemetry> {
        &self.telemetry
    }

    /// Admits one validated load: reserves FIFO order and queues its decode.
    pub fn admit(&mut self, handle: TextureHandle, pending: PendingUpload, decode: QueuedDecode) {
        self.order.push_back(handle);
        self.pending.insert(handle, pending);
        self.queue.push_back(decode);
    }

    /// Removes one pending upload and its queue entries without residue.
    pub fn remove_pending(&mut self, handle: TextureHandle) -> Option<PendingUpload> {
        self.queue.retain(|job| job.handle != handle);
        self.order.retain(|queued| *queued != handle);
        self.decoded.remove(&handle);
        self.notified.remove(&handle);
        self.pending.remove(&handle)
    }

    /// Decides FIFO decode admission under the shared byte budget.
    pub fn decode_plan() -> BatchPlan {
        BatchPlan::new(WORKING_SET_BUDGET_BYTES, DECODE_RESERVATION_BYTES)
    }
}

/// Terminal reason for one failed upload; the caller maps it to its own
/// error type and emits its own failure event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UploadFailure {
    /// Decoded dimensions disagree with the admission contract.
    DimensionMismatch,
    /// The requirement cannot be satisfied by the decoded chain.
    Requirement(TextureScheduleError),
    /// Backend validation, allocation, or submission failed.
    Native(AllocationError),
    /// Registry tracking failed after native submission.
    Tracking(TextureError),
    /// Internal ledger accounting overflowed.
    Ledger,
}

/// Outcome of one submitted fine level.
#[derive(Clone, Copy, Debug)]
pub struct FineSubmitted {
    /// Texture that gained one resident fine level.
    pub handle: TextureHandle,
    /// Contiguous coarse residency count once this level publishes.
    pub target: u32,
    /// Transfer completion gating publication.
    pub token: CompletionToken,
    /// Payload bytes admitted under background budget.
    pub bytes: u64,
}

/// Outcome of one terminally failed texture in the fine pump.
#[derive(Clone, Copy, Debug)]
pub struct FineFailed {
    /// Texture that failed terminally.
    pub handle: TextureHandle,
    /// Registry slot to fail.
    pub id: TextureId,
    /// Machine-readable failure reason.
    pub failure: UploadFailure,
}

/// Submits retained fine mips coarsest-first after required readiness.
///
/// Each level goes through the validated region-update vehicle under
/// background admission; one pump drains every fitting level, backpressure
/// defers the remainder to the next pump, and terminal failures surface as
/// [`FineFailed`] rather than sampling an incomplete chain silently.
/// Admission order across textures is handle-sorted so retries are
/// deterministic. Textures whose required prefix is unobserved, unpublished,
/// or untracked are skipped silently.
#[allow(
    clippy::implicit_hasher,
    reason = "callers always pass std maps; a hasher parameter buys no leverage"
)]
pub fn pump_fine_uploads<B: TextureBackendContext>(
    pipe: &mut TexturePipeline,
    registry: &mut TextureRegistry,
    backend: &mut B,
    natives: &mut HashMap<TextureHandle, B::Texture>,
    pool: &mut SharedTransferPool,
) -> (Vec<FineSubmitted>, Vec<FineFailed>) {
    if pipe.fine.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut handles: Vec<_> = pipe.fine.keys().copied().collect();
    handles.sort_by_key(|handle| handle.into_raw());
    let mut submitted = Vec::new();
    let mut failed = Vec::new();
    for handle in handles {
        let Some(info) = pipe.submitted.get(&handle) else {
            pipe.fine.remove(&handle);
            continue;
        };
        let id = info.id;
        // Required readiness gates phase two: frames must already sample the
        // required prefix before finer levels consume transfer budget.
        if pipe.ready.contains_key(&handle) {
            continue;
        }
        let Ok(required) = registry.required_mips(id) else {
            continue;
        };
        if pipe.published.get(&handle).copied().unwrap_or(0) < required {
            continue;
        }
        let Some(native) = natives.get_mut(&handle) else {
            pipe.fine.remove(&handle);
            continue;
        };
        let mut queue = pipe.fine.remove(&handle).unwrap_or_default();
        while let Some(entry) = queue.pop() {
            let bytes = entry.mip.bytes.len() as u64;
            if pool.acquire_background(bytes).is_err() {
                queue.push(entry);
                break;
            }
            let region = TextureRegion {
                mip_level: entry.level,
                x: 0,
                y: 0,
                width: entry.mip.width,
                height: entry.mip.height,
                bytes: &entry.mip.bytes,
            };
            let token = match backend.update_texture_region(native, &region) {
                Ok(token) => token,
                Err(AllocationError::OutOfMemory) => {
                    // Native staging exhaustion is transient backpressure,
                    // not permanent texture failure.
                    pool.release_background(bytes);
                    queue.push(entry);
                    break;
                }
                Err(error) => {
                    pool.release_background(bytes);
                    queue.clear();
                    failed.push(FineFailed {
                        handle,
                        id,
                        failure: UploadFailure::Native(error),
                    });
                    break;
                }
            };
            if registry
                .mark_mips_submitted(id, entry.target, token)
                .is_err()
            {
                pool.release_background(bytes);
                queue.push(entry);
                break;
            }
            pipe.last_transfer.insert(handle, token);
            // Ledger overflow is an accounting bug, never caller input; fail
            // the texture terminally rather than leaking background bytes. A
            // missing work entry means its ledgers already retired; the level
            // still counts as submitted.
            if let Some(work) = pipe.work.get_mut(&handle)
                && work.note_fine(token, bytes).is_err()
            {
                pool.release_background(bytes);
                queue.clear();
                failed.push(FineFailed {
                    handle,
                    id,
                    failure: UploadFailure::Ledger,
                });
                break;
            }
            pipe.telemetry.record_staging_bytes(bytes);
            submitted.push(FineSubmitted {
                handle,
                target: entry.target,
                token,
                bytes,
            });
        }
        if !queue.is_empty() {
            pipe.fine.insert(handle, queue);
        }
    }
    (submitted, failed)
}

/// One texture whose ledgers retired at the observed transfer completion.
#[derive(Clone, Copy, Debug)]
pub struct Reclaimed {
    /// Texture whose ledgers released.
    pub handle: TextureHandle,
    /// Required bytes released; the caller drops them from the pool.
    pub required_bytes: u64,
    /// Background bytes released; the caller drops them from the pool.
    pub fine_bytes: u64,
    /// True when both ledgers cleared and no fine work remains.
    pub cleared: bool,
}

/// Releases required-prefix and fine-upload credits at their own completion
/// tokens. The caller drops the returned bytes from the shared pool and
/// forgets staging accounting for cleared textures.
pub fn reclaim_transfer_work(pipe: &mut TexturePipeline, completed: u64) -> Vec<Reclaimed> {
    if pipe.work.is_empty() {
        return Vec::new();
    }
    let handles = pipe.work.keys().copied().collect::<Vec<_>>();
    let mut reclaimed = Vec::new();
    for handle in handles {
        let pending_fine = pipe.fine.contains_key(&handle);
        let Some(work) = pipe.work.get_mut(&handle) else {
            continue;
        };
        let (required_bytes, fine_bytes) = work.release_completed(completed);
        let cleared = work.is_clear() && !pending_fine;
        if cleared {
            pipe.work.remove(&handle);
            pipe.transfer_bytes.remove(&handle);
        }
        if required_bytes > 0 || fine_bytes > 0 || cleared {
            reclaimed.push(Reclaimed {
                handle,
                required_bytes,
                fine_bytes,
                cleared,
            });
        }
    }
    reclaimed
}

/// Reports whether any submitted texture hides its required prefix behind an
/// unpublished descriptor. Handles in `failed` are terminal and skipped.
#[allow(
    clippy::implicit_hasher,
    reason = "callers always pass std sets; a hasher parameter buys no leverage"
)]
pub fn unpublished_required(
    pipe: &TexturePipeline,
    registry: &TextureRegistry,
    failed: &HashSet<TextureHandle>,
) -> bool {
    pipe.submitted.iter().any(|(handle, info)| {
        !failed.contains(handle)
            && registry
                .required_mips(info.id)
                .is_ok_and(|required| pipe.published.get(handle).copied().unwrap_or(0) < required)
    })
}

/// Installs one submitted texture's required-prefix descriptor ahead of
/// completion. Returns true when the binding now references the real view.
/// A false return is transient descriptor contention the caller surfaces
/// retryably; it never records a frame sampling fallback for a required
/// texture.
///
/// # Errors
///
/// Returns [`AllocationError`] for descriptor loss or view-creation failure.
#[allow(
    clippy::implicit_hasher,
    reason = "callers always pass std maps; a hasher parameter buys no leverage"
)]
pub fn reference_required_prefix<B: TextureBackendContext>(
    pipe: &mut TexturePipeline,
    registry: &TextureRegistry,
    backend: &mut B,
    natives: &mut HashMap<TextureHandle, B::Texture>,
    handle: TextureHandle,
) -> Result<bool, AllocationError> {
    let Some(info) = pipe.submitted.get(&handle) else {
        return Ok(true);
    };
    let Ok(required) = registry.required_mips(info.id) else {
        return Ok(true);
    };
    if pipe.published.get(&handle).copied().unwrap_or(0) >= required {
        return Ok(true);
    }
    if !backend.texture_descriptors_ready()? {
        return Ok(false);
    }
    let Some(native) = natives.get_mut(&handle) else {
        return Ok(true);
    };
    // Transient contention stays retryable like the completion-gated
    // advance; only allocation, validation, and device errors propagate.
    backend.reference_texture_prefix(native, required)?;
    pipe.published.insert(handle, required);
    Ok(true)
}

/// Advances residency for every submitted texture whose transfers completed.
///
/// Publication expands coarse-first around pending hidden updates but never
/// shrinks an already published view while its region overwrite completes.
/// Only fence-gate contention stays in place retryably; other backend errors
/// propagate.
///
/// # Errors
///
/// Returns [`AllocationError`] when descriptors cannot be read or a view
/// fails beyond transient contention.
#[allow(
    clippy::implicit_hasher,
    reason = "callers always pass std maps; a hasher parameter buys no leverage"
)]
pub fn advance_residency<B: TextureBackendContext>(
    pipe: &mut TexturePipeline,
    registry: &TextureRegistry,
    backend: &mut B,
    natives: &mut HashMap<TextureHandle, B::Texture>,
    completed: u64,
) -> Result<(), AllocationError> {
    let advances = pipe
        .submitted
        .iter()
        .filter_map(|(handle, info)| {
            let available = registry.resident_mips(info.id).ok()?;
            let target = pipe
                .targets
                .get(handle)
                .copied()
                .unwrap_or(info.total)
                .min(available);
            let published = pipe.published.get(handle).copied().unwrap_or(0);
            let native = natives.get(handle)?;
            let native_available = native
                .mip_transfer_values()
                .iter()
                .rev()
                .take_while(|value| **value != 0 && **value <= completed)
                .count();
            // Continue coarse-first expansion around a pending hidden update,
            // but never silently shrink an already published view while its
            // region overwrite completes.
            let target = if target > published {
                target.min(u32::try_from(native_available).ok()?.max(published))
            } else {
                target
            };
            let ready = mip_range_completion(native.mip_transfer_values(), target)?;
            (target != published && ready <= completed).then_some((*handle, target))
        })
        .collect::<Vec<_>>();
    if advances.is_empty() {
        return Ok(());
    }
    if !backend.texture_descriptors_ready()? {
        return Ok(());
    }
    for (handle, resident_mips) in advances {
        let Some(native) = natives.get_mut(&handle) else {
            continue;
        };
        match backend.publish_texture_mips(native, resident_mips) {
            Ok(()) => {
                pipe.published.insert(handle, resident_mips);
            }
            // Fence-gate contention is transient: the recorded target stays
            // and the next poll retries.
            Err(AllocationError::NativeFailure) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Observes transfer completion for one texture: a completed copy is not
/// sample-ready until real coarse residency publishes. Returns true exactly
/// once, when initial readiness fires; the caller emits its bind-ready event.
/// Handoff latency is recorded inside.
pub fn observe_ready(
    pipe: &mut TexturePipeline,
    registry: &TextureRegistry,
    handle: TextureHandle,
    completed: u64,
) -> bool {
    // Resolve through pending uploads as well as submitted textures: the
    // requirement is recorded at admission, so the gate must hold even
    // before native submission populates the live map.
    let required = pipe
        .pending
        .get(&handle)
        .map(|pending| pending.id)
        .or_else(|| pipe.submitted.get(&handle).map(|info| info.id))
        .and_then(|id| registry.required_mips(id).ok())
        .unwrap_or(1);
    // Optional residency still needs its first real coarse mip published:
    // a zero requirement never means a zero-mip view.
    let published = required.max(1);
    let ready = pipe
        .ready
        .get(&handle)
        .is_some_and(|token| token.value <= completed)
        && pipe
            .published
            .get(&handle)
            .is_some_and(|mips| *mips >= published);
    if !ready {
        return false;
    }
    pipe.ready.remove(&handle);
    let handoff = pipe.handoffs.remove(&handle);
    // Region-update re-arms consume the transfer wait here but must not
    // notify twice: only the first readiness per load returns a transition.
    if !pipe.notified.insert(handle) {
        return false;
    }
    if let Some(submitted_at) = handoff {
        pipe.telemetry.record_handoff_latency(
            u64::try_from(submitted_at.elapsed().as_micros()).unwrap_or(u64::MAX),
        );
    }
    true
}

/// Marks terminal loss: sweeps queued decodes, clears transfer ledgers and
/// retained payloads, and cancels every pending registry slot. The caller
/// emits terminal events and marks its own lifecycle state.
pub fn drop_device_state(
    pipe: &mut TexturePipeline,
    registry: &mut TextureRegistry,
    pool: &mut SharedTransferPool,
) {
    pipe.transfer_bytes.clear();
    for (_, work) in pipe.work.drain() {
        let (required_bytes, fine_bytes) = work.release_all();
        pool.release_texture(required_bytes);
        pool.release_background(fine_bytes);
    }
    pipe.fine.clear();
    // Terminal loss ends every load lifetime; no handle observes again.
    pipe.notified.clear();
    cancel_all_pending(pipe, registry, pool);
}

/// Cancels every pending upload: flags workers, cancels registry slots, and
/// clears queues. Terminal cancellation drops every texture reservation at
/// once; background geometry entries retire separately by transfer token.
pub fn cancel_all_pending(
    pipe: &mut TexturePipeline,
    registry: &mut TextureRegistry,
    pool: &mut SharedTransferPool,
) {
    for (_, pending) in pipe.pending.drain() {
        pending.cancelled.store(true, Ordering::Release);
        let _ = registry.cancel_upload(pending.id);
    }
    pipe.queue.clear();
    pipe.order.clear();
    pipe.decoded.clear();
    pool.clear_texture_ledger();
}

/// Collects one ready decode result into FIFO order. Returns the reservation
/// bytes the caller must release when the job is stale (cancelled after
/// admission); ready results carry no release.
pub fn collect_result(
    pipe: &mut TexturePipeline,
    handle: TextureHandle,
    decoded: Result<DecodedTexture, TextureError>,
) -> u64 {
    if let Some(pending) = pipe.pending.get_mut(&handle) {
        pending.decoded_bytes = decoded.as_ref().ok().map(|decoded| {
            decoded.mips.iter().fold(0_u64, |total, mip| {
                total.saturating_add(mip.bytes.len() as u64)
            })
        });
        pipe.decoded.insert(handle, decoded);
        0
    } else {
        // Cancelled/stale jobs still report completion internally so their
        // reservation cannot strand the FIFO; their public terminal event
        // was emitted by the cancelling path.
        DECODE_RESERVATION_BYTES
    }
}

impl TexturePipeline {
    /// Drops every tracked upload without emitting events or touching
    /// ledgers; the caller retires shared-budget bytes and native resources
    /// through its own terminal paths first.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.queue.clear();
        self.order.clear();
        self.decoded.clear();
        self.submitted.clear();
        self.fine.clear();
        self.published.clear();
        self.ready.clear();
        self.notified.clear();
        self.work.clear();
        self.last_transfer.clear();
        self.transfer_bytes.clear();
        self.targets.clear();
        self.handoffs.clear();
    }
}

impl TexturePipeline {
    /// Pops the next queued decode for worker admission.
    pub fn pop_queued(&mut self) -> Option<QueuedDecode> {
        self.queue.pop_front()
    }

    /// Requeues a decode at the head after failed budget admission.
    pub fn push_queued_front(&mut self, job: QueuedDecode) {
        self.queue.push_front(job);
    }

    /// Pops the next queued decode with required-positive work prioritized.
    ///
    /// Required jobs dispatch first in admission order so an optional head
    /// never occupies the last worker while required work waits; optional
    /// jobs keep their own admission order behind them. Reservation
    /// accounting is unchanged: exactly one job leaves per call.
    pub fn pop_queued_prioritized(&mut self) -> Option<QueuedDecode> {
        let index = self
            .queue
            .iter()
            .position(|job| {
                self.pending
                    .get(&job.handle)
                    .is_none_or(|pending| pending.config.required_mips != 0)
            })
            .unwrap_or(0);
        self.queue.remove(index)
    }

    /// Reports whether any pending upload gates frame submission.
    ///
    /// Only positive requirements gate: optional zero-mip uploads decode and
    /// submit asynchronously while frames keep recording.
    pub fn has_required_pending(&self) -> bool {
        self.pending
            .values()
            .any(|pending| pending.config.required_mips != 0)
    }

    /// Returns mutable pending uploads for fallback backfill.
    pub fn pending_mut(&mut self) -> &mut HashMap<TextureHandle, PendingUpload> {
        &mut self.pending
    }

    /// Returns published counts for test-fabricated residency.
    pub fn published_mut(&mut self) -> &mut HashMap<TextureHandle, u32> {
        &mut self.published
    }

    /// Returns required-prefix tokens for test-fabricated readiness.
    pub fn ready_mut(&mut self) -> &mut HashMap<TextureHandle, CompletionToken> {
        &mut self.ready
    }

    /// Returns handoff instants for test-fabricated latency.
    pub fn handoffs_mut(&mut self) -> &mut HashMap<TextureHandle, Instant> {
        &mut self.handoffs
    }

    /// Returns residency targets for test-fabricated growth.
    pub fn targets_mut(&mut self) -> &mut HashMap<TextureHandle, u32> {
        &mut self.targets
    }
}

impl TexturePipeline {
    /// Forgets one texture's transfer tracking and returns its held ledgers
    /// for shared-pool release. Decode failures hold no ledgers and take the
    /// same path.
    pub fn forget_transfer(&mut self, handle: TextureHandle) -> (u64, u64) {
        self.ready.remove(&handle);
        self.fine.remove(&handle);
        self.handoffs.remove(&handle);
        self.transfer_bytes.remove(&handle);
        self.work
            .remove(&handle)
            .map_or((0, 0), PrefixTransferWork::release_all)
    }

    /// Removes one submitted record (and its target) without touching
    /// ledgers; the caller retires those through its own terminal paths.
    pub fn remove_submitted(&mut self, handle: TextureHandle) -> Option<SubmittedInfo> {
        self.targets.remove(&handle);
        self.last_transfer.remove(&handle);
        self.published.remove(&handle);
        self.notified.remove(&handle);
        self.submitted.remove(&handle)
    }

    /// Forgets every submitted residency record for a terminally failed
    /// texture: submitted info, publication, targets, latest transfer, and
    /// any decoded payload. Ledgers stay caller-owned; run `forget_transfer`
    /// first so shared-pool bytes release through the same path as every
    /// other failure.
    pub fn forget_submitted(&mut self, handle: TextureHandle) {
        self.submitted.remove(&handle);
        self.published.remove(&handle);
        self.targets.remove(&handle);
        self.last_transfer.remove(&handle);
        self.decoded.remove(&handle);
        self.notified.remove(&handle);
    }

    /// Returns latest transfer tokens for region-update bookkeeping.
    pub fn last_transfer_mut(&mut self) -> &mut HashMap<TextureHandle, CompletionToken> {
        &mut self.last_transfer
    }

    /// Returns staging bytes for region-update bookkeeping.
    pub fn transfer_bytes_mut(&mut self) -> &mut HashMap<TextureHandle, u64> {
        &mut self.transfer_bytes
    }
}

impl TexturePipeline {
    /// Takes one decoded result without submitting it; the caller releases
    /// its reservation when present.
    pub fn take_decoded(
        &mut self,
        handle: TextureHandle,
    ) -> Option<Result<DecodedTexture, TextureError>> {
        self.decoded.remove(&handle)
    }
}
