//! Backend-neutral texture upload scheduling.
//!
//! This crate owns the generic texture-management policy behind backend
//! traits: required-mip readiness selection, FIFO decode batching, and the
//! shared transfer budget consumed by both the texture scheduler and
//! geometry/buffer uploads. Backend adapters own native allocations, submit
//! device copies, and report [`CompletionToken`] progress; this crate never
//! touches native handles.

#![forbid(unsafe_code)]

/// Decoded-texture ingestion: sources, destinations, validation, and telemetry.
pub mod texture;
pub use texture::{
    DecodedMip, DecodedTexture, MAX_TEXTURE_BYTES, PreparedTextureDecode, TextureConfig,
    TextureDecodeCallback, TextureDecoder, TextureDestination, TextureSource,
    TextureUploadTelemetry, TextureUploadTelemetrySnapshot, coarse_to_fine_mip_levels,
    generate_mips, register_texture_decoder, unregister_texture_decoder,
};

pub mod pipeline;
pub use pipeline::{
    DECODE_RESERVATION_BYTES, FineFailed, FineSubmitted, FineUpload, PendingUpload, QueuedDecode,
    Reclaimed, RetiredEntry, Rollback, SubmitOutcome, SubmittedInfo, TexturePipeline,
    UploadFailure, WORKING_SET_BUDGET_BYTES,
};
pub mod decode_driver;
pub use decode_driver::{
    DecodeDriver, DecodeDriverError, DecodedTextureJob, DispatchReport, MAX_CPU_POOL_THREADS,
};
mod registry;
pub use registry::{TextureId, TextureRegistry};

use std::collections::VecDeque;

use ez_gfx_hal::{CompletionToken, QueueKind, ReusableStagingPool};

/// Failures produced while planning generic texture uploads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureScheduleError {
    /// A mip count is zero or exceeds the decoded chain.
    InvalidMipCount,
    /// A byte reservation overflows the shared budget.
    BudgetOverflow,
}

impl core::fmt::Display for TextureScheduleError {
    /// Formats the schedule error using its debug name.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TextureScheduleError {}

/// Requires the entire decoded chain, whatever its length.
///
/// The count is unknown until decode completes, so resolve this with
/// [`resolve_required_mips`] once the decoded total is known. Snapshot and
/// capture paths use it to pin fully-streamed frames through the normal frame
/// GPU wait instead of parsing container headers in application code.
///
/// The requirement names a count of the coarsest mips: `0` requests no mips
/// for initial rendering (the stable binding keeps sampling fallback until
/// real residency publishes), `1` waits for the smallest mip only, `total`
/// waits for the full chain. In largest-to-smallest storage order a positive
/// prefix is the tail; in native submission order (which is coarse-first) it
/// is the head.
pub const REQUIRED_MIPS_FULL: u32 = u32::MAX;

/// Validates a required-prefix request against a decoded chain.
///
/// # Errors
///
/// Returns [`TextureScheduleError::InvalidMipCount`] when the chain is
/// empty or the requirement exceeds it. Over-sized requirements fail
/// instead of clamping so misconfiguration cannot silently weaken a wait.
/// [`REQUIRED_MIPS_FULL`] resolves to the decoded total instead of failing.
pub fn resolve_required_mips(count: u32, total: u32) -> Result<u32, TextureScheduleError> {
    // Decoded chains always carry at least one mip; an empty total means
    // the caller measured the wrong chain, not a satisfiable request.
    if total == 0 {
        return Err(TextureScheduleError::InvalidMipCount);
    }
    // The sentinel resolves after the empty check above. Zero stays zero:
    // optional residency imposes no requirement, so there is nothing to
    // validate against the decoded total.
    if count == REQUIRED_MIPS_FULL {
        return Ok(total);
    }
    if count == 0 {
        return Ok(0);
    }
    if count > total {
        return Err(TextureScheduleError::InvalidMipCount);
    }
    Ok(count)
}

/// Backend source of per-mip transfer values behind the upload interface.
///
/// Adapters implement this for their native texture; the generic scheduler
/// only reads transfer values and selects tokens. Values run in
/// largest-to-smallest storage order (mip level zero first), so the required
/// coarse prefix is the tail of this slice. See [`mip_range_completion`].
pub trait MipTransferValues {
    /// Returns one transfer value per mip, ordered largest to smallest.
    fn mip_transfer_values(&self) -> &[u64];
}

/// Backend texture behind the upload interface: transfer progress plus cancellation.
///
/// Backend crates implement this for their private native texture type; the
/// generic scheduler only reads coarse-first progress and flips the
/// cancellation flag. Native handles never cross this interface.
pub trait TextureBackendTexture: MipTransferValues {
    /// Returns the latest transfer value submitted for this texture.
    fn last_transfer_value(&self) -> u64;
}

/// Backend context behind the texture-upload traits.
///
/// Backend crates implement this for their private native context; `ez-gfx`
/// monomorphizes one generic caller per enum variant, so dispatch stays
/// static and this crate never depends on a backend. All payload types are
/// HAL or core types. Staging-cache eviction stays behind
/// [`ReclaimableStaging`], blanket-implemented for every
/// [`ReusableStagingPool`](ez_gfx_hal::ReusableStagingPool).
pub trait TextureBackendContext {
    /// Private native texture owned by the implementing backend.
    type Texture: TextureBackendTexture;

    /// Creates full sampled storage but submits only the coarsest `prefix` mips.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when validation, allocation, or
    /// submission fails.
    fn create_texture_with_prefix(
        &mut self,
        format: ez_gfx_hal::TextureFormat,
        mips: &[ez_gfx_hal::ImageMip<'_>],
        binding: u32,
        sampler: ez_gfx_hal::TextureSamplerDesc,
        prefix: u32,
    ) -> Result<(Self::Texture, Vec<CompletionToken>), ez_gfx_hal::AllocationError>;

    /// Submits one retained fine-mip region update through the validated vehicle.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when the region or submission fails.
    fn update_texture_region(
        &mut self,
        texture: &mut Self::Texture,
        region: &ez_gfx_hal::TextureRegion<'_>,
    ) -> Result<CompletionToken, ez_gfx_hal::AllocationError>;

    /// Installs the transfer-independent view for a resident coarse prefix.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when view creation fails.
    fn reference_texture_prefix(
        &mut self,
        texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), ez_gfx_hal::AllocationError>;

    /// Exposes a resident coarse prefix through the stable binding.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when the range is not resident
    /// or view creation fails.
    fn publish_texture_mips(
        &mut self,
        texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), ez_gfx_hal::AllocationError>;

    /// Reports whether descriptors may be rewritten without disturbing frames.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when the gate cannot be read.
    fn texture_descriptors_ready(&self) -> Result<bool, ez_gfx_hal::AllocationError>;

    /// Reports whether a retired texture escapes every submitted frame.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when completion cannot be read.
    fn texture_retirement_ready(
        &self,
        completion: CompletionToken,
    ) -> Result<bool, ez_gfx_hal::AllocationError>;

    /// Queues a retired texture for deferred native destruction.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when the texture cannot be queued.
    fn destroy_texture(
        &mut self,
        texture: Self::Texture,
    ) -> Result<(), ez_gfx_hal::AllocationError>;

    /// Returns the latest retired texture-transfer timeline value.
    ///
    /// # Errors
    ///
    /// Returns [`ez_gfx_hal::AllocationError`] when the timeline is unavailable.
    fn completed_texture_transfer_value(&self) -> Result<u64, ez_gfx_hal::AllocationError>;

    /// Prevents transfer-owner jobs not yet recorded from copying this texture.
    fn cancel_texture_transfers(texture: &Self::Texture);
}

/// Returns the completion token gating the required coarse prefix, or `None`
/// when the requirement is optional.
///
/// `completions` runs in native submission order, which is coarse-first: the
/// first value covers the smallest mip and values advance toward level zero.
/// A positive required prefix therefore ends at index `required - 1`. This is
/// the opposite end from [`MipTransferValues`]; passing storage-ordered values
/// here would gate the wrong mip. Callers needing the mechanical single-mip
/// submission behind an optional requirement resolve index `0` themselves;
/// every backend rejects an empty prefix, so zero never indexes here.
///
/// # Errors
///
/// Returns [`TextureScheduleError::InvalidMipCount`] when the chain is empty,
/// the requirement exceeds it, or a parallel completion is missing.
pub fn required_completion_token(
    completions: &[CompletionToken],
    required: u32,
) -> Result<Option<CompletionToken>, TextureScheduleError> {
    if required == 0 {
        return Ok(None);
    }
    let total =
        u32::try_from(completions.len()).map_err(|_| TextureScheduleError::InvalidMipCount)?;
    let required = resolve_required_mips(required, total)?;
    // Completions run coarse-first, so the required prefix ends at index
    // `required - 1`; resolution guarantees the index is in range.
    completions
        .get(required as usize - 1)
        .copied()
        .map(Some)
        .ok_or(TextureScheduleError::InvalidMipCount)
}

/// Returns the latest transfer value covering a resident coarse prefix.
///
/// A zero value inside the exposed range means a hidden fine update is still
/// in flight, so no completion is reported rather than a partial one.
pub fn mip_range_completion(values: &[u64], resident_mips: u32) -> Option<u64> {
    // A hidden fine update must not delay a coarse view; exposed updated mips use their max.
    let count = usize::try_from(resident_mips).ok()?;
    let range = values.get(values.len().checked_sub(count)?..)?;
    if range.contains(&0) {
        return None;
    }
    range.iter().copied().max()
}

/// Returns contiguous residency targets for deferred fine mips, largest first.
///
/// Storage order is largest-first, so the unsubmitted fine levels are the head
/// of the chain and each target names the contiguous coarse residency count
/// once that level submits. Consumers pop from the back so the level adjacent
/// to the required prefix submits first. A full-prefix submit retains nothing.
/// `required` must already be resolved against `total` (see
/// [`resolve_required_mips`]); callers pass the mechanical single-mip prefix
/// behind optional residency, never zero. An out-of-range requirement yields
/// no targets and the caller fails the texture through its terminal path.
pub fn fine_residency_targets(total_mips: u32, required: u32) -> Vec<u32> {
    // An empty or inverted range collects to nothing, so full-prefix submits
    // and out-of-range requirements retain no targets by construction.
    (required.saturating_add(1)..=total_mips).rev().collect()
}

/// Tracks one texture's split required/background transfer ledgers.
///
/// The required decode reservation holds through the required-prefix
/// completion token, not the final fine mip; each phase-two fine payload
/// holds background bytes under its own latest token. Both ledgers release
/// at their own completion value so one shared [`SharedTransferPool`] sees
/// a single admission domain.
#[derive(Clone, Copy, Debug)]
pub struct PrefixTransferWork {
    required_completion: CompletionToken,
    required_bytes: u64,
    fine_completion: Option<CompletionToken>,
    /// Phase-two fine payload bytes awaiting their latest completion.
    fine_bytes: u64,
}

impl PrefixTransferWork {
    /// Tracks a required reservation under its prefix completion token.
    /// Completion tokens retire on their own queue timeline, so callers pass
    /// transfer-queue tokens; foreign queues never reach these values.
    pub fn new(required_completion: CompletionToken, required_bytes: u64) -> Self {
        debug_assert_eq!(
            required_completion.queue,
            QueueKind::TextureTransfer,
            "prefix ledger tracks transfer-queue completion"
        );
        Self {
            required_completion,
            required_bytes,
            fine_completion: None,
            fine_bytes: 0,
        }
    }

    /// Adds one submitted fine payload under its completion token.
    ///
    /// # Errors
    ///
    /// Returns [`TextureScheduleError::BudgetOverflow`] when the running fine
    /// total overflows its `u64` representation.
    pub fn note_fine(
        &mut self,
        completion: CompletionToken,
        bytes: u64,
    ) -> Result<(), TextureScheduleError> {
        debug_assert_eq!(
            completion.queue,
            QueueKind::TextureTransfer,
            "fine ledger tracks transfer-queue completion"
        );
        // Saturating would hide a real accounting bug; fail loudly instead so
        // the caller fails the texture rather than leaking ledger bytes.
        self.fine_bytes = self
            .fine_bytes
            .checked_add(bytes)
            .ok_or(TextureScheduleError::BudgetOverflow)?;
        self.fine_completion = Some(completion);
        Ok(())
    }

    /// Releases whichever ledgers retired at `completed`, returning both.
    pub fn release_completed(&mut self, completed: u64) -> (u64, u64) {
        let required = if self.required_completion.value <= completed {
            core::mem::take(&mut self.required_bytes)
        } else {
            0
        };
        let fine = if self
            .fine_completion
            .is_some_and(|completion| completion.value <= completed)
        {
            self.fine_completion = None;
            core::mem::take(&mut self.fine_bytes)
        } else {
            0
        };
        (required, fine)
    }

    /// Releases both ledgers unconditionally, returning each byte count.
    pub fn release_all(mut self) -> (u64, u64) {
        (
            core::mem::take(&mut self.required_bytes),
            core::mem::take(&mut self.fine_bytes),
        )
    }

    /// Returns true once both ledgers fully released.
    pub const fn is_clear(&self) -> bool {
        self.required_bytes == 0 && self.fine_bytes == 0
    }
}

/// Decides FIFO decode admission under a shared byte budget.
/// Mirrors the scheduler window: each active decode reserves the per-request
/// maximum, and an empty pool admits one request alone so a future larger
/// valid request cannot starve permanently.
#[derive(Clone, Copy, Debug)]
pub struct BatchPlan {
    /// Maximum decode plus in-flight transfer bytes admitted together.
    pub budget: u64,
    /// Per-request reservation while the true output size is unknown.
    pub reservation: u64,
}

impl BatchPlan {
    /// Creates a batching window from a budget and per-request reservation.
    pub const fn new(budget: u64, reservation: u64) -> Self {
        Self {
            budget,
            reservation,
        }
    }

    /// Returns whether one more decode may start.
    ///
    /// Mirrors [`SharedTransferPool::acquire_required`]: only the texture
    /// ledger counts, so geometry and buffer pressure never blocks required
    /// texture work; an empty texture ledger still admits one reservation
    /// alone so oversized requests cannot deadlock.
    pub fn admits(&self, active: usize, threads: usize, pool: &SharedTransferPool) -> bool {
        if active >= threads {
            return false;
        }
        // Saturating addition keeps a near-full ledger from wrapping into a
        // false admission; overflow then falls back to the empty-ledger rule.
        pool.texture_ledger().saturating_add(self.reservation) <= self.budget
            || pool.texture_ledger() == 0
    }
}

/// One transfer-queue-shared byte budget for texture and geometry uploads.
///
/// Texture decode reservations hold bytes through the required-prefix
/// token; geometry and buffer uploads record their staging bytes until the
/// transfer timeline retires them. Both managers consult the same instance so
/// one hardware transfer queue sees one admission domain.
///
/// Priority policy: admission order across textures stays FIFO so publication
/// stays deterministic. Required texture work outranks background work only
/// through admission: required reservations keep the empty-ledger progress
/// exception while background reservations never force admission.
#[derive(Clone, Debug)]
pub struct SharedTransferPool {
    capacity: u64,
    /// Required texture decode/transfer reservations, released at
    /// required-prefix completion, cancel, or loss.
    texture_bytes: u64,
    /// Admitted background reservations (phase-two fine mips, native staging
    /// growth), released explicitly after submission or on failure.
    background_bytes: u64,
    /// Transfer-gated staging leases from geometry and buffer uploads.
    staged: VecDeque<(CompletionToken, u64)>,
    /// Running byte total of `staged` so `in_use` stays O(1).
    staged_bytes: u64,
}
impl SharedTransferPool {
    /// Creates an empty pool with a finite byte capacity.
    pub const fn new(capacity: u64) -> Self {
        Self {
            capacity,
            texture_bytes: 0,
            background_bytes: 0,
            staged: VecDeque::new(),
            staged_bytes: 0,
        }
    }

    /// Returns the configured capacity.
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Returns admitted but unreleased texture reservation bytes.
    pub const fn texture_ledger(&self) -> u64 {
        self.texture_bytes
    }

    /// Returns admitted but unreleased background reservation bytes.
    pub const fn background_ledger(&self) -> u64 {
        self.background_bytes
    }

    /// Returns admitted but unreleased bytes across all classes.
    pub fn in_use(&self) -> u64 {
        // Staged leases retire lazily at the next upload or explicit
        // reclaim; the maintained total keeps this O(1) on hot paths.
        self.texture_bytes
            .saturating_add(self.background_bytes)
            .saturating_add(self.staged_bytes)
    }

    /// Admits a required texture reservation, preempting background pressure.
    ///
    /// Only the texture ledger counts toward capacity: geometry, buffer, and
    /// phase-two bytes already in flight never block required texture work.
    /// An empty texture ledger still admits one reservation alone so progress
    /// cannot deadlock on a valid oversized request.
    ///
    /// # Errors
    ///
    /// Returns [`TextureScheduleError::BudgetOverflow`] when the reservation
    /// would exceed capacity while other texture bytes are outstanding.
    pub fn acquire_required(&mut self, bytes: u64) -> Result<(), TextureScheduleError> {
        // checked_add fails before mutating so a failed admission never
        // strands partial bytes in the ledger.
        let next = self
            .texture_bytes
            .checked_add(bytes)
            .ok_or(TextureScheduleError::BudgetOverflow)?;
        if next > self.capacity && self.texture_bytes != 0 {
            return Err(TextureScheduleError::BudgetOverflow);
        }
        self.texture_bytes = next;
        Ok(())
    }

    /// Admits background bytes only when they fit the remaining budget.
    ///
    /// # Errors
    ///
    /// Returns [`TextureScheduleError::BudgetOverflow`] when the request does
    /// not fit, including the empty-pool case: background work never forces
    /// admission the way required texture mips do.
    pub fn acquire_background(&mut self, bytes: u64) -> Result<(), TextureScheduleError> {
        let next = self
            .in_use()
            .checked_add(bytes)
            .ok_or(TextureScheduleError::BudgetOverflow)?;
        if next > self.capacity {
            return Err(TextureScheduleError::BudgetOverflow);
        }
        self.background_bytes = self
            .background_bytes
            .checked_add(bytes)
            .ok_or(TextureScheduleError::BudgetOverflow)?;
        Ok(())
    }

    /// Abandons the whole texture ledger at terminal cancellation; every
    /// pending reservation dies with the queue it belonged to.
    pub fn clear_texture_ledger(&mut self) {
        self.texture_bytes = 0;
    }

    /// Releases texture bytes; a debug underflow is loud while release
    /// builds stay saturating.
    pub fn release_texture(&mut self, bytes: u64) {
        debug_assert!(
            bytes <= self.texture_bytes,
            "texture ledger release exceeds admission"
        );
        self.texture_bytes = self.texture_bytes.saturating_sub(bytes);
    }

    /// Releases background bytes; a debug underflow is loud while release
    /// builds stay saturating.
    pub fn release_background(&mut self, bytes: u64) {
        debug_assert!(
            bytes <= self.background_bytes,
            "background ledger release exceeds admission"
        );
        self.background_bytes = self.background_bytes.saturating_sub(bytes);
    }

    /// Records a background upload retired by a transfer-queue token.
    pub fn note_transfer(&mut self, token: CompletionToken, bytes: u64) {
        // Only transfer-queue tokens gate reuse; foreign queues never retire
        // entries here, matching the native staging pools.
        if token.queue == QueueKind::Transfer {
            self.staged.push_back((token, bytes));
            self.staged_bytes = self.staged_bytes.saturating_add(bytes);
        }
    }

    /// Drops background entries whose transfer token completed.
    pub fn reclaim_transfers(&mut self, queue: QueueKind, completed_value: u64) {
        // Completion counters only retire their own queue timeline; a larger
        // value from another queue must not release transfer-owned bytes.
        let mut released = 0_u64;
        self.staged.retain(|(token, bytes)| {
            let done = token.queue == queue && token.value <= completed_value;
            if done {
                released = released.saturating_add(*bytes);
            }
            !done
        });
        debug_assert!(
            released <= self.staged_bytes,
            "staged reclaim exceeds noted transfers"
        );
        self.staged_bytes = self.staged_bytes.saturating_sub(released);
    }
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;

    fn token(value: u64) -> CompletionToken {
        CompletionToken::new(QueueKind::TextureTransfer, value).unwrap()
    }

    fn transfer_token(value: u64) -> CompletionToken {
        CompletionToken::new(QueueKind::Transfer, value).unwrap()
    }

    #[test]
    fn zero_is_optional_and_positive_requirements_validate_against_total() {
        assert_eq!(resolve_required_mips(0, 3), Ok(0));
        assert_eq!(resolve_required_mips(2, 3), Ok(2));
        assert_eq!(
            resolve_required_mips(4, 3),
            Err(TextureScheduleError::InvalidMipCount)
        );
        assert_eq!(
            resolve_required_mips(0, 0),
            Err(TextureScheduleError::InvalidMipCount)
        );
    }
    #[test]
    fn required_mips_full_resolves_to_decoded_total() {
        // The full-chain sentinel selects the final token, so snapshots pin
        // streaming without header parsing.
        assert_eq!(REQUIRED_MIPS_FULL, u32::MAX);
        assert_eq!(resolve_required_mips(REQUIRED_MIPS_FULL, 1), Ok(1));
        assert_eq!(resolve_required_mips(REQUIRED_MIPS_FULL, 11), Ok(11));
        assert_eq!(
            resolve_required_mips(REQUIRED_MIPS_FULL, 0),
            Err(TextureScheduleError::InvalidMipCount)
        );
        let completions = vec![token(7), token(9), token(11)];
        assert_eq!(
            required_completion_token(&completions, REQUIRED_MIPS_FULL),
            Ok(Some(token(11)))
        );
    }

    #[test]
    fn required_token_is_absent_for_optional_residency() {
        // Native submissions remain coarse-first, but zero has no required
        // completion dependency. Positive requirements select their prefix end.
        let coarse = token(7);
        let mid = token(9);
        let fine = token(11);
        let completions = vec![coarse, mid, fine];
        assert_eq!(required_completion_token(&completions, 0), Ok(None));
        assert_eq!(required_completion_token(&completions, 1), Ok(Some(coarse)));
        assert_eq!(required_completion_token(&completions, 2), Ok(Some(mid)));
        assert_eq!(required_completion_token(&completions, 3), Ok(Some(fine)));
        assert_eq!(
            required_completion_token(&completions, 4),
            Err(TextureScheduleError::InvalidMipCount)
        );
        assert_eq!(
            required_completion_token(&[], 1),
            Err(TextureScheduleError::InvalidMipCount)
        );
    }

    #[test]
    fn fine_split_defers_unsubmitted_levels_coarsest_first() {
        // Retained targets run largest-first with ascending contiguous
        // residency counts; consumers pop from the back so the level adjacent
        // to the required prefix submits first. A full-prefix submit retains
        // nothing.
        assert_eq!(fine_residency_targets(3, 3), Vec::<u32>::new());
        assert_eq!(fine_residency_targets(4, 1), vec![4, 3, 2]);
        assert_eq!(fine_residency_targets(4, 2), vec![4, 3]);
        assert_eq!(fine_residency_targets(1, 1), Vec::<u32>::new());
    }

    #[test]
    fn mip_range_completion_reports_only_fully_updated_prefixes() {
        // Values run largest-to-smallest, so the exposed coarse prefix is the
        // tail; any zero inside it means a hidden update is still in flight.
        for (values, resident, expected) in [
            (&[7, 2, 1][..], 1, Some(1)),
            (&[7, 9, 1][..], 2, Some(9)),
            (&[7, 9, 1][..], 3, Some(9)),
            (&[0, 2, 1][..], 2, Some(2)),
            (&[0, 2, 1][..], 3, None),
            (&[7, 9, 0][..], 2, None),
            (&[7, 9, 11][..], 2, Some(11)),
            (&[1][..], 0, None),
            (&[1][..], 2, None),
            (&[][..], 1, None),
        ] {
            assert_eq!(mip_range_completion(values, resident), expected);
        }
    }

    #[test]
    fn batch_plan_bounds_threads_and_preempts_background_pressure() {
        let plan = BatchPlan::new(256, 64);
        let mut pool = SharedTransferPool::new(256);
        assert!(plan.admits(0, 2, &pool));
        pool.acquire_required(192).unwrap();
        assert!(plan.admits(1, 2, &pool));
        pool.acquire_required(64).unwrap();
        assert!(!plan.admits(1, 2, &pool));
        assert!(!plan.admits(2, 2, &pool));
        // Background pressure never blocks required work: the ledger is
        // empty, so admission proceeds despite 200 in-flight bytes.
        pool.clear_texture_ledger();
        pool.note_transfer(transfer_token(1), 200);
        assert!(plan.admits(0, 2, &pool));
        pool.acquire_required(64).unwrap();
        assert_eq!(pool.in_use(), 264);
        assert!(plan.admits(0, 2, &pool));
    }

    #[test]
    fn shared_pool_separates_ledgers_and_retires_staged_by_queue() {
        let mut pool = SharedTransferPool::new(100);
        pool.acquire_required(64).unwrap();
        assert_eq!((pool.in_use(), pool.texture_ledger()), (64, 64));
        // Background fits in the remainder but never forces admission.
        pool.acquire_background(32).unwrap();
        assert_eq!((pool.in_use(), pool.background_ledger()), (96, 32));
        assert_eq!(
            pool.acquire_background(8),
            Err(TextureScheduleError::BudgetOverflow)
        );
        pool.release_background(32);
        pool.release_texture(64);
        assert_eq!(pool.in_use(), 0);
        // An empty texture ledger admits one reservation alone.
        pool.acquire_required(100).unwrap();
        pool.release_texture(100);

        pool.note_transfer(transfer_token(3), 40);
        // Foreign-queue tokens never gate staged reuse.
        pool.note_transfer(token(9), 40);
        assert_eq!(pool.in_use(), 40);
        pool.reclaim_transfers(QueueKind::TextureTransfer, u64::MAX);
        assert_eq!(pool.in_use(), 40);
        pool.reclaim_transfers(QueueKind::Transfer, 2);
        assert_eq!(pool.in_use(), 40);
        pool.reclaim_transfers(QueueKind::Transfer, 3);
        assert_eq!(pool.in_use(), 0);
    }

    #[test]
    fn mip_transfer_values_adapter_drives_generic_selection() {
        struct Fake {
            values: Vec<u64>,
        }
        impl MipTransferValues for Fake {
            fn mip_transfer_values(&self) -> &[u64] {
                &self.values
            }
        }
        let texture = Fake {
            values: vec![5, 8, 0],
        };
        // Generic code reads values without knowing the native type.
        assert_eq!(mip_range_completion(texture.mip_transfer_values(), 2), None);
        assert_eq!(mip_range_completion(texture.mip_transfer_values(), 1), None);
        let ready = Fake {
            values: vec![5, 8, 9],
        };
        assert_eq!(
            mip_range_completion(ready.mip_transfer_values(), 3),
            Some(9)
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Error reported while decoding or tracking textures.
/// Texture processing error.
pub enum TextureError {
    /// Encoded bytes or decoded mip data are malformed.
    InvalidData,
    /// The texture format or layout is not supported.
    Unsupported,
    /// Texture dimensions, byte size, or mip count exceed supported limits.
    TooLarge,
    /// A registry limit is zero.
    InvalidCapacity,
    /// No texture slot or descriptor binding remains available.
    CapacityExceeded,
    /// The requested operation is not allowed in the texture's current state.
    InvalidState,
    /// The texture has no resident mip levels yet.
    NotReady,
    /// The texture handle does not identify a current allocation.
    NotFound,
    /// A reused slot can no longer advance its generation counter.
    GenerationExhausted,
}

/// Uniform eviction interface over heterogeneous staging caches.
///
/// Native staging buckets cannot move across backends: backend allocation
/// types stay private, so no generic crate can own the storage. This trait
/// shares what can be shared -- accounting and eviction order -- letting one
/// orchestrator trim the largest retained cache across every pool under one
/// ceiling without touching any bucket.
pub trait ReclaimableStaging {
    /// Backend allocation retained by an evicted bucket.
    type Staging;
    /// Currently retained bucket capacity in bytes.
    fn retained_bytes(&self) -> u64;
    /// Largest completed bucket capacity without evicting it.
    fn largest_completed_capacity(&self, queue: QueueKind, completed: u64) -> Option<u64>;
    /// Removes and returns the largest completed bucket, if any.
    fn pop_largest_completed(&mut self, queue: QueueKind, completed: u64) -> Option<Self::Staging>;
}

impl<T> ReclaimableStaging for ReusableStagingPool<T> {
    type Staging = T;

    /// Delegates to the pool's non-allocating telemetry read.
    fn retained_bytes(&self) -> u64 {
        self.retained_bytes()
    }

    /// Delegates to the pool's largest-completed peek.
    fn largest_completed_capacity(&self, queue: QueueKind, completed: u64) -> Option<u64> {
        self.largest_completed_capacity(queue, completed)
    }

    /// Delegates to the pool's largest-completed removal.
    fn pop_largest_completed(&mut self, queue: QueueKind, completed: u64) -> Option<T> {
        self.pop_largest_completed(queue, completed)
    }
}
