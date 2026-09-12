//! FIFO submission: decoded payloads to native prefix storage.
//!
//! [`submit_ready_uploads`] walks pipeline FIFO order, converts each ready
//! decode through [`submit_one`], and reports per-texture outcomes the caller
//! translates into its own errors and events. Backend interaction stays
//! generic over [`TextureBackendContext`]; native handles never enter this
//! module except inside caller-owned maps keyed by handle.

use super::super::texture::{DecodedMip, DecodedTexture};
use super::super::{
    DECODE_RESERVATION_BYTES, PrefixTransferWork, TextureBackendContext, TextureBackendTexture,
    TextureId, TextureRegistry, fine_residency_targets, required_completion_token,
    resolve_required_mips,
};
use super::{FineUpload, PendingUpload, TexturePipeline};
use ez_gfx_core::handle::TextureHandle;
use ez_gfx_hal::{AllocationError, CompletionToken, ImageMip, QueueKind};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Native rollback payload for a submission that failed after the backend
/// accepted it; the caller destroys the texture and retires its binding.
#[derive(Debug)]
pub struct Rollback<T> {
    /// Registry slot to fail; already retired when `rollback` is present.
    pub id: TextureId,
    /// Descriptor binding the caller must reset before destruction.
    pub binding: u32,
    /// Native texture the caller must destroy.
    pub texture: T,
    /// Transfer completion gating destruction.
    pub completion: CompletionToken,
}

/// Outcome of one FIFO submission step.
#[derive(Debug)]
pub enum SubmitOutcome<T> {
    /// Native submission succeeded; the decode reservation stays held
    /// through the required-prefix completion token.
    Submitted {
        /// Texture whose required prefix is now in flight.
        handle: TextureHandle,
    },
    /// Terminal failure; the caller releases the decode reservation,
    /// destroys any rollback texture, and records the failure.
    Failed {
        /// Texture that failed terminally.
        handle: TextureHandle,
        /// Registry slot to fail; already retired when `rollback` is present.
        id: TextureId,
        /// Machine-readable failure reason.
        failure: super::UploadFailure,
        /// Present when the backend accepted storage before tracking failed.
        rollback: Option<Rollback<T>>,
    },
    /// Texture withdrawn without submission.
    Dropped {
        /// Texture withdrawn without submission.
        handle: TextureHandle,
    },
}

/// Submits ready decodes with required-positive work prioritized.
///
/// Within one requirement class FIFO holds: an undecoded required head blocks
/// later required decodes, and an undecoded optional head blocks later
/// optional decodes. Across classes a decoded required texture bypasses an
/// undecoded optional head so optional work never head-of-line blocks
/// required work; publication still never reorders inside either class.
///
/// The caller releases one decode reservation per [`SubmitOutcome::Failed`]
/// or [`SubmitOutcome::Dropped`]; [`SubmitOutcome::Submitted`] holds its
/// reservation through the required-prefix completion token. `fallback_ready` gates
/// native submission: no real upload may displace a missing fallback
/// descriptor.
#[allow(
    clippy::implicit_hasher,
    reason = "callers always pass std maps; a hasher parameter buys no leverage"
)]
pub fn submit_ready_uploads<B: TextureBackendContext>(
    pipe: &mut TexturePipeline,
    registry: &mut TextureRegistry,
    backend: &mut B,
    natives: &mut HashMap<TextureHandle, B::Texture>,
    fallback_ready: bool,
) -> Vec<SubmitOutcome<B::Texture>> {
    if !fallback_ready {
        return Vec::new();
    }
    let mut outcomes = Vec::new();
    while let Some(handle) = next_submittable(pipe) {
        // The selected handle always carries a decoded result; remove it
        // from FIFO order wherever the bypass found it.
        let Some(position) = pipe.order.iter().position(|queued| *queued == handle) else {
            break;
        };
        pipe.order.remove(position);
        let Some(decoded) = pipe.decoded.remove(&handle) else {
            break;
        };
        let Some(pending) = pipe.pending.remove(&handle) else {
            outcomes.push(SubmitOutcome::Dropped { handle });
            continue;
        };
        if pending.cancelled.load(Ordering::Acquire) {
            let _ = registry.cancel_upload(pending.id);
            outcomes.push(SubmitOutcome::Dropped { handle });
            continue;
        }
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                let _ = registry.cancel_upload(pending.id);
                outcomes.push(SubmitOutcome::Failed {
                    handle,
                    id: pending.id,
                    failure: super::UploadFailure::Tracking(error),
                    rollback: None,
                });
                continue;
            }
        };
        outcomes.push(submit_one(
            pipe, registry, backend, natives, handle, &pending, decoded,
        ));
    }
    outcomes
}

/// Selects the next decoded texture honoring the required-bypass policy.
///
/// A decoded FIFO head submits immediately. An undecoded optional head yields
/// to the first decoded required texture behind it; an undecoded required
/// head (or no decoded required texture) submits nothing so its class keeps
/// FIFO order.
fn next_submittable(pipe: &TexturePipeline) -> Option<TextureHandle> {
    let front = pipe.order().front().copied()?;
    if pipe.decoded().contains_key(&front) {
        return Some(front);
    }
    let head_optional = pipe
        .pending()
        .get(&front)
        .is_some_and(|pending| pending.config.required_mips == 0);
    if !head_optional {
        return None;
    }
    pipe.order().iter().copied().find(|handle| {
        pipe.decoded().contains_key(handle)
            && pipe
                .pending()
                .get(handle)
                .is_some_and(|pending| pending.config.required_mips != 0)
    })
}

/// Converts one decoded payload to native work: validates the admission
/// contract, resolves the requirement against the decoded total, creates
/// prefix storage through the backend traits, tracks every completion token in
/// the registry, and retains unsubmitted fine levels for the phase-two pump.
fn submit_one<B: TextureBackendContext>(
    pipe: &mut TexturePipeline,
    registry: &mut TextureRegistry,
    backend: &mut B,
    natives: &mut HashMap<TextureHandle, B::Texture>,
    handle: TextureHandle,
    pending: &PendingUpload,
    decoded: DecodedTexture,
) -> SubmitOutcome<B::Texture> {
    // A terminal dimension mismatch consumes no native work; the registry
    // slot is cancelled rather than retired since nothing was exposed.
    if (pending.config.width != 0 && decoded.width != pending.config.width)
        || (pending.config.height != 0 && decoded.height != pending.config.height)
        || (pending.config.mip_count != 0 && decoded.mip_count != pending.config.mip_count)
    {
        let _ = registry.cancel_upload(pending.id);
        return SubmitOutcome::Failed {
            handle,
            id: pending.id,
            failure: super::UploadFailure::DimensionMismatch,
            rollback: None,
        };
    }
    // The decoded total is known only here, so an over-sized requirement
    // fails now as a terminal typed error instead of clamping to a weaker
    // wait. Zero resolves to zero: optional residency imposes no requirement.
    let required = match resolve_required_mips(pending.config.required_mips, decoded.mip_count) {
        Ok(required) => required,
        Err(error) => {
            let _ = registry.cancel_upload(pending.id);
            return SubmitOutcome::Failed {
                handle,
                id: pending.id,
                failure: super::UploadFailure::Requirement(error),
                rollback: None,
            };
        }
    };
    // Mechanical single-coarse submission behind optional residency: every
    // backend rejects prefix zero and zero never indexes a completion token,
    // so the native upload always covers at least the coarsest mip while the
    // registry stores the semantic zero.
    let prefix = required.max(1);
    let mips = decoded
        .mips
        .iter()
        .map(|mip| ImageMip {
            width: mip.width,
            height: mip.height,
            bytes: &mip.bytes,
        })
        .collect::<Vec<_>>();
    let submitted_at = Instant::now();
    let binding = match registry.reserved_binding(pending.id) {
        Ok(binding) => binding,
        Err(error) => {
            let _ = registry.cancel_upload(pending.id);
            return SubmitOutcome::Failed {
                handle,
                id: pending.id,
                failure: super::UploadFailure::Tracking(error),
                rollback: None,
            };
        }
    };
    // Full storage is allocated but only the submitted coarse prefix uploads,
    // so required work never head-of-line blocks behind earlier textures'
    // fine levels in the backend transfer FIFO.
    let (texture, completions) = match backend.create_texture_with_prefix(
        decoded.format,
        &mips,
        binding,
        pending.config.sampler,
        prefix,
    ) {
        Ok(created) => created,
        Err(error) => {
            let _ = registry.cancel_upload(pending.id);
            return SubmitOutcome::Failed {
                handle,
                id: pending.id,
                failure: super::UploadFailure::Native(error),
                rollback: None,
            };
        }
    };
    // The backend returns exactly the submitted prefix tokens in
    // coarse-first order; anything else breaks the required-token index.
    if completions.len() != prefix as usize {
        return rollback::<B>(
            registry,
            handle,
            pending.id,
            binding,
            texture,
            super::UploadFailure::Native(AllocationError::NativeFailure),
        );
    }
    // The mechanical prefix is always nonzero, so token selection resolves;
    // the semantic zero is stored in the registry below, never indexed here.
    let Ok(Some(required_token)) = required_completion_token(&completions, prefix) else {
        return rollback::<B>(
            registry,
            handle,
            pending.id,
            binding,
            texture,
            super::UploadFailure::Native(AllocationError::NativeFailure),
        );
    };
    let mut completions = completions.into_iter();
    // The length check above guarantees a nonempty prefix; the guard keeps
    // the accepted native texture on the rollback path regardless.
    let Some(first) = completions.next() else {
        return rollback::<B>(
            registry,
            handle,
            pending.id,
            binding,
            texture,
            super::UploadFailure::Native(AllocationError::NativeFailure),
        );
    };
    // The registry stores the semantic requirement (zero for optional
    // residency) while every mechanical value above used the prefix.
    if let Err(failure) = track_submission(registry, pending.id, required, first, completions) {
        return rollback::<B>(registry, handle, pending.id, binding, texture, failure);
    }
    // The prefix end is the latest submitted token; the length check above
    // keeps it identical to the required token.
    let last = required_token;
    // Only the submitted prefix occupies staging; retained fine payloads
    // stay owner-side until phase two admits them.
    let staging_bytes = decoded
        .mips
        .iter()
        .rev()
        .take(prefix as usize)
        .fold(0_u64, |total, mip| {
            total.saturating_add(mip.bytes.len() as u64)
        });
    pipe.submitted.insert(
        handle,
        super::SubmittedInfo {
            id: pending.id,
            width: decoded.width,
            height: decoded.height,
            total: decoded.mip_count,
            format: decoded.format,
        },
    );
    pipe.targets.insert(handle, decoded.mip_count);
    pipe.published.insert(handle, 0);
    pipe.last_transfer.insert(handle, last);
    pipe.ready.insert(handle, required_token);
    pipe.transfer_bytes.insert(handle, staging_bytes);
    pipe.work.insert(
        handle,
        PrefixTransferWork::new(required_token, DECODE_RESERVATION_BYTES),
    );
    // Views borrow the chain above; the backend call ends the borrow so the
    // remaining levels move into phase-two retention without copying.
    let total = decoded.mip_count;
    retain_fine(pipe, handle, total, prefix, decoded.mips);
    pipe.handoffs.insert(handle, submitted_at);
    natives.insert(handle, texture);
    SubmitOutcome::Submitted { handle }
}

/// Tracks every submitted completion token plus the requirement in the
/// registry, in coarse-first order. The registry owns the requirement so
/// readiness and residency paths read one authority.
///
/// # Errors
///
/// Returns [`super::UploadFailure`] when token arithmetic overflows or registry
/// tracking rejects a transition; the caller rolls back like any tracking
/// failure.
fn track_submission(
    registry: &mut TextureRegistry,
    id: TextureId,
    required: u32,
    first: CompletionToken,
    rest: std::vec::IntoIter<CompletionToken>,
) -> Result<(), super::UploadFailure> {
    registry
        .mark_submitted(id, first)
        .map_err(super::UploadFailure::Tracking)?;
    for (index, completion) in rest.enumerate() {
        // The first token covers one mip; each following token adds one more
        // contiguous coarse level.
        let resident_mips = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(2))
            .ok_or(super::UploadFailure::Native(AllocationError::NativeFailure))?;
        registry
            .mark_mips_submitted(id, resident_mips, completion)
            .map_err(super::UploadFailure::Tracking)?;
    }
    registry
        .set_required_mips(id, required)
        .map_err(super::UploadFailure::Tracking)
}

/// Retires the registry slot and packages the accepted native texture for
/// caller-side destruction after its binding resets to fallback.
fn rollback<B: TextureBackendContext>(
    registry: &mut TextureRegistry,
    handle: TextureHandle,
    id: TextureId,
    binding: u32,
    texture: B::Texture,
    failure: super::UploadFailure,
) -> SubmitOutcome<B::Texture> {
    let _ = registry.retire(id);
    // A zero transfer value still builds a valid token; the value only gates
    // destruction ordering, never correctness of the retired binding reset.
    let completion =
        CompletionToken::new(QueueKind::TextureTransfer, texture.last_transfer_value());
    let (completion, failure) = match (completion, failure) {
        (Ok(completion), failure) => (completion, failure),
        // The slot is already retired; surfacing token construction as the
        // failure keeps one terminal path instead of two.
        (Err(_), _) => (
            CompletionToken::new(QueueKind::TextureTransfer, 0)
                .unwrap_or_else(|_| panic_no_token()),
            super::UploadFailure::Native(AllocationError::NativeFailure),
        ),
    };
    SubmitOutcome::Failed {
        handle,
        id,
        failure,
        rollback: Some(Rollback {
            id,
            binding,
            texture,
            completion,
        }),
    }
}

/// Token construction over `u64::MIN` cannot fail; this marks the
/// unreachable fallback for the compiler.
fn panic_no_token() -> CompletionToken {
    unreachable!("zero transfer value always builds a completion token")
}

/// Retains the unsubmitted fine levels for the phase-two pump.
///
/// Storage order is largest-first, so the coarsest remainder sits at the back
/// of the retained queue behind its ascending residency targets.
fn retain_fine(
    pipe: &mut TexturePipeline,
    handle: TextureHandle,
    total: u32,
    required: u32,
    mips: Vec<DecodedMip>,
) {
    // A full-prefix submit retains nothing; the reservation still releases at
    // the required-prefix completion token through the shared work entry. Targets come
    // from one helper so submission order stays coarsest-first in one place.
    // Zipping stops at the shorter side, so a short chain retains fewer
    // levels exactly like the previous indexed lookup.
    let retained = fine_residency_targets(total, required)
        .into_iter()
        .zip(mips)
        .enumerate()
        .filter_map(|(level, (target, mip))| {
            // Levels ascend from zero over the largest-first chain while
            // targets descend from the total; chains are validated small.
            Some(FineUpload {
                target,
                level: u32::try_from(level).ok()?,
                mip,
            })
        })
        .collect::<Vec<_>>();
    if !retained.is_empty() {
        pipe.fine.insert(handle, retained);
    }
}
