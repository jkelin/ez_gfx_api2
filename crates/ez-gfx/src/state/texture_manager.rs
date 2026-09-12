//! Thin adapters between [`ContextState`] and the manager-owned texture pipeline.
//!
//! Every backend-neutral policy lives in `ez_gfx_texture_manager::pipeline`
//! and runs through the [`TextureBackendContext`](super::TextureBackendContext)
//! traits with static dispatch. These adapters only extract context state,
//! translate outcomes into typed errors and upload events, and drive the
//! frame-submission gate that keeps required prefixes out of fallback.

use super::texture::{
    fail_texture_job, map_upload_failure, publish_advance_step, reclaim_retired_textures,
    record_texture_failure, schedule_texture_decodes, texture_descriptors_ready,
};
use super::{
    ContextState, DECODE_RESERVATION_BYTES, Error, Result, RuntimePhase, TextureHandle,
    completed_texture_transfer_native, map_allocation, note_device_lost,
    poll_native_frame_completion, runtime_record,
};
use ez_gfx_texture_manager::pipeline::{
    SubmitOutcome, cancel_all_pending, pump_fine_uploads, reclaim_transfer_work,
    reference_required_prefix, submit_ready_uploads, unpublished_required,
};
use std::collections::HashSet;
use std::time::Instant;

/// Advances every ready stage; empty stages are no-ops and native failures propagate.
pub(super) fn pump_async_textures(context: &mut ContextState) -> Result<usize> {
    reclaim_retired_textures(context)?;
    reclaim_completed_work(context)?;
    collect_decode_results(context);

    let completed = submit_ready_uploads_adapter(context);
    pump_fine_uploads_adapter(context);
    schedule_texture_decodes(context)?;
    reclaim_retired_textures(context)?;
    Ok(completed)
}
/// Maximum owner-thread wait for required-prefix CPU decode at frame submission.
const SUBMIT_GATE_DECODE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Maximum owner-thread wait for submitted frames to drain out of descriptor slots.
const SUBMIT_GATE_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Drives every required texture's prefix to a referenced, GPU-waited draw.
///
/// Frame submission calls this before recording sampling work so no required
/// texture can render its fallback binding before its configured prefix.
/// Optional zero-mip textures never block here: their decodes submit
/// asynchronously and their bindings keep sampling fallback until real
/// residency publishes. Pending CPU decodes for required textures are pumped
/// to native submission (waiting only on decode and admission; transfer
/// completion is never polled here and stays GPU-gated through the required
/// tokens the frame graph attaches. Transfer-independent descriptors install
/// once no submitted frame can still observe the slot; prior frames drain
/// under a bound that observes graphics completion only. A required texture
/// therefore never samples fallback in the recorded frame. Failed textures
/// are skipped for the existing terminal error paths.
///
/// # Errors
///
/// Returns an error for native submission failures, when required decodes do
/// not finish within the submission bound, or when prior frames do not drain
/// within the descriptor bound.
pub(super) fn gate_required_textures_for_submit(
    context: &mut ContextState,
    requires_heap: bool,
) -> Result<()> {
    // Shaders without a bindless heap sample no textures; skip driving entirely.
    if !requires_heap {
        return Ok(());
    }
    // Steady-state frames skip the gate once no required texture is pending
    // or unpublished. Optional work below gets one bounded nonblocking pump
    // per frame so zero-only workloads progress without ever waiting.
    if !has_required_pending(&context.texture_pipeline) && !has_unpublished_required(context) {
        if !context.texture_pipeline.pending().is_empty() {
            pump_async_textures(context)?;
        }
        return Ok(());
    }
    // Native submission is impossible before device admission publishes the
    // fallback; frames then keep streaming instead of stalling admission.
    if !context.texture_fallback.is_ready() {
        return Ok(());
    }
    // Decode is finite CPU work on a live pool, so this waits only on
    // required decode and admission. Optional pending entries never appear
    // in this condition, so one zero texture cannot stall the frame.
    // Every terminal path drains its pending entry, and reclaim inside the
    // pump frees budget as the independently progressing GPU completes work.
    let deadline = Instant::now() + SUBMIT_GATE_DECODE_TIMEOUT;
    while has_required_pending(&context.texture_pipeline) {
        pump_async_textures(context)?;
        if !has_required_pending(&context.texture_pipeline) {
            break;
        }
        if Instant::now() >= deadline {
            return Err(Error::NotReady);
        }
        std::thread::yield_now();
    }
    if !has_unpublished_required(context) {
        return Ok(());
    }
    // Descriptor rewrites must avoid submitted frames, so reap them here. This
    // observes graphics completion only; texture transfer completion stays
    // GPU-gated through the tokens attached below.
    let drain_deadline = Instant::now() + SUBMIT_GATE_DRAIN_TIMEOUT;
    while !texture_descriptors_ready(&context.native)? {
        poll_native_frame_completion(&mut context.native)?;
        if texture_descriptors_ready(&context.native)? {
            break;
        }
        if Instant::now() >= drain_deadline {
            return Err(Error::NotReady);
        }
        std::thread::yield_now();
    }
    let handles: Vec<TextureHandle> = context.textures.keys().copied().collect();
    for handle in handles {
        // Post-drain contention is transient; surface it retryably rather than
        // recording a frame that samples fallback for a required texture.
        if !reference_required_prefix_early(context, handle)? {
            return Err(Error::NotReady);
        }
    }
    Ok(())
}

/// Reports whether any pending upload gates frame submission.
///
/// Only positive requirements gate; optional zero-mip uploads progress
/// asynchronously through ordinary pumps and event polling.
pub(super) fn has_required_pending(
    pipe: &ez_gfx_texture_manager::pipeline::TexturePipeline,
) -> bool {
    pipe.has_required_pending()
}

fn has_unpublished_required(context: &ContextState) -> bool {
    // Terminal failures publish through the error paths, never the prefix gate.
    let failed: HashSet<TextureHandle> = context.texture_failures.keys().copied().collect();
    unpublished_required(
        &context.texture_pipeline,
        &context.texture_registry,
        &failed,
    )
}

/// Installs one submitted texture's required-prefix descriptor ahead of completion.
///
/// Returns true when the binding now references the real view (or already did).
/// A false return is transient descriptor contention the caller surfaces
/// retryably; it never records a frame sampling fallback for a required texture.
///
/// # Errors
///
/// Returns an error for descriptor loss or view-creation failure.
fn reference_required_prefix_early(
    context: &mut ContextState,
    handle: TextureHandle,
) -> Result<bool> {
    if context.texture_failures.contains_key(&handle) {
        return Ok(true);
    }
    // Transient contention stays retryable like the completion-gated advance;
    // only allocation, validation, and device errors propagate.
    match reference_required_prefix(
        &mut context.texture_pipeline,
        &context.texture_registry,
        &mut context.native,
        &mut context.textures,
        handle,
    ) {
        // Installed prefixes advance like any publication; contention stays
        // retryable without recording fallback-sampling work.
        Ok(true) => publish_advance_step(Ok(())),
        Ok(false) => Ok(false),
        Err(error) => publish_advance_step(Err(error)),
    }
}

/// Releases required-prefix and fine-upload credits at their own completion tokens.
fn reclaim_completed_work(context: &mut ContextState) -> Result<()> {
    if context.texture_pipeline.work().is_empty() {
        return Ok(());
    }
    let completed =
        completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
    for reclaimed in reclaim_transfer_work(&mut context.texture_pipeline, completed) {
        context
            .transfer_pool
            .release_texture(reclaimed.required_bytes);
        context
            .transfer_pool
            .release_background(reclaimed.fine_bytes);
    }
    Ok(())
}

/// Collects every ready result; stale jobs release their reservation credit.
pub(super) fn collect_decode_results(context: &mut ContextState) {
    context
        .decode_textures
        .collect(&mut context.texture_pipeline, &mut context.transfer_pool);
}

/// Submits FIFO-ready decodes through the pipeline and translates outcomes.
fn submit_ready_uploads_adapter(context: &mut ContextState) -> usize {
    let outcomes = submit_ready_uploads(
        &mut context.texture_pipeline,
        &mut context.texture_registry,
        &mut context.native,
        &mut context.textures,
        context.texture_fallback.is_ready(),
    );
    let mut completed = 0;
    for outcome in outcomes {
        match outcome {
            SubmitOutcome::Submitted { handle } => {
                completed += 1;
                let decode =
                    runtime_record(context, handle.into_raw(), RuntimePhase::Decode, Ok(()));
                context.observability.push_event(decode);
                let upload =
                    runtime_record(context, handle.into_raw(), RuntimePhase::Upload, Ok(()));
                context.observability.push_event(upload);
            }
            SubmitOutcome::Failed {
                handle,
                id,
                failure,
                rollback,
            } => {
                // The decode reservation releases on every terminal path; only
                // successful submissions hold it through completion. Storage
                // accepted before tracking failed still needs its binding
                // reset and deferred destruction.
                if let Some(rollback) = rollback {
                    context.retired_textures.push(super::RetiredTexture {
                        id: rollback.id,
                        binding: rollback.binding,
                        native: rollback.texture,
                        completion: rollback.completion,
                    });
                }
                let error = map_upload_failure(failure);
                fail_texture_job(context, handle, id, error);
                if error == Error::DeviceLost {
                    note_device_lost(context);
                }
                completed += 1;
                context
                    .transfer_pool
                    .release_texture(DECODE_RESERVATION_BYTES);
            }
            SubmitOutcome::Dropped { .. } => {
                context
                    .transfer_pool
                    .release_texture(DECODE_RESERVATION_BYTES);
            }
        }
    }
    completed
}

/// Pumps retained fine mips and records terminal fine failures.
fn pump_fine_uploads_adapter(context: &mut ContextState) {
    let (submitted, failed) = pump_fine_uploads(
        &mut context.texture_pipeline,
        &mut context.texture_registry,
        &mut context.native,
        &mut context.textures,
        &mut context.transfer_pool,
    );
    let _ = submitted;
    for failure in failed {
        record_texture_failure(context, failure.handle, map_upload_failure(failure.failure));
    }
}

/// Cancels every pending upload; terminal cancellation drops every texture
/// reservation at once. The caller emits terminal events for drained handles.
pub(super) fn cancel_all_pending_textures(context: &mut ContextState) {
    cancel_all_pending(
        &mut context.texture_pipeline,
        &mut context.texture_registry,
        &mut context.transfer_pool,
    );
}
