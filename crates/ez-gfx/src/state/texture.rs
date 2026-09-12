use crate::Result;

#[cfg(windows)]
use super::Dx12Context;
#[cfg(target_vendor = "apple")]
use super::MetalContext;
use super::{
    Arc, AtomicBool, CompletionToken, ContextHandle, ContextState, DECODE_RESERVATION_BYTES, Error,
    Instant, MAX_TEXTURE_BYTES, MipTransferValues, NativeContext, NativeTexture, Ordering,
    PendingUpload, QueuedDecode, ResourceKind, RetiredTexture, RetiredTextureBinding, RuntimePhase,
    TextureBackendContext, TextureBackendTexture, TextureDecoder, TextureFormat, TextureHandle,
    TextureId, TextureRegion, TextureUploadTelemetrySnapshot, UploadEvent, UploadResource,
    UploadStatus, VulkanContext, completed_texture_transfer_native, destroy_native_texture,
    map_allocation, map_lifecycle, map_texture, poll_native_frame_completion,
    publish_reserved_fallback, pump_async_textures, runtime_record, runtime_status,
    with_context_mut,
};
use ez_gfx_runtime::ContextHealth;
#[path = "texture_backend.rs"]
mod texture_backend;

/// Maps a pipeline failure to the context error type.
///
/// Dimension mismatches are admission-contract violations; native failures
/// keep their allocation mapping (including terminal device loss); ledger
/// overflow is an internal accounting bug, never caller input.
pub(super) fn map_upload_failure(failure: super::UploadFailure) -> Error {
    use super::UploadFailure;
    match failure {
        UploadFailure::DimensionMismatch => Error::InvalidArgument,
        UploadFailure::Requirement(error) => super::map_schedule(error),
        UploadFailure::Native(error) => map_allocation(error),
        UploadFailure::Tracking(error) => map_texture(error),
        UploadFailure::Ledger => Error::NativeFailure,
    }
}

/// Admission contract owned by the texture manager; re-exported here so the
/// safe facade path (`state::TextureConfig`, `ez_gfx::TextureConfig`) is unchanged.
pub use ez_gfx_texture_manager::texture::TextureConfig;

/// Validates a texture sampler against the shared native contract.
///
/// The C boundary rejects non-finite and out-of-range anisotropy; the safe entry
/// point must enforce the same invariant so native lowering (Vulkan passthrough,
/// DX12 floor, Metal clamp-to-16) never observes values the C API cannot send.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for non-finite or out-of-range anisotropy.
pub(super) fn validate_texture_sampler(sampler: &ez_gfx_hal::TextureSamplerDesc) -> Result<()> {
    // 1.0 disables anisotropy everywhere; 16.0 is the largest representable level.
    if !sampler.max_anisotropy.is_finite() || !(1.0..=16.0).contains(&sampler.max_anisotropy) {
        return Err(Error::InvalidArgument);
    }
    Ok(())
}

#[cfg(test)]
mod publish_tests {
    use super::Error;
    use super::publish_advance_step;

    #[test]
    fn only_fence_gate_contention_stays_retryable() {
        // DX12's graphics-fence gate reports transient contention as a plain
        // native failure; the advance path retries it instead of surfacing it.
        assert_eq!(
            publish_advance_step(Err(ez_gfx_hal::AllocationError::NativeFailure)),
            Ok(false)
        );
    }

    #[test]
    fn allocation_and_capability_errors_stay_terminal() {
        // OOM, validation, and capability failures must surface terminally
        // through the shared mapping, never hang in `NotReady`.
        assert_eq!(
            publish_advance_step(Err(ez_gfx_hal::AllocationError::ZeroSize)),
            Err(Error::InvalidArgument)
        );
        assert_eq!(
            publish_advance_step(Err(ez_gfx_hal::AllocationError::Unsupported)),
            Err(Error::Unsupported)
        );
        assert_eq!(
            publish_advance_step(Err(ez_gfx_hal::AllocationError::OutOfMemory)),
            Err(Error::NativeFailure)
        );
    }
    #[test]
    fn publication_success_and_device_loss_pass_through() {
        assert_eq!(publish_advance_step(Ok(())), Ok(true));
        assert_eq!(
            publish_advance_step(Err(ez_gfx_hal::AllocationError::DeviceLost)),
            Err(Error::DeviceLost)
        );
    }
}

/// Queues texture decode and upload without blocking the caller.
///
/// The manager copies source bytes, reserves the stable binding, and admits FIFO decode and
/// transfer waves under the shared transfer budget. Callers retain no asynchronous lifetime
/// or batch-size obligation.
///
/// # Errors
///
/// Returns an error for invalid input, an unvalidated sampler, exhausted handles, or a stale
/// context.
pub fn load_texture(
    context: ContextHandle,
    bytes: &[u8],
    config: &TextureConfig,
) -> Result<TextureHandle> {
    if bytes.is_empty() || bytes.len() > MAX_TEXTURE_BYTES {
        return Err(Error::InvalidArgument);
    }
    validate_texture_sampler(&config.sampler)?;
    let owned = bytes.to_vec().into_boxed_slice();
    let source_bytes = owned.len() as u64;
    let config = *config;

    with_context_mut(context, |context| {
        reclaim_retired_textures(context)?;
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if !context.texture_fallback.permits_load() {
            return Err(Error::NotReady);
        }
        context
            .decode_textures
            .ensure_started()
            .map_err(|_| Error::NativeFailure)?;
        let compression = match &context.native {
            NativeContext::Vulkan(native) => native.adapter_info().map_or(
                ez_gfx_core::capability::CompressionSupport::NONE,
                |adapter| adapter.capabilities().compression,
            ),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native.adapter_info().capabilities().compression,
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native.adapter_info().capabilities().compression,
        };
        let prepared = TextureDecoder::prepare(config.source, compression, config.destination)
            .map_err(map_texture)?;
        let texture = context
            .texture_registry
            .begin_upload()
            .map_err(map_texture)?;
        let handle = match context.identity.insert(ResourceKind::Texture) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = context.texture_registry.cancel_upload(texture);
                return Err(map_lifecycle(error));
            }
        };
        let Ok(typed) = TextureHandle::from_packed(handle) else {
            let _ = context.identity.remove(handle, ResourceKind::Texture);
            let _ = context.texture_registry.cancel_upload(texture);
            return Err(Error::NativeFailure);
        };
        let binding = context
            .texture_registry
            .reserved_binding(texture)
            .map_err(map_texture)?;
        let fallback_ready = context.texture_fallback.is_ready();
        if fallback_ready && let Err(error) = publish_reserved_fallback(context, binding) {
            let _ = context.identity.remove(handle, ResourceKind::Texture);
            let _ = context.texture_registry.cancel_upload(texture);
            return Err(error);
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        context.texture_pipeline.admit(
            typed,
            PendingUpload {
                id: texture,
                cancelled: cancelled.clone(),
                config,
                fallback_published: fallback_ready,
                source_bytes,
                decoded_bytes: None,
                admitted_at: Instant::now(),
            },
            QueuedDecode {
                handle: typed,
                prepared,
                bytes: owned,
                generate: config.generate_mips,
                cancelled,
            },
        );
        let record = runtime_record(context, typed.into_raw(), RuntimePhase::Admission, Ok(()));
        context.observability.push_event(record);
        context.upload_events.push(UploadEvent {
            resource: UploadResource::Texture(typed),
            status: UploadStatus::SourceStaged,
        });
        schedule_texture_decodes(context)?;
        Ok(typed)
    })
}

/// Fills the private decode window without blocking a worker or exposing policy to callers.
///
/// Each active job reserves the per-request maximum because encoded formats and custom decoders
/// need not reveal their output size before decoding.
pub(super) fn schedule_texture_decodes(context: &mut ContextState) -> Result<()> {
    // The driver pops FIFO order, reserves transfer bytes atomically with
    // each spawn, and reports failed spawns for the terminal failure path.
    // Late pool construction failure leaves queue and ledgers untouched, so
    // only shutdown maps here; it cannot race the owner thread.
    let report = context
        .decode_textures
        .dispatch(&mut context.texture_pipeline, &mut context.transfer_pool)
        .map_err(|_| Error::NativeFailure)?;
    for (handle, id) in report.failed {
        fail_texture_job(context, handle, id, Error::QueueFull);
    }
    Ok(())
}

pub(super) fn record_texture_failure(
    context: &mut ContextState,
    handle: TextureHandle,
    error: Error,
) {
    // Submitted failures cancel every unsignalable gate and release both
    // ledgers; decode failures have none of this state and take the same path.
    if let Some(native) = context.textures.get(&handle) {
        cancel_native_texture_transfers(native);
    }
    // The retained native texture stays owned until context teardown; only
    // transfer tracking is forgotten here. Submitted residency records clear
    // too so a failed-then-unloaded texture leaves no pipeline residue.
    let (required_bytes, fine_bytes) = context.texture_pipeline.forget_transfer(handle);
    context.transfer_pool.release_texture(required_bytes);
    context.transfer_pool.release_background(fine_bytes);
    context.texture_pipeline.forget_submitted(handle);
    // Retain the public handle long enough for deterministic polling.
    context.texture_failures.insert(handle, error);
    let record = runtime_record(context, handle.into_raw(), RuntimePhase::Decode, Err(error));
    context.observability.push_event(record);
    context.upload_events.push(UploadEvent {
        resource: UploadResource::Texture(handle),
        status: UploadStatus::Failed(runtime_status(Err(error))),
    });
}

fn retire_pending_texture_binding(context: &mut ContextState, id: TextureId) -> Result<()> {
    if context.texture_fallback.is_ready() {
        context.texture_registry.retire(id).map_err(map_texture)?;
        context
            .retired_texture_bindings
            .push(RetiredTextureBinding { id });
    } else {
        context
            .texture_registry
            .cancel_upload(id)
            .map_err(map_texture)?;
    }
    Ok(())
}

pub(super) fn fail_texture_job(
    context: &mut ContextState,
    handle: TextureHandle,
    id: TextureId,
    error: Error,
) {
    let _ = retire_pending_texture_binding(context, id);
    record_texture_failure(context, handle, error);
}

/// Marks terminal loss once, emits terminal upload events, and sweeps queued decodes.
pub(super) fn note_device_lost(context: &mut ContextState) {
    let pending: Vec<_> = context.texture_pipeline.pending().keys().copied().collect();
    let ready: Vec<_> = context.texture_pipeline.ready().keys().copied().collect();
    for texture in pending.into_iter().chain(ready) {
        context.upload_events.push(UploadEvent {
            resource: UploadResource::Texture(texture),
            status: UploadStatus::Failed(super::RuntimeStatus::DeviceLost),
        });
    }
    let _ = context.identity.mark_lost();
    // Terminal failure events retire every pending outcome and ledger.
    ez_gfx_texture_manager::pipeline::drop_device_state(
        &mut context.texture_pipeline,
        &mut context.texture_registry,
        &mut context.transfer_pool,
    );
    super::texture_manager::cancel_all_pending_textures(context);
}
fn cancel_native_texture_transfers(texture: &NativeTexture) {
    NativeContext::cancel_texture_transfers(texture);
}

pub(super) fn texture_descriptors_ready(context: &NativeContext) -> Result<bool> {
    context.texture_descriptors_ready().map_err(map_allocation)
}

fn reclaim_retired_texture_bindings(context: &mut ContextState) -> Result<()> {
    if context.retired_texture_bindings.is_empty() || !texture_descriptors_ready(&context.native)? {
        return Ok(());
    }
    for retired in context.retired_texture_bindings.drain(..) {
        context
            .texture_registry
            .release_retired(retired.id)
            .map_err(map_texture)?;
    }
    Ok(())
}

pub(super) fn reclaim_retired_textures(context: &mut ContextState) -> Result<()> {
    reclaim_retired_texture_bindings(context)?;
    let mut index = 0;
    while index < context.retired_textures.len() {
        let retired = &context.retired_textures[index];
        if !context
            .native
            .texture_retirement_ready(retired.completion)
            .map_err(map_allocation)?
        {
            index += 1;
            continue;
        }
        let binding = retired.binding;
        // Reset the slot before destroying its real resource; retirement readiness proves no
        // submitted frame can still observe the old descriptor.
        publish_reserved_fallback(context, binding)?;
        let retired = context.retired_textures.swap_remove(index);
        destroy_native_texture(&mut context.native, retired.native).map_err(map_allocation)?;
        context
            .texture_registry
            .release_retired(retired.id)
            .map_err(map_texture)?;
    }
    Ok(())
}

pub(super) fn publish_native_texture_mips(
    context: &mut NativeContext,
    texture: &mut NativeTexture,
    resident_mips: u32,
) -> std::result::Result<(), ez_gfx_hal::AllocationError> {
    context.publish_texture_mips(texture, resident_mips)
}

/// Maps a mip-publication outcome onto residency progress.
///
/// Only fence-gate contention is transient: `Ok(false)` keeps the recorded target
/// and surfaces `NotReady` until the next poll. Allocation, validation, capability,
/// and device errors stay terminal through the shared mapping.
pub(super) fn publish_advance_step(
    result: std::result::Result<(), ez_gfx_hal::AllocationError>,
) -> Result<bool> {
    match result {
        Ok(()) => Ok(true),
        Err(ez_gfx_hal::AllocationError::NativeFailure) => Ok(false),
        Err(error) => Err(map_allocation(error)),
    }
}

fn advance_texture_residency(context: &mut ContextState, completed: u64) -> Result<()> {
    // Reap finished frames before consulting the gate: completed submissions must
    // unblock publication during sustained rendering without wait_idle/readback.
    poll_native_frame_completion(&mut context.native)?;
    context
        .texture_registry
        .poll(ez_gfx_hal::QueueKind::TextureTransfer, completed)
        .map_err(map_texture)?;
    ez_gfx_texture_manager::pipeline::advance_residency(
        &mut context.texture_pipeline,
        &context.texture_registry,
        &mut context.native,
        &mut context.textures,
        completed,
    )
    .map_err(map_allocation)
}

pub(super) fn record_texture_ready(
    context: &mut ContextState,
    texture: TextureHandle,
    completed: u64,
) {
    // A completed copy is not sample-ready until the required coarse prefix is
    // published. The frame token already selects the required prefix, so this
    // descriptor gate is the second half of the same wait. Region updates keep
    // an existing published view but still require their new copy token.
    if !ez_gfx_texture_manager::pipeline::observe_ready(
        &mut context.texture_pipeline,
        &context.texture_registry,
        texture,
        completed,
    ) {
        return;
    }
    let record = runtime_record(context, texture.into_raw(), RuntimePhase::Bind, Ok(()));
    context.observability.push_event(record);
    context.upload_events.push(UploadEvent {
        resource: UploadResource::Texture(texture),
        status: UploadStatus::DeviceReady,
    });
}

/// Returns the async texture decode worker thread count for a context.
///
/// This observes the worker policy sized at creation from
/// `ContextOptions::texture_decode_workers` (zero selects the default topology),
/// whether or not the lazily built pool has decoded anything yet, letting FFI
/// callers verify the C descriptor value arrived without a C-side query export.
///
/// # Errors
///
/// Returns an error when the context handle is invalid or stale.
pub fn texture_decode_worker_count(context: ContextHandle) -> Result<u32> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        // Pool sizes always fit `u32`; the fallback only guards the conversion.
        u32::try_from(context.decode_textures.worker_count()).map_err(|_| Error::NativeFailure)
    })
}

/// Returns a texture's reserved stable binding.
///
/// A successful load makes this slot sample the context fallback until real publication.
///
/// # Errors
///
/// Returns an error when the context or texture handle is invalid, stale, or terminally failed.
pub fn texture_binding(context: ContextHandle, texture: TextureHandle) -> Result<u32> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let id = context
            .texture_pipeline
            .pending()
            .get(&texture)
            .map(|pending| pending.id)
            .or_else(|| {
                context
                    .texture_pipeline
                    .submitted()
                    .get(&texture)
                    .map(|info| info.id)
            })
            .ok_or(Error::InvalidContext)?;
        context
            .texture_registry
            .reserved_binding(id)
            .map_err(map_texture)
    })
}

#[cfg(feature = "ffi")]
/// Returns the stored texture extent.
///
/// # Errors
///
/// Returns an error when the context or texture is invalid, failed, or not ready.
pub fn texture_extent(context: ContextHandle, texture: TextureHandle) -> Result<(u32, u32)> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        pump_async_textures(context)?;
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }
        if context.texture_pipeline.pending().contains_key(&texture) {
            return Err(Error::NotReady);
        }
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let info = context
            .texture_pipeline
            .submitted()
            .get(&texture)
            .ok_or(Error::InvalidContext)?;
        Ok((info.width, info.height))
    })
}

/// Returns resident and total mip counts.
///
/// # Errors
///
/// Returns an error when the context or texture handle is invalid or stale.
pub fn texture_residency(context: ContextHandle, texture: TextureHandle) -> Result<(u32, u32)> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        pump_async_textures(context)?;
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }
        if context.texture_pipeline.pending().contains_key(&texture) {
            return Err(Error::NotReady);
        }
        let handle = texture.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let completed =
            completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
        advance_texture_residency(context, completed)?;
        record_texture_ready(context, texture, completed);
        let total = context
            .texture_pipeline
            .submitted()
            .get(&texture)
            .map(|info| info.total)
            .ok_or(Error::InvalidContext)?;
        let resident = context
            .texture_pipeline
            .published()
            .get(&texture)
            .copied()
            .unwrap_or(0);
        Ok((resident, total))
    })
}

/// Sets the contiguous coarse mip count exposed through the stable texture binding.
///
/// Decreasing the count evicts finer mips logically. Increasing it re-admits already uploaded
/// levels as their transfer completion permits. The request remains recorded when `NotReady`.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn set_texture_residency(
    context: ContextHandle,
    texture: TextureHandle,
    resident_mips: u32,
) -> Result<()> {
    super::result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        pump_async_textures(context)?;
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        if context.texture_pipeline.pending().contains_key(&texture) {
            return Err(Error::NotReady);
        }
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }
        let total = context
            .texture_pipeline
            .submitted()
            .get(&texture)
            .map(|info| info.total)
            .ok_or(Error::InvalidContext)?;
        if resident_mips == 0 || resident_mips > total {
            return Err(Error::InvalidArgument);
        }
        context
            .texture_pipeline
            .targets_mut()
            .insert(texture, resident_mips);
        let completed =
            completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
        advance_texture_residency(context, completed)?;
        record_texture_ready(context, texture, completed);
        let published = context
            .texture_pipeline
            .published()
            .get(&texture)
            .copied()
            .unwrap_or(0);
        if published == resident_mips {
            Ok(())
        } else {
            Err(Error::NotReady)
        }
    }))
}

pub(super) fn progress_texture_upload_events(context: &mut ContextState) -> Result<()> {
    pump_async_textures(context)?;
    if context.identity.health() == ContextHealth::Lost {
        return Err(Error::DeviceLost);
    }
    // Nothing submitted means no residency can advance; skip the native
    // completion query so fabricated or pre-submission states keep their
    // existing behavior.
    if context.texture_pipeline.ready().is_empty()
        && context.texture_pipeline.submitted().is_empty()
    {
        return Ok(());
    }
    let completed =
        completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
    // Fine levels can complete after required readiness leaves the ready map.
    // Residency has to advance on every ordinary poll so deferred mips publish.
    advance_texture_residency(context, completed)?;
    if context.texture_pipeline.ready().is_empty() {
        return Ok(());
    }
    let ready: Vec<_> = context.texture_pipeline.ready().keys().copied().collect();
    for texture in ready {
        record_texture_ready(context, texture, completed);
    }
    Ok(())
}

pub(super) fn update_native_texture_region(
    context: &mut NativeContext,
    texture: &mut NativeTexture,
    region: &TextureRegion<'_>,
) -> std::result::Result<ez_gfx_hal::CompletionToken, ez_gfx_hal::AllocationError> {
    // The manager traits synchronously copy the borrowed region before returning its token.
    context.update_texture_region(texture, region)
}

pub(super) fn validate_texture_update(
    format: TextureFormat,
    width: u32,
    height: u32,
    mip_count: u32,
    region: TextureRegion<'_>,
) -> Result<()> {
    // Validation happens before native admission so rejected slices are never retained.
    ez_gfx_hal::validate_texture_region(format, width, height, mip_count, region)
        .map_err(|_| Error::InvalidArgument)
}

pub(super) fn map_texture_update_error(error: ez_gfx_hal::AllocationError) -> Error {
    // Native staging exhaustion is transient backpressure, not permanent texture failure.
    if error == ez_gfx_hal::AllocationError::OutOfMemory {
        Error::QueueFull
    } else {
        map_allocation(error)
    }
}

/// Copies and asynchronously uploads one validated texture sub-region.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn update_texture_region(
    context: ContextHandle,
    texture: TextureHandle,
    region: TextureRegion<'_>,
) -> Result<()> {
    super::result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        pump_async_textures(context)?;
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        if context.texture_pipeline.pending().contains_key(&texture) {
            return Err(Error::NotReady);
        }
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }

        let completed =
            completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
        advance_texture_residency(context, completed)?;
        let info = context
            .texture_pipeline
            .submitted()
            .get(&texture)
            .copied()
            .ok_or(Error::InvalidContext)?;
        context
            .texture_registry
            .resident_mips(info.id)
            .map_err(map_texture)?;
        validate_texture_update(info.format, info.width, info.height, info.total, region)?;
        let published = context
            .texture_pipeline
            .published()
            .get(&texture)
            .copied()
            .unwrap_or(0);
        let touches_view = published != 0 && region.mip_level >= info.total - published;

        let native = context
            .textures
            .get_mut(&texture)
            .ok_or(Error::InvalidContext)?;
        // The backend owns the copied staging bytes after this call returns.
        let completion = update_native_texture_region(&mut context.native, native, &region)
            .map_err(map_texture_update_error)?;
        context
            .texture_pipeline
            .last_transfer_mut()
            .insert(texture, completion);
        // Writes outside the published coarse view keep sampling available while finer work runs.
        if touches_view {
            context
                .texture_pipeline
                .ready_mut()
                .insert(texture, completion);
            // A region rewrite supersedes the initial upload size for pending diagnostics.
            context
                .texture_pipeline
                .transfer_bytes_mut()
                .insert(texture, region.bytes.len() as u64);
            context
                .texture_pipeline
                .handoffs_mut()
                .insert(texture, Instant::now());
        }
        context
            .texture_pipeline
            .telemetry()
            .record_staging_bytes(region.bytes.len() as u64);
        Ok(())
    }))
}

/// Returns context-wide lock-free texture pipeline counters.
///
/// # Errors
///
/// Returns an error when the context handle is invalid, stale, or unhealthy.
pub fn texture_upload_telemetry(context: ContextHandle) -> Result<TextureUploadTelemetrySnapshot> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        // Relaxed per-counter reads are monotonic diagnostics, not a transactional sample.
        Ok(context.texture_pipeline.telemetry().snapshot())
    })
}

fn retire_live_texture(context: &mut ContextState, texture: TextureHandle) -> Result<()> {
    let completion = context
        .texture_pipeline
        .last_transfer()
        .get(&texture)
        .copied()
        .ok_or(Error::InvalidContext)?;
    let id = context
        .texture_pipeline
        .submitted()
        .get(&texture)
        .map(|info| info.id)
        .ok_or(Error::InvalidContext)?;
    let binding = context
        .texture_registry
        .reserved_binding(id)
        .map_err(map_texture)?;
    context
        .identity
        .resolve(texture.packed(), ResourceKind::Texture)
        .map_err(map_lifecycle)?;
    context.texture_registry.retire(id).map_err(map_texture)?;
    context
        .identity
        .remove(texture.packed(), ResourceKind::Texture)
        .map_err(map_lifecycle)?;
    let native = context
        .textures
        .remove(&texture)
        .ok_or(Error::InvalidContext)?;
    cancel_native_texture_transfers(&native);
    // Cancelled transfers never reach their fence values, so release both
    // ledgers here rather than leaving them for completion reclamation.
    let (required_bytes, fine_bytes) = context.texture_pipeline.forget_transfer(texture);
    context.transfer_pool.release_texture(required_bytes);
    context.transfer_pool.release_background(fine_bytes);
    context.texture_pipeline.remove_submitted(texture);
    context.retired_textures.push(RetiredTexture {
        id,
        binding,
        native,
        completion,
    });
    Ok(())
}

fn remove_pending_decode_state(
    context: &mut ContextState,
    texture: TextureHandle,
) -> Option<PendingUpload> {
    // Only decoded-but-unsubmitted results hold a releasable reservation;
    // queued entries have none yet and active decodes release on collection.
    let had_decoded = context.texture_pipeline.take_decoded(texture).is_some();
    let pending = context.texture_pipeline.remove_pending(texture);
    if had_decoded {
        context
            .transfer_pool
            .release_texture(DECODE_RESERVATION_BYTES);
    }
    pending
}

/// Cancels a texture before or after native transfer-worker admission.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn cancel_texture_load(context: ContextHandle, texture: TextureHandle) -> Result<()> {
    super::result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let phase = if context.texture_pipeline.pending().contains_key(&texture) {
            RuntimePhase::Decode
        } else if context.texture_pipeline.ready().contains_key(&texture) {
            RuntimePhase::Upload
        } else {
            return Err(Error::InvalidArgument);
        };
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        if let Some(id) = context
            .texture_pipeline
            .pending()
            .get(&texture)
            .map(|pending| pending.id)
        {
            retire_pending_texture_binding(context, id)?;
            context
                .identity
                .remove(texture.packed(), ResourceKind::Texture)
                .map_err(map_lifecycle)?;
            let pending =
                remove_pending_decode_state(context, texture).ok_or(Error::InvalidContext)?;
            pending.cancelled.store(true, Ordering::Release);
        } else {
            retire_live_texture(context, texture)?;
        }
        let record = runtime_record(context, texture.into_raw(), phase, Err(Error::Cancelled));
        context.observability.push_event(record);
        context.upload_events.push(UploadEvent {
            resource: UploadResource::Texture(texture),
            status: UploadStatus::Cancelled,
        });
        Ok(())
    }))
}

/// Unloads a texture or cancels its queued/native work.
#[cfg(feature = "ffi")]
pub fn unload_texture(context: ContextHandle, texture: TextureHandle) {
    let _ = with_context_mut(context, |context| {
        pump_async_textures(context)?;
        let pending = context
            .texture_pipeline
            .pending()
            .get(&texture)
            .map(|pending| pending.id);
        let failed = context.texture_failures.contains_key(&texture);
        if pending.is_none() && !failed && !context.textures.contains_key(&texture) {
            return Err(Error::InvalidContext);
        }
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        if let Some(id) = pending {
            retire_pending_texture_binding(context, id)?;
            context
                .identity
                .remove(texture.packed(), ResourceKind::Texture)
                .map_err(map_lifecycle)?;
            let pending =
                remove_pending_decode_state(context, texture).ok_or(Error::InvalidContext)?;
            pending.cancelled.store(true, Ordering::Release);
            return Ok(());
        }
        if failed {
            context
                .identity
                .remove(texture.packed(), ResourceKind::Texture)
                .map_err(map_lifecycle)?;
            context.texture_failures.remove(&texture);
            return Ok(());
        }
        retire_live_texture(context, texture)
    });
}

#[cfg(test)]
mod sampler_tests {
    use super::*;
    use ez_gfx_hal::{SamplerAddressMode, SamplerFilter};

    fn sampler(anisotropy: f32) -> ez_gfx_hal::TextureSamplerDesc {
        ez_gfx_hal::TextureSamplerDesc {
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            max_anisotropy: anisotropy,
            address_u: SamplerAddressMode::Clamp,
            address_v: SamplerAddressMode::Clamp,
            address_w: SamplerAddressMode::Clamp,
        }
    }

    #[test]
    fn sampler_validation_rejects_nonfinite_and_out_of_range_anisotropy() {
        // Mirrors the C boundary (`ez_gfx_texture_load` range check): NaN would
        // otherwise disable Vulkan anisotropy while Metal clamps it to 16.
        for anisotropy in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -1.0,
            0.999,
            16.001,
            17.0,
        ] {
            assert_eq!(
                validate_texture_sampler(&sampler(anisotropy)),
                Err(Error::InvalidArgument),
                "anisotropy {anisotropy} must be rejected"
            );
        }
    }

    #[test]
    fn sampler_validation_accepts_shared_boundary_range() {
        for anisotropy in [1.0, 2.0, 8.0, 16.0] {
            assert!(
                validate_texture_sampler(&sampler(anisotropy)).is_ok(),
                "anisotropy {anisotropy} must be accepted"
            );
        }
    }
}
