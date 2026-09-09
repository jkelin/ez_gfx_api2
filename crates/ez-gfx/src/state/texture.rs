use crate::Result;

#[cfg(windows)]
use super::Dx12Context;
#[cfg(target_vendor = "apple")]
use super::MetalContext;
use super::{
    Arc, AtomicBool, CompletionToken, ContextHandle, ContextState, DecodedTextureJob, Error,
    ImageMip, Instant, NativeContext, NativeTexture, Ordering, PendingTexture, QueueKind,
    ResourceKind, RetiredTexture, RuntimePhase, TextureDecoder, TextureDestination, TextureError,
    TextureFormat, TextureHandle, TextureId, TextureRegion, TextureSource,
    TextureUploadTelemetrySnapshot, UploadEvent, UploadResource, UploadStatus, VulkanContext,
    completed_texture_transfer_native, destroy_native_texture, generate_mips, map_allocation,
    map_lifecycle, map_texture, poll_native_frame_completion, runtime_record, runtime_status,
    with_context_mut,
};
use ez_gfx_runtime::ContextHealth;

#[derive(Clone, Copy, Debug)]
/// Dimensions, mip policy, and sampling configuration for a texture.
pub struct TextureConfig {
    /// Base width in pixels.
    pub width: u32,
    /// Base height in pixels.
    pub height: u32,
    /// Requested mip count, or zero to use the decoded chain.
    pub mip_count: u32,
    /// Requested GPU storage or automatic capability-based selection.
    pub destination: TextureDestination,
    /// Texture sampling configuration.
    pub sampler: ez_gfx_hal::TextureSamplerDesc,
}

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

/// Queues texture decode, mip generation, and transfer preparation without blocking the caller.
///
/// The input bytes are copied before this function returns; callers retain no asynchronous
/// lifetime obligation.
///
/// # Errors
///
/// Returns an error for invalid input, an unvalidated sampler, exhausted handles,
/// worker backpressure, or a stale context.
pub fn load_texture(
    context: ContextHandle,
    source: TextureSource,
    bytes: &[u8],
    generate: bool,
    config: &TextureConfig,
) -> Result<TextureHandle> {
    if bytes.is_empty() || bytes.len() > ez_gfx_runtime::texture::MAX_TEXTURE_BYTES {
        return Err(Error::InvalidArgument);
    }
    validate_texture_sampler(&config.sampler)?;
    let owned = bytes.to_vec().into_boxed_slice();
    let config = *config;

    with_context_mut(context, |context| {
        reclaim_retired_textures(context)?;
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
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
        let typed = TextureHandle::from_packed(handle).map_err(|_| Error::NativeFailure)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        context.pending_textures.insert(
            typed,
            PendingTexture {
                id: texture,
                cancelled: cancelled.clone(),
                config,
                admitted_at: Instant::now(),
            },
        );
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
        let prepared = match TextureDecoder::prepare(source, compression, config.destination) {
            Ok(prepared) => prepared,
            Err(error) => {
                context.pending_textures.remove(&typed);
                let _ = context.identity.remove(handle, ResourceKind::Texture);
                let _ = context.texture_registry.cancel_upload(texture);
                return Err(map_texture(error));
            }
        };
        let telemetry = context.texture_telemetry.clone();
        let ready = context.async_textures.ready_tx.clone();
        #[cfg(test)]
        let decode_gate = context.async_textures.decode_gate.clone();
        let submitted = context.async_textures.pool.submit(move || {
            #[cfg(test)]
            if let Some(gate) = decode_gate {
                gate.wait();
            }
            let started = Instant::now();
            let decoded = if cancelled.load(Ordering::Acquire) {
                Err(TextureError::NotFound)
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    prepared.decode(&owned).and_then(|texture| {
                        if generate {
                            generate_mips(texture)
                        } else {
                            Ok(texture)
                        }
                    })
                }))
                .unwrap_or(Err(TextureError::InvalidData))
            };
            let decode_microseconds =
                u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
            telemetry.record_decode(decode_microseconds);
            // A cancelled request has already retired its public and registry handles.
            if !cancelled.load(Ordering::Acquire) {
                let _ = ready.send(DecodedTextureJob {
                    handle: typed,
                    decoded,
                });
            }
        });
        if submitted.is_err() {
            context.pending_textures.remove(&typed);
            let _ = context.identity.remove(handle, ResourceKind::Texture);
            let _ = context.texture_registry.cancel_upload(texture);
            return Err(Error::QueueFull);
        }
        let record = runtime_record(context, typed.into_raw(), RuntimePhase::Admission, Ok(()));
        context.observability.push_event(record);
        context.upload_events.push(UploadEvent {
            resource: UploadResource::Texture(typed),
            status: UploadStatus::SourceStaged,
        });
        Ok(typed)
    })
}

fn record_texture_failure(context: &mut ContextState, handle: TextureHandle, error: Error) {
    // Decode failures retain the public handle long enough for deterministic polling.
    context.texture_failures.insert(handle, error);
    let record = runtime_record(context, handle.into_raw(), RuntimePhase::Decode, Err(error));
    context.observability.push_event(record);
    context.upload_events.push(UploadEvent {
        resource: UploadResource::Texture(handle),
        status: UploadStatus::Failed(runtime_status(Err(error))),
    });
}

fn fail_texture_job(
    context: &mut ContextState,
    handle: TextureHandle,
    id: TextureId,
    error: Error,
) {
    let _ = context.texture_registry.cancel_upload(id);
    record_texture_failure(context, handle, error);
}

/// Synchronously cancels every queued decode, reusing destroy's drain loop.
pub(super) fn cancel_all_pending_textures(context: &mut ContextState) {
    for (_, pending) in context.pending_textures.drain() {
        pending.cancelled.store(true, Ordering::Release);
        let _ = context.texture_registry.cancel_upload(pending.id);
    }
}

/// Marks terminal loss once, emits terminal upload events, and sweeps queued decodes.
pub(super) fn note_device_lost(context: &mut ContextState) {
    let pending: Vec<_> = context.pending_textures.keys().copied().collect();
    let ready: Vec<_> = context.texture_ready.keys().copied().collect();
    for texture in pending.into_iter().chain(ready) {
        context.upload_events.push(UploadEvent {
            resource: UploadResource::Texture(texture),
            status: UploadStatus::Failed(super::RuntimeStatus::DeviceLost),
        });
    }
    let _ = context.identity.mark_lost();
    cancel_all_pending_textures(context);
}

pub(super) fn pump_async_textures(context: &mut ContextState) -> Result<usize> {
    reclaim_retired_textures(context)?;
    let mut completed = 0;
    while let Ok(job) = context.async_textures.ready_rx.try_recv() {
        let Some(pending) = context.pending_textures.remove(&job.handle) else {
            continue;
        };
        if pending.cancelled.load(Ordering::Acquire) {
            continue;
        }
        let decoded = match job.decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                fail_texture_job(context, job.handle, pending.id, map_texture(error));
                completed += 1;
                continue;
            }
        };
        if (pending.config.width != 0 && decoded.width != pending.config.width)
            || (pending.config.height != 0 && decoded.height != pending.config.height)
            || (pending.config.mip_count != 0 && decoded.mip_count != pending.config.mip_count)
        {
            fail_texture_job(context, job.handle, pending.id, Error::InvalidArgument);
            completed += 1;
            continue;
        }
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
        let binding = context
            .texture_registry
            .reserved_binding(pending.id)
            .map_err(map_texture)?;
        let created = match &mut context.native {
            NativeContext::Vulkan(native) => native
                .create_texture(decoded.format, &mips, binding, pending.config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Vulkan(texture), tokens)),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native
                .create_texture(decoded.format, &mips, binding, pending.config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Dx12(texture), tokens)),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native
                .create_texture(decoded.format, &mips, binding, pending.config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Metal(texture), tokens)),
        };
        let (native, completions) = match created {
            Ok(created) => created,
            Err(error) => {
                // Pump records per-job failures without returning, so a loss observed
                // here must sweep directly or siblings would stay NotReady.
                let mapped = map_allocation(error);
                fail_texture_job(context, job.handle, pending.id, mapped);
                if mapped == Error::DeviceLost {
                    note_device_lost(context);
                }
                completed += 1;
                continue;
            }
        };
        if completions.len() != decoded.mip_count as usize {
            rollback_texture_upload(context, pending.id, native)?;
            record_texture_failure(context, job.handle, Error::NativeFailure);
            completed += 1;
            continue;
        }
        let last = *completions.last().ok_or(Error::NativeFailure)?;
        let mut completions = completions.into_iter();
        let first = completions.next().ok_or(Error::NativeFailure)?;
        let tracked = context
            .texture_registry
            .mark_submitted(pending.id, first)
            .map_err(map_texture)
            .and_then(|()| {
                for (index, completion) in completions.enumerate() {
                    let resident_mips = u32::try_from(index)
                        .ok()
                        .and_then(|index| index.checked_add(2))
                        .ok_or(Error::NativeFailure)?;
                    context
                        .texture_registry
                        .mark_mips_submitted(pending.id, resident_mips, completion)
                        .map_err(map_texture)?;
                }
                Ok(())
            });
        if let Err(error) = tracked {
            rollback_texture_upload(context, pending.id, native)?;
            record_texture_failure(context, job.handle, error);
            completed += 1;
            continue;
        }
        context.texture_telemetry.record_queue_latency(
            u64::try_from(submitted_at.duration_since(pending.admitted_at).as_micros())
                .unwrap_or(u64::MAX),
        );
        let staging_bytes = decoded.mips.iter().fold(0_u64, |total, mip| {
            total.saturating_add(mip.bytes.len() as u64)
        });
        context
            .texture_telemetry
            .record_staging_bytes(staging_bytes);
        context.textures.insert(
            job.handle,
            (
                pending.id,
                native,
                decoded.width,
                decoded.height,
                decoded.mip_count,
            ),
        );
        context.texture_formats.insert(job.handle, decoded.format);
        context.texture_published_mips.insert(job.handle, 0);
        context
            .texture_residency_targets
            .insert(job.handle, decoded.mip_count);
        context.texture_last_transfer.insert(job.handle, last);
        context.texture_ready.insert(job.handle, first);
        context.texture_handoffs.insert(job.handle, submitted_at);
        let decode = runtime_record(context, job.handle.into_raw(), RuntimePhase::Decode, Ok(()));
        context.observability.push_event(decode);
        let upload = runtime_record(context, job.handle.into_raw(), RuntimePhase::Upload, Ok(()));
        context.observability.push_event(upload);
        completed += 1;
    }
    reclaim_retired_textures(context)?;
    Ok(completed)
}

pub(super) fn rollback_texture_upload(
    context: &mut ContextState,
    texture: TextureId,
    native: NativeTexture,
) -> Result<()> {
    let completion = native_texture_last_completion(&native)?;
    cancel_native_texture_transfers(&native);
    context
        .texture_registry
        .retire(texture)
        .map_err(map_texture)?;
    context.retired_textures.push(RetiredTexture {
        id: texture,
        native,
        completion,
    });
    Ok(())
}

fn native_texture_last_completion(texture: &NativeTexture) -> Result<CompletionToken> {
    let value = match texture {
        NativeTexture::Vulkan(texture) => texture.last_transfer_value(),
        #[cfg(windows)]
        NativeTexture::Dx12(texture) => texture.last_transfer_value(),
        #[cfg(target_vendor = "apple")]
        NativeTexture::Metal(texture) => texture.last_transfer_value(),
    };
    CompletionToken::new(QueueKind::TextureTransfer, value).map_err(|_| Error::NativeFailure)
}

pub(super) fn mip_range_completion(values: &[u64], resident_mips: u32) -> Option<u64> {
    // A hidden fine update must not delay a coarse view; exposed updated mips use their max.
    let count = usize::try_from(resident_mips).ok()?;
    let range = values.get(values.len().checked_sub(count)?..)?;
    if range.contains(&0) {
        return None;
    }
    range.iter().copied().max()
}

fn native_texture_mip_values(texture: &NativeTexture) -> &[u64] {
    match texture {
        NativeTexture::Vulkan(texture) => texture.mip_transfer_values(),
        #[cfg(windows)]
        NativeTexture::Dx12(texture) => texture.mip_transfer_values(),
        #[cfg(target_vendor = "apple")]
        NativeTexture::Metal(texture) => texture.mip_transfer_values(),
    }
}

fn cancel_native_texture_transfers(texture: &NativeTexture) {
    match texture {
        NativeTexture::Vulkan(texture) => VulkanContext::cancel_texture_transfers(texture),
        #[cfg(windows)]
        NativeTexture::Dx12(texture) => Dx12Context::cancel_texture_transfers(texture),
        #[cfg(target_vendor = "apple")]
        NativeTexture::Metal(texture) => MetalContext::cancel_texture_transfers(texture),
    }
}

fn native_texture_retirement_ready(
    context: &NativeContext,
    texture: &NativeTexture,
    completion: CompletionToken,
) -> std::result::Result<bool, ez_gfx_hal::AllocationError> {
    match (context, texture) {
        (NativeContext::Vulkan(context), NativeTexture::Vulkan(_)) => {
            context.texture_retirement_ready(completion)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeTexture::Dx12(_)) => {
            context.texture_retirement_ready(completion)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeTexture::Metal(_)) => {
            context.texture_retirement_ready(completion)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

fn reclaim_retired_textures(context: &mut ContextState) -> Result<()> {
    let mut index = 0;
    while index < context.retired_textures.len() {
        let retired = &context.retired_textures[index];
        if !native_texture_retirement_ready(&context.native, &retired.native, retired.completion)
            .map_err(map_allocation)?
        {
            index += 1;
            continue;
        }
        let retired = context.retired_textures.swap_remove(index);
        destroy_native_texture(&mut context.native, retired.native).map_err(map_allocation)?;
        context
            .texture_registry
            .release_retired(retired.id)
            .map_err(map_texture)?;
    }
    Ok(())
}

fn publish_native_texture_mips(
    context: &mut NativeContext,
    texture: &mut NativeTexture,
    resident_mips: u32,
) -> std::result::Result<(), ez_gfx_hal::AllocationError> {
    // Variant mismatches indicate corrupt context-owned state, never caller input.
    match (context, texture) {
        (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
            context.publish_texture_mips(texture, resident_mips)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
            context.publish_texture_mips(texture, resident_mips)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
            context.publish_texture_mips(texture, resident_mips)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
}

/// Maps a mip-publication outcome onto residency progress.
///
/// Only fence-gate contention is transient: `Ok(false)` keeps the recorded target
/// and surfaces `NotReady` until the next poll. Allocation, validation, capability,
/// and device errors stay terminal through the shared mapping.
fn publish_advance_step(
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
    let advances = context
        .textures
        .iter()
        .filter_map(|(handle, (id, texture, _, _, total))| {
            let available = context.texture_registry.resident_mips(*id).ok()?;
            let target = context
                .texture_residency_targets
                .get(handle)
                .copied()
                .unwrap_or(*total)
                .min(available);
            let published = context
                .texture_published_mips
                .get(handle)
                .copied()
                .unwrap_or(0);
            let native_available = native_texture_mip_values(texture)
                .iter()
                .rev()
                .take_while(|value| **value != 0 && **value <= completed)
                .count();
            // Continue coarse-first expansion around a pending hidden update, but never
            // silently shrink an already published view while its region overwrite completes.
            let target = if target > published {
                target.min(u32::try_from(native_available).ok()?.max(published))
            } else {
                target
            };
            let ready = mip_range_completion(native_texture_mip_values(texture), target)?;
            (target != published && ready <= completed).then_some((*handle, target))
        })
        .collect::<Vec<_>>();
    if advances.is_empty() {
        return Ok(());
    }

    // Only DX12's descriptor fence query can fail; preserve its device-loss error.
    let descriptors_ready = match &context.native {
        NativeContext::Vulkan(context) => context.texture_descriptor_update_ready(),
        #[cfg(windows)]
        NativeContext::Dx12(context) => context
            .texture_descriptor_update_ready()
            .map_err(map_allocation)?,
        #[cfg(target_vendor = "apple")]
        NativeContext::Metal(context) => context.texture_descriptor_update_ready(),
    };
    if !descriptors_ready {
        return Ok(());
    }
    for (handle, resident_mips) in advances {
        let (_, texture, _, _, _) = context
            .textures
            .get_mut(&handle)
            .ok_or(Error::InvalidContext)?;
        let published = publish_advance_step(publish_native_texture_mips(
            &mut context.native,
            texture,
            resident_mips,
        ))?;
        if published {
            context.texture_published_mips.insert(handle, resident_mips);
        }
    }
    Ok(())
}

pub(super) fn record_texture_ready(
    context: &mut ContextState,
    texture: TextureHandle,
    completed: u64,
) {
    // A completed copy is not sample-ready until its first coarse descriptor is published.
    // Region updates keep an existing published view but still require their new copy token.
    let is_ready = context
        .texture_ready
        .get(&texture)
        .is_some_and(|token| token.value <= completed)
        && context
            .texture_published_mips
            .get(&texture)
            .is_some_and(|mips| *mips != 0);
    if !is_ready {
        return;
    }

    context.texture_ready.remove(&texture);
    if let Some(submitted_at) = context.texture_handoffs.remove(&texture) {
        context.texture_telemetry.record_handoff_latency(
            u64::try_from(submitted_at.elapsed().as_micros()).unwrap_or(u64::MAX),
        );
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
/// This observes the pool sized at creation from `ContextOptions::texture_decode_workers`
/// (zero selects the default topology), letting FFI callers verify the C descriptor
/// value arrived without a C-side query export.
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
        u32::try_from(context.async_textures.pool.thread_count()).map_err(|_| Error::NativeFailure)
    })
}

/// Returns a texture binding index.
///
/// # Errors
///
/// Returns an error when the context or texture handle is invalid or stale.
pub fn texture_binding(context: ContextHandle, texture: TextureHandle) -> Result<u32> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        pump_async_textures(context)?;
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }
        if context.pending_textures.contains_key(&texture) {
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
        if context.texture_ready.contains_key(&texture) {
            return Err(Error::NotReady);
        }
        let (id, _, _, _, _) = context
            .textures
            .get(&texture)
            .ok_or(Error::InvalidContext)?;
        context
            .texture_registry
            .binding_index(*id)
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
        if context.pending_textures.contains_key(&texture) {
            return Err(Error::NotReady);
        }
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let (_, _, width, height, _) = context
            .textures
            .get(&texture)
            .ok_or(Error::InvalidContext)?;
        Ok((*width, *height))
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
        if context.pending_textures.contains_key(&texture) {
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
        let (_, _, _, _, total) = context
            .textures
            .get(&texture)
            .ok_or(Error::InvalidContext)?;
        let resident = context
            .texture_published_mips
            .get(&texture)
            .copied()
            .unwrap_or(0);
        Ok((resident, *total))
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
        if context.pending_textures.contains_key(&texture) {
            return Err(Error::NotReady);
        }
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }
        let total = context
            .textures
            .get(&texture)
            .map(|(_, _, _, _, total)| *total)
            .ok_or(Error::InvalidContext)?;
        if resident_mips == 0 || resident_mips > total {
            return Err(Error::InvalidArgument);
        }
        context
            .texture_residency_targets
            .insert(texture, resident_mips);
        let completed =
            completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
        advance_texture_residency(context, completed)?;
        record_texture_ready(context, texture, completed);
        let published = context
            .texture_published_mips
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

/// Progresses every owner-thread texture upload and publishes resulting events.
pub(super) fn progress_texture_upload_events(context: &mut ContextState) -> Result<()> {
    pump_async_textures(context)?;
    if context.identity.health() == ContextHealth::Lost {
        return Err(Error::DeviceLost);
    }
    // An uninitialized context has no texture transfer timeline. Pending CPU decode alone
    // still needs polling, but querying native completion before admission is an error.
    if context.texture_ready.is_empty() {
        return Ok(());
    }
    let completed =
        completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
    advance_texture_residency(context, completed)?;
    let ready: Vec<_> = context.texture_ready.keys().copied().collect();
    for texture in ready {
        record_texture_ready(context, texture, completed);
    }
    Ok(())
}

fn update_native_texture_region(
    context: &mut NativeContext,
    texture: &mut NativeTexture,
    region: &TextureRegion<'_>,
) -> std::result::Result<ez_gfx_hal::CompletionToken, ez_gfx_hal::AllocationError> {
    // Native methods synchronously copy the borrowed region before returning its token.
    match (context, texture) {
        (NativeContext::Vulkan(context), NativeTexture::Vulkan(texture)) => {
            context.update_texture_region(texture, region)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeTexture::Dx12(texture)) => {
            context.update_texture_region(texture, region)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeTexture::Metal(texture)) => {
            context.update_texture_region(texture, region)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => Err(ez_gfx_hal::AllocationError::NativeFailure),
    }
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
        if context.pending_textures.contains_key(&texture) {
            return Err(Error::NotReady);
        }
        if let Some(error) = context.texture_failures.get(&texture).copied() {
            return Err(error);
        }

        let completed =
            completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
        advance_texture_residency(context, completed)?;
        let (id, _, width, height, mip_count) = context
            .textures
            .get(&texture)
            .ok_or(Error::InvalidContext)?;
        context
            .texture_registry
            .resident_mips(*id)
            .map_err(map_texture)?;
        let format = context
            .texture_formats
            .get(&texture)
            .copied()
            .ok_or(Error::InvalidContext)?;
        validate_texture_update(format, *width, *height, *mip_count, region)?;
        let published = context
            .texture_published_mips
            .get(&texture)
            .copied()
            .unwrap_or(0);
        let touches_view = published != 0 && region.mip_level >= *mip_count - published;

        let (_, native, _, _, _) = context
            .textures
            .get_mut(&texture)
            .ok_or(Error::InvalidContext)?;
        // The backend owns the copied staging bytes after this call returns.
        let completion = update_native_texture_region(&mut context.native, native, &region)
            .map_err(map_texture_update_error)?;
        context.texture_last_transfer.insert(texture, completion);
        // Writes outside the published coarse view keep sampling available while finer work runs.
        if touches_view {
            context.texture_ready.insert(texture, completion);
            context.texture_handoffs.insert(texture, Instant::now());
        }
        context
            .texture_telemetry
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
        Ok(context.texture_telemetry.snapshot())
    })
}

fn retire_live_texture(context: &mut ContextState, texture: TextureHandle) -> Result<()> {
    let completion = context
        .texture_last_transfer
        .get(&texture)
        .copied()
        .ok_or(Error::InvalidContext)?;
    let id = context
        .textures
        .get(&texture)
        .map(|(id, _, _, _, _)| *id)
        .ok_or(Error::InvalidContext)?;
    context
        .identity
        .resolve(texture.packed(), ResourceKind::Texture)
        .map_err(map_lifecycle)?;
    context.texture_registry.retire(id).map_err(map_texture)?;
    context
        .identity
        .remove(texture.packed(), ResourceKind::Texture)
        .map_err(map_lifecycle)?;
    let (_, native, _, _, _) = context
        .textures
        .remove(&texture)
        .ok_or(Error::InvalidContext)?;
    cancel_native_texture_transfers(&native);
    context.texture_ready.remove(&texture);
    context.texture_handoffs.remove(&texture);
    context.texture_formats.remove(&texture);
    context.texture_published_mips.remove(&texture);
    context.texture_residency_targets.remove(&texture);
    context.texture_last_transfer.remove(&texture);
    context.retired_textures.push(RetiredTexture {
        id,
        native,
        completion,
    });
    Ok(())
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
        let phase = if context.pending_textures.contains_key(&texture) {
            RuntimePhase::Decode
        } else if context.texture_ready.contains_key(&texture) {
            RuntimePhase::Upload
        } else {
            return Err(Error::InvalidArgument);
        };
        context
            .identity
            .resolve(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        if let Some(id) = context
            .pending_textures
            .get(&texture)
            .map(|pending| pending.id)
        {
            context
                .texture_registry
                .cancel_upload(id)
                .map_err(map_texture)?;
            context
                .identity
                .remove(texture.packed(), ResourceKind::Texture)
                .map_err(map_lifecycle)?;
            let pending = context
                .pending_textures
                .remove(&texture)
                .ok_or(Error::InvalidContext)?;
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
            .pending_textures
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
            context
                .texture_registry
                .cancel_upload(id)
                .map_err(map_texture)?;
            context
                .identity
                .remove(texture.packed(), ResourceKind::Texture)
                .map_err(map_lifecycle)?;
            let pending = context
                .pending_textures
                .remove(&texture)
                .ok_or(Error::InvalidContext)?;
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
