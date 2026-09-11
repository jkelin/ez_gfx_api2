use super::texture::{
    TEXTURE_DECODE_RESERVATION, fail_texture_job, reclaim_retired_textures, record_texture_failure,
    rollback_texture_upload, schedule_texture_decodes,
};
use super::{
    ContextState, DecodedTexture, Error, NativeContext, NativeTexture, PendingTexture, Result,
    RuntimePhase, TextureHandle, TextureTransferWork, completed_texture_transfer_native,
    map_allocation, map_texture, note_device_lost, runtime_record,
};
use ez_gfx_hal::ImageMip;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Advances every ready stage; empty stages are no-ops and native failures propagate.
pub(super) fn pump_async_textures(context: &mut ContextState) -> Result<usize> {
    reclaim_retired_textures(context)?;
    reclaim_texture_transfer_reservations(context)?;
    collect_decode_results(context);

    let completed = submit_decoded_textures(context)?;
    schedule_texture_decodes(context)?;
    reclaim_retired_textures(context)?;
    Ok(completed)
}

/// Releases credits only after final mip completion; an empty transfer set avoids a native query.
fn reclaim_texture_transfer_reservations(context: &mut ContextState) -> Result<()> {
    if context.texture_transfer_work.is_empty() {
        return Ok(());
    }

    let completed =
        completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
    // Transfer reservations survive initial coarse publication: staging is reusable only after
    // the final mip token, so releasing on `DeviceReady` would undercount fine uploads.
    let retired = context
        .texture_transfer_work
        .iter()
        .filter_map(|(handle, work)| (work.completion.value <= completed).then_some(*handle))
        .collect::<Vec<_>>();
    for handle in retired {
        if let Some(work) = context.texture_transfer_work.remove(&handle) {
            context.async_textures.working_bytes = context
                .async_textures
                .working_bytes
                .saturating_sub(work.bytes);
            context.texture_transfer_bytes.remove(&handle);
        }
    }
    Ok(())
}

/// Collects every ready result; cancelled jobs release their otherwise orphaned credit.
pub(super) fn collect_decode_results(context: &mut ContextState) {
    while let Ok(job) = context.async_textures.ready_rx.try_recv() {
        context.async_textures.active = context.async_textures.active.saturating_sub(1);
        if let Some(pending) = context.pending_textures.get_mut(&job.handle) {
            pending.decoded_bytes = job.decoded.as_ref().ok().map(|decoded| {
                decoded.mips.iter().fold(0_u64, |total, mip| {
                    total.saturating_add(mip.bytes.len() as u64)
                })
            });
            context
                .async_textures
                .decoded
                .insert(job.handle, job.decoded);
        } else {
            // Cancelled/stale jobs still report completion internally so their reservation cannot
            // strand the FIFO; their public terminal event was emitted by the cancelling path.
            context.async_textures.working_bytes = context
                .async_textures
                .working_bytes
                .saturating_sub(TEXTURE_DECODE_RESERVATION);
        }
    }
}

/// Preserves FIFO publication by stopping at the first decode that is not ready.
fn submit_decoded_textures(context: &mut ContextState) -> Result<usize> {
    // CPU decode may run before device initialization, but no real native upload may displace
    // a missing or incompletely published fallback descriptor.
    if !context.texture_fallback.is_ready() {
        return Ok(0);
    }
    let mut completed = 0;
    loop {
        let Some(handle) = context.async_textures.order.front().copied() else {
            break;
        };
        let Some(decoded) = context.async_textures.decoded.remove(&handle) else {
            break;
        };
        context.async_textures.order.pop_front();
        let Some(pending) = context.pending_textures.remove(&handle) else {
            release_decode_reservation(context);
            continue;
        };
        if pending.cancelled.load(Ordering::Acquire) {
            release_decode_reservation(context);
            continue;
        }
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                release_decode_reservation(context);
                fail_texture_job(context, handle, pending.id, map_texture(error));
                completed += 1;
                continue;
            }
        };

        submit_decoded_texture(context, handle, &pending, &decoded)?;
        completed += 1;
    }
    Ok(completed)
}

/// Converts one decoded payload to native work; terminal admission failures consume its credit.
fn submit_decoded_texture(
    context: &mut ContextState,
    handle: TextureHandle,
    pending: &PendingTexture,
    decoded: &DecodedTexture,
) -> Result<()> {
    if (pending.config.width != 0 && decoded.width != pending.config.width)
        || (pending.config.height != 0 && decoded.height != pending.config.height)
        || (pending.config.mip_count != 0 && decoded.mip_count != pending.config.mip_count)
    {
        release_decode_reservation(context);
        fail_texture_job(context, handle, pending.id, Error::InvalidArgument);
        return Ok(());
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
            release_decode_reservation(context);
            let mapped = map_allocation(error);
            fail_texture_job(context, handle, pending.id, mapped);
            if mapped == Error::DeviceLost {
                note_device_lost(context);
            }
            return Ok(());
        }
    };
    if completions.len() != decoded.mip_count as usize {
        release_decode_reservation(context);
        rollback_texture_upload(context, pending.id, native)?;
        record_texture_failure(context, handle, Error::NativeFailure);
        return Ok(());
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
        release_decode_reservation(context);
        rollback_texture_upload(context, pending.id, native)?;
        record_texture_failure(context, handle, error);
        return Ok(());
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
        handle,
        (
            pending.id,
            native,
            decoded.width,
            decoded.height,
            decoded.mip_count,
        ),
    );
    context.texture_formats.insert(handle, decoded.format);
    context.texture_published_mips.insert(handle, 0);
    context
        .texture_residency_targets
        .insert(handle, decoded.mip_count);
    context.texture_last_transfer.insert(handle, last);
    context.texture_ready.insert(handle, first);
    context.texture_transfer_bytes.insert(handle, staging_bytes);
    context.texture_transfer_work.insert(
        handle,
        TextureTransferWork {
            completion: last,
            bytes: TEXTURE_DECODE_RESERVATION,
        },
    );
    context.texture_handoffs.insert(handle, submitted_at);
    let decode = runtime_record(context, handle.into_raw(), RuntimePhase::Decode, Ok(()));
    context.observability.push_event(decode);
    let upload = runtime_record(context, handle.into_raw(), RuntimePhase::Upload, Ok(()));
    context.observability.push_event(upload);
    Ok(())
}

/// Saturation keeps cancellation and asynchronous completion races from underflowing diagnostics.
fn release_decode_reservation(context: &mut ContextState) {
    context.async_textures.working_bytes = context
        .async_textures
        .working_bytes
        .saturating_sub(TEXTURE_DECODE_RESERVATION);
}
