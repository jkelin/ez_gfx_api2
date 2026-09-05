use super::{
    Arc, AtomicBool, ContextHandle, ContextState, DecodedTextureJob, EzGfxResult, ImageMip,
    NativeContext, NativeTexture, Ordering, PendingTexture, ResourceKind, RuntimePhase,
    TextureDecoder, TextureError, TextureHandle, TextureId, TextureSource,
    completed_texture_transfer_native, destroy_native_texture, generate_mips, map_allocation,
    map_hal, map_lifecycle, map_texture, runtime_record, wait_native_idle, with_context_mut,
};

#[derive(Clone, Copy, Debug)]
/// Dimensions, mip policy, and sampling configuration for a texture.
pub struct TextureConfig {
    /// Base width in pixels.
    pub width: u32,
    /// Base height in pixels.
    pub height: u32,
    /// Requested mip count, or zero to use the decoded chain.
    pub mip_count: u32,
    /// Texture sampling configuration.
    pub sampler: ez_gfx_hal::TextureSamplerDesc,
}

/// Queues texture decode, mip generation, and transfer preparation without blocking the caller.
///
/// The input bytes are copied before this function returns; callers retain no asynchronous
/// lifetime obligation.
///
/// # Errors
///
/// Returns an error for invalid input, exhausted handles, worker backpressure, or a stale context.
pub fn load_texture(
    context: ContextHandle,
    source: TextureSource,
    bytes: &[u8],
    generate: bool,
    config: &TextureConfig,
) -> Result<TextureHandle, EzGfxResult> {
    if bytes.is_empty() || bytes.len() > ez_gfx_runtime::texture::MAX_TEXTURE_BYTES {
        return Err(EzGfxResult::InvalidArgument);
    }
    let owned = bytes.to_vec().into_boxed_slice();
    let config = *config;

    with_context_mut(context, |context| {
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
        let typed = TextureHandle::from_packed(handle).map_err(|_| EzGfxResult::NativeFailure)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        context.pending_textures.insert(
            typed,
            PendingTexture {
                id: texture,
                cancelled: cancelled.clone(),
                config,
            },
        );
        let ready = context.async_textures.ready_tx.clone();
        #[cfg(test)]
        let decode_gate = context.async_textures.decode_gate.clone();
        let submitted = context
            .async_textures
            .pool
            .submit_sized(owned.len(), move || {
                #[cfg(test)]
                if let Some(gate) = decode_gate {
                    gate.wait();
                }
                let decoded = if cancelled.load(Ordering::Acquire) {
                    Err(TextureError::NotFound)
                } else {
                    std::panic::catch_unwind(|| {
                        TextureDecoder::decode(source, &owned).and_then(|texture| {
                            if generate {
                                generate_mips(texture)
                            } else {
                                Ok(texture)
                            }
                        })
                    })
                    .unwrap_or(Err(TextureError::InvalidData))
                };
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
            return Err(EzGfxResult::QueueFull);
        }
        let record = runtime_record(
            context,
            typed.into_raw(),
            RuntimePhase::Admission,
            EzGfxResult::Ok,
        );
        context.observability.push_event(record);
        Ok(typed)
    })
}

fn fail_texture_job(
    context: &mut ContextState,
    handle: TextureHandle,
    id: TextureId,
    error: EzGfxResult,
) {
    let _ = context.texture_registry.cancel_upload(id);
    context.texture_failures.insert(handle, error);
    let record = runtime_record(context, handle.into_raw(), RuntimePhase::Decode, error);
    context.observability.push_event(record);
}

pub(super) fn pump_async_textures(context: &mut ContextState) -> Result<usize, EzGfxResult> {
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
            fail_texture_job(
                context,
                job.handle,
                pending.id,
                EzGfxResult::InvalidArgument,
            );
            completed += 1;
            continue;
        }
        let mips = decoded
            .mips
            .iter()
            .map(|mip| ImageMip {
                width: mip.width,
                height: mip.height,
                bytes: &mip.rgba8,
            })
            .collect::<Vec<_>>();
        let binding = context
            .texture_registry
            .reserved_binding(pending.id)
            .map_err(map_texture)?;
        let created = match &mut context.native {
            NativeContext::Vulkan(native) => native
                .create_texture_rgba8(&mips, binding, pending.config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Vulkan(texture), tokens)),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native
                .create_texture_rgba8(&mips, binding, pending.config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Dx12(texture), tokens)),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native
                .create_texture_rgba8(&mips, binding, pending.config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Metal(texture), tokens)),
        };
        let (native, completions) = match created {
            Ok(created) => created,
            Err(error) => {
                fail_texture_job(context, job.handle, pending.id, map_allocation(error));
                completed += 1;
                continue;
            }
        };
        let Some(ready) = completions.last().copied() else {
            rollback_texture_upload(context, pending.id, native)?;
            fail_texture_job(context, job.handle, pending.id, EzGfxResult::NativeFailure);
            completed += 1;
            continue;
        };
        let mut completions = completions.into_iter();
        let first = completions.next().ok_or(EzGfxResult::NativeFailure)?;
        context
            .texture_registry
            .mark_submitted(pending.id, first)
            .map_err(map_texture)?;
        for (index, completion) in completions.enumerate() {
            let mip = u32::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(2))
                .ok_or(EzGfxResult::NativeFailure)?;
            context
                .texture_registry
                .mark_mips_submitted(pending.id, mip, completion)
                .map_err(map_texture)?;
        }
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
        context.texture_ready.insert(job.handle, ready);
        let decode = runtime_record(
            context,
            job.handle.into_raw(),
            RuntimePhase::Decode,
            EzGfxResult::Ok,
        );
        context.observability.push_event(decode);
        let upload = runtime_record(
            context,
            job.handle.into_raw(),
            RuntimePhase::Upload,
            EzGfxResult::Ok,
        );
        context.observability.push_event(upload);
        completed += 1;
    }
    Ok(completed)
}

pub(super) fn rollback_texture_upload(
    context: &mut ContextState,
    texture: TextureId,
    native: NativeTexture,
) -> Result<(), EzGfxResult> {
    let idle = wait_native_idle(&mut context.native).map_err(map_hal);
    let destroyed = destroy_native_texture(&mut context.native, native).map_err(map_allocation);
    let canceled = context
        .texture_registry
        .cancel_upload(texture)
        .map_err(map_texture);
    idle.and(destroyed).and(canceled)
}

fn record_texture_ready(context: &mut ContextState, texture: TextureHandle, completed: u64) {
    let is_ready = context
        .texture_ready
        .get(&texture)
        .is_some_and(|token| token.value <= completed);
    if !is_ready {
        return;
    }

    context.texture_ready.remove(&texture);
    let record = runtime_record(
        context,
        texture.into_raw(),
        RuntimePhase::Bind,
        EzGfxResult::Ok,
    );
    context.observability.push_event(record);
}

/// Returns a texture binding index.
///
/// # Errors
///
/// Returns an error when the context or texture handle is invalid or stale.
pub fn texture_binding(context: ContextHandle, texture: TextureHandle) -> Result<u32, EzGfxResult> {
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
            return Err(EzGfxResult::NotReady);
        }
        let handle = texture.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let completed =
            completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
        context
            .texture_registry
            .poll(ez_gfx_hal::QueueKind::TextureTransfer, completed)
            .map_err(map_texture)?;
        record_texture_ready(context, texture, completed);
        let (id, _, _, _, _) = context
            .textures
            .get(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        context
            .texture_registry
            .binding_index(*id)
            .map_err(map_texture)
    })
}

/// Returns resident and total mip counts.
///
/// # Errors
///
/// Returns an error when the context or texture handle is invalid or stale.
pub fn texture_residency(
    context: ContextHandle,
    texture: TextureHandle,
) -> Result<(u32, u32), EzGfxResult> {
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
            return Err(EzGfxResult::NotReady);
        }
        let handle = texture.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let completed =
            completed_texture_transfer_native(&mut context.native).map_err(map_allocation)?;
        context
            .texture_registry
            .poll(ez_gfx_hal::QueueKind::TextureTransfer, completed)
            .map_err(map_texture)?;
        record_texture_ready(context, texture, completed);
        let (id, _, _, _, total) = context
            .textures
            .get(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        let resident = match context.texture_registry.resident_mips(*id) {
            Ok(resident) => resident,
            Err(TextureError::NotReady) => 0,
            Err(error) => return Err(map_texture(error)),
        };
        Ok((resident, *total))
    })
}

/// Polls one asynchronous texture request through decode and GPU transfer completion.
pub fn poll_texture_load(context: ContextHandle, texture: TextureHandle) -> EzGfxResult {
    match texture_binding(context, texture) {
        Ok(_) => EzGfxResult::Ok,
        Err(error) => error,
    }
}

/// Cancels a texture that has not reached native transfer submission.
pub fn cancel_texture_load(context: ContextHandle, texture: TextureHandle) -> EzGfxResult {
    super::result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let pending = context
            .pending_textures
            .remove(&texture)
            .ok_or(EzGfxResult::InvalidArgument)?;
        pending.cancelled.store(true, Ordering::Release);
        context
            .texture_registry
            .cancel_upload(pending.id)
            .map_err(map_texture)?;
        context
            .identity
            .remove(texture.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let record = runtime_record(
            context,
            texture.into_raw(),
            RuntimePhase::Decode,
            EzGfxResult::Cancelled,
        );
        context.observability.push_event(record);
        Ok(())
    }))
}

/// Unloads a texture or cancels its queued decode.
pub fn unload_texture(context: ContextHandle, texture: TextureHandle) {
    let _ = with_context_mut(context, |context| {
        pump_async_textures(context)?;
        let handle = texture.packed();
        context
            .identity
            .remove(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        if let Some(pending) = context.pending_textures.remove(&texture) {
            pending.cancelled.store(true, Ordering::Release);
            return context
                .texture_registry
                .cancel_upload(pending.id)
                .map_err(map_texture);
        }
        if context.texture_failures.remove(&texture).is_some() {
            return Ok(());
        }
        wait_native_idle(&mut context.native).map_err(map_hal)?;
        let (id, allocation, _, _, _) = context
            .textures
            .remove(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        context.texture_ready.remove(&texture);
        context.texture_registry.unload(id).map_err(map_texture)?;
        destroy_native_texture(&mut context.native, allocation).map_err(map_allocation)
    });
}
