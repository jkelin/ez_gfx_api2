use super::{
    EzGfxResult, FfiContext, ImageMip, NativeContext, NativeTexture, PackedHandle, ResourceKind,
    RuntimePhase, TextureDecoder, TextureError, TextureId, TextureSource,
    completed_transfer_native, destroy_native_texture, generate_mips, map_allocation, map_hal,
    map_lifecycle, map_texture, runtime_record, wait_native_idle, with_context_mut,
};

pub struct TextureConfig {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub sampler: ez_gfx_hal::TextureSamplerDesc,
}

pub fn load_texture(
    context: u64,
    source: TextureSource,
    bytes: &[u8],
    generate: bool,
    config: &TextureConfig,
) -> Result<PackedHandle, EzGfxResult> {
    let decoded = TextureDecoder::decode(source, bytes)
        .and_then(|texture| {
            if generate {
                generate_mips(texture)
            } else {
                Ok(texture)
            }
        })
        .map_err(map_texture)?;
    if (config.width != 0 && decoded.width != config.width)
        || (config.height != 0 && decoded.height != config.height)
        || (config.mip_count != 0 && decoded.mip_count != config.mip_count)
    {
        return Err(EzGfxResult::InvalidArgument);
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
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let texture = context
            .texture_registry
            .begin_upload()
            .map_err(map_texture)?;
        let binding = match context.texture_registry.reserved_binding(texture) {
            Ok(binding) => binding,
            Err(error) => {
                let _ = context.texture_registry.cancel_upload(texture);
                return Err(map_texture(error));
            }
        };
        let created = match &mut context.native {
            NativeContext::Vulkan(native) => native
                .create_texture_rgba8(&mips, binding, config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Vulkan(texture), tokens)),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native
                .create_texture_rgba8(&mips, binding, config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Dx12(texture), tokens)),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native
                .create_texture_rgba8(&mips, binding, config.sampler)
                .map(|(texture, tokens)| (NativeTexture::Metal(texture), tokens)),
        };
        let (native, completions) = match created {
            Ok(created) => created,
            Err(error) => {
                let _ = context.texture_registry.cancel_upload(texture);
                return Err(map_allocation(error));
            }
        };
        let ready = completions
            .last()
            .copied()
            .ok_or(EzGfxResult::NativeFailure)?;
        let mut completions = completions.into_iter();
        let submitted = completions
            .next()
            .ok_or(EzGfxResult::NativeFailure)
            .and_then(|completion| {
                context
                    .texture_registry
                    .mark_submitted(texture, completion)
                    .map_err(map_texture)
            })
            .and_then(|()| {
                for (index, completion) in completions.enumerate() {
                    let mip = u32::try_from(index)
                        .ok()
                        .and_then(|index| index.checked_add(2))
                        .ok_or(EzGfxResult::NativeFailure)?;
                    context
                        .texture_registry
                        .mark_mips_submitted(texture, mip, completion)
                        .map_err(map_texture)?;
                }
                Ok(())
            });
        if let Err(error) = submitted {
            rollback_texture_upload(context, texture, native)?;
            return Err(error);
        }
        let handle = match context.identity.insert(ResourceKind::Texture) {
            Ok(handle) => handle,
            Err(error) => {
                rollback_texture_upload(context, texture, native)?;
                return Err(map_lifecycle(error));
            }
        };
        context.textures.insert(
            handle.get(),
            (
                texture,
                native,
                decoded.width,
                decoded.height,
                decoded.mip_count,
            ),
        );
        context.texture_ready.insert(handle.get(), ready);
        let record = runtime_record(context, handle.get(), RuntimePhase::Upload, EzGfxResult::Ok);
        context.observability.push_event(record);
        Ok(handle)
    })
}

pub(super) fn rollback_texture_upload(
    context: &mut FfiContext,
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

pub fn texture_binding(context: u64, texture: u64) -> Result<u32, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let completed = completed_transfer_native(&context.native).map_err(map_allocation)?;
        context
            .texture_registry
            .poll(ez_gfx_hal::QueueKind::Transfer, completed)
            .map_err(map_texture)?;
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

pub fn texture_residency(context: u64, texture: u64) -> Result<(u32, u32), EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let completed = completed_transfer_native(&context.native).map_err(map_allocation)?;
        context
            .texture_registry
            .poll(ez_gfx_hal::QueueKind::Transfer, completed)
            .map_err(map_texture)?;
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

pub fn unload_texture(context: u64, texture: u64) {
    if texture == 0 {
        return;
    }
    let _ = with_context_mut(context, |context| {
        wait_native_idle(&mut context.native).map_err(map_hal)?;
        let handle = PackedHandle::from_raw(texture).map_err(|_| EzGfxResult::InvalidContext)?;
        context
            .identity
            .remove(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let (id, allocation, _, _, _) = context
            .textures
            .remove(&texture)
            .ok_or(EzGfxResult::InvalidContext)?;
        context.texture_ready.remove(&texture);
        context.texture_registry.unload(id).map_err(map_texture)?;
        destroy_native_texture(&mut context.native, allocation).map_err(map_allocation)
    });
}
