//! Managed render-target lifecycle: allocation, format probing, and teardown.
//!
//! Slice 1 owns creation, per-target clear storage, and destruction over the
//! native single-mip sampled-image constructors, plus multisampled render
//! storage that resolves into the sampled image. No heap writes or pass
//! attachments happen here; the stored declaration (including its clear value
//! and sample count) feeds the render-pass slice. Render targets lease heap

use super::{
    ContextHandle, ContextState, EzGfxResult, NativeTexture, RenderTargetHandle, ResourceKind,
    destroy_native_texture, map_allocation, map_lifecycle, map_texture, native_texture_compression,
    result_status, with_context_mut,
};
use ez_gfx_runtime::target::{Format, TargetDeclaration, TargetError, TargetUsage};
use ez_gfx_runtime::texture::TextureId;

/// Resolves the clear color applied when a pass clears this target.
///
/// Declarations without color data clear to transparent black.
pub(super) fn render_target_clear_color(record: &RenderTargetRecord) -> [f32; 4] {
    match record.declaration.clear() {
        ez_gfx_runtime::target::ClearValue::Color(values) => values,
        _ => [0.0, 0.0, 0.0, 0.0],
    }
}

/// A live managed render target: one sampled single-mip image plus its contract.
pub(super) struct RenderTargetRecord {
    pub(super) native: NativeTexture,
    pub(super) declaration: TargetDeclaration,
    pub(super) format: Format,
    pub(super) width: u32,
    pub(super) height: u32,
    /// Heap slot leased from the shared texture registry. Its index is the
    /// descriptor binding, so textures and targets draw from one free-list
    /// and can never collide.
    pub(super) id: TextureId,
}

/// Creates a sampled color render target from a declaration and explicit extents.
///
/// Probes the active adapter, admits BC/ASTC compression like texture uploads,
/// and stores the declaration (including its clear value and sample count)
/// for the render-pass slice. Multisampled declarations allocate render
/// storage that resolves into the sampled image at end of pass. Depth,
/// storage, and sampled-only declarations are deferred.
/// # Errors
/// Returns [`EzGfxResult::InvalidArgument`] for empty extents or a malformed
/// declaration, [`EzGfxResult::Unsupported`] for non-color usage or an
/// unresolvable format, [`EzGfxResult::NativeFailure`] for oversized targets,
/// an exhausted binding range, or native allocation failure.
pub fn create_render_target(
    context: ContextHandle,
    declaration: &TargetDeclaration,
    width: u32,
    height: u32,
) -> Result<RenderTargetHandle, EzGfxResult> {
    if declaration.usage() != TargetUsage::Color {
        // Depth, storage, and sampled-only targets need pass-attachment and
        // descriptor-table work owned by later slices.
        return Err(EzGfxResult::Unsupported);
    }
    if width == 0 || height == 0 {
        return Err(EzGfxResult::InvalidArgument);
    }
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let compression = native_texture_compression(&context.native);
        let capabilities = match &context.native {
            super::NativeContext::Vulkan(native) => native
                .probe_target_formats()
                .map_err(|_| EzGfxResult::NativeFailure)?,
            #[cfg(windows)]
            super::NativeContext::Dx12(native) => native
                .probe_target_formats()
                .map_err(|_| EzGfxResult::NativeFailure)?,
            #[cfg(target_vendor = "apple")]
            super::NativeContext::Metal(native) => native
                .probe_target_formats()
                .map_err(|_| EzGfxResult::NativeFailure)?,
        };
        let format = capabilities
            .resolve_with_compression(declaration, compression)
            .map_err(map_target_error)?;
        // The native constructor repeats these checks; fail before leasing a
        // binding so rejected requests leave no allocator residue.
        let bytes_per_texel: u64 = match format {
            Format::Rgba8Unorm | Format::Bgra8Srgb => 4,
            Format::Rgba16Float => 8,
            _ => return Err(EzGfxResult::Unsupported),
        };
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(bytes_per_texel))
            .and_then(|single| single.checked_mul(u64::from(declaration.samples())))
            .ok_or(EzGfxResult::NativeFailure)?;
        if bytes > ez_gfx_runtime::texture::MAX_TEXTURE_BYTES as u64 {
            return Err(EzGfxResult::NativeFailure);
        }
        let id = context
            .texture_registry
            .begin_upload()
            .map_err(map_texture)?;
        let binding = match context.texture_registry.reserved_binding(id) {
            Ok(binding) => binding,
            Err(error) => {
                release_heap_slot(context, id);
                return Err(map_texture(error));
            }
        };
        let native = match (&mut context.native, format) {
            (super::NativeContext::Vulkan(native), format) => native
                .create_render_target(format, width, height, binding, declaration.samples())
                .map(NativeTexture::Vulkan)
                .map_err(map_allocation),
            #[cfg(windows)]
            (super::NativeContext::Dx12(native), format) => native
                .create_render_target(format, width, height, binding, declaration.samples())
                .map(NativeTexture::Dx12)
                .map_err(map_allocation),
            #[cfg(target_vendor = "apple")]
            (super::NativeContext::Metal(native), format) => native
                .create_render_target(format, width, height, binding, declaration.samples())
                .map(NativeTexture::Metal)
                .map_err(map_allocation),
        };
        let native = match native {
            Ok(native) => native,
            Err(error) => {
                release_heap_slot(context, id);
                return Err(error);
            }
        };
        let handle = match context.identity.insert(ResourceKind::RenderTarget) {
            Ok(handle) => handle,
            Err(error) => {
                release_heap_slot(context, id);
                let _ = destroy_native_texture(&mut context.native, native);
                return Err(map_lifecycle(error));
            }
        };
        let Ok(typed) = RenderTargetHandle::from_packed(handle) else {
            release_heap_slot(context, id);
            let _ = destroy_native_texture(&mut context.native, native);
            return Err(EzGfxResult::NativeFailure);
        };
        context.render_targets.insert(
            typed,
            RenderTargetRecord {
                native,
                declaration: declaration.clone(),
                format,
                width,
                height,
                id,
            },
        );
        Ok(typed)
    })
}

/// Destroys a render target, freeing its image and binding immediately.
///
/// Unknown or already-destroyed handles are ignored so teardown paths stay
/// infallible, mirroring texture unload.
pub fn destroy_render_target(context: ContextHandle, target: RenderTargetHandle) {
    let _ = with_context_mut(context, |context| {
        // A destroyed target must never stay bound: clear the override even
        // when the handle is already gone so no stale binding survives.
        if context.frame_render_target == Some(target) {
            context.frame_render_target = None;
        }
        let Some(record) = context.render_targets.remove(&target) else {
            return Ok(());
        };
        context
            .identity
            .remove(target.into(), ResourceKind::RenderTarget)
            .map_err(map_lifecycle)?;
        release_heap_slot(context, record.id);
        destroy_native_texture(&mut context.native, record.native).map_err(map_allocation)?;
        Ok(())
    });
}

/// Reports the resolved storage format of a live render target.
///
/// # Errors
///
/// Returns [`EzGfxResult::InvalidArgument`] for an unknown or destroyed handle.
pub fn render_target_format(
    context: ContextHandle,
    target: RenderTargetHandle,
) -> Result<Format, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        context
            .render_targets
            .get(&target)
            .map(|record| record.format)
            .ok_or(EzGfxResult::InvalidArgument)
    })
}

/// Reports the extents of a live render target.
///
/// # Errors
///
/// Returns [`EzGfxResult::InvalidArgument`] for an unknown or destroyed handle.
pub fn render_target_extent(
    context: ContextHandle,
    target: RenderTargetHandle,
) -> Result<(u32, u32), EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        context
            .render_targets
            .get(&target)
            .map(|record| (record.width, record.height))
            .ok_or(EzGfxResult::InvalidArgument)
    })
}

/// Reports the stored clear value of a live render target.
///
/// The render-pass slice applies this when the target is attached; storage
/// here keeps per-target clear data with the target instead of the pass.
///
/// # Errors
///
/// Returns [`EzGfxResult::InvalidArgument`] for an unknown or destroyed handle.
pub fn render_target_clear(
    context: ContextHandle,
    target: RenderTargetHandle,
) -> Result<ez_gfx_runtime::target::ClearValue, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        context
            .render_targets
            .get(&target)
            .map(|record| record.declaration.clear())
            .ok_or(EzGfxResult::InvalidArgument)
    })
}
/// Probes whether one format admits a sampled color target at the given sample count.
///
/// Builds a single-candidate color declaration and resolves it against the
/// active adapter, so depth, storage, above-ceiling sample, and compression
/// rejections surface here exactly as they would at creation time. Probing
/// leases nothing from the shared heap.
///
/// # Errors
///
/// Returns [`EzGfxResult::InvalidArgument`] for a sample count outside
/// `1 | 2 | 4 | 8`, [`EzGfxResult::Unsupported`] when no candidate satisfies
/// the declaration, and [`EzGfxResult::NativeFailure`] when the device cannot
/// be probed.
pub fn probe_render_target_format(
    context: ContextHandle,
    format: Format,
    samples: u8,
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if !matches!(samples, 1 | 2 | 4 | 8) {
            return Err(EzGfxResult::InvalidArgument);
        }
        let compression = native_texture_compression(&context.native);
        let capabilities = match &context.native {
            super::NativeContext::Vulkan(native) => native
                .probe_target_formats()
                .map_err(|_| EzGfxResult::NativeFailure)?,
            #[cfg(windows)]
            super::NativeContext::Dx12(native) => native
                .probe_target_formats()
                .map_err(|_| EzGfxResult::NativeFailure)?,
            #[cfg(target_vendor = "apple")]
            super::NativeContext::Metal(native) => native
                .probe_target_formats()
                .map_err(|_| EzGfxResult::NativeFailure)?,
        };
        let declaration = TargetDeclaration::new(
            "probe",
            TargetUsage::Color,
            1.0,
            samples,
            vec![format],
            ez_gfx_runtime::target::ClearValue::None,
            true,
        )
        .map_err(map_target_error)?;
        capabilities
            .resolve_with_compression(&declaration, compression)
            .map(|_| ())
            .map_err(map_target_error)
    }))
}

pub(super) fn destroy_all_render_targets(context: &mut ContextState) {
    context.frame_render_target = None;
    let records: Vec<_> = context.render_targets.drain().collect();
    for (handle, record) in records {
        let _ = context
            .identity
            .remove(handle.into(), ResourceKind::RenderTarget);
        release_heap_slot(context, record.id);
        let _ = destroy_native_texture(&mut context.native, record.native);
    }
}

/// Returns one heap slot to the shared texture registry.
///
/// Slots stay in the `Allocated` state for the whole target lifetime, so
/// `cancel_upload` releases them without emitting texture unload events.
fn release_heap_slot(context: &mut ContextState, id: TextureId) {
    let _ = context.texture_registry.cancel_upload(id);
}

fn map_target_error(error: TargetError) -> EzGfxResult {
    match error {
        TargetError::UnsupportedFormat => EzGfxResult::Unsupported,
        TargetError::DuplicateSupport => EzGfxResult::NativeFailure,
        TargetError::InvalidName
        | TargetError::InvalidScale
        | TargetError::InvalidSamples
        | TargetError::NoCandidates
        | TargetError::DuplicateCandidate
        | TargetError::InvalidClear
        | TargetError::ClearTypeMismatch => EzGfxResult::InvalidArgument,
    }
}
