fn validate_surface_request(
    surface: Option<FrameSurface<'_>>,
) -> Result<ResolvedFrameSurface<'_>, HalError> {
    match surface {
        Some((surface, extent, mode)) if extent.0 != 0 && extent.1 != 0 => {
            if !surface.presentation_modes().contains(mode) {
                return Err(HalError::Unsupported);
            }
            Ok((Some(surface), extent, mode))
        }
        Some(_) => Err(HalError::InvalidArgument),
        None => Ok((None, (0, 0), PresentationMode::Fifo)),
    }
}
#[cfg(test)]
const SOURCE_DRAW_INDEXED_ARGUMENT_BYTES: u64 = 20;
// Native record: leading base-instance root-constant dword, four draw dwords,
// and a zeroed start-instance dword. The command signature strides 24 bytes.
const NATIVE_DRAW_INDEXED_ARGUMENT_BYTES: u64 = 24;
struct DxFramePlan {
    uses_surface: bool,
    presents: bool,
    external_waits: ArrayVec<CompletionToken, 2>,
}

// D3D12 copy commands are invalid between BeginRenderPass and EndRenderPass.
fn validate_indirect_copy_phase(pass_active: bool) -> Result<(), HalError> {
    if pass_active {
        Err(HalError::InvalidArgument)
    } else {
        Ok(())
    }
}
#[cfg(test)]
fn source_indirect_command_bytes(draw_count: u32) -> u64 {
    u64::from(draw_count) * SOURCE_DRAW_INDEXED_ARGUMENT_BYTES
}

fn native_indirect_command_bytes(draw_count: u32) -> u64 {
    u64::from(draw_count) * NATIVE_DRAW_INDEXED_ARGUMENT_BYTES
}

#[allow(
    clippy::too_many_lines,
    reason = "one validation pass keeps cross-action frame invariants local"
)]
fn bindings_match_pipeline(
    bindings: &dyn super::NativeBufferBindingSource,
    writable: &[bool],
) -> Result<bool, HalError> {
    if bindings.len() != writable.len() {
        return Ok(false);
    }
    let mut valid = true;
    bindings.visit(&mut |index, binding| {
        valid &= writable.get(index).is_some_and(|expected| {
            binding.writable == *expected && binding.offset < binding.allocation.allocation.size()
        });
        Ok(())
    })?;
    Ok(valid)
}

fn indirect_binding_writable(
    bindings: &dyn super::NativeBufferBindingSource,
    indirect: &super::ID3D12Resource,
) -> Result<Option<bool>, HalError> {
    let indirect = windows::core::Interface::as_raw(indirect);
    let mut found = None;
    bindings.visit(&mut |_, binding| {
        if windows::core::Interface::as_raw(&binding.allocation.resource) == indirect {
            found = Some(binding.writable);
        }
        Ok(())
    })?;
    Ok(found)
}

/// Rejects misshapen mesh dispatches without allocation.
///
/// Mesh work shades the active pass directly: no index buffer, no indirect
/// signature, and no indirect copies participate. The pipeline must be a mesh
/// state object; grid shape and reflected thread sizes run through the shared
/// dispatch validation.
///
/// # Errors
///
/// Returns [`HalError::InvalidArgument`] for a passless dispatch, a non-mesh or
/// task-mismatched pipeline, mismatched bindings, or a grid the shared validation rejects.
fn validate_mesh_plan(
    dispatch: &super::NativeMeshDispatch<'_>,
    pass_active: bool,
) -> Result<(), HalError> {
    if !pass_active
        || !dispatch.pipeline.mesh
        || dispatch.has_task != dispatch.pipeline.task_stage
        || !bindings_match_pipeline(dispatch.bindings, &dispatch.pipeline.buffer_writable)?
        || super::check_mesh_dispatch(
            dispatch.has_task,
            dispatch.groups,
            dispatch.mesh_workgroup_size,
            dispatch.task_workgroup_size,
        )
        .is_err()
    {
        return Err(HalError::InvalidArgument);
    }
    Ok(())
}

fn track_external_wait(
    external_waits: &mut ArrayVec<CompletionToken, 2>,
    token: CompletionToken,
) -> Result<(), HalError> {
    if !matches!(
        token.queue,
        QueueKind::Transfer | QueueKind::TextureTransfer
    ) {
        return Err(HalError::InvalidArgument);
    }
    if let Some(existing) = external_waits
        .iter_mut()
        .find(|existing| existing.queue == token.queue)
    {
        if token.value > existing.value {
            *existing = token;
        }
    } else {
        external_waits
            .try_push(token)
            .map_err(|_| HalError::InvalidArgument)?;
    }
    Ok(())
}

fn validate_begin_pass(
    pass: &super::ExecutionPass,
    colors: &[super::PassAttachment<'_>],
    extent: (u32, u32),
    uses_surface: &mut bool,
    pass_active: &mut bool,
) -> Result<(), HalError> {
    *uses_surface |= colors
        .iter()
        .any(|attachment| matches!(attachment.resource, NativeFrameResource::Surface));
    let mut target_extent = None;
    let mut valid = !*pass_active
        && pass.colors.len() == 1
        && colors.len() == 1
        && matches!(pass.samples, 1 | 2 | 4 | 8);
    if let Some(attachment) = colors.first() {
        target_extent = match attachment.resource {
            NativeFrameResource::Surface => {
                valid &= pass.samples == 1;
                Some(extent)
            }
            NativeFrameResource::RenderTarget(texture) => {
                valid &= pass.depth.is_none() || texture.depth.is_some();
                valid &= texture.msaa.as_ref().map_or(1, |msaa| msaa.samples) == pass.samples;
                Some((texture.width, texture.height))
            }
            _ => None,
        };
    }
    let Some((target_width, target_height)) = target_extent else {
        return Err(HalError::InvalidArgument);
    };
    if !valid
        || pass.area[0]
            .checked_add(pass.area[2])
            .is_none_or(|end| end > target_width)
        || pass.area[1]
            .checked_add(pass.area[3])
            .is_none_or(|end| end > target_height)
    {
        return Err(HalError::InvalidArgument);
    }
    *pass_active = true;
    Ok(())
}

fn validate_graphics_draw(
    draw: &super::NativeDrawIndexed<'_>,
    pass_active: bool,
) -> Result<(), HalError> {
    let indirect_size = native_indirect_command_bytes(draw.draw_count)
        .checked_add(COUNTER_BUFFER_ELEMENT_OFFSET)
        .ok_or(HalError::InvalidArgument)?;
    if !pass_active
        || draw.draw_count == 0
        || !bindings_match_pipeline(draw.bindings, &draw.pipeline.buffer_writable)?
        || draw.pipeline.topology.is_none()
        || draw.pipeline.signature.is_none()
        || draw.pipeline.base_instance_root == u32::MAX
        || draw.index_size == 0
        || draw.index_size > u64::from(u32::MAX)
        || draw.indirect_size < indirect_size
    {
        return Err(HalError::InvalidArgument);
    }
    Ok(())
}
