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
const DRAW_INDEXED_ARGUMENT_BYTES: u64 = core::mem::size_of::<
    windows::Win32::Graphics::Direct3D12::D3D12_DRAW_INDEXED_ARGUMENTS,
>() as u64;

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
fn indirect_command_bytes(draw_count: u32) -> u64 {
    u64::from(draw_count) * DRAW_INDEXED_ARGUMENT_BYTES
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
