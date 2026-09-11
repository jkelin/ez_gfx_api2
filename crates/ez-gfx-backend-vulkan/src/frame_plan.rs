use super::{CompletionToken, HalError, NativeFrameAction, NativeFrameResource, QueueKind};
use crate::NativeFrameActionSource;
use arrayvec::ArrayVec;
use ez_gfx_hal::COUNTER_BUFFER_ELEMENT_OFFSET;

pub(super) struct FramePlan {
    pub(super) uses_surface: bool,
    pub(super) presents: bool,
    pub(super) external_wait: ArrayVec<CompletionToken, 2>,
}

/// Requires exact sample-count agreement between a pass and its color target.
///
/// Surfaces stay single-sample; a multisampled pass needs a multisampled
/// target rendering at exactly the pass count, and vice versa.
fn samples_agree(resource: &NativeFrameResource<'_>, samples: u8) -> bool {
    match resource {
        NativeFrameResource::Surface => samples == 1,
        NativeFrameResource::RenderTarget(texture) => {
            texture.msaa.as_ref().map_or(1, |msaa| msaa.samples) == samples
        }
        _ => false,
    }
}

/// Rejects misshapen or passless mesh dispatches without allocation.
///
/// Mesh work shades the active pass directly: no index or indirect buffers
/// participate. Grid ceilings and reflected threadgroup sizes need the retained
/// native limits, so recording rechecks them fully.
///
/// # Errors
///
/// Returns [`HalError::InvalidArgument`] for a passless, empty, overflowing,
/// stage-inconsistent, or pipeline-mismatched dispatch.
pub(super) fn validate_mesh_plan(
    draw: &super::super::NativeMeshDraw<'_>,
    extent: (u32, u32),
    pass_active: bool,
) -> Result<(), HalError> {
    if !pass_active
        || draw.width == 0
        || draw.height == 0
        || draw.width > extent.0
        || draw.height > extent.1
        || draw.groups.contains(&0)
        || draw.task_workgroup_size.is_some() != draw.has_task
        || !matches!(
            draw.pipeline.kind,
            super::super::NativePipelineKind::Mesh { task_stage }
                if task_stage == draw.has_task
        )
    {
        return Err(HalError::InvalidArgument);
    }
    Ok(())
}

// Overflow and any edge beyond the attachment both reject the pass.
const fn pass_area_fits(area: [u32; 4], extent: (u32, u32)) -> bool {
    match (area[0].checked_add(area[2]), area[1].checked_add(area[3])) {
        (Some(right), Some(bottom)) => right <= extent.0 && bottom <= extent.1,
        _ => false,
    }
}

pub(super) fn validate_frame_plan(
    actions: &(impl NativeFrameActionSource + ?Sized),
    extent: (u32, u32),
    surface_available: bool,
    capture_presented: bool,
) -> Result<FramePlan, HalError> {
    let mut present_count = 0_usize;
    let mut uses_surface = false;
    let mut external_wait = ArrayVec::<CompletionToken, 2>::new();
    let mut pass_active = false;
    let mut saw_present = false;
    let mut pass_extent = None;
    actions.visit(&mut |_, action| {
        if saw_present {
            return Err(HalError::InvalidArgument);
        }
        match action {
            NativeFrameAction::Wait(token) => {
                if !matches!(
                    token.queue,
                    QueueKind::Transfer | QueueKind::TextureTransfer
                ) {
                    return Err(HalError::InvalidArgument);
                }
                if let Some(existing) = external_wait
                    .iter_mut()
                    .find(|existing| existing.queue == token.queue)
                {
                    if token.value > existing.value {
                        *existing = *token;
                    }
                } else {
                    external_wait
                        .try_push(*token)
                        .map_err(|_| HalError::InvalidArgument)?;
                }
            }
            NativeFrameAction::Barrier { barrier, resource } => {
                uses_surface |= matches!(
                    resource,
                    NativeFrameResource::Surface | NativeFrameResource::Depth
                );
                match (resource, barrier.range) {
                    (
                        NativeFrameResource::Buffer(allocation),
                        ez_gfx_hal::ExecutionRange::Buffer(range),
                    ) if range
                        .offset
                        .checked_add(range.size)
                        .is_some_and(|end| end <= allocation.allocation.size()) => {}
                    (
                        NativeFrameResource::Texture(_)
                        | NativeFrameResource::Surface
                        | NativeFrameResource::Depth
                        | NativeFrameResource::RenderTarget(_),
                        ez_gfx_hal::ExecutionRange::Image(_),
                    ) => {}
                    _ => return Err(HalError::InvalidArgument),
                }
            }
            NativeFrameAction::BeginPass { pass, colors } => {
                uses_surface |= colors
                    .iter()
                    .any(|attachment| matches!(attachment.resource, NativeFrameResource::Surface));
                let mut target_extent = None;
                let mut valid = !pass_active
                    && pass.colors.len() == 1
                    && colors.len() == 1
                    && matches!(pass.samples, 1 | 2 | 4 | 8);
                if let Some(attachment) = colors.first() {
                    valid &= samples_agree(&attachment.resource, pass.samples);
                    target_extent = match attachment.resource {
                        NativeFrameResource::Surface => Some(extent),
                        NativeFrameResource::RenderTarget(texture) => {
                            valid &= pass.depth.is_none();
                            Some((texture.width, texture.height))
                        }
                        _ => None,
                    };
                }
                let Some((target_width, target_height)) = target_extent else {
                    return Err(HalError::InvalidArgument);
                };
                if !valid || !pass_area_fits(pass.area, (target_width, target_height)) {
                    return Err(HalError::InvalidArgument);
                }
                pass_active = true;
                pass_extent = target_extent;
            }
            NativeFrameAction::Compute(dispatch) => {
                if pass_active
                    || dispatch.groups.contains(&0)
                    || dispatch.pipeline.kind != super::super::NativePipelineKind::Compute
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::Graphics(draw) => {
                let indirect_size = u64::from(draw.draw_count)
                    .checked_mul(20)
                    .and_then(|size| size.checked_add(COUNTER_BUFFER_ELEMENT_OFFSET))
                    .ok_or(HalError::InvalidArgument)?;
                if !pass_active
                    || draw.pipeline.kind != super::super::NativePipelineKind::Graphics
                    || draw.width == 0
                    || draw.height == 0
                    || pass_extent.is_none_or(|pass_extent| {
                        draw.width > pass_extent.0 || draw.height > pass_extent.1
                    })
                    || draw.draw_count == 0
                    || draw.indirect_buffer.allocation.size() < indirect_size
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::Mesh(draw) => {
                validate_mesh_plan(draw, pass_extent.unwrap_or_default(), pass_active)?;
            }
            NativeFrameAction::TextureReadback { width, height, .. } => {
                if pass_active || *width == 0 || *height == 0 {
                    return Err(HalError::InvalidArgument);
                }
            }
            NativeFrameAction::EndPass => {
                if !pass_active {
                    return Err(HalError::InvalidArgument);
                }
                pass_active = false;
                pass_extent = None;
            }
            NativeFrameAction::Present => {
                if pass_active {
                    return Err(HalError::InvalidArgument);
                }
                present_count += 1;
                saw_present = true;
                uses_surface = true;
            }
        }
        Ok(())
    })?;
    let presents = present_count == 1;
    if pass_active
        || present_count > 1
        || uses_surface && !surface_available
        || (uses_surface || capture_presented) && !presents
    {
        return Err(HalError::InvalidArgument);
    }
    Ok(FramePlan {
        uses_surface,
        presents,
        external_wait,
    })
}

#[cfg(test)]
mod plan_allocation_tests {
    use super::validate_frame_plan;
    use crate::NativeFrameAction;
    use ez_gfx_hal::{CompletionToken, QueueKind};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static BYTES: AtomicUsize = AtomicUsize::new(0);

    struct Counter;

    // SAFETY: every operation delegates unchanged pointers and layouts to
    // `System`; the atomics only observe sizes and never affect ownership.
    unsafe impl GlobalAlloc for Counter {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the unchanged request is delegated to the system allocator.
            let pointer = unsafe { System.alloc(layout) };
            if ENABLED.load(Ordering::Relaxed) && !pointer.is_null() {
                CALLS.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: the pointer and layout came from the system allocator above.
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Counter = Counter;

    #[test]
    fn spill_cardinality_plan_validation_performs_no_allocations() {
        // Sixty-five actions exceed every former inline action-scratch threshold.
        let actions: [NativeFrameAction<'_>; 65] = std::array::from_fn(|index| {
            NativeFrameAction::Wait(CompletionToken {
                queue: if index % 2 == 0 {
                    QueueKind::Transfer
                } else {
                    QueueKind::TextureTransfer
                },
                value: index as u64 + 1,
            })
        });
        for _ in 0..4 {
            validate_frame_plan(&actions, (1, 1), false, false).unwrap();
        }
        CALLS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        for _ in 0..500 {
            validate_frame_plan(&actions, (1, 1), false, false).unwrap();
        }
        ENABLED.store(false, Ordering::Relaxed);
        assert_eq!(CALLS.load(Ordering::Relaxed), 0);
        assert_eq!(BYTES.load(Ordering::Relaxed), 0);
    }
}
