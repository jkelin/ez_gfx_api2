use crate::RenderTargetLoad;
use crate::Result;

use super::{
    Access, Backend, BufferHandle, BufferRange, ContextHandle, ContextState, CounterBufferHandle,
    DiagnosticLevel, DynamicPipelineState, Error, ExecutableNode, ExecutionAction,
    ExecutionBarrier, ExecutionPass, Format, FrameBindingSource, FrameBufferBindingRecord,
    FrameExecutionBackend, FrameExecutionPlan, FrameNativeResource, GeometryAllocation, HashMap,
    ImageRange, LoadOp, MAX_PIPELINE_CACHE_ENTRIES, MeshPipelineKeyDesc, NativeAllocation,
    NativeContext, NativePipeline, NativeShader, NativeSurface, NativeTexture, NodeDesc,
    PackedHandle, PassInfo, PipelineKey, QueueKind, RenderTargetHandle, RenderTargetRecord,
    ResourceAccess, ResourceDesc, ResourceId, ResourceKind, ResourceLifetime, ResourceState,
    RuntimePhase, SURFACE_DEFAULT_CLEAR, ShaderHandle, ShaderRecord, ShaderStage, StoreOp,
    SubmittedInfo, SurfaceHandle, TextureHandle, last_native_frame_completion, map_frame, map_hal,
    map_lifecycle, native_layouts, native_mesh_dispatch_limits, pipeline_layout_key,
    prepare_frame_binding_scratch, result_status, runtime_record, wait_native_idle,
    with_context_mut,
};

#[cfg(target_vendor = "apple")]
use super::MetalWorkgroupSizes;

mod binding;
#[cfg(test)]
pub(in crate::state) use binding::BindingProjection;
mod transients;
use transients::{invalidate_unsafe_transients, recycle_consumed_transients};

#[cfg(test)]
mod transient_tests;
type NativeTextureMap = HashMap<TextureHandle, NativeTexture>;

include!("lowering.rs");

/// Begins frame recording.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
#[cfg_attr(
    not(any(feature = "ffi", test)),
    allow(
        dead_code,
        reason = "the C raw boundary begins target-less readback frames"
    )
)]
pub fn frame_begin(context: ContextHandle) -> Result<()> {
    result_status(with_context_mut(context, start_recording))
}

pub(super) fn start_recording(context: &mut ContextState) -> Result<()> {
    context
        .identity
        .check_thread_and_health()
        .map_err(map_lifecycle)?;
    let frame_serial = context
        .frame_serial
        .checked_add(1)
        .ok_or(Error::NativeFailure)?;
    context.frame.begin().map_err(|error| map_frame(&error))?;
    if let Err(error) = super::buffers::reclaim_available_transients(context) {
        context.frame.abort();
        return Err(error);
    }
    context.frame_serial = frame_serial;
    context.frame_resources.clear();
    context.frame_native_resources.clear();
    context.frame_vertex_heaps.clear();
    context.frame_arena_buffers.clear();
    context.frame_index = None;
    context.active_surface = None;
    context.frame_surface = None;
    context.frame_render_target = None;
    context.frame_render_target_load = RenderTargetLoad::Clear;
    context.frame_render_target_states.clear();
    context.frame_depth = None;
    context.frame_has_graphics = false;
    context.last_readbacks.clear();
    context.frame_presented = false;
    context.frame_capture_surface = None;
    Ok(())
}

/// Returns the serial of the currently recording raw transaction.
///
/// # Errors
///
/// Returns an error when the context is stale, foreign, unhealthy, or not recording.
#[cfg_attr(
    not(feature = "ffi"),
    allow(dead_code, reason = "only the C frame registry mirrors raw serials")
)]
#[doc(hidden)]
pub fn current_frame_serial(context: ContextHandle) -> Result<u64> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording {
            return Err(Error::NotReady);
        }
        Ok(context.frame_serial)
    })
}

/// Requests capture of this frame's presented surface without changing persistent cache policy.
///
/// # Errors
///
/// Returns an error unless `surface` is the active surface of a recording frame.
pub fn frame_request_presented_readback(
    context: ContextHandle,
    surface: SurfaceHandle,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording
            || context.active_surface != Some(surface)
        {
            return Err(Error::NotReady);
        }
        context.frame_capture_surface = Some(surface);
        Ok(())
    }))
}
/// Records a validated texture-to-texture copy.
///
/// # Errors
///
/// Returns [`Error::InvalidContext`] for unknown handles, [`Error::InvalidArgument`]
/// for incompatible formats or regions, or [`Error::NotReady`] when no frame is recording.
pub fn copy_texture_regions(
    context: ContextHandle,
    source: TextureHandle,
    destination: TextureHandle,
    region: ez_gfx_hal::TextureCopyRegion,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        context
            .identity
            .resolve(source.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        context
            .identity
            .resolve(destination.packed(), ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let src = context
            .texture_pipeline
            .submitted()
            .get(&source)
            .copied()
            .ok_or(Error::InvalidContext)?;
        let dst = context
            .texture_pipeline
            .submitted()
            .get(&destination)
            .copied()
            .ok_or(Error::InvalidContext)?;
        ez_gfx_hal::validate_texture_copy(
            src.format,
            dst.format,
            (src.width, src.height),
            (dst.width, dst.height),
            source == destination,
            region,
        )
        .map_err(|_| Error::InvalidArgument)?;
        if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording {
            return Err(Error::NotReady);
        }
        let source_resource = intern_texture_resource(context, source)?;
        let destination_resource = intern_texture_resource(context, destination)?;
        let range = ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?;
        let read = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferRead,
        )
        .map_err(|_| Error::InvalidArgument)?;
        let write = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferWrite,
        )
        .map_err(|_| Error::InvalidArgument)?;
        let node = NodeDesc::new("copy-texture", QueueKind::Transfer)
            .access(Access::image(source_resource, range, read))
            .access(Access::image(destination_resource, range, write));
        context
            .frame
            .record_node(
                node,
                ExecutableNode::CopyTexture {
                    source,
                    destination,
                    region,
                },
            )
            .map_err(|error| map_frame(&error))?;
        Ok(())
    }))
}

pub(super) fn validate_render_target_copy(
    source_format: Format,
    destination_format: Format,
    source_extent: (u32, u32),
    destination_extent: (u32, u32),
    same_target: bool,
    region: ez_gfx_hal::TextureCopyRegion,
) -> Result<()> {
    let [width, height] = region.extent;
    if source_format != destination_format
        || region.source_mip != 0
        || region.destination_mip != 0
        || width == 0
        || height == 0
    {
        return Err(Error::InvalidArgument);
    }
    let source_end = (
        region.source_origin[0].checked_add(width),
        region.source_origin[1].checked_add(height),
    );
    let destination_end = (
        region.destination_origin[0].checked_add(width),
        region.destination_origin[1].checked_add(height),
    );
    let (Some(source_right), Some(source_bottom)) = source_end else {
        return Err(Error::InvalidArgument);
    };
    let (Some(destination_right), Some(destination_bottom)) = destination_end else {
        return Err(Error::InvalidArgument);
    };
    if source_right > source_extent.0
        || source_bottom > source_extent.1
        || destination_right > destination_extent.0
        || destination_bottom > destination_extent.1
    {
        return Err(Error::InvalidArgument);
    }
    let overlaps = region.source_origin[0] < destination_right
        && region.destination_origin[0] < source_right
        && region.source_origin[1] < destination_bottom
        && region.destination_origin[1] < source_bottom;
    if same_target && overlaps {
        return Err(Error::InvalidArgument);
    }
    Ok(())
}

/// Records a validated copy between persistent managed render targets.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for incompatible targets or regions and
/// [`Error::NotReady`] when no frame is recording.
pub fn copy_render_target_regions(
    context: ContextHandle,
    source: RenderTargetHandle,
    destination: RenderTargetHandle,
    region: ez_gfx_hal::TextureCopyRegion,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .resolve(source.packed(), ResourceKind::RenderTarget)
            .map_err(map_lifecycle)?;
        context
            .identity
            .resolve(destination.packed(), ResourceKind::RenderTarget)
            .map_err(map_lifecycle)?;
        let source_record = context
            .render_targets
            .get(&source)
            .ok_or(Error::InvalidContext)?;
        let destination_record = context
            .render_targets
            .get(&destination)
            .ok_or(Error::InvalidContext)?;
        if source_record.declaration.samples() != 1 || destination_record.declaration.samples() != 1
        {
            return Err(Error::InvalidArgument);
        }
        validate_render_target_copy(
            source_record.format,
            destination_record.format,
            (source_record.width, source_record.height),
            (destination_record.width, destination_record.height),
            source == destination,
            region,
        )?;
        if context.frame.state() != ez_gfx_runtime::frame::FrameState::Recording {
            return Err(Error::NotReady);
        }
        let source_resource = intern_render_target_resource(context, source)?;
        let destination_resource = intern_render_target_resource(context, destination)?;
        let range = ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?;
        let read = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferRead,
        )
        .map_err(|_| Error::InvalidArgument)?;
        let write = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferWrite,
        )
        .map_err(|_| Error::InvalidArgument)?;
        context
            .frame
            .record_node(
                NodeDesc::new("copy-render-target", QueueKind::Transfer)
                    .access(Access::image(source_resource, range, read))
                    .access(Access::image(destination_resource, range, write)),
                ExecutableNode::CopyRenderTarget {
                    source,
                    destination,
                    region,
                },
            )
            .map_err(|error| map_frame(&error))?;
        context.frame_render_target_states.insert(source, read);
        context
            .frame_render_target_states
            .insert(destination, write);
        Ok(())
    }))
}

const fn should_capture_presented(snapshot_cache: bool, frame_request: bool) -> bool {
    snapshot_cache || frame_request
}

fn frame_target_extent(
    context: &ContextState,
    surface: Option<&super::SurfaceRecord>,
) -> (u32, u32) {
    surface
        .and_then(|surface| surface.state.extent())
        .or_else(|| {
            context
                .frame_render_target
                .and_then(|target| context.render_targets.get(&target))
                .map(|record| (record.width, record.height))
        })
        .unwrap_or((0, 0))
}

fn expected_frame_output_count(payloads: &[ExecutableNode], capture: bool) -> usize {
    payloads
        .iter()
        .filter(|payload| {
            matches!(
                payload,
                ExecutableNode::TextureReadback { .. }
                    | ExecutableNode::RenderTargetReadback { .. }
            )
        })
        .count()
        .saturating_add(usize::from(capture))
}

include!("resources.rs");
/// Enqueues texture readback in the current frame.
///
/// Readback captures the full stored image as RGBA8. Block-compressed storage has
/// no RGBA texel grid, so compressed textures are rejected with
/// [`Error::InvalidArgument`]; sample them through a shader instead.
/// A logically demoted texture reports [`Error::NotReady`] until its full
/// chain is resident again, so the captured extent always matches the request.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn frame_enqueue_readback(context: ContextHandle, texture: TextureHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = texture.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        // Compressed blocks cannot fill an RGBA8 capture; demoted views expose a
        if context.texture_pipeline.pending().contains_key(&texture) {
            return Err(Error::NotReady);
        }
        let info = context
            .texture_pipeline
            .submitted()
            .get(&texture)
            .copied()
            .ok_or(Error::InvalidContext)?;
        if info.format.is_compressed() {
            return Err(Error::InvalidArgument);
        }
        let resident = context
            .texture_pipeline
            .published()
            .get(&texture)
            .copied()
            .unwrap_or(0);
        if resident != info.total {
            return Err(Error::NotReady);
        }
        let resource = intern_texture_resource(context, texture)?;
        let range = ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?;
        let state = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferRead,
        )
        .map_err(|_| Error::InvalidArgument)?;
        context
            .frame
            .record_node(
                NodeDesc::new("texture-readback", QueueKind::Transfer)
                    .access(Access::image(resource, range, state)),
                ExecutableNode::TextureReadback { texture },
            )
            .map_err(|error| map_frame(&error))?;
        Ok(())
    }))
}
/// Interns a managed render target as a frame-graph image resource.
///
/// Unlike textures, targets carry no initial state: the first barrier starts
/// from undefined, and the compiler derives attachment-to-sampled transitions
/// from pass accesses.
fn intern_render_target_resource(
    context: &mut ContextState,
    target: RenderTargetHandle,
) -> Result<ResourceId> {
    context
        .identity
        .resolve(target.packed(), ResourceKind::RenderTarget)
        .map_err(map_lifecycle)?;
    if let Some(resource) = context.frame_resources.get(&target.packed()) {
        return Ok(*resource);
    }
    let record = context
        .render_targets
        .get(&target)
        .ok_or(Error::InvalidContext)?;
    let last_state = record.last_state;
    let desc = ResourceDesc::image(
        record.width,
        record.height,
        1,
        1,
        record.format,
        record.declaration.samples(),
        ResourceLifetime::External,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    if let Some(state) = last_state {
        context
            .frame
            .set_resource_initial_state(resource, state)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_resources.insert(target.packed(), resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::RenderTarget(target));
    Ok(resource)
}
/// Enqueues full-image RGBA8 readback of an `Rgba8Unorm` managed render target.
///
/// # Errors
/// Returns an error when the target is invalid, not `Rgba8Unorm`, or not part of a recording frame.
pub fn frame_enqueue_render_target_readback(
    context: ContextHandle,
    target: RenderTargetHandle,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let record = context
            .render_targets
            .get(&target)
            .ok_or(Error::InvalidContext)?;
        // The backend copy contract returns bytes unchanged as RGBA8. Reject
        // channel-swizzled and wider texels before recording any graph work.
        if record.format != Format::Rgba8Unorm {
            return Err(Error::InvalidArgument);
        }
        let resource = intern_render_target_resource(context, target)?;
        let range = ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?;
        let state = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferRead,
        )
        .map_err(|_| Error::InvalidArgument)?;
        context
            .frame
            .record_node(
                NodeDesc::new("render-target-readback", QueueKind::Transfer)
                    .access(Access::image(resource, range, state)),
                ExecutableNode::RenderTargetReadback { target },
            )
            .map_err(|error| map_frame(&error))?;
        context.frame_render_target_states.insert(target, state);
        Ok(())
    }))
}

/// Enqueues a barrier-only sampled read after rendering or copying a managed target.
///
/// # Errors
///
/// Returns an error when the target is stale or has not participated in this frame.
pub fn frame_enqueue_render_target_sample(
    context: ContextHandle,
    target: RenderTargetHandle,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        if context.frame_render_target != Some(target)
            && !context.frame_resources.contains_key(&target.packed())
        {
            return Err(Error::InvalidContext);
        }
        let resource = intern_render_target_resource(context, target)?;
        let sampled = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::AllGraphics,
            ResourceAccess::SampledRead,
        )
        .map_err(|_| Error::InvalidArgument)?;
        context
            .frame
            .record_node(
                NodeDesc::new("render-target-sample", QueueKind::Graphics).access(Access::image(
                    resource,
                    ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?,
                    sampled,
                )),
                ExecutableNode::RenderTargetSample { target },
            )
            .map_err(|error| map_frame(&error))?;
        context.frame_render_target_states.insert(target, sampled);
        Ok(())
    }))
}

include!("execution.rs");

/// Submits the recorded frame.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn frame_submit(context: ContextHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let mut native_started = false;
        let frame_serial = context.frame_serial;
        let result = (|| {
            // Updates admitted after draw recording still precede submission. Refresh their
            // dependencies so a cached resource entry cannot retain an older ready value.
            for (texture, completion) in context.texture_pipeline.ready() {
                if let Some(resource) = context.frame_resources.get(&texture.packed()) {
                    context
                        .frame
                        .set_resource_ready(*resource, *completion)
                        .map_err(|error| map_frame(&error))?;
                }
            }
            // Target-only frames present nothing; the image stays sampled.
            let presenting = context.frame_has_graphics && context.frame_render_target.is_none();
            if presenting {
                let surface = context.active_surface.ok_or(Error::NotReady)?;
                let resource = context.frame_surface.ok_or(Error::NotReady)?;
                let present = ResourceState::new(
                    QueueKind::Graphics,
                    ShaderStage::None,
                    ResourceAccess::Present,
                )
                .map_err(|_| Error::InvalidArgument)?;
                context
                    .frame
                    .record_node(
                        NodeDesc::new("present", QueueKind::Graphics).access(Access::image(
                            resource,
                            ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?,
                            present,
                        )),
                        ExecutableNode::Present { surface },
                    )
                    .map_err(|error| map_frame(&error))?;
            }
            let submission = context.frame.submit().map_err(|error| map_frame(&error))?;
            native_started = true;
            let execution = {
                let mut adapter = NativeFrameAdapter {
                    context,
                    binding_resources: &submission.binding_resources,
                };
                adapter
                    .execute(&submission.plan, &submission.nodes)
                    .and_then(|()| last_native_frame_completion(&adapter.context.native))
            };
            context
                .frame
                .finish(submission)
                .map_err(|error| map_frame(&error))?;
            let completion = execution?;
            super::geometry::finalize_recording_range_drops(
                context,
                frame_serial,
                Some(completion),
            )?;
            recycle_consumed_transients(context, completion)?;
            super::buffers::reclaim_available_transients(context)?;
            for packed in &context.frame_arena_buffers {
                if let Ok(handle) = BufferHandle::from_packed(*packed)
                    && let Some(arena) = context.gpu_arenas.get_mut(&handle)
                {
                    arena.last_use = Some(completion);
                }
            }
            for (target, state) in context.frame_render_target_states.drain() {
                if let Some(record) = context.render_targets.get_mut(&target) {
                    record.last_state = Some(state);
                }
            }
            let record = runtime_record(context, 0, RuntimePhase::Submit, Ok(()));
            context.observability.push_event(record);
            Ok(())
        })();
        if let Err(status) = result {
            context.frame.abort();
            if !native_started || wait_native_idle(&mut context.native).is_ok() {
                rollback_transient_internment(context);
                let completion = last_native_frame_completion(&context.native).ok();
                super::geometry::finalize_recording_range_drops(context, frame_serial, completion)?;
            } else {
                invalidate_unsafe_transients(context);
            }
            let record = runtime_record(context, 0, RuntimePhase::Submit, Err(status));
            context
                .observability
                .push_diagnostic(DiagnosticLevel::Error, record);
        }
        super::shader::release_frame_shaders(context);
        result
    }))
}
/// Aborts the current recording transaction without submitting it.
///
/// Interned transient buffers return to their pre-recording state. No backend
/// work starts, so rollback never needs an idle wait or quarantine path.
///
/// # Errors
///
/// Returns an error for a stale, foreign, or wrong-thread context.
fn abort_recording_state(context: &mut ContextState) -> Result<()> {
    let frame_serial = context.frame_serial;
    context.frame.abort();
    super::shader::release_frame_shaders(context);
    rollback_transient_internment(context);
    context.frame_resources.clear();
    let completion = last_native_frame_completion(&context.native).ok();
    super::geometry::finalize_recording_range_drops(context, frame_serial, completion)?;
    context.frame_native_resources.clear();
    context.frame_vertex_heaps.clear();
    context.frame_arena_buffers.clear();
    context.frame_index = None;
    context.frame_surface = None;
    context.frame_render_target = None;
    context.frame_render_target_load = RenderTargetLoad::Clear;
    context.frame_render_target_states.clear();
    context.frame_depth = None;
    context.frame_has_graphics = false;
    context.frame_presented = false;
    context.frame_capture_surface = None;
    Ok(())
}

/// Aborts the active recording frame and releases every retained frame resource.
///
/// # Errors
///
/// Returns an error when the context is stale, unhealthy, on the wrong thread, or cleanup fails.
pub fn frame_abort(context: ContextHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        abort_recording_state(context)
    }))
}

include!("cleanup.rs");

/// Returns every completed frame readback in recording order.
///
/// # Errors
///
/// Returns an error when the context is invalid or no completed readback is available.
#[cfg_attr(feature = "ffi", doc(hidden))]
#[cfg_attr(
    not(feature = "ffi"),
    allow(
        dead_code,
        reason = "completed readbacks are consumed only by the FFI facade"
    )
)]
pub fn frame_readbacks(context: ContextHandle) -> Result<Vec<Vec<u8>>> {
    with_context_mut(context, |context| {
        if context.last_readbacks.is_empty() {
            return Err(Error::NotReady);
        }
        Ok(context.last_readbacks.clone())
    })
}

struct NativeFrameAdapter<'a> {
    context: &'a mut ContextState,
    binding_resources: &'a [ez_gfx_runtime::binding::ResourceIdentity],
}

impl FrameExecutionBackend<ExecutableNode> for NativeFrameAdapter<'_> {
    type Error = Error;

    fn execute(
        &mut self,
        plan: &FrameExecutionPlan,
        payloads: &[ExecutableNode],
    ) -> std::result::Result<(), Self::Error> {
        if matches!(self.context.native, NativeContext::Vulkan(_)) {
            return execute_vulkan_frame_plan(self.context, plan, payloads, self.binding_resources);
        }
        #[cfg(windows)]
        if matches!(self.context.native, NativeContext::Dx12(_)) {
            return execute_dx12_frame_plan(self.context, plan, payloads, self.binding_resources);
        }
        #[cfg(target_vendor = "apple")]
        if matches!(self.context.native, NativeContext::Metal(_)) {
            return execute_metal_frame_plan(self.context, plan, payloads, self.binding_resources);
        }
        Err(Error::NativeFailure)
    }
}

#[cfg(windows)]
mod dx12;
#[cfg(target_vendor = "apple")]
mod metal;
mod vulkan;

#[cfg(windows)]
use dx12::execute_dx12_frame_plan;
#[cfg(target_vendor = "apple")]
use metal::execute_metal_frame_plan;
use vulkan::execute_vulkan_frame_plan;
