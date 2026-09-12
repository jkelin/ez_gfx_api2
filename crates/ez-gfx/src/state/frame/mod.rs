use crate::Result;

use super::{
    Access, Backend, BufferRange, ContextHandle, ContextState, CounterBufferHandle,
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
    prepare_frame_binding_scratch, result_status, runtime_record, wait_native_idle, with_context_mut,
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
    context.frame_index = None;
    context.active_surface = None;
    context.frame_surface = None;
    context.frame_render_target = None;
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

fn intern_buffer_resource(context: &mut ContextState, handle: PackedHandle) -> Result<ResourceId> {
    if let Some(resource) = context.frame_resources.get(&handle) {
        return Ok(*resource);
    }
    let size = context
        .allocations
        .get(&handle)
        .map(|(size, _)| *size)
        .ok_or(Error::InvalidContext)?;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = context.allocation_ready.get(&handle).copied() {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_resources.insert(handle, resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Buffer(handle));
    Ok(resource)
}

fn intern_surface_resource(context: &mut ContextState) -> Result<ResourceId> {
    if let Some(resource) = context.frame_surface {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(Error::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(Error::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Bgra8Srgb,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let present = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::None,
        ResourceAccess::Present,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, present)
        .map_err(|error| map_frame(&error))?;
    context.frame_surface = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Surface(surface));
    Ok(resource)
}

fn intern_depth_resource(context: &mut ContextState) -> Result<ResourceId> {
    if let Some(resource) = context.frame_depth {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(Error::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(Error::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Depth32Float,
        1,
        ResourceLifetime::Transient,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    context.frame_depth = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Depth);
    Ok(resource)
}

fn intern_index_resource(context: &mut ContextState) -> Result<ResourceId> {
    if let Some(resource) = context.frame_index {
        return Ok(resource);
    }
    let heap = context.index_heap.as_ref().ok_or(Error::NotReady)?;
    let size = heap.size;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = heap.ready {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_index = Some(resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Index);
    Ok(resource)
}

fn intern_vertex_heap_resource(
    context: &mut ContextState,
    name: &str,
) -> Result<(ResourceId, u64)> {
    let heap = context.vertex_heaps.get(name).ok_or(Error::NotReady)?;
    let heap_id = heap.heap_id.ok_or(Error::NativeFailure)?;
    if let Some(resource) = context.frame_vertex_heaps.get(&heap_id) {
        return Ok((*resource, heap.size));
    }
    let size = heap.size;
    let ready = heap.ready;
    let desc = ResourceDesc::buffer(size, 16, ResourceLifetime::External)
        .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, initial)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = ready {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_vertex_heaps.insert(heap_id, resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::VertexHeap(heap_id));
    Ok((resource, size))
}

fn add_binding_accesses(
    context: &mut ContextState,
    mut node: NodeDesc,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: binding::BindingProjection<'_>,
    queue: QueueKind,
    stage: ShaderStage,
    combined_indirect: Option<CounterBufferHandle>,
) -> Result<NodeDesc> {
    for requirement in layout.requirements() {
        if requirement.kind == ez_gfx_runtime::binding::BindingKind::VertexHeap {
            let (resource, size) = intern_vertex_heap_resource(context, &requirement.name)?;
            let access_state = ResourceState::new(
                queue,
                stage,
                if requirement.writable {
                    ResourceAccess::StorageReadWrite
                } else {
                    ResourceAccess::StorageRead
                },
            )
            .map_err(|_| Error::InvalidArgument)?;
            node = node.access(Access::buffer(
                resource,
                BufferRange::new(0, size).map_err(|_| Error::InvalidArgument)?,
                access_state,
            ));
            continue;
        }
        let binding = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(Error::InvalidArgument)?;
        let handle = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Buffer(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => {
                return Err(Error::Unsupported);
            }
        };
        if combined_indirect.is_some_and(|indirect| indirect.packed() == handle) {
            continue;
        }
        let size = context
            .allocations
            .get(&handle)
            .map(|(size, _)| *size)
            .ok_or(Error::InvalidContext)?;
        let resource = intern_buffer_resource(context, handle)?;
        let writable = requirement.writable;
        let access_state = ResourceState::new(
            queue,
            stage,
            if writable {
                ResourceAccess::StorageReadWrite
            } else {
                ResourceAccess::StorageRead
            },
        )
        .map_err(|_| Error::InvalidArgument)?;
        node = node.access(Access::buffer(
            resource,
            BufferRange::new(0, size).map_err(|_| Error::InvalidArgument)?,
            access_state,
        ));
    }
    Ok(node)
}

fn intern_texture_resource(
    context: &mut ContextState,
    texture: TextureHandle,
) -> Result<ResourceId> {
    if let Some(resource) = context.frame_resources.get(&texture.packed()) {
        return Ok(*resource);
    }
    let info = context
        .texture_pipeline
        .submitted()
        .get(&texture)
        .copied()
        .ok_or(Error::InvalidContext)?;
    let desc = ResourceDesc::image(
        info.width,
        info.height,
        1,
        1,
        Format::Rgba8Unorm,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context
        .frame
        .set_resource_initial_state(resource, sampled)
        .map_err(|error| map_frame(&error))?;
    if let Some(ready) = context.texture_pipeline.ready().get(&texture).copied() {
        context
            .frame
            .set_resource_ready(resource, ready)
            .map_err(|error| map_frame(&error))?;
    }
    context.frame_resources.insert(texture.packed(), resource);
    context
        .frame_native_resources
        .insert(resource, FrameNativeResource::Texture(texture));
    Ok(resource)
}
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
        Ok(())
    }))
}

// Attachment readiness is checked before any indexed-only resource is interned.
fn graphics_pass_node(
    context: &mut ContextState,
    pipeline_layout: ez_gfx_runtime::binding::PipelineLayout,
    name: &'static str,
) -> Result<NodeDesc> {
    // A bound render target replaces the surface color attachment; depth
    // pipelines stay surface-only. Draws into multisampled targets stay
    // unsupported until pipelines carry sample counts; clears resolve without
    // any draw.
    let (color, depth, width, height, samples) = if let Some(target) = context.frame_render_target {
        if pipeline_layout.depth_required() {
            return Err(Error::Unsupported);
        }
        let resource = intern_render_target_resource(context, target)?;
        let record = context
            .render_targets
            .get(&target)
            .ok_or(Error::InvalidContext)?;
        if record.declaration.samples() != 1 {
            return Err(Error::Unsupported);
        }
        (
            resource,
            None,
            record.width,
            record.height,
            record.declaration.samples(),
        )
    } else {
        let surface = intern_surface_resource(context)?;
        let depth = if pipeline_layout.depth_required() {
            Some(intern_depth_resource(context)?)
        } else {
            None
        };
        let (width, height) = context
            .active_surface
            .and_then(|surface| context.surfaces.get(&surface))
            .and_then(|surface| surface.state.extent())
            .ok_or(Error::NotReady)?;
        (surface, depth, width, height, 1)
    };
    let load = if context.frame_has_graphics {
        LoadOp::Load
    } else {
        LoadOp::Clear
    };
    let pass = if let Some(depth) = depth {
        PassInfo::new(
            vec![color],
            Some(depth),
            [0, 0, width, height],
            samples,
            load,
            StoreOp::Store,
        )
    } else {
        PassInfo::single_color(color, [0, 0, width, height], samples, load, StoreOp::Store)
    }
    .map_err(|_| Error::InvalidArgument)?;
    let color_state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let mut node = NodeDesc::new(name, QueueKind::Graphics)
        .access(Access::image(
            color,
            ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?,
            color_state,
        ))
        .pass(pass);
    if let Some(depth) = depth {
        let depth_state = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::Fragment,
            ResourceAccess::DepthStencilWrite,
        )
        .map_err(|_| Error::InvalidArgument)?;
        node = node.access(Access::image(
            depth,
            ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?,
            depth_state,
        ));
    }
    Ok(node)
}

fn graphics_node(
    context: &mut ContextState,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: binding::BindingProjection<'_>,
    indirect: CounterBufferHandle,
    pipeline_layout: ez_gfx_runtime::binding::PipelineLayout,
) -> Result<NodeDesc> {
    let mut node = graphics_pass_node(context, pipeline_layout, "graphics")?;
    let index_resource = intern_index_resource(context)?;
    let index_size = context.index_heap.as_ref().ok_or(Error::NotReady)?.size;
    let index_state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::AllGraphics,
        ResourceAccess::IndexRead,
    )
    .map_err(|_| Error::InvalidArgument)?;
    node = node.access(Access::buffer(
        index_resource,
        BufferRange::new(0, index_size).map_err(|_| Error::InvalidArgument)?,
        index_state,
    ));
    let indirect_binding = layout.requirements().iter().find_map(|requirement| {
        let binding = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)?;
        matches!(
            binding.resource,
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle) if handle == indirect
        )
        .then_some(requirement.writable)
    });
    let indirect_size = context
        .allocations
        .get(&indirect.packed())
        .map(|(size, _)| *size)
        .ok_or(Error::InvalidContext)?;
    let indirect_resource = intern_buffer_resource(context, indirect.packed())?;
    let indirect_state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::AllGraphics,
        match indirect_binding {
            Some(true) => ResourceAccess::IndirectStorageReadWrite,
            Some(false) => ResourceAccess::IndirectStorageRead,
            None => ResourceAccess::IndirectRead,
        },
    )
    .map_err(|_| Error::InvalidArgument)?;
    node = node.access(Access::buffer(
        indirect_resource,
        BufferRange::new(0, indirect_size).map_err(|_| Error::InvalidArgument)?,
        indirect_state,
    ));
    node = add_binding_accesses(
        context,
        node,
        layout,
        bindings,
        QueueKind::Graphics,
        ShaderStage::AllGraphics,
        Some(indirect),
    )?;
    add_texture_accesses(context, node, QueueKind::Graphics, ShaderStage::AllGraphics)
}

fn mesh_node(
    context: &mut ContextState,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: binding::BindingProjection<'_>,
    pipeline_layout: ez_gfx_runtime::binding::PipelineLayout,
) -> Result<NodeDesc> {
    let node = graphics_pass_node(context, pipeline_layout, "mesh")?;
    let node = add_binding_accesses(
        context,
        node,
        layout,
        bindings,
        QueueKind::Graphics,
        ShaderStage::AllGraphics,
        None,
    )?;
    add_texture_accesses(context, node, QueueKind::Graphics, ShaderStage::AllGraphics)
}

fn add_texture_accesses(
    context: &mut ContextState,
    mut node: NodeDesc,
    queue: QueueKind,
    stage: ShaderStage,
) -> Result<NodeDesc> {
    // Unpublished textures cannot be sampled yet; unrelated uploads must not stall the heap.
    let texture_handles: Vec<_> = context
        .texture_pipeline
        .published()
        .iter()
        .filter_map(|(texture, mips)| (*mips != 0).then_some(*texture))
        .collect();
    for texture in texture_handles {
        let resource = intern_texture_resource(context, texture)?;
        let sampled = ResourceState::new(queue, stage, ResourceAccess::SampledRead)
            .map_err(|_| Error::InvalidArgument)?;
        node = node.access(Access::image(
            resource,
            ImageRange::all(1, 1).map_err(|_| Error::InvalidArgument)?,
            sampled,
        ));
    }
    Ok(node)
}

/// Records an indexed graphics operation.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn execute_graphics(
    context: ContextHandle,
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
    counter: CounterBufferHandle,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    state: DynamicPipelineState,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        for shader in [vertex_shader, fragment_shader] {
            context
                .identity
                .resolve(shader.packed(), ResourceKind::Shader)
                .map_err(map_lifecycle)?;
        }
        let counter_handle = counter.packed();
        context
            .identity
            .resolve(counter_handle, ResourceKind::CounterBuffer)
            .map_err(map_lifecycle)?;
        let vertex = context
            .shaders
            .get(&vertex_shader)
            .filter(|record| record.stage == ez_gfx_artifact::Stage::Vertex)
            .ok_or(Error::InvalidContext)?;
        let fragment = context
            .shaders
            .get(&fragment_shader)
            .filter(|record| record.stage == ez_gfx_artifact::Stage::Fragment)
            .ok_or(Error::InvalidContext)?;
        let layout = vertex
            .runtime
            .bindings(ez_gfx_artifact::Stage::Vertex)
            .and_then(|vertex_layout| {
                fragment
                    .runtime
                    .bindings(ez_gfx_artifact::Stage::Fragment)
                    .and_then(|fragment_layout| vertex_layout.merge(&fragment_layout))
            })
            .map_err(|_| Error::InvalidArgument)?;
        let bindings = binding::BindingProjection::new(&layout, bindings);
        validate_binding_handles(context, bindings)?;
        bindings.validate().map_err(|_| Error::InvalidArgument)?;
        let draw_capacity = context
            .indirects
            .get(&counter)
            .ok_or(Error::InvalidContext)?
            .capacity();
        let pipeline_layout = vertex
            .runtime
            .pipeline_layout(ez_gfx_artifact::Stage::Vertex)
            .and_then(|vertex_layout| {
                fragment
                    .runtime
                    .pipeline_layout(ez_gfx_artifact::Stage::Fragment)
                    .and_then(|fragment_layout| vertex_layout.merge(&fragment_layout))
            })
            .map_err(|_| Error::InvalidArgument)?;
        // Required-prefix textures cannot sample fallback: drive pending decodes
        // to referenced descriptors here so heap accesses below attach GPU waits.
        // Heapless shaders sample no textures and skip driving entirely.
        super::texture_manager::gate_required_textures_for_submit(
            context,
            pipeline_layout.texture_heap().is_some(),
        )?;
        let node = graphics_node(context, &layout, bindings, counter, pipeline_layout)?;
        let payload_layout = layout.clone();
        context
            .frame
            .record_bound_node(node, bindings.resources(), move |bindings| {
                ExecutableNode::Graphics {
                    vertex_shader,
                    fragment_shader,
                    counter,
                    draw_capacity,
                    bindings,
                    layout: payload_layout,
                    pipeline_layout,
                    state,
                }
            })
            .map_err(|error| map_frame(&error))?;
        context.frame_shaders.insert(vertex_shader);
        context.frame_shaders.insert(fragment_shader);
        mark_transient_bindings_interned(context, bindings)?;
        mark_transient_interned(context, counter_handle)?;
        context.frame_has_graphics = true;
        Ok(())
    }))
}
/// Preserves malformed reflection as an argument error while separating valid
/// shader shapes that the selected device cannot execute.
pub(super) const fn map_mesh_dispatch(error: ez_gfx_hal::MeshDispatchError) -> Error {
    match error {
        ez_gfx_hal::MeshDispatchError::InvalidGroups
        | ez_gfx_hal::MeshDispatchError::InvalidWorkgroup => Error::InvalidArgument,
        ez_gfx_hal::MeshDispatchError::UnsupportedWorkgroup => Error::Unsupported,
    }
}

/// Records a direct mesh graphics operation.
///
/// # Errors
///
/// Returns an error before graph mutation when capabilities, handles, reflection, groups, or
/// bindings are invalid.
pub fn execute_mesh(
    context: ContextHandle,
    stages: ez_gfx_hal::MeshStages<ShaderHandle>,
    groups: [u32; 3],
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    state: ez_gfx_hal::MeshPipelineState,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let result = (|| {
            context
                .identity
                .resolve(stages.mesh.packed(), ResourceKind::Shader)
                .map_err(map_lifecycle)?;
            context
                .identity
                .resolve(stages.fragment.packed(), ResourceKind::Shader)
                .map_err(map_lifecycle)?;
            if let Some(task) = stages.task {
                context
                    .identity
                    .resolve(task.packed(), ResourceKind::Shader)
                    .map_err(map_lifecycle)?;
            }
            let capabilities = super::context_shader_capabilities(context)?;
            if !capabilities.mesh || stages.task.is_some() && !capabilities.task {
                return Err(Error::Unsupported);
            }

            let mesh = context
                .shaders
                .get(&stages.mesh)
                .filter(|record| record.stage == ez_gfx_artifact::Stage::Mesh)
                .ok_or(Error::InvalidContext)?;
            let fragment = context
                .shaders
                .get(&stages.fragment)
                .filter(|record| record.stage == ez_gfx_artifact::Stage::Fragment)
                .ok_or(Error::InvalidContext)?;
            let task = stages
                .task
                .map(|handle| {
                    context
                        .shaders
                        .get(&handle)
                        .filter(|record| record.stage == ez_gfx_artifact::Stage::Task)
                        .ok_or(Error::InvalidContext)
                })
                .transpose()?;

            let mut layout = mesh
                .runtime
                .bindings(ez_gfx_artifact::Stage::Mesh)
                .map_err(|_| Error::InvalidArgument)?;
            let mut pipeline_layout = mesh
                .runtime
                .pipeline_layout(ez_gfx_artifact::Stage::Mesh)
                .map_err(|_| Error::InvalidArgument)?;
            if let Some(task) = task {
                layout = task
                    .runtime
                    .bindings(ez_gfx_artifact::Stage::Task)
                    .and_then(|task_layout| task_layout.merge(&layout))
                    .map_err(|_| Error::InvalidArgument)?;
                pipeline_layout = task
                    .runtime
                    .pipeline_layout(ez_gfx_artifact::Stage::Task)
                    .and_then(|task_layout| task_layout.merge(&pipeline_layout))
                    .map_err(|_| Error::InvalidArgument)?;
            }
            layout = layout
                .merge(
                    &fragment
                        .runtime
                        .bindings(ez_gfx_artifact::Stage::Fragment)
                        .map_err(|_| Error::InvalidArgument)?,
                )
                .map_err(|_| Error::InvalidArgument)?;
            pipeline_layout = pipeline_layout
                .merge(
                    &fragment
                        .runtime
                        .pipeline_layout(ez_gfx_artifact::Stage::Fragment)
                        .map_err(|_| Error::InvalidArgument)?,
                )
                .map_err(|_| Error::InvalidArgument)?;
            let mesh_threads = mesh
                .runtime
                .workgroup_size()
                .map_err(|_| Error::InvalidArgument)?;
            let task_threads = task
                .map(|record| record.runtime.workgroup_size())
                .transpose()
                .map_err(|_| Error::InvalidArgument)?;
            let limits = native_mesh_dispatch_limits(&context.native, stages.task.is_some())
                .map_err(map_hal)?;
            ez_gfx_hal::validate_mesh_dispatch(groups, mesh_threads, task_threads, limits)
                .map_err(map_mesh_dispatch)?;

            let stage_layouts = ez_gfx_hal::MeshStages {
                task: task.map(|record| record.runtime.physical_layout_identity()),
                mesh: mesh.runtime.physical_layout_identity(),
                fragment: fragment.runtime.physical_layout_identity(),
            };
            let projected = binding::BindingProjection::new(&layout, bindings);
            validate_binding_handles(context, projected)?;
            projected.validate().map_err(|_| Error::InvalidArgument)?;

            let node = mesh_node(context, &layout, projected, pipeline_layout)?;
            let payload_layout = layout.clone();
            context
                .frame
                .record_bound_node(node, projected.resources(), move |bindings| {
                    ExecutableNode::Mesh {
                        stages,
                        groups,
                        bindings,
                        layout: payload_layout,
                        stage_layouts,
                        pipeline_layout,
                        state,
                    }
                })
                .map_err(|error| map_frame(&error))?;
            if let Some(task) = stages.task {
                context.frame_shaders.insert(task);
            }
            context.frame_shaders.insert(stages.mesh);
            context.frame_shaders.insert(stages.fragment);
            mark_transient_bindings_interned(context, projected)?;
            context.frame_has_graphics = true;
            Ok(())
        })();
        if result.is_err() {
            let _ = abort_recording_state(context);
        }
        result
    }))
}

/// Records a compute dispatch.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn execute_compute(
    context: ContextHandle,
    shader: ShaderHandle,
    groups: [u32; 3],
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let handle = shader.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        let record = context
            .shaders
            .get(&shader)
            .filter(|record| record.stage == ez_gfx_artifact::Stage::Compute)
            .ok_or(Error::InvalidContext)?;
        if groups.contains(&0) {
            return Err(Error::InvalidArgument);
        }
        let layout = record
            .runtime
            .bindings(ez_gfx_artifact::Stage::Compute)
            .map_err(|_| Error::InvalidArgument)?;
        let bindings = binding::BindingProjection::new(&layout, bindings);
        validate_binding_handles(context, bindings)?;
        bindings.validate().map_err(|_| Error::InvalidArgument)?;
        let heap_demanded = record
            .runtime
            .pipeline_layout(ez_gfx_artifact::Stage::Compute)
            .map(|layout| layout.texture_heap().is_some())
            .map_err(|_| Error::InvalidArgument)?;
        // Heapless shaders sample no textures and skip driving entirely.
        super::texture_manager::gate_required_textures_for_submit(context, heap_demanded)?;
        let node = add_binding_accesses(
            context,
            NodeDesc::new("compute", QueueKind::Compute),
            &layout,
            bindings,
            QueueKind::Compute,
            ShaderStage::Compute,
            None,
        )?;
        let node = add_texture_accesses(context, node, QueueKind::Compute, ShaderStage::Compute)?;
        let payload_layout = layout.clone();
        context
            .frame
            .record_bound_node(node, bindings.resources(), move |bindings| {
                ExecutableNode::Compute {
                    shader,
                    groups,
                    bindings,
                    layout: payload_layout,
                }
            })
            .map_err(|error| map_frame(&error))?;
        context.frame_shaders.insert(shader);
        mark_transient_bindings_interned(context, bindings)?;
        Ok(())
    }))
}

fn validate_binding_handles(
    context: &ContextState,
    bindings: binding::BindingProjection<'_>,
) -> Result<()> {
    for binding in bindings.iter() {
        let (packed, kind) = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Buffer(handle) => {
                (handle.packed(), ResourceKind::Buffer)
            }
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle) => {
                (handle.packed(), ResourceKind::CounterBuffer)
            }
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(handle) => {
                context
                    .identity
                    .resolve(handle.packed(), ResourceKind::RenderTarget)
                    .map_err(map_lifecycle)?;
                continue;
            }
        };
        context
            .identity
            .resolve(packed, kind)
            .map_err(map_lifecycle)?;
        let usage = context
            .transient_buffers
            .get(&packed)
            .ok_or(Error::InvalidContext)?
            .usage;
        match usage {
            super::TransientUse::Available => {}
            super::TransientUse::Interned(frame) if frame == context.frame_serial => {}
            super::TransientUse::Interned(_) => return Err(Error::NotReady),
        }
    }
    Ok(())
}

fn mark_transient_bindings_interned(
    context: &mut ContextState,
    bindings: binding::BindingProjection<'_>,
) -> Result<()> {
    for binding in bindings.iter() {
        let handle = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Buffer(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::Counter(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => continue,
        };
        mark_transient_interned(context, handle)?;
    }
    Ok(())
}

fn mark_transient_interned(context: &mut ContextState, handle: PackedHandle) -> Result<()> {
    let buffer = context
        .transient_buffers
        .get_mut(&handle)
        .ok_or(Error::InvalidContext)?;
    match buffer.usage {
        super::TransientUse::Available => {
            buffer.usage = super::TransientUse::Interned(context.frame_serial);
            Ok(())
        }
        super::TransientUse::Interned(frame) if frame == context.frame_serial => Ok(()),
        super::TransientUse::Interned(_) => Err(Error::NotReady),
    }
}

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
                    .and_then(|()| {
                        // The raw Windows test executor submits no command list, so
                        // publish its monotonic observation as the frame completion.
                        #[cfg(all(test, windows))]
                        if adapter.context.raw_native_frame_test_probe.enabled {
                            return ez_gfx_hal::CompletionToken::new(
                                QueueKind::Graphics,
                                adapter.context.raw_native_frame_test_probe.submits as u64,
                            )
                            .map_err(|_| Error::NativeFailure);
                        }
                        last_native_frame_completion(&adapter.context.native)
                    })
            };
            // Even failed native encoding no longer strands the reusable CPU buffers.
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
    context.frame_index = None;
    context.frame_surface = None;
    context.frame_render_target = None;
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
