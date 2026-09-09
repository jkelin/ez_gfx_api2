use crate::Result;

#[cfg(windows)]
use super::dx12_bindings;
#[cfg(target_vendor = "apple")]
use super::metal_bindings;
use super::{
    Access, Backend, BufferRange, ContextHandle, ContextState, CounterBufferHandle,
    DiagnosticLevel, DynamicPipelineState, Error, ExecutableNode, ExecutionAction,
    ExecutionBarrier, ExecutionError, ExecutionPass, Format, FrameExecutionBackend,
    FrameExecutionPlan, FrameNativeResource, GeometryAllocation, HashMap, ImageRange, LoadOp,
    MAX_PIPELINE_CACHE_ENTRIES, NativeAllocation, NativeContext, NativePipeline, NativeShader,
    NativeSurface, NativeTexture, NodeDesc, PackedHandle, PassInfo, PipelineKey, QueueKind,
    RenderTargetHandle, RenderTargetRecord, ResourceAccess, ResourceDesc, ResourceId, ResourceKind,
    ResourceLifetime, ResourceState, RuntimePhase, SURFACE_DEFAULT_CLEAR, ShaderHandle,
    ShaderRecord, ShaderStage, StoreOp, TextureFormat, TextureHandle, TextureId,
    execute_compiled_graph, last_native_frame_completion, map_frame, map_hal, map_lifecycle,
    native_layouts, pipeline_layout_key, result_status, runtime_record, vulkan_bindings,
    wait_native_idle, with_context_mut,
};

#[cfg(test)]
mod transient_tests;
type NativeTextureMap = HashMap<TextureHandle, (TextureId, NativeTexture, u32, u32, u32)>;

/// Begins frame recording.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
#[cfg_attr(
    not(any(feature = "ffi", test)),
    allow(
        dead_code,
        reason = "the C raw seam begins target-less readback frames"
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
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
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
    let (_, _, width, height, _) = context
        .textures
        .get(&texture)
        .ok_or(Error::InvalidContext)?;
    let desc = ResourceDesc::image(
        *width,
        *height,
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
    if let Some(ready) = context.texture_ready.get(&texture).copied() {
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
        // smaller extent than the request. Both are boundary rejections, applied
        // identically before any backend records the copy.
        if context.pending_textures.contains_key(&texture) {
            return Err(Error::NotReady);
        }
        let format = context
            .texture_formats
            .get(&texture)
            .copied()
            .unwrap_or(TextureFormat::Rgba8Unorm);
        if format.is_compressed() {
            return Err(Error::InvalidArgument);
        }
        let (_, _, _, _, total) = context
            .textures
            .get(&texture)
            .ok_or(Error::InvalidContext)?;
        let total = *total;
        let resident = context
            .texture_published_mips
            .get(&texture)
            .copied()
            .unwrap_or(0);
        if resident != total {
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
/// Enqueues full-image RGBA8 readback of a managed render target.
///
/// # Errors
/// Returns an error when the target is invalid, compressed, or not part of a recording frame.
pub fn frame_enqueue_render_target_readback(
    context: ContextHandle,
    target: RenderTargetHandle,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let record = context
            .render_targets
            .get(&target)
            .ok_or(Error::InvalidContext)?;
        if matches!(
            record.format,
            Format::Bc7Unorm | Format::Astc4x4Unorm | Format::Depth32Float
        ) {
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

// Missing surface/index resources and invalid ranges fail before the frame node is recorded.
fn graphics_node(
    context: &mut ContextState,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    indirect: CounterBufferHandle,
    pipeline_layout: ez_gfx_runtime::binding::PipelineLayout,
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
    let pass = PassInfo::new(
        vec![color],
        depth,
        [0, 0, width, height],
        samples,
        load,
        StoreOp::Store,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let color_state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let mut node = NodeDesc::new("graphics", QueueKind::Graphics)
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

fn add_texture_accesses(
    context: &mut ContextState,
    mut node: NodeDesc,
    queue: QueueKind,
    stage: ShaderStage,
) -> Result<NodeDesc> {
    // Unpublished textures cannot be sampled yet; unrelated uploads must not stall the heap.
    let texture_handles: Vec<_> = context
        .texture_published_mips
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
    shader: ShaderHandle,
    counter: CounterBufferHandle,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    state: DynamicPipelineState,
) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        let shader_handle = shader.packed();
        context
            .identity
            .resolve(shader_handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        let counter_handle = counter.packed();
        context
            .identity
            .resolve(counter_handle, ResourceKind::CounterBuffer)
            .map_err(map_lifecycle)?;
        validate_binding_handles(context, bindings)?;
        let record = context.shaders.get(&shader).ok_or(Error::InvalidContext)?;
        let layout = record
            .runtime
            .bindings(ez_gfx_artifact::Stage::Vertex)
            .and_then(|vertex| {
                record
                    .runtime
                    .bindings(ez_gfx_artifact::Stage::Fragment)
                    .and_then(|fragment| vertex.merge(&fragment))
            })
            .map_err(|_| Error::InvalidArgument)?;
        layout
            .validate(bindings)
            .map_err(|_| Error::InvalidArgument)?;
        let draw_capacity = context
            .indirects
            .get(&counter)
            .ok_or(Error::InvalidContext)?
            .capacity();
        let pipeline_layout = *record
            .graphics_layout
            .as_ref()
            .ok_or(Error::InvalidArgument)?;
        let node = graphics_node(context, &layout, bindings, counter, pipeline_layout)?;
        context
            .frame
            .record_node(
                node,
                ExecutableNode::Graphics {
                    shader,
                    counter,
                    draw_capacity,
                    bindings: bindings.to_vec(),
                    layout,
                    pipeline_layout,
                    state,
                },
            )
            .map_err(|error| map_frame(&error))?;
        mark_transient_bindings_interned(context, bindings)?;
        mark_transient_interned(context, counter_handle)?;
        context.frame_has_graphics = true;
        Ok(())
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
        validate_binding_handles(context, bindings)?;
        let record = context.shaders.get(&shader).ok_or(Error::InvalidContext)?;
        if groups.contains(&0) {
            return Err(Error::InvalidArgument);
        }
        let layout = record
            .runtime
            .bindings(ez_gfx_artifact::Stage::Compute)
            .map_err(|_| Error::InvalidArgument)?;
        layout
            .validate(bindings)
            .map_err(|_| Error::InvalidArgument)?;
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
        context
            .frame
            .record_node(
                node,
                ExecutableNode::Compute {
                    shader,
                    groups,
                    bindings: bindings.to_vec(),
                    layout,
                },
            )
            .map_err(|error| map_frame(&error))?;
        mark_transient_bindings_interned(context, bindings)?;
        Ok(())
    }))
}

fn validate_binding_handles(
    context: &ContextState,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
) -> Result<()> {
    for binding in bindings {
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
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
) -> Result<()> {
    for binding in bindings {
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
            for (texture, completion) in &context.texture_ready {
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
            let mut adapter = NativeFrameAdapter { context };
            execute_compiled_graph(&submission.graph, &submission.nodes, &mut adapter)
                .map_err(|error| map_execution(&error))?;
            let completion = last_native_frame_completion(&adapter.context.native)?;
            adapter
                .context
                .frame
                .finish()
                .map_err(|error| map_frame(&error))?;
            super::geometry::finalize_recording_range_drops(
                adapter.context,
                frame_serial,
                Some(completion),
            )?;
            recycle_consumed_transients(adapter.context, completion)?;
            super::buffers::reclaim_available_transients(adapter.context)?;
            let record = runtime_record(adapter.context, 0, RuntimePhase::Submit, Ok(()));
            adapter.context.observability.push_event(record);
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
pub fn frame_abort(context: ContextHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let frame_serial = context.frame_serial;
        context.frame.abort();
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
        Ok(())
    }))
}

fn rollback_transient_internment(context: &mut ContextState) {
    for buffer in context.transient_buffers.values_mut() {
        if buffer.usage == super::TransientUse::Interned(context.frame_serial) {
            buffer.usage = super::TransientUse::Available;
        }
    }
}

fn invalidate_unsafe_transients(context: &mut ContextState) {
    let handles = context
        .transient_buffers
        .iter()
        .filter_map(|(handle, buffer)| {
            (buffer.usage == super::TransientUse::Interned(context.frame_serial)).then_some(*handle)
        })
        .collect::<Vec<_>>();
    for handle in handles {
        if let Ok(kind) = context.identity.resource_kind(handle) {
            let _ = context.identity.remove(handle, kind);
            if kind == ResourceKind::CounterBuffer
                && let Ok(typed) = CounterBufferHandle::from_packed(handle)
            {
                context.indirects.remove(&typed);
            }
        }
        context.transient_buffers.remove(&handle);
        // Native ownership is uncertain after a failed idle drain. Keep the
        // allocation quarantined in `allocations` for terminal context cleanup.
    }
}

fn recycle_consumed_transients(
    context: &mut ContextState,
    completion: ez_gfx_hal::CompletionToken,
) -> Result<()> {
    let handles = context
        .transient_buffers
        .iter()
        .filter_map(|(handle, buffer)| {
            (buffer.usage == super::TransientUse::Interned(context.frame_serial)).then_some(*handle)
        })
        .collect::<Vec<_>>();
    for handle in handles {
        let kind = context
            .identity
            .resource_kind(handle)
            .map_err(map_lifecycle)?;
        context
            .identity
            .remove(handle, kind)
            .map_err(map_lifecycle)?;
        let metadata = context
            .transient_buffers
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        context.allocation_ready.remove(&handle);
        let (_, allocation) = context
            .allocations
            .remove(&handle)
            .ok_or(Error::InvalidContext)?;
        match kind {
            ResourceKind::Buffer => context
                .buffer_pool
                .entry(metadata.element_size)
                .or_insert_with(|| ez_gfx_hal::ReusableStagingPool::new(256))
                .put(metadata.byte_capacity, allocation, Some(completion)),
            ResourceKind::CounterBuffer => {
                let typed =
                    CounterBufferHandle::from_packed(handle).map_err(|_| Error::NativeFailure)?;
                context.indirects.remove(&typed);
                context
                    .counter_pool
                    .put(metadata.byte_capacity, allocation, Some(completion));
            }
            _ => return Err(Error::InvalidContext),
        }
    }
    Ok(())
}

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

fn map_execution(error: &ExecutionError<Error>) -> Error {
    match error {
        ExecutionError::Backend(error) => *error,
        ExecutionError::MissingPayload { .. }
        | ExecutionError::UnexpectedPayloads
        | ExecutionError::InvalidCompiledRange => Error::InvalidArgument,
    }
}

struct NativeFrameAdapter<'a> {
    context: &'a mut ContextState,
}

impl FrameExecutionBackend<ExecutableNode> for NativeFrameAdapter<'_> {
    type Error = Error;

    fn execute(
        &mut self,
        plan: &FrameExecutionPlan,
        payloads: &[ExecutableNode],
    ) -> std::result::Result<(), Self::Error> {
        if matches!(self.context.native, NativeContext::Vulkan(_)) {
            return execute_vulkan_frame_plan(self.context, plan, payloads);
        }
        #[cfg(windows)]
        if matches!(self.context.native, NativeContext::Dx12(_)) {
            return execute_dx12_frame_plan(self.context, plan, payloads);
        }
        #[cfg(target_vendor = "apple")]
        if matches!(self.context.native, NativeContext::Metal(_)) {
            return execute_metal_frame_plan(self.context, plan, payloads);
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
