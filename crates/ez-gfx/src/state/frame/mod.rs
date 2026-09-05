#[cfg(windows)]
use super::dx12_bindings;
#[cfg(target_vendor = "apple")]
use super::metal_bindings;
use super::{
    Access, Backend, BufferRange, ContextHandle, ContextState, DiagnosticLevel,
    DynamicPipelineState, ExecutableNode, ExecutionAction, ExecutionError, EzGfxResult, Format,
    FrameExecutionBackend, FrameExecutionPlan, FrameNativeResource, HashMap, ImageRange,
    IndirectBufferHandle, LoadOp, MAX_PIPELINE_CACHE_ENTRIES, NativeAllocation, NativeContext,
    NativePipeline, NativeShader, NativeSurface, NativeTexture, NodeDesc, PackedHandle, PassInfo,
    PipelineKey, QueueKind, ResourceAccess, ResourceDesc, ResourceId, ResourceKind,
    ResourceLifetime, ResourceState, RuntimePhase, ShaderHandle, ShaderRecord, ShaderStage,
    StoreOp, TextureHandle, TextureId, execute_compiled_graph, map_frame, map_hal, map_lifecycle,
    native_layouts, pipeline_layout_key, result_status, runtime_record, vulkan_bindings,
    with_context_mut,
};
type NativeTextureMap = HashMap<TextureHandle, (TextureId, NativeTexture, u32, u32, u32)>;

/// Begins frame recording.
pub fn frame_begin(context: ContextHandle) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        context.frame.begin().map_err(|error| map_frame(&error))?;
        context.frame_resources.clear();
        context.frame_native_resources.clear();
        context.frame_index = None;
        context.frame_surface = None;
        context.frame_depth = None;
        context.frame_has_graphics = false;
        context.last_readback.clear();
        context.frame_presented = false;
        Ok(())
    }))
}

fn intern_buffer_resource(
    context: &mut ContextState,
    handle: PackedHandle,
) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_resources.get(&handle) {
        return Ok(*resource);
    }
    let size = context
        .allocations
        .get(&handle)
        .map(|(size, _)| *size)
        .ok_or(EzGfxResult::InvalidContext)?;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
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

fn intern_surface_resource(context: &mut ContextState) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_surface {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(EzGfxResult::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(EzGfxResult::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Bgra8Srgb,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let present = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::None,
        ResourceAccess::Present,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
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

fn intern_depth_resource(context: &mut ContextState) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_depth {
        return Ok(resource);
    }
    let surface = context.active_surface.ok_or(EzGfxResult::NotReady)?;
    let (width, height) = context
        .surfaces
        .get(&surface)
        .and_then(|surface| surface.state.extent())
        .ok_or(EzGfxResult::NotReady)?;
    let desc = ResourceDesc::image(
        width,
        height,
        1,
        1,
        Format::Depth32Float,
        1,
        ResourceLifetime::Transient,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
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

fn intern_index_resource(context: &mut ContextState) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_index {
        return Ok(resource);
    }
    let heap = context.index_heap.as_ref().ok_or(EzGfxResult::NotReady)?;
    let size = heap.size;
    let desc = ResourceDesc::buffer(size, 4, ResourceLifetime::External)
        .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let initial = ResourceState::new(
        QueueKind::Transfer,
        ShaderStage::None,
        ResourceAccess::TransferWrite,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
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

fn add_binding_accesses(
    context: &mut ContextState,
    mut node: NodeDesc,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    queue: QueueKind,
    stage: ShaderStage,
    combined_indirect: Option<IndirectBufferHandle>,
) -> Result<NodeDesc, EzGfxResult> {
    for requirement in layout.requirements() {
        let binding = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)
            .ok_or(EzGfxResult::InvalidArgument)?;
        let handle = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) => handle.packed(),
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(_) => {
                return Err(EzGfxResult::Unsupported);
            }
        };
        if combined_indirect.is_some_and(|indirect| indirect.packed() == handle) {
            continue;
        }
        let size = context
            .allocations
            .get(&handle)
            .map(|(size, _)| *size)
            .ok_or(EzGfxResult::InvalidContext)?;
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
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        node = node.access(Access::buffer(
            resource,
            BufferRange::new(0, size).map_err(|_| EzGfxResult::InvalidArgument)?,
            access_state,
        ));
    }
    Ok(node)
}

fn intern_texture_resource(
    context: &mut ContextState,
    texture: TextureHandle,
) -> Result<ResourceId, EzGfxResult> {
    if let Some(resource) = context.frame_resources.get(&texture.packed()) {
        return Ok(*resource);
    }
    let (_, _, width, height, _) = context
        .textures
        .get(&texture)
        .ok_or(EzGfxResult::InvalidContext)?;
    let desc = ResourceDesc::image(
        *width,
        *height,
        1,
        1,
        Format::Rgba8Unorm,
        1,
        ResourceLifetime::External,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    let resource = context
        .frame
        .add_resource(desc)
        .map_err(|error| map_frame(&error))?;
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
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
pub fn frame_enqueue_readback(context: ContextHandle, texture: TextureHandle) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = texture.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Texture)
            .map_err(map_lifecycle)?;
        let resource = intern_texture_resource(context, texture)?;
        let range = ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?;
        let state = ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferRead,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
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
// Missing surface/index resources and invalid ranges fail before the frame node is recorded.
fn graphics_node(
    context: &mut ContextState,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    indirect: IndirectBufferHandle,
    pipeline_layout: ez_gfx_runtime::binding::PipelineLayout,
) -> Result<NodeDesc, EzGfxResult> {
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
        .ok_or(EzGfxResult::NotReady)?;
    let load = if context.frame_has_graphics {
        LoadOp::Load
    } else {
        LoadOp::Clear
    };
    let pass = PassInfo::new(
        vec![surface],
        depth,
        [0, 0, width, height],
        1,
        load,
        StoreOp::Store,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    let color_state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    let mut node = NodeDesc::new("graphics", QueueKind::Graphics)
        .access(Access::image(
            surface,
            ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
            color_state,
        ))
        .pass(pass);
    if let Some(depth) = depth {
        let depth_state = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::Fragment,
            ResourceAccess::DepthStencilWrite,
        )
        .map_err(|_| EzGfxResult::InvalidArgument)?;
        node = node.access(Access::image(
            depth,
            ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
            depth_state,
        ));
    }
    let index_resource = intern_index_resource(context)?;
    let index_size = context
        .index_heap
        .as_ref()
        .ok_or(EzGfxResult::NotReady)?
        .size;
    let index_state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::AllGraphics,
        ResourceAccess::IndexRead,
    )
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    node = node.access(Access::buffer(
        index_resource,
        BufferRange::new(0, index_size).map_err(|_| EzGfxResult::InvalidArgument)?,
        index_state,
    ));
    let indirect_binding = layout.requirements().iter().find_map(|requirement| {
        let binding = bindings
            .iter()
            .find(|binding| binding.name == requirement.name)?;
        matches!(
            binding.resource,
            ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) if handle == indirect
        )
        .then_some(requirement.writable)
    });
    let indirect_size = context
        .allocations
        .get(&indirect.packed())
        .map(|(size, _)| *size)
        .ok_or(EzGfxResult::InvalidContext)?;
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
    .map_err(|_| EzGfxResult::InvalidArgument)?;
    node = node.access(Access::buffer(
        indirect_resource,
        BufferRange::new(0, indirect_size).map_err(|_| EzGfxResult::InvalidArgument)?,
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
) -> Result<NodeDesc, EzGfxResult> {
    // Unpublished textures cannot be sampled yet; unrelated uploads must not stall the heap.
    let texture_handles: Vec<_> = context
        .texture_published_mips
        .iter()
        .filter_map(|(texture, mips)| (*mips != 0).then_some(*texture))
        .collect();
    for texture in texture_handles {
        let resource = intern_texture_resource(context, texture)?;
        let sampled = ResourceState::new(queue, stage, ResourceAccess::SampledRead)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        node = node.access(Access::image(
            resource,
            ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
            sampled,
        ));
    }
    Ok(node)
}

/// Records an indexed graphics operation.
pub fn render_add_graphics(
    context: ContextHandle,
    shader: ShaderHandle,
    indirect: IndirectBufferHandle,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    state: DynamicPipelineState,
    push_constants: &[u8],
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let shader_handle = shader.packed();
        context
            .identity
            .resolve(shader_handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        let indirect_handle = indirect.packed();
        context
            .identity
            .resolve(indirect_handle, ResourceKind::Indirect)
            .map_err(map_lifecycle)?;
        validate_binding_handles(context, bindings)?;
        let record = context
            .shaders
            .get(&shader)
            .ok_or(EzGfxResult::InvalidContext)?;
        let layout = record
            .runtime
            .bindings(ez_gfx_artifact::Stage::Vertex)
            .and_then(|vertex| {
                record
                    .runtime
                    .bindings(ez_gfx_artifact::Stage::Fragment)
                    .and_then(|fragment| vertex.merge(&fragment))
            })
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        layout
            .validate(bindings)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        let draw_count = context
            .indirects
            .get(&indirect)
            .ok_or(EzGfxResult::InvalidContext)?
            .draw_count();
        if draw_count == 0 || push_constants.len() > 128 || !push_constants.len().is_multiple_of(4)
        {
            return Err(EzGfxResult::InvalidArgument);
        }
        let pipeline_layout = *record
            .graphics_layout
            .as_ref()
            .ok_or(EzGfxResult::InvalidArgument)?;
        let node = graphics_node(context, &layout, bindings, indirect, pipeline_layout)?;
        context
            .frame
            .record_node(
                node,
                ExecutableNode::Graphics {
                    shader,
                    indirect,
                    draw_count,
                    bindings: bindings.to_vec(),
                    layout,
                    pipeline_layout,
                    state,
                    push_constants: push_constants.to_vec(),
                },
            )
            .map_err(|error| map_frame(&error))?;
        context.frame_has_graphics = true;
        Ok(())
    }))
}
/// Records a compute dispatch.
pub fn render_add_compute(
    context: ContextHandle,
    shader: ShaderHandle,
    groups: [u32; 3],
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
    push_constants: &[u8],
) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
        let handle = shader.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Shader)
            .map_err(map_lifecycle)?;
        validate_binding_handles(context, bindings)?;
        let record = context
            .shaders
            .get(&shader)
            .ok_or(EzGfxResult::InvalidContext)?;
        if groups.contains(&0)
            || push_constants.len() > 128
            || !push_constants.len().is_multiple_of(4)
        {
            return Err(EzGfxResult::InvalidArgument);
        }
        let layout = record
            .runtime
            .bindings(ez_gfx_artifact::Stage::Compute)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
        layout
            .validate(bindings)
            .map_err(|_| EzGfxResult::InvalidArgument)?;
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
                    push_constants: push_constants.to_vec(),
                },
            )
            .map_err(|error| map_frame(&error))?;
        Ok(())
    }))
}

fn validate_binding_handles(
    context: &ContextState,
    bindings: &[ez_gfx_runtime::binding::PublicBinding],
) -> Result<(), EzGfxResult> {
    for binding in bindings {
        let (packed, kind) = match binding.resource {
            ez_gfx_runtime::binding::ResourceIdentity::Structured(handle) => {
                (handle.packed(), ResourceKind::Structured)
            }
            ez_gfx_runtime::binding::ResourceIdentity::Indirect(handle) => {
                (handle.packed(), ResourceKind::Indirect)
            }
            ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(handle) => {
                (handle.packed(), ResourceKind::RenderTarget)
            }
        };
        context
            .identity
            .resolve(packed, kind)
            .map_err(map_lifecycle)?;
    }
    Ok(())
}

/// Submits the recorded frame.
pub fn frame_submit(context: ContextHandle) -> EzGfxResult {
    result_status(with_context_mut(context, |context| {
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
            if context.frame_has_graphics {
                let surface = context.active_surface.ok_or(EzGfxResult::NotReady)?;
                let resource = context.frame_surface.ok_or(EzGfxResult::NotReady)?;
                let present = ResourceState::new(
                    QueueKind::Graphics,
                    ShaderStage::None,
                    ResourceAccess::Present,
                )
                .map_err(|_| EzGfxResult::InvalidArgument)?;
                context
                    .frame
                    .record_node(
                        NodeDesc::new("present", QueueKind::Graphics).access(Access::image(
                            resource,
                            ImageRange::all(1, 1).map_err(|_| EzGfxResult::InvalidArgument)?,
                            present,
                        )),
                        ExecutableNode::Present { surface },
                    )
                    .map_err(|error| map_frame(&error))?;
            }
            let submission = context.frame.submit().map_err(|error| map_frame(&error))?;
            let mut adapter = NativeFrameAdapter { context };
            execute_compiled_graph(&submission.graph, &submission.nodes, &mut adapter)
                .map_err(|error| map_execution(&error))?;
            adapter
                .context
                .frame
                .finish()
                .map_err(|error| map_frame(&error))?;
            let record = runtime_record(adapter.context, 0, RuntimePhase::Submit, EzGfxResult::Ok);
            adapter.context.observability.push_event(record);
            Ok(())
        })();
        if let Err(status) = result {
            context.frame.abort();
            let record = runtime_record(context, 0, RuntimePhase::Submit, status);
            context
                .observability
                .push_diagnostic(DiagnosticLevel::Error, record);
        }
        result
    }))
}

/// Returns the completed frame readback.
///
/// # Errors
///
/// Returns an error when the context is invalid or no completed readback is available.
pub fn frame_readback(context: ContextHandle) -> Result<Vec<u8>, EzGfxResult> {
    with_context_mut(context, |context| {
        if context.last_readback.is_empty() {
            return Err(EzGfxResult::NotReady);
        }
        Ok(context.last_readback.clone())
    })
}

fn map_execution(error: &ExecutionError<EzGfxResult>) -> EzGfxResult {
    match error {
        ExecutionError::Backend(error) => *error,
        ExecutionError::MissingPayload { .. }
        | ExecutionError::UnexpectedPayloads
        | ExecutionError::InvalidCompiledRange => EzGfxResult::InvalidArgument,
    }
}

struct NativeFrameAdapter<'a> {
    context: &'a mut ContextState,
}

impl FrameExecutionBackend<ExecutableNode> for NativeFrameAdapter<'_> {
    type Error = EzGfxResult;

    fn execute(
        &mut self,
        plan: &FrameExecutionPlan,
        payloads: &[ExecutableNode],
    ) -> Result<(), Self::Error> {
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
        Err(EzGfxResult::NativeFailure)
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
