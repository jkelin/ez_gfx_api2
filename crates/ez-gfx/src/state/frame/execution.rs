// Attachment readiness is checked before any indexed-only resource is interned.
fn mark_attached_target_written(context: &mut ContextState) -> Result<()> {
    let Some(target) = context.frame_render_target else {
        return Ok(());
    };
    let state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    )
    .map_err(|_| Error::InvalidArgument)?;
    context.frame_render_target_states.insert(target, state);
    Ok(())
}

fn graphics_pass_node(
    context: &mut ContextState,
    pipeline_layout: ez_gfx_runtime::binding::PipelineLayout,
    name: &'static str,
) -> Result<NodeDesc> {
    // A managed target may carry its own backend depth companion; the backend
    // validates that attachment during pass lowering.
    let target_had_prior_access = context
        .frame_render_target
        .is_some_and(|target| context.frame_resources.contains_key(&target.packed()));
    let (color, depth, width, height, samples) = if let Some(target) = context.frame_render_target {
        let resource = intern_render_target_resource(context, target)?;
        let (width, height, samples) = {
            let record = context
                .render_targets
                .get(&target)
                .ok_or(Error::InvalidContext)?;
            (record.width, record.height, record.declaration.samples())
        };
        let depth = if pipeline_layout.depth_required() {
            Some(intern_depth_resource(context)?)
        } else {
            None
        };
        (resource, depth, width, height, samples)
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
    let load = if context.frame_has_graphics
        || context.frame_render_target_load == RenderTargetLoad::Preserve
        || target_had_prior_access
    {
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
        mark_attached_target_written(context)?;
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
            mark_attached_target_written(context)?;
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
    if !bindings.arenas_are_read_only(&context.frame_arena_buffers) {
        return Err(Error::Unsupported);
    }
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
        if kind == ResourceKind::Buffer
            && BufferHandle::from_packed(packed)
                .is_ok_and(|handle| context.gpu_arenas.contains_key(&handle))
        {
            continue;
        }
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
        if context.frame_arena_buffers.contains(&handle) {
            continue;
        }
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
