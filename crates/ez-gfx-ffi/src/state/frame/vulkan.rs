use super::{
    Backend, ExecutableNode, ExecutionAction, EzGfxResult, FfiContext, FrameExecutionPlan,
    FrameNativeResource, HashMap, MAX_PIPELINE_CACHE_ENTRIES, NativeAllocation, NativeContext,
    NativePipeline, NativeShader, NativeSurface, NativeTexture, NativeTextureMap, PipelineKey,
    ResourceId, ShaderRecord, map_hal, native_layouts, pipeline_layout_key, vulkan_bindings,
};

struct VulkanActionState<'a> {
    allocations: &'a HashMap<u64, (u64, NativeAllocation)>,
    textures: &'a NativeTextureMap,
    pipelines: &'a HashMap<PipelineKey, NativePipeline>,
    resources: &'a HashMap<ResourceId, FrameNativeResource>,
    index: Option<&'a ez_gfx_backend_vulkan::NativeAllocation>,
    extent: (u32, u32),
}

// A headless frame leaves the cached format and graphics pipelines unchanged.
fn prepare_vulkan_surface(
    native: &mut ez_gfx_backend_vulkan::NativeContext,
    pipelines: &mut HashMap<PipelineKey, NativePipeline>,
    graphics_format: &mut Option<u32>,
    surface: Option<&ez_gfx_backend_vulkan::NativeSurface>,
    extent: (u32, u32),
) -> Result<(), EzGfxResult> {
    if let Some(surface) = surface {
        native
            .prepare_surface(surface, extent.0, extent.1)
            .map_err(map_hal)?;
        let format = native.graphics_format_key();
        if graphics_format.is_some_and(|cached| cached != format) {
            // Old-format graphics pipelines remain valid native objects but cannot be reused.
            let stale = pipelines
                .extract_if(|key, _| matches!(key, PipelineKey::Graphics { .. }))
                .map(|(_, pipeline)| pipeline)
                .collect::<Vec<_>>();
            for pipeline in stale {
                let NativePipeline::Vulkan(pipeline) = pipeline else {
                    return Err(EzGfxResult::NativeFailure);
                };
                native.destroy_pipeline(pipeline);
            }
        }
        *graphics_format = Some(format);
    }
    Ok(())
}

// Unsupported shader variants fail without inserting a partial pipeline-cache entry.
fn prepare_vulkan_pipelines(
    native: &mut ez_gfx_backend_vulkan::NativeContext,
    shaders: &HashMap<u64, ShaderRecord>,
    pipelines: &mut HashMap<PipelineKey, NativePipeline>,
    payloads: &[ExecutableNode],
) -> Result<Vec<Option<PipelineKey>>, EzGfxResult> {
    let mut pipeline_keys: Vec<Option<PipelineKey>> = (0..payloads.len()).map(|_| None).collect();
    for (node_index, payload) in payloads.iter().enumerate() {
        let (key, pipeline) = match payload {
            ExecutableNode::Compute { shader, layout, .. } => {
                let record = shaders.get(shader).ok_or(EzGfxResult::InvalidContext)?;
                let compute = record
                    .compute
                    .as_ref()
                    .ok_or(EzGfxResult::InvalidArgument)?;
                let NativeShader::Vulkan(native_shader) = &record.native else {
                    return Err(EzGfxResult::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let key = PipelineKey::Compute {
                    backend: Backend::Vulkan,
                    shader: *shader,
                    shader_digest: record.digest,
                    product: compute.0,
                    entry: compute.1.clone(),
                    layouts: pipeline_layout_key(&layouts),
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Vulkan(
                        native
                            .create_compute_pipeline(native_shader, compute.0, &compute.1, &layouts)
                            .map_err(map_hal)?,
                    ))
                };
                (key, pipeline)
            }
            ExecutableNode::Graphics {
                shader,
                layout,
                pipeline_layout,
                state,
                ..
            } => {
                let record = shaders.get(shader).ok_or(EzGfxResult::InvalidContext)?;
                let graphics = record
                    .graphics
                    .as_ref()
                    .ok_or(EzGfxResult::InvalidArgument)?;
                let NativeShader::Vulkan(native_shader) = &record.native else {
                    return Err(EzGfxResult::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let depth_required = pipeline_layout.depth_required();
                let key = PipelineKey::Graphics {
                    backend: Backend::Vulkan,
                    shader: *shader,
                    shader_digest: record.digest,
                    vertex_product: graphics.0,
                    vertex_entry: graphics.1.clone(),
                    fragment_product: graphics.2,
                    fragment_entry: graphics.3.clone(),
                    texture_heap: None,
                    layouts: pipeline_layout_key(&layouts),
                    state: *state,
                    depth_required,
                    color_format: native.graphics_format_key(),
                    depth_format: u32::from(depth_required),
                    sample_count: 1,
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Vulkan(
                        native
                            .create_graphics_pipeline(
                                native_shader,
                                ez_gfx_backend_vulkan::NativeGraphicsPipelineDesc {
                                    vertex_index: graphics.0,
                                    fragment_index: graphics.2,
                                    state: *state,
                                    depth_required,
                                    layouts: &layouts,
                                },
                            )
                            .map_err(map_hal)?,
                    ))
                };
                (key, pipeline)
            }
            ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => continue,
        };
        if let Some(pipeline) = pipeline {
            if pipelines.len() == MAX_PIPELINE_CACHE_ENTRIES {
                native.wait_idle().map_err(map_hal)?;
                let stale = pipelines
                    .drain()
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>();
                for stale_pipeline in stale {
                    let NativePipeline::Vulkan(stale_pipeline) = stale_pipeline else {
                        return Err(EzGfxResult::NativeFailure);
                    };
                    native.destroy_pipeline(stale_pipeline);
                }
            }
            pipelines.insert(key.clone(), pipeline);
        }
        pipeline_keys[node_index] = Some(key);
    }
    Ok(pipeline_keys)
}

// Missing resources and mismatched backend variants abort action construction before submission.
fn vulkan_actions<'a>(
    state: &'a VulkanActionState<'a>,
    plan: &'a FrameExecutionPlan,
    payloads: &'a [ExecutableNode],
    binding_sets: &'a [Vec<ez_gfx_backend_vulkan::NativeBufferBinding<'a>>],
    pipeline_keys: &[Option<PipelineKey>],
) -> Result<Vec<ez_gfx_backend_vulkan::NativeFrameAction<'a>>, EzGfxResult> {
    let mut actions = Vec::with_capacity(plan.actions.len());
    for action in &plan.actions {
        match action {
            ExecutionAction::Wait(wait) => {
                if let Some(token) = wait.external {
                    actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Wait(token));
                }
            }
            ExecutionAction::Barrier(barrier) => {
                let resource = state
                    .resources
                    .get(&ResourceId::from_index(barrier.resource))
                    .ok_or(EzGfxResult::InvalidArgument)?;
                let resource = match *resource {
                    FrameNativeResource::Buffer(handle) => {
                        let NativeAllocation::Vulkan(allocation) = &state
                            .allocations
                            .get(&handle)
                            .ok_or(EzGfxResult::InvalidContext)?
                            .1
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        ez_gfx_backend_vulkan::NativeFrameResource::Buffer(allocation)
                    }
                    FrameNativeResource::Texture(handle) => {
                        let (_, NativeTexture::Vulkan(texture), _, _, _) = state
                            .textures
                            .get(&handle)
                            .ok_or(EzGfxResult::InvalidContext)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        ez_gfx_backend_vulkan::NativeFrameResource::Texture(texture)
                    }
                    FrameNativeResource::Surface(_) => {
                        ez_gfx_backend_vulkan::NativeFrameResource::Surface
                    }
                    FrameNativeResource::Depth => ez_gfx_backend_vulkan::NativeFrameResource::Depth,
                    FrameNativeResource::Index => {
                        ez_gfx_backend_vulkan::NativeFrameResource::Buffer(
                            state.index.ok_or(EzGfxResult::NotReady)?,
                        )
                    }
                };
                actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                });
            }
            ExecutionAction::BeginPass(pass) => {
                actions.push(ez_gfx_backend_vulkan::NativeFrameAction::BeginPass(pass));
            }
            ExecutionAction::ExecuteNode(node) => {
                let index_node = *node as usize;
                let payload = payloads
                    .get(index_node)
                    .ok_or(EzGfxResult::InvalidArgument)?;
                match payload {
                    ExecutableNode::Compute {
                        groups,
                        push_constants,
                        ..
                    } => {
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        let NativePipeline::Vulkan(pipeline) =
                            state.pipelines.get(key).ok_or(EzGfxResult::NativeFailure)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Compute(
                            ez_gfx_backend_vulkan::NativeComputeDispatch {
                                pipeline,
                                groups: *groups,
                                push_constants,
                                bindings: &binding_sets[index_node],
                            },
                        ));
                    }
                    ExecutableNode::Graphics {
                        indirect,
                        draw_count,
                        push_constants,
                        ..
                    } => {
                        let NativeAllocation::Vulkan(indirect) = &state
                            .allocations
                            .get(indirect)
                            .ok_or(EzGfxResult::InvalidContext)?
                            .1
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        let NativePipeline::Vulkan(pipeline) =
                            state.pipelines.get(key).ok_or(EzGfxResult::NativeFailure)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Graphics(
                            ez_gfx_backend_vulkan::NativeDrawIndexed {
                                width: state.extent.0,
                                height: state.extent.1,
                                pipeline,
                                index_buffer: state.index.ok_or(EzGfxResult::NotReady)?,
                                indirect_buffer: indirect,
                                draw_count: *draw_count,
                                push_constants,
                                bindings: &binding_sets[index_node],
                            },
                        ));
                    }
                    ExecutableNode::TextureReadback { texture } => {
                        let (_, NativeTexture::Vulkan(texture), width, height, _) = state
                            .textures
                            .get(texture)
                            .ok_or(EzGfxResult::InvalidContext)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::TextureReadback {
                            texture,
                            width: *width,
                            height: *height,
                        });
                    }
                    ExecutableNode::Present { .. } => {
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Present);
                    }
                }
            }
            ExecutionAction::EndPass => {
                actions.push(ez_gfx_backend_vulkan::NativeFrameAction::EndPass);
            }
        }
    }
    Ok(actions)
}

pub(super) fn execute_vulkan_frame_plan(
    context: &mut FfiContext,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
) -> Result<(), EzGfxResult> {
    let surface_handle = payloads.iter().find_map(|payload| match payload {
        ExecutableNode::Present { surface } => Some(*surface),
        _ => None,
    });
    let mut surface = surface_handle
        .map(|handle| {
            context
                .surfaces
                .remove(&handle)
                .ok_or(EzGfxResult::InvalidContext)
        })
        .transpose()?;
    let extent = surface
        .as_ref()
        .and_then(|surface| surface.state.extent())
        .unwrap_or((0, 0));
    let capture = surface
        .as_ref()
        .is_some_and(|surface| surface.state.snapshot_cache());

    let index = match context.index_heap.as_ref().map(|heap| &heap.allocation) {
        Some(NativeAllocation::Vulkan(index)) => Some(index),
        Some(_) => return Err(EzGfxResult::NativeFailure),
        None => None,
    };
    if surface
        .as_ref()
        .is_some_and(|surface| !matches!(surface.native, NativeSurface::Vulkan(_)))
    {
        if let (Some(handle), Some(surface)) = (surface_handle, surface) {
            context.surfaces.insert(handle, surface);
        }
        return Err(EzGfxResult::NativeFailure);
    }
    let mut native_surface = surface.as_mut().map(|surface| {
        let NativeSurface::Vulkan(surface) = &mut surface.native else {
            unreachable!("surface variant validated");
        };
        surface
    });
    let pipeline_keys = {
        let NativeContext::Vulkan(native) = &mut context.native else {
            return Err(EzGfxResult::NativeFailure);
        };
        prepare_vulkan_surface(
            native,
            &mut context.pipelines,
            &mut context.graphics_format,
            native_surface.as_deref(),
            extent,
        )?;
        prepare_vulkan_pipelines(native, &context.shaders, &mut context.pipelines, payloads)?
    };
    let binding_sets = payloads
        .iter()
        .map(|payload| match payload {
            ExecutableNode::Compute {
                layout, bindings, ..
            }
            | ExecutableNode::Graphics {
                layout, bindings, ..
            } => vulkan_bindings(layout, bindings, &context.allocations).map_err(map_hal),
            ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => {
                Ok(Vec::new())
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state = VulkanActionState {
        allocations: &context.allocations,
        textures: &context.textures,
        pipelines: &context.pipelines,
        resources: &context.frame_native_resources,
        index,
        extent,
    };
    let actions = vulkan_actions(&state, plan, payloads, &binding_sets, &pipeline_keys)?;
    let execution = {
        let NativeContext::Vulkan(native) = &mut context.native else {
            return Err(EzGfxResult::NativeFailure);
        };
        native
            .execute_frame(
                native_surface
                    .as_deref_mut()
                    .map(|surface| (surface, extent)),
                &actions,
                capture,
            )
            .map_err(map_hal)
    };
    drop(actions);
    let outcome = match execution {
        Ok(outputs) => {
            let texture_readbacks = payloads
                .iter()
                .filter(|payload| matches!(payload, ExecutableNode::TextureReadback { .. }))
                .count();
            if texture_readbacks != 0 {
                let Some(readback) = outputs.get(texture_readbacks - 1) else {
                    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
                        context.surfaces.insert(handle, surface);
                    }
                    return Err(EzGfxResult::NativeFailure);
                };
                context.last_readback.clone_from(readback);
            }
            if capture && let Some(native_surface) = native_surface.as_deref() {
                context.last_readback = native_surface.presented_rgba8().to_vec();
            }
            context.frame_presented = payloads
                .iter()
                .any(|payload| matches!(payload, ExecutableNode::Present { .. }));
            Ok(())
        }
        Err(error) => Err(error),
    };
    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
        context.surfaces.insert(handle, surface);
    }
    outcome
}
