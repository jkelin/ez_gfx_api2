use super::*;

#[cfg(target_vendor = "apple")]
pub(super) fn execute_metal_frame_plan(
    context: &mut ContextState,
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

    let mut binding_sets = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let bindings = match payload {
            ExecutableNode::Graphics {
                layout, bindings, ..
            }
            | ExecutableNode::Compute {
                layout, bindings, ..
            } => metal_bindings(layout, bindings, &context.allocations).map_err(map_hal)?,
            ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => Vec::new(),
        };
        binding_sets.push(bindings);
    }
    let native_textures: Vec<_> = context
        .textures
        .values()
        .map(|(_, texture, _, _, _)| match texture {
            NativeTexture::Metal(texture) => Ok(texture),
            _ => Err(EzGfxResult::NativeFailure),
        })
        .collect::<Result<_, _>>()?;
    let index = match context.index_heap.as_ref().map(|heap| &heap.allocation) {
        Some(NativeAllocation::Metal(index)) => Some(index),
        Some(_) => return Err(EzGfxResult::NativeFailure),
        None => None,
    };

    let NativeContext::Metal(native) = &mut context.native else {
        return Err(EzGfxResult::NativeFailure);
    };
    let mut pipeline_keys = vec![None; payloads.len()];
    let mut texture_heaps = vec![None; payloads.len()];
    for (node_index, payload) in payloads.iter().enumerate() {
        let (key, pipeline) = match payload {
            ExecutableNode::Compute { shader, layout, .. } => {
                let record = context
                    .shaders
                    .get(shader)
                    .ok_or(EzGfxResult::InvalidContext)?;
                let compute = record
                    .compute
                    .as_ref()
                    .ok_or(EzGfxResult::InvalidArgument)?;
                let NativeShader::Metal(native_shader) = &record.native else {
                    return Err(EzGfxResult::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let key = PipelineKey::Compute {
                    backend: Backend::Metal,
                    shader: *shader,
                    shader_digest: record.digest,
                    product: compute.0,
                    entry: compute.1.clone(),
                    layouts: pipeline_layout_key(&layouts),
                };
                let pipeline = if context.pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Metal(
                        native
                            .create_compute_pipeline(native_shader, compute.0, &compute.1)
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
                let record = context
                    .shaders
                    .get(shader)
                    .ok_or(EzGfxResult::InvalidContext)?;
                let graphics = record
                    .graphics
                    .as_ref()
                    .ok_or(EzGfxResult::InvalidArgument)?;
                let NativeShader::Metal(native_shader) = &record.native else {
                    return Err(EzGfxResult::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let texture_heap = pipeline_layout
                    .texture_heap()
                    .map(|layout| {
                        ez_gfx_hal::ShaderTextureHeapLayout::new(
                            layout.space,
                            layout.binding,
                            layout.capacity,
                            layout.argument_stride,
                            layout.texture_argument_offset,
                            layout.sampler_argument_offset,
                        )
                    })
                    .transpose()
                    .map_err(map_hal)?;
                let depth_required = pipeline_layout.depth_required();
                let key = PipelineKey::Graphics {
                    backend: Backend::Metal,
                    shader: *shader,
                    shader_digest: record.digest,
                    vertex_product: graphics.0,
                    vertex_entry: graphics.1.clone(),
                    fragment_product: graphics.2,
                    fragment_entry: graphics.3.clone(),
                    layouts: pipeline_layout_key(&layouts),
                    texture_heap,
                    state: *state,
                    depth_required,
                    color_format: 80,
                    depth_format: if depth_required { 252 } else { 0 },
                    sample_count: 1,
                };
                let pipeline = if context.pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Metal(
                        native
                            .create_graphics_pipeline(
                                native_shader,
                                graphics,
                                *state,
                                depth_required,
                                texture_heap,
                            )
                            .map_err(map_hal)?,
                    ))
                };
                texture_heaps[node_index] = texture_heap;
                (key, pipeline)
            }
            ExecutableNode::TextureReadback { .. } | ExecutableNode::Present { .. } => continue,
        };
        if let Some(pipeline) = pipeline {
            if context.pipelines.len() == MAX_PIPELINE_CACHE_ENTRIES {
                native.wait_idle().map_err(map_hal)?;
                let stale = context
                    .pipelines
                    .drain()
                    .map(|(_, pipeline)| pipeline)
                    .collect::<Vec<_>>();
                for pipeline in stale {
                    let NativePipeline::Metal(pipeline) = pipeline else {
                        return Err(EzGfxResult::NativeFailure);
                    };
                    native.destroy_pipeline(pipeline);
                }
            }
            context.pipelines.insert(key.clone(), pipeline);
        }
        pipeline_keys[node_index] = Some(key);
    }

    let mut actions = Vec::with_capacity(plan.actions.len());
    for action in &plan.actions {
        match action {
            ExecutionAction::Wait(wait) => {
                if let Some(token) = wait.external {
                    actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Wait(token));
                }
            }
            ExecutionAction::Barrier(barrier) => {
                let resource = context
                    .frame_native_resources
                    .get(&ResourceId::from_index(barrier.resource))
                    .ok_or(EzGfxResult::InvalidArgument)?;
                let resource = match *resource {
                    FrameNativeResource::Buffer(handle) => {
                        let NativeAllocation::Metal(allocation) = &context
                            .allocations
                            .get(&handle)
                            .ok_or(EzGfxResult::InvalidContext)?
                            .1
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        ez_gfx_backend_metal::native::NativeFrameResource::Buffer(allocation)
                    }
                    FrameNativeResource::Texture(handle) => {
                        let (_, NativeTexture::Metal(texture), _, _, _) = context
                            .textures
                            .get(&handle)
                            .ok_or(EzGfxResult::InvalidContext)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        ez_gfx_backend_metal::native::NativeFrameResource::Texture(texture)
                    }
                    FrameNativeResource::Surface(_) => {
                        ez_gfx_backend_metal::native::NativeFrameResource::Surface
                    }
                    FrameNativeResource::Depth => {
                        ez_gfx_backend_metal::native::NativeFrameResource::Depth
                    }
                    FrameNativeResource::Index => {
                        ez_gfx_backend_metal::native::NativeFrameResource::Buffer(
                            index.ok_or(EzGfxResult::NotReady)?,
                        )
                    }
                };
                actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                });
            }
            ExecutionAction::BeginPass(pass) => {
                actions.push(ez_gfx_backend_metal::native::NativeFrameAction::BeginPass(
                    pass,
                ));
            }
            ExecutionAction::ExecuteNode(node) => {
                let index_node = *node as usize;
                let payload = payloads
                    .get(index_node)
                    .ok_or(EzGfxResult::InvalidArgument)?;
                match payload {
                    ExecutableNode::Graphics {
                        indirect,
                        draw_count,
                        pipeline_layout,
                        state,
                        push_constants,
                        ..
                    } => {
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        let NativePipeline::Metal(pipeline) = context
                            .pipelines
                            .get(key)
                            .ok_or(EzGfxResult::NativeFailure)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        let NativeAllocation::Metal(indirect) = &context
                            .allocations
                            .get(&indirect.packed())
                            .ok_or(EzGfxResult::InvalidContext)?
                            .1
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        let texture_heap = texture_heaps[index_node];
                        actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Graphics(
                            ez_gfx_backend_metal::native::NativeGraphicsDraw {
                                pipeline,
                                depth_required: pipeline_layout.depth_required(),
                                texture_heap,
                                state: *state,
                                index: index.ok_or(EzGfxResult::NotReady)?,
                                indirect,
                                draw_count: *draw_count,
                                push_constants,
                                bindings: &binding_sets[index_node],
                                textures: &native_textures,
                            },
                        ));
                    }
                    ExecutableNode::Compute {
                        groups,
                        push_constants,
                        ..
                    } => {
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(EzGfxResult::InvalidArgument)?;
                        let NativePipeline::Metal(pipeline) = context
                            .pipelines
                            .get(key)
                            .ok_or(EzGfxResult::NativeFailure)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Compute(
                            ez_gfx_backend_metal::native::NativeComputeDispatch {
                                pipeline,
                                groups: *groups,
                                push_constants,
                                bindings: &binding_sets[index_node],
                            },
                        ));
                    }
                    ExecutableNode::TextureReadback { texture } => {
                        let (_, NativeTexture::Metal(texture), width, height, _) = context
                            .textures
                            .get(texture)
                            .ok_or(EzGfxResult::InvalidContext)?
                        else {
                            return Err(EzGfxResult::NativeFailure);
                        };
                        actions.push(
                            ez_gfx_backend_metal::native::NativeFrameAction::TextureReadback {
                                texture,
                                width: *width,
                                height: *height,
                            },
                        );
                    }
                    ExecutableNode::Present { surface } => {
                        if Some(*surface) != surface_handle {
                            return Err(EzGfxResult::InvalidArgument);
                        }
                        actions.push(ez_gfx_backend_metal::native::NativeFrameAction::Present);
                    }
                }
            }
            ExecutionAction::EndPass => {
                actions.push(ez_gfx_backend_metal::native::NativeFrameAction::EndPass);
            }
        }
    }

    let result = match (
        &mut context.native,
        surface.as_mut().map(|surface| &mut surface.native),
    ) {
        (NativeContext::Metal(native), Some(NativeSurface::Metal(surface))) => native
            .execute_frame(Some((surface, extent)), &actions, capture)
            .map_err(map_hal),
        (NativeContext::Metal(native), None) => {
            native.execute_frame(None, &actions, false).map_err(map_hal)
        }
        _ => Err(EzGfxResult::NativeFailure),
    };
    if let Ok(Some(readback)) = &result {
        context.last_readback = readback.clone();
    }
    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
        context.surfaces.insert(handle, surface);
    }
    result?;
    context.frame_presented = payloads
        .iter()
        .any(|payload| matches!(payload, ExecutableNode::Present { .. }));
    Ok(())
}
