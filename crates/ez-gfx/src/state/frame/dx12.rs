use crate::Result;

use super::{
    Backend, ContextState, Error, ExecutableNode, ExecutionAction, ExecutionBarrier, ExecutionPass,
    FrameExecutionPlan, FrameNativeResource, GeometryAllocation, HashMap,
    MAX_PIPELINE_CACHE_ENTRIES, NativeAllocation, NativeContext, NativePipeline, NativeShader,
    NativeSurface, NativeTexture, NativeTextureMap, PackedHandle, PipelineKey, RenderTargetHandle,
    RenderTargetRecord, ResourceId, SURFACE_DEFAULT_CLEAR, ShaderHandle, ShaderRecord,
    dx12_bindings, map_hal, native_layouts, pipeline_layout_key,
};

struct DxActionState<'a> {
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
    textures: &'a NativeTextureMap,
    render_targets: &'a HashMap<RenderTargetHandle, RenderTargetRecord>,
    pipelines: &'a HashMap<PipelineKey, NativePipeline>,
    resources: &'a HashMap<ResourceId, FrameNativeResource>,
    index: Option<&'a ez_gfx_backend_dx12::native::NativeAllocation>,
    index_size: u64,
    extent: (u32, u32),
}

// Unsupported shader variants fail without inserting a partial pipeline-cache entry.
fn prepare_dx12_pipelines(
    native: &mut ez_gfx_backend_dx12::native::NativeContext,
    shaders: &HashMap<ShaderHandle, ShaderRecord>,
    pipelines: &mut HashMap<PipelineKey, NativePipeline>,
    payloads: &[ExecutableNode],
) -> Result<Vec<Option<PipelineKey>>> {
    let mut pipeline_keys: Vec<Option<PipelineKey>> = (0..payloads.len()).map(|_| None).collect();
    for (node_index, payload) in payloads.iter().enumerate() {
        let (key, pipeline) = match payload {
            ExecutableNode::Compute { shader, layout, .. } => {
                let record = shaders.get(shader).ok_or(Error::InvalidContext)?;
                let compute = record.compute.as_ref().ok_or(Error::InvalidArgument)?;
                let NativeShader::Dx12(native_shader) = &record.native else {
                    return Err(Error::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let key = PipelineKey::Compute {
                    backend: Backend::Dx12,
                    shader: *shader,
                    shader_digest: record.digest,
                    product: compute.0,
                    entry: compute.1.clone(),
                    layouts: pipeline_layout_key(&layouts),
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Dx12(
                        native
                            .create_compute_pipeline(native_shader, compute.0, &layouts)
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
                let record = shaders.get(shader).ok_or(Error::InvalidContext)?;
                let graphics = record.graphics.as_ref().ok_or(Error::InvalidArgument)?;
                let NativeShader::Dx12(native_shader) = &record.native else {
                    return Err(Error::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let depth_required = pipeline_layout.depth_required();
                let key = PipelineKey::Graphics {
                    backend: Backend::Dx12,
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
                    color_format: 28,
                    depth_format: if depth_required { 40 } else { 0 },
                    sample_count: 1,
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Dx12(
                        native
                            .create_graphics_pipeline(
                                native_shader,
                                graphics.0,
                                graphics.2,
                                *state,
                                depth_required,
                                &layouts,
                            )
                            .map_err(map_hal)?,
                    ))
                };
                (key, pipeline)
            }
            ExecutableNode::TextureReadback { .. }
            | ExecutableNode::RenderTargetReadback { .. }
            | ExecutableNode::Present { .. } => continue,
        };
        if let Some(pipeline) = pipeline {
            if pipelines.len() == MAX_PIPELINE_CACHE_ENTRIES {
                native.wait_idle().map_err(map_hal)?;
                pipelines.clear();
            }
            pipelines.insert(key.clone(), pipeline);
        }
        pipeline_keys[node_index] = Some(key);
    }
    Ok(pipeline_keys)
}

// Barrier resource indices resolve to the native buffer, texture, surface,
// depth, render-target, or index resource transitioned before submission.
fn dx12_barrier_resource<'a>(
    state: &'a DxActionState<'a>,
    barrier: &ExecutionBarrier,
) -> Result<ez_gfx_backend_dx12::native::NativeFrameResource<'a>> {
    let resource = state
        .resources
        .get(&ResourceId::from_index(barrier.resource))
        .ok_or(Error::InvalidArgument)?;
    Ok(match *resource {
        FrameNativeResource::Buffer(handle) => {
            let NativeAllocation::Dx12(allocation) = &state
                .allocations
                .get(&handle)
                .ok_or(Error::InvalidContext)?
                .1
            else {
                return Err(Error::NativeFailure);
            };
            ez_gfx_backend_dx12::native::NativeFrameResource::Buffer(allocation)
        }
        FrameNativeResource::Texture(handle) => {
            let (_, NativeTexture::Dx12(texture), _, _, _) =
                state.textures.get(&handle).ok_or(Error::InvalidContext)?
            else {
                return Err(Error::NativeFailure);
            };
            ez_gfx_backend_dx12::native::NativeFrameResource::Texture(texture)
        }
        FrameNativeResource::Surface(_) => {
            ez_gfx_backend_dx12::native::NativeFrameResource::Surface
        }
        FrameNativeResource::Depth => ez_gfx_backend_dx12::native::NativeFrameResource::Depth,
        FrameNativeResource::RenderTarget(handle) => {
            let record = state
                .render_targets
                .get(&handle)
                .ok_or(Error::InvalidContext)?;
            let NativeTexture::Dx12(texture) = &record.native else {
                return Err(Error::NativeFailure);
            };
            ez_gfx_backend_dx12::native::NativeFrameResource::RenderTarget(texture)
        }
        FrameNativeResource::Index => ez_gfx_backend_dx12::native::NativeFrameResource::Buffer(
            state.index.ok_or(Error::NotReady)?,
        ),
        FrameNativeResource::VertexHeap(heap_id) => {
            let heap = state
                .vertex_heaps
                .values()
                .find(|heap| heap.heap_id == Some(heap_id))
                .ok_or(Error::InvalidContext)?;
            let NativeAllocation::Dx12(allocation) = &heap.allocation else {
                return Err(Error::NativeFailure);
            };
            ez_gfx_backend_dx12::native::NativeFrameResource::Buffer(allocation)
        }
    })
}

// Pass color indices resolve to surface or render-target attachments here;
// textures, buffers, and depth images are never color attachments. Surfaces
// keep the legacy clear.
fn dx12_pass_colors<'a>(
    state: &'a DxActionState<'a>,
    pass: &ExecutionPass,
) -> Result<Vec<ez_gfx_backend_dx12::native::PassAttachment<'a>>> {
    let mut colors = Vec::with_capacity(pass.colors.len());
    for index in &pass.colors {
        let resource = state
            .resources
            .get(&ResourceId::from_index(*index))
            .ok_or(Error::InvalidArgument)?;
        colors.push(match *resource {
            FrameNativeResource::Surface(_) => ez_gfx_backend_dx12::native::PassAttachment {
                resource: ez_gfx_backend_dx12::native::NativeFrameResource::Surface,
                clear: SURFACE_DEFAULT_CLEAR,
            },
            FrameNativeResource::RenderTarget(handle) => {
                let record = state
                    .render_targets
                    .get(&handle)
                    .ok_or(Error::InvalidContext)?;
                let NativeTexture::Dx12(texture) = &record.native else {
                    return Err(Error::NativeFailure);
                };
                ez_gfx_backend_dx12::native::PassAttachment {
                    resource: ez_gfx_backend_dx12::native::NativeFrameResource::RenderTarget(
                        texture,
                    ),
                    clear: super::super::render_target::render_target_clear_color(record),
                }
            }
            _ => return Err(Error::InvalidArgument),
        });
    }
    Ok(colors)
}

// Missing resources and mismatched backend variants abort action construction before submission.
fn dx12_actions<'a>(
    state: &'a DxActionState<'a>,
    plan: &'a FrameExecutionPlan,
    payloads: &'a [ExecutableNode],
    binding_sets: &'a [Vec<ez_gfx_backend_dx12::native::NativeBufferBinding<'a>>],
    pipeline_keys: &[Option<PipelineKey>],
) -> Result<Vec<ez_gfx_backend_dx12::native::NativeFrameAction<'a>>> {
    let mut actions = Vec::with_capacity(plan.actions.len());
    for action in &plan.actions {
        match action {
            ExecutionAction::Wait(wait) => {
                if let Some(token) = wait.external {
                    actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Wait(token));
                }
            }
            ExecutionAction::Barrier(barrier) => {
                let resource = dx12_barrier_resource(state, barrier)?;
                actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                });
            }
            ExecutionAction::BeginPass(pass) => {
                let colors = dx12_pass_colors(state, pass)?;
                actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::BeginPass {
                    pass,
                    colors,
                });
            }
            ExecutionAction::ExecuteNode(node) => {
                let index_node = *node as usize;
                match payloads.get(index_node).ok_or(Error::InvalidArgument)? {
                    ExecutableNode::Compute {
                        groups,
                        push_constants,
                        ..
                    } => {
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(Error::InvalidArgument)?;
                        let NativePipeline::Dx12(pipeline) =
                            state.pipelines.get(key).ok_or(Error::NativeFailure)?
                        else {
                            return Err(Error::NativeFailure);
                        };
                        actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Compute(
                            ez_gfx_backend_dx12::native::NativeComputeDispatch {
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
                        let (indirect_size, NativeAllocation::Dx12(indirect)) = state
                            .allocations
                            .get(&indirect.packed())
                            .ok_or(Error::InvalidContext)?
                        else {
                            return Err(Error::NativeFailure);
                        };
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(Error::InvalidArgument)?;
                        let NativePipeline::Dx12(pipeline) =
                            state.pipelines.get(key).ok_or(Error::NativeFailure)?
                        else {
                            return Err(Error::NativeFailure);
                        };
                        actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Graphics(
                            ez_gfx_backend_dx12::native::NativeDrawIndexed {
                                width: state.extent.0,
                                height: state.extent.1,
                                pipeline,
                                index_buffer: state.index.ok_or(Error::NotReady)?,
                                index_size: state.index_size,
                                indirect_buffer: indirect,
                                indirect_size: *indirect_size,
                                draw_count: *draw_count,
                                push_constants,
                                bindings: &binding_sets[index_node],
                            },
                        ));
                    }
                    ExecutableNode::TextureReadback { texture } => {
                        let (_, NativeTexture::Dx12(texture), width, height, _) =
                            state.textures.get(texture).ok_or(Error::InvalidContext)?
                        else {
                            return Err(Error::NativeFailure);
                        };
                        actions.push(
                            ez_gfx_backend_dx12::native::NativeFrameAction::TextureReadback {
                                texture,
                                width: *width,
                                height: *height,
                            },
                        );
                    }
                    ExecutableNode::RenderTargetReadback { target } => {
                        let record = state
                            .render_targets
                            .get(target)
                            .ok_or(Error::InvalidContext)?;
                        let NativeTexture::Dx12(texture) = &record.native else {
                            return Err(Error::NativeFailure);
                        };
                        actions.push(
                            ez_gfx_backend_dx12::native::NativeFrameAction::TextureReadback {
                                texture,
                                width: record.width,
                                height: record.height,
                            },
                        );
                    }
                    ExecutableNode::Present { .. } => {
                        actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::Present);
                    }
                }
            }
            ExecutionAction::EndPass => {
                actions.push(ez_gfx_backend_dx12::native::NativeFrameAction::EndPass);
            }
        }
    }
    Ok(actions)
}

#[cfg(windows)]
pub(super) fn execute_dx12_frame_plan(
    context: &mut ContextState,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
) -> Result<()> {
    let surface_handle = payloads.iter().find_map(|payload| match payload {
        ExecutableNode::Present { surface } => Some(*surface),
        _ => None,
    });
    let mut surface = surface_handle
        .map(|handle| {
            context
                .surfaces
                .remove(&handle)
                .ok_or(Error::InvalidContext)
        })
        .transpose()?;
    // Target-only frames size draws and validations from the target extents.
    let extent = surface
        .as_ref()
        .and_then(|surface| surface.state.extent())
        .or_else(|| {
            context
                .frame_render_target
                .and_then(|target| context.render_targets.get(&target))
                .map(|record| (record.width, record.height))
        })
        .unwrap_or((0, 0));
    let capture = surface
        .as_ref()
        .is_some_and(|surface| surface.state.snapshot_cache());
    let (index, index_size) = match context.index_heap.as_ref() {
        Some(heap) => match &heap.allocation {
            NativeAllocation::Dx12(index) => (Some(index), heap.size),
            NativeAllocation::Vulkan(_) => return Err(Error::NativeFailure),
        },
        None => (None, 0),
    };
    if surface
        .as_ref()
        .is_some_and(|surface| !matches!(surface.native, NativeSurface::Dx12(_)))
    {
        if let (Some(handle), Some(surface)) = (surface_handle, surface) {
            context.surfaces.insert(handle, surface);
        }
        return Err(Error::NativeFailure);
    }
    let mut native_surface = surface.as_mut().map(|surface| {
        let NativeSurface::Dx12(surface) = &mut surface.native else {
            unreachable!("surface variant validated");
        };
        surface
    });
    let binding_sets = payloads
        .iter()
        .map(|payload| match payload {
            ExecutableNode::Compute {
                layout, bindings, ..
            }
            | ExecutableNode::Graphics {
                layout, bindings, ..
            } => dx12_bindings(
                layout,
                bindings,
                &context.allocations,
                &context.vertex_heaps,
            )
            .map_err(map_hal),
            ExecutableNode::TextureReadback { .. }
            | ExecutableNode::RenderTargetReadback { .. }
            | ExecutableNode::Present { .. } => Ok(Vec::new()),
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let pipeline_keys = {
        let NativeContext::Dx12(native) = &mut context.native else {
            return Err(Error::NativeFailure);
        };
        prepare_dx12_pipelines(native, &context.shaders, &mut context.pipelines, payloads)?
    };
    let state = DxActionState {
        allocations: &context.allocations,
        vertex_heaps: &context.vertex_heaps,
        textures: &context.textures,
        render_targets: &context.render_targets,
        pipelines: &context.pipelines,
        resources: &context.frame_native_resources,
        index,
        index_size,
        extent,
    };
    let actions = dx12_actions(&state, plan, payloads, &binding_sets, &pipeline_keys)?;
    let execution = {
        let NativeContext::Dx12(native) = &mut context.native else {
            return Err(Error::NativeFailure);
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
                .filter(|payload| {
                    matches!(
                        payload,
                        ExecutableNode::TextureReadback { .. }
                            | ExecutableNode::RenderTargetReadback { .. }
                    )
                })
                .count();
            if texture_readbacks != 0 {
                let Some(readback) = outputs.get(texture_readbacks - 1) else {
                    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
                        context.surfaces.insert(handle, surface);
                    }
                    return Err(Error::NativeFailure);
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
