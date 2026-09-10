use crate::Result;
use ez_gfx_core::capability::PresentationMode;

use super::{
    Backend, ContextState, Error, ExecutableNode, ExecutionAction, FrameExecutionPlan,
    FrameNativeResource, GeometryAllocation, HashMap, MAX_PIPELINE_CACHE_ENTRIES, NativeAllocation,
    NativeContext, NativePipeline, NativeShader, NativeSurface, NativeTexture, PipelineKey,
    RenderTargetHandle, RenderTargetRecord, ResourceId, SURFACE_DEFAULT_CLEAR, ShaderRecord,
    TextureHandle, TextureId, map_hal, metal_bindings, native_layouts, pipeline_layout_key,
    should_capture_presented,
};
use ez_gfx_backend_metal::native::{
    NativeAllocation as MetalAllocation, NativeBufferBinding as MetalBufferBinding,
    NativeContext as MetalContext, NativeFrameAction as MetalFrameAction,
    NativeFrameResource as MetalFrameResource, NativeTexture as MetalTexture,
    PassAttachment as MetalPassAttachment,
};
use ez_gfx_core::handle::{PackedHandle, ShaderHandle, SurfaceHandle};
use ez_gfx_hal::{DynamicPipelineState, ShaderTextureHeapLayout};
use ez_gfx_runtime::binding::{PipelineLayout, ReflectedBindings, TextureHeapLayout};

type PreparedComputePipeline = (
    PipelineKey,
    Option<NativePipeline>,
    Option<ShaderTextureHeapLayout>,
    [u32; 3],
);
type PreparedGraphicsPipeline = (
    PipelineKey,
    Option<NativePipeline>,
    Option<ShaderTextureHeapLayout>,
);
type MetalTextureRecords = HashMap<TextureHandle, (TextureId, NativeTexture, u32, u32, u32)>;

#[cfg(target_vendor = "apple")]
struct PreparedMetalPipelines {
    keys: Vec<Option<PipelineKey>>,
    texture_heaps: Vec<Option<ShaderTextureHeapLayout>>,
    workgroup_sizes: Vec<Option<[u32; 3]>>,
}

fn metal_texture_heap(
    layout: Option<&TextureHeapLayout>,
) -> Result<Option<ShaderTextureHeapLayout>> {
    layout
        .map(|layout| {
            ShaderTextureHeapLayout::new(
                layout.space,
                layout.binding,
                layout.capacity,
                layout.argument_stride,
                layout.texture_argument_offset,
                layout.sampler_argument_offset,
            )
        })
        .transpose()
        .map_err(map_hal)
}

fn prepare_compute_pipeline(
    shaders: &HashMap<ShaderHandle, ShaderRecord>,
    pipelines: &HashMap<PipelineKey, NativePipeline>,
    native: &MetalContext,
    shader: ShaderHandle,
    layout: &ReflectedBindings,
) -> Result<PreparedComputePipeline> {
    let record = shaders.get(&shader).ok_or(Error::InvalidContext)?;
    let NativeShader::Metal(native_shader) = &record.native else {
        return Err(Error::NativeFailure);
    };
    let layouts = native_layouts(layout).map_err(map_hal)?;
    let compute_layout = record
        .runtime
        .pipeline_layout(ez_gfx_artifact::Stage::Compute)
        .map_err(|_| Error::InvalidArgument)?;
    let texture_heap = metal_texture_heap(compute_layout.texture_heap())?;
    let workgroup_size = record
        .runtime
        .compute_workgroup_size()
        .map_err(|_| Error::InvalidArgument)?;
    let key = PipelineKey::Compute {
        backend: Backend::Metal,
        shader,
        shader_digest: record.digest,
        entry: record.entry.clone(),
        layouts: pipeline_layout_key(&layouts),
    };
    let pipeline = if pipelines.contains_key(&key) {
        None
    } else {
        Some(NativePipeline::Metal(
            native
                .create_compute_pipeline(native_shader, record.product, &record.entry, texture_heap)
                .map_err(map_hal)?,
        ))
    };

    Ok((key, pipeline, texture_heap, workgroup_size))
}

fn prepare_graphics_pipeline(
    shaders: &HashMap<ShaderHandle, ShaderRecord>,
    pipelines: &HashMap<PipelineKey, NativePipeline>,
    native: &MetalContext,
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
    layout: &ReflectedBindings,
    pipeline_layout: &PipelineLayout,
    state: DynamicPipelineState,
) -> Result<PreparedGraphicsPipeline> {
    let vertex = shaders.get(&vertex_shader).ok_or(Error::InvalidContext)?;
    let fragment = shaders.get(&fragment_shader).ok_or(Error::InvalidContext)?;
    let NativeShader::Metal(vertex_native) = &vertex.native else {
        return Err(Error::NativeFailure);
    };
    let NativeShader::Metal(fragment_native) = &fragment.native else {
        return Err(Error::NativeFailure);
    };
    let layouts = native_layouts(layout).map_err(map_hal)?;
    let texture_heap = metal_texture_heap(pipeline_layout.texture_heap())?;
    let vertex_layout = vertex
        .runtime
        .pipeline_layout(ez_gfx_artifact::Stage::Vertex)
        .map_err(|_| Error::InvalidArgument)?;
    let vertex_texture_heap = metal_texture_heap(vertex_layout.texture_heap())?;
    let fragment_layout = fragment
        .runtime
        .pipeline_layout(ez_gfx_artifact::Stage::Fragment)
        .map_err(|_| Error::InvalidArgument)?;
    let fragment_texture_heap = metal_texture_heap(fragment_layout.texture_heap())?;
    let depth_required = pipeline_layout.depth_required();
    let key = PipelineKey::Graphics {
        backend: Backend::Metal,
        vertex_shader,
        vertex_digest: vertex.digest,
        vertex_entry: vertex.entry.clone(),
        fragment_shader,
        fragment_digest: fragment.digest,
        fragment_entry: fragment.entry.clone(),
        layouts: pipeline_layout_key(&layouts),
        texture_heap,
        state,
        depth_required,
        color_format: 80,
        depth_format: if depth_required { 252 } else { 0 },
        sample_count: 1,
    };
    let pipeline = if pipelines.contains_key(&key) {
        None
    } else {
        Some(NativePipeline::Metal(
            native
                .create_graphics_pipeline(
                    vertex_native,
                    fragment_native,
                    &(vertex.product, vertex.entry.clone()),
                    &(fragment.product, fragment.entry.clone()),
                    state,
                    depth_required,
                    vertex_texture_heap,
                    fragment_texture_heap,
                )
                .map_err(map_hal)?,
        ))
    };

    Ok((key, pipeline, texture_heap))
}

fn cache_metal_pipeline(
    pipelines: &mut HashMap<PipelineKey, NativePipeline>,
    native: &mut MetalContext,
    key: PipelineKey,
    pipeline: Option<NativePipeline>,
) -> Result<()> {
    let Some(pipeline) = pipeline else {
        return Ok(());
    };
    if pipelines.len() == MAX_PIPELINE_CACHE_ENTRIES {
        native.wait_idle().map_err(map_hal)?;
        let stale = pipelines
            .drain()
            .map(|(_, pipeline)| pipeline)
            .collect::<Vec<_>>();
        for pipeline in stale {
            let NativePipeline::Metal(pipeline) = pipeline else {
                return Err(Error::NativeFailure);
            };
            native.destroy_pipeline(pipeline);
        }
    }
    pipelines.insert(key, pipeline);
    Ok(())
}

fn prepare_metal_pipelines(
    shaders: &HashMap<ShaderHandle, ShaderRecord>,
    pipelines: &mut HashMap<PipelineKey, NativePipeline>,
    native: &mut MetalContext,
    payloads: &[ExecutableNode],
) -> Result<PreparedMetalPipelines> {
    let mut prepared = PreparedMetalPipelines {
        keys: vec![None; payloads.len()],
        texture_heaps: vec![None; payloads.len()],
        workgroup_sizes: vec![None; payloads.len()],
    };
    for (node_index, payload) in payloads.iter().enumerate() {
        let (key, pipeline) = match payload {
            ExecutableNode::Compute { shader, layout, .. } => {
                let (key, pipeline, texture_heap, workgroup_size) =
                    prepare_compute_pipeline(shaders, pipelines, native, *shader, layout)?;
                prepared.texture_heaps[node_index] = texture_heap;
                prepared.workgroup_sizes[node_index] = Some(workgroup_size);
                (key, pipeline)
            }
            ExecutableNode::Graphics {
                vertex_shader,
                fragment_shader,
                layout,
                pipeline_layout,
                state,
                ..
            } => {
                let (key, pipeline, texture_heap) = prepare_graphics_pipeline(
                    shaders,
                    pipelines,
                    native,
                    *vertex_shader,
                    *fragment_shader,
                    layout,
                    pipeline_layout,
                    *state,
                )?;
                prepared.texture_heaps[node_index] = texture_heap;
                (key, pipeline)
            }
            ExecutableNode::TextureReadback { .. }
            | ExecutableNode::RenderTargetReadback { .. }
            | ExecutableNode::Present { .. } => continue,
        };
        cache_metal_pipeline(pipelines, native, key.clone(), pipeline)?;
        prepared.keys[node_index] = Some(key);
    }
    Ok(prepared)
}

struct MetalActionInputs<'a> {
    payloads: &'a [ExecutableNode],
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
    textures: &'a MetalTextureRecords,
    render_targets: &'a HashMap<RenderTargetHandle, RenderTargetRecord>,
    pipelines: &'a HashMap<PipelineKey, NativePipeline>,
    frame_resources: &'a HashMap<ResourceId, FrameNativeResource>,
    binding_sets: &'a [Vec<MetalBufferBinding<'a>>],
    native_textures: &'a [&'a MetalTexture],
    index: Option<&'a MetalAllocation>,
    index_size: u64,
    surface: Option<SurfaceHandle>,
    prepared: &'a PreparedMetalPipelines,
}

impl<'a> MetalActionInputs<'a> {
    fn node(&self, index: usize) -> Result<MetalFrameAction<'a>> {
        let payload = self.payloads.get(index).ok_or(Error::InvalidArgument)?;
        match payload {
            ExecutableNode::Graphics {
                counter,
                draw_capacity,
                pipeline_layout,
                state,
                ..
            } => {
                let key = self.prepared.keys[index]
                    .as_ref()
                    .ok_or(Error::InvalidArgument)?;
                let NativePipeline::Metal(pipeline) =
                    self.pipelines.get(key).ok_or(Error::NativeFailure)?
                else {
                    return Err(Error::NativeFailure);
                };
                let (indirect_size, NativeAllocation::Metal(indirect)) = self
                    .allocations
                    .get(&counter.packed())
                    .ok_or(Error::InvalidContext)?
                else {
                    return Err(Error::NativeFailure);
                };
                Ok(MetalFrameAction::Graphics(
                    ez_gfx_backend_metal::native::NativeGraphicsDraw {
                        pipeline,
                        depth_required: pipeline_layout.depth_required(),
                        texture_heap: self.prepared.texture_heaps[index],
                        state: *state,
                        index: self.index.ok_or(Error::NotReady)?,
                        index_size: self.index_size,
                        indirect,
                        indirect_size: *indirect_size,
                        draw_count: *draw_capacity,
                        bindings: &self.binding_sets[index],
                        textures: self.native_textures,
                    },
                ))
            }
            ExecutableNode::Compute { groups, .. } => {
                let key = self.prepared.keys[index]
                    .as_ref()
                    .ok_or(Error::InvalidArgument)?;
                let NativePipeline::Metal(pipeline) =
                    self.pipelines.get(key).ok_or(Error::NativeFailure)?
                else {
                    return Err(Error::NativeFailure);
                };
                Ok(MetalFrameAction::Compute(
                    ez_gfx_backend_metal::native::NativeComputeDispatch {
                        pipeline,
                        groups: *groups,
                        threads_per_group: self.prepared.workgroup_sizes[index]
                            .ok_or(Error::InvalidArgument)?,
                        bindings: &self.binding_sets[index],
                        texture_heap: self.prepared.texture_heaps[index],
                        textures: self.native_textures,
                    },
                ))
            }
            ExecutableNode::TextureReadback { texture } => {
                let (_, NativeTexture::Metal(texture), width, height, _) =
                    self.textures.get(texture).ok_or(Error::InvalidContext)?
                else {
                    return Err(Error::NativeFailure);
                };
                Ok(MetalFrameAction::TextureReadback {
                    texture,
                    width: *width,
                    height: *height,
                })
            }
            ExecutableNode::RenderTargetReadback { target } => {
                let record = self
                    .render_targets
                    .get(target)
                    .ok_or(Error::InvalidContext)?;
                let NativeTexture::Metal(texture) = &record.native else {
                    return Err(Error::NativeFailure);
                };
                Ok(MetalFrameAction::TextureReadback {
                    texture,
                    width: record.width,
                    height: record.height,
                })
            }
            ExecutableNode::Present { surface } => {
                if Some(*surface) != self.surface {
                    return Err(Error::InvalidArgument);
                }
                Ok(MetalFrameAction::Present)
            }
        }
    }
}

fn build_metal_actions<'a>(
    plan: &'a FrameExecutionPlan,
    inputs: &'a MetalActionInputs<'a>,
) -> Result<Vec<MetalFrameAction<'a>>> {
    let mut actions = Vec::with_capacity(plan.actions.len());
    for action in &plan.actions {
        match action {
            ExecutionAction::Wait(wait) => {
                if let Some(token) = wait.external {
                    actions.push(MetalFrameAction::Wait(token));
                }
            }
            ExecutionAction::Barrier(barrier) => {
                let resource = inputs
                    .frame_resources
                    .get(&ResourceId::from_index(barrier.resource))
                    .ok_or(Error::InvalidArgument)?;
                let resource = match *resource {
                    FrameNativeResource::Buffer(handle) => {
                        let NativeAllocation::Metal(allocation) = &inputs
                            .allocations
                            .get(&handle)
                            .ok_or(Error::InvalidContext)?
                            .1
                        else {
                            return Err(Error::NativeFailure);
                        };
                        MetalFrameResource::Buffer(allocation)
                    }
                    FrameNativeResource::Texture(handle) => {
                        let (_, NativeTexture::Metal(texture), _, _, _) =
                            inputs.textures.get(&handle).ok_or(Error::InvalidContext)?
                        else {
                            return Err(Error::NativeFailure);
                        };
                        MetalFrameResource::Texture(texture)
                    }
                    FrameNativeResource::Surface(_) => MetalFrameResource::Surface,
                    FrameNativeResource::Depth => MetalFrameResource::Depth,
                    FrameNativeResource::RenderTarget(handle) => {
                        let record = inputs
                            .render_targets
                            .get(&handle)
                            .ok_or(Error::InvalidContext)?;
                        let NativeTexture::Metal(texture) = &record.native else {
                            return Err(Error::NativeFailure);
                        };
                        MetalFrameResource::RenderTarget(texture)
                    }
                    FrameNativeResource::Index => {
                        MetalFrameResource::Buffer(inputs.index.ok_or(Error::NotReady)?)
                    }
                    FrameNativeResource::VertexHeap(heap_id) => {
                        let heap = inputs
                            .vertex_heaps
                            .values()
                            .find(|heap| heap.heap_id == Some(heap_id))
                            .ok_or(Error::InvalidContext)?;
                        let NativeAllocation::Metal(allocation) = &heap.allocation else {
                            return Err(Error::NativeFailure);
                        };
                        MetalFrameResource::Buffer(allocation)
                    }
                };
                actions.push(MetalFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                });
            }
            ExecutionAction::BeginPass(pass) => {
                // Resource indices resolve to surface or render-target
                // attachments here; textures, buffers, and depth images are
                // never color attachments. Surfaces keep the legacy clear.
                let mut colors = Vec::with_capacity(pass.colors.len());
                for index in &pass.colors {
                    let resource = inputs
                        .frame_resources
                        .get(&ResourceId::from_index(*index))
                        .ok_or(Error::InvalidArgument)?;
                    colors.push(match *resource {
                        FrameNativeResource::Surface(_) => MetalPassAttachment {
                            resource: MetalFrameResource::Surface,
                            clear: SURFACE_DEFAULT_CLEAR,
                        },
                        FrameNativeResource::RenderTarget(handle) => {
                            let record = inputs
                                .render_targets
                                .get(&handle)
                                .ok_or(Error::InvalidContext)?;
                            let NativeTexture::Metal(texture) = &record.native else {
                                return Err(Error::NativeFailure);
                            };
                            MetalPassAttachment {
                                resource: MetalFrameResource::RenderTarget(texture),
                                clear: super::super::render_target::render_target_clear_color(
                                    record,
                                ),
                            }
                        }
                        _ => return Err(Error::InvalidArgument),
                    });
                }
                actions.push(MetalFrameAction::BeginPass { pass, colors });
            }
            ExecutionAction::ExecuteNode(node) => actions.push(inputs.node(*node as usize)?),
            ExecutionAction::EndPass => actions.push(MetalFrameAction::EndPass),
        }
    }
    Ok(actions)
}

#[cfg(target_vendor = "apple")]
pub(super) fn execute_metal_frame_plan(
    context: &mut ContextState,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
) -> Result<()> {
    let surface_handle = payloads.iter().find_map(|payload| match payload {
        ExecutableNode::Present { surface } => Some(*surface),
        ExecutableNode::Graphics { .. }
        | ExecutableNode::Compute { .. }
        | ExecutableNode::TextureReadback { .. }
        | ExecutableNode::RenderTargetReadback { .. } => None,
    });
    let mut surface = surface_handle
        .map(|handle| {
            context
                .surfaces
                .remove(&handle)
                .ok_or(Error::InvalidContext)
        })
        .transpose()?;
    let presentation_mode = surface
        .as_ref()
        .map_or(PresentationMode::Fifo, |surface| surface.presentation_mode);
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
    let capture = should_capture_presented(
        surface
            .as_ref()
            .is_some_and(|surface| surface.state.snapshot_cache()),
        context.frame_capture_surface.is_some(),
    );

    let mut binding_sets = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let bindings = match payload {
            ExecutableNode::Graphics {
                layout, bindings, ..
            }
            | ExecutableNode::Compute {
                layout, bindings, ..
            } => metal_bindings(
                layout,
                bindings,
                &context.allocations,
                &context.vertex_heaps,
            )
            .map_err(map_hal)?,
            ExecutableNode::TextureReadback { .. }
            | ExecutableNode::RenderTargetReadback { .. }
            | ExecutableNode::Present { .. } => Vec::new(),
        };
        binding_sets.push(bindings);
    }
    let native_textures: Vec<_> = context
        .textures
        .iter()
        .filter(|(handle, _)| {
            context
                .texture_published_mips
                .get(handle)
                .is_some_and(|mips| *mips != 0)
        })
        .map(|(_, (_, texture, _, _, _))| match texture {
            NativeTexture::Metal(texture) => Ok(texture),
            NativeTexture::Vulkan(_) => Err(Error::NativeFailure),
        })
        .collect::<std::result::Result<_, _>>()?;
    let (index, index_size) = match context.index_heap.as_ref() {
        Some(heap) => match &heap.allocation {
            NativeAllocation::Metal(index) => (Some(index), heap.size),
            NativeAllocation::Vulkan(_) => return Err(Error::NativeFailure),
        },
        None => (None, 0),
    };
    let NativeContext::Metal(native) = &mut context.native else {
        return Err(Error::NativeFailure);
    };
    let prepared =
        prepare_metal_pipelines(&context.shaders, &mut context.pipelines, native, payloads)?;
    let inputs = MetalActionInputs {
        payloads,
        allocations: &context.allocations,
        vertex_heaps: &context.vertex_heaps,
        textures: &context.textures,
        render_targets: &context.render_targets,
        pipelines: &context.pipelines,
        frame_resources: &context.frame_native_resources,
        binding_sets: &binding_sets,
        native_textures: &native_textures,
        index,
        index_size,
        surface: surface_handle,
        prepared: &prepared,
    };
    let actions = build_metal_actions(plan, &inputs)?;
    let result = match surface.as_mut().map(|surface| &mut surface.native) {
        Some(NativeSurface::Metal(surface)) => native
            .execute_frame(
                Some((surface, extent, presentation_mode)),
                &actions,
                capture,
            )
            .map_err(map_hal),
        None => native.execute_frame(None, &actions, false).map_err(map_hal),
        Some(NativeSurface::Vulkan(_)) => Err(Error::NativeFailure),
    };
    let output_count_valid = match result.as_ref() {
        Err(_) => true,
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
            outputs.len() == texture_readbacks + usize::from(capture)
        }
    };
    if let Ok(outputs) = &result {
        context.last_readbacks.clone_from(outputs);
    }
    if let (Some(handle), Some(surface)) = (surface_handle, surface) {
        context.surfaces.insert(handle, surface);
    }
    result?;
    if !output_count_valid {
        return Err(Error::NativeFailure);
    }
    context.frame_presented = payloads
        .iter()
        .any(|payload| matches!(payload, ExecutableNode::Present { .. }));
    Ok(())
}
