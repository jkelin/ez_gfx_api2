use crate::Result;

use super::{
    Backend, ContextState, Error, ExecutableNode, ExecutionAction, ExecutionBarrier, ExecutionPass,
    FrameExecutionPlan, FrameNativeResource, GeometryAllocation, HashMap,
    MAX_PIPELINE_CACHE_ENTRIES, NativeAllocation, NativeContext, NativePipeline, NativeShader,
    NativeSurface, NativeTexture, NativeTextureMap, PackedHandle, PipelineKey, RenderTargetHandle,
    RenderTargetRecord, ResourceId, SURFACE_DEFAULT_CLEAR, ShaderHandle, ShaderRecord, map_hal,
    native_layouts, pipeline_layout_key, vulkan_bindings,
};

struct VulkanActionState<'a> {
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
    textures: &'a NativeTextureMap,
    render_targets: &'a HashMap<RenderTargetHandle, RenderTargetRecord>,
    resources: &'a HashMap<ResourceId, FrameNativeResource>,
    pipelines: &'a HashMap<PipelineKey, NativePipeline>,
    index: Option<&'a ez_gfx_backend_vulkan::NativeAllocation>,
    extent: (u32, u32),
}

// On multi-backend targets, variant projections reject mismatched state before native calls.
trait VulkanRef {
    type Native;

    fn vulkan(&self) -> Result<&Self::Native>;
}

trait VulkanMut {
    type Native;

    fn vulkan_mut(&mut self) -> Result<&mut Self::Native>;
}

trait IntoVulkan {
    type Native;

    fn into_vulkan(self) -> Result<Self::Native>;
}

macro_rules! impl_vulkan_ref {
    ($wrapper:ty, $native:ty, $variant:path) => {
        impl VulkanRef for $wrapper {
            type Native = $native;

            fn vulkan(&self) -> Result<&Self::Native> {
                match self {
                    $variant(native) => Ok(native),
                    #[cfg(any(windows, target_vendor = "apple"))]
                    _ => Err(Error::NativeFailure),
                }
            }
        }
    };
}

impl_vulkan_ref!(
    NativeAllocation,
    ez_gfx_backend_vulkan::NativeAllocation,
    NativeAllocation::Vulkan
);
impl_vulkan_ref!(
    NativePipeline,
    ez_gfx_backend_vulkan::NativePipeline,
    NativePipeline::Vulkan
);
impl_vulkan_ref!(
    NativeShader,
    ez_gfx_backend_vulkan::NativeShader,
    NativeShader::Vulkan
);
impl_vulkan_ref!(
    NativeTexture,
    ez_gfx_backend_vulkan::NativeTexture,
    NativeTexture::Vulkan
);

impl VulkanMut for NativeContext {
    type Native = ez_gfx_backend_vulkan::NativeContext;

    fn vulkan_mut(&mut self) -> Result<&mut Self::Native> {
        match self {
            Self::Vulkan(native) => Ok(native),
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(Error::NativeFailure),
        }
    }
}

impl VulkanMut for NativeSurface {
    type Native = ez_gfx_backend_vulkan::NativeSurface;

    fn vulkan_mut(&mut self) -> Result<&mut Self::Native> {
        match self {
            Self::Vulkan(native) => Ok(native),
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(Error::NativeFailure),
        }
    }
}

impl IntoVulkan for NativePipeline {
    type Native = ez_gfx_backend_vulkan::NativePipeline;

    fn into_vulkan(self) -> Result<Self::Native> {
        match self {
            Self::Vulkan(native) => Ok(native),
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(Error::NativeFailure),
        }
    }
}

// A headless frame leaves the cached format and graphics pipelines unchanged.
fn prepare_vulkan_surface(
    native: &mut ez_gfx_backend_vulkan::NativeContext,
    pipelines: &mut HashMap<PipelineKey, NativePipeline>,
    graphics_format: &mut Option<u32>,
    surface: Option<&ez_gfx_backend_vulkan::NativeSurface>,
    extent: (u32, u32),
) -> Result<()> {
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
                native.destroy_pipeline(pipeline.into_vulkan()?);
            }
        }
        *graphics_format = Some(format);
    }
    Ok(())
}

// Unsupported shader variants fail without inserting a partial pipeline-cache entry.
fn prepare_vulkan_pipelines(
    native: &mut ez_gfx_backend_vulkan::NativeContext,
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
                let native_shader = record.native.vulkan()?;
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
                let record = shaders.get(shader).ok_or(Error::InvalidContext)?;
                let graphics = record.graphics.as_ref().ok_or(Error::InvalidArgument)?;
                let native_shader = record.native.vulkan()?;
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
            ExecutableNode::TextureReadback { .. }
            | ExecutableNode::RenderTargetReadback { .. }
            | ExecutableNode::Present { .. } => continue,
        };
        if let Some(pipeline) = pipeline {
            if pipelines.len() == MAX_PIPELINE_CACHE_ENTRIES {
                native.wait_idle().map_err(map_hal)?;
                let stale = pipelines
                    .drain()
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>();
                for stale_pipeline in stale {
                    native.destroy_pipeline(stale_pipeline.into_vulkan()?);
                }
            }
            pipelines.insert(key.clone(), pipeline);
        }
        pipeline_keys[node_index] = Some(key);
    }
    Ok(pipeline_keys)
}

// Barrier resource indices resolve to the native buffer, texture, surface,
// render-target, depth, or index resource transitioned before submission.
fn vulkan_barrier_resource<'a>(
    state: &'a VulkanActionState<'a>,
    barrier: &ExecutionBarrier,
) -> Result<ez_gfx_backend_vulkan::NativeFrameResource<'a>> {
    let resource = state
        .resources
        .get(&ResourceId::from_index(barrier.resource))
        .ok_or(Error::InvalidArgument)?;
    Ok(match *resource {
        FrameNativeResource::Buffer(handle) => {
            let allocation = state
                .allocations
                .get(&handle)
                .ok_or(Error::InvalidContext)?
                .1
                .vulkan()?;
            ez_gfx_backend_vulkan::NativeFrameResource::Buffer(allocation)
        }
        FrameNativeResource::Texture(handle) => {
            let (_, texture, _, _, _) = state.textures.get(&handle).ok_or(Error::InvalidContext)?;
            ez_gfx_backend_vulkan::NativeFrameResource::Texture(texture.vulkan()?)
        }
        FrameNativeResource::Surface(_) => ez_gfx_backend_vulkan::NativeFrameResource::Surface,
        FrameNativeResource::RenderTarget(handle) => {
            let record = state
                .render_targets
                .get(&handle)
                .ok_or(Error::InvalidContext)?;
            ez_gfx_backend_vulkan::NativeFrameResource::RenderTarget(record.native.vulkan()?)
        }
        FrameNativeResource::Depth => ez_gfx_backend_vulkan::NativeFrameResource::Depth,
        FrameNativeResource::Index => {
            ez_gfx_backend_vulkan::NativeFrameResource::Buffer(state.index.ok_or(Error::NotReady)?)
        }
        FrameNativeResource::VertexHeap(heap_id) => {
            let allocation = state
                .vertex_heaps
                .values()
                .find(|heap| heap.heap_id == Some(heap_id))
                .ok_or(Error::InvalidContext)?;
            ez_gfx_backend_vulkan::NativeFrameResource::Buffer(allocation.allocation.vulkan()?)
        }
    })
}

// Pass color indices resolve to surface or render-target attachments here;
// textures, buffers, and depth images are never color attachments. Surfaces
// keep the legacy clear.
fn vulkan_pass_colors<'a>(
    state: &'a VulkanActionState<'a>,
    pass: &ExecutionPass,
) -> Result<Vec<ez_gfx_backend_vulkan::PassAttachment<'a>>> {
    let mut colors = Vec::with_capacity(pass.colors.len());
    for index in &pass.colors {
        let resource = state
            .resources
            .get(&ResourceId::from_index(*index))
            .ok_or(Error::InvalidArgument)?;
        colors.push(match *resource {
            FrameNativeResource::Surface(_) => ez_gfx_backend_vulkan::PassAttachment {
                resource: ez_gfx_backend_vulkan::NativeFrameResource::Surface,
                clear: SURFACE_DEFAULT_CLEAR,
            },
            FrameNativeResource::RenderTarget(handle) => {
                let record = state
                    .render_targets
                    .get(&handle)
                    .ok_or(Error::InvalidContext)?;
                ez_gfx_backend_vulkan::PassAttachment {
                    resource: ez_gfx_backend_vulkan::NativeFrameResource::RenderTarget(
                        record.native.vulkan()?,
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
fn vulkan_actions<'a>(
    state: &'a VulkanActionState<'a>,
    plan: &'a FrameExecutionPlan,
    payloads: &'a [ExecutableNode],
    binding_sets: &'a [Vec<ez_gfx_backend_vulkan::NativeBufferBinding<'a>>],
    pipeline_keys: &[Option<PipelineKey>],
) -> Result<Vec<ez_gfx_backend_vulkan::NativeFrameAction<'a>>> {
    let mut actions = Vec::with_capacity(plan.actions.len());
    for action in &plan.actions {
        match action {
            ExecutionAction::Wait(wait) => {
                if let Some(token) = wait.external {
                    actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Wait(token));
                }
            }
            ExecutionAction::Barrier(barrier) => {
                let resource = vulkan_barrier_resource(state, barrier)?;
                actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                });
            }
            ExecutionAction::BeginPass(pass) => {
                let colors = vulkan_pass_colors(state, pass)?;
                actions.push(ez_gfx_backend_vulkan::NativeFrameAction::BeginPass { pass, colors });
            }
            ExecutionAction::ExecuteNode(node) => {
                let index_node = *node as usize;
                let payload = payloads.get(index_node).ok_or(Error::InvalidArgument)?;
                match payload {
                    ExecutableNode::Compute { groups, .. } => {
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(Error::InvalidArgument)?;
                        let pipeline = state
                            .pipelines
                            .get(key)
                            .ok_or(Error::NativeFailure)?
                            .vulkan()?;
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Compute(
                            ez_gfx_backend_vulkan::NativeComputeDispatch {
                                pipeline,
                                groups: *groups,
                                bindings: &binding_sets[index_node],
                            },
                        ));
                    }
                    ExecutableNode::Graphics {
                        counter,
                        draw_capacity,
                        ..
                    } => {
                        let indirect = state
                            .allocations
                            .get(&counter.packed())
                            .ok_or(Error::InvalidContext)?
                            .1
                            .vulkan()?;
                        let key = pipeline_keys[index_node]
                            .as_ref()
                            .ok_or(Error::InvalidArgument)?;
                        let pipeline = state
                            .pipelines
                            .get(key)
                            .ok_or(Error::NativeFailure)?
                            .vulkan()?;
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::Graphics(
                            ez_gfx_backend_vulkan::NativeDrawIndexed {
                                width: state.extent.0,
                                height: state.extent.1,
                                pipeline,
                                index_buffer: state.index.ok_or(Error::NotReady)?,
                                indirect_buffer: indirect,
                                draw_count: *draw_capacity,
                                bindings: &binding_sets[index_node],
                            },
                        ));
                    }
                    ExecutableNode::TextureReadback { texture } => {
                        let (_, texture, width, height, _) =
                            state.textures.get(texture).ok_or(Error::InvalidContext)?;
                        let texture = texture.vulkan()?;
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::TextureReadback {
                            texture,
                            width: *width,
                            height: *height,
                        });
                    }
                    ExecutableNode::RenderTargetReadback { target } => {
                        let record = state
                            .render_targets
                            .get(target)
                            .ok_or(Error::InvalidContext)?;
                        actions.push(ez_gfx_backend_vulkan::NativeFrameAction::TextureReadback {
                            texture: record.native.vulkan()?,
                            width: record.width,
                            height: record.height,
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

    let index = context
        .index_heap
        .as_ref()
        .map(|heap| heap.allocation.vulkan())
        .transpose()?;
    if surface
        .as_ref()
        .is_some_and(|surface| !matches!(surface.native, NativeSurface::Vulkan(_)))
    {
        if let (Some(handle), Some(surface)) = (surface_handle, surface) {
            context.surfaces.insert(handle, surface);
        }
        return Err(Error::NativeFailure);
    }
    if surface.as_ref().is_some_and(
        |surface| matches!(&surface.native, NativeSurface::Vulkan(native) if native.is_headless()),
    ) {
        if let (Some(handle), Some(surface)) = (surface_handle, surface) {
            context.surfaces.insert(handle, surface);
        }
        return Err(Error::Unsupported);
    }
    let mut native_surface = surface
        .as_mut()
        .map(|surface| surface.native.vulkan_mut())
        .transpose()?;
    let pipeline_keys = {
        let native = context.native.vulkan_mut()?;
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
            } => vulkan_bindings(
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
    let state = VulkanActionState {
        allocations: &context.allocations,
        vertex_heaps: &context.vertex_heaps,
        textures: &context.textures,
        render_targets: &context.render_targets,
        pipelines: &context.pipelines,
        resources: &context.frame_native_resources,
        index,
        extent,
    };
    let actions = vulkan_actions(&state, plan, payloads, &binding_sets, &pipeline_keys)?;
    let execution = {
        let native = context.native.vulkan_mut()?;
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
            let expected = texture_readbacks + usize::from(capture);
            if outputs.len() != expected {
                if let (Some(handle), Some(surface)) = (surface_handle, surface) {
                    context.surfaces.insert(handle, surface);
                }
                return Err(Error::NativeFailure);
            }
            context.last_readbacks = outputs;
            if capture && let Some(native_surface) = native_surface.as_deref() {
                native_surface.presented_rgba8().clone_into(
                    context
                        .last_readbacks
                        .last_mut()
                        .expect("capture output exists"),
                );
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
