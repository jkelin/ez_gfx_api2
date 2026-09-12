use crate::Result;
use ez_gfx_core::capability::PresentationMode;

use super::{
    Backend, ContextState, Error, ExecutableNode, ExecutionAction, ExecutionBarrier, ExecutionPass,
    FrameBindingSource, FrameBufferBindingRecord, FrameExecutionPlan, FrameNativeResource,
    GeometryAllocation, HashMap, MAX_PIPELINE_CACHE_ENTRIES, MeshPipelineKeyDesc, NativeAllocation,
    NativeContext, NativePipeline, NativeShader, NativeSurface, NativeTexture, NativeTextureMap,
    PackedHandle, PipelineKey, RenderTargetHandle, RenderTargetRecord, ResourceId,
    SURFACE_DEFAULT_CLEAR, ShaderHandle, ShaderRecord, SubmittedInfo, TextureHandle, map_hal,
    native_layouts,
    pipeline_layout_key, prepare_frame_binding_scratch, should_capture_presented,
};

use arrayvec::ArrayVec;

struct DxActionState<'a> {
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
    textures: &'a NativeTextureMap,
    submitted: &'a HashMap<TextureHandle, SubmittedInfo>,
    render_targets: &'a HashMap<RenderTargetHandle, RenderTargetRecord>,
    pipelines: &'a HashMap<PipelineKey, NativePipeline>,
    resources: &'a HashMap<ResourceId, FrameNativeResource>,
    index: Option<&'a ez_gfx_backend_dx12::native::NativeAllocation>,
    index_size: u64,
    extent: (u32, u32),
}
// `None` uses the swapchain format; non-color runtime formats fail before PSO creation.
const fn dx12_color_format(format: Option<ez_gfx_runtime::target::Format>) -> Result<u32> {
    use ez_gfx_runtime::target::Format;
    match format {
        None => Ok(29),
        Some(Format::Rgba8Unorm) => Ok(28),
        Some(Format::Bgra8Srgb) => Ok(91),
        Some(Format::Rgba16Float) => Ok(10),
        Some(_) => Err(Error::Unsupported),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the backend mesh seam keeps native state, cache storage, and immutable node inputs explicit"
)]
fn prepare_dx12_mesh_pipeline(
    native: &mut ez_gfx_backend_dx12::native::NativeContext,
    shaders: &HashMap<ShaderHandle, ShaderRecord>,
    pipelines: &HashMap<PipelineKey, NativePipeline>,
    key_slot: &mut Option<PipelineKey>,
    stages: &ez_gfx_hal::MeshStages<ShaderHandle>,
    layout: &ez_gfx_runtime::binding::ReflectedBindings,
    stage_layouts: &ez_gfx_hal::MeshStages<ez_gfx_runtime::binding::StageLayoutIdentity>,
    pipeline_layout: &ez_gfx_runtime::binding::PipelineLayout,
    state: ez_gfx_hal::MeshPipelineState,
    color_format: Option<ez_gfx_runtime::target::Format>,
) -> Result<Option<NativePipeline>> {
    let task = stages
        .task
        .map(|handle| shaders.get(&handle).ok_or(Error::InvalidContext))
        .transpose()?;
    let mesh = shaders.get(&stages.mesh).ok_or(Error::InvalidContext)?;
    let fragment = shaders.get(&stages.fragment).ok_or(Error::InvalidContext)?;
    let task_native = task
        .map(|record| match &record.native {
            NativeShader::Dx12(native) => Ok(native),
            NativeShader::Vulkan(_) => Err(Error::NativeFailure),
        })
        .transpose()?;
    let NativeShader::Dx12(mesh_native) = &mesh.native else {
        return Err(Error::NativeFailure);
    };
    let NativeShader::Dx12(fragment_native) = &fragment.native else {
        return Err(Error::NativeFailure);
    };

    let task_workgroup_size = task
        .map(|record| record.runtime.workgroup_size())
        .transpose()
        .map_err(|_| Error::InvalidArgument)?;
    let mesh_workgroup_size = mesh
        .runtime
        .workgroup_size()
        .map_err(|_| Error::InvalidArgument)?;
    let depth_required = pipeline_layout.depth_required();
    let key = PipelineKey::prepare_mesh_slot(
        key_slot,
        MeshPipelineKeyDesc {
            backend: Backend::Dx12,
            task_shader: stages.task,
            task_digest: task.map(|record| record.digest),
            task_entry: task.map(|record| record.entry.as_str()),
            mesh_shader: stages.mesh,
            mesh_digest: mesh.digest,
            mesh_entry: &mesh.entry,
            fragment_shader: stages.fragment,
            fragment_digest: fragment.digest,
            fragment_entry: &fragment.entry,
            stage_layouts,
            state,
            depth_required,
            color_format: dx12_color_format(color_format)?,
            depth_format: if depth_required { 40 } else { 0 },
            sample_count: 1,
        },
    );
    if pipelines.contains_key(key) {
        return Ok(None);
    }
    let layouts = native_layouts(layout).map_err(map_hal)?;
    let pipeline = NativePipeline::Dx12(
        native
            .create_mesh_pipeline(ez_gfx_backend_dx12::native::NativeMeshPipelineDesc {
                task: task_native.zip(task.map(|record| record.product)),
                mesh: (mesh_native, mesh.product),
                fragment: (fragment_native, fragment.product),
                state,
                color_format,
                layouts: &layouts,
                depth_required,
                task_workgroup_size,
                mesh_workgroup_size,
            })
            .map_err(map_hal)?,
    );
    Ok(Some(pipeline))
}

// Unsupported shader variants fail without inserting a partial pipeline-cache entry.
fn prepare_dx12_pipelines(
    native: &mut ez_gfx_backend_dx12::native::NativeContext,
    shaders: &HashMap<ShaderHandle, ShaderRecord>,
    pipelines: &mut HashMap<PipelineKey, NativePipeline>,
    payloads: &[ExecutableNode],
    pipeline_keys: &mut Vec<Option<PipelineKey>>,
    color_format: Option<ez_gfx_runtime::target::Format>,
) -> Result<()> {
    pipeline_keys.truncate(payloads.len());
    pipeline_keys.resize_with(payloads.len(), || None);
    for (node_index, payload) in payloads.iter().enumerate() {
        if let ExecutableNode::Mesh {
            stages,
            layout,
            stage_layouts,
            pipeline_layout,
            state,
            ..
        } = payload
        {
            let pipeline = prepare_dx12_mesh_pipeline(
                native,
                shaders,
                pipelines,
                &mut pipeline_keys[node_index],
                stages,
                layout,
                stage_layouts,
                pipeline_layout,
                *state,
                color_format,
            )?;
            if let Some(pipeline) = pipeline {
                if pipelines.len() == MAX_PIPELINE_CACHE_ENTRIES {
                    native.wait_idle().map_err(map_hal)?;
                    pipelines.clear();
                }
                let key = pipeline_keys[node_index]
                    .as_ref()
                    .ok_or(Error::InvalidArgument)?
                    .clone();
                pipelines.insert(key, pipeline);
            }
            continue;
        }
        let (key, pipeline) = match payload {
            ExecutableNode::Compute { shader, layout, .. } => {
                let record = shaders.get(shader).ok_or(Error::InvalidContext)?;
                let NativeShader::Dx12(native_shader) = &record.native else {
                    return Err(Error::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let key = PipelineKey::Compute {
                    backend: Backend::Dx12,
                    shader: *shader,
                    shader_digest: record.digest,
                    entry: record.entry.clone(),
                    layouts: pipeline_layout_key(&layouts),
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Dx12(
                        native
                            .create_compute_pipeline(native_shader, record.product, &layouts)
                            .map_err(map_hal)?,
                    ))
                };
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
                let vertex = shaders.get(vertex_shader).ok_or(Error::InvalidContext)?;
                let fragment = shaders.get(fragment_shader).ok_or(Error::InvalidContext)?;
                let NativeShader::Dx12(vertex_native) = &vertex.native else {
                    return Err(Error::NativeFailure);
                };
                let NativeShader::Dx12(fragment_native) = &fragment.native else {
                    return Err(Error::NativeFailure);
                };
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let depth_required = pipeline_layout.depth_required();
                let key = PipelineKey::Graphics {
                    backend: Backend::Dx12,
                    vertex_shader: *vertex_shader,
                    vertex_digest: vertex.digest,
                    vertex_entry: vertex.entry.clone(),
                    fragment_shader: *fragment_shader,
                    fragment_digest: fragment.digest,
                    fragment_entry: fragment.entry.clone(),
                    texture_heap: None,
                    layouts: pipeline_layout_key(&layouts),
                    state: *state,
                    depth_required,
                    color_format: dx12_color_format(color_format)?,
                    depth_format: if depth_required { 40 } else { 0 },
                    sample_count: 1,
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Dx12(
                        native
                            .create_graphics_pipeline(
                                vertex_native,
                                fragment_native,
                                vertex.product,
                                fragment.product,
                                *state,
                                depth_required,
                                color_format,
                                &layouts,
                            )
                            .map_err(map_hal)?,
                    ))
                };
                (key, pipeline)
            }
            ExecutableNode::Mesh { .. } => unreachable!("mesh payload handled above"),
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
    Ok(())
}

// Barrier resource indices resolve to the native buffer, texture, surface,
// depth, render-target, or index resource transitioned before submission.
fn dx12_barrier_resource<'resources>(
    state: &DxActionState<'resources>,
    barrier: &ExecutionBarrier,
) -> Result<ez_gfx_backend_dx12::native::NativeFrameResource<'resources>> {
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
            let NativeTexture::Dx12(texture) =
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
fn dx12_pass_colors<'resources>(
    state: &DxActionState<'resources>,
    pass: &ExecutionPass,
) -> Result<ArrayVec<ez_gfx_backend_dx12::native::PassAttachment<'resources>, 1>> {
    let mut colors = ArrayVec::new();
    for index in &pass.colors {
        let resource = state
            .resources
            .get(&ResourceId::from_index(*index))
            .ok_or(Error::InvalidArgument)?;
        colors
            .try_push(match *resource {
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
            })
            .map_err(|_| Error::InvalidArgument)?;
    }
    Ok(colors)
}

struct DxActionSource<'a, 'resources> {
    state: DxActionState<'resources>,
    plan: &'a FrameExecutionPlan,
    payloads: &'a [ExecutableNode],
    pipeline_keys: &'a [Option<PipelineKey>],
    shaders: &'resources HashMap<ShaderHandle, ShaderRecord>,
    action_indices: &'a [usize],
    binding_records: &'a [FrameBufferBindingRecord],
    binding_ranges: &'a [core::ops::Range<usize>],
}

impl DxActionSource<'_, '_> {
    fn binding_source(
        &self,
        node: usize,
    ) -> std::result::Result<FrameBindingSource<'_>, ez_gfx_hal::HalError> {
        let range = self
            .binding_ranges
            .get(node)
            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
            .clone();
        Ok(FrameBindingSource {
            records: self
                .binding_records
                .get(range)
                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
            allocations: self.state.allocations,
            vertex_heaps: self.state.vertex_heaps,
        })
    }

    fn mesh_dispatch<'draw>(
        &'draw self,
        node: usize,
        stages: &ez_gfx_hal::MeshStages<ShaderHandle>,
        groups: [u32; 3],
        bindings: &'draw FrameBindingSource<'_>,
    ) -> std::result::Result<
        ez_gfx_backend_dx12::native::NativeMeshDispatch<'draw>,
        ez_gfx_hal::HalError,
    > {
        let key = self.pipeline_keys[node]
            .as_ref()
            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
        let NativePipeline::Dx12(pipeline) = self
            .state
            .pipelines
            .get(key)
            .ok_or(ez_gfx_hal::HalError::NativeFailure)?
        else {
            return Err(ez_gfx_hal::HalError::NativeFailure);
        };
        let mesh_record = self
            .shaders
            .get(&stages.mesh)
            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
        let task_workgroup_size = stages
            .task
            .map(|handle| {
                self.shaders
                    .get(&handle)
                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                    .runtime
                    .workgroup_size()
                    .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)
            })
            .transpose()?;
        Ok(ez_gfx_backend_dx12::native::NativeMeshDispatch {
            pipeline,
            groups,
            has_task: stages.task.is_some(),
            mesh_workgroup_size: mesh_record
                .runtime
                .workgroup_size()
                .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?,
            task_workgroup_size,
            bindings,
        })
    }
}

impl ez_gfx_backend_dx12::native::NativeFrameActionSource for DxActionSource<'_, '_> {
    fn len(&self) -> usize {
        self.action_indices.len()
    }

    fn with_action(
        &self,
        index: usize,
        visitor: &mut dyn FnMut(
            &ez_gfx_backend_dx12::native::NativeFrameAction<'_>,
        ) -> std::result::Result<(), ez_gfx_hal::HalError>,
    ) -> std::result::Result<(), ez_gfx_hal::HalError> {
        let action = self
            .plan
            .actions
            .get(
                *self
                    .action_indices
                    .get(index)
                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
            )
            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
        let binding_source;
        let native = match action {
            ExecutionAction::Wait(wait) => ez_gfx_backend_dx12::native::NativeFrameAction::Wait(
                wait.external.ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
            ),
            ExecutionAction::Barrier(barrier) => {
                let resource = dx12_barrier_resource(&self.state, barrier)
                    .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?;
                ez_gfx_backend_dx12::native::NativeFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                }
            }
            ExecutionAction::BeginPass(pass) => {
                let colors = dx12_pass_colors(&self.state, pass)
                    .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?;
                ez_gfx_backend_dx12::native::NativeFrameAction::BeginPass { pass, colors }
            }
            ExecutionAction::ExecuteNode(node) => {
                let node = *node as usize;
                match self
                    .payloads
                    .get(node)
                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                {
                    ExecutableNode::Compute { groups, .. } => {
                        binding_source = self.binding_source(node)?;
                        let key = self.pipeline_keys[node]
                            .as_ref()
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativePipeline::Dx12(pipeline) = self
                            .state
                            .pipelines
                            .get(key)
                            .ok_or(ez_gfx_hal::HalError::NativeFailure)?
                        else {
                            return Err(ez_gfx_hal::HalError::NativeFailure);
                        };
                        ez_gfx_backend_dx12::native::NativeFrameAction::Compute(
                            ez_gfx_backend_dx12::native::NativeComputeDispatch {
                                pipeline,
                                groups: *groups,
                                bindings: &binding_source,
                            },
                        )
                    }
                    ExecutableNode::Graphics {
                        counter,
                        draw_capacity,
                        ..
                    } => {
                        binding_source = self.binding_source(node)?;
                        let (indirect_size, NativeAllocation::Dx12(indirect)) = self
                            .state
                            .allocations
                            .get(&counter.packed())
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        let key = self.pipeline_keys[node]
                            .as_ref()
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativePipeline::Dx12(pipeline) = self
                            .state
                            .pipelines
                            .get(key)
                            .ok_or(ez_gfx_hal::HalError::NativeFailure)?
                        else {
                            return Err(ez_gfx_hal::HalError::NativeFailure);
                        };
                        ez_gfx_backend_dx12::native::NativeFrameAction::Graphics(
                            ez_gfx_backend_dx12::native::NativeDrawIndexed {
                                width: self.state.extent.0,
                                height: self.state.extent.1,
                                pipeline,
                                index_buffer: self
                                    .state
                                    .index
                                    .ok_or(ez_gfx_hal::HalError::NotReady)?,
                                index_size: self.state.index_size,
                                indirect_buffer: indirect,
                                indirect_size: *indirect_size,
                                draw_count: *draw_capacity,
                                bindings: &binding_source,
                            },
                        )
                    }
                    ExecutableNode::Mesh { stages, groups, .. } => {
                        binding_source = self.binding_source(node)?;
                        ez_gfx_backend_dx12::native::NativeFrameAction::Mesh(self.mesh_dispatch(
                            node,
                            stages,
                            *groups,
                            &binding_source,
                        )?)
                    }
                    ExecutableNode::TextureReadback { texture } => {
                        let info = self
                            .state
                            .submitted
                            .get(texture)
                            .copied()
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativeTexture::Dx12(texture) = self
                            .state
                            .textures
                            .get(texture)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        ez_gfx_backend_dx12::native::NativeFrameAction::TextureReadback {
                            texture,
                            width: info.width,
                            height: info.height,
                        }
                    }
                    ExecutableNode::RenderTargetReadback { target } => {
                        let record = self
                            .state
                            .render_targets
                            .get(target)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativeTexture::Dx12(texture) = &record.native else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        ez_gfx_backend_dx12::native::NativeFrameAction::TextureReadback {
                            texture,
                            width: record.width,
                            height: record.height,
                        }
                    }
                    ExecutableNode::Present { .. } => {
                        ez_gfx_backend_dx12::native::NativeFrameAction::Present
                    }
                }
            }
            ExecutionAction::EndPass => ez_gfx_backend_dx12::native::NativeFrameAction::EndPass,
        };
        visitor(&native)
    }
}

#[cfg(all(test, windows))]
fn validate_raw_dx12_shader(
    context: &ContextState,
    handle: ShaderHandle,
    stage: ez_gfx_artifact::Stage,
) -> Result<()> {
    let shader = context
        .shaders
        .get(&handle)
        .filter(|shader| shader.stage == stage)
        .ok_or(Error::NativeFailure)?;
    if !matches!(shader.native, NativeShader::Dx12(_)) {
        return Err(Error::NativeFailure);
    }
    Ok(())
}

#[cfg(all(test, windows))]
fn execute_raw_native_frame_test_probe(
    context: &mut ContextState,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
) -> Option<Result<()>> {
    if !context.raw_native_frame_test_probe.enabled {
        return None;
    }

    let observed = (|| {
        let mut mesh_executions = 0;
        let mut graphics_executions = 0;
        // Only executable plan actions count: unreferenced payloads are not
        // executions, and an invalid node index fails closed.
        for action in &plan.actions {
            let ExecutionAction::ExecuteNode(node) = action else {
                continue;
            };
            let node = usize::try_from(*node).map_err(|_| Error::NativeFailure)?;
            match payloads.get(node).ok_or(Error::NativeFailure)? {
                ExecutableNode::Mesh { stages, .. } => {
                    if let Some(task) = stages.task {
                        validate_raw_dx12_shader(context, task, ez_gfx_artifact::Stage::Task)?;
                    }
                    validate_raw_dx12_shader(context, stages.mesh, ez_gfx_artifact::Stage::Mesh)?;
                    validate_raw_dx12_shader(
                        context,
                        stages.fragment,
                        ez_gfx_artifact::Stage::Fragment,
                    )?;
                    mesh_executions += 1;
                }
                ExecutableNode::Graphics {
                    vertex_shader,
                    fragment_shader,
                    ..
                } => {
                    validate_raw_dx12_shader(
                        context,
                        *vertex_shader,
                        ez_gfx_artifact::Stage::Vertex,
                    )?;
                    validate_raw_dx12_shader(
                        context,
                        *fragment_shader,
                        ez_gfx_artifact::Stage::Fragment,
                    )?;
                    graphics_executions += 1;
                }
                _ => {}
            }
        }
        context.raw_native_frame_test_probe.submits += 1;
        context.raw_native_frame_test_probe.mesh_executions += mesh_executions;
        context.raw_native_frame_test_probe.graphics_executions += graphics_executions;
        Ok(())
    })();
    Some(observed)
}

#[allow(
    clippy::too_many_lines,
    reason = "linear native lowering plus a test-only raw-plan observation gate"
)]
#[cfg(windows)]
pub(super) fn execute_dx12_frame_plan(
    context: &mut ContextState,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
    binding_resources: &[ez_gfx_runtime::binding::ResourceIdentity],
) -> Result<()> {
    #[cfg(test)]
    if let Some(observed) = execute_raw_native_frame_test_probe(context, plan, payloads) {
        return observed;
    }

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
    let presentation_mode = surface
        .as_ref()
        .map_or(PresentationMode::Fifo, |surface| surface.presentation_mode);
    let extent = super::frame_target_extent(context, surface.as_ref());
    let capture = should_capture_presented(
        surface
            .as_ref()
            .is_some_and(|surface| surface.state.snapshot_cache()),
        context.frame_capture_surface.is_some(),
    );
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
    context.frame_action_indices.clear();
    context
        .frame_action_indices
        .extend(
            plan.actions
                .iter()
                .enumerate()
                .filter_map(|(index, action)| {
                    (!matches!(action, ExecutionAction::Wait(wait) if wait.external.is_none()))
                        .then_some(index)
                }),
        );
    let color_format = context
        .frame_render_target
        .and_then(|target| context.render_targets.get(&target))
        .map(|record| record.format);
    {
        let NativeContext::Dx12(native) = &mut context.native else {
            return Err(Error::NativeFailure);
        };
        prepare_dx12_pipelines(
            native,
            &context.shaders,
            &mut context.pipelines,
            payloads,
            &mut context.frame_pipeline_keys,
            color_format,
        )?;
    }
    prepare_frame_binding_scratch(
        payloads,
        binding_resources,
        &context.allocations,
        &context.vertex_heaps,
        &mut context.frame_binding_scratch,
        &mut context.frame_binding_ranges,
    )
    .map_err(map_hal)?;
    super::account_frame_lowering_scratch(
        &mut context.frame,
        &mut context.frame_pipeline_keys,
        &mut context.frame_action_indices,
        &mut context.frame_binding_scratch,
        &mut context.frame_binding_ranges,
        &mut context.frame_texture_handles,
    )?;
    let source = DxActionSource {
        state: DxActionState {
            allocations: &context.allocations,
            vertex_heaps: &context.vertex_heaps,
            textures: &context.textures,
            submitted: context.texture_pipeline.submitted(),
            render_targets: &context.render_targets,
            pipelines: &context.pipelines,
            resources: &context.frame_native_resources,
            index,
            index_size,
            extent,
        },
        plan,
        payloads,
        pipeline_keys: &context.frame_pipeline_keys,
        shaders: &context.shaders,
        action_indices: &context.frame_action_indices,
        binding_records: &context.frame_binding_scratch,
        binding_ranges: &context.frame_binding_ranges,
    };
    let execution = {
        let NativeContext::Dx12(native) = &mut context.native else {
            return Err(Error::NativeFailure);
        };
        native
            .execute_frame_source(
                native_surface
                    .as_deref_mut()
                    .map(|surface| (surface, extent, presentation_mode)),
                &source,
                capture,
            )
            .map_err(map_hal)
    };
    let outcome = match execution {
        Ok(outputs) => {
            let expected = super::expected_frame_output_count(payloads, capture);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ez_gfx_runtime::target::Format;

    #[test]
    fn target_and_surface_formats_keep_distinct_pipeline_keys() {
        let surface = dx12_color_format(None).unwrap();
        let rgba = dx12_color_format(Some(Format::Rgba8Unorm)).unwrap();
        let bgra_srgb = dx12_color_format(Some(Format::Bgra8Srgb)).unwrap();

        assert_eq!((surface, rgba, bgra_srgb), (29, 28, 91));
        assert_ne!(surface, rgba);
        assert_ne!(surface, bgra_srgb);
    }
}
