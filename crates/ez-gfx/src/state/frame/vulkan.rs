use crate::{Result, state::SurfaceRecord};
use ez_gfx_core::capability::PresentationMode;

use super::{
    Backend, ContextState, Error, ExecutableNode, ExecutionAction, ExecutionBarrier, ExecutionPass,
    FrameBindingSource, FrameBufferBindingRecord, FrameExecutionPlan, FrameNativeResource,
    GeometryAllocation, HashMap, MAX_PIPELINE_CACHE_ENTRIES, MeshPipelineKeyDesc, NativeAllocation,
    NativeContext, NativePipeline, NativeShader, NativeSurface, NativeTexture, NativeTextureMap,
    PackedHandle, PipelineKey, RenderTargetHandle, RenderTargetRecord, ResourceId,
    SURFACE_DEFAULT_CLEAR, ShaderHandle, ShaderRecord, SubmittedInfo, TextureHandle, map_hal,
    native_layouts, pipeline_layout_key, prepare_frame_binding_scratch, should_capture_presented,
};

use arrayvec::ArrayVec;

struct VulkanActionState<'a> {
    allocations: &'a HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'a HashMap<String, GeometryAllocation>,
    textures: &'a NativeTextureMap,
    submitted: &'a HashMap<TextureHandle, SubmittedInfo>,
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
    presentation_mode: PresentationMode,
) -> Result<()> {
    if let Some(surface) = surface {
        native
            .prepare_surface(surface, extent.0, extent.1, presentation_mode)
            .map_err(map_hal)?;
        let format = native.graphics_format_key(None).map_err(map_hal)?;
        if graphics_format.is_some_and(|cached| cached != format) {
            // Render-pass formats are baked into both indexed and mesh pipeline objects.
            let stale = pipelines
                .extract_if(|key, _| key.is_render())
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

#[allow(
    clippy::too_many_arguments,
    reason = "the backend mesh seam keeps native state, cache storage, and immutable node inputs explicit"
)]
fn prepare_vulkan_mesh_pipeline(
    native: &mut ez_gfx_backend_vulkan::NativeContext,
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
    let task_native = task.map(|record| record.native.vulkan()).transpose()?;
    let mesh_native = mesh.native.vulkan()?;
    let fragment_native = fragment.native.vulkan()?;

    let task_workgroup_size = task
        .map(|record| record.runtime.workgroup_size())
        .transpose()
        .map_err(|_| Error::InvalidArgument)?;
    let mesh_workgroup_size = mesh
        .runtime
        .workgroup_size()
        .map_err(|_| Error::InvalidArgument)?;
    let depth_required = pipeline_layout.depth_required();
    let native_color_format = native.graphics_format_key(color_format).map_err(map_hal)?;
    let key = PipelineKey::prepare_mesh_slot(
        key_slot,
        MeshPipelineKeyDesc {
            backend: Backend::Vulkan,
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
            color_format: native_color_format,
            depth_format: u32::from(depth_required),
            sample_count: 1,
        },
    );
    if pipelines.contains_key(key) {
        return Ok(None);
    }
    let layouts = native_layouts(layout).map_err(map_hal)?;
    let pipeline = NativePipeline::Vulkan(
        native
            .create_mesh_pipeline(ez_gfx_backend_vulkan::NativeMeshPipelineDesc {
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
fn prepare_vulkan_pipelines(
    native: &mut ez_gfx_backend_vulkan::NativeContext,
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
            let pipeline = prepare_vulkan_mesh_pipeline(
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
                    let stale = pipelines
                        .drain()
                        .map(|(_, value)| value)
                        .collect::<Vec<_>>();
                    for stale_pipeline in stale {
                        native.destroy_pipeline(stale_pipeline.into_vulkan()?);
                    }
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
                let native_shader = record.native.vulkan()?;
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let key = PipelineKey::Compute {
                    backend: Backend::Vulkan,
                    shader: *shader,
                    shader_digest: record.digest,
                    entry: record.entry.clone(),
                    layouts: pipeline_layout_key(&layouts),
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Vulkan(
                        native
                            .create_compute_pipeline(
                                native_shader,
                                record.product,
                                &record.entry,
                                &layouts,
                            )
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
                let vertex_native = vertex.native.vulkan()?;
                let fragment_native = fragment.native.vulkan()?;
                let layouts = native_layouts(layout).map_err(map_hal)?;
                let depth_required = pipeline_layout.depth_required();
                let native_color_format =
                    native.graphics_format_key(color_format).map_err(map_hal)?;
                let key = PipelineKey::Graphics {
                    backend: Backend::Vulkan,
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
                    color_format: native_color_format,
                    depth_format: u32::from(depth_required),
                    sample_count: 1,
                };
                let pipeline = if pipelines.contains_key(&key) {
                    None
                } else {
                    Some(NativePipeline::Vulkan(
                        native
                            .create_graphics_pipeline(
                                vertex_native,
                                fragment_native,
                                ez_gfx_backend_vulkan::NativeGraphicsPipelineDesc {
                                    vertex_index: vertex.product,
                                    fragment_index: fragment.product,
                                    state: *state,
                                    color_format,
                                    depth_required,
                                    layouts: &layouts,
                                },
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
    Ok(())
}

// Barrier resource indices resolve to the native buffer, texture, surface,
// render-target, depth, or index resource transitioned before submission.
fn vulkan_barrier_resource<'resources>(
    state: &VulkanActionState<'resources>,
    barrier: &ExecutionBarrier,
) -> Result<ez_gfx_backend_vulkan::NativeFrameResource<'resources>> {
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
            let texture = state.textures.get(&handle).ok_or(Error::InvalidContext)?;
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
fn vulkan_pass_colors<'resources>(
    state: &VulkanActionState<'resources>,
    pass: &ExecutionPass,
) -> Result<ArrayVec<ez_gfx_backend_vulkan::PassAttachment<'resources>, 1>> {
    let mut colors = ArrayVec::new();
    for index in &pass.colors {
        let resource = state
            .resources
            .get(&ResourceId::from_index(*index))
            .ok_or(Error::InvalidArgument)?;
        colors
            .try_push(match *resource {
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
            })
            .map_err(|_| Error::InvalidArgument)?;
    }
    Ok(colors)
}

// Rebuilds one borrowed native view at a time from retained plan records.
// No reference survives its visitor call.
struct VulkanActionSource<'a, 'resources> {
    state: VulkanActionState<'resources>,
    plan: &'a FrameExecutionPlan,
    payloads: &'a [ExecutableNode],
    pipeline_keys: &'a [Option<PipelineKey>],
    shaders: &'resources HashMap<ShaderHandle, ShaderRecord>,
    binding_records: &'a [FrameBufferBindingRecord],
    binding_ranges: &'a [core::ops::Range<usize>],
}

impl VulkanActionSource<'_, '_> {
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

    fn mesh_draw<'draw>(
        &'draw self,
        node: usize,
        stages: &ez_gfx_hal::MeshStages<ShaderHandle>,
        groups: [u32; 3],
        bindings: &'draw FrameBindingSource<'_>,
    ) -> std::result::Result<ez_gfx_backend_vulkan::NativeMeshDraw<'draw>, ez_gfx_hal::HalError>
    {
        let key = self.pipeline_keys[node]
            .as_ref()
            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
        let pipeline = self
            .state
            .pipelines
            .get(key)
            .ok_or(ez_gfx_hal::HalError::NativeFailure)?
            .vulkan()
            .map_err(|_| ez_gfx_hal::HalError::NativeFailure)?;
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
        Ok(ez_gfx_backend_vulkan::NativeMeshDraw {
            width: self.state.extent.0,
            height: self.state.extent.1,
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

impl ez_gfx_backend_vulkan::NativeFrameActionSource for VulkanActionSource<'_, '_> {
    fn len(&self) -> usize {
        self.plan
            .actions
            .iter()
            .filter(
                |action| !matches!(action, ExecutionAction::Wait(wait) if wait.external.is_none()),
            )
            .count()
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(
            usize,
            &ez_gfx_backend_vulkan::NativeFrameAction<'_>,
        ) -> std::result::Result<(), ez_gfx_hal::HalError>,
    ) -> std::result::Result<(), ez_gfx_hal::HalError> {
        let mut output_index = 0_usize;
        for action in &self.plan.actions {
            let binding_source;
            let native = match action {
                ExecutionAction::Wait(wait) => {
                    let Some(token) = wait.external else {
                        continue;
                    };
                    ez_gfx_backend_vulkan::NativeFrameAction::Wait(token)
                }
                ExecutionAction::Barrier(barrier) => {
                    let resource = vulkan_barrier_resource(&self.state, barrier)
                        .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?;
                    ez_gfx_backend_vulkan::NativeFrameAction::Barrier {
                        barrier: *barrier,
                        resource,
                    }
                }
                ExecutionAction::BeginPass(pass) => {
                    let colors = vulkan_pass_colors(&self.state, pass)
                        .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?;
                    ez_gfx_backend_vulkan::NativeFrameAction::BeginPass { pass, colors }
                }
                ExecutionAction::ExecuteNode(node) => {
                    let node = *node as usize;
                    let payload = self
                        .payloads
                        .get(node)
                        .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                    match payload {
                        ExecutableNode::Compute { groups, .. } => {
                            binding_source = self.binding_source(node)?;
                            let key = self.pipeline_keys[node]
                                .as_ref()
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                            let pipeline = self
                                .state
                                .pipelines
                                .get(key)
                                .ok_or(ez_gfx_hal::HalError::NativeFailure)?
                                .vulkan()
                                .map_err(|_| ez_gfx_hal::HalError::NativeFailure)?;
                            ez_gfx_backend_vulkan::NativeFrameAction::Compute(
                                ez_gfx_backend_vulkan::NativeComputeDispatch {
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
                            let indirect = self
                                .state
                                .allocations
                                .get(&counter.packed())
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                                .1
                                .vulkan()
                                .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?;
                            let key = self.pipeline_keys[node]
                                .as_ref()
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                            let pipeline = self
                                .state
                                .pipelines
                                .get(key)
                                .ok_or(ez_gfx_hal::HalError::NativeFailure)?
                                .vulkan()
                                .map_err(|_| ez_gfx_hal::HalError::NativeFailure)?;
                            ez_gfx_backend_vulkan::NativeFrameAction::Graphics(
                                ez_gfx_backend_vulkan::NativeDrawIndexed {
                                    width: self.state.extent.0,
                                    height: self.state.extent.1,
                                    pipeline,
                                    index_buffer: self
                                        .state
                                        .index
                                        .ok_or(ez_gfx_hal::HalError::NotReady)?,
                                    indirect_buffer: indirect,
                                    draw_count: *draw_capacity,
                                    bindings: &binding_source,
                                },
                            )
                        }
                        ExecutableNode::Mesh { stages, groups, .. } => {
                            binding_source = self.binding_source(node)?;
                            ez_gfx_backend_vulkan::NativeFrameAction::Mesh(self.mesh_draw(
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
                            let texture = self
                                .state
                                .textures
                                .get(texture)
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                            ez_gfx_backend_vulkan::NativeFrameAction::TextureReadback {
                                texture: texture
                                    .vulkan()
                                    .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?,
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
                            ez_gfx_backend_vulkan::NativeFrameAction::TextureReadback {
                                texture: record
                                    .native
                                    .vulkan()
                                    .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?,
                                width: record.width,
                                height: record.height,
                            }
                        }
                        ExecutableNode::Present { .. } => {
                            ez_gfx_backend_vulkan::NativeFrameAction::Present
                        }
                    }
                }
                ExecutionAction::EndPass => ez_gfx_backend_vulkan::NativeFrameAction::EndPass,
            };
            visitor(output_index, &native)?;
            output_index += 1;
        }
        Ok(())
    }
}

fn presentation_mode(surface: Option<&SurfaceRecord>) -> PresentationMode {
    surface.map_or(PresentationMode::Fifo, |surface| surface.presentation_mode)
}

pub(super) fn execute_vulkan_frame_plan(
    context: &mut ContextState,
    plan: &FrameExecutionPlan,
    payloads: &[ExecutableNode],
    binding_resources: &[ez_gfx_runtime::binding::ResourceIdentity],
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
    let presentation_mode = presentation_mode(surface.as_ref());
    let extent = super::frame_target_extent(context, surface.as_ref());
    let capture = should_capture_presented(
        surface
            .as_ref()
            .is_some_and(|surface| surface.state.snapshot_cache()),
        context.frame_capture_surface.is_some(),
    );

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
    let render_target_format = context
        .frame_render_target
        .and_then(|target| context.render_targets.get(&target))
        .map(|record| record.format);
    {
        let native = context.native.vulkan_mut()?;
        prepare_vulkan_surface(
            native,
            &mut context.pipelines,
            &mut context.graphics_format,
            native_surface.as_deref(),
            extent,
            presentation_mode,
        )?;
        prepare_vulkan_pipelines(
            native,
            &context.shaders,
            &mut context.pipelines,
            payloads,
            &mut context.frame_pipeline_keys,
            render_target_format,
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
        #[cfg(target_vendor = "apple")]
        &mut context.frame_texture_heaps,
        #[cfg(target_vendor = "apple")]
        &mut context.frame_workgroup_sizes,
    )?;
    let source = VulkanActionSource {
        state: VulkanActionState {
            allocations: &context.allocations,
            vertex_heaps: &context.vertex_heaps,
            textures: &context.textures,
            submitted: context.texture_pipeline.submitted(),
            render_targets: &context.render_targets,
            pipelines: &context.pipelines,
            resources: &context.frame_native_resources,
            index,
            extent,
        },
        plan,
        payloads,
        pipeline_keys: &context.frame_pipeline_keys,
        shaders: &context.shaders,
        binding_records: &context.frame_binding_scratch,
        binding_ranges: &context.frame_binding_ranges,
    };
    let execution = {
        let native = context.native.vulkan_mut()?;
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
