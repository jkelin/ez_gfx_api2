use crate::Result;
use ez_gfx_core::capability::PresentationMode;

use super::super::map_texture;
use super::{
    Backend, ContextState, Error, ExecutableNode, ExecutionAction, FrameBindingSource,
    FrameBufferBindingRecord, FrameExecutionPlan, FrameNativeResource, GeometryAllocation, HashMap,
    MAX_PIPELINE_CACHE_ENTRIES, MeshPipelineKeyDesc, MetalWorkgroupSizes, NativeAllocation,
    NativeContext, NativePipeline, NativeShader, NativeSurface, NativeTexture, PipelineKey,
    RenderTargetHandle, RenderTargetRecord, ResourceId, SURFACE_DEFAULT_CLEAR, ShaderRecord,
    SubmittedInfo, TextureHandle, TextureId, map_hal, native_layouts, pipeline_layout_key,
    prepare_frame_binding_scratch, should_capture_presented,
};
use arrayvec::ArrayVec;
use ez_gfx_backend_metal::native::{
    NativeAllocation as MetalAllocation, NativeContext as MetalContext,
    NativeFrameAction as MetalFrameAction, NativeFrameResource as MetalFrameResource,
    NativeSampledTexture as MetalSampledTexture, PassAttachment as MetalPassAttachment,
};
use ez_gfx_core::{
    capability::MAX_BINDLESS_SAMPLED_TEXTURES,
    handle::{PackedHandle, ShaderHandle, SurfaceHandle},
};
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
type PreparedMeshPipeline = (
    Option<NativePipeline>,
    Option<ShaderTextureHeapLayout>,
    MetalWorkgroupSizes,
);
type MetalTextureRecords = HashMap<TextureHandle, NativeTexture>;

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
        .workgroup_size()
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
// `None` is the surface cache domain; runtime format discriminants keep targets distinct.
fn metal_color_format_key(format: Option<ez_gfx_runtime::target::Format>) -> u32 {
    match format {
        None => 0,
        Some(format) => u32::from(format as u8),
    }
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
    color_format: Option<ez_gfx_runtime::target::Format>,
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
        color_format: metal_color_format_key(color_format),
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
                    color_format,
                    depth_required,
                    vertex_texture_heap,
                    fragment_texture_heap,
                )
                .map_err(map_hal)?,
        ))
    };

    Ok((key, pipeline, texture_heap))
}

fn prepare_mesh_pipeline(
    shaders: &HashMap<ShaderHandle, ShaderRecord>,
    pipelines: &HashMap<PipelineKey, NativePipeline>,
    key_slot: &mut Option<PipelineKey>,
    native: &MetalContext,
    stages: ez_gfx_hal::MeshStages<ShaderHandle>,
    stage_layouts: &ez_gfx_hal::MeshStages<ez_gfx_runtime::binding::StageLayoutIdentity>,
    pipeline_layout: &PipelineLayout,
    state: ez_gfx_hal::MeshPipelineState,
    color_format: Option<ez_gfx_runtime::target::Format>,
) -> Result<PreparedMeshPipeline> {
    let mesh = shaders.get(&stages.mesh).ok_or(Error::InvalidContext)?;
    let fragment = shaders.get(&stages.fragment).ok_or(Error::InvalidContext)?;
    let task = stages
        .task
        .map(|handle| shaders.get(&handle).ok_or(Error::InvalidContext))
        .transpose()?;
    let NativeShader::Metal(mesh_native) = &mesh.native else {
        return Err(Error::NativeFailure);
    };
    let NativeShader::Metal(fragment_native) = &fragment.native else {
        return Err(Error::NativeFailure);
    };
    let task_native = task
        .map(|record| match &record.native {
            NativeShader::Metal(shader) => Ok(shader),
            _ => Err(Error::NativeFailure),
        })
        .transpose()?;

    let texture_heap = metal_texture_heap(pipeline_layout.texture_heap())?;
    let task_texture_heap = task
        .map(|record| {
            record
                .runtime
                .pipeline_layout(ez_gfx_artifact::Stage::Task)
                .map_err(|_| Error::InvalidArgument)
                .and_then(|layout| metal_texture_heap(layout.texture_heap()))
        })
        .transpose()?
        .flatten();
    let mesh_texture_heap = metal_texture_heap(
        mesh.runtime
            .pipeline_layout(ez_gfx_artifact::Stage::Mesh)
            .map_err(|_| Error::InvalidArgument)?
            .texture_heap(),
    )?;
    let fragment_texture_heap = metal_texture_heap(
        fragment
            .runtime
            .pipeline_layout(ez_gfx_artifact::Stage::Fragment)
            .map_err(|_| Error::InvalidArgument)?
            .texture_heap(),
    )?;
    let task_threads = task
        .map(|record| record.runtime.workgroup_size())
        .transpose()
        .map_err(|_| Error::InvalidArgument)?;
    let mesh_threads = mesh
        .runtime
        .workgroup_size()
        .map_err(|_| Error::InvalidArgument)?;
    let depth_required = pipeline_layout.depth_required();
    let key = PipelineKey::prepare_mesh_slot(
        key_slot,
        MeshPipelineKeyDesc {
            backend: Backend::Metal,
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
            color_format: metal_color_format_key(color_format),
            depth_format: if depth_required { 252 } else { 0 },
            sample_count: 1,
        },
    );
    let pipeline = if pipelines.contains_key(key) {
        None
    } else {
        let task_buffer_layouts = task
            .map(|record| {
                record
                    .runtime
                    .bindings(ez_gfx_artifact::Stage::Task)
                    .map_err(|_| Error::InvalidArgument)
                    .and_then(|layout| native_layouts(&layout).map_err(map_hal))
            })
            .transpose()?
            .unwrap_or_default();
        let mesh_buffer_layouts = native_layouts(
            &mesh
                .runtime
                .bindings(ez_gfx_artifact::Stage::Mesh)
                .map_err(|_| Error::InvalidArgument)?,
        )
        .map_err(map_hal)?;
        let fragment_buffer_layouts = native_layouts(
            &fragment
                .runtime
                .bindings(ez_gfx_artifact::Stage::Fragment)
                .map_err(|_| Error::InvalidArgument)?,
        )
        .map_err(map_hal)?;
        let task_identity = task.map(|record| (record.product, record.entry.clone()));
        Some(NativePipeline::Metal(
            native
                .create_mesh_pipeline(
                    task_native,
                    mesh_native,
                    fragment_native,
                    task_identity.as_ref(),
                    &(mesh.product, mesh.entry.clone()),
                    &(fragment.product, fragment.entry.clone()),
                    state,
                    color_format,
                    depth_required,
                    task_texture_heap,
                    mesh_texture_heap,
                    fragment_texture_heap,
                    &task_buffer_layouts,
                    &mesh_buffer_layouts,
                    &fragment_buffer_layouts,
                    task_threads,
                    mesh_threads,
                )
                .map_err(map_hal)?,
        ))
    };
    Ok((
        pipeline,
        texture_heap,
        MetalWorkgroupSizes::Mesh {
            task: task_threads,
            mesh: mesh_threads,
        },
    ))
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
    keys: &mut Vec<Option<PipelineKey>>,
    texture_heaps: &mut Vec<Option<ShaderTextureHeapLayout>>,
    workgroup_sizes: &mut Vec<Option<MetalWorkgroupSizes>>,
    color_format: Option<ez_gfx_runtime::target::Format>,
) -> Result<()> {
    keys.truncate(payloads.len());
    keys.resize_with(payloads.len(), || None);
    texture_heaps.clear();
    texture_heaps.resize(payloads.len(), None);
    workgroup_sizes.clear();
    workgroup_sizes.resize(payloads.len(), None);
    for (node_index, payload) in payloads.iter().enumerate() {
        if let ExecutableNode::Mesh {
            stages,
            pipeline_layout,
            stage_layouts,
            state,
            ..
        } = payload
        {
            let (pipeline, texture_heap, workgroups) = prepare_mesh_pipeline(
                shaders,
                pipelines,
                &mut keys[node_index],
                native,
                *stages,
                stage_layouts,
                pipeline_layout,
                *state,
                color_format,
            )?;
            texture_heaps[node_index] = texture_heap;
            workgroup_sizes[node_index] = Some(workgroups);
            if let Some(pipeline) = pipeline {
                let key = keys[node_index]
                    .as_ref()
                    .ok_or(Error::InvalidArgument)?
                    .clone();
                cache_metal_pipeline(pipelines, native, key, Some(pipeline))?;
            }
            continue;
        }
        let (key, pipeline) = match payload {
            ExecutableNode::Compute { shader, layout, .. } => {
                let (key, pipeline, texture_heap, workgroup_size) =
                    prepare_compute_pipeline(shaders, pipelines, native, *shader, layout)?;
                texture_heaps[node_index] = texture_heap;
                workgroup_sizes[node_index] = Some(MetalWorkgroupSizes::Compute(workgroup_size));
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
                    color_format,
                )?;
                texture_heaps[node_index] = texture_heap;
                (key, pipeline)
            }
            ExecutableNode::Mesh { .. } => unreachable!("mesh payload handled above"),
            ExecutableNode::TextureReadback { .. }
            | ExecutableNode::RenderTargetReadback { .. }
            | ExecutableNode::Present { .. } => continue,
        };
        cache_metal_pipeline(pipelines, native, key.clone(), pipeline)?;
        keys[node_index] = Some(key);
    }
    Ok(())
}

struct MetalActionSource<'a, 'resources> {
    payloads: &'a [ExecutableNode],
    plan: &'a FrameExecutionPlan,
    action_indices: &'a [usize],
    allocations: &'resources HashMap<PackedHandle, (u64, NativeAllocation)>,
    vertex_heaps: &'resources HashMap<String, GeometryAllocation>,
    textures: &'resources MetalTextureRecords,
    submitted: &'resources HashMap<TextureHandle, SubmittedInfo>,
    render_targets: &'resources HashMap<RenderTargetHandle, RenderTargetRecord>,
    pipelines: &'resources HashMap<PipelineKey, NativePipeline>,
    frame_resources: &'resources HashMap<ResourceId, FrameNativeResource>,
    native_textures: &'a [MetalSampledTexture<'resources>],
    index: Option<&'resources MetalAllocation>,
    index_size: u64,
    surface: Option<SurfaceHandle>,
    keys: &'a [Option<PipelineKey>],
    binding_records: &'a [FrameBufferBindingRecord],
    binding_ranges: &'a [core::ops::Range<usize>],
    texture_heaps: &'a [Option<ShaderTextureHeapLayout>],
    workgroup_sizes: &'a [Option<MetalWorkgroupSizes>],
}

impl ez_gfx_backend_metal::native::NativeFrameActionSource for MetalActionSource<'_, '_> {
    fn len(&self) -> usize {
        self.action_indices.len()
    }

    fn with_action(
        &self,
        index: usize,
        visitor: &mut dyn FnMut(
            &MetalFrameAction<'_>,
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
            ExecutionAction::Wait(wait) => {
                MetalFrameAction::Wait(wait.external.ok_or(ez_gfx_hal::HalError::InvalidArgument)?)
            }
            ExecutionAction::Barrier(barrier) => {
                let resource = self
                    .frame_resources
                    .get(&ResourceId::from_index(barrier.resource))
                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                let resource = match *resource {
                    FrameNativeResource::Buffer(handle) => {
                        let NativeAllocation::Metal(allocation) = &self
                            .allocations
                            .get(&handle)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                            .1
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameResource::Buffer(allocation)
                    }
                    FrameNativeResource::Texture(handle) => {
                        let NativeTexture::Metal(texture) = self
                            .textures
                            .get(&handle)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameResource::Texture(texture)
                    }
                    FrameNativeResource::Surface(_) => MetalFrameResource::Surface,
                    FrameNativeResource::Depth => MetalFrameResource::Depth,
                    FrameNativeResource::RenderTarget(handle) => {
                        let record = self
                            .render_targets
                            .get(&handle)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativeTexture::Metal(texture) = &record.native else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameResource::RenderTarget(texture)
                    }
                    FrameNativeResource::Index => MetalFrameResource::Buffer(
                        self.index.ok_or(ez_gfx_hal::HalError::NotReady)?,
                    ),
                    FrameNativeResource::VertexHeap(heap_id) => {
                        let heap = self
                            .vertex_heaps
                            .values()
                            .find(|heap| heap.heap_id == Some(heap_id))
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativeAllocation::Metal(allocation) = &heap.allocation else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameResource::Buffer(allocation)
                    }
                };
                MetalFrameAction::Barrier {
                    barrier: *barrier,
                    resource,
                }
            }
            ExecutionAction::BeginPass(pass) => {
                let mut colors = arrayvec::ArrayVec::new();
                for color in &pass.colors {
                    let resource = self
                        .frame_resources
                        .get(&ResourceId::from_index(*color))
                        .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                    colors
                        .try_push(match *resource {
                            FrameNativeResource::Surface(_) => MetalPassAttachment {
                                resource: MetalFrameResource::Surface,
                                clear: SURFACE_DEFAULT_CLEAR,
                            },
                            FrameNativeResource::RenderTarget(handle) => {
                                let record = self
                                    .render_targets
                                    .get(&handle)
                                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                                let NativeTexture::Metal(texture) = &record.native else {
                                    return Err(ez_gfx_hal::HalError::InvalidArgument);
                                };
                                MetalPassAttachment {
                                    resource: MetalFrameResource::RenderTarget(texture),
                                    clear: super::super::render_target::render_target_clear_color(
                                        record,
                                    ),
                                }
                            }
                            _ => return Err(ez_gfx_hal::HalError::InvalidArgument),
                        })
                        .map_err(|_| ez_gfx_hal::HalError::InvalidArgument)?;
                }
                MetalFrameAction::BeginPass { pass, colors }
            }
            ExecutionAction::ExecuteNode(node) => {
                let node = *node as usize;
                match self
                    .payloads
                    .get(node)
                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                {
                    ExecutableNode::Graphics {
                        counter,
                        draw_capacity,
                        pipeline_layout,
                        state,
                        ..
                    } => {
                        let range = self
                            .binding_ranges
                            .get(node)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                            .clone();
                        binding_source = FrameBindingSource {
                            records: self
                                .binding_records
                                .get(range)
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
                            allocations: self.allocations,
                            vertex_heaps: self.vertex_heaps,
                        };
                        let NativePipeline::Metal(pipeline) = self
                            .pipelines
                            .get(
                                self.keys[node]
                                    .as_ref()
                                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
                            )
                            .ok_or(ez_gfx_hal::HalError::NativeFailure)?
                        else {
                            return Err(ez_gfx_hal::HalError::NativeFailure);
                        };
                        let (indirect_size, NativeAllocation::Metal(indirect)) = self
                            .allocations
                            .get(&counter.packed())
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameAction::Graphics(
                            ez_gfx_backend_metal::native::NativeGraphicsDraw {
                                pipeline,
                                depth_required: pipeline_layout.depth_required(),
                                texture_heap: self.texture_heaps[node],
                                state: *state,
                                index: self.index.ok_or(ez_gfx_hal::HalError::NotReady)?,
                                index_size: self.index_size,
                                indirect,
                                indirect_size: *indirect_size,
                                draw_count: *draw_capacity,
                                bindings: &binding_source,
                                textures: self.native_textures,
                            },
                        )
                    }
                    ExecutableNode::Mesh {
                        groups,
                        pipeline_layout,
                        state,
                        ..
                    } => {
                        let range = self
                            .binding_ranges
                            .get(node)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                            .clone();
                        binding_source = FrameBindingSource {
                            records: self
                                .binding_records
                                .get(range)
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
                            allocations: self.allocations,
                            vertex_heaps: self.vertex_heaps,
                        };
                        let NativePipeline::Metal(pipeline) = self
                            .pipelines
                            .get(
                                self.keys[node]
                                    .as_ref()
                                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
                            )
                            .ok_or(ez_gfx_hal::HalError::NativeFailure)?
                        else {
                            return Err(ez_gfx_hal::HalError::NativeFailure);
                        };
                        let MetalWorkgroupSizes::Mesh { task, mesh } =
                            self.workgroup_sizes[node]
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameAction::Mesh(ez_gfx_backend_metal::native::NativeMeshDraw {
                            pipeline,
                            groups: *groups,
                            task_threads_per_group: task,
                            mesh_threads_per_group: mesh,
                            depth_required: pipeline_layout.depth_required(),
                            state: *state,
                            texture_heap: self.texture_heaps[node],
                            bindings: &binding_source,
                            textures: self.native_textures,
                        })
                    }
                    ExecutableNode::Compute { groups, .. } => {
                        let range = self
                            .binding_ranges
                            .get(node)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                            .clone();
                        binding_source = FrameBindingSource {
                            records: self
                                .binding_records
                                .get(range)
                                .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
                            allocations: self.allocations,
                            vertex_heaps: self.vertex_heaps,
                        };
                        let NativePipeline::Metal(pipeline) = self
                            .pipelines
                            .get(
                                self.keys[node]
                                    .as_ref()
                                    .ok_or(ez_gfx_hal::HalError::InvalidArgument)?,
                            )
                            .ok_or(ez_gfx_hal::HalError::NativeFailure)?
                        else {
                            return Err(ez_gfx_hal::HalError::NativeFailure);
                        };
                        let MetalWorkgroupSizes::Compute(threads_per_group) = self.workgroup_sizes
                            [node]
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameAction::Compute(
                            ez_gfx_backend_metal::native::NativeComputeDispatch {
                                pipeline,
                                groups: *groups,
                                threads_per_group,
                                bindings: &binding_source,
                                texture_heap: self.texture_heaps[node],
                                textures: self.native_textures,
                            },
                        )
                    }
                    ExecutableNode::TextureReadback { texture } => {
                        let info = self
                            .submitted
                            .get(texture)
                            .copied()
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativeTexture::Metal(texture) = self
                            .textures
                            .get(texture)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameAction::TextureReadback {
                            texture,
                            width: info.width,
                            height: info.height,
                        }
                    }
                    ExecutableNode::RenderTargetReadback { target } => {
                        let record = self
                            .render_targets
                            .get(target)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?;
                        let NativeTexture::Metal(texture) = &record.native else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        MetalFrameAction::TextureReadback {
                            texture,
                            width: record.width,
                            height: record.height,
                        }
                    }
                    ExecutableNode::Present { surface } => {
                        if Some(*surface) != self.surface {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        }
                        MetalFrameAction::Present
                    }
                }
            }
            ExecutionAction::EndPass => MetalFrameAction::EndPass,
        };
        visitor(&native)
    }
}
pub(super) fn execute_metal_frame_plan(
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
    let NativeTexture::Metal(fallback) = context.texture_fallback.ready().ok_or(Error::NotReady)?
    else {
        return Err(Error::NativeFailure);
    };
    let mut native_textures = ArrayVec::<_, { MAX_BINDLESS_SAMPLED_TEXTURES as usize }>::new();
    // Only live and pending handles participate. Failed handles and both retirement queues have
    // ended their documented binding lifetime, so their fallback aliases are intentionally absent.
    for (handle, texture) in &context.textures {
        let id = context
            .texture_pipeline
            .submitted()
            .get(handle)
            .map(|info| info.id);
        let sampled = if context
            .texture_pipeline
            .published()
            .get(handle)
            .is_some_and(|mips| *mips != 0)
        {
            let NativeTexture::Metal(texture) = texture else {
                return Err(Error::NativeFailure);
            };
            texture.sampled()
        } else {
            let binding = context
                .texture_registry
                .reserved_binding(id.ok_or(Error::InvalidContext)?)
                .map_err(map_texture)?;
            fallback.fallback_sampled(binding)
        };
        native_textures
            .try_push(sampled)
            .map_err(|_| Error::NativeFailure)?;
    }
    for pending in context.texture_pipeline.pending().values() {
        let binding = context
            .texture_registry
            .reserved_binding(pending.id)
            .map_err(map_texture)?;
        native_textures
            .try_push(fallback.fallback_sampled(binding))
            .map_err(|_| Error::NativeFailure)?;
    }
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
    let color_format = context
        .frame_render_target
        .and_then(|target| context.render_targets.get(&target))
        .map(|record| record.format);
    prepare_metal_pipelines(
        &context.shaders,
        &mut context.pipelines,
        native,
        payloads,
        &mut context.frame_pipeline_keys,
        &mut context.frame_texture_heaps,
        &mut context.frame_workgroup_sizes,
        color_format,
    )?;
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
        &mut context.frame_texture_heaps,
        &mut context.frame_workgroup_sizes,
    )?;
    let source = MetalActionSource {
        payloads,
        plan,
        action_indices: &context.frame_action_indices,
        allocations: &context.allocations,
        vertex_heaps: &context.vertex_heaps,
        textures: &context.textures,
        submitted: context.texture_pipeline.submitted(),
        render_targets: &context.render_targets,
        pipelines: &context.pipelines,
        frame_resources: &context.frame_native_resources,
        native_textures: &native_textures,
        binding_records: &context.frame_binding_scratch,
        binding_ranges: &context.frame_binding_ranges,
        index,
        index_size,
        surface: surface_handle,
        keys: &context.frame_pipeline_keys,
        texture_heaps: &context.frame_texture_heaps,
        workgroup_sizes: &context.frame_workgroup_sizes,
    };
    let result = match surface.as_mut().map(|surface| &mut surface.native) {
        Some(NativeSurface::Metal(surface)) => native
            .execute_frame_source(Some((surface, extent, presentation_mode)), &source, capture)
            .map_err(map_hal),
        None => native
            .execute_frame_source(None, &source, false)
            .map_err(map_hal),
        Some(NativeSurface::Vulkan(_)) => Err(Error::NativeFailure),
    };
    let output_count_valid = result.as_ref().map_or(true, |outputs| {
        outputs.len() == super::expected_frame_output_count(payloads, capture)
    });
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
