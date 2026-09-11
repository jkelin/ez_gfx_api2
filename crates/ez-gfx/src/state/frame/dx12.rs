use crate::Result;
use ez_gfx_core::capability::PresentationMode;

use super::{
    Backend, ContextState, Error, ExecutableNode, ExecutionAction, ExecutionBarrier, ExecutionPass,
    FrameExecutionPlan, FrameNativeResource, GeometryAllocation, HashMap,
    MAX_PIPELINE_CACHE_ENTRIES, NativeAllocation, NativeContext, NativePipeline, NativeShader,
    NativeSurface, NativeTexture, NativeTextureMap, PackedHandle, PipelineKey, RenderTargetHandle,
    FrameBindingSource, FrameBufferBindingRecord, RenderTargetRecord, ResourceId,
    SURFACE_DEFAULT_CLEAR, ShaderHandle, ShaderRecord, map_hal, native_layouts,
    pipeline_layout_key, prepare_frame_binding_scratch, should_capture_presented,
};

use arrayvec::ArrayVec;

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
    pipeline_keys: &mut Vec<Option<PipelineKey>>,
) -> Result<()> {
    pipeline_keys.clear();
    pipeline_keys.resize_with(payloads.len(), || None);
    for (node_index, payload) in payloads.iter().enumerate() {
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
                                vertex_native,
                                fragment_native,
                                vertex.product,
                                fragment.product,
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
                FrameNativeResource::Surface(_) => {
                    ez_gfx_backend_dx12::native::PassAttachment {
                        resource: ez_gfx_backend_dx12::native::NativeFrameResource::Surface,
                        clear: SURFACE_DEFAULT_CLEAR,
                    }
                }
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
    action_indices: &'a [usize],
    binding_records: &'a [FrameBufferBindingRecord],
    binding_ranges: &'a [core::ops::Range<usize>],
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
            .get(*self.action_indices.get(index).ok_or(ez_gfx_hal::HalError::InvalidArgument)?)
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
                            allocations: self.state.allocations,
                            vertex_heaps: self.state.vertex_heaps,
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
                            allocations: self.state.allocations,
                            vertex_heaps: self.state.vertex_heaps,
                        };
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
                    ExecutableNode::TextureReadback { texture } => {
                        let (_, NativeTexture::Dx12(texture), width, height, _) = self
                            .state
                            .textures
                            .get(texture)
                            .ok_or(ez_gfx_hal::HalError::InvalidArgument)?
                        else {
                            return Err(ez_gfx_hal::HalError::InvalidArgument);
                        };
                        ez_gfx_backend_dx12::native::NativeFrameAction::TextureReadback {
                            texture,
                            width: *width,
                            height: *height,
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

#[cfg(windows)]
pub(super) fn execute_dx12_frame_plan(
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
    context.frame_action_indices.extend(
        plan.actions
            .iter()
            .enumerate()
            .filter_map(|(index, action)| {
                (!matches!(action, ExecutionAction::Wait(wait) if wait.external.is_none()))
                    .then_some(index)
            }),
    );
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
    )?;
    let source = DxActionSource {
        state: DxActionState {
            allocations: &context.allocations,
            vertex_heaps: &context.vertex_heaps,
            textures: &context.textures,
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
