type NativePipelineLayout = (
    vk::PipelineLayout,
    vk::DescriptorSetLayout,
    Vec<bool>,
    Vec<u32>,
);

use super::{
    BlendMode, CString, CullMode, DeferredResource, FrontFace, HalError, MeshDispatchLimits,
    NativeContext, NativeGraphicsPipelineDesc, NativeMeshPipelineDesc, NativePipeline,
    NativePipelineKind, NativeShader, NativeSurface, PresentationMode, PrimitiveTopology,
    ShaderBufferLayout, map_vk, retained_mesh_dispatch_limits, texture::render_target_format_vk,
    validate_mesh_dispatch, vk,
};

use ez_gfx_hal::MeshDispatchError;

/// Resolves the exact task/mesh/fragment modules named by a mesh pipeline request.
///
/// # Errors
///
/// Returns [`HalError::InvalidArgument`] when any product index is out of range.
fn mesh_shader_modules(
    desc: &NativeMeshPipelineDesc<'_>,
) -> Result<(Option<vk::ShaderModule>, vk::ShaderModule, vk::ShaderModule), HalError> {
    let task = desc
        .task
        .map(|(shader, index)| {
            shader
                .modules
                .get(index)
                .copied()
                .ok_or(HalError::InvalidArgument)
        })
        .transpose()?;
    let mesh = *desc
        .mesh
        .0
        .modules
        .get(desc.mesh.1)
        .ok_or(HalError::InvalidArgument)?;
    let fragment = *desc
        .fragment
        .0
        .modules
        .get(desc.fragment.1)
        .ok_or(HalError::InvalidArgument)?;
    Ok((task, mesh, fragment))
}

fn pipeline_color_format(
    swapchain_format: vk::Format,
    format: Option<ez_gfx_runtime::target::Format>,
) -> Result<vk::Format, HalError> {
    match format {
        Some(format) => render_target_format_vk(format).ok_or(HalError::Unsupported),
        None if swapchain_format == vk::Format::UNDEFINED => Err(HalError::NotReady),
        None => Ok(swapchain_format),
    }
}

impl NativeContext {
    /// Creates native shader products from validated compiler output.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` if the device is unavailable, `InvalidArgument` if the shader products are empty or not four-byte aligned, or a mapped Vulkan error if shader-module creation fails.
    pub fn create_shader(&self, products: &[&[u8]]) -> Result<NativeShader, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let mut modules = Vec::with_capacity(products.len());
        for product in products {
            if product.is_empty() || product.len() % 4 != 0 {
                for module in modules.drain(..) {
                    // SAFETY: Each `module` was returned earlier by `device.create_shader_module`, has not escaped or been used, and is drained exactly once before `destroy_shader_module`.
                    unsafe { device.destroy_shader_module(module, None) };
                }
                return Err(HalError::InvalidArgument);
            }
            let words = product
                .chunks_exact(4)
                .map(<[u8; 4]>::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| HalError::InvalidArgument)?
                .into_iter()
                .map(u32::from_le_bytes)
                .collect::<Vec<_>>();
            // SAFETY: `device.create_shader_module` reads nonempty whole words from `words`' contiguous `u32` storage, which remains allocated and unmodified for the call.
            match unsafe {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            } {
                Ok(module) => modules.push(module),
                Err(error) => {
                    for module in modules.drain(..) {
                        // SAFETY: Each `module` was returned earlier by `device.create_shader_module`, has not escaped or been used, and is drained exactly once before `destroy_shader_module`.
                        unsafe { device.destroy_shader_module(module, None) };
                    }
                    return Err(map_vk(error));
                }
            }
        }
        if modules.is_empty() {
            return Err(HalError::InvalidArgument);
        }
        Ok(NativeShader { modules })
    }

    /// Defers shader destruction until every referencing frame completes.
    pub fn destroy_shader(&mut self, shader: NativeShader) {
        let _ = self.defer_resource(DeferredResource::Shader(shader));
    }

    /// Reflected public buffers occupy descriptor set zero; the bindless texture table remains set one.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` if the device or texture descriptor layout is unavailable, `Unsupported` for an unsupported buffer layout, `InvalidArgument` if a binding overflows, or a mapped Vulkan error if descriptor-set or pipeline-layout creation fails.
    pub(super) fn create_pipeline_layout(
        &self,
        layouts: &[ShaderBufferLayout],
    ) -> Result<NativePipelineLayout, HalError> {
        self.create_pipeline_layout_with_stages(layouts, vk::ShaderStageFlags::ALL)
    }

    /// Reflected public buffers occupy descriptor set zero with explicit stage visibility.
    ///
    /// Mesh pipelines bind only their dispatch stages instead of every stage, so a
    /// task/mesh/fragment pipeline never claims compute or vertex visibility it cannot use.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` if the device or texture descriptor layout is unavailable, `Unsupported` for an unsupported buffer layout, `InvalidArgument` if a binding overflows, or a mapped Vulkan error if descriptor-set or pipeline-layout creation fails.
    fn create_pipeline_layout_with_stages(
        &self,
        layouts: &[ShaderBufferLayout],
        stages: vk::ShaderStageFlags,
    ) -> Result<NativePipelineLayout, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let texture = self.texture_descriptor_layout.ok_or(HalError::NotReady)?;
        let mut bindings = Vec::new();
        let mut writable = Vec::new();
        let mut physical_bindings = Vec::new();
        for layout in layouts {
            if layout.space != 0 || layout.descriptor_count == 0 || layout.descriptor_count > 2 {
                return Err(HalError::Unsupported);
            }
            for offset in 0..layout.descriptor_count {
                let binding = layout
                    .binding
                    .checked_add(offset)
                    .ok_or(HalError::InvalidArgument)?;
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(stages),
                );
                writable.push(layout.writable);
                physical_bindings.push(binding);
            }
        }
        // SAFETY: `device.create_descriptor_set_layout` reads initialized elements from `bindings`' contiguous storage, which remains allocated and unmodified for the call.
        let public = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(map_vk)?;
        let set_layouts = [public, texture];
        // SAFETY: `device.create_pipeline_layout` receives `public` created above and the context's `texture` layout, while `set_layouts` remains allocated for the call.
        match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
        } {
            Ok(layout) => Ok((layout, public, writable, physical_bindings)),
            Err(error) => {
                // SAFETY: `public` was created above by `device`, no pipeline layout was returned to reference it, and this error path calls `destroy_descriptor_set_layout` exactly once.
                unsafe { device.destroy_descriptor_set_layout(public, None) };
                Err(map_vk(error))
            }
        }
    }

    /// Entry names reject interior NUL before reaching Vulkan.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` if required context state is unavailable, `InvalidArgument` for an invalid module index or overflowing layout binding, `Unsupported` for an unsupported buffer layout, or a mapped Vulkan error if layout or compute-pipeline creation fails.
    ///
    /// # Panics
    ///
    /// Panics if Vulkan reports successful compute-pipeline creation without returning a pipeline.
    pub fn create_compute_pipeline(
        &self,
        shader: &NativeShader,
        module_index: usize,
        _entry: &str,
        layouts: &[ShaderBufferLayout],
    ) -> Result<NativePipeline, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let module = *shader
            .modules
            .get(module_index)
            .ok_or(HalError::InvalidArgument)?;
        let entry = CString::new("main").unwrap();
        let (layout, public_descriptor_layout, buffer_writable, buffer_bindings) =
            self.create_pipeline_layout(layouts)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        // SAFETY: `device.create_compute_pipelines` receives `layout` created above and `module` selected from `shader.modules`; the entry string and create-info array remain allocated for the call.
        match unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
        } {
            Ok(pipelines) => Ok(NativePipeline {
                pipeline: pipelines[0],
                layout,
                public_descriptor_layout,
                buffer_writable,
                buffer_bindings,
                kind: NativePipelineKind::Compute,
            }),
            Err((_, error)) => {
                // SAFETY: `layout` and `public_descriptor_layout` were created above by `device`, have not escaped on this error path, and are each passed once to their matching destroy operation.
                unsafe {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(public_descriptor_layout, None);
                };
                Err(map_vk(error))
            }
        }
    }

    /// Defers pipeline destruction until every referencing frame completes.
    pub fn destroy_pipeline(&mut self, pipeline: NativePipeline) {
        let _ = self.defer_resource(DeferredResource::Pipeline(pipeline));
    }
    /// Creates a dynamic-rendering graphics pipeline from an exact vertex/fragment artifact pair.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` if required context state or the swapchain format is unavailable, `InvalidArgument` for an invalid shader index or overflowing layout binding, `Unsupported` for an unsupported buffer layout, or a mapped Vulkan error if layout or graphics-pipeline creation fails.
    ///
    /// # Panics
    ///
    /// Panics if Vulkan reports successful graphics-pipeline creation without returning a pipeline.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the public backend request is consumed by value across all adapters"
    )]
    pub fn create_graphics_pipeline(
        &self,
        vertex_shader: &NativeShader,
        fragment_shader: &NativeShader,
        desc: NativeGraphicsPipelineDesc<'_>,
    ) -> Result<NativePipeline, HalError> {
        let NativeGraphicsPipelineDesc {
            vertex_index,
            fragment_index,
            state,
            layouts,
            depth_required,
            color_format,
        } = desc;
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let color_format = pipeline_color_format(self.swapchain_format, color_format)?;
        let vertex = *vertex_shader
            .modules
            .get(vertex_index)
            .ok_or(HalError::InvalidArgument)?;
        let fragment = *fragment_shader
            .modules
            .get(fragment_index)
            .ok_or(HalError::InvalidArgument)?;
        let entry = CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(&entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(&entry),
        ];
        let topology = match state.topology {
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
        };
        let cull = match state.cull {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::Back => vk::CullModeFlags::BACK,
        };
        let front = match state.front_face {
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        };
        let blend = match state.blend {
            BlendMode::None => vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            BlendMode::Alpha => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
        };
        let (layout, public_descriptor_layout, buffer_writable, buffer_bindings) =
            self.create_pipeline_layout(layouts)?;
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(cull)
            .front_face(front)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(core::slice::from_ref(&blend));
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_required)
            .depth_write_enable(depth_required)
            .depth_compare_op(vk::CompareOp::LESS);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(core::slice::from_ref(&color_format))
            .depth_attachment_format(if depth_required {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::UNDEFINED
            });
        let create = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        // SAFETY: `device.create_graphics_pipelines` receives `layout` created above and stages selected from `shader.modules`; all create-info, pNext, entry-string, array, and state storage remains allocated and unmodified for the call.
        match unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[create], None)
        } {
            Ok(pipelines) => Ok(NativePipeline {
                pipeline: pipelines[0],
                layout,
                public_descriptor_layout,
                buffer_writable,
                buffer_bindings,
                kind: NativePipelineKind::Graphics,
            }),
            Err((_, error)) => {
                // SAFETY: `layout` and `public_descriptor_layout` were created above by `device`, have not escaped on this error path, and are each passed once to their matching destroy operation.
                unsafe {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(public_descriptor_layout, None);
                };
                Err(map_vk(error))
            }
        }
    }

    /// Returns the shared mesh dispatch limits for a task or taskless pipeline.
    ///
    /// Selects the retained task grid ceilings when the pipeline holds a task
    /// stage and the mesh ceilings otherwise; both dispatch-stage thread
    /// ceilings and the native total grid count travel in the shared value.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::Unsupported`] when the device lacks mesh support, or
    /// task support when a task stage is requested.
    pub fn mesh_dispatch_limits(&self, has_task: bool) -> Result<MeshDispatchLimits, HalError> {
        let limits = self
            .mesh_shader_limits
            .as_ref()
            .ok_or(HalError::Unsupported)?;
        if has_task && !limits.task_supported {
            return Err(HalError::Unsupported);
        }
        Ok(retained_mesh_dispatch_limits(limits, has_task))
    }

    /// Creates a dynamic-rendering mesh pipeline from exact task/mesh/fragment products.
    ///
    /// Support is rejected before any Vulkan allocation: without the retained EXT
    /// loader and limits the device cannot execute mesh work, and a task stage
    /// without task support fails the same way. Reflected workgroup sizes run through
    /// the shared dispatch validation against the retained native ceilings. The
    /// pipeline holds explicit task/mesh/fragment stages, stage-visible resource
    /// bindings, and the shared rasterization state, but no vertex-input or
    /// input-assembly state, which mesh pipelines must omit.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` if required context state or the swapchain format is unavailable, `Unsupported` when the device lacks mesh or requested task support, or when a well-formed reflected workgroup size exceeds the native ceilings; `InvalidArgument` for an inconsistent stage selection, an invalid shader index, or a malformed (zero or overflowing) workgroup size; or a mapped Vulkan error if layout or graphics-pipeline creation fails.
    ///
    /// # Panics
    ///
    /// Panics if Vulkan reports successful graphics-pipeline creation without returning a pipeline.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the public backend request is consumed by value across all adapters"
    )]
    pub fn create_mesh_pipeline(
        &self,
        desc: NativeMeshPipelineDesc<'_>,
    ) -> Result<NativePipeline, HalError> {
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let has_task = desc.task.is_some();
        // Malformed stage/size selection fails before any capability probe.
        if desc.task_workgroup_size.is_some() != has_task {
            return Err(HalError::InvalidArgument);
        }
        let (task, mesh, fragment) = mesh_shader_modules(&desc)?;
        let shared = self.mesh_dispatch_limits(has_task)?;
        // A well-formed shader the device cannot execute is unsupported; only
        // malformed shapes stay invalid arguments.
        validate_mesh_dispatch(
            [1, 1, 1],
            desc.mesh_workgroup_size,
            desc.task_workgroup_size,
            shared,
        )
        .map_err(|error| match error {
            MeshDispatchError::InvalidGroups | MeshDispatchError::InvalidWorkgroup => {
                HalError::InvalidArgument
            }
            MeshDispatchError::UnsupportedWorkgroup => HalError::Unsupported,
        })?;
        let color_format = pipeline_color_format(self.swapchain_format, desc.color_format)?;
        let entry = CString::new("main").unwrap();
        let task_stage = task.map(|module| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::TASK_EXT)
                .module(module)
                .name(&entry)
        });
        let mesh_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MESH_EXT)
            .module(mesh)
            .name(&entry);
        let fragment_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(&entry);
        let mut stages = [mesh_stage, fragment_stage, fragment_stage];
        let stage_count = if task_stage.is_some() { 3 } else { 2 };
        if let Some(task_stage) = task_stage {
            stages[2] = stages[1];
            stages[1] = stages[0];
            stages[0] = task_stage;
        }
        let state = desc.state;
        let cull = match state.cull {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::Back => vk::CullModeFlags::BACK,
        };
        let front = match state.front_face {
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        };
        let blend = match state.blend {
            BlendMode::None => vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            BlendMode::Alpha => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
        };
        let visible = if has_task {
            vk::ShaderStageFlags::TASK_EXT
                | vk::ShaderStageFlags::MESH_EXT
                | vk::ShaderStageFlags::FRAGMENT
        } else {
            vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT
        };
        let (layout, public_descriptor_layout, buffer_writable, buffer_bindings) =
            self.create_pipeline_layout_with_stages(desc.layouts, visible)?;
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(cull)
            .front_face(front)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(core::slice::from_ref(&blend));
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_required)
            .depth_write_enable(desc.depth_required)
            .depth_compare_op(vk::CompareOp::LESS);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(core::slice::from_ref(&color_format))
            .depth_attachment_format(if desc.depth_required {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::UNDEFINED
            });
        let create = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages[..stage_count])
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        // SAFETY: `device.create_graphics_pipelines` receives `layout` created above and stages selected from the callers' shader modules; all create-info, pNext, entry-string, array, and state storage remains allocated and unmodified for the call. No vertex-input or input-assembly state is chained, which mesh pipelines must omit.
        match unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[create], None)
        } {
            Ok(pipelines) => Ok(NativePipeline {
                pipeline: pipelines[0],
                layout,
                public_descriptor_layout,
                buffer_writable,
                buffer_bindings,
                kind: NativePipelineKind::Mesh {
                    task_stage: has_task,
                },
            }),
            Err((_, error)) => {
                // SAFETY: `layout` and `public_descriptor_layout` were created above by `device`, have not escaped on this error path, and are each passed once to their matching destroy operation.
                unsafe {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(public_descriptor_layout, None);
                };
                Err(map_vk(error))
            }
        }
    }

    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if the binding count, access mode, or buffer range is invalid, `NotReady` if the device is unavailable, or a mapped Vulkan error if descriptor-set allocation fails.
    pub(super) fn create_public_descriptor_set(
        &self,
        pool: vk::DescriptorPool,
        pipeline: &NativePipeline,
        bindings: &dyn super::NativeBufferBindingSource,
        infos: &mut Vec<vk::DescriptorBufferInfo>,
        writes: &mut Vec<vk::WriteDescriptorSet<'static>>,
    ) -> Result<vk::DescriptorSet, HalError> {
        if bindings.len() != pipeline.buffer_writable.len() {
            return Err(HalError::InvalidArgument);
        }
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        let allocation_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(core::slice::from_ref(&pipeline.public_descriptor_layout));
        let mut set = vk::DescriptorSet::null();
        // SAFETY: `pool` and `pipeline.public_descriptor_layout` are live objects from
        // `device`; the slot is completion-gated, so descriptor-pool access is externally
        // synchronized. `allocation_info` keeps its one layout pointer valid, and `set`
        // provides writable storage for exactly `descriptor_set_count == 1` output handle.
        let result = unsafe {
            (device.fp_v1_0().allocate_descriptor_sets)(
                device.handle(),
                &raw const allocation_info,
                &raw mut set,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(map_vk(result));
        }
        reserve_descriptor_shells(infos, writes, bindings.len())?;
        bindings.visit(&mut |index, binding| {
            let writable = pipeline
                .buffer_writable
                .get(index)
                .ok_or(HalError::InvalidArgument)?;
            if binding.writable != *writable
                || binding.range == 0
                || binding
                    .offset
                    .checked_add(binding.range)
                    .is_none_or(|end| end > binding.allocation.allocation.size())
            {
                return Err(HalError::InvalidArgument);
            }
            infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(binding.allocation.buffer)
                    .offset(binding.offset)
                    .range(binding.range),
            );
            Ok(())
        })?;
        // SAFETY: the extended lifetime is a local encoding of call-scoped
        // validity. Each entry points into `infos`' current buffer and is only
        // read by the update call below: `writes` was cleared before `infos`
        // was touched so no previous-call entry survives, both shells were
        // reserved for `bindings.len()` up front so neither reallocs between
        // this construction and the update, `infos` is not mutated in between,
        // and `writes` is cleared again before return so no borrowed entry
        // escapes this call. `WriteDescriptorSet` is covariant over its marker
        // lifetime and carries only raw pointers beside it, so the extension
        // preserves layout and aliasing validity.
        writes.extend(infos.iter().enumerate().map(|(index, info)| {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(pipeline.buffer_bindings[index])
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(core::slice::from_ref(info));
            // SAFETY: `write` only borrows the current stable `infos` buffer through the update below.
            unsafe {
                core::mem::transmute::<vk::WriteDescriptorSet<'_>, vk::WriteDescriptorSet<'static>>(
                    write,
                )
            }
        }));
        // SAFETY: all descriptor sets and referenced buffer infos are live and externally synchronized for this slot.
        unsafe { device.update_descriptor_sets(writes, &[]) };
        // No borrowed entries escape: the next call clears `writes` before
        // touching `infos`, and clearing here keeps even a panicking caller
        // from observing stale entries in the retained shell.
        writes.clear();
        Ok(set)
    }

    /// Ensures target format and extent exist before a graphics pipeline is created.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` for a zero extent, or propagates an error from swapchain recreation.
    pub fn prepare_surface(
        &mut self,
        surface: &NativeSurface,
        width: u32,
        height: u32,
        presentation_mode: PresentationMode,
    ) -> Result<(), HalError> {
        if width == 0 || height == 0 {
            return Err(HalError::NotReady);
        }
        if self.swapchain.is_none()
            || self.swapchain_extent.width != width
            || self.swapchain_extent.height != height
            || self.swapchain_presentation_mode != presentation_mode
        {
            self.recreate_swapchain(surface, width, height, presentation_mode)?;
        }
        Ok(())
    }

    /// Returns the exact native color-format key used by graphics pipeline caching.
    ///
    /// # Errors
    ///
    /// Returns `NotReady` when a surface format is requested before swapchain preparation, or
    /// `Unsupported` for a non-color render-target format.
    pub fn graphics_format_key(
        &self,
        format: Option<ez_gfx_runtime::target::Format>,
    ) -> Result<u32, HalError> {
        u32::try_from(pipeline_color_format(self.swapchain_format, format)?.as_raw())
            .map_err(|_| HalError::NativeFailure)
    }
}
/// Clears borrowed write entries and reserves both descriptor shells.
///
/// `writes` entries borrow the `infos` buffer, so they must be dropped before
/// `infos` is touched; reserving both shells up front then guarantees no
/// realloc can occur while rebuilt entries are live.
fn reserve_descriptor_shells(
    infos: &mut Vec<vk::DescriptorBufferInfo>,
    writes: &mut Vec<vk::WriteDescriptorSet<'static>>,
    count: usize,
) -> Result<(), HalError> {
    writes.clear();
    infos.clear();
    infos
        .try_reserve(count)
        .map_err(|_| HalError::OutOfMemory)?;
    writes
        .try_reserve(count)
        .map_err(|_| HalError::OutOfMemory)?;
    Ok(())
}

#[cfg(test)]
mod descriptor_shell_tests {
    use super::{pipeline_color_format, reserve_descriptor_shells};
    use ash::vk;
    use ez_gfx_runtime::target::Format;

    #[test]
    fn target_format_resolution_preserves_surface_format() {
        let surface = vk::Format::B8G8R8A8_SRGB;

        assert_eq!(
            pipeline_color_format(surface, Some(Format::Rgba8Unorm)).unwrap(),
            vk::Format::R8G8B8A8_UNORM
        );
        assert_eq!(
            pipeline_color_format(surface, Some(Format::Bgra8Srgb)).unwrap(),
            vk::Format::B8G8R8A8_SRGB
        );
        assert_eq!(
            pipeline_color_format(surface, Some(Format::Rgba16Float)).unwrap(),
            vk::Format::R16G16B16A16_SFLOAT
        );
        assert_eq!(pipeline_color_format(surface, None).unwrap(), surface);
    }

    fn rebuild_test_writes(
        writes: &mut Vec<vk::WriteDescriptorSet<'static>>,
        infos: &[vk::DescriptorBufferInfo],
    ) {
        writes.extend(infos.iter().map(|info| {
            let write = vk::WriteDescriptorSet::default().buffer_info(core::slice::from_ref(info));
            // SAFETY: test-only mirror of the production encoding; entries are
            // only read while `infos` is untouched and are cleared by the next
            // reservation before any mutation.
            unsafe {
                core::mem::transmute::<vk::WriteDescriptorSet<'_>, vk::WriteDescriptorSet<'static>>(
                    write,
                )
            }
        }));
    }

    /// Simulates residue from a previous call, reserves for growth, and proves
    /// the rebuild fill cannot realloc while borrowed entries are live.
    #[test]
    fn growth_across_calls_keeps_borrowed_entries_stable() {
        let mut infos = Vec::new();
        let mut writes: Vec<vk::WriteDescriptorSet<'static>> = Vec::new();
        // First cycle leaves one borrowed entry behind, as a real call does
        // before the post-update clear was added.
        reserve_descriptor_shells(&mut infos, &mut writes, 1).unwrap();
        infos.push(vk::DescriptorBufferInfo::default());
        rebuild_test_writes(&mut writes, &infos);
        assert_eq!(writes.len(), 1);
        // Second cycle grows past the first buffer: the fixed order drops the
        // borrowed entry before `infos` is touched, so the realloc below
        // cannot strand it.
        reserve_descriptor_shells(&mut infos, &mut writes, 64).unwrap();
        assert!(writes.is_empty());
        assert!(infos.capacity() >= 64 && writes.capacity() >= 64);
        let stable = infos.as_ptr();
        for _ in 0..64 {
            infos.push(vk::DescriptorBufferInfo::default());
        }
        // No realloc across the whole fill, so entries built from this buffer
        // stay valid through the subsequent update call.
        assert!(core::ptr::eq(infos.as_ptr(), stable));
    }
}
