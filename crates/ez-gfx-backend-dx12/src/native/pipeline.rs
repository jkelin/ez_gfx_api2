use super::{
    BlendMode, CullMode, D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC,
    D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_BLEND_SRC_ALPHA,
    D3D12_BLEND_ZERO, D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COMMAND_SIGNATURE_DESC,
    D3D12_COMPARISON_FUNC_ALWAYS, D3D12_COMPARISON_FUNC_LESS, D3D12_COMPUTE_PIPELINE_STATE_DESC,
    D3D12_CULL_MODE_BACK, D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_NONE, D3D12_DEPTH_STENCIL_DESC,
    D3D12_DEPTH_STENCILOP_DESC, D3D12_DEPTH_WRITE_MASK_ALL, D3D12_DESCRIPTOR_RANGE,
    D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND, D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
    D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_FILL_MODE_SOLID, D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_DESC_0,
    D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, D3D12_LOGIC_OP_NOOP, D3D12_MESH_SHADER_TIER_1,
    D3D12_PIPELINE_STATE_STREAM_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
    D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC, D3D12_ROOT_DESCRIPTOR,
    D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER_TYPE_SRV,
    D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_RT_FORMAT_ARRAY, D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
    D3D12_STENCIL_OP_KEEP, D3D12SerializeRootSignature, DXGI_FORMAT, DXGI_FORMAT_D32_FLOAT,
    DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC, DeferredResource, DynamicPipelineState, FrontFace,
    HalError, ID3D12Device2, ID3D12PipelineState, ID3D12RootSignature, ID3DBlob, Interface,
    MeshDispatchLimits, MeshPipelineState, NativeContext, NativeMeshPipelineDesc, NativePipeline,
    NativeShader, PrimitiveTopology, ShaderBufferLayout, TEXTURE_DESCRIPTOR_CAPACITY, map_windows,
    ptr, retained_mesh_dispatch_limits, validate_mesh_dispatch,
};

use ez_gfx_hal::MeshDispatchError;

/// One pipeline-state-stream subobject: a type tag followed by its payload.
///
/// `align(8)` upholds the D3D12 stream layout rule that every subobject starts at
/// pointer granularity; without it a preceding odd-sized payload would misalign
/// the following tag.
#[repr(C, align(8))]
struct StreamSubobject<T> {
    subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    payload: T,
}

/// Mesh pipeline-state-stream tail shared by the task and taskless variants.
///
/// Nesting preserves the 8-byte granularity: every subobject is 8-aligned, so a
/// nested block starting on an 8-byte boundary keeps all inner tags aligned.
#[repr(C)]
struct MeshPipelineTail {
    mesh: StreamSubobject<D3D12_SHADER_BYTECODE>,
    pixel: StreamSubobject<D3D12_SHADER_BYTECODE>,
    blend: StreamSubobject<D3D12_BLEND_DESC>,
    rasterizer: StreamSubobject<D3D12_RASTERIZER_DESC>,
    depth_stencil: StreamSubobject<D3D12_DEPTH_STENCIL_DESC>,
    render_targets: StreamSubobject<D3D12_RT_FORMAT_ARRAY>,
    depth_stencil_format: StreamSubobject<DXGI_FORMAT>,
    sample_desc: StreamSubobject<DXGI_SAMPLE_DESC>,
    sample_mask: StreamSubobject<u32>,
}

/// Taskless mesh pipeline-state stream: root signature plus the shared tail.
#[repr(C)]
struct MeshPipelineStream {
    root: StreamSubobject<Option<ID3D12RootSignature>>,
    tail: MeshPipelineTail,
}

/// Task mesh pipeline-state stream: root, amplification stage, shared tail.
///
/// A present-but-empty amplification subobject is not valid D3D12, so the task
/// stage is omitted structurally instead of zeroed.
#[repr(C)]
struct TaskMeshPipelineStream {
    root: StreamSubobject<Option<ID3D12RootSignature>>,
    amplification: StreamSubobject<D3D12_SHADER_BYTECODE>,
    tail: MeshPipelineTail,
}

/// Returns one DXIL product slice by index.
///
/// # Errors
///
/// Returns [`HalError::InvalidArgument`] when the product index is out of range.
fn mesh_product(shader: &NativeShader, index: usize) -> Result<&[u8], HalError> {
    shader
        .products
        .get(index)
        .map(Vec::as_slice)
        .ok_or(HalError::InvalidArgument)
}

/// Builds the shared blend, rasterizer, and depth-stencil state for a mesh stream.
///
/// Mirrors the indexed graphics path: one sRGB render target, less-depth, and
/// stencil kept off.
///
/// # Errors
///
/// Returns [`HalError::NativeFailure`] when the color-write mask does not fit in `u8`.
fn mesh_render_state(
    state: MeshPipelineState,
    depth_required: bool,
) -> Result<
    (
        D3D12_BLEND_DESC,
        D3D12_RASTERIZER_DESC,
        D3D12_DEPTH_STENCIL_DESC,
    ),
    HalError,
> {
    let target = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: (state.blend == BlendMode::Alpha).into(),
        LogicOpEnable: false.into(),
        SrcBlend: if state.blend == BlendMode::Alpha {
            D3D12_BLEND_SRC_ALPHA
        } else {
            D3D12_BLEND_ONE
        },
        DestBlend: if state.blend == BlendMode::Alpha {
            D3D12_BLEND_INV_SRC_ALPHA
        } else {
            D3D12_BLEND_ZERO
        },
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: if state.blend == BlendMode::Alpha {
            D3D12_BLEND_INV_SRC_ALPHA
        } else {
            D3D12_BLEND_ZERO
        },
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: u8::try_from(D3D12_COLOR_WRITE_ENABLE_ALL.0)
            .map_err(|_| HalError::NativeFailure)?,
    };
    let blend = D3D12_BLEND_DESC {
        AlphaToCoverageEnable: false.into(),
        IndependentBlendEnable: false.into(),
        RenderTarget: [target; 8],
    };
    let raster = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: match state.cull {
            CullMode::None => D3D12_CULL_MODE_NONE,
            CullMode::Front => D3D12_CULL_MODE_FRONT,
            CullMode::Back => D3D12_CULL_MODE_BACK,
        },
        FrontCounterClockwise: (state.front_face == FrontFace::CounterClockwise).into(),
        DepthClipEnable: true.into(),
        ..Default::default()
    };
    let stencil = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: depth_required.into(),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS,
        StencilEnable: false.into(),
        StencilReadMask: u8::MAX,
        StencilWriteMask: u8::MAX,
        FrontFace: stencil,
        BackFace: stencil,
    };
    Ok((blend, raster, depth_stencil))
}

impl NativeContext {
    // `None` is the swapchain format; depth, compressed, and storage formats are invalid here.
    fn render_pipeline_color_format(
        format: Option<ez_gfx_runtime::target::Format>,
    ) -> Result<DXGI_FORMAT, HalError> {
        use ez_gfx_runtime::target::Format;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R16G16B16A16_FLOAT,
        };
        match format {
            None => Ok(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB),
            Some(Format::Rgba8Unorm) => Ok(DXGI_FORMAT_R8G8B8A8_UNORM),
            Some(Format::Bgra8Srgb) => Ok(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB),
            Some(Format::Rgba16Float) => Ok(DXGI_FORMAT_R16G16B16A16_FLOAT),
            Some(_) => Err(HalError::Unsupported),
        }
    }

    /// DXIL products remain owned until PSO creation; empty products are rejected at admission.
    ///
    /// # Errors
    ///
    /// Returns an error if no shader products are provided or any product is empty.
    pub fn create_shader(&self, products: &[&[u8]]) -> Result<NativeShader, HalError> {
        if products.is_empty() || products.iter().any(|product| product.is_empty()) {
            return Err(HalError::InvalidArgument);
        }
        Ok(NativeShader {
            products: products.iter().map(|product| product.to_vec()).collect(),
        })
    }

    /// Reflected buffers use root SRV/UAVs; separate texture and sampler tables occupy the remaining roots.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid or overflowing layout values, root-signature serialization failure or a missing serialized blob, or device root-signature creation failure.
    pub(super) fn create_root_signature(
        &self,
        layouts: &[ShaderBufferLayout],
        graphics: bool,
    ) -> Result<(ID3D12RootSignature, Vec<bool>), HalError> {
        let descriptor_count = layouts
            .iter()
            .try_fold(0usize, |total, layout| {
                total.checked_add(layout.descriptor_count as usize)
            })
            .ok_or(HalError::InvalidArgument)?;
        let texture_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: TEXTURE_DESCRIPTOR_CAPACITY,
            BaseShaderRegister: 0,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };
        let sampler_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
            NumDescriptors: TEXTURE_DESCRIPTOR_CAPACITY,
            BaseShaderRegister: 0,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };
        let mut parameters = Vec::with_capacity(descriptor_count + 2);
        let mut writable = Vec::with_capacity(descriptor_count);
        for layout in layouts {
            if layout.descriptor_count == 0 || layout.descriptor_count > 2 {
                return Err(HalError::InvalidArgument);
            }
            for offset in 0..layout.descriptor_count {
                parameters.push(D3D12_ROOT_PARAMETER {
                    ParameterType: if layout.writable {
                        D3D12_ROOT_PARAMETER_TYPE_UAV
                    } else {
                        D3D12_ROOT_PARAMETER_TYPE_SRV
                    },
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: layout
                                .binding
                                .checked_add(offset)
                                .ok_or(HalError::InvalidArgument)?,
                            RegisterSpace: layout.space,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                });
                writable.push(layout.writable);
            }
        }
        parameters.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &raw const texture_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        parameters.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &raw const sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        let flags = if graphics {
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
        } else {
            D3D12_ROOT_SIGNATURE_FLAG_NONE
        };
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: u32::try_from(parameters.len())
                .map_err(|_| HalError::InvalidArgument)?,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: ptr::null(),
            Flags: flags,
        };
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: `desc.pParameters` spans `parameters.len()` initialized entries, its descriptor-table pointers reference `texture_range` and `sampler_range`, and `desc` plus the `blob` output storage remain allocated for `D3D12SerializeRootSignature`.
        unsafe {
            D3D12SerializeRootSignature(
                &raw const desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &raw mut blob,
                None,
            )
        }
        .map_err(map_windows)?;
        let blob = blob.ok_or(HalError::NativeFailure)?;
        // SAFETY: `blob.GetBufferPointer()` addresses `blob.GetBufferSize()` bytes of blob-owned storage, and `blob` outlives the resulting `serialized` slice.
        let serialized = unsafe {
            core::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        let root =
            // SAFETY: `serialized` references the complete serialized root-signature bytes in `blob` for the duration of `ID3D12Device::CreateRootSignature`.
            unsafe { self.device.CreateRootSignature(0, serialized) }.map_err(map_windows)?;
        Ok((root, writable))
    }

    /// Creates a compute pipeline from one compiled shader entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the product index is invalid, root-signature creation fails, or the device cannot create the compute pipeline state.
    pub fn create_compute_pipeline(
        &self,
        shader: &NativeShader,
        product_index: usize,
        layouts: &[ShaderBufferLayout],
    ) -> Result<NativePipeline, HalError> {
        let bytes = shader
            .products
            .get(product_index)
            .ok_or(HalError::InvalidArgument)?;
        let (root, buffer_writable) = self.create_root_signature(layouts, false)?;
        let state_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: core::mem::ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        // SAFETY: `state_desc` remains allocated for `CreateComputePipelineState`, and its root-signature clone and `bytes` shader storage remain available for the duration of the call.
        let state = unsafe {
            self.device
                .CreateComputePipelineState(&raw const state_desc)
        }
        .map_err(map_windows)?;
        Ok(NativePipeline {
            state,
            root,
            topology: None,
            signature: None,
            buffer_writable,
            mesh: false,
            task_stage: false,
        })
    }

    /// Creates a graphics pipeline from the requested shader entries and render state.
    ///
    /// # Errors
    ///
    /// Returns an error if either shader index is invalid, root-signature creation fails, triangle-fan topology is requested, the color-write mask does not fit in `u8`, or the device cannot create the graphics pipeline state or command signature.
    #[allow(
        clippy::too_many_arguments,
        reason = "the backend boundary receives two explicit shaders, their product indices, and pipeline state"
    )]
    pub fn create_graphics_pipeline(
        &self,
        vertex_shader: &NativeShader,
        fragment_shader: &NativeShader,
        vertex_index: usize,
        fragment_index: usize,
        state: DynamicPipelineState,
        depth_required: bool,
        color_format: Option<ez_gfx_runtime::target::Format>,
        layouts: &[ShaderBufferLayout],
    ) -> Result<NativePipeline, HalError> {
        let vertex = vertex_shader
            .products
            .get(vertex_index)
            .ok_or(HalError::InvalidArgument)?;
        let fragment = fragment_shader
            .products
            .get(fragment_index)
            .ok_or(HalError::InvalidArgument)?;
        let (root, buffer_writable) = self.create_root_signature(layouts, true)?;
        let (topology, topology_type) = match state.topology {
            PrimitiveTopology::TriangleList => (
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
            PrimitiveTopology::PointList => (
                D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
            ),
            PrimitiveTopology::LineList => (
                D3D_PRIMITIVE_TOPOLOGY_LINELIST,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            ),
            PrimitiveTopology::LineStrip => (
                D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            ),
            PrimitiveTopology::TriangleStrip => (
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
            PrimitiveTopology::TriangleFan => return Err(HalError::Unsupported),
        };
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: (state.blend == BlendMode::Alpha).into(),
            LogicOpEnable: false.into(),
            SrcBlend: if state.blend == BlendMode::Alpha {
                D3D12_BLEND_SRC_ALPHA
            } else {
                D3D12_BLEND_ONE
            },
            DestBlend: if state.blend == BlendMode::Alpha {
                D3D12_BLEND_INV_SRC_ALPHA
            } else {
                D3D12_BLEND_ZERO
            },
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: if state.blend == BlendMode::Alpha {
                D3D12_BLEND_INV_SRC_ALPHA
            } else {
                D3D12_BLEND_ZERO
            },
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: u8::try_from(D3D12_COLOR_WRITE_ENABLE_ALL.0)
                .map_err(|_| HalError::NativeFailure)?,
        };
        let blend = D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [target; 8],
        };
        let raster = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: match state.cull {
                CullMode::None => D3D12_CULL_MODE_NONE,
                CullMode::Front => D3D12_CULL_MODE_FRONT,
                CullMode::Back => D3D12_CULL_MODE_BACK,
            },
            FrontCounterClockwise: (state.front_face == FrontFace::CounterClockwise).into(),
            DepthClipEnable: true.into(),
            ..Default::default()
        };
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: depth_required.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            StencilReadMask: u8::MAX,
            StencilWriteMask: u8::MAX,
            FrontFace: stencil,
            BackFace: stencil,
        };
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = Self::render_pipeline_color_format(color_format)?;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: core::mem::ManuallyDrop::new(Some(root.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vertex.as_ptr().cast(),
                BytecodeLength: vertex.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: fragment.as_ptr().cast(),
                BytecodeLength: fragment.len(),
            },
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: raster,
            DepthStencilState: depth_stencil,
            PrimitiveTopologyType: topology_type,
            NumRenderTargets: 1,
            RTVFormats: formats,
            DSVFormat: if depth_required {
                DXGI_FORMAT_D32_FLOAT
            } else {
                DXGI_FORMAT_UNKNOWN
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let pipeline =
            // SAFETY: `desc` remains allocated for `CreateGraphicsPipelineState`, and its root-signature clone plus `vertex` and `fragment` shader storage remain available for the duration of the call.
            unsafe { self.device.CreateGraphicsPipelineState(&raw const desc) }.map_err(map_windows)?;
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
        };
        let signature_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 20,
            NumArgumentDescs: 1,
            pArgumentDescs: &raw const argument,
            NodeMask: 0,
        };
        let mut signature = None;
        // SAFETY: `signature_desc` and its single `argument` descriptor remain allocated for `CreateCommandSignature`, and `signature` is initialized output storage for the returned `ID3D12CommandSignature`.
        unsafe {
            self.device
                .CreateCommandSignature(&raw const signature_desc, None, &raw mut signature)
        }
        .map_err(map_windows)?;
        Ok(NativePipeline {
            state: pipeline,
            root,
            topology: Some(topology),
            signature,
            buffer_writable,
            mesh: false,
            task_stage: false,
        })
    }

    /// Returns the shared mesh dispatch limits.
    ///
    /// Tier 1 enables amplification and mesh stages together, so both stage
    /// selections share one gate; the parameter keeps the cross-backend seam
    /// identical to Vulkan's. Thread and total grid ceilings are the specified
    /// mesh-shader constants.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::Unsupported`] when the cached tier is below tier 1.
    pub fn mesh_dispatch_limits(&self, _has_task: bool) -> Result<MeshDispatchLimits, HalError> {
        if self.mesh_shader_tier.0 < D3D12_MESH_SHADER_TIER_1.0 {
            return Err(HalError::Unsupported);
        }
        Ok(retained_mesh_dispatch_limits())
    }

    /// Creates a mesh pipeline from the requested task/mesh/fragment DXIL entries.
    ///
    /// The cached mesh-shader tier gates before any state creation: below tier 1
    /// neither the amplification/mesh/pixel stream nor `DispatchMesh` exists.
    /// Reflected workgroup sizes run through the shared dispatch validation with a
    /// unit grid standing in for the grid the later dispatch supplies. The stream
    /// carries the optional amplification stage, the required mesh and pixel
    /// stages, and the existing blend, rasterizer, depth, render-target, and root
    /// state, but no input layout, primitive topology, or indirect signature,
    /// which mesh execution never uses.
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` when the cached tier is below tier 1, the device
    /// cannot QI the pipeline-state-stream device interface, or a well-formed
    /// reflected workgroup size exceeds the specified ceilings; `InvalidArgument`
    /// for an inconsistent stage selection, an invalid product index, or a
    /// malformed (zero or overflowing) workgroup size; or a mapped Windows error
    /// when root-signature or pipeline-state creation fails.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the public backend request is consumed by value across all adapters"
    )]
    pub fn create_mesh_pipeline(
        &self,
        desc: NativeMeshPipelineDesc<'_>,
    ) -> Result<NativePipeline, HalError> {
        let has_task = desc.task.is_some();
        // Malformed stage/size selection fails before any capability probe.
        if desc.task_workgroup_size.is_some() != has_task {
            return Err(HalError::InvalidArgument);
        }
        let task_bytes = desc
            .task
            .map(|(shader, index)| mesh_product(shader, index))
            .transpose()?;
        let mesh_bytes = mesh_product(desc.mesh.0, desc.mesh.1)?;
        let fragment_bytes = mesh_product(desc.fragment.0, desc.fragment.1)?;
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
        // Same root layout as the indexed path: reflected buffers plus texture
        // and sampler tables. The input-assembler flag is inert here because the
        // stream carries no input-layout subobject.
        let (root, buffer_writable) = self.create_root_signature(desc.layouts, true)?;
        let shader = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let (blend, raster, depth_stencil) = mesh_render_state(desc.state, desc.depth_required)?;
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = Self::render_pipeline_color_format(desc.color_format)?;
        let tail = MeshPipelineTail {
            mesh: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
                payload: shader(mesh_bytes),
            },
            pixel: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
                payload: shader(fragment_bytes),
            },
            blend: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
                payload: blend,
            },
            rasterizer: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
                payload: raster,
            },
            depth_stencil: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
                payload: depth_stencil,
            },
            render_targets: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
                payload: D3D12_RT_FORMAT_ARRAY {
                    RTFormats: formats,
                    NumRenderTargets: 1,
                },
            },
            depth_stencil_format: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
                payload: if desc.depth_required {
                    DXGI_FORMAT_D32_FLOAT
                } else {
                    DXGI_FORMAT_UNKNOWN
                },
            },
            sample_desc: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
                payload: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
            },
            sample_mask: StreamSubobject {
                subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
                payload: u32::MAX,
            },
        };
        let root_subobject = StreamSubobject {
            subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            payload: Some(root.clone()),
        };
        // The stream device interface arrives through QI; absence means the
        // runtime predates pipeline-state streams and cannot host mesh state.
        let device: ID3D12Device2 = self.device.cast().map_err(|_| HalError::Unsupported)?;
        let state: ID3D12PipelineState = if let Some(task_bytes) = task_bytes {
            let stream = TaskMeshPipelineStream {
                root: root_subobject,
                amplification: StreamSubobject {
                    subobject_type: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
                    payload: shader(task_bytes),
                },
                tail,
            };
            let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
                SizeInBytes: core::mem::size_of_val(&stream),
                pPipelineStateSubobjectStream: (&raw const stream).cast_mut().cast(),
            };
            // SAFETY: `stream` and `desc` remain allocated for `CreatePipelineState`, its root-signature clone and DXIL byte storage outlive the call, and every subobject starts at 8-byte granularity by construction.
            unsafe { device.CreatePipelineState(&raw const desc) }.map_err(map_windows)?
        } else {
            let stream = MeshPipelineStream {
                root: root_subobject,
                tail,
            };
            let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
                SizeInBytes: core::mem::size_of_val(&stream),
                pPipelineStateSubobjectStream: (&raw const stream).cast_mut().cast(),
            };
            // SAFETY: `stream` and `desc` remain allocated for `CreatePipelineState`, its root-signature clone and DXIL byte storage outlive the call, and every subobject starts at 8-byte granularity by construction.
            unsafe { device.CreatePipelineState(&raw const desc) }.map_err(map_windows)?
        };
        Ok(NativePipeline {
            state,
            root,
            topology: None,
            signature: None,
            buffer_writable,
            mesh: true,
            task_stage: has_task,
        })
    }

    /// Defers pipeline destruction until every referencing frame completes.
    pub fn destroy_pipeline(&mut self, pipeline: NativePipeline) {
        let _ = self.defer_resource(DeferredResource::Pipeline(pipeline));
    }
    /// Releases a shader product; DXIL is copied during pipeline creation.
    pub fn destroy_shader(&self, _shader: NativeShader) {}
}

#[cfg(test)]
mod mesh_stream_tests {
    use super::{MeshPipelineStream, StreamSubobject, TaskMeshPipelineStream};
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn stream_subobjects_start_at_pointer_granularity() {
        // D3D12 requires every stream subobject to start at pointer granularity;
        // the aligned wrapper upholds this for any payload, including odd-sized
        // ones such as the render-target format array.
        assert_eq!(align_of::<StreamSubobject<u32>>(), 8);
        assert_eq!(
            align_of::<StreamSubobject<super::D3D12_RT_FORMAT_ARRAY>>(),
            8
        );
        assert_eq!(size_of::<MeshPipelineStream>() % 8, 0);
        assert_eq!(size_of::<TaskMeshPipelineStream>() % 8, 0);
        assert_eq!(offset_of!(MeshPipelineStream, root) % 8, 0);
        assert_eq!(offset_of!(MeshPipelineStream, tail) % 8, 0);
        assert_eq!(offset_of!(TaskMeshPipelineStream, root) % 8, 0);
        assert_eq!(offset_of!(TaskMeshPipelineStream, amplification) % 8, 0);
        assert_eq!(offset_of!(TaskMeshPipelineStream, tail) % 8, 0);
    }
}
