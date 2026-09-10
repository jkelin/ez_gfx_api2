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
    D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, D3D12_LOGIC_OP_NOOP,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE, D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER,
    D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
    D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL, D3D12_STENCIL_OP_KEEP,
    D3D12SerializeRootSignature, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC, DeferredResource, DynamicPipelineState, FrontFace,
    HalError, ID3D12RootSignature, ID3DBlob, NativeContext, NativePipeline, NativeShader,
    PrimitiveTopology, ShaderBufferLayout, TEXTURE_DESCRIPTOR_CAPACITY, map_windows, ptr,
};

impl NativeContext {
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
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
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
        })
    }

    /// Defers pipeline destruction until every referencing frame completes.
    pub fn destroy_pipeline(&mut self, pipeline: NativePipeline) {
        let _ = self.defer_resource(DeferredResource::Pipeline(pipeline));
    }
    /// Releases a shader product; DXIL is copied during pipeline creation.
    pub fn destroy_shader(&self, _shader: NativeShader) {}
}
