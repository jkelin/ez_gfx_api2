use ez_gfx_core::{Backend, capability::MAX_BINDLESS_SAMPLED_TEXTURES};

pub const BACKEND: Backend = Backend::Dx12;
pub const SUPPORTED_ON_TARGET: bool = cfg!(windows);
pub const TEXTURE_DESCRIPTOR_CAPACITY: u32 = MAX_BINDLESS_SAMPLED_TEXTURES;

#[cfg(windows)]
pub mod native {
    use crate::{BACKEND, TEXTURE_DESCRIPTOR_CAPACITY};
    use core::{ffi::c_void, ptr};

    use ez_gfx_core::capability::{
        AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport, SemanticProfile,
    };
    use ez_gfx_hal::{
        AllocationError, AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BlendMode,
        BufferTransfer, CompletionToken, CullMode, DynamicPipelineState, ExecutionBarrier,
        ExecutionPass, FrontFace, HalError, ImageMip, MemoryAllocator, MemoryClass,
        PrimitiveTopology, QueueKind, ResourceAccess, SamplerAddressMode, SamplerFilter,
        ShaderBufferLayout, TextureSamplerDesc, validate_rgba8_mips,
    };
    use gpu_allocator::{
        MemoryLocation,
        d3d12::{
            Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc, ID3D12DeviceVersion,
        },
    };
    use windows::Win32::Graphics::Direct3D::{
        D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    };
    use windows::Win32::Graphics::Direct3D12::{
        D3D_ROOT_SIGNATURE_VERSION_1, D3D_SHADER_MODEL_6_5, D3D12_BLEND_DESC,
        D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_BLEND_SRC_ALPHA,
        D3D12_BLEND_ZERO, D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0,
        D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COMMAND_SIGNATURE_DESC, D3D12_COMPARISON_FUNC_ALWAYS,
        D3D12_COMPARISON_FUNC_LESS, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_CULL_MODE_BACK,
        D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_NONE, D3D12_DEPTH_STENCIL_DESC,
        D3D12_DEPTH_STENCIL_VALUE, D3D12_DEPTH_STENCILOP_DESC, D3D12_DEPTH_WRITE_MASK_ALL,
        D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_RANGE,
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND, D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_FEATURE_DATA_SHADER_MODEL,
        D3D12_FEATURE_SHADER_MODEL, D3D12_FILL_MODE_SOLID, D3D12_FILTER_ANISOTROPIC,
        D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT,
        D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_INDEX_BUFFER_VIEW, D3D12_INDIRECT_ARGUMENT_DESC,
        D3D12_INDIRECT_ARGUMENT_DESC_0, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
        D3D12_LOGIC_OP_NOOP, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC, D3D12_RESOURCE_BARRIER,
        D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_READ,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_INDEX_BUFFER,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER,
        D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_DESCRIPTOR_TABLE,
        D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER_TYPE_SRV,
        D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SAMPLER_DESC, D3D12_SHADER_BYTECODE,
        D3D12_SHADER_VISIBILITY_ALL, D3D12_STENCIL_OP_KEEP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_COPY_LOCATION,
        D3D12_TEXTURE_COPY_LOCATION_0, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_VIEWPORT,
        D3D12SerializeRootSignature, ID3D12CommandSignature, ID3D12DescriptorHeap,
        ID3D12PipelineState, ID3D12RootSignature,
    };
    use windows::{
        Win32::{
            Foundation::{CloseHandle, HANDLE, HWND, RECT},
            Graphics::{
                Direct3D::{D3D_FEATURE_LEVEL_12_1, ID3DBlob},
                Direct3D12::{
                    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
                    D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT, D3D12_FEATURE_D3D12_OPTIONS,
                    D3D12_FEATURE_DATA_D3D12_OPTIONS, D3D12_FENCE_FLAG_NONE, D3D12_RANGE,
                    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandList,
                    ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
                    ID3D12Resource,
                },
                Dxgi::{
                    Common::{
                        DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
                        DXGI_FORMAT_R32_UINT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
                    },
                    CreateDXGIFactory1, DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_ERROR_NOT_FOUND,
                    DXGI_ERROR_UNSUPPORTED, DXGI_PRESENT, DXGI_SCALING_STRETCH,
                    DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_EFFECT_FLIP_DISCARD,
                    DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIAdapter4, IDXGIFactory4, IDXGISwapChain4,
                },
            },
            System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
        },
        core::Interface,
    };

    pub struct NativeAllocation {
        resource: ID3D12Resource,
        allocation: Allocation,
        mapped_address: usize,
    }

    pub struct NativeBufferBinding<'a> {
        pub allocation: &'a NativeAllocation,
        pub offset: u64,
        pub writable: bool,
    }

    pub struct NativeDrawIndexed<'a> {
        pub width: u32,
        pub height: u32,
        pub pipeline: &'a NativePipeline,
        pub index_buffer: &'a NativeAllocation,
        pub indirect_buffer: &'a NativeAllocation,
        pub draw_count: u32,
        pub push_constants: &'a [u8],
        pub bindings: &'a [NativeBufferBinding<'a>],
    }

    pub struct NativeComputeDispatch<'a> {
        pub pipeline: &'a NativePipeline,
        pub groups: [u32; 3],
        pub push_constants: &'a [u8],
        pub bindings: &'a [NativeBufferBinding<'a>],
    }

    pub enum NativeFrameResource<'a> {
        Buffer(&'a NativeAllocation),
        Texture(&'a NativeTexture),
        Surface,
        Depth,
    }

    pub enum NativeFrameAction<'a> {
        Wait(CompletionToken),
        Barrier {
            barrier: ExecutionBarrier,
            resource: NativeFrameResource<'a>,
        },
        BeginPass(&'a ExecutionPass),
        Compute(NativeComputeDispatch<'a>),
        Graphics(NativeDrawIndexed<'a>),
        TextureReadback {
            texture: &'a NativeTexture,
            width: u32,
            height: u32,
        },
        EndPass,
        Present,
    }

    pub struct NativeShader {
        products: Vec<Vec<u8>>,
    }

    impl NativeShader {
        pub fn products(&self) -> impl ExactSizeIterator<Item = &[u8]> {
            self.products.iter().map(Vec::as_slice)
        }
    }

    pub struct NativePipeline {
        state: ID3D12PipelineState,
        root: ID3D12RootSignature,
        topology: Option<D3D_PRIMITIVE_TOPOLOGY>,
        signature: Option<ID3D12CommandSignature>,
        buffer_writable: Vec<bool>,
    }

    pub struct NativeTexture {
        resource: ID3D12Resource,
        allocation: Allocation,
        pub binding: u32,
    }

    struct RetiredAllocation {
        allocation: NativeAllocation,
        completion: CompletionToken,
    }

    struct PendingCopy {
        completion: CompletionToken,
        _allocator: ID3D12CommandAllocator,
        _list: ID3D12GraphicsCommandList,
    }

    struct SurfaceDepth {
        resource: ID3D12Resource,
        allocation: Allocation,
        heap: ID3D12DescriptorHeap,
    }

    pub struct NativeSurface {
        window: usize,
        swapchain: Option<IDXGISwapChain4>,
        buffers: Vec<ID3D12Resource>,
        rtv_heap: Option<ID3D12DescriptorHeap>,
        width: u32,
        height: u32,
        presented: Vec<u8>,
        depth: Option<SurfaceDepth>,
    }

    impl NativeSurface {
        /// The HWND is borrowed and never destroyed by the graphics context.
        pub fn new(window: *mut c_void) -> Result<Self, HalError> {
            if window.is_null() {
                return Err(HalError::InvalidArgument);
            }
            Ok(Self {
                window: window as usize,
                swapchain: None,
                width: 0,
                height: 0,
                buffers: Vec::new(),
                rtv_heap: None,
                presented: Vec::new(),
                depth: None,
            })
        }

        pub const fn window(&self) -> usize {
            self.window
        }

        pub fn presented_rgba8(&self) -> &[u8] {
            &self.presented
        }
    }

    pub struct NativeContext {
        pub adapter: IDXGIAdapter4,
        pub device: ID3D12Device,
        queue: ID3D12CommandQueue,
        fence: ID3D12Fence,
        fence_event: HANDLE,
        next_fence: u64,
        allocator: Option<Allocator>,
        retired: Vec<RetiredAllocation>,
        pending_copies: Vec<PendingCopy>,
        adapter_info: AdapterInfo,
        descriptors: ID3D12DescriptorHeap,
        descriptor_stride: u32,
        samplers: ID3D12DescriptorHeap,
        sampler_stride: u32,
    }

    // SAFETY: D3D12/DXGI interfaces are agile and the event handle is process-wide; higher layers
    // serialize mutation and enforce frame-recording thread affinity.
    unsafe impl Send for NativeContext {}
    impl NativeContext {
        /// Enumeration exhaustion reports `DXGI_ERROR_UNSUPPORTED`; software adapters are ignored unless explicitly allowed.
        pub fn create_default(allow_software: bool) -> windows::core::Result<Self> {
            let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }?;
            let mut index = 0;
            loop {
                let adapter = match unsafe { factory.EnumAdapters1(index) } {
                    Ok(adapter) => adapter.cast::<IDXGIAdapter4>()?,
                    Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => {
                        return Err(windows::core::Error::new(
                            DXGI_ERROR_UNSUPPORTED,
                            "no D3D12 adapter meets the semantic capability floor",
                        ));
                    }
                    Err(error) => return Err(error),
                };
                index += 1;
                let description = unsafe { adapter.GetDesc3() }?;
                if !allow_software && description.Flags.contains(DXGI_ADAPTER_FLAG3_SOFTWARE) {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_1, &mut device) }
                    .is_err()
                {
                    continue;
                }
                let Some(device) = device else {
                    continue;
                };
                let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
                if unsafe {
                    device.CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS,
                        (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS).cast(),
                        core::mem::size_of_val(&options) as u32,
                    )
                }
                .is_err()
                {
                    continue;
                }
                let mut shader_model = D3D12_FEATURE_DATA_SHADER_MODEL {
                    HighestShaderModel: D3D_SHADER_MODEL_6_5,
                };
                if unsafe {
                    device.CheckFeatureSupport(
                        D3D12_FEATURE_SHADER_MODEL,
                        (&mut shader_model as *mut D3D12_FEATURE_DATA_SHADER_MODEL).cast(),
                        core::mem::size_of_val(&shader_model) as u32,
                    )
                }
                .is_err()
                {
                    continue;
                }
                let capabilities = AdapterCapabilities {
                    bindless_sampled_textures: if options.ResourceBindingTier.0 >= 3 {
                        TEXTURE_DESCRIPTOR_CAPACITY
                    } else {
                        0
                    },
                    bindless_storage_resources: if options.ResourceBindingTier.0 >= 3 {
                        1_000_000
                    } else {
                        0
                    },
                    bindless_samplers: if options.ResourceBindingTier.0 >= 3 {
                        TEXTURE_DESCRIPTOR_CAPACITY
                    } else {
                        0
                    },
                    max_indirect_draw_count: u32::MAX,
                    shader_model: u32::try_from(shader_model.HighestShaderModel.0)
                        .map(|value| ((value >> 4) << 8) | (value & 0x0f))
                        .unwrap_or(0),
                    timeline_synchronization: true,
                    resource_aliasing: true,
                    dynamic_rendering: true,
                    presentation: true,
                    compression: CompressionSupport::BC,
                };
                if SemanticProfile::V1.admit(&capabilities).is_err() {
                    continue;
                }
                let stable_id = adapter_id(&description);
                let name = String::from_utf16_lossy(&description.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let driver = format!(
                    "luid-{:08x}-{:08x}",
                    description.AdapterLuid.HighPart, description.AdapterLuid.LowPart
                );
                let class = if description.DedicatedVideoMemory > 0 {
                    AdapterClass::Discrete
                } else {
                    AdapterClass::Integrated
                };
                let adapter_info =
                    AdapterInfo::new(BACKEND, stable_id, name, driver, class, capabilities)
                        .map_err(|_| {
                            windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL)
                        })?;
                let queue: ID3D12CommandQueue = unsafe {
                    device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                }?;
                let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
                let fence_event = unsafe { CreateEventW(None, false, false, None) }?;
                let allocator = Allocator::new(&AllocatorCreateDesc {
                    device: ID3D12DeviceVersion::Device(device.clone()),
                    debug_settings: Default::default(),
                    allocation_sizes: Default::default(),
                })
                .map_err(|_| {
                    windows::core::Error::from_hresult(windows::Win32::Foundation::E_OUTOFMEMORY)
                })?;
                let descriptors: ID3D12DescriptorHeap = unsafe {
                    device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: TEXTURE_DESCRIPTOR_CAPACITY,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                }?;
                let descriptor_stride = unsafe {
                    device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                };
                let samplers: ID3D12DescriptorHeap = unsafe {
                    device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                        NumDescriptors: TEXTURE_DESCRIPTOR_CAPACITY,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                }?;
                let sampler_stride = unsafe {
                    device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
                } as usize;
                let sampler_desc = D3D12_SAMPLER_DESC {
                    Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    MaxLOD: f32::MAX,
                    ..Default::default()
                };
                let mut sampler = unsafe { samplers.GetCPUDescriptorHandleForHeapStart() };
                for _ in 0..TEXTURE_DESCRIPTOR_CAPACITY {
                    unsafe { device.CreateSampler(&sampler_desc, sampler) };
                    sampler.ptr += sampler_stride;
                }
                return Ok(Self {
                    adapter,
                    device,
                    queue,
                    fence,
                    fence_event,
                    next_fence: 1,
                    allocator: Some(allocator),
                    retired: Vec::new(),
                    pending_copies: Vec::new(),
                    adapter_info,
                    descriptors,
                    descriptor_stride,
                    samplers,
                    sampler_stride: sampler_stride as u32,
                });
            }
        }

        pub fn adapter_info(&self) -> &AdapterInfo {
            &self.adapter_info
        }

        /// The surface's HWND remains host-owned; admission already occurred during context creation.
        pub fn init_device(&self, surface: &NativeSurface) -> Result<AdapterInfo, HalError> {
            if surface.window == 0 {
                return Err(HalError::InvalidArgument);
            }
            Ok(self.adapter_info.clone())
        }

        pub fn wait_idle(&mut self) -> Result<(), HalError> {
            let value = self.next_fence;
            self.next_fence = self
                .next_fence
                .checked_add(1)
                .ok_or(HalError::NativeFailure)?;
            unsafe { self.queue.Signal(&self.fence, value) }.map_err(map_windows)?;
            if unsafe { self.fence.GetCompletedValue() } < value {
                unsafe { self.fence.SetEventOnCompletion(value, self.fence_event) }
                    .map_err(map_windows)?;
                unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
            }
            Ok(())
        }

        /// Acquires the current flip-model back buffer and presents it; zero extent remains minimized.
        pub fn acquire_present(
            &mut self,
            surface: &mut NativeSurface,
            width: u32,
            height: u32,
        ) -> Result<(), HalError> {
            self.ensure_swapchain(surface, width, height)?;
            unsafe {
                surface
                    .swapchain
                    .as_ref()
                    .expect("initialized")
                    .Present(1, DXGI_PRESENT(0))
            }
            .ok()
            .map_err(map_windows)?;
            Ok(())
        }

        fn ensure_swapchain(
            &mut self,
            surface: &mut NativeSurface,
            width: u32,
            height: u32,
        ) -> Result<(), HalError> {
            if width == 0 || height == 0 {
                return Err(HalError::NotReady);
            }
            if surface.swapchain.is_none() {
                let factory: IDXGIFactory4 =
                    unsafe { CreateDXGIFactory1() }.map_err(map_windows)?;
                let desc = DXGI_SWAP_CHAIN_DESC1 {
                    Width: width,
                    Height: height,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    Stereo: false.into(),
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    BufferCount: 3,
                    Scaling: DXGI_SCALING_STRETCH,
                    SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                    AlphaMode: DXGI_ALPHA_MODE_IGNORE,
                    Flags: 0,
                };
                let created = unsafe {
                    factory.CreateSwapChainForHwnd(
                        &self.queue,
                        HWND(surface.window as *mut _),
                        &desc,
                        None,
                        None,
                    )
                }
                .map_err(map_windows)?;
                surface.swapchain = Some(created.cast().map_err(map_windows)?);
                surface.width = width;
                surface.height = height;
            } else if surface.width != width || surface.height != height {
                self.wait_idle()?;
                self.destroy_surface_depth(surface)?;
                surface.buffers.clear();
                surface.rtv_heap = None;
                unsafe {
                    surface
                        .swapchain
                        .as_ref()
                        .expect("initialized")
                        .ResizeBuffers(
                            3,
                            width,
                            height,
                            DXGI_FORMAT_R8G8B8A8_UNORM,
                            DXGI_SWAP_CHAIN_FLAG(0),
                        )
                }
                .map_err(map_windows)?;
                surface.width = width;
                surface.height = height;
            }
            self.ensure_surface_targets(surface)
        }

        fn ensure_surface_targets(&self, surface: &mut NativeSurface) -> Result<(), HalError> {
            if !surface.buffers.is_empty() {
                return Ok(());
            }
            let swapchain = surface.swapchain.as_ref().ok_or(HalError::NotReady)?;
            let heap: ID3D12DescriptorHeap = unsafe {
                self.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 3,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                        NodeMask: 0,
                    })
            }
            .map_err(map_windows)?;
            let stride = unsafe {
                self.device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
            } as usize;
            let mut handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
            for index in 0..3 {
                let buffer: ID3D12Resource =
                    unsafe { swapchain.GetBuffer(index) }.map_err(map_windows)?;
                unsafe { self.device.CreateRenderTargetView(&buffer, None, handle) };
                surface.buffers.push(buffer);
                handle.ptr += stride;
            }
            surface.rtv_heap = Some(heap);
            Ok(())
        }

        fn ensure_surface_depth(&mut self, surface: &mut NativeSurface) -> Result<(), HalError> {
            if surface.depth.is_some() {
                return Ok(());
            }
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: u64::from(surface.width),
                Height: surface.height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            };
            let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
                self.allocator
                    .as_ref()
                    .ok_or(HalError::NativeFailure)?
                    .device(),
                &desc,
                "ez-gfx-depth",
                MemoryLocation::GpuOnly,
            );
            let allocation = self
                .allocator
                .as_mut()
                .ok_or(HalError::NativeFailure)?
                .allocate(&allocation_desc)
                .map_err(map_allocator_hal)?;
            let clear = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: 1.0,
                        Stencil: 0,
                    },
                },
            };
            let mut resource = None;
            if let Err(error) = unsafe {
                self.device.CreatePlacedResource(
                    allocation.heap(),
                    allocation.offset(),
                    &desc,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    Some(&clear),
                    &mut resource,
                )
            } {
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator remains initialized")
                    .free(allocation);
                return Err(map_windows(error));
            }
            let Some(resource) = resource else {
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator remains initialized")
                    .free(allocation);
                return Err(HalError::NativeFailure);
            };
            let heap: ID3D12DescriptorHeap = match unsafe {
                self.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                        NumDescriptors: 1,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                        NodeMask: 0,
                    })
            } {
                Ok(heap) => heap,
                Err(error) => {
                    drop(resource);
                    let _ = self
                        .allocator
                        .as_mut()
                        .expect("allocator remains initialized")
                        .free(allocation);
                    return Err(map_windows(error));
                }
            };
            unsafe {
                self.device.CreateDepthStencilView(
                    &resource,
                    None,
                    heap.GetCPUDescriptorHandleForHeapStart(),
                )
            };
            surface.depth = Some(SurfaceDepth {
                resource,
                allocation,
                heap,
            });
            Ok(())
        }

        fn destroy_surface_depth(&mut self, surface: &mut NativeSurface) -> Result<(), HalError> {
            let Some(depth) = surface.depth.take() else {
                return Ok(());
            };
            drop(depth.resource);
            drop(depth.heap);
            self.allocator
                .as_mut()
                .ok_or(HalError::NativeFailure)?
                .free(depth.allocation)
                .map_err(map_allocator_hal)
        }

        pub fn destroy_surface(&mut self, mut surface: NativeSurface) {
            let _ = self.wait_idle();
            let _ = self.destroy_surface_depth(&mut surface);
        }
        /// DXIL products remain owned until PSO creation; empty products are rejected at admission.
        pub fn create_shader(&self, products: &[&[u8]]) -> Result<NativeShader, HalError> {
            if products.is_empty() || products.iter().any(|product| product.is_empty()) {
                return Err(HalError::InvalidArgument);
            }
            Ok(NativeShader {
                products: products.iter().map(|product| product.to_vec()).collect(),
            })
        }

        /// Reflected buffers use root SRV/UAVs; separate texture and sampler tables occupy the remaining roots.
        fn create_root_signature(
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
            let mut parameters = Vec::with_capacity(descriptor_count + 3);
            parameters.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 32,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
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
                        pDescriptorRanges: &texture_range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
            parameters.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &sampler_range,
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
                NumParameters: parameters.len() as u32,
                pParameters: parameters.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: ptr::null(),
                Flags: flags,
            };
            let mut blob: Option<ID3DBlob> = None;
            unsafe {
                D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
            }
            .map_err(map_windows)?;
            let blob = blob.ok_or(HalError::NativeFailure)?;
            let serialized = unsafe {
                core::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
            };
            let root =
                unsafe { self.device.CreateRootSignature(0, serialized) }.map_err(map_windows)?;
            Ok((root, writable))
        }

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
            let state = unsafe { self.device.CreateComputePipelineState(&state_desc) }
                .map_err(map_windows)?;
            Ok(NativePipeline {
                state,
                root,
                topology: None,
                signature: None,
                buffer_writable,
            })
        }

        pub fn create_graphics_pipeline(
            &self,
            shader: &NativeShader,
            vertex_index: usize,
            fragment_index: usize,
            state: DynamicPipelineState,
            depth_required: bool,
            layouts: &[ShaderBufferLayout],
        ) -> Result<NativePipeline, HalError> {
            let vertex = shader
                .products
                .get(vertex_index)
                .ok_or(HalError::InvalidArgument)?;
            let fragment = shader
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
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
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
            formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
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
                unsafe { self.device.CreateGraphicsPipelineState(&desc) }.map_err(map_windows)?;
            let argument = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
            };
            let signature_desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: 20,
                NumArgumentDescs: 1,
                pArgumentDescs: &argument,
                NodeMask: 0,
            };
            let mut signature = None;
            unsafe {
                self.device
                    .CreateCommandSignature(&signature_desc, None, &mut signature)
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

        /// Records one immutable graph plan into one command list and presents after every action.
        pub fn execute_frame(
            &mut self,
            surface: Option<(&mut NativeSurface, (u32, u32))>,
            actions: &[NativeFrameAction<'_>],
            capture_presented: bool,
        ) -> Result<Vec<Vec<u8>>, HalError> {
            if actions.is_empty() {
                return Err(HalError::InvalidArgument);
            }
            let (mut surface, extent) = match surface {
                Some((surface, extent)) if extent.0 != 0 && extent.1 != 0 => {
                    (Some(surface), extent)
                }
                Some(_) => return Err(HalError::InvalidArgument),
                None => (None, (0, 0)),
            };
            let present_count = actions
                .iter()
                .filter(|action| matches!(action, NativeFrameAction::Present))
                .count();
            let presents = present_count == 1;
            let uses_surface = actions.iter().any(|action| {
                matches!(
                    action,
                    NativeFrameAction::BeginPass(_)
                        | NativeFrameAction::Graphics(_)
                        | NativeFrameAction::Present
                        | NativeFrameAction::Barrier {
                            resource: NativeFrameResource::Surface | NativeFrameResource::Depth,
                            ..
                        }
                )
            });
            if present_count > 1
                || uses_surface && surface.is_none()
                || (uses_surface || capture_presented) && !presents
            {
                return Err(HalError::InvalidArgument);
            }
            let external_waits = actions
                .iter()
                .filter_map(|action| match action {
                    NativeFrameAction::Wait(token) if token.queue == QueueKind::Transfer => {
                        Some(Ok(token.value))
                    }
                    NativeFrameAction::Wait(_) => Some(Err(HalError::InvalidArgument)),
                    _ => None,
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pass_active = false;
            let mut saw_present = false;
            for action in actions {
                if saw_present {
                    return Err(HalError::InvalidArgument);
                }
                match action {
                    NativeFrameAction::Wait(_) | NativeFrameAction::Barrier { .. } => {}
                    NativeFrameAction::BeginPass(pass) => {
                        if pass_active
                            || pass.colors.len() != 1
                            || pass.samples != 1
                            || pass.area[0]
                                .checked_add(pass.area[2])
                                .is_none_or(|end| end > extent.0)
                            || pass.area[1]
                                .checked_add(pass.area[3])
                                .is_none_or(|end| end > extent.1)
                        {
                            return Err(HalError::InvalidArgument);
                        }
                        pass_active = true;
                    }
                    NativeFrameAction::Compute(dispatch) => {
                        if pass_active
                            || dispatch.groups.contains(&0)
                            || dispatch.push_constants.len() > 128
                            || !dispatch.push_constants.len().is_multiple_of(4)
                            || dispatch.bindings.len() != dispatch.pipeline.buffer_writable.len()
                            || dispatch
                                .bindings
                                .iter()
                                .zip(&dispatch.pipeline.buffer_writable)
                                .any(|(binding, writable)| {
                                    binding.writable != *writable
                                        || binding.offset >= binding.allocation.allocation.size()
                                })
                        {
                            return Err(HalError::InvalidArgument);
                        }
                    }
                    NativeFrameAction::Graphics(draw) => {
                        let indirect_size = u64::from(draw.draw_count)
                            .checked_mul(20)
                            .ok_or(HalError::InvalidArgument)?;
                        if !pass_active
                            || draw.draw_count == 0
                            || draw.push_constants.len() > 128
                            || !draw.push_constants.len().is_multiple_of(4)
                            || draw.bindings.len() != draw.pipeline.buffer_writable.len()
                            || draw.pipeline.topology.is_none()
                            || draw.pipeline.signature.is_none()
                            || draw.index_buffer.allocation.size() > u64::from(u32::MAX)
                            || draw.indirect_buffer.allocation.size() < indirect_size
                            || draw
                                .bindings
                                .iter()
                                .zip(&draw.pipeline.buffer_writable)
                                .any(|(binding, writable)| {
                                    binding.writable != *writable
                                        || binding.offset >= binding.allocation.allocation.size()
                                })
                        {
                            return Err(HalError::InvalidArgument);
                        }
                    }
                    NativeFrameAction::TextureReadback { width, height, .. } => {
                        if pass_active || *width == 0 || *height == 0 {
                            return Err(HalError::InvalidArgument);
                        }
                    }
                    NativeFrameAction::EndPass => {
                        if !pass_active {
                            return Err(HalError::InvalidArgument);
                        }
                        pass_active = false;
                    }
                    NativeFrameAction::Present => {
                        if pass_active {
                            return Err(HalError::InvalidArgument);
                        }
                        saw_present = true;
                    }
                }
            }
            if pass_active {
                return Err(HalError::InvalidArgument);
            }
            if uses_surface {
                self.ensure_swapchain(
                    surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                    extent.0,
                    extent.1,
                )?;
            }
            if actions.iter().any(|action| {
                matches!(
                    action,
                    NativeFrameAction::BeginPass(pass) if pass.depth.is_some()
                )
            }) {
                self.ensure_surface_depth(
                    surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                )?;
            }

            let surface_resources = surface
                .as_ref()
                .map(|surface| {
                    let swapchain = surface
                        .swapchain
                        .as_ref()
                        .ok_or(HalError::NotReady)?
                        .clone();
                    let image_index = unsafe { swapchain.GetCurrentBackBufferIndex() } as usize;
                    let back_buffer = surface
                        .buffers
                        .get(image_index)
                        .ok_or(HalError::NativeFailure)?
                        .clone();
                    let heap = surface.rtv_heap.as_ref().ok_or(HalError::NativeFailure)?;
                    let stride = unsafe {
                        self.device
                            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
                    } as usize;
                    let mut rtv = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
                    rtv.ptr += image_index * stride;
                    let dsv = surface
                        .depth
                        .as_ref()
                        .map(|depth| unsafe { depth.heap.GetCPUDescriptorHandleForHeapStart() });
                    Ok((swapchain, back_buffer, rtv, dsv))
                })
                .transpose()?;
            let (swapchain, back_buffer, rtv, dsv) = match surface_resources {
                Some((swapchain, back_buffer, rtv, dsv)) => {
                    (Some(swapchain), Some(back_buffer), Some(rtv), dsv)
                }
                None => (None, None, None, None),
            };

            let allocator: ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .map_err(map_windows)?;
            let list: ID3D12GraphicsCommandList = unsafe {
                self.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None::<&ID3D12PipelineState>,
                )
            }
            .map_err(map_windows)?;
            let mut readbacks = Vec::new();
            for action in actions {
                let resource = match action {
                    NativeFrameAction::TextureReadback { texture, .. } => {
                        Some(texture.resource.clone())
                    }
                    NativeFrameAction::Present if capture_presented => back_buffer.clone(),
                    _ => None,
                };
                let Some(resource) = resource else {
                    continue;
                };
                let desc = unsafe { resource.GetDesc() };
                let mut footprint = Default::default();
                let mut rows = 0;
                let mut row_size = 0;
                let mut total = 0;
                unsafe {
                    self.device.GetCopyableFootprints(
                        &desc,
                        0,
                        1,
                        0,
                        Some(&mut footprint),
                        Some(&mut rows),
                        Some(&mut row_size),
                        Some(&mut total),
                    );
                }
                let request =
                    match AllocationRequest::new(total, 256, MemoryClass::Readback, true, None) {
                        Ok(request) => request,
                        Err(_) => {
                            for (_, _, _, _, allocation) in readbacks {
                                let _ = self.free(allocation);
                            }
                            return Err(HalError::InvalidArgument);
                        }
                    };
                let allocation = match self.allocate(request) {
                    Ok(allocation) => allocation,
                    Err(_) => {
                        for (_, _, _, _, allocation) in readbacks {
                            let _ = self.free(allocation);
                        }
                        return Err(HalError::NativeFailure);
                    }
                };
                let dimensions = match action {
                    NativeFrameAction::TextureReadback { width, height, .. } => (*width, *height),
                    NativeFrameAction::Present => extent,
                    _ => unreachable!(),
                };
                readbacks.push((dimensions.0, dimensions.1, footprint, total, allocation));
            }
            let mut indirect_copies: Vec<Option<NativeAllocation>> =
                (0..actions.len()).map(|_| None).collect();
            for (action_index, action) in actions.iter().enumerate() {
                let NativeFrameAction::Graphics(draw) = action else {
                    continue;
                };
                let indirect_raw = windows::core::Interface::as_raw(&draw.indirect_buffer.resource);
                if !draw.bindings.iter().any(|binding| {
                    windows::core::Interface::as_raw(&binding.allocation.resource) == indirect_raw
                }) {
                    continue;
                }
                let request = match AllocationRequest::new(
                    draw.indirect_buffer.allocation.size(),
                    16,
                    MemoryClass::Device,
                    false,
                    None,
                ) {
                    Ok(request) => request,
                    Err(_) => {
                        for (_, _, _, _, allocation) in readbacks {
                            let _ = self.free(allocation);
                        }
                        for allocation in indirect_copies.into_iter().flatten() {
                            let _ = self.free(allocation);
                        }
                        return Err(HalError::InvalidArgument);
                    }
                };
                indirect_copies[action_index] = match self.allocate(request) {
                    Ok(allocation) => Some(allocation),
                    Err(_) => {
                        for (_, _, _, _, allocation) in readbacks {
                            let _ = self.free(allocation);
                        }
                        for allocation in indirect_copies.into_iter().flatten() {
                            let _ = self.free(allocation);
                        }
                        return Err(HalError::NativeFailure);
                    }
                };
            }

            let descriptor_heap = self.descriptors.clone();
            let sampler_heap = self.samplers.clone();
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: extent.0 as f32,
                Height: extent.1 as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: 0,
                top: 0,
                right: extent.0 as i32,
                bottom: extent.1 as i32,
            };
            let mut pass_active = false;
            let mut discard_store = false;
            let mut discard_depth = false;
            let mut readback_index = 0_usize;
            let encoded = (|| {
                unsafe {
                    list.SetDescriptorHeaps(&[
                        Some(descriptor_heap.clone()),
                        Some(sampler_heap.clone()),
                    ]);
                    list.RSSetViewports(&[viewport]);
                    list.RSSetScissorRects(&[scissor]);
                }
                for (action_index, action) in actions.iter().enumerate() {
                    match action {
                        NativeFrameAction::Wait(_) => {}
                        NativeFrameAction::Barrier { barrier, resource } => {
                            let native = match resource {
                                NativeFrameResource::Buffer(allocation) => {
                                    allocation.resource.clone()
                                }
                                NativeFrameResource::Texture(texture) => texture.resource.clone(),
                                NativeFrameResource::Surface => back_buffer
                                    .as_ref()
                                    .ok_or(HalError::InvalidArgument)?
                                    .clone(),
                                NativeFrameResource::Depth => surface
                                    .as_ref()
                                    .and_then(|surface| surface.depth.as_ref())
                                    .ok_or(HalError::NotReady)?
                                    .resource
                                    .clone(),
                            };
                            let before = barrier
                                .before
                                .map(|state| dx12_resource_state(state.access))
                                .unwrap_or(D3D12_RESOURCE_STATE_COMMON);
                            let after = dx12_resource_state(barrier.after.access);
                            unsafe {
                                if before == after && after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                {
                                    list.ResourceBarrier(&[uav_barrier(native)]);
                                } else if before != after {
                                    list.ResourceBarrier(&[transition_barrier(
                                        native, before, after,
                                    )]);
                                }
                            }
                        }
                        NativeFrameAction::BeginPass(pass) => {
                            if pass_active
                                || pass.colors.len() != 1
                                || pass.samples != 1
                                || pass.area[0]
                                    .checked_add(pass.area[2])
                                    .is_none_or(|end| end > extent.0)
                                || pass.area[1]
                                    .checked_add(pass.area[3])
                                    .is_none_or(|end| end > extent.1)
                            {
                                return Err(HalError::InvalidArgument);
                            }
                            let rtv = rtv.ok_or(HalError::InvalidArgument)?;
                            unsafe {
                                list.OMSetRenderTargets(
                                    1,
                                    Some(&rtv),
                                    false,
                                    pass.depth
                                        .and(dsv)
                                        .as_ref()
                                        .map(|handle| handle as *const _),
                                );
                                match pass.load {
                                    AttachmentLoadOp::Load => {}
                                    AttachmentLoadOp::Clear => {
                                        list.ClearRenderTargetView(
                                            rtv,
                                            &[0.1, 0.1, 0.1, 1.0],
                                            None,
                                        );
                                        if let Some(dsv) = pass.depth.and(dsv) {
                                            list.ClearDepthStencilView(
                                                dsv,
                                                D3D12_CLEAR_FLAG_DEPTH,
                                                1.0,
                                                0,
                                                None,
                                            );
                                        }
                                    }
                                    AttachmentLoadOp::Discard => {
                                        list.DiscardResource(
                                            back_buffer
                                                .as_ref()
                                                .ok_or(HalError::InvalidArgument)?,
                                            None,
                                        );
                                        if pass.depth.is_some()
                                            && let Some(depth) = surface
                                                .as_ref()
                                                .and_then(|surface| surface.depth.as_ref())
                                        {
                                            list.DiscardResource(&depth.resource, None);
                                        }
                                    }
                                }
                            }
                            discard_store = pass.store == AttachmentStoreOp::Discard;
                            discard_depth = pass.depth.is_some();
                            pass_active = true;
                        }
                        NativeFrameAction::Compute(dispatch) => {
                            if pass_active {
                                return Err(HalError::InvalidArgument);
                            }
                            let pipeline = dispatch.pipeline;
                            unsafe {
                                list.SetPipelineState(&pipeline.state);
                                list.SetComputeRootSignature(&pipeline.root);
                                let table = pipeline.buffer_writable.len() as u32 + 1;
                                list.SetComputeRootDescriptorTable(
                                    table,
                                    descriptor_heap.GetGPUDescriptorHandleForHeapStart(),
                                );
                                list.SetComputeRootDescriptorTable(
                                    table + 1,
                                    sampler_heap.GetGPUDescriptorHandleForHeapStart(),
                                );
                                bind_dx12_compute_buffers(&list, pipeline, dispatch.bindings)?;
                                if !dispatch.push_constants.is_empty() {
                                    list.SetComputeRoot32BitConstants(
                                        0,
                                        dispatch.push_constants.len() as u32 / 4,
                                        dispatch.push_constants.as_ptr().cast(),
                                        0,
                                    );
                                }
                                list.Dispatch(
                                    dispatch.groups[0],
                                    dispatch.groups[1],
                                    dispatch.groups[2],
                                );
                            }
                        }
                        NativeFrameAction::Graphics(draw) => {
                            if !pass_active {
                                return Err(HalError::InvalidArgument);
                            }
                            let pipeline = draw.pipeline;
                            let topology = pipeline.topology.ok_or(HalError::InvalidArgument)?;
                            let signature = pipeline
                                .signature
                                .as_ref()
                                .ok_or(HalError::InvalidArgument)?;
                            let index_size = u32::try_from(draw.index_buffer.allocation.size())
                                .map_err(|_| HalError::InvalidArgument)?;
                            let index_view = D3D12_INDEX_BUFFER_VIEW {
                                BufferLocation: unsafe {
                                    draw.index_buffer.resource.GetGPUVirtualAddress()
                                },
                                SizeInBytes: index_size,
                                Format: DXGI_FORMAT_R32_UINT,
                            };
                            let indirect_resource =
                                if let Some(copy) = indirect_copies[action_index].as_ref() {
                                    let indirect_raw = windows::core::Interface::as_raw(
                                        &draw.indirect_buffer.resource,
                                    );
                                    let binding = draw
                                        .bindings
                                        .iter()
                                        .find(|binding| {
                                            windows::core::Interface::as_raw(
                                                &binding.allocation.resource,
                                            ) == indirect_raw
                                        })
                                        .ok_or(HalError::InvalidArgument)?;
                                    let shader_state = if binding.writable {
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                    } else {
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                                    };
                                    unsafe {
                                        list.ResourceBarrier(&[
                                            transition_barrier(
                                                draw.indirect_buffer.resource.clone(),
                                                shader_state,
                                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                                            ),
                                            transition_barrier(
                                                copy.resource.clone(),
                                                D3D12_RESOURCE_STATE_COMMON,
                                                D3D12_RESOURCE_STATE_COPY_DEST,
                                            ),
                                        ]);
                                        list.CopyBufferRegion(
                                            &copy.resource,
                                            0,
                                            &draw.indirect_buffer.resource,
                                            0,
                                            draw.indirect_buffer.allocation.size(),
                                        );
                                        list.ResourceBarrier(&[
                                            transition_barrier(
                                                draw.indirect_buffer.resource.clone(),
                                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                shader_state,
                                            ),
                                            transition_barrier(
                                                copy.resource.clone(),
                                                D3D12_RESOURCE_STATE_COPY_DEST,
                                                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                            ),
                                        ]);
                                    }
                                    &copy.resource
                                } else {
                                    &draw.indirect_buffer.resource
                                };
                            unsafe {
                                list.SetPipelineState(&pipeline.state);
                                list.SetGraphicsRootSignature(&pipeline.root);
                                let table = pipeline.buffer_writable.len() as u32 + 1;
                                list.SetGraphicsRootDescriptorTable(
                                    table,
                                    descriptor_heap.GetGPUDescriptorHandleForHeapStart(),
                                );
                                list.SetGraphicsRootDescriptorTable(
                                    table + 1,
                                    sampler_heap.GetGPUDescriptorHandleForHeapStart(),
                                );
                                bind_dx12_graphics_buffers(&list, pipeline, draw.bindings)?;
                                if !draw.push_constants.is_empty() {
                                    list.SetGraphicsRoot32BitConstants(
                                        0,
                                        draw.push_constants.len() as u32 / 4,
                                        draw.push_constants.as_ptr().cast(),
                                        0,
                                    );
                                }
                                list.IASetPrimitiveTopology(topology);
                                list.IASetIndexBuffer(Some(&index_view));
                                list.ExecuteIndirect(
                                    signature,
                                    draw.draw_count,
                                    indirect_resource,
                                    0,
                                    None,
                                    0,
                                );
                            }
                        }
                        NativeFrameAction::TextureReadback { texture, .. } => {
                            if pass_active {
                                return Err(HalError::InvalidArgument);
                            }
                            let (_, _, footprint, _, readback) = readbacks
                                .get(readback_index)
                                .ok_or(HalError::InvalidArgument)?;
                            unsafe {
                                copy_texture_to_readback(
                                    &list,
                                    &texture.resource,
                                    &readback.resource,
                                    *footprint,
                                );
                            }
                            readback_index += 1;
                        }
                        NativeFrameAction::EndPass => {
                            if !pass_active {
                                return Err(HalError::InvalidArgument);
                            }
                            if discard_store {
                                unsafe {
                                    list.DiscardResource(
                                        back_buffer.as_ref().ok_or(HalError::InvalidArgument)?,
                                        None,
                                    );
                                    if discard_depth
                                        && let Some(depth) = surface
                                            .as_ref()
                                            .and_then(|surface| surface.depth.as_ref())
                                    {
                                        list.DiscardResource(&depth.resource, None);
                                    }
                                }
                            }
                            pass_active = false;
                            discard_store = false;
                            discard_depth = false;
                        }
                        NativeFrameAction::Present => {
                            if pass_active {
                                return Err(HalError::InvalidArgument);
                            }
                            if capture_presented {
                                let (_, _, footprint, _, readback) = readbacks
                                    .get(readback_index)
                                    .ok_or(HalError::InvalidArgument)?;
                                unsafe {
                                    let back_buffer =
                                        back_buffer.as_ref().ok_or(HalError::InvalidArgument)?;
                                    list.ResourceBarrier(&[transition_barrier(
                                        back_buffer.clone(),
                                        D3D12_RESOURCE_STATE_PRESENT,
                                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                                    )]);
                                    copy_texture_to_readback(
                                        &list,
                                        back_buffer,
                                        &readback.resource,
                                        *footprint,
                                    );
                                    list.ResourceBarrier(&[transition_barrier(
                                        back_buffer.clone(),
                                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                                        D3D12_RESOURCE_STATE_PRESENT,
                                    )]);
                                }
                                readback_index += 1;
                            }
                        }
                    }
                }
                if pass_active {
                    return Err(HalError::InvalidArgument);
                }
                unsafe { list.Close().map_err(map_windows) }
            })();
            if let Err(error) = encoded {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                for allocation in indirect_copies.into_iter().flatten() {
                    let _ = self.free(allocation);
                }
                return Err(error);
            }
            let command: ID3D12CommandList = match list.cast() {
                Ok(command) => command,
                Err(error) => {
                    for (_, _, _, _, allocation) in readbacks {
                        let _ = self.free(allocation);
                    }
                    for allocation in indirect_copies.into_iter().flatten() {
                        let _ = self.free(allocation);
                    }
                    return Err(map_windows(error));
                }
            };
            for value in external_waits {
                if let Err(error) = unsafe { self.queue.Wait(&self.fence, value) } {
                    for (_, _, _, _, allocation) in readbacks {
                        let _ = self.free(allocation);
                    }
                    for allocation in indirect_copies.into_iter().flatten() {
                        let _ = self.free(allocation);
                    }
                    return Err(map_windows(error));
                }
            }
            unsafe {
                self.queue.ExecuteCommandLists(&[Some(command)]);
                if presents
                    && let Err(error) = swapchain
                        .as_ref()
                        .ok_or(HalError::InvalidArgument)?
                        .Present(1, DXGI_PRESENT(0))
                        .ok()
                {
                    let _ = self.wait_idle();
                    for (_, _, _, _, allocation) in readbacks {
                        let _ = self.free(allocation);
                    }
                    for allocation in indirect_copies.into_iter().flatten() {
                        let _ = self.free(allocation);
                    }
                    return Err(map_windows(error));
                }
            }
            if let Err(error) = self.wait_idle() {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                for allocation in indirect_copies.into_iter().flatten() {
                    let _ = self.free(allocation);
                }
                return Err(error);
            }
            let mut cleanup_error = None;
            for allocation in indirect_copies.into_iter().flatten() {
                if self.free(allocation).is_err() {
                    cleanup_error = Some(HalError::NativeFailure);
                }
            }
            if let Some(error) = cleanup_error {
                for (_, _, _, _, allocation) in readbacks {
                    let _ = self.free(allocation);
                }
                return Err(error);
            }

            let mut outputs = Vec::with_capacity(readbacks.len());
            let mut remaining = readbacks.into_iter();
            while let Some((width, height, footprint, total, mut readback)) = remaining.next() {
                let output = (|| {
                    self.invalidate(&mut readback, 0, total)
                        .map_err(|_| HalError::NativeFailure)?;
                    let source = self
                        .mapped_slice(&readback)
                        .map_err(|_| HalError::NativeFailure)?;
                    let row_bytes = (width as usize)
                        .checked_mul(4)
                        .ok_or(HalError::InvalidArgument)?;
                    let pixel_bytes = row_bytes
                        .checked_mul(height as usize)
                        .ok_or(HalError::InvalidArgument)?;
                    let mut pixels = vec![0; pixel_bytes];
                    for row in 0..height as usize {
                        let source_start = row
                            .checked_mul(footprint.Footprint.RowPitch as usize)
                            .ok_or(HalError::InvalidArgument)?;
                        let source_end = source_start
                            .checked_add(row_bytes)
                            .ok_or(HalError::InvalidArgument)?;
                        let destination_start = row
                            .checked_mul(row_bytes)
                            .ok_or(HalError::InvalidArgument)?;
                        let destination_end = destination_start
                            .checked_add(row_bytes)
                            .ok_or(HalError::InvalidArgument)?;
                        pixels
                            .get_mut(destination_start..destination_end)
                            .ok_or(HalError::NativeFailure)?
                            .copy_from_slice(
                                source
                                    .get(source_start..source_end)
                                    .ok_or(HalError::NativeFailure)?,
                            );
                    }
                    Ok(pixels)
                })();
                let freed = self.free(readback).map_err(|_| HalError::NativeFailure);
                match (output, freed) {
                    (Ok(pixels), Ok(())) => outputs.push(pixels),
                    (Err(error), _) | (_, Err(error)) => {
                        for (_, _, _, _, allocation) in remaining {
                            let _ = self.free(allocation);
                        }
                        return Err(error);
                    }
                }
            }
            if capture_presented && let Some(pixels) = outputs.last() {
                surface.ok_or(HalError::InvalidArgument)?.presented = pixels.clone();
            }
            Ok(outputs)
        }

        fn destroy_unpublished_texture(
            &mut self,
            resource: ID3D12Resource,
            allocation: Allocation,
            upload: Option<NativeAllocation>,
            wait_for_queue: bool,
        ) {
            if wait_for_queue && self.wait_idle().is_ok() {
                let completed = unsafe { self.fence.GetCompletedValue() };
                self.pending_copies
                    .retain(|copy| copy.completion.value > completed);
            }
            if let Some(upload) = upload {
                let _ = self.free(upload);
            }
            drop(resource);
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
        }

        /// Creates an RGBA8 mip chain and submits each level under a distinct fence value.
        pub fn create_texture_rgba8(
            &mut self,
            mips: &[ImageMip<'_>],
            binding: u32,
            sampler_desc: TextureSamplerDesc,
        ) -> Result<(NativeTexture, Vec<CompletionToken>), AllocationError> {
            validate_rgba8_mips(mips).map_err(|_| AllocationError::ZeroSize)?;
            if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
                return Err(AllocationError::ZeroSize);
            }
            let width = mips[0].width;
            let height = mips[0].height;
            let mip_count =
                u16::try_from(mips.len()).map_err(|_| AllocationError::NativeFailure)?;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: u64::from(width),
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: mip_count,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_NONE,
            };
            let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
                self.allocator
                    .as_ref()
                    .ok_or(AllocationError::NativeFailure)?
                    .device(),
                &desc,
                "ez-gfx-texture",
                MemoryLocation::GpuOnly,
            );
            let allocation = self
                .allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .allocate(&allocation_desc)
                .map_err(map_allocator)?;
            let mut resource: Option<ID3D12Resource> = None;
            if let Err(error) = unsafe {
                self.device.CreatePlacedResource(
                    allocation.heap(),
                    allocation.offset(),
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut resource,
                )
            } {
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator remains initialized")
                    .free(allocation);
                return Err(map_allocation_windows(error));
            }
            let resource = match resource {
                Some(resource) => resource,
                None => {
                    let _ = self
                        .allocator
                        .as_mut()
                        .expect("allocator remains initialized")
                        .free(allocation);
                    return Err(AllocationError::NativeFailure);
                }
            };
            let mut footprints = vec![Default::default(); mips.len()];
            let mut row_counts = vec![0_u32; mips.len()];
            let mut row_sizes = vec![0_u64; mips.len()];
            let mut upload_size = 0;
            unsafe {
                self.device.GetCopyableFootprints(
                    &desc,
                    0,
                    mips.len() as u32,
                    0,
                    Some(footprints.as_mut_ptr()),
                    Some(row_counts.as_mut_ptr()),
                    Some(row_sizes.as_mut_ptr()),
                    Some(&mut upload_size),
                );
            }
            let upload_request =
                match AllocationRequest::new(upload_size, 256, MemoryClass::Upload, true, None) {
                    Ok(request) => request,
                    Err(_) => {
                        self.destroy_unpublished_texture(resource, allocation, None, false);
                        return Err(AllocationError::ZeroSize);
                    }
                };
            let mut upload = match self.allocate(upload_request) {
                Ok(upload) => upload,
                Err(error) => {
                    self.destroy_unpublished_texture(resource, allocation, None, false);
                    return Err(error);
                }
            };
            let populated = (|| {
                let target = self.mapped_slice_mut(&mut upload)?;
                for (mip, footprint) in mips.iter().zip(&footprints) {
                    let row_bytes = mip.width as usize * 4;
                    for row in 0..mip.height as usize {
                        let source_start = row * row_bytes;
                        let destination_start =
                            footprint.Offset as usize + row * footprint.Footprint.RowPitch as usize;
                        target[destination_start..destination_start + row_bytes]
                            .copy_from_slice(&mip.bytes[source_start..source_start + row_bytes]);
                    }
                }
                self.flush(&mut upload, 0, upload_size)
            })();
            if let Err(error) = populated {
                self.destroy_unpublished_texture(resource, allocation, Some(upload), false);
                return Err(error);
            }
            let mut queue_touched = false;
            let submitted = (|| {
                let mut completions = Vec::with_capacity(mips.len());
                for (level, footprint) in footprints.into_iter().enumerate() {
                    let allocator: ID3D12CommandAllocator = unsafe {
                        self.device
                            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    }
                    .map_err(map_allocation_windows)?;
                    let list: ID3D12GraphicsCommandList = unsafe {
                        self.device.CreateCommandList(
                            0,
                            D3D12_COMMAND_LIST_TYPE_DIRECT,
                            &allocator,
                            None,
                        )
                    }
                    .map_err(map_allocation_windows)?;
                    let source = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: core::mem::ManuallyDrop::new(Some(upload.resource.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: footprint,
                        },
                    };
                    let destination = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: level as u32,
                        },
                    };
                    unsafe { list.CopyTextureRegion(&destination, 0, 0, 0, &source, None) };
                    let transition = D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
                        Subresource: level as u32,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                        StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    };
                    let barrier = D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            Transition: core::mem::ManuallyDrop::new(transition),
                        },
                    };
                    unsafe {
                        list.ResourceBarrier(&[barrier]);
                        list.Close()
                    }
                    .map_err(map_allocation_windows)?;
                    let command: ID3D12CommandList = list.cast().map_err(map_allocation_windows)?;
                    unsafe { self.queue.ExecuteCommandLists(&[Some(command)]) };
                    queue_touched = true;
                    let value = self.next_fence;
                    self.next_fence = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
                    unsafe { self.queue.Signal(&self.fence, value) }
                        .map_err(map_allocation_windows)?;
                    let completion = CompletionToken::new(QueueKind::Transfer, value)
                        .map_err(|_| AllocationError::NativeFailure)?;
                    self.pending_copies.push(PendingCopy {
                        completion,
                        _allocator: allocator,
                        _list: list,
                    });
                    completions.push(completion);
                }
                Ok::<_, AllocationError>(completions)
            })();
            let completions = match submitted {
                Ok(completions) => completions,
                Err(error) => {
                    self.destroy_unpublished_texture(
                        resource,
                        allocation,
                        Some(upload),
                        queue_touched,
                    );
                    return Err(error);
                }
            };
            if let Err(error) = self.wait_idle().map_err(map_hal_allocation) {
                self.destroy_unpublished_texture(resource, allocation, Some(upload), true);
                return Err(error);
            }
            self.pending_copies
                .retain(|copy| copy.completion.value > unsafe { self.fence.GetCompletedValue() });
            if let Err(error) = self.free(upload) {
                self.destroy_unpublished_texture(resource, allocation, None, false);
                return Err(error);
            }
            let mut handle = unsafe { self.descriptors.GetCPUDescriptorHandleForHeapStart() };
            handle.ptr += binding as usize * self.descriptor_stride as usize;
            unsafe {
                self.device
                    .CreateShaderResourceView(&resource, None, handle)
            };
            let filter = if sampler_desc.max_anisotropy > 1.0 {
                D3D12_FILTER_ANISOTROPIC
            } else {
                match (sampler_desc.min_filter, sampler_desc.mag_filter) {
                    (SamplerFilter::Nearest, SamplerFilter::Nearest) => {
                        D3D12_FILTER_MIN_MAG_MIP_POINT
                    }
                    (SamplerFilter::Nearest, SamplerFilter::Linear) => {
                        D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT
                    }
                    (SamplerFilter::Linear, SamplerFilter::Nearest) => {
                        D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR
                    }
                    (SamplerFilter::Linear, SamplerFilter::Linear) => {
                        D3D12_FILTER_MIN_MAG_MIP_LINEAR
                    }
                }
            };
            let address = |mode| match mode {
                SamplerAddressMode::Clamp => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                SamplerAddressMode::Repeat => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            };
            let sampler = D3D12_SAMPLER_DESC {
                Filter: filter,
                AddressU: address(sampler_desc.address_u),
                AddressV: address(sampler_desc.address_v),
                AddressW: address(sampler_desc.address_w),
                MaxAnisotropy: sampler_desc.max_anisotropy as u32,
                MaxLOD: f32::MAX,
                ..Default::default()
            };
            let mut sampler_handle = unsafe { self.samplers.GetCPUDescriptorHandleForHeapStart() };
            sampler_handle.ptr += binding as usize * self.sampler_stride as usize;
            unsafe { self.device.CreateSampler(&sampler, sampler_handle) };
            Ok((
                NativeTexture {
                    resource,
                    allocation,
                    binding,
                },
                completions,
            ))
        }

        pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
            drop(texture.resource);
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(texture.allocation)
                .map_err(map_allocator)
        }

        /// Copies the shader-readable image through a GPU readback footprint and returns tightly packed RGBA8 rows.
        pub fn readback_texture_rgba8(
            &mut self,
            texture: &NativeTexture,
            width: u32,
            height: u32,
        ) -> Result<Vec<u8>, AllocationError> {
            let desc = unsafe { texture.resource.GetDesc() };
            if desc.Width != u64::from(width) || desc.Height != height {
                return Err(AllocationError::NativeFailure);
            }
            let mut footprint = Default::default();
            let mut rows = 0;
            let mut row_size = 0;
            let mut total = 0;
            unsafe {
                self.device.GetCopyableFootprints(
                    &desc,
                    0,
                    1,
                    0,
                    Some(&mut footprint),
                    Some(&mut rows),
                    Some(&mut row_size),
                    Some(&mut total),
                );
            }
            let mut readback = self.allocate(
                AllocationRequest::new(total, 256, MemoryClass::Readback, true, None)
                    .map_err(|_| AllocationError::ZeroSize)?,
            )?;
            let allocator: ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .map_err(map_allocation_windows)?;
            let list: ID3D12GraphicsCommandList = unsafe {
                self.device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            }
            .map_err(map_allocation_windows)?;
            let to_copy = D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: core::mem::ManuallyDrop::new(Some(texture.resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                StateAfter: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE,
            };
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: core::mem::ManuallyDrop::new(to_copy),
                },
            };
            unsafe { list.ResourceBarrier(&[barrier]) };
            let source = D3D12_TEXTURE_COPY_LOCATION {
                pResource: core::mem::ManuallyDrop::new(Some(texture.resource.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let destination = D3D12_TEXTURE_COPY_LOCATION {
                pResource: core::mem::ManuallyDrop::new(Some(readback.resource.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            unsafe { list.CopyTextureRegion(&destination, 0, 0, 0, &source, None) };
            let to_shader = D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: core::mem::ManuallyDrop::new(Some(texture.resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE,
                StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            };
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: core::mem::ManuallyDrop::new(to_shader),
                },
            };
            unsafe {
                list.ResourceBarrier(&[barrier]);
                list.Close()
            }
            .map_err(map_allocation_windows)?;
            let command: ID3D12CommandList = list.cast().map_err(map_allocation_windows)?;
            unsafe { self.queue.ExecuteCommandLists(&[Some(command)]) };
            let value = self.next_fence;
            self.next_fence = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
            unsafe { self.queue.Signal(&self.fence, value) }.map_err(map_allocation_windows)?;
            if unsafe { self.fence.GetCompletedValue() } < value {
                unsafe { self.fence.SetEventOnCompletion(value, self.fence_event) }
                    .map_err(map_allocation_windows)?;
                unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
            }
            self.invalidate(&mut readback, 0, total)?;
            let source = self.mapped_slice(&readback)?;
            let mut pixels = vec![0; width as usize * height as usize * 4];
            for row in 0..height as usize {
                let source_start = row * footprint.Footprint.RowPitch as usize;
                let destination_start = row * width as usize * 4;
                pixels[destination_start..destination_start + width as usize * 4]
                    .copy_from_slice(&source[source_start..source_start + width as usize * 4]);
            }
            self.free(readback)?;
            Ok(pixels)
        }
        pub fn destroy_shader(&self, _shader: NativeShader) {}
    }

    impl MemoryAllocator for NativeContext {
        type Allocation = NativeAllocation;

        fn allocate(
            &mut self,
            request: AllocationRequest,
        ) -> Result<Self::Allocation, AllocationError> {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
                Width: request.size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: if matches!(
                    request.memory_class,
                    MemoryClass::Device | MemoryClass::Transient
                ) {
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                } else {
                    D3D12_RESOURCE_FLAG_NONE
                },
            };
            let location = match request.memory_class {
                MemoryClass::Device | MemoryClass::Transient => MemoryLocation::GpuOnly,
                MemoryClass::Upload => MemoryLocation::CpuToGpu,
                MemoryClass::Readback => MemoryLocation::GpuToCpu,
            };
            let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
                self.allocator
                    .as_ref()
                    .ok_or(AllocationError::NativeFailure)?
                    .device(),
                &desc,
                "ez-gfx-buffer",
                location,
            );
            if allocation_desc.alignment < request.alignment {
                return Err(AllocationError::InvalidAlignment);
            }
            let allocation = self
                .allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .allocate(&allocation_desc)
                .map_err(map_allocator)?;
            let initial_state = match request.memory_class {
                MemoryClass::Upload => D3D12_RESOURCE_STATE_GENERIC_READ,
                MemoryClass::Readback => D3D12_RESOURCE_STATE_COPY_DEST,
                _ => D3D12_RESOURCE_STATE_COMMON,
            };
            let mut resource: Option<ID3D12Resource> = None;
            let created = unsafe {
                self.device.CreatePlacedResource(
                    allocation.heap(),
                    allocation.offset(),
                    &desc,
                    initial_state,
                    None,
                    &mut resource,
                )
            };
            if let Err(error) = created {
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator remains initialized")
                    .free(allocation);
                return Err(map_allocation_windows(error));
            }
            let resource = resource.ok_or(AllocationError::NativeFailure)?;
            let mut mapped_address = 0;
            if request.mapped {
                let mut mapped: *mut c_void = ptr::null_mut();
                unsafe { resource.Map(0, None, Some(&mut mapped)) }
                    .map_err(map_allocation_windows)?;
                if mapped.is_null() {
                    return Err(AllocationError::NotHostVisible);
                }
                mapped_address = mapped as usize;
            }
            Ok(NativeAllocation {
                resource,
                allocation,
                mapped_address,
            })
        }

        fn mapped_slice<'a>(
            &self,
            allocation: &'a Self::Allocation,
        ) -> Result<&'a [u8], AllocationError> {
            if allocation.mapped_address == 0 || allocation.allocation.size() > isize::MAX as u64 {
                return Err(AllocationError::NotHostVisible);
            }
            Ok(unsafe {
                core::slice::from_raw_parts(
                    allocation.mapped_address as *const u8,
                    allocation.allocation.size() as usize,
                )
            })
        }

        fn mapped_slice_mut<'a>(
            &mut self,
            allocation: &'a mut Self::Allocation,
        ) -> Result<&'a mut [u8], AllocationError> {
            if allocation.mapped_address == 0 || allocation.allocation.size() > isize::MAX as u64 {
                return Err(AllocationError::NotHostVisible);
            }
            Ok(unsafe {
                core::slice::from_raw_parts_mut(
                    allocation.mapped_address as *mut u8,
                    allocation.allocation.size() as usize,
                )
            })
        }

        fn flush(
            &mut self,
            allocation: &mut Self::Allocation,
            offset: u64,
            size: u64,
        ) -> Result<(), AllocationError> {
            validate_range(allocation.allocation.size(), offset, size)?;
            if allocation.mapped_address == 0 {
                return Err(AllocationError::NotHostVisible);
            }
            let written = D3D12_RANGE {
                Begin: offset as usize,
                End: (offset + size) as usize,
            };
            unsafe {
                allocation.resource.Unmap(0, Some(&written));
            }
            let mut mapped: *mut c_void = ptr::null_mut();
            unsafe { allocation.resource.Map(0, None, Some(&mut mapped)) }
                .map_err(map_allocation_windows)?;
            allocation.mapped_address = mapped as usize;
            Ok(())
        }

        fn invalidate(
            &mut self,
            allocation: &mut Self::Allocation,
            offset: u64,
            size: u64,
        ) -> Result<(), AllocationError> {
            validate_range(allocation.allocation.size(), offset, size)?;
            if allocation.mapped_address == 0 {
                return Err(AllocationError::NotHostVisible);
            }
            let no_write = D3D12_RANGE { Begin: 0, End: 0 };
            unsafe {
                allocation.resource.Unmap(0, Some(&no_write));
            }
            let read = D3D12_RANGE {
                Begin: offset as usize,
                End: (offset + size) as usize,
            };
            let mut mapped: *mut c_void = ptr::null_mut();
            unsafe { allocation.resource.Map(0, Some(&read), Some(&mut mapped)) }
                .map_err(map_allocation_windows)?;
            allocation.mapped_address = mapped as usize;
            Ok(())
        }
        fn free(&mut self, allocation: Self::Allocation) -> Result<(), AllocationError> {
            if allocation.mapped_address != 0 {
                let no_write = D3D12_RANGE { Begin: 0, End: 0 };
                unsafe {
                    allocation.resource.Unmap(0, Some(&no_write));
                }
            }
            drop(allocation.resource);
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(allocation.allocation)
                .map_err(map_allocator)
        }

        fn retire(
            &mut self,
            allocation: Self::Allocation,
            completion: CompletionToken,
        ) -> Result<(), AllocationError> {
            self.retired.push(RetiredAllocation {
                allocation,
                completion,
            });
            Ok(())
        }

        fn reclaim(&mut self, queue: QueueKind, completed: u64) -> Result<usize, AllocationError> {
            let mut count = 0;
            let mut index = 0;
            while index < self.retired.len() {
                if self.retired[index].completion.queue == queue
                    && self.retired[index].completion.value <= completed
                {
                    let retired = self.retired.swap_remove(index);
                    self.free(retired.allocation)?;
                    count += 1;
                } else {
                    index += 1;
                }
            }
            Ok(count)
        }
    }

    impl BufferTransfer for NativeContext {
        fn copy_buffer(
            &mut self,
            source: &Self::Allocation,
            destination: &Self::Allocation,
            source_offset: u64,
            destination_offset: u64,
            size: u64,
        ) -> Result<CompletionToken, AllocationError> {
            validate_range(source.allocation.size(), source_offset, size)?;
            validate_range(destination.allocation.size(), destination_offset, size)?;
            let allocator: ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .map_err(map_allocation_windows)?;
            let list: ID3D12GraphicsCommandList = unsafe {
                self.device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            }
            .map_err(map_allocation_windows)?;
            unsafe {
                list.CopyBufferRegion(
                    &destination.resource,
                    destination_offset,
                    &source.resource,
                    source_offset,
                    size,
                )
            };
            unsafe { list.Close() }.map_err(map_allocation_windows)?;
            let command: ID3D12CommandList = list.cast().map_err(map_allocation_windows)?;
            unsafe { self.queue.ExecuteCommandLists(&[Some(command)]) };
            let value = self.next_fence;
            self.next_fence = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
            unsafe { self.queue.Signal(&self.fence, value) }.map_err(map_allocation_windows)?;
            let completion = CompletionToken::new(QueueKind::Transfer, value)
                .map_err(|_| AllocationError::NativeFailure)?;
            self.pending_copies
                .retain(|copy| copy.completion.value > unsafe { self.fence.GetCompletedValue() });
            self.pending_copies.push(PendingCopy {
                completion,
                _allocator: allocator,
                _list: list,
            });
            Ok(completion)
        }

        fn completed_transfer_value(&self) -> Result<u64, AllocationError> {
            let value = unsafe { self.fence.GetCompletedValue() };
            if value == u64::MAX {
                return Err(AllocationError::DeviceLost);
            }
            Ok(value)
        }
    }
    fn dx12_resource_state(access: ResourceAccess) -> D3D12_RESOURCE_STATES {
        match access {
            ResourceAccess::SampledRead | ResourceAccess::StorageRead => {
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                    | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            }
            ResourceAccess::StorageWrite | ResourceAccess::StorageReadWrite => {
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            }
            ResourceAccess::IndexRead => D3D12_RESOURCE_STATE_INDEX_BUFFER,
            ResourceAccess::IndirectRead => D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ResourceAccess::IndirectStorageRead => {
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                    | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            }
            ResourceAccess::IndirectStorageReadWrite => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ResourceAccess::ColorAttachmentWrite => D3D12_RESOURCE_STATE_RENDER_TARGET,
            ResourceAccess::DepthStencilRead => D3D12_RESOURCE_STATE_DEPTH_READ,
            ResourceAccess::DepthStencilWrite => D3D12_RESOURCE_STATE_DEPTH_WRITE,
            ResourceAccess::TransferRead => D3D12_RESOURCE_STATE_COPY_SOURCE,
            ResourceAccess::TransferWrite => D3D12_RESOURCE_STATE_COPY_DEST,
            ResourceAccess::Present => D3D12_RESOURCE_STATE_PRESENT,
        }
    }

    fn transition_barrier(
        resource: ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: core::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: core::mem::ManuallyDrop::new(Some(resource)),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    fn uav_barrier(resource: ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: core::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: core::mem::ManuallyDrop::new(Some(resource)),
                }),
            },
        }
    }

    unsafe fn copy_texture_to_readback(
        list: &ID3D12GraphicsCommandList,
        source: &ID3D12Resource,
        destination: &ID3D12Resource,
        footprint: windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    ) {
        let source = D3D12_TEXTURE_COPY_LOCATION {
            pResource: core::mem::ManuallyDrop::new(Some(source.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let destination = D3D12_TEXTURE_COPY_LOCATION {
            pResource: core::mem::ManuallyDrop::new(Some(destination.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        unsafe {
            list.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
        }
    }

    unsafe fn bind_dx12_compute_buffers(
        list: &ID3D12GraphicsCommandList,
        pipeline: &NativePipeline,
        bindings: &[NativeBufferBinding<'_>],
    ) -> Result<(), HalError> {
        for (index, (binding, writable)) in
            bindings.iter().zip(&pipeline.buffer_writable).enumerate()
        {
            if binding.writable != *writable
                || binding.offset >= binding.allocation.allocation.size()
            {
                return Err(HalError::InvalidArgument);
            }
            let address = unsafe {
                binding
                    .allocation
                    .resource
                    .GetGPUVirtualAddress()
                    .checked_add(binding.offset)
            }
            .ok_or(HalError::InvalidArgument)?;
            unsafe {
                if *writable {
                    list.SetComputeRootUnorderedAccessView(index as u32 + 1, address);
                } else {
                    list.SetComputeRootShaderResourceView(index as u32 + 1, address);
                }
            }
        }
        Ok(())
    }

    unsafe fn bind_dx12_graphics_buffers(
        list: &ID3D12GraphicsCommandList,
        pipeline: &NativePipeline,
        bindings: &[NativeBufferBinding<'_>],
    ) -> Result<(), HalError> {
        for (index, (binding, writable)) in
            bindings.iter().zip(&pipeline.buffer_writable).enumerate()
        {
            if binding.writable != *writable
                || binding.offset >= binding.allocation.allocation.size()
            {
                return Err(HalError::InvalidArgument);
            }
            let address = unsafe {
                binding
                    .allocation
                    .resource
                    .GetGPUVirtualAddress()
                    .checked_add(binding.offset)
            }
            .ok_or(HalError::InvalidArgument)?;
            unsafe {
                if *writable {
                    list.SetGraphicsRootUnorderedAccessView(index as u32 + 1, address);
                } else {
                    list.SetGraphicsRootShaderResourceView(index as u32 + 1, address);
                }
            }
        }
        Ok(())
    }

    impl Drop for NativeContext {
        fn drop(&mut self) {
            let _ = self.wait_idle();
            while let Some(retired) = self.retired.pop() {
                let _ = self.free(retired.allocation);
            }
            drop(self.allocator.take());
            unsafe {
                let _ = CloseHandle(self.fence_event);
            }
        }
    }

    fn adapter_id(desc: &windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_DESC3) -> [u8; 16] {
        let mut id = [0_u8; 16];
        id[0..4].copy_from_slice(&desc.VendorId.to_le_bytes());
        id[4..8].copy_from_slice(&desc.DeviceId.to_le_bytes());
        id[8..12].copy_from_slice(&desc.AdapterLuid.LowPart.to_le_bytes());
        id[12..16].copy_from_slice(&desc.AdapterLuid.HighPart.to_le_bytes());
        id
    }

    fn validate_range(length: u64, offset: u64, size: u64) -> Result<(), AllocationError> {
        if size == 0 || offset.checked_add(size).is_none_or(|end| end > length) {
            return Err(AllocationError::NativeFailure);
        }
        Ok(())
    }

    fn map_hal_allocation(error: HalError) -> AllocationError {
        match error {
            HalError::OutOfMemory => AllocationError::OutOfMemory,
            HalError::DeviceLost => AllocationError::DeviceLost,
            _ => AllocationError::NativeFailure,
        }
    }

    fn map_windows(_: windows::core::Error) -> HalError {
        HalError::NativeFailure
    }
    fn map_allocation_windows(error: windows::core::Error) -> AllocationError {
        if error.code() == windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED {
            AllocationError::DeviceLost
        } else {
            AllocationError::NativeFailure
        }
    }
    fn map_allocator(error: gpu_allocator::AllocationError) -> AllocationError {
        match error {
            gpu_allocator::AllocationError::OutOfMemory => AllocationError::OutOfMemory,
            _ => AllocationError::NativeFailure,
        }
    }

    fn map_allocator_hal(error: gpu_allocator::AllocationError) -> HalError {
        match error {
            gpu_allocator::AllocationError::OutOfMemory => HalError::OutOfMemory,
            _ => HalError::NativeFailure,
        }
    }

    const _: () = assert!(IDXGIAdapter4::IID.data1 != 0);
}
