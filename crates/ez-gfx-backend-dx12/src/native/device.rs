use super::{
    AdapterCapabilities, AdapterClass, AdapterInfo, AllocationError, AllocationSizes, Allocator,
    AllocatorCreateDesc, BACKEND, CompletionToken, CompressionSupport, CreateDXGIFactory1,
    CreateEventW, D3D_FEATURE_LEVEL_12_1, D3D_SHADER_MODEL_6_5, D3D12_COMMAND_LIST_TYPE_COPY,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_D3D12_OPTIONS7,
    D3D12_FEATURE_DATA_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
    D3D12_FEATURE_DATA_SHADER_MODEL, D3D12_FEATURE_SHADER_MODEL, D3D12_FENCE_FLAG_NONE,
    D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_MESH_SHADER_TIER, D3D12_MESH_SHADER_TIER_1,
    D3D12_MESH_SHADER_TIER_NOT_SUPPORTED, D3D12_RANGE, D3D12_SAMPLER_DESC,
    D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12CreateDevice, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_ERROR_NOT_FOUND, DXGI_ERROR_UNSUPPORTED,
    DeferredNativeResource, DeferredResource, E_INVALIDARG, HalError, ID3D12CommandQueue,
    ID3D12DescriptorHeap, ID3D12Device, ID3D12DeviceVersion, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12GraphicsCommandList6, IDXGIAdapter4, IDXGIFactory4, INFINITE,
    Interface, MemoryAllocator, NativeContext, NativeSurface, QueueKind, SemanticProfile,
    ShaderCapabilities, TEXTURE_DESCRIPTOR_CAPACITY, WAIT_FAILED, WAIT_OBJECT_0,
    WaitForSingleObject, adapter_id, create_frame_slots, map_allocator, map_windows, transfer,
};

fn initialize_context(
    adapter: IDXGIAdapter4,
    device: ID3D12Device,
    adapter_info: AdapterInfo,
    mesh_shader_tier: D3D12_MESH_SHADER_TIER,
) -> windows::core::Result<NativeContext> {
    // SAFETY: CreateCommandQueue reads the initialized D3D12_COMMAND_QUEUE_DESC only during the call, while device retains the ID3D12Device COM receiver.
    let queue: ID3D12CommandQueue = unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })
    }?;
    // SAFETY: the descriptor requests a native copy queue from the same live device.
    let transfer_queue: ID3D12CommandQueue = unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY,
            ..Default::default()
        })
    }?;
    // SAFETY: the descriptor requests a second native copy queue from the same live device.
    let texture_queue: ID3D12CommandQueue = unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY,
            ..Default::default()
        })
    }?;
    // SAFETY: CreateFence receives defined value and flag constants, while device retains the ID3D12Device receiver and windows-rs provides ID3D12Fence out storage.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
    // SAFETY: the transfer fence is created by and retained with the same live device.
    let transfer_fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
    // SAFETY: the texture fence is created by and retained with the same live device.
    let texture_fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
    // SAFETY: CreateEventW permits null security-attribute and name pointers, and windows-rs provides HANDLE result storage for the call.
    let fence_event = unsafe { CreateEventW(None, false, false, None) }?;
    let allocator = Allocator::new(&AllocatorCreateDesc {
        device: ID3D12DeviceVersion::Device(device.clone()),
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        allocation_sizes: AllocationSizes::new(
            DEFAULT_ALLOCATION_BLOCK_POLICY.initial_device,
            DEFAULT_ALLOCATION_BLOCK_POLICY.initial_host,
        )
        .with_max_device_memblock_size(DEFAULT_ALLOCATION_BLOCK_POLICY.maximum_device)
        .with_max_host_memblock_size(DEFAULT_ALLOCATION_BLOCK_POLICY.maximum_host),
    })
    .map_err(|_| windows::core::Error::from_hresult(windows::Win32::Foundation::E_OUTOFMEMORY))?;
    // SAFETY: CreateDescriptorHeap reads the initialized CBV_SRV_UAV heap descriptor only during the call, while device retains the ID3D12Device COM receiver.
    let descriptors: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: TEXTURE_DESCRIPTOR_CAPACITY,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    }?;
    // SAFETY: GetDescriptorHandleIncrementSize takes no raw arguments; device retains the COM receiver and CBV_SRV_UAV is a defined descriptor-heap type.
    let descriptor_stride =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
    // SAFETY: CreateDescriptorHeap reads the initialized SAMPLER heap descriptor only during the call, while device retains the ID3D12Device COM receiver.
    let samplers: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: TEXTURE_DESCRIPTOR_CAPACITY,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    }?;
    // SAFETY: GetDescriptorHandleIncrementSize takes no raw arguments; device retains the COM receiver and SAMPLER is a defined descriptor-heap type.
    let sampler_stride =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) }
            as usize;
    let sampler_desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    // SAFETY: GetCPUDescriptorHandleForHeapStart is called on the SAMPLER heap created above, whose ID3D12DescriptorHeap reference is retained through the call.
    let mut sampler = unsafe { samplers.GetCPUDescriptorHandleForHeapStart() };
    for _ in 0..TEXTURE_DESCRIPTOR_CAPACITY {
        // SAFETY: CreateSampler reads sampler_desc during the call, and sampler denotes the current slot among the heap's TEXTURE_DESCRIPTOR_CAPACITY stride-spaced CPU descriptors.
        unsafe { device.CreateSampler(&raw const sampler_desc, sampler) };
        sampler.ptr += sampler_stride;
    }
    let frame_slots = create_frame_slots(&device)?;
    let transfer_worker = transfer::start_worker(
        &device,
        transfer_queue.clone(),
        queue.clone(),
        transfer_fence.clone(),
    )
    .map_err(|_| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))?;
    let texture_worker =
        transfer::start_worker(&device, texture_queue, queue.clone(), texture_fence.clone())
            .map_err(|_| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))?;
    let context = NativeContext {
        adapter,
        device,
        mesh_shader_tier,
        queue,
        fence,
        fence_event,
        transfer_fence,
        texture_fence,
        transfer_worker: Some(transfer_worker),
        texture_worker: Some(texture_worker),
        next_fence: 1,
        idle_drained: false,
        #[cfg(test)]
        wait_idle_failure: None,
        allocator: Some(allocator),
        next_transfer_fence: 1,
        next_texture_fence: 1,
        retired: Vec::new(),
        texture_staging: ez_gfx_hal::ReusableStagingPool::new(256),
        frame_slots,
        frame_cursor: 0,
        deferred: Vec::new(),
        adapter_info,
        descriptors,
        descriptor_stride,
        samplers,
        sampler_stride: u32::try_from(sampler_stride).unwrap_or(u32::MAX),
    };
    // The cached tier and the normalized stages derive from the same probe;
    // a mismatch would admit an adapter whose execution gate disagrees.
    debug_assert_eq!(
        context.mesh_shader_tier.0 >= D3D12_MESH_SHADER_TIER_1.0,
        context.adapter_info.capabilities().shader_stages.mesh,
        "cached mesh tier agrees with normalized shader stages"
    );
    Ok(context)
}

/// Maps the optional OPTIONS7 query result to a mesh-shader tier.
///
/// `E_INVALIDARG` means the runtime predates the OPTIONS7 revision, so
/// neither the tier query nor the mesh-dispatch command-list interface
/// exists, and maps to unsupported. Any other failure is a genuine device
/// problem and propagates to the caller.
///
/// # Errors
///
/// Returns the query failure unless it is `E_INVALIDARG`.
fn mesh_tier_from_options7(
    result: windows::core::Result<D3D12_FEATURE_DATA_D3D12_OPTIONS7>,
) -> windows::core::Result<D3D12_MESH_SHADER_TIER> {
    match result {
        Ok(options) => Ok(options.MeshShaderTier),
        Err(error) if error.code() == E_INVALIDARG => Ok(D3D12_MESH_SHADER_TIER_NOT_SUPPORTED),
        Err(error) => Err(error),
    }
}

/// Mesh execution records `DispatchMesh` through this interface; the OPTIONS7
/// probe below already gates its availability on the same runtime floor, so a
/// cast failure here reports a genuine device problem.
pub(super) fn mesh_dispatch_list(
    list: &ID3D12GraphicsCommandList,
) -> windows::core::Result<ID3D12GraphicsCommandList6> {
    list.cast()
}

/// Probes one DXGI adapter for identity and capabilities, creating a temporary
/// device for feature queries. The device is returned for context creation;
/// enumeration drops it. Software adapters classify as `Software` (not by
/// video memory) so catalog policy can gate them.
///
/// Returns `Ok(None)` for adapters that cannot host D3D12 at all (undescribed
/// or device-less); callers skip those without failing the walk. Every other
/// failure is a genuine device problem and propagates through `Err`.
///
/// The mesh-shader tier comes from the optional `D3D12_FEATURE_D3D12_OPTIONS7`
/// query: tier 1 enables both task (amplification) and mesh stages. The tier
/// is returned alongside the device so context creation caches the native
/// probe result.
///
/// # Errors
///
/// Returns an error if required feature queries or adapter metadata
/// construction fails.
fn describe_adapter(
    adapter: &IDXGIAdapter4,
) -> windows::core::Result<Option<(AdapterInfo, ID3D12Device, D3D12_MESH_SHADER_TIER)>> {
    // SAFETY: GetDesc3 uses adapter's retained IDXGIAdapter4 receiver, and windows-rs provides correctly sized DXGI_ADAPTER_DESC3 out storage for the call.
    let Ok(description) = (unsafe { adapter.GetDesc3() }) else {
        // Undescribable adapters (removed, non-DXGI) skip the walk, never fail it.
        return Ok(None);
    };
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: D3D12CreateDevice receives adapter through its IDXGIAdapter4 wrapper and a writable, aligned Option<ID3D12Device> out slot that lives through the call.
    if (unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_1, &raw mut device) }).is_err() {
        // Adapters without a D3D12 device (basic display, pre-12.1) skip the
        // walk; only created devices face the required queries below.
        return Ok(None);
    }
    let Some(device) = device else {
        return Ok(None);
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: CheckFeatureSupport(D3D12_OPTIONS) receives a writable, aligned options pointer with the exact D3D12_FEATURE_DATA_D3D12_OPTIONS size, and options lives through the call.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut options).cast(),
            u32::try_from(core::mem::size_of_val(&options)).unwrap_or(u32::MAX),
        )
    }?;
    let mut shader_model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_5,
    };
    // SAFETY: CheckFeatureSupport(SHADER_MODEL) receives a writable, aligned shader_model pointer with the exact D3D12_FEATURE_DATA_SHADER_MODEL size, and shader_model lives through the call.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut shader_model).cast(),
            u32::try_from(core::mem::size_of_val(&shader_model)).unwrap_or(u32::MAX),
        )
    }?;
    let mut mesh_options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    // SAFETY: CheckFeatureSupport(D3D12_OPTIONS7) receives a writable, aligned mesh_options pointer with the exact D3D12_FEATURE_DATA_D3D12_OPTIONS7 size, and mesh_options lives through the call.
    let options7 = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&raw mut mesh_options).cast(),
                u32::try_from(core::mem::size_of_val(&mesh_options)).unwrap_or(u32::MAX),
            )
            .map(|()| mesh_options)
    };
    let mesh_tier = mesh_tier_from_options7(options7)?;
    // Tier 1 is the only supported tier in this SDK revision; compare by
    // value floor so a future higher tier still implies both stages.
    let mesh_supported = mesh_tier.0 >= D3D12_MESH_SHADER_TIER_1.0;
    let shader_stages = ShaderCapabilities::normalized(mesh_supported, mesh_supported);
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
        shader_stages,
        timeline_synchronization: true,
        resource_aliasing: true,
        dynamic_rendering: true,
        presentation: true,
        compression: CompressionSupport::BC,
    };
    let stable_id = adapter_id(&description);
    let name = String::from_utf16_lossy(&description.Description)
        .trim_end_matches('\0')
        .to_owned();
    let driver = format!(
        "luid-{:08x}-{:08x}",
        description.AdapterLuid.HighPart, description.AdapterLuid.LowPart
    );
    // The software flag decides the class: WARP reports no dedicated memory
    // but must still gate on software policy, never on video-memory heuristics.
    let class = if description.Flags.contains(DXGI_ADAPTER_FLAG3_SOFTWARE) {
        AdapterClass::Software
    } else if description.DedicatedVideoMemory > 0 {
        AdapterClass::Discrete
    } else {
        AdapterClass::Integrated
    };
    let adapter_info = AdapterInfo::new(BACKEND, stable_id, name, driver, class, capabilities)
        .map_err(|_| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))?;
    Ok(Some((adapter_info, device, mesh_tier)))
}

impl NativeContext {
    /// Notes completed-frame reclamation for the shared polling path.
    ///
    /// DX12 needs no software reap: the descriptor gate reads the live graphics
    /// fence on every poll, so completed submissions already unblock publication.
    /// This hook keeps the shared dispatch uniform across backends.
    ///
    /// # Errors
    ///
    /// Never returns an error; the `Result` keeps the shared dispatch uniform.
    pub fn poll_frame_completion(&mut self) -> Result<(), HalError> {
        Ok(())
    }
    /// Returns the most recently submitted graphics-frame token.
    pub fn last_frame_completion(&self) -> Option<CompletionToken> {
        CompletionToken::new(QueueKind::Graphics, self.next_fence.saturating_sub(1)).ok()
    }

    /// Returns the completed graphics fence value.
    ///
    /// # Errors
    ///
    /// DX12 reports device removal by returning the reserved failure fence value.
    pub fn completed_frame_value(&mut self) -> Result<u64, AllocationError> {
        // SAFETY: the context retains the graphics fence through this query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            return Err(AllocationError::DeviceLost);
        }
        Ok(completed)
    }

    /// Enumeration exhaustion reports `DXGI_ERROR_UNSUPPORTED`; software adapters are ignored unless explicitly allowed.
    ///
    /// # Errors
    ///
    /// Returns an error if DXGI factory creation, adapter enumeration or querying, adapter metadata construction, or native context initialization fails, or if no adapter meets the required capabilities.
    pub fn create_default(allow_software: bool) -> windows::core::Result<Self> {
        // SAFETY: CreateDXGIFactory1 takes no input pointers, and windows-rs provides correctly typed IDXGIFactory4 out storage and adopts the returned COM reference.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }?;
        let mut index = 0;
        loop {
            // SAFETY: EnumAdapters1 uses factory's retained IDXGIFactory4 receiver, and windows-rs provides correctly typed adapter out storage for the call.
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
            // Adapter-local inability to describe or create a D3D12 device remains skippable;
            // every failure from a created device's required queries propagates.
            let Some((adapter_info, device, mesh_shader_tier)) = describe_adapter(&adapter)? else {
                continue;
            };
            if !allow_software && adapter_info.class() == AdapterClass::Software {
                continue;
            }
            if SemanticProfile::V1
                .admit(adapter_info.capabilities())
                .is_err()
            {
                continue;
            }
            return initialize_context(adapter, device, adapter_info, mesh_shader_tier);
        }
    }

    /// Enumerates every describable DXGI adapter, admitted or not.
    ///
    /// Rejected adapters stay listed so rejection diagnostics can name them.
    /// Exhaustion ends the walk; one undescribable adapter skips itself.
    ///
    /// # Errors
    ///
    /// Returns an error if DXGI factory creation, adapter enumeration, or a created device's
    /// required capability query fails.
    pub fn enumerate_adapters() -> windows::core::Result<Vec<AdapterInfo>> {
        // SAFETY: CreateDXGIFactory1 takes no input pointers, and windows-rs provides correctly typed IDXGIFactory4 out storage and adopts the returned COM reference.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }?;
        let mut adapters = Vec::new();
        let mut index = 0;
        loop {
            // SAFETY: EnumAdapters1 uses factory's retained IDXGIFactory4 receiver, and windows-rs provides correctly typed adapter out storage for the call.
            let raw = match unsafe { factory.EnumAdapters1(index) } {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => return Ok(adapters),
                Err(error) => return Err(error),
            };
            let Ok(adapter) = raw.cast::<IDXGIAdapter4>() else {
                index += 1;
                continue;
            };
            index += 1;
            if let Some((adapter_info, _, _)) = describe_adapter(&adapter)? {
                adapters.push(adapter_info);
            }
        }
    }

    /// Creates a context for one explicitly selected adapter.
    ///
    /// Ranking is bypassed but admission never is: an unknown identity fails
    /// `InvalidArgument`, disallowed software fails `InvalidArgument`, and a
    /// matched but inadmissible adapter fails `Unsupported`.
    ///
    /// # Errors
    ///
    /// Returns an error if DXGI factory creation, adapter querying, or native
    /// context initialization fails, or if no enumerated adapter matches the
    /// requested identity under admission policy.
    pub fn create_for_adapter(stable_id: [u8; 16], allow_software: bool) -> Result<Self, HalError> {
        // SAFETY: CreateDXGIFactory1 takes no input pointers, and windows-rs provides correctly typed IDXGIFactory4 out storage and adopts the returned COM reference.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }.map_err(map_windows)?;
        let mut index = 0;
        loop {
            // SAFETY: EnumAdapters1 uses factory's retained IDXGIFactory4 receiver, and windows-rs provides correctly typed adapter out storage for the call.
            let adapter = match unsafe { factory.EnumAdapters1(index) } {
                Ok(adapter) => adapter.cast::<IDXGIAdapter4>().map_err(map_windows)?,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => {
                    return Err(HalError::InvalidArgument);
                }
                Err(error) => return Err(map_windows(error)),
            };
            index += 1;
            // Preserve adapter-local skips, but never turn a created device's query failure into
            // a misleading unknown-adapter result.
            let Some((adapter_info, device, mesh_shader_tier)) =
                describe_adapter(&adapter).map_err(map_windows)?
            else {
                continue;
            };
            if adapter_info.stable_id() != stable_id {
                continue;
            }
            // Explicit selection bypasses ranking but never bypasses admission.
            if adapter_info.class() == AdapterClass::Software && !allow_software {
                return Err(HalError::InvalidArgument);
            }
            if SemanticProfile::V1
                .admit(adapter_info.capabilities())
                .is_err()
            {
                return Err(HalError::Unsupported);
            }
            return initialize_context(adapter, device, adapter_info, mesh_shader_tier)
                .map_err(map_windows);
        }
    }

    /// Reports on-demand allocator and device-memory telemetry.
    ///
    /// Calls `generate_report`, which allocates; never call per frame. Counts
    /// and sizes saturate instead of wrapping. Swapchain and depth storage are
    /// surface-owned on this backend, so those estimates report zero here.
    pub fn memory_telemetry(&self) -> ez_gfx_hal::BackendMemoryTelemetry {
        // Report generation walks live blocks, so this observes the allocator
        // exactly once per explicit query rather than sampling per frame.
        let allocator = self.allocator.as_ref().map(|allocator| {
            let report = allocator.generate_report();
            ez_gfx_hal::AllocatorTelemetry {
                live_bytes: report.total_allocated_bytes,
                block_bytes: report.total_capacity_bytes,
                block_count: u32::try_from(report.blocks.len()).unwrap_or(u32::MAX),
                allocation_count: u32::try_from(report.allocations.len()).unwrap_or(u32::MAX),
            }
        });
        ez_gfx_hal::BackendMemoryTelemetry {
            allocator,
            swapchain_images: 0,
            swapchain_extent: (0, 0),
            swapchain_format: 0,
            swapchain_bytes: 0,
            depth_bytes: 0,
            frame_slots: u32::try_from(self.frame_slots.len()).unwrap_or(u32::MAX),
        }
    }

    /// Returns the immutable identity and capabilities of the admitted adapter.
    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    /// The surface's HWND remains host-owned; admission already occurred during context creation.
    ///
    /// # Errors
    ///
    /// Returns an error when the surface window handle is zero or DXGI capability probing fails.
    pub fn init_device(&self, surface: &mut NativeSurface) -> Result<AdapterInfo, HalError> {
        if surface.window == 0 {
            return Err(HalError::InvalidArgument);
        }
        // Probe before the first present so the public surface capability snapshot is authoritative.
        // SAFETY: factory creation takes no caller-provided pointers.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }.map_err(map_windows)?;
        surface.allow_tearing = super::surface::factory_allows_tearing(&factory);
        Ok(self.adapter_info.clone())
    }

    fn wait_for_fence(
        &self,
        fence: &ID3D12Fence,
        value: u64,
        timeout: u32,
    ) -> Result<(), HalError> {
        // SAFETY: the context retains this device fence throughout every status read.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed == u64::MAX {
            return Err(HalError::DeviceLost);
        }
        if completed >= value {
            return Ok(());
        }

        // SAFETY: the context keeps both the fence and its waitable event alive through the wait.
        unsafe { fence.SetEventOnCompletion(value, self.fence_event) }.map_err(map_windows)?;
        // SAFETY: fence_event is a live waitable event retained by this context.
        let status = unsafe { WaitForSingleObject(self.fence_event, timeout) };
        if status == WAIT_FAILED {
            return Err(HalError::NativeFailure);
        }
        if status != WAIT_OBJECT_0 {
            return Err(HalError::NativeFailure);
        }

        // An event wake is not accepted as drain proof until the fence itself confirms completion.
        // SAFETY: the retained fence remains live after the event wake.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed == u64::MAX {
            Err(HalError::DeviceLost)
        } else if completed < value {
            Err(HalError::NativeFailure)
        } else {
            Ok(())
        }
    }

    /// Waits for submitted native work and reclaims completed deferred resources.
    ///
    /// # Errors
    ///
    /// Returns an error if fence signaling, event registration/waiting, post-wake completion
    /// confirmation, worker drain, or deferred resource reclamation fails.
    pub fn wait_idle(&mut self) -> Result<(), HalError> {
        self.idle_drained = false;
        let mut native_drained = true;
        let mut worker_failed = false;
        let mut worker_lost = false;
        for worker in [&mut self.transfer_worker, &mut self.texture_worker] {
            if let Some(worker) = worker {
                if let Err(error) = worker.flush() {
                    // Terminal cleanup drains actual COPY/DIRECT submissions, not
                    // the application completion a failed callback never signaled.
                    worker.shutdown();
                    native_drained &= worker.drained();
                    worker_failed = true;
                    worker_lost |= error == ez_gfx_hal::TransferWorkerError::DeviceLost
                        || worker.device_lost();
                }
            } else {
                worker_failed = true;
                native_drained = false;
            }
        }
        #[cfg(test)]
        if let Some(error) = self.wait_idle_failure.take() {
            // Fault injection occupies the native-wait failure point, before release is proven.
            return Err(error);
        }
        let value = self.next_fence;
        self.next_fence = self
            .next_fence
            .checked_add(1)
            .ok_or(HalError::NativeFailure)?;
        // SAFETY: Signal uses the queue and fence created from the same retained D3D12 device.
        unsafe { self.queue.Signal(&self.fence, value) }.map_err(map_windows)?;
        self.wait_for_fence(&self.fence, value, INFINITE)?;
        // SAFETY: both transfer fences belong to this live device and their workers retain them.
        let transfer_value = if self
            .transfer_worker
            .as_ref()
            .is_some_and(ez_gfx_hal::TransferWorker::failed)
        {
            0
        } else {
            self.next_transfer_fence.saturating_sub(1)
        };
        if transfer_value != 0 {
            if let Some(error) = self
                .transfer_worker
                .as_ref()
                .and_then(ez_gfx_hal::TransferWorker::terminal_error)
            {
                return Err(error.to_hal_error());
            }
            self.wait_for_fence(&self.transfer_fence, transfer_value, INFINITE)?;
        }
        // SAFETY: the texture fence remains live while the context owns it.
        let texture_value = if self
            .texture_worker
            .as_ref()
            .is_some_and(ez_gfx_hal::TransferWorker::failed)
        {
            0
        } else {
            self.next_texture_fence.saturating_sub(1)
        };
        if texture_value != 0 {
            if let Some(error) = self
                .texture_worker
                .as_ref()
                .and_then(ez_gfx_hal::TransferWorker::terminal_error)
            {
                return Err(error.to_hal_error());
            }
            self.wait_for_fence(&self.texture_fence, texture_value, INFINITE)?;
        }
        self.idle_drained = native_drained;
        if worker_failed {
            return Err(if worker_lost {
                HalError::DeviceLost
            } else {
                HalError::NativeFailure
            });
        }
        self.reclaim(QueueKind::Transfer, transfer_value)
            .map_err(|_| HalError::NativeFailure)?;
        self.reclaim(QueueKind::TextureTransfer, texture_value)
            .map_err(|_| HalError::NativeFailure)?;
        self.reclaim_deferred()
            .map_err(|_| HalError::NativeFailure)?;
        Ok(())
    }

    /// Reports whether the last idle attempt proved native storage safe to release.
    /// Call `wait_idle` immediately before consulting this terminal-cleanup status.
    pub const fn is_drained(&self) -> bool {
        self.idle_drained
    }

    ///
    /// # Errors
    ///
    /// Returns an error if immediately destroying a deferred allocation or texture finds no allocator or fails to free its allocation.
    pub(super) fn defer_resource(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        let fence_value = self.next_fence.saturating_sub(1);
        // SAFETY: GetCompletedValue takes no raw arguments, and self.fence's wrapper retains the ID3D12Fence COM receiver throughout the call.
        if fence_value == 0 || unsafe { self.fence.GetCompletedValue() } >= fence_value {
            return self.destroy_deferred_now(resource);
        }
        self.deferred.push(DeferredNativeResource {
            fence_value,
            resource,
        });
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if destroying a completed deferred allocation or texture finds no allocator or fails to free its allocation.
    pub(super) fn reclaim_deferred(&mut self) -> Result<(), AllocationError> {
        // SAFETY: GetCompletedValue takes no raw arguments, and self.fence's wrapper retains the ID3D12Fence COM receiver throughout the call.
        let completed = unsafe { self.fence.GetCompletedValue() };
        let mut ready = Vec::new();
        let mut index = 0;
        while index < self.deferred.len() {
            if self.deferred[index].fence_value <= completed {
                ready.push(self.deferred.swap_remove(index).resource);
            } else {
                index += 1;
            }
        }
        for resource in ready {
            self.destroy_deferred_now(resource)?;
        }
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if destroying an allocation or texture finds no allocator or fails to free its allocation.
    pub(super) fn destroy_deferred_now(
        &mut self,
        resource: DeferredResource,
    ) -> Result<(), AllocationError> {
        match resource {
            DeferredResource::Allocation(allocation) => {
                if allocation.mapped_address != 0 {
                    let no_write = D3D12_RANGE { Begin: 0, End: 0 };
                    // SAFETY: mapped_address != 0 records a mapping of allocation.resource subresource 0, and no_write remains initialized and readable until Unmap returns before the resource is dropped.
                    unsafe { allocation.resource.Unmap(0, Some(&raw const no_write)) };
                }
                drop(allocation.resource);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(allocation.allocation)
                    .map_err(|error| map_allocator(&error))
            }
            DeferredResource::Pipeline(pipeline) => {
                drop(pipeline);
                Ok(())
            }
            DeferredResource::Texture(texture) => {
                // The MSAA render storage retires with the sampled image; its
                // RTV handle needs no destroy, dying with the shared heap.
                if let Some(msaa) = texture.msaa {
                    drop(msaa.resource);
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(msaa.allocation)
                        .map_err(|error| map_allocator(&error))?;
                }
                drop(texture.resource);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(texture.allocation)
                    .map_err(|error| map_allocator(&error))
            }
        }
    }
}

#[cfg(test)]
mod adapter_tests {
    use super::super::{NativeMeshPipelineDesc, NativeShader};
    use super::*;
    use ez_gfx_hal::{BlendMode, CullMode, FrontFace, MeshPipelineState};
    use std::collections::BTreeSet;

    #[test]
    fn enumeration_reports_unique_named_adapters() {
        // No surface is created, shown, or activated by this test.
        let adapters = NativeContext::enumerate_adapters().expect("DXGI enumerates adapters");
        assert!(!adapters.is_empty());
        let mut identities = BTreeSet::new();
        for adapter in &adapters {
            assert_ne!(adapter.stable_id(), [0; 16]);
            assert!(!adapter.name().is_empty());
            assert!(!adapter.driver().is_empty());
            assert!(identities.insert(adapter.stable_id()));
        }
    }

    #[test]
    fn explicit_selection_rejects_unknown_identity() {
        // No surface is created, shown, or activated by this test.
        assert_eq!(
            NativeContext::create_for_adapter([0xA5; 16], false).map(|_| ()),
            Err(HalError::InvalidArgument)
        );
    }
    #[test]
    fn explicit_selection_admits_enumerated_adapter() {
        // No surface is created, shown, or activated by this test.
        let adapters = NativeContext::enumerate_adapters().expect("DXGI enumerates adapters");
        let wanted = adapters.first().expect("at least one adapter").stable_id();
        let context = NativeContext::create_for_adapter(wanted, false)
            .expect("enumerated adapter initializes");
        assert_eq!(context.adapter_info().stable_id(), wanted);
    }

    #[test]
    fn options7_invalid_argument_means_mesh_is_unsupported() {
        let error = windows::core::Error::from_hresult(E_INVALIDARG);

        assert_eq!(
            mesh_tier_from_options7(Err(error)).unwrap(),
            D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
        );
    }

    #[test]
    fn options7_non_invalid_argument_failure_propagates() {
        let error = windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL);

        assert_eq!(
            mesh_tier_from_options7(Err(error.clone()))
                .unwrap_err()
                .code(),
            error.code()
        );
    }

    #[test]
    fn mesh_dispatch_accepts_valid_task_and_taskless_grids() {
        use super::super::check_mesh_dispatch;

        assert_eq!(
            check_mesh_dispatch(true, [4, 2, 1], [128, 1, 1], Some([32, 1, 1])),
            Ok(())
        );
        assert_eq!(
            check_mesh_dispatch(false, [4, 2, 1], [128, 1, 1], None),
            Ok(())
        );
    }

    #[test]
    fn mesh_dispatch_rejects_misshapen_grids_and_inconsistent_stages() {
        use super::super::check_mesh_dispatch;

        // Zero dimensions and over-ceiling grids fail.
        assert_eq!(
            check_mesh_dispatch(false, [0, 1, 1], [8, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
        assert_eq!(
            check_mesh_dispatch(
                false,
                [
                    super::super::D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION + 1,
                    1,
                    1,
                ],
                [8, 1, 1],
                None,
            ),
            Err(HalError::InvalidArgument)
        );
        // A task size without a task stage, or a missing one with it, describes
        // no executable dispatch.
        assert_eq!(
            check_mesh_dispatch(false, [1, 1, 1], [8, 1, 1], Some([8, 1, 1])),
            Err(HalError::InvalidArgument)
        );
        assert_eq!(
            check_mesh_dispatch(true, [1, 1, 1], [8, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
        // Zero thread dimensions fail before any ceiling comparison.
        assert_eq!(
            check_mesh_dispatch(true, [1, 1, 1], [8, 1, 1], Some([0, 1, 1])),
            Err(HalError::InvalidArgument)
        );
    }

    #[test]
    fn mesh_dispatch_enforces_spec_total_grid_and_thread_caps() {
        use super::super::check_mesh_dispatch;

        // The mesh-shader specification caps the DispatchMesh workgroup-count
        // product at 2^22: the boundary product dispatches, one step over fails.
        assert_eq!(
            check_mesh_dispatch(false, [4096, 1024, 1], [8, 1, 1], None),
            Ok(())
        );
        assert_eq!(
            check_mesh_dispatch(false, [4096, 4096, 1], [8, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
        // Amplification and mesh threadgroup sizes cap at 128 threads each.
        assert_eq!(
            check_mesh_dispatch(false, [1, 1, 1], [128, 1, 1], None),
            Ok(())
        );
        assert_eq!(
            check_mesh_dispatch(false, [1, 1, 1], [256, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
        assert_eq!(
            check_mesh_dispatch(true, [1, 1, 1], [8, 1, 1], Some([128, 1, 1])),
            Ok(())
        );
        assert_eq!(
            check_mesh_dispatch(true, [1, 1, 1], [8, 1, 1], Some([256, 1, 1])),
            Err(HalError::InvalidArgument)
        );
    }

    fn mesh_raster() -> MeshPipelineState {
        MeshPipelineState {
            cull: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            blend: BlendMode::None,
        }
    }

    fn mesh_describe<'a>(
        empty: &'a NativeShader,
        task: Option<(&'a NativeShader, usize)>,
    ) -> NativeMeshPipelineDesc<'a> {
        NativeMeshPipelineDesc {
            task,
            mesh: (empty, 0),
            fragment: (empty, 0),
            state: mesh_raster(),
            color_format: None,
            layouts: &[],
            depth_required: false,
            task_workgroup_size: task.map(|_| [1, 1, 1]),
            mesh_workgroup_size: [32, 1, 1],
        }
    }

    #[test]
    fn mesh_pipeline_rejects_unsupported_before_state_creation() {
        // No surface is created, shown, or activated by this test.
        let adapters = NativeContext::enumerate_adapters().expect("DXGI enumerates adapters");
        let wanted = adapters.first().expect("at least one adapter").stable_id();
        let context = NativeContext::create_for_adapter(wanted, false)
            .expect("enumerated adapter initializes");
        let stages = context.adapter_info().capabilities().shader_stages;
        // The public limits getter follows the same tier gate without allocation.
        assert_eq!(context.mesh_dispatch_limits(false).is_ok(), stages.mesh);
        assert_eq!(context.mesh_dispatch_limits(true).is_ok(), stages.task);
        // Tier-1 limits are the specified mesh-shader constants.
        if stages.mesh {
            let limits = context.mesh_dispatch_limits(false).expect("tier-1 limits");
            assert_eq!(limits.max_groups, [65_535; 3]);
            assert_eq!(limits.max_total_groups, 1 << 22);
            assert_eq!(limits.max_mesh_threads, 128);
            assert_eq!(limits.max_task_threads, 128);
        }
        // Empty shaders carry no products, so a tier-1 device fails on the
        // product index while an older one fails on the earlier tier gate.
        let empty = NativeShader {
            products: Vec::new(),
        };
        assert_eq!(
            context
                .create_mesh_pipeline(mesh_describe(&empty, None))
                .map(|_| ()),
            if stages.mesh {
                Err(HalError::InvalidArgument)
            } else {
                Err(HalError::Unsupported)
            }
        );
        // Tier 1 always implies the task stage, so the task case follows the
        // same gate: unsupported below tier 1, product index above it.
        assert_eq!(
            context
                .create_mesh_pipeline(mesh_describe(&empty, Some((&empty, 0))))
                .map(|_| ()),
            if stages.task {
                Err(HalError::InvalidArgument)
            } else {
                Err(HalError::Unsupported)
            }
        );
        // An inconsistent stage/size selection is invalid on any device.
        let mismatched = NativeMeshPipelineDesc {
            task: None,
            task_workgroup_size: Some([1, 1, 1]),
            ..mesh_describe(&empty, None)
        };
        assert_eq!(
            context.create_mesh_pipeline(mismatched).map(|_| ()),
            Err(HalError::InvalidArgument)
        );
        // Garbage bytes pass index checks without reaching native creation, so a
        // well-formed but oversized workgroup is unsupported on any device,
        // while a zero workgroup is malformed only where tier 1 is supported.
        let present = NativeShader {
            products: vec![vec![0xAA]],
        };
        let oversized = NativeMeshPipelineDesc {
            mesh: (&present, 0),
            fragment: (&present, 0),
            mesh_workgroup_size: [1024, 1, 1],
            ..mesh_describe(&present, None)
        };
        assert_eq!(
            context.create_mesh_pipeline(oversized).map(|_| ()),
            Err(HalError::Unsupported)
        );
        let empty_workgroup = NativeMeshPipelineDesc {
            mesh_workgroup_size: [0, 1, 1],
            ..mesh_describe(&present, None)
        };
        assert_eq!(
            context.create_mesh_pipeline(empty_workgroup).map(|_| ()),
            if stages.mesh {
                Err(HalError::InvalidArgument)
            } else {
                Err(HalError::Unsupported)
            }
        );
    }
}
