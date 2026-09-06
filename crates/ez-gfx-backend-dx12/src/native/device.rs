use super::{
    AdapterCapabilities, AdapterClass, AdapterInfo, AllocationError, AllocationSizes, Allocator,
    AllocatorCreateDesc, BACKEND, CompressionSupport, CreateDXGIFactory1, CreateEventW,
    D3D_FEATURE_LEVEL_12_1, D3D_SHADER_MODEL_6_5, D3D12_COMMAND_LIST_TYPE_COPY,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_FEATURE_D3D12_OPTIONS,
    D3D12_FEATURE_DATA_D3D12_OPTIONS, D3D12_FEATURE_DATA_SHADER_MODEL, D3D12_FEATURE_SHADER_MODEL,
    D3D12_FENCE_FLAG_NONE, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_RANGE, D3D12_SAMPLER_DESC,
    D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12CreateDevice, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_ERROR_NOT_FOUND, DXGI_ERROR_UNSUPPORTED,
    DeferredNativeResource, DeferredResource, HalError, ID3D12CommandQueue, ID3D12DescriptorHeap,
    ID3D12Device, ID3D12DeviceVersion, ID3D12Fence, IDXGIAdapter4, IDXGIFactory4, INFINITE,
    Interface, MemoryAllocator, NativeContext, NativeSurface, QueueKind, SemanticProfile,
    TEXTURE_DESCRIPTOR_CAPACITY, WaitForSingleObject, adapter_id, create_frame_slots,
    map_allocator, map_windows, transfer,
};

fn initialize_context(
    adapter: IDXGIAdapter4,
    device: ID3D12Device,
    adapter_info: AdapterInfo,
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
    Ok(NativeContext {
        adapter,
        device,
        queue,
        fence,
        fence_event,
        transfer_fence,
        texture_fence,
        transfer_worker: Some(transfer_worker),
        texture_worker: Some(texture_worker),
        next_fence: 1,
        idle_drained: false,
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
    })
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
            // SAFETY: GetDesc3 uses adapter's retained IDXGIAdapter4 receiver, and windows-rs provides correctly sized DXGI_ADAPTER_DESC3 out storage for the call.
            let description = unsafe { adapter.GetDesc3() }?;
            if !allow_software && description.Flags.contains(DXGI_ADAPTER_FLAG3_SOFTWARE) {
                continue;
            }
            let mut device: Option<ID3D12Device> = None;
            // SAFETY: D3D12CreateDevice receives adapter through its IDXGIAdapter4 wrapper and a writable, aligned Option<ID3D12Device> out slot that lives through the call.
            if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_1, &raw mut device) }
                .is_err()
            {
                continue;
            }
            let Some(device) = device else {
                continue;
            };
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            // SAFETY: CheckFeatureSupport(D3D12_OPTIONS) receives a writable, aligned options pointer with the exact D3D12_FEATURE_DATA_D3D12_OPTIONS size, and options lives through the call.
            if unsafe {
                device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS,
                    (&raw mut options).cast(),
                    u32::try_from(core::mem::size_of_val(&options)).unwrap_or(u32::MAX),
                )
            }
            .is_err()
            {
                continue;
            }
            let mut shader_model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_5,
            };
            // SAFETY: CheckFeatureSupport(SHADER_MODEL) receives a writable, aligned shader_model pointer with the exact D3D12_FEATURE_DATA_SHADER_MODEL size, and shader_model lives through the call.
            if unsafe {
                device.CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut shader_model).cast(),
                    u32::try_from(core::mem::size_of_val(&shader_model)).unwrap_or(u32::MAX),
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
                AdapterInfo::new(BACKEND, stable_id, name, driver, class, capabilities).map_err(
                    |_| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL),
                )?;
            return initialize_context(adapter, device, adapter_info);
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
    /// Returns `HalError::InvalidArgument` if the surface window handle is zero.
    pub fn init_device(&self, surface: &NativeSurface) -> Result<AdapterInfo, HalError> {
        if surface.window == 0 {
            return Err(HalError::InvalidArgument);
        }
        Ok(self.adapter_info.clone())
    }

    /// Waits for submitted native work and reclaims completed deferred resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the fence counter overflows, fence signaling or event registration fails, or deferred resource reclamation fails.
    pub fn wait_idle(&mut self) -> Result<(), HalError> {
        self.idle_drained = false;
        let mut native_drained = true;
        let mut worker_failed = false;
        for worker in [&mut self.transfer_worker, &mut self.texture_worker] {
            if let Some(worker) = worker {
                if worker.flush().is_err() {
                    // Terminal cleanup drains actual COPY/DIRECT submissions, not
                    // the application completion a failed callback never signaled.
                    worker.shutdown();
                    native_drained &= worker.drained();
                    worker_failed = true;
                }
            } else {
                worker_failed = true;
                native_drained = false;
            }
        }
        let value = self.next_fence;
        self.next_fence = self
            .next_fence
            .checked_add(1)
            .ok_or(HalError::NativeFailure)?;
        // SAFETY: Signal uses the command queue and fence created from the same ID3D12Device and retained in self; value is the next monotonically allocated fence value.
        unsafe { self.queue.Signal(&self.fence, value) }.map_err(map_windows)?;
        // SAFETY: GetCompletedValue takes no raw arguments, and self.fence's wrapper retains the ID3D12Fence COM receiver throughout the call.
        if unsafe { self.fence.GetCompletedValue() } < value {
            // SAFETY: SetEventOnCompletion receives the CreateEventW event stored in self, and exclusive self access keeps that HANDLE and the ID3D12Fence receiver retained through the call.
            unsafe { self.fence.SetEventOnCompletion(value, self.fence_event) }
                .map_err(map_windows)?;
            // SAFETY: WaitForSingleObject receives fence_event, a waitable event HANDLE returned by CreateEventW, and exclusive self access keeps it unclosed for the wait.
            unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
        }
        // SAFETY: the retained graphics fence reports device removal as u64::MAX.
        if unsafe { self.fence.GetCompletedValue() } == u64::MAX {
            return Err(HalError::DeviceLost);
        }
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
        // SAFETY: the transfer fence remains live while polling its worker's terminal value.
        let mut transfer_completed = unsafe { self.transfer_fence.GetCompletedValue() };
        while transfer_value != 0 && transfer_completed < transfer_value {
            if self
                .transfer_worker
                .as_ref()
                .is_some_and(ez_gfx_hal::TransferWorker::failed)
            {
                return Err(HalError::NativeFailure);
            }
            // SAFETY: the worker signals every accepted transfer value, and this short timed wait
            // lets the caller observe worker or device failure rather than hanging indefinitely.
            unsafe {
                self.transfer_fence
                    .SetEventOnCompletion(transfer_value, self.fence_event)
            }
            .map_err(map_windows)?;
            // SAFETY: `fence_event` remains live and waitable throughout this method.
            unsafe { WaitForSingleObject(self.fence_event, 10) };
            // SAFETY: the transfer fence remains live after the timed event wait.
            transfer_completed = unsafe { self.transfer_fence.GetCompletedValue() };
        }
        if transfer_completed == u64::MAX {
            return Err(HalError::DeviceLost);
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
        // SAFETY: the texture fence remains live while polling its worker's terminal value.
        let mut texture_completed = unsafe { self.texture_fence.GetCompletedValue() };
        while texture_value != 0 && texture_completed < texture_value {
            if self
                .texture_worker
                .as_ref()
                .is_some_and(ez_gfx_hal::TransferWorker::failed)
            {
                return Err(HalError::NativeFailure);
            }
            // SAFETY: the worker signals every accepted texture-transfer value, and this timed wait
            // lets the caller observe worker or device failure rather than hanging indefinitely.
            unsafe {
                self.texture_fence
                    .SetEventOnCompletion(texture_value, self.fence_event)
            }
            .map_err(map_windows)?;
            // SAFETY: `fence_event` remains live and waitable throughout this method.
            unsafe { WaitForSingleObject(self.fence_event, 10) };
            // SAFETY: the texture fence remains live after the timed event wait.
            texture_completed = unsafe { self.texture_fence.GetCompletedValue() };
        }
        if texture_completed == u64::MAX {
            return Err(HalError::DeviceLost);
        }
        self.idle_drained = native_drained;
        if worker_failed {
            return Err(HalError::NativeFailure);
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
