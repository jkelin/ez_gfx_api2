use super::{
    Allocation, AllocationCreateDesc, AllocationError, AllocationRequest, CompletionToken,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FILTER_ANISOTROPIC,
    D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
    D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_TRANSITION_BARRIER,
    D3D12_SAMPLER_DESC, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC, DeferredResource,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12GraphicsCommandList, ID3D12Resource, INFINITE,
    ImageMip, Interface, MemoryAllocator, MemoryClass, MemoryLocation, NativeAllocation,
    NativeContext, NativeTexture, QueueKind, SamplerAddressMode, SamplerFilter,
    TEXTURE_DESCRIPTOR_CAPACITY, TextureSamplerDesc, WaitForSingleObject, map_allocation_windows,
    map_allocator, validate_rgba8_mips,
};

fn validate_texture_request(mips: &[ImageMip<'_>], binding: u32) -> Result<(), AllocationError> {
    validate_rgba8_mips(mips).map_err(|_| AllocationError::ZeroSize)?;
    if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
        return Err(AllocationError::ZeroSize);
    }
    Ok(())
}

fn publish_texture(
    context: &NativeContext,
    resource: ID3D12Resource,
    allocation: Allocation,
    binding: u32,
    sampler_desc: TextureSamplerDesc,
    completions: Vec<CompletionToken>,
) -> (NativeTexture, Vec<CompletionToken>) {
    // SAFETY: `context.descriptors` retains the `ID3D12DescriptorHeap` COM reference while `GetCPUDescriptorHandleForHeapStart` returns that heap's base CPU handle.
    let mut handle = unsafe { context.descriptors.GetCPUDescriptorHandleForHeapStart() };
    handle.ptr += binding as usize * context.descriptor_stride as usize;
    // SAFETY: `CreateShaderResourceView` receives the resource created by `context.device` and the range-checked `binding` slot in `context.descriptors`; both resource and descriptor heap outlast the call.
    unsafe {
        context
            .device
            .CreateShaderResourceView(&resource, None, handle);
    };
    let filter = if sampler_desc.max_anisotropy > 1.0 {
        D3D12_FILTER_ANISOTROPIC
    } else {
        match (sampler_desc.min_filter, sampler_desc.mag_filter) {
            (SamplerFilter::Nearest, SamplerFilter::Nearest) => D3D12_FILTER_MIN_MAG_MIP_POINT,
            (SamplerFilter::Nearest, SamplerFilter::Linear) => {
                D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT
            }
            (SamplerFilter::Linear, SamplerFilter::Nearest) => {
                D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR
            }
            (SamplerFilter::Linear, SamplerFilter::Linear) => D3D12_FILTER_MIN_MAG_MIP_LINEAR,
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
        MaxAnisotropy: anisotropy_u32(sampler_desc.max_anisotropy),
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    // SAFETY: `context.samplers` retains the `ID3D12DescriptorHeap` COM reference while `GetCPUDescriptorHandleForHeapStart` returns that heap's base CPU handle.
    let mut sampler_handle = unsafe { context.samplers.GetCPUDescriptorHandleForHeapStart() };
    sampler_handle.ptr += binding as usize * context.sampler_stride as usize;
    // SAFETY: `CreateSampler` reads the initialized local `sampler` for this call and writes the descriptor to the range-checked `binding` slot in `context.samplers`.
    unsafe {
        context
            .device
            .CreateSampler(&raw const sampler, sampler_handle);
    };
    (
        NativeTexture {
            resource,
            allocation,
            binding,
        },
        completions,
    )
}

fn anisotropy_u32(value: f32) -> u32 {
    for level in 1_u16..16 {
        if value < f32::from(level + 1) {
            return u32::from(level);
        }
    }
    16
}

impl NativeContext {
    fn create_texture_resource(
        &mut self,
        mips: &[ImageMip<'_>],
    ) -> Result<(ID3D12Resource, Allocation, D3D12_RESOURCE_DESC), AllocationError> {
        let width = mips[0].width;
        let height = mips[0].height;
        let mip_count = u16::try_from(mips.len()).map_err(|_| AllocationError::NativeFailure)?;
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
            .map_err(|error| map_allocator(&error))?;
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: `CreatePlacedResource` receives the allocator-produced heap and offset for `desc`; `desc` and writable `resource` storage are initialized, aligned locals that outlast the call.
        if let Err(error) = unsafe {
            self.device.CreatePlacedResource(
                allocation.heap(),
                allocation.offset(),
                &raw const desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &raw mut resource,
            )
        } {
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
            return Err(map_allocation_windows(&error));
        }
        let Some(resource) = resource else {
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
            return Err(AllocationError::NativeFailure);
        };
        Ok((resource, allocation, desc))
    }
}

impl NativeContext {
    pub(super) fn destroy_unpublished_texture(
        &mut self,
        resource: ID3D12Resource,
        allocation: Allocation,
        upload: Option<NativeAllocation>,
    ) {
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

    fn populate_texture_upload(
        &mut self,
        mips: &[ImageMip<'_>],
        footprints: &[windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
        upload: &mut NativeAllocation,
        upload_size: u64,
    ) -> Result<(), AllocationError> {
        let target = self.mapped_slice_mut(upload)?;
        for (mip, footprint) in mips.iter().zip(footprints) {
            let row_bytes = usize::try_from(mip.width)
                .ok()
                .and_then(|width| width.checked_mul(4))
                .ok_or(AllocationError::NativeFailure)?;
            for row in 0..usize::try_from(mip.height).map_err(|_| AllocationError::NativeFailure)? {
                let source_start = row * row_bytes;
                let destination_start = usize::try_from(footprint.Offset)
                    .map_err(|_| AllocationError::NativeFailure)?
                    + row * footprint.Footprint.RowPitch as usize;
                target[destination_start..destination_start + row_bytes]
                    .copy_from_slice(&mip.bytes[source_start..source_start + row_bytes]);
            }
        }
        self.flush(upload, 0, upload_size)
    }

    /// Creates an RGBA8 mip chain and submits each level under a distinct fence value.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid RGBA8 mip chain or out-of-range binding, failed texture or upload setup, failed D3D12 resource, command, or fence operations, fence-value overflow, or failed synchronization or cleanup.
    pub fn create_texture_rgba8(
        &mut self,
        mips: &[ImageMip<'_>],
        binding: u32,
        sampler_desc: TextureSamplerDesc,
    ) -> Result<(NativeTexture, Vec<CompletionToken>), AllocationError> {
        validate_texture_request(mips, binding)?;
        let (resource, allocation, desc) = self.create_texture_resource(mips)?;
        let mut footprints = vec![
            windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            mips.len()
        ];
        let mut row_counts = vec![0_u32; mips.len()];
        let mut row_sizes = vec![0_u64; mips.len()];
        let mut upload_size = 0;
        // SAFETY: every output array has exactly `mips.len()` initialized slots.
        unsafe {
            self.device.GetCopyableFootprints(
                &raw const desc,
                0,
                u32::try_from(mips.len()).map_err(|_| AllocationError::NativeFailure)?,
                0,
                Some(footprints.as_mut_ptr()),
                Some(row_counts.as_mut_ptr()),
                Some(row_sizes.as_mut_ptr()),
                Some(&raw mut upload_size),
            );
        }
        let bucket =
            ez_gfx_hal::staging_bucket_size(upload_size, ez_gfx_hal::DEFAULT_STAGING_POLICY)
                .map_err(|_| AllocationError::OutOfMemory)?;
        let completed = self.completed_texture_transfer_value()?;
        for stale in self.texture_staging.trim(completed) {
            self.free(stale)?;
        }
        let request = AllocationRequest::new(bucket, 256, MemoryClass::Upload, true, None)
            .map_err(|_| AllocationError::ZeroSize)?;
        let mut upload =
            if let Some((_, upload)) = self.texture_staging.take(upload_size, completed) {
                upload
            } else {
                match self.allocate(request) {
                    Ok(upload) => upload,
                    Err(error) => {
                        self.destroy_unpublished_texture(resource, allocation, None);
                        return Err(error);
                    }
                }
            };
        if let Err(error) =
            self.populate_texture_upload(mips, &footprints, &mut upload, upload_size)
        {
            self.destroy_unpublished_texture(resource, allocation, Some(upload));
            return Err(error);
        }
        let first = self.next_texture_fence;
        self.next_texture_fence = first
            .checked_add(mips.len() as u64)
            .ok_or(AllocationError::NativeFailure)?;
        let completions = (first..self.next_texture_fence)
            .map(|value| CompletionToken::new(QueueKind::TextureTransfer, value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| AllocationError::NativeFailure)?;
        let completion = *completions.last().ok_or(AllocationError::NativeFailure)?;
        let value = completion.value;
        let submitted = self
            .texture_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .submit(super::transfer::Dx12TransferJob {
                value,
                bytes: upload_size,
                copy: super::transfer::Dx12TransferCopy::Texture {
                    source: upload.resource.clone(),
                    destination: resource.clone(),
                    footprints,
                },
            });
        if let Err(error) = submitted {
            self.destroy_unpublished_texture(resource, allocation, Some(upload));
            return Err(match error {
                ez_gfx_hal::TransferWorkerError::Full => AllocationError::OutOfMemory,
                ez_gfx_hal::TransferWorkerError::Failed => AllocationError::NativeFailure,
            });
        }
        let capacity = upload.allocation.size();
        self.texture_staging.put(capacity, upload, Some(completion));
        Ok(publish_texture(
            self,
            resource,
            allocation,
            binding,
            sampler_desc,
            completions,
        ))
    }

    /// Defers texture destruction until every referencing frame completes.
    ///
    /// # Errors
    ///
    /// Returns an error if `defer_resource` fails to queue the texture for deferred destruction.
    pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
        self.defer_resource(DeferredResource::Texture(texture))
    }

    /// Copies the shader-readable image through a GPU readback footprint and returns tightly packed RGBA8 rows.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions, allocation, synchronization, or the native copy fails.
    pub fn readback_texture_rgba8(
        &mut self,
        texture: &NativeTexture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, AllocationError> {
        // SAFETY: `GetDesc` reads the `ID3D12Resource` referenced by `texture`, whose COM reference remains stored in `texture` throughout the call.
        let desc = unsafe { texture.resource.GetDesc() };
        if desc.Width != u64::from(width) || desc.Height != height {
            return Err(AllocationError::NativeFailure);
        }
        let mut footprint =
            windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0;
        let mut row_size = 0;
        let mut total = 0;
        // SAFETY: `GetCopyableFootprints` reads initialized `desc` and writes one footprint plus the `rows`, `row_size`, and `total` locals, all of whose storage outlasts the call.
        unsafe {
            self.device.GetCopyableFootprints(
                &raw const desc,
                0,
                1,
                0,
                Some(&raw mut footprint),
                Some(&raw mut rows),
                Some(&raw mut row_size),
                Some(&raw mut total),
            );
        }
        let mut readback = self.allocate(
            AllocationRequest::new(total, 256, MemoryClass::Readback, true, None)
                .map_err(|_| AllocationError::ZeroSize)?,
        )?;
        // SAFETY: `CreateCommandAllocator` receives the defined `DIRECT` command-list type, and `self` retains the device COM reference throughout the call.
        let allocator: ID3D12CommandAllocator = unsafe {
            self.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|error| map_allocation_windows(&error))?;
        // SAFETY: `CreateCommandList` receives the newly created, unused `DIRECT` allocator from the same device; the allocator COM reference outlasts the call.
        let list: ID3D12GraphicsCommandList = unsafe {
            self.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
        }
        .map_err(|error| map_allocation_windows(&error))?;
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
        // SAFETY: `barrier.Type` selects its initialized `Transition` arm for the shader-resource-to-copy-source transition, and the temporary barrier slice and cloned texture-resource reference outlast `ResourceBarrier`.
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
        // SAFETY: `CopyTextureRegion` reads initialized source and destination locations; their local storage and cloned texture and readback resource references outlast the call, and `footprint` came from `GetCopyableFootprints`.
        unsafe { list.CopyTextureRegion(&raw const destination, 0, 0, 0, &raw const source, None) };
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
        // SAFETY: `barrier.Type` selects its initialized `Transition` arm for the copy-source-to-shader-resource transition, the temporary slice lasts through `ResourceBarrier`, and the recording list remains stored through `Close`.
        unsafe {
            list.ResourceBarrier(&[barrier]);
            list.Close()
        }
        .map_err(|error| map_allocation_windows(&error))?;
        let command: ID3D12CommandList = list
            .cast()
            .map_err(|error| map_allocation_windows(&error))?;
        // SAFETY: `ExecuteCommandLists` receives a command cast from the just-closed list, and the temporary command-array storage plus the allocator and list remain in scope throughout the call.
        unsafe { self.queue.ExecuteCommandLists(&[Some(command)]) };
        let value = self.next_fence;
        self.next_fence = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        // SAFETY: `Signal` receives this queue's fence and the monotonic value reserved from `next_fence`; `self` retains both COM references throughout the call.
        unsafe { self.queue.Signal(&self.fence, value) }
            .map_err(|error| map_allocation_windows(&error))?;
        // SAFETY: `GetCompletedValue` only reads the counter of the fence just signaled by this queue, and `self` retains that fence COM reference throughout the call.
        if unsafe { self.fence.GetCompletedValue() } < value {
            // SAFETY: `SetEventOnCompletion` receives the submitted fence value and `self.fence_event`, the context's waitable event handle, which is not closed while `self` is borrowed.
            unsafe { self.fence.SetEventOnCompletion(value, self.fence_event) }
                .map_err(|error| map_allocation_windows(&error))?;
            // SAFETY: `WaitForSingleObject` receives `self.fence_event`, the event just registered by `SetEventOnCompletion`, and `self` keeps that handle open for the entire wait.
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
}

impl NativeContext {
    /// Returns the completed value of the independent texture transfer timeline.
    ///
    /// # Errors
    ///
    /// Returns `NativeFailure` when the worker failed and `DeviceLost` for a removed device.
    pub fn completed_texture_transfer_value(&self) -> Result<u64, AllocationError> {
        if self
            .texture_worker
            .as_ref()
            .is_some_and(ez_gfx_hal::TransferWorker::failed)
        {
            return Err(AllocationError::NativeFailure);
        }
        // SAFETY: the context retains the texture fence and device for this query.
        let value = unsafe { self.texture_fence.GetCompletedValue() };
        if value == u64::MAX {
            return Err(AllocationError::DeviceLost);
        }
        Ok(value)
    }
}
