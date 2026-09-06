use super::{
    Allocation, AllocationCreateDesc, AllocationError, AllocationRequest, CompletionToken,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    D3D12_FILTER_ANISOTROPIC, D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
    D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_FILTER_MIN_MAG_MIP_POINT,
    D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_TRANSITION_BARRIER,
    D3D12_SAMPLER_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
    D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_TEX2D_SRV, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
    D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_UNKNOWN, DXGI_SAMPLE_DESC, DeferredResource, ID3D12CommandAllocator,
    ID3D12CommandList, ID3D12DescriptorHeap, ID3D12GraphicsCommandList, ID3D12Resource, INFINITE,
    ImageMip, Interface, MemoryAllocator, MemoryClass, MemoryLocation, NativeAllocation,
    NativeContext, NativeTexture, QueueKind, SamplerAddressMode, SamplerFilter,
    TEXTURE_DESCRIPTOR_CAPACITY, TextureFormat, TextureRegion, TextureSamplerDesc,
    WaitForSingleObject, map_allocation_windows, map_allocator, validate_texture_mips,
    validate_texture_region,
};

fn texture_format_dxgi(
    format: TextureFormat,
) -> Option<windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT> {
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM,
        DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB,
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    };

    match format {
        TextureFormat::Rgba8Unorm => Some(DXGI_FORMAT_R8G8B8A8_UNORM),
        TextureFormat::Rgba8Srgb => Some(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB),
        TextureFormat::Bc1Unorm => Some(DXGI_FORMAT_BC1_UNORM),
        TextureFormat::Bc1Srgb => Some(DXGI_FORMAT_BC1_UNORM_SRGB),
        TextureFormat::Bc3Unorm => Some(DXGI_FORMAT_BC3_UNORM),
        TextureFormat::Bc3Srgb => Some(DXGI_FORMAT_BC3_UNORM_SRGB),
        TextureFormat::Bc7Unorm => Some(DXGI_FORMAT_BC7_UNORM),
        TextureFormat::Bc7Srgb => Some(DXGI_FORMAT_BC7_UNORM_SRGB),
        // Desktop DXGI has no ASTC resource format.
        TextureFormat::Astc4x4Unorm | TextureFormat::Astc4x4Srgb => None,
    }
}

/// Maps D3D12 format-support bits onto render-target roles for one format.
///
/// Depth formats never report color or storage roles; sampling follows the
/// texture bit on every format. Multisample counts stay single-sample.
fn support_for_target_format(
    format: ez_gfx_runtime::target::Format,
    support: windows::Win32::Graphics::Direct3D12::D3D12_FORMAT_SUPPORT1,
) -> Option<ez_gfx_runtime::target::FormatSupport> {
    use ez_gfx_core::capability::CompressionSupport;
    use ez_gfx_runtime::target::Format;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL, D3D12_FORMAT_SUPPORT1_RENDER_TARGET,
        D3D12_FORMAT_SUPPORT1_TEXTURE2D, D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW,
    };
    let flags = support.0;
    // A depth candidate without attachment support is omitted so resolution
    // fails with a diagnostic instead of selecting an unusable format.
    if format == Format::Depth32Float && flags & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL.0 == 0 {
        return None;
    }
    let (color, sampled, storage) = match format {
        Format::Depth32Float => (false, flags & D3D12_FORMAT_SUPPORT1_TEXTURE2D.0 != 0, false),
        _ => (
            flags & D3D12_FORMAT_SUPPORT1_RENDER_TARGET.0 != 0,
            flags & D3D12_FORMAT_SUPPORT1_TEXTURE2D.0 != 0,
            flags & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0,
        ),
    };
    // Single-sample support always validates; only the role bits vary per device.
    Some(
        ez_gfx_runtime::target::FormatSupport::new(
            format,
            color,
            sampled,
            storage,
            1,
            CompressionSupport::NONE,
        )
        .expect("single-sample support is always valid"),
    )
}
fn validate_texture_request(
    format: TextureFormat,
    mips: &[ImageMip<'_>],
    binding: u32,
) -> Result<(), AllocationError> {
    validate_texture_mips(format, mips).map_err(|_| AllocationError::ZeroSize)?;
    if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
        return Err(AllocationError::ZeroSize);
    }

    // Only the base must fill BC blocks; smaller mips retain their logical edge dimensions.
    let [block_width, block_height, _] = format.block();
    if !mips[0].width.is_multiple_of(block_width) || !mips[0].height.is_multiple_of(block_height) {
        return Err(AllocationError::Unsupported);
    }

    Ok(())
}

fn write_texture_view(
    context: &NativeContext,
    resource: &ID3D12Resource,
    format: TextureFormat,
    binding: u32,
    mip_count: u32,
    resident_mips: u32,
) {
    // The resident range is always the contiguous coarse suffix of the native mip chain.
    let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: texture_format_dxgi(format).expect("validated DX12 texture format"),
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: mip_count - resident_mips,
                MipLevels: resident_mips,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the descriptor heap remains live and `binding` was range-checked.
    let mut handle = unsafe { context.descriptors.GetCPUDescriptorHandleForHeapStart() };
    handle.ptr += binding as usize * context.descriptor_stride as usize;
    // SAFETY: the explicit SRV references a live texture and a nonempty in-range mip suffix.
    unsafe {
        context
            .device
            .CreateShaderResourceView(resource, Some(&raw const view), handle);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "construction receives the validated native texture contract as one explicit handoff"
)]
fn publish_texture(
    context: &NativeContext,
    resource: ID3D12Resource,
    allocation: Allocation,
    format: TextureFormat,
    width: u32,
    height: u32,
    binding: u32,
    sampler_desc: TextureSamplerDesc,
    cancellation: std::sync::Arc<std::sync::atomic::AtomicBool>,
    completions: Vec<CompletionToken>,
) -> (NativeTexture, Vec<CompletionToken>) {
    let mip_count = u32::try_from(completions.len()).expect("validated mip count fits u32");
    write_texture_view(context, &resource, format, binding, mip_count, 1);
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
    // SAFETY: the sampler heap remains live and `binding` was range-checked.
    let mut sampler_handle = unsafe { context.samplers.GetCPUDescriptorHandleForHeapStart() };
    sampler_handle.ptr += binding as usize * context.sampler_stride as usize;
    // SAFETY: the initialized sampler is written into the range-checked stable binding.
    unsafe {
        context
            .device
            .CreateSampler(&raw const sampler, sampler_handle);
    };
    let mip_completions = completions.iter().rev().map(|token| token.value).collect();
    (
        NativeTexture {
            resource,
            allocation,
            format,
            width,
            height,
            mip_count,
            resident_mips: 1,
            mip_completions,
            cancellation,
            binding,
            rtv: None,
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
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS,
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
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
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
        row_counts: &[u32],
        row_sizes: &[u64],
        upload: &mut NativeAllocation,
        upload_size: u64,
    ) -> Result<(), AllocationError> {
        let target = self.mapped_slice_mut(upload)?;
        for (((mip, footprint), row_count), row_size) in
            mips.iter().zip(footprints).zip(row_counts).zip(row_sizes)
        {
            let row_bytes =
                usize::try_from(*row_size).map_err(|_| AllocationError::NativeFailure)?;
            let rows = usize::try_from(*row_count).map_err(|_| AllocationError::NativeFailure)?;
            if row_bytes
                .checked_mul(rows)
                .ok_or(AllocationError::NativeFailure)?
                != mip.bytes.len()
            {
                return Err(AllocationError::NativeFailure);
            }
            for row in 0..rows {
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

    /// Creates a sampled mip chain and submits each level under a distinct fence value.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid mip chain, format, or binding, failed texture/upload setup,
    /// native resource failure, arithmetic overflow, or failed synchronization.
    pub fn create_texture(
        &mut self,
        format: TextureFormat,
        mips: &[ImageMip<'_>],
        binding: u32,
        sampler_desc: TextureSamplerDesc,
    ) -> Result<(NativeTexture, Vec<CompletionToken>), AllocationError> {
        validate_texture_request(format, mips, binding)?;
        let dxgi = texture_format_dxgi(format).ok_or(AllocationError::NativeFailure)?;
        let (resource, allocation, desc) =
            self.create_texture_resource(mips, dxgi, D3D12_RESOURCE_FLAG_NONE)?;
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
        if let Err(error) = self.populate_texture_upload(
            mips,
            &footprints,
            &row_counts,
            &row_sizes,
            &mut upload,
            upload_size,
        ) {
            self.destroy_unpublished_texture(resource, allocation, Some(upload));
            return Err(error);
        }
        let cancellation = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let first = self.next_texture_fence;
        let next = first
            .checked_add(mips.len() as u64)
            .ok_or(AllocationError::NativeFailure)?;
        let completions = (first..next)
            .map(|value| CompletionToken::new(QueueKind::TextureTransfer, value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| AllocationError::NativeFailure)?;
        let completion = *completions.last().ok_or(AllocationError::NativeFailure)?;
        let worker = self
            .texture_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?;
        let mut jobs = Vec::with_capacity(mips.len());
        for ((subresource, footprint), token) in
            footprints.iter().enumerate().rev().zip(completions.iter())
        {
            let subresource =
                u32::try_from(subresource).map_err(|_| AllocationError::NativeFailure)?;
            let rows = u64::from(row_counts[subresource as usize]);
            let bytes = u64::from(footprint.Footprint.RowPitch)
                .checked_mul(rows)
                .ok_or(AllocationError::NativeFailure)?;
            jobs.push(super::transfer::Dx12TransferJob {
                value: token.value,
                bytes,
                cancelled: Some(cancellation.clone()),
                copy: super::transfer::Dx12TransferCopy::Texture {
                    source: upload.resource.clone(),
                    destination: resource.clone(),
                    footprint: *footprint,
                    subresource,
                    destination_x: 0,
                    destination_y: 0,
                    transition_from_shader: false,
                    stream_stage: u32::try_from(token.value - first)
                        .map_err(|_| AllocationError::NativeFailure)?,
                },
            });
        }
        // The worker accepts the entire chain or nothing, so rejection has no in-flight resources.
        if let Err(error) = worker.submit_batch(jobs) {
            self.destroy_unpublished_texture(resource, allocation, Some(upload));
            return Err(match error {
                ez_gfx_hal::TransferWorkerError::Full => AllocationError::OutOfMemory,
                ez_gfx_hal::TransferWorkerError::Failed => AllocationError::NativeFailure,
            });
        }
        // Commit only admitted values; idle must never wait for rejected uploads.
        self.next_texture_fence = next;
        let capacity = upload.allocation.size();
        self.texture_staging.put(capacity, upload, Some(completion));
        Ok(publish_texture(
            self,
            resource,
            allocation,
            format,
            mips[0].width,
            mips[0].height,
            binding,
            sampler_desc,
            cancellation,
            completions,
        ))
    }

    /// Creates an uninitialized single-mip color resource for managed render-target use.
    ///
    /// The allocated DXGI format derives from the runtime format directly. The stored
    /// texture format is the closest block-compatible value for record shape only,
    /// with inert transfer fields: route the record only through render-target entry
    /// points, never through upload, publish, or region-update paths. The safe layer
    /// owns the true format.
    ///
    /// # Errors
    ///
    /// Returns an error for zero dimensions, excessive aggregate bytes, an
    /// unsupported (non-color) format, or native allocation failure.
    pub fn create_render_target(
        &mut self,
        format: ez_gfx_runtime::target::Format,
        width: u32,
        height: u32,
        binding: u32,
    ) -> Result<NativeTexture, AllocationError> {
        use ez_gfx_runtime::target::Format;
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT,
        };
        let (dxgi, hal_format, bytes_per_texel): (_, TextureFormat, u64) = match format {
            Format::Rgba8Unorm => (DXGI_FORMAT_R8G8B8A8_UNORM, TextureFormat::Rgba8Unorm, 4),
            Format::Bgra8Srgb => (DXGI_FORMAT_B8G8R8A8_UNORM, TextureFormat::Rgba8Srgb, 4),
            Format::Rgba16Float => (DXGI_FORMAT_R16G16B16A16_FLOAT, TextureFormat::Rgba8Unorm, 8),
            _ => return Err(AllocationError::Unsupported),
        };
        if width == 0 || height == 0 {
            return Err(AllocationError::ZeroSize);
        }
        if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
            return Err(AllocationError::ZeroSize);
        }
        // A render target holds exactly one mip; bound it by the texture budget.
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(bytes_per_texel))
            .ok_or(AllocationError::NativeFailure)?;
        if bytes > u64::try_from(ez_gfx_runtime::texture::MAX_TEXTURE_BYTES).unwrap_or(u64::MAX) {
            return Err(AllocationError::OutOfMemory);
        }
        // A single mip bypasses block-alignment validation; dimensions stay logical.
        let mips = [ImageMip {
            width,
            height,
            bytes: &[],
        }];
        let (resource, allocation, _) =
            self.create_texture_resource(&mips, dxgi, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)?;
        // A private single-entry RTV heap avoids sharing the surface heaps;
        // the heap drops with the texture record after GPU retirement.
        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            self.device.CreateDescriptorHeap(
                &windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_DESC {
                    Type:
                        windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags:
                        windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                },
            )
        }
        .map_err(|error| map_allocation_windows(&error))?;
        // SAFETY: the heap retains one CPU descriptor slot for the call.
        let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let rtv_desc =
            windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC {
                Format: dxgi,
                ViewDimension:
                    windows::Win32::Graphics::Direct3D12::D3D12_RTV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
        // SAFETY: `rtv` addresses the heap's single slot and `desc` selects a
        // matching view of the retained resource through the call.
        unsafe {
            self.device
                .CreateRenderTargetView(&resource, Some(&raw const rtv_desc), rtv);
        }
        Ok(NativeTexture {
            resource,
            allocation,
            format: hal_format,
            width,
            height,
            mip_count: 1,
            resident_mips: 1,
            mip_completions: vec![0],
            cancellation: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            binding,
            rtv: Some((rtv_heap, rtv)),
        })
    }

    /// Copies one validated tightly packed region through reusable upload staging.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid region, staging exhaustion, timeline overflow, or failed
    /// worker admission. Borrowed bytes are copied before this method returns.
    pub fn update_texture_region(
        &mut self,
        texture: &mut NativeTexture,
        region: &TextureRegion<'_>,
    ) -> Result<CompletionToken, AllocationError> {
        validate_texture_region(
            texture.format,
            texture.width,
            texture.height,
            texture.mip_count,
            *region,
        )
        .map_err(|_| AllocationError::ZeroSize)?;

        // Footprint queries treat this as a base resource, so clipped mip edges still need
        // complete physical blocks. The real destination mip keeps its logical dimensions.
        let [block_width, block_height, _] = texture.format.block();
        let width = region
            .width
            .div_ceil(block_width)
            .checked_mul(block_width)
            .ok_or(AllocationError::ZeroSize)?;
        let height = region
            .height
            .div_ceil(block_height)
            .checked_mul(block_height)
            .ok_or(AllocationError::ZeroSize)?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: texture_format_dxgi(texture.format).ok_or(AllocationError::NativeFailure)?,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut footprint =
            windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0_u32;
        let mut row_size = 0_u64;
        let mut upload_size = 0_u64;
        // SAFETY: all output pointers reference initialized writable locals for one subresource.
        unsafe {
            self.device.GetCopyableFootprints(
                &raw const desc,
                0,
                1,
                0,
                Some(&raw mut footprint),
                Some(&raw mut rows),
                Some(&raw mut row_size),
                Some(&raw mut upload_size),
            );
        }
        let row_bytes = usize::try_from(row_size).map_err(|_| AllocationError::NativeFailure)?;
        let row_count = usize::try_from(rows).map_err(|_| AllocationError::NativeFailure)?;
        if row_bytes
            .checked_mul(row_count)
            .ok_or(AllocationError::NativeFailure)?
            != region.bytes.len()
        {
            return Err(AllocationError::NativeFailure);
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
                self.allocate(request)?
            };
        let populate = (|| -> Result<(), AllocationError> {
            let target = self.mapped_slice_mut(&mut upload)?;
            for row in 0..row_count {
                let source_start = row * row_bytes;
                let destination_start = row * footprint.Footprint.RowPitch as usize;
                target[destination_start..destination_start + row_bytes]
                    .copy_from_slice(&region.bytes[source_start..source_start + row_bytes]);
            }
            self.flush(&mut upload, 0, upload_size)
        })();
        if let Err(error) = populate {
            self.free(upload)?;
            return Err(error);
        }

        let value = self.next_texture_fence;
        let next = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        let completion = CompletionToken::new(QueueKind::TextureTransfer, value)
            .map_err(|_| AllocationError::NativeFailure)?;
        let submitted = self
            .texture_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .submit(super::transfer::Dx12TransferJob {
                value,
                bytes: upload_size,
                cancelled: Some(texture.cancellation.clone()),
                copy: super::transfer::Dx12TransferCopy::Texture {
                    source: upload.resource.clone(),
                    destination: texture.resource.clone(),
                    footprint,
                    subresource: region.mip_level,
                    destination_x: region.x,
                    destination_y: region.y,
                    transition_from_shader: true,
                    stream_stage: 0,
                },
            });
        if let Err(error) = submitted {
            self.free(upload)?;
            return Err(match error {
                ez_gfx_hal::TransferWorkerError::Full => AllocationError::OutOfMemory,
                ez_gfx_hal::TransferWorkerError::Failed => AllocationError::NativeFailure,
            });
        }
        self.next_texture_fence = next;
        let capacity = upload.allocation.size();
        self.texture_staging.put(capacity, upload, Some(completion));
        *texture
            .mip_completions
            .get_mut(usize::try_from(region.mip_level).map_err(|_| AllocationError::NativeFailure)?)
            .ok_or(AllocationError::NativeFailure)? = value;
        Ok(completion)
    }

    /// Rewrites the stable binding to expose exactly the requested contiguous coarse mip range.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/out-of-range range, an unfinished mip transfer, worker
    /// failure, or device loss.
    pub fn publish_texture_mips(
        &mut self,
        texture: &mut NativeTexture,
        resident_mips: u32,
    ) -> Result<(), AllocationError> {
        if resident_mips == 0 || resident_mips > texture.mip_count {
            return Err(AllocationError::ZeroSize);
        }
        let first = usize::try_from(texture.mip_count - resident_mips)
            .map_err(|_| AllocationError::NativeFailure)?;
        let required = texture.mip_completions[first..]
            .iter()
            .copied()
            .max()
            .ok_or(AllocationError::NativeFailure)?;
        if self.completed_texture_transfer_value()? < required {
            return Err(AllocationError::NativeFailure);
        }
        // SAFETY: the context retains the graphics fence throughout this nonblocking counter read.
        let graphics_completed = unsafe { self.fence.GetCompletedValue() };
        if graphics_completed == u64::MAX {
            return Err(AllocationError::DeviceLost);
        }
        // Shader-visible descriptors cannot be overwritten while an earlier frame may still
        // consume the slot. Publishing remains nonblocking and can be retried after polling.
        if graphics_completed < self.next_fence.saturating_sub(1) {
            return Err(AllocationError::NativeFailure);
        }
        if texture.resident_mips == resident_mips {
            return Ok(());
        }
        write_texture_view(
            self,
            &texture.resource,
            texture.format,
            texture.binding,
            texture.mip_count,
            resident_mips,
        );
        texture.resident_mips = resident_mips;
        Ok(())
    }

    /// Queries device format support for render-target formats.
    ///
    /// Reports render, sampled, and storage roles for RGBA8, BGRA sRGB, and
    /// RGBA16F plus depth attachment support for D32 float. Multisample counts
    /// stay single-sample; resolve targets select separately.
    ///
    /// # Errors
    ///
    /// Returns an error when the device rejects the format-support query.
    pub fn probe_target_formats(
        &self,
    ) -> Result<ez_gfx_runtime::target::FormatCapabilities, AllocationError> {
        use ez_gfx_runtime::target::Format;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_DATA_FORMAT_SUPPORT, D3D12_FEATURE_FORMAT_SUPPORT,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        };
        let query = |format, dxgi| {
            let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                Format: dxgi,
                ..Default::default()
            };
            // SAFETY: `data` is a live aligned query record sized exactly, and the
            // device outlives the call.
            unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    (&raw mut data).cast(),
                    u32::try_from(core::mem::size_of_val(&data)).unwrap_or(u32::MAX),
                )
            }
            .map_err(|error| map_allocation_windows(&error))?;
            Ok::<_, AllocationError>(support_for_target_format(format, data.Support1))
        };
        let supports = [
            query(Format::Rgba8Unorm, DXGI_FORMAT_R8G8B8A8_UNORM)?,
            query(Format::Bgra8Srgb, DXGI_FORMAT_B8G8R8A8_UNORM)?,
            query(Format::Rgba16Float, DXGI_FORMAT_R16G16B16A16_FLOAT)?,
            query(Format::Depth32Float, DXGI_FORMAT_D32_FLOAT)?,
        ]
        .into_iter()
        .flatten()
        .collect();
        ez_gfx_runtime::target::FormatCapabilities::new(supports)
            .map_err(|_| AllocationError::NativeFailure)
    }
    /// Reports whether bindless descriptor rewrites can avoid every submitted frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the graphics completion fence reports device loss.
    pub fn texture_descriptor_update_ready(&self) -> Result<bool, AllocationError> {
        // SAFETY: the context retains the graphics fence throughout this counter read.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            return Err(AllocationError::DeviceLost);
        }
        Ok(self
            .frame_slots
            .iter()
            .all(|frame| frame.fence_value == 0 || frame.fence_value <= completed))
    }

    /// Prevents transfer-owner jobs not yet recorded by the native queue from copying this texture.
    pub fn cancel_texture_transfers(texture: &NativeTexture) {
        texture
            .cancellation
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Reports whether transfer and graphics-frame users have released a logically dead texture.
    ///
    /// # Errors
    ///
    /// Returns an error when either native fence cannot be queried.
    pub fn texture_retirement_ready(
        &self,
        completion: CompletionToken,
    ) -> Result<bool, AllocationError> {
        let transfer_done = self.completed_texture_transfer_value()? >= completion.value;
        // SAFETY: the context retains the graphics fence throughout this counter read.
        let graphics_completed = unsafe { self.fence.GetCompletedValue() };
        if graphics_completed == u64::MAX {
            return Err(AllocationError::DeviceLost);
        }
        let frames_done = self
            .frame_slots
            .iter()
            .all(|frame| frame.fence_value == 0 || frame.fence_value <= graphics_completed);
        Ok(transfer_done && frames_done)
    }

    /// Defers texture destruction behind both accepted texture transfers and graphics work.
    ///
    /// # Errors
    ///
    /// Returns an error if worker submission draining or graphics-fence signaling fails.
    pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
        self.texture_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .flush()
            .map_err(|_| AllocationError::NativeFailure)?;
        let retirement = self.next_fence;
        self.next_fence = retirement
            .checked_add(1)
            .ok_or(AllocationError::NativeFailure)?;
        // SAFETY: worker flush established all prior texture queue handoffs on this graphics
        // queue, so its following signal retires both those transfers and earlier frame uses.
        unsafe { self.queue.Signal(&self.fence, retirement) }
            .map_err(|error| map_allocation_windows(&error))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_admission_distinguishes_bc_base_alignment_from_mip_edges() {
        for format in [
            TextureFormat::Bc1Unorm,
            TextureFormat::Bc1Srgb,
            TextureFormat::Bc3Unorm,
            TextureFormat::Bc3Srgb,
            TextureFormat::Bc7Unorm,
            TextureFormat::Bc7Srgb,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8Srgb,
        ] {
            // Either base axis can violate BC alignment, including sub-block bases.
            for (width, height) in [(7, 3), (7, 4), (4, 3), (1, 1), (8, 4)] {
                let bytes =
                    vec![0; usize::try_from(format.level_bytes(width, height).unwrap()).unwrap()];
                let mips = [ImageMip {
                    width,
                    height,
                    bytes: &bytes,
                }];
                let expected = if format.is_compressed() && (width % 4 != 0 || height % 4 != 0) {
                    Err(AllocationError::Unsupported)
                } else {
                    Ok(())
                };
                assert_eq!(validate_texture_request(format, &mips, 0), expected);
            }

            // The aligned base permits odd 14x6 and 7x3 mips and a sub-block tail.
            let dimensions = [(28, 12), (14, 6), (7, 3), (3, 1), (1, 1)];
            let payloads = dimensions.map(|(width, height)| {
                vec![0; usize::try_from(format.level_bytes(width, height).unwrap()).unwrap()]
            });
            let mips: Vec<_> = dimensions
                .iter()
                .zip(&payloads)
                .map(|(&(width, height), bytes)| ImageMip {
                    width,
                    height,
                    bytes,
                })
                .collect();
            assert_eq!(validate_texture_request(format, &mips, 0), Ok(()));
        }
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;
    use ez_gfx_runtime::target::Format;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL, D3D12_FORMAT_SUPPORT1_RENDER_TARGET,
        D3D12_FORMAT_SUPPORT1_TEXTURE2D, D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW,
    };

    #[test]
    fn support_bits_select_render_sampled_and_storage_roles() {
        // Full support admits every render-target role for RGBA8.
        let full = D3D12_FORMAT_SUPPORT1_RENDER_TARGET
            | D3D12_FORMAT_SUPPORT1_TEXTURE2D
            | D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
        let support = support_for_target_format(Format::Rgba8Unorm, full).unwrap();
        assert!(support.color && support.sampled && support.storage);
    }

    #[test]
    fn depth_support_never_reports_color() {
        // Depth aspects carry no color role regardless of support bits.
        let depth = support_for_target_format(
            Format::Depth32Float,
            D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL | D3D12_FORMAT_SUPPORT1_TEXTURE2D,
        )
        .unwrap();
        assert!(!depth.color);
        assert!(depth.sampled);
    }

    #[test]
    fn depth_without_attachment_support_is_omitted() {
        // Omitting the record makes resolution fail with UnsupportedFormat
        // instead of selecting an unusable depth format.
        assert!(
            support_for_target_format(Format::Depth32Float, D3D12_FORMAT_SUPPORT1_TEXTURE2D)
                .is_none()
        );
    }
}
