use super::{
    Allocation, AllocationCreateDesc, AllocationError, AllocationRequest, CompletionToken,
    DeferredResource, ImageMip, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLHeap, MTLOrigin, MTLPixelFormat,
    MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerState, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage, MemoryAllocator, MemoryClass, NSRange, NativeContext, NativeTexture,
    ProtocolObject, QueueKind, Retained, SamplerAddressMode, SamplerFilter,
    TEXTURE_DESCRIPTOR_CAPACITY, TextureFormat, TextureRegion, TextureSamplerDesc, ThreadBound,
    map_allocator, validate_texture_mips, validate_texture_region,
};

fn texture_format_metal(format: TextureFormat) -> MTLPixelFormat {
    match format {
        TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        TextureFormat::Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        TextureFormat::Bc1Unorm => MTLPixelFormat::BC1_RGBA,
        TextureFormat::Bc1Srgb => MTLPixelFormat::BC1_RGBA_sRGB,
        TextureFormat::Bc3Unorm => MTLPixelFormat::BC3_RGBA,
        TextureFormat::Bc3Srgb => MTLPixelFormat::BC3_RGBA_sRGB,
        TextureFormat::Bc7Unorm => MTLPixelFormat::BC7_RGBAUnorm,
        TextureFormat::Bc7Srgb => MTLPixelFormat::BC7_RGBAUnorm_sRGB,
        TextureFormat::Astc4x4Unorm => MTLPixelFormat::ASTC_4x4_LDR,
        TextureFormat::Astc4x4Srgb => MTLPixelFormat::ASTC_4x4_sRGB,
    }
}

fn texture_view(
    storage: &ProtocolObject<dyn MTLTexture>,
    format: TextureFormat,
    mip_count: u32,
    resident_mips: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, AllocationError> {
    // A Metal view cannot expose zero levels; callers validate the nonempty contiguous coarse tail.
    if resident_mips == 0 || resident_mips > mip_count {
        return Err(AllocationError::ZeroSize);
    }
    let first_level = mip_count - resident_mips;
    // SAFETY: the nonempty level range is bounded by `mip_count`, and 2D textures have one slice.
    unsafe {
        storage.newTextureViewWithPixelFormat_textureType_levels_slices(
            texture_format_metal(format),
            storage.textureType(),
            NSRange::new(first_level as usize, resident_mips as usize),
            NSRange::new(0, 1),
        )
    }
    .ok_or(AllocationError::NativeFailure)
}

/// Releases the published resolve-texture parts when multisampled setup fails.
///
/// The resolve storage, view, and sampler never published, so they drop
/// immediately; the allocation frees through the context allocator.
fn release_resolve_parts(
    context: &mut NativeContext,
    view: Retained<ProtocolObject<dyn MTLTexture>>,
    storage: Retained<ProtocolObject<dyn MTLTexture>>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    allocation: Allocation,
) {
    drop(view);
    drop(storage);
    drop(sampler);
    if let Some(allocator) = context.allocator.as_mut() {
        let _ = allocator.free(&allocation);
    }
}

impl NativeContext {
    /// Creates full sampled storage and admits copy descriptions coarse to fine.
    /// Encoding runs on the transfer owner; exclusive context access serializes admission and
    /// allocator changes. Retained storage and flushed staging outlive every admitted copy.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid mips or format, allocation failure, or rejected Metal commands.
    /// Returned completion tokens follow coarse-to-fine submission order.
    #[expect(
        clippy::too_many_lines,
        reason = "keep native allocation rollback beside all fallible upload-admission steps"
    )]
    pub fn create_texture(
        &mut self,
        format: TextureFormat,
        mips: &[ImageMip<'_>],
        binding: u32,
        sampler_desc: TextureSamplerDesc,
    ) -> Result<(NativeTexture, Vec<CompletionToken>), AllocationError> {
        validate_texture_mips(format, mips).map_err(|_| AllocationError::ZeroSize)?;
        if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
            return Err(AllocationError::ZeroSize);
        }
        let width = mips[0].width;
        let height = mips[0].height;
        let mip_count = u32::try_from(mips.len()).map_err(|_| AllocationError::NativeFailure)?;
        let texture_width = usize::try_from(width).map_err(|_| AllocationError::NativeFailure)?;
        let texture_height = usize::try_from(height).map_err(|_| AllocationError::NativeFailure)?;
        let total = mips
            .iter()
            .try_fold(0_u64, |sum, mip| sum.checked_add(mip.bytes.len() as u64))
            .ok_or(AllocationError::NativeFailure)?;
        let bucket = ez_gfx_hal::staging_bucket_size(total, ez_gfx_hal::DEFAULT_STAGING_POLICY)
            .map_err(|_| AllocationError::OutOfMemory)?;
        let completed = self.reclaim_texture_transfers()?;
        let request = AllocationRequest::new(bucket, 4, MemoryClass::Upload, true, None)?;
        // SAFETY: dimensions and mip count were validated against every bounded payload.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                texture_format_metal(format),
                texture_width,
                texture_height,
                mips.len() > 1,
            )
        };
        // SAFETY: `desc` is live and the validated mip count is consumed during this send.
        unsafe { desc.setMipmapLevelCount(mips.len()) };
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
        desc.setStorageMode(MTLStorageMode::Private);
        let allocation_desc = AllocationCreateDesc::texture(&self.device, "ez-gfx-texture", &desc);
        let allocation = self
            .allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .allocate(&allocation_desc)
            .map_err(map_allocator)?;
        let Ok(allocation_offset) = usize::try_from(allocation.offset()) else {
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(&allocation)
                .map_err(map_allocator)?;
            return Err(AllocationError::NativeFailure);
        };
        // SAFETY: the allocation belongs to this heap and the checked offset describes it.
        let Some(storage) = (unsafe {
            allocation
                .heap()
                .newTextureWithDescriptor_offset(&desc, allocation_offset)
        }) else {
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(&allocation)
                .map_err(map_allocator)?;
            return Err(AllocationError::OutOfMemory);
        };
        let mut upload = if let Some((_, upload)) = self.texture_staging.take(total, ez_gfx_hal::QueueKind::TextureTransfer, completed) {
            upload
        } else {
            match self.allocate(request) {
                Ok(upload) => upload,
                Err(error) => {
                    drop(storage);
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(&allocation)
                        .map_err(map_allocator)?;
                    return Err(error);
                }
            }
        };
        let submitted = (|| {
            let target = self.mapped_slice_mut(&mut upload)?;
            let mut offset = 0_usize;
            // Coarse-first packing matches command submission and completion-token order.
            for mip in mips.iter().rev() {
                target[offset..offset + mip.bytes.len()].copy_from_slice(mip.bytes);
                offset += mip.bytes.len();
            }
            self.flush(&mut upload, 0, total)?;

            let sampler_descriptor = MTLSamplerDescriptor::new();
            sampler_descriptor.setMinFilter(match sampler_desc.min_filter {
                SamplerFilter::Nearest => MTLSamplerMinMagFilter::Nearest,
                SamplerFilter::Linear => MTLSamplerMinMagFilter::Linear,
            });
            sampler_descriptor.setMagFilter(match sampler_desc.mag_filter {
                SamplerFilter::Nearest => MTLSamplerMinMagFilter::Nearest,
                SamplerFilter::Linear => MTLSamplerMinMagFilter::Linear,
            });
            // `mipFilter` defaults to `notMipmapped`: without this mapping every
            // minified fragment would sample view level zero regardless of the chain.
            sampler_descriptor.setMipFilter(match sampler_desc.min_filter {
                SamplerFilter::Nearest => MTLSamplerMipFilter::Nearest,
                SamplerFilter::Linear => MTLSamplerMipFilter::Linear,
            });
            let address = |mode| match mode {
                SamplerAddressMode::Clamp => MTLSamplerAddressMode::ClampToEdge,
                SamplerAddressMode::Repeat => MTLSamplerAddressMode::Repeat,
            };
            sampler_descriptor.setSAddressMode(address(sampler_desc.address_u));
            sampler_descriptor.setTAddressMode(address(sampler_desc.address_v));
            sampler_descriptor.setRAddressMode(address(sampler_desc.address_w));
            let anisotropy = (1_u16..16)
                .find(|level| sampler_desc.max_anisotropy < f32::from(*level + 1))
                .map_or(16, usize::from);
            sampler_descriptor.setMaxAnisotropy(anisotropy);
            sampler_descriptor.setSupportArgumentBuffers(true);
            let sampler = self
                .device
                .newSamplerStateWithDescriptor(&sampler_descriptor)
                .ok_or(AllocationError::NativeFailure)?;
            let view = texture_view(&storage, format, mip_count, 1)?;

            let cancellation = std::sync::Arc::new(super::transfer::TransferCancellation::new());
            let first = self.next_texture_value;
            let next = first
                .checked_add(mips.len() as u64)
                .ok_or(AllocationError::NativeFailure)?;
            let completions = (first..next)
                .map(|value| CompletionToken::new(QueueKind::TextureTransfer, value))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| AllocationError::NativeFailure)?;
            let completion = *completions.last().ok_or(AllocationError::NativeFailure)?;
            let mut jobs = Vec::with_capacity(mips.len());
            let mut pending = Vec::with_capacity(mips.len());
            let mut source_offset = 0_usize;
            let [block_width, block_height, block_bytes] = format.block();
            for (level, mip) in mips.iter().enumerate().rev() {
                let mip_width =
                    usize::try_from(mip.width).map_err(|_| AllocationError::NativeFailure)?;
                let mip_height =
                    usize::try_from(mip.height).map_err(|_| AllocationError::NativeFailure)?;
                let row_bytes = usize::try_from(mip.width.div_ceil(block_width))
                    .ok()
                    .and_then(|blocks| blocks.checked_mul(block_bytes as usize))
                    .ok_or(AllocationError::NativeFailure)?;
                let rows = usize::try_from(mip.height.div_ceil(block_height))
                    .map_err(|_| AllocationError::NativeFailure)?;
                let image_bytes = row_bytes
                    .checked_mul(rows)
                    .ok_or(AllocationError::NativeFailure)?;
                if image_bytes != mip.bytes.len() {
                    return Err(AllocationError::NativeFailure);
                }
                let submission = std::sync::Arc::new(super::transfer::TextureSubmission::default());
                let token = completions[jobs.len()];
                jobs.push(super::transfer::TextureTransferJob {
                    value: token.value,
                    bytes: image_bytes as u64,
                    stage: jobs.len() as u64,
                    graphics_wait: false,
                    copy: super::transfer::TextureCopy {
                        source: upload.buffer.clone(),
                        destination: storage.clone(),
                        source_offset,
                        row_bytes,
                        image_bytes,
                        size: MTLSize {
                            width: mip_width,
                            height: mip_height,
                            depth: 1,
                        },
                        level,
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    },
                    cancellation: cancellation.clone(),
                    submission: submission.clone(),
                });
                pending.push(super::transfer::PendingTextureTransfer {
                    value: token.value,
                    submission,
                });
                source_offset = source_offset
                    .checked_add(image_bytes)
                    .ok_or(AllocationError::NativeFailure)?;
            }
            self.texture_worker
                .as_ref()
                .ok_or(AllocationError::NativeFailure)?
                .submit_batch(jobs)
                .map_err(ez_gfx_hal::TransferWorkerError::to_allocation_error)?;
            self.drain_complete = false;
            // Rejected admission owns no completion value.
            self.next_texture_value = next;
            self.pending_texture_transfers.extend(pending);
            Ok((sampler, view, cancellation, completion, completions))
        })();
        match submitted {
            Ok((sampler, view, cancellation, completion, completions)) => {
                let capacity = upload.allocation.size();
                self.texture_staging.put(capacity, upload, Some(completion));
                let mut mip_completions = vec![0; mips.len()];
                for (token, level) in completions.iter().zip((0..mips.len()).rev()) {
                    mip_completions[level] = token.value;
                }
                Ok((
                    NativeTexture {
                        texture: ThreadBound::new(view),
                        allocation: ThreadBound::new(allocation),
                        sampler: ThreadBound::new(sampler),
                        format,
                        width,
                        height,
                        mip_count,
                        resident_mips: 1,
                        mip_completions,
                        cancellation,
                        binding,
                        msaa: None,
                    },
                    completions,
                ))
            }
            Err(error) => {
                let _ = self.free(upload);
                drop(storage);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&allocation)
                    .map_err(map_allocator)?;
                Err(error)
            }
        }
    }

    /// Creates an uninitialized single-mip color texture for managed render-target use.
    ///
    /// The sampled texture carries render-target, shader-read, and
    /// pixel-format-view usage with no initial contents; the first render pass
    /// transitions and clears it. With `samples > 1` a second multisampled
    /// texture renders the pass and resolves into the sampled texture, which
    /// stays the only sampled, readback, and descriptor texture. The returned
    /// texture reuses the texture record with inert transfer fields: route it
    /// only through render-target entry points, never through upload, publish,
    /// or region-update paths. The safe layer owns the true format.
    ///
    /// # Errors
    ///
    /// Returns an error for zero dimensions, an unsupported sample count or
    /// (non-color) format, excessive aggregate bytes, or native allocation failure.
    pub fn create_render_target(
        &mut self,
        format: ez_gfx_runtime::target::Format,
        width: u32,
        height: u32,
        binding: u32,
        samples: u8,
    ) -> Result<NativeTexture, AllocationError> {
        use ez_gfx_runtime::target::Format;
        let (pixel_format, hal_format, bytes_per_texel) = match format {
            Format::Rgba8Unorm => (MTLPixelFormat::RGBA8Unorm, TextureFormat::Rgba8Unorm, 4),
            Format::Bgra8Srgb => (MTLPixelFormat::BGRA8Unorm_sRGB, TextureFormat::Rgba8Srgb, 4),
            Format::Rgba16Float => (MTLPixelFormat::RGBA16Float, TextureFormat::Rgba8Unorm, 8),
            _ => return Err(AllocationError::Unsupported),
        };
        // 8-sample render targets stay out of scope on Metal hardware.
        if !matches!(samples, 1 | 2 | 4) {
            return Err(AllocationError::Unsupported);
        }
        if width == 0 || height == 0 {
            return Err(AllocationError::ZeroSize);
        }
        if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
            return Err(AllocationError::ZeroSize);
        }
        // A render target holds exactly one mip; bound it by the texture budget,
        // scaled by the sample count for multisampled storage.
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(bytes_per_texel))
            .and_then(|single| single.checked_mul(u64::from(samples)))
            .ok_or(AllocationError::NativeFailure)?;
        if bytes > u64::try_from(ez_gfx_runtime::texture::MAX_TEXTURE_BYTES).unwrap_or(u64::MAX) {
            return Err(AllocationError::OutOfMemory);
        }
        let texture_width = usize::try_from(width).map_err(|_| AllocationError::NativeFailure)?;
        let texture_height = usize::try_from(height).map_err(|_| AllocationError::NativeFailure)?;
        // SAFETY: validated dimensions and a single mip level are consumed during this send.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format,
                texture_width,
                texture_height,
                false,
            )
        };
        // SAFETY: `desc` is live and the single mip count is consumed during this send.
        unsafe { desc.setMipmapLevelCount(1) };
        desc.setUsage(
            MTLTextureUsage::RenderTarget
                | MTLTextureUsage::ShaderRead
                | MTLTextureUsage::PixelFormatView,
        );
        desc.setStorageMode(MTLStorageMode::Private);
        let allocation = self
            .allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .allocate(&AllocationCreateDesc::texture(
                &self.device,
                "ez-gfx-render-target",
                &desc,
            ))
            .map_err(map_allocator)?;
        let Ok(allocation_offset) = usize::try_from(allocation.offset()) else {
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(&allocation)
                .map_err(map_allocator)?;
            return Err(AllocationError::NativeFailure);
        };
        // SAFETY: the allocation belongs to this heap and the checked offset describes it.
        let Some(storage) = (unsafe {
            allocation
                .heap()
                .newTextureWithDescriptor_offset(&desc, allocation_offset)
        }) else {
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(&allocation)
                .map_err(map_allocator)?;
            return Err(AllocationError::OutOfMemory);
        };
        let sampler_descriptor = MTLSamplerDescriptor::new();
        sampler_descriptor.setMinFilter(MTLSamplerMinMagFilter::Nearest);
        sampler_descriptor.setMagFilter(MTLSamplerMinMagFilter::Nearest);
        sampler_descriptor.setMipFilter(MTLSamplerMipFilter::Nearest);
        sampler_descriptor.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sampler_descriptor.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sampler_descriptor.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sampler_descriptor.setMaxAnisotropy(1);
        sampler_descriptor.setSupportArgumentBuffers(true);
        let sampler = match self
            .device
            .newSamplerStateWithDescriptor(&sampler_descriptor)
        {
            Some(sampler) => sampler,
            None => {
                drop(storage);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&allocation)
                    .map_err(map_allocator)?;
                return Err(AllocationError::NativeFailure);
            }
        };
        // The view must use the true pixel format, not the record-shape stand-in.
        // SAFETY: the single-level range is bounded by the one-mip storage, with one slice.
        let view = match unsafe {
            storage.newTextureViewWithPixelFormat_textureType_levels_slices(
                pixel_format,
                storage.textureType(),
                NSRange::new(0, 1),
                NSRange::new(0, 1),
            )
        } {
            Some(view) => view,
            None => {
                drop(storage);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&allocation)
                    .map_err(map_allocator)?;
                return Err(AllocationError::NativeFailure);
            }
        };
        // Single-sample targets render directly into the sampled texture; the
        // multisampled texture below stays absent.
        let msaa = if samples == 1 {
            None
        } else {
            // SAFETY: validated dimensions and a single mip level are consumed during this send.
            let msaa_desc = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    pixel_format,
                    texture_width,
                    texture_height,
                    false,
                )
            };
            msaa_desc.setTextureType(MTLTextureType::Type2DMultisample);
            // SAFETY: 2 and 4 are valid Metal sample counts, checked above.
            unsafe { msaa_desc.setSampleCount(usize::from(samples)) };
            // SAFETY: `msaa_desc` is live and the single mip count is consumed during this send.
            unsafe { msaa_desc.setMipmapLevelCount(1) };
            msaa_desc.setUsage(MTLTextureUsage::RenderTarget);
            msaa_desc.setStorageMode(MTLStorageMode::Private);
            let msaa_allocation = match self
                .allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .allocate(&AllocationCreateDesc::texture(
                    &self.device,
                    "ez-gfx-render-target-msaa",
                    &msaa_desc,
                )) {
                Ok(allocation) => allocation,
                Err(error) => {
                    release_resolve_parts(self, view, storage, sampler, allocation);
                    return Err(map_allocator(error));
                }
            };
            let Ok(msaa_offset) = usize::try_from(msaa_allocation.offset()) else {
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&msaa_allocation)
                    .map_err(map_allocator)?;
                release_resolve_parts(self, view, storage, sampler, allocation);
                return Err(AllocationError::NativeFailure);
            };
            // SAFETY: the allocation belongs to this heap and the checked offset describes it.
            let Some(msaa_storage) = (unsafe {
                msaa_allocation
                    .heap()
                    .newTextureWithDescriptor_offset(&msaa_desc, msaa_offset)
            }) else {
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&msaa_allocation)
                    .map_err(map_allocator)?;
                release_resolve_parts(self, view, storage, sampler, allocation);
                return Err(AllocationError::OutOfMemory);
            };
            Some(super::MsaaStorage {
                texture: ThreadBound::new(msaa_storage),
                allocation: ThreadBound::new(msaa_allocation),
                samples,
            })
        };
        Ok(NativeTexture {
            texture: ThreadBound::new(view),
            allocation: ThreadBound::new(allocation),
            sampler: ThreadBound::new(sampler),
            format: hal_format,
            width,
            height,
            mip_count: 1,
            resident_mips: 1,
            mip_completions: vec![0],
            cancellation: std::sync::Arc::new(super::transfer::TransferCancellation::new()),
            binding,
            msaa,
        })
    }

    /// Replaces one validated mip subregion through the dedicated texture-transfer queue.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed block alignment or byte counts, allocation failure, or
    /// rejected Metal commands.
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
        let storage = texture
            .texture
            .parentTexture()
            .ok_or(AllocationError::NativeFailure)?;
        let bytes = region.bytes.len() as u64;
        let bucket = ez_gfx_hal::staging_bucket_size(bytes, ez_gfx_hal::DEFAULT_STAGING_POLICY)
            .map_err(|_| AllocationError::OutOfMemory)?;
        let completed = self.reclaim_texture_transfers()?;
        let request = AllocationRequest::new(bucket, 4, MemoryClass::Upload, true, None)?;
        let mut upload = if let Some((_, upload)) = self.texture_staging.take(bytes, ez_gfx_hal::QueueKind::TextureTransfer, completed) {
            upload
        } else {
            self.allocate(request)?
        };
        let submitted = (|| {
            // Borrowed update bytes are copied into owned staging before the worker can submit.
            self.mapped_slice_mut(&mut upload)?[..region.bytes.len()].copy_from_slice(region.bytes);
            self.flush(&mut upload, 0, bytes)?;

            let [block_width, block_height, block_bytes] = texture.format.block();
            let blocks_per_row = region.width.div_ceil(block_width);
            let block_rows = region.height.div_ceil(block_height);
            let row_bytes = usize::try_from(blocks_per_row)
                .ok()
                .and_then(|blocks| blocks.checked_mul(block_bytes as usize))
                .ok_or(AllocationError::NativeFailure)?;
            let image_bytes = row_bytes
                .checked_mul(
                    usize::try_from(block_rows).map_err(|_| AllocationError::NativeFailure)?,
                )
                .ok_or(AllocationError::NativeFailure)?;
            if image_bytes != region.bytes.len() {
                return Err(AllocationError::NativeFailure);
            }
            let value = self.next_texture_value;
            let release = self
                .queue
                .commandBuffer()
                .ok_or(AllocationError::NativeFailure)?;
            release.encodeSignalEvent_value(&self.texture_graphics_event, value);
            // The worker encodes the graphics-release wait before the shared update blit.
            let copy = super::transfer::TextureCopy {
                source: upload.buffer.clone(),
                destination: storage,
                source_offset: 0,
                row_bytes,
                image_bytes,
                size: MTLSize {
                    width: region.width as usize,
                    height: region.height as usize,
                    depth: 1,
                },
                level: region.mip_level as usize,
                origin: MTLOrigin {
                    x: region.x as usize,
                    y: region.y as usize,
                    z: 0,
                },
            };
            let submission = std::sync::Arc::new(super::transfer::TextureSubmission::default());
            let next = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
            let completion = CompletionToken::new(QueueKind::TextureTransfer, value)
                .map_err(|_| AllocationError::NativeFailure)?;
            self.texture_worker
                .as_ref()
                .ok_or(AllocationError::NativeFailure)?
                .submit(super::transfer::TextureTransferJob {
                    value,
                    bytes,
                    stage: u64::MAX,
                    graphics_wait: true,
                    copy,
                    cancellation: texture.cancellation.clone(),
                    submission: submission.clone(),
                })
                .map_err(ez_gfx_hal::TransferWorkerError::to_allocation_error)?;
            self.drain_complete = false;
            self.next_texture_value = next;
            // Commit the producer before any subsequent graphics frame can wait on the copy.
            // Failed worker admission never commits this otherwise unnecessary marker.
            release.commit();
            self.pending_texture_transfers
                .push(super::transfer::PendingTextureTransfer { value, submission });
            Ok(completion)
        })();
        match submitted {
            Ok(completion) => {
                let capacity = upload.allocation.size();
                self.texture_staging.put(capacity, upload, Some(completion));
                texture.mip_completions[region.mip_level as usize] = completion.value;
                Ok(completion)
            }
            Err(error) => {
                let _ = self.free(upload);
                Err(error)
            }
        }
    }

    /// Reports whether bindless argument-buffer rewrites can avoid every submitted frame.
    pub fn texture_descriptor_update_ready(&self) -> bool {
        self.frame_tracker.in_flight_mask() == 0
    }

    /// Prevents transfer-owner jobs not yet committed to Metal from copying this texture.
    pub fn cancel_texture_transfers(texture: &NativeTexture) {
        texture.cancellation.cancel();
    }

    /// Reports whether transfer and graphics-frame users have released a logically dead texture.
    ///
    /// # Errors
    ///
    /// Returns an error when native completion cannot be queried.
    pub fn texture_retirement_ready(
        &self,
        completion: CompletionToken,
    ) -> Result<bool, AllocationError> {
        Ok(self.completed_texture_transfer_value()? >= completion.value
            && self.frame_tracker.in_flight_mask() == 0)
    }

    /// Exposes exactly the requested contiguous coarse mip tail to sampling.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/oversized range, incomplete uploads, or view creation failure.
    pub fn publish_texture_mips(
        &mut self,
        texture: &mut NativeTexture,
        resident_mips: u32,
    ) -> Result<(), AllocationError> {
        if resident_mips == 0 || resident_mips > texture.mip_count {
            return Err(AllocationError::ZeroSize);
        }
        let completed = self.reclaim_texture_transfers()?;
        let first_level = (texture.mip_count - resident_mips) as usize;
        // Every exposed level must contain its latest submitted update; zero denotes no upload.
        if texture.mip_completions[first_level..]
            .iter()
            .any(|value| *value == 0 || *value > completed)
        {
            return Err(AllocationError::NativeFailure);
        }
        if resident_mips == texture.resident_mips {
            return Ok(());
        }
        let storage = texture
            .texture
            .parentTexture()
            .ok_or(AllocationError::NativeFailure)?;
        let sampled = texture_view(&storage, texture.format, texture.mip_count, resident_mips)?;
        let old = core::mem::replace(&mut texture.texture, ThreadBound::new(sampled));
        // Standard Metal command buffers retain resources declared by `useResource`, so any
        // already-submitted frame owns the replaced view through completion.
        drop(old);
        texture.resident_mips = resident_mips;
        Ok(())
    }

    /// Copies a private RGBA8 texture into shared CPU-visible storage and returns packed rows.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, allocation failure, or rejected Metal commands.
    pub fn readback_texture_rgba8(
        &mut self,
        texture: &NativeTexture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, AllocationError> {
        let size = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|value| value.checked_mul(4))
            .ok_or(AllocationError::ZeroSize)?;
        if width == 0 || height == 0 {
            return Err(AllocationError::ZeroSize);
        }
        let storage = texture
            .texture
            .parentTexture()
            .ok_or(AllocationError::NativeFailure)?;
        let mut readback = self.allocate(
            AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                .map_err(|_| AllocationError::ZeroSize)?,
        )?;
        let result = (|| {
            let command = self
                .queue
                .commandBuffer()
                .ok_or(AllocationError::NativeFailure)?;
            // A direct readback bypasses frame-graph waits but still consumes base-mip writes.
            let ready = *texture
                .mip_completions
                .first()
                .ok_or(AllocationError::NativeFailure)?;
            command.encodeWaitForEvent_value(&self.texture_completion_event, ready);
            let blit = command
                .blitCommandEncoder()
                .ok_or(AllocationError::NativeFailure)?;
            // SAFETY: `copyFromTexture` writes the checked RGBA8 extent into the equally sized `readback.buffer` allocation, both storages outlive command completion, and `blit` is ended exactly once.
            unsafe {
                let readback_width =
                    usize::try_from(width).map_err(|_| AllocationError::NativeFailure)?;
                let readback_height =
                    usize::try_from(height).map_err(|_| AllocationError::NativeFailure)?;
                let readback_size =
                    usize::try_from(size).map_err(|_| AllocationError::NativeFailure)?;
                let row_bytes = readback_width
                    .checked_mul(4)
                    .ok_or(AllocationError::NativeFailure)?;
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(&storage, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: readback_width, height: readback_height, depth: 1 }, &readback.buffer, 0, row_bytes, readback_size);
                blit.endEncoding();
            }
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(AllocationError::NativeFailure);
            }
            self.invalidate(&mut readback, 0, size)?;
            let readback_size =
                usize::try_from(size).map_err(|_| AllocationError::NativeFailure)?;
            Ok(self.mapped_slice(&readback)?[..readback_size].to_vec())
        })();
        match (result, self.free(readback)) {
            (Ok(pixels), Ok(())) => Ok(pixels),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    /// Defers texture destruction until every referencing frame completes.
    ///
    /// # Errors
    ///
    /// Returns an error if deferring the texture exceeds the deferred-resource capacity.
    pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
        self.defer_resource(DeferredResource::Texture(texture))
    }
}

impl NativeContext {
    fn reclaim_texture_transfers(&mut self) -> Result<u64, AllocationError> {
        let completed = self.completed_texture_transfer_value()?;
        self.completed_texture_value = completed;
        self.pending_texture_transfers
            .retain(|pending| pending.value > completed);
        for stale in self.texture_staging.trim(ez_gfx_hal::QueueKind::TextureTransfer, completed) {
            self.free(stale)?;
        }
        Ok(completed)
    }

    /// Returns the completed value of the independent texture transfer stream.
    ///
    /// # Errors
    ///
    /// Returns an error when worker submission or a committed command buffer failed.
    pub fn completed_texture_transfer_value(&self) -> Result<u64, AllocationError> {
        if let Some(error) = self
            .texture_worker
            .as_ref()
            .and_then(ez_gfx_hal::TransferWorker::terminal_error)
        {
            return Err(error.to_allocation_error());
        }
        let mut completed = self.completed_texture_value;
        for pending in &self.pending_texture_transfers {
            if !pending.submission.completed()? {
                break;
            }
            completed = completed.max(pending.value);
        }
        Ok(completed)
    }
}
