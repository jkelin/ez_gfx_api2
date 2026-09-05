use super::{
    AllocationCreateDesc, AllocationError, AllocationRequest, CompletionToken, DeferredResource,
    ImageMip, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLHeap, MTLOrigin, MTLPixelFormat, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSize, MTLStorageMode, MTLTextureDescriptor,
    MTLTextureUsage, MemoryAllocator, MemoryClass, NativeContext, NativeTexture, QueueKind,
    SamplerAddressMode, SamplerFilter, TEXTURE_DESCRIPTOR_CAPACITY, TextureSamplerDesc,
    ThreadBound, map_allocator, validate_rgba8_mips,
};

impl NativeContext {
    /// Creates a sampled RGBA8 texture and uploads its complete mip chain.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid mips, allocation failure, or rejected Metal commands.
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
        let texture_width = usize::try_from(width).map_err(|_| AllocationError::NativeFailure)?;
        let texture_height = usize::try_from(height).map_err(|_| AllocationError::NativeFailure)?;
        // SAFETY: dimensions and mip count were validated against every bounded payload.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                texture_width,
                texture_height,
                mips.len() > 1,
            )
        };
        // SAFETY: `desc` is live and the validated mip count is consumed during this send.
        unsafe { desc.setMipmapLevelCount(mips.len()) };
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(MTLStorageMode::Private);
        let allocation_desc = AllocationCreateDesc::texture(&self.device, "ez-gfx-texture", &desc);
        let allocation = self
            .allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .allocate(&allocation_desc)
            .map_err(map_allocator)?;
        let allocation_offset =
            usize::try_from(allocation.offset()).map_err(|_| AllocationError::NativeFailure)?;
        // SAFETY: the allocation belongs to this heap and the checked offset describes it.
        let Some(texture) = (unsafe {
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
        let total = mips
            .iter()
            .try_fold(0_u64, |sum, mip| sum.checked_add(mip.bytes.len() as u64));
        let Some(total) = total else {
            drop(texture);
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(&allocation)
                .map_err(map_allocator)?;
            return Err(AllocationError::NativeFailure);
        };
        let bucket = ez_gfx_hal::staging_bucket_size(total, ez_gfx_hal::DEFAULT_STAGING_POLICY)
            .map_err(|_| AllocationError::OutOfMemory)?;
        let completed = self.completed_texture_transfer_value()?;
        self.completed_texture_value = completed;
        self.pending_texture_transfers
            .retain(|pending| pending.value > completed);
        for stale in self.texture_staging.trim(completed) {
            self.free(stale)?;
        }
        let request = AllocationRequest::new(bucket, 4, MemoryClass::Upload, true, None)?;
        let mut upload = if let Some((_, upload)) = self.texture_staging.take(total, completed) {
            upload
        } else {
            match self.allocate(request) {
                Ok(upload) => upload,
                Err(error) => {
                    drop(texture);
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
            for mip in mips {
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
            let command = self
                .texture_queue
                .commandBuffer()
                .ok_or(AllocationError::NativeFailure)?;
            let blit = command
                .blitCommandEncoder()
                .ok_or(AllocationError::NativeFailure)?;
            let mut source_offset = 0_usize;
            for (level, mip) in mips.iter().enumerate() {
                let mip_width =
                    usize::try_from(mip.width).map_err(|_| AllocationError::NativeFailure)?;
                let mip_height =
                    usize::try_from(mip.height).map_err(|_| AllocationError::NativeFailure)?;
                let row_bytes = mip_width
                    .checked_mul(4)
                    .ok_or(AllocationError::NativeFailure)?;
                // SAFETY: every source interval was copied and flushed above; the validated mip geometry and retained objects outlive command completion.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &upload.buffer,
                        source_offset,
                        row_bytes,
                        mip.bytes.len(),
                        MTLSize { width: mip_width, height: mip_height, depth: 1 },
                        &texture,
                        0,
                        level,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
                }
                source_offset = source_offset
                    .checked_add(mip.bytes.len())
                    .ok_or(AllocationError::NativeFailure)?;
            }
            blit.endEncoding();
            let first = self.next_texture_value;
            self.next_texture_value = first
                .checked_add(mips.len() as u64)
                .ok_or(AllocationError::NativeFailure)?;
            let completions = (first..self.next_texture_value)
                .map(|value| CompletionToken::new(QueueKind::TextureTransfer, value))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| AllocationError::NativeFailure)?;
            let completion = *completions.last().ok_or(AllocationError::NativeFailure)?;
            let value = completion.value;
            let pending_command = command.clone();
            self.texture_worker
                .as_ref()
                .ok_or(AllocationError::NativeFailure)?
                .submit(super::transfer::MetalTransferJob {
                    value,
                    bytes: total,
                    command: super::transfer::TransferCommand::new(command),
                })
                .map_err(|error| match error {
                    ez_gfx_hal::TransferWorkerError::Full => AllocationError::OutOfMemory,
                    ez_gfx_hal::TransferWorkerError::Failed => AllocationError::NativeFailure,
                })?;
            self.pending_texture_transfers.push(super::PendingTransfer {
                value,
                command: ThreadBound::new(pending_command),
            });
            Ok((sampler, completion, completions))
        })();
        match submitted {
            Ok((sampler, completion, completions)) => {
                let capacity = upload.allocation.size();
                self.texture_staging.put(capacity, upload, Some(completion));
                Ok((
                    NativeTexture {
                        texture: ThreadBound::new(texture),
                        allocation: ThreadBound::new(allocation),
                        sampler: ThreadBound::new(sampler),
                        binding,
                    },
                    completions,
                ))
            }
            Err(error) => {
                let _ = self.free(upload);
                drop(texture);
                self.allocator
                    .as_mut()
                    .ok_or(AllocationError::NativeFailure)?
                    .free(&allocation)
                    .map_err(map_allocator)?;
                Err(error)
            }
        }
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
        let mut readback = self.allocate(
            AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                .map_err(|_| AllocationError::ZeroSize)?,
        )?;
        let result = (|| {
            let command = self
                .queue
                .commandBuffer()
                .ok_or(AllocationError::NativeFailure)?;
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
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(&texture.texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: readback_width, height: readback_height, depth: 1 }, &readback.buffer, 0, row_bytes, readback_size);
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
    /// Returns the completed value of the independent texture transfer stream.
    pub fn completed_texture_transfer_value(&self) -> Result<u64, AllocationError> {
        if self
            .texture_worker
            .as_ref()
            .is_some_and(ez_gfx_hal::TransferWorker::failed)
        {
            return Err(AllocationError::NativeFailure);
        }
        let mut completed = self.completed_texture_value;
        for pending in &self.pending_texture_transfers {
            match pending.command.status() {
                MTLCommandBufferStatus::Completed if pending.command.error().is_none() => {
                    completed = completed.max(pending.value);
                }
                MTLCommandBufferStatus::Error => return Err(AllocationError::NativeFailure),
                _ => break,
            }
        }
        Ok(completed)
    }
}
