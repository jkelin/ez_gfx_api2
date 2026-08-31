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
        // SAFETY: `desc` is the newly allocated `MTLTextureDescriptor`, and `mips.len()` is the validated mip-chain count consumed by `setMipmapLevelCount` during this send.
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
        let uploaded = (|| {
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
            let mut completions = Vec::with_capacity(mips.len());
            for (level, mip) in mips.iter().enumerate() {
                let size = mip.bytes.len() as u64;
                let mut upload = self.allocate(
                    AllocationRequest::new(size, 4, MemoryClass::Upload, true, None)
                        .map_err(|_| AllocationError::ZeroSize)?,
                )?;
                let submitted = (|| {
                    self.mapped_slice_mut(&mut upload)?[..mip.bytes.len()]
                        .copy_from_slice(mip.bytes);
                    self.flush(&mut upload, 0, size)?;
                    let command = self
                        .queue
                        .commandBuffer()
                        .ok_or(AllocationError::NativeFailure)?;
                    let blit = command
                        .blitCommandEncoder()
                        .ok_or(AllocationError::NativeFailure)?;
                    // SAFETY: `copyFromBuffer` reads the initialized, flushed `upload.buffer` range described by the checked RGBA8 strides into the validated `texture` mip level, both storages outlive command completion, and `blit` is ended exactly once.
                    unsafe {
                        let mip_width = usize::try_from(mip.width)
                            .map_err(|_| AllocationError::NativeFailure)?;
                        let mip_height = usize::try_from(mip.height)
                            .map_err(|_| AllocationError::NativeFailure)?;
                        let upload_size =
                            usize::try_from(size).map_err(|_| AllocationError::NativeFailure)?;
                        let row_bytes = mip_width
                            .checked_mul(4)
                            .ok_or(AllocationError::NativeFailure)?;
                        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&upload.buffer, 0, row_bytes, upload_size, MTLSize { width: mip_width, height: mip_height, depth: 1 }, &texture, 0, level, MTLOrigin { x: 0, y: 0, z: 0 });
                        blit.endEncoding();
                    }
                    command.commit();
                    let value = self.next_transfer_value;
                    self.next_transfer_value =
                        value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
                    command.waitUntilCompleted();
                    if command.status() != MTLCommandBufferStatus::Completed
                        || command.error().is_some()
                    {
                        return Err(AllocationError::NativeFailure);
                    }
                    self.completed_transfer_value = value;
                    CompletionToken::new(QueueKind::Transfer, value)
                        .map_err(|_| AllocationError::NativeFailure)
                })();
                let freed = self.free(upload);
                let completion = match (submitted, freed) {
                    (Ok(completion), Ok(())) => completion,
                    (Err(error), _) | (_, Err(error)) => return Err(error),
                };
                completions.push(completion);
            }
            Ok((sampler, completions))
        })();
        match uploaded {
            Ok((sampler, completions)) => Ok((
                NativeTexture {
                    texture: ThreadBound::new(texture),
                    allocation: ThreadBound::new(allocation),
                    sampler: ThreadBound::new(sampler),
                    binding,
                },
                completions,
            )),
            Err(error) => {
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
