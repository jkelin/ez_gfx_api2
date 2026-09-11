use super::{
    Allocation, AllocationCreateDesc, AllocationError, AllocationRequest, AllocationScheme,
    CompletionToken, DeferredResource, ImageMip, MemoryAllocator, MemoryClass, MemoryLocation,
    NativeAllocation, NativeContext, NativeTexture, QueueKind, SAMPLER_DESCRIPTOR_BINDING,
    SamplerAddressMode, SamplerFilter, TEXTURE_DESCRIPTOR_BINDING, TEXTURE_DESCRIPTOR_CAPACITY,
    TextureFormat, TextureRegion, TextureSamplerDesc, map_allocation_vk, map_allocator, map_vk,
    sampler_create_info,
    transfer::{VulkanTransferCopy, VulkanTransferJob},
    validate_texture_mips, validate_texture_region, vk,
};

fn texture_format_vk(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Bc1Unorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TextureFormat::Bc1Srgb => vk::Format::BC1_RGBA_SRGB_BLOCK,
        TextureFormat::Bc3Unorm => vk::Format::BC3_UNORM_BLOCK,
        TextureFormat::Bc3Srgb => vk::Format::BC3_SRGB_BLOCK,
        TextureFormat::Bc7Unorm => vk::Format::BC7_UNORM_BLOCK,
        TextureFormat::Bc7Srgb => vk::Format::BC7_SRGB_BLOCK,
        TextureFormat::Astc4x4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
        TextureFormat::Astc4x4Srgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
    }
}

fn resident_mip_range(mip_count: u32, resident_mips: u32) -> Option<(u32, u32)> {
    // Zero residency cannot be represented by a Vulkan image view; over-residency would expose
    // storage outside the allocation's mip chain.
    (resident_mips != 0 && resident_mips <= mip_count)
        .then(|| (mip_count - resident_mips, resident_mips))
}

fn texture_region_copy(region: &TextureRegion<'_>) -> Result<vk::BufferImageCopy, AllocationError> {
    Ok(vk::BufferImageCopy::default()
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: region.mip_level,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D {
            x: i32::try_from(region.x).map_err(|_| AllocationError::NativeFailure)?,
            y: i32::try_from(region.y).map_err(|_| AllocationError::NativeFailure)?,
            z: 0,
        })
        .image_extent(vk::Extent3D {
            width: region.width,
            height: region.height,
            depth: 1,
        }))
}

fn write_texture_descriptor(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    binding: u32,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let image_descriptor = vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let sampler_descriptor = vk::DescriptorImageInfo::default().sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(TEXTURE_DESCRIPTOR_BINDING)
            .dst_array_element(binding)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(core::slice::from_ref(&image_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(SAMPLER_DESCRIPTOR_BINDING)
            .dst_array_element(binding)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(core::slice::from_ref(&sampler_descriptor)),
    ];
    // SAFETY: the descriptor set, view, and sampler belong to this live device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

struct TextureUpload<'a> {
    mips: &'a [ImageMip<'a>],
    image: vk::Image,
    total: u64,
    cancellation: &'a std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl NativeContext {
    fn upload_texture_mips(
        &mut self,
        mut upload: NativeAllocation,
        request: &TextureUpload<'_>,
    ) -> Result<Vec<CompletionToken>, AllocationError> {
        let target = self.mapped_slice_mut(&mut upload)?;
        let mut offsets = Vec::with_capacity(request.mips.len());
        let mut offset = 0_usize;
        for mip in request.mips {
            let end = offset
                .checked_add(mip.bytes.len())
                .ok_or(AllocationError::NativeFailure)?;
            target
                .get_mut(offset..end)
                .ok_or(AllocationError::NativeFailure)?
                .copy_from_slice(mip.bytes);
            offsets.push(offset as u64);
            offset = end;
        }
        self.flush(&mut upload, 0, request.total)?;

        let first = self.next_texture_value;
        let next = first
            .checked_add(request.mips.len() as u64)
            .ok_or(AllocationError::NativeFailure)?;
        let completions = (first..next)
            .map(|value| CompletionToken::new(QueueKind::TextureTransfer, value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| AllocationError::NativeFailure)?;
        let worker = self
            .texture_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?;
        let mut jobs = Vec::with_capacity(request.mips.len());
        for (resident_index, level) in (0..request.mips.len()).rev().enumerate() {
            let mip = &request.mips[level];
            let region = vk::BufferImageCopy::default()
                .buffer_offset(offsets[level])
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: u32::try_from(level).map_err(|_| AllocationError::NativeFailure)?,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: mip.width,
                    height: mip.height,
                    depth: 1,
                });
            jobs.push(VulkanTransferJob {
                value: completions[resident_index].value,
                bytes: mip.bytes.len() as u64,
                cancelled: Some(request.cancellation.clone()),
                copy: VulkanTransferCopy::Texture {
                    source: upload.buffer,
                    destination: request.image,
                    region,
                    initialized: false,
                    stream_stage: u32::try_from(resident_index)
                        .map_err(|_| AllocationError::NativeFailure)?,
                },
            });
        }
        // Atomic admission prevents a rejected fine mip from freeing an already-live coarse copy.
        if let Err(error) = worker.submit_batch(jobs) {
            self.texture_staging
                .put(upload.allocation.size(), upload, None);
            return Err(error.to_allocation_error());
        }
        // Rejected chains leave the accepted completion highwater unchanged.
        self.next_texture_value = next;
        let completion = *completions.last().ok_or(AllocationError::NativeFailure)?;
        let capacity = upload.allocation.size();
        self.texture_staging.put(capacity, upload, Some(completion));
        Ok(completions)
    }
}

impl NativeContext {
    pub(super) fn destroy_unpublished_texture(
        &mut self,
        device: &ash::Device,
        image: vk::Image,
        view: Option<vk::ImageView>,
        sampler: Option<vk::Sampler>,
        allocation: Allocation,
    ) {
        // SAFETY: the unpublished sampler, view, and image have no descriptor references or pending queue use, and the image allocation is freed only after these destroy operations.
        unsafe {
            if let Some(sampler) = sampler {
                device.destroy_sampler(sampler, None);
            }
            if let Some(view) = view {
                device.destroy_image_view(view, None);
            }
            device.destroy_image(image, None);
        }
        if let Some(allocator) = self.allocator.as_mut() {
            let _ = allocator.free(allocation);
        }
    }

    fn create_texture_image(
        &mut self,
        device: &ash::Device,
        width: u32,
        height: u32,
        mip_count: u32,
        format: TextureFormat,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, Allocation), AllocationError> {
        let create = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(texture_format_vk(format))
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `create` is initialized without dangling extension pointers and its stack storage lives through `create_image`; no allocation callbacks are supplied.
        let image = unsafe { device.create_image(&create, None) }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
        // SAFETY: `image` is the undestroyed result of `device.create_image` immediately above, so `get_image_memory_requirements` may query it before binding.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = match self
            .allocator
            .as_mut()
            .expect("allocator checked before image creation")
            .allocate(&AllocationCreateDesc {
                name: "ez-gfx-texture",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: allocation failed before `image` was bound, published, or submitted, so `destroy_image` cannot race pending device use.
                unsafe { device.destroy_image(image, None) };
                return Err(map_allocator(&error));
            }
        };
        if let Err(error) =
            // SAFETY: `allocation` was created from this `image`'s requirements, so its memory type, aligned offset, and range satisfy `bind_image_memory` and remain allocated through the call.
            unsafe {
                device.bind_image_memory(image, allocation.memory(), allocation.offset())
            }
        {
            // SAFETY: `bind_image_memory` failed before publication or submission, so `image` has no pending use and is destroyed before `allocation` is freed.
            unsafe { device.destroy_image(image, None) };
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator initialized")
                .free(allocation);
            return Err(map_allocation_vk(map_vk(error)));
        }
        Ok((image, allocation))
    }

    /// Creates a sampled mip chain and submits each level under a distinct timeline value.
    ///
    /// # Errors
    ///
    /// Returns an error if the mip chain, format, or binding is invalid, required context state is
    /// unavailable, arithmetic overflows, or native allocation/submission fails.
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
        if sampler_desc.max_anisotropy > 1.0 && !self.sampler_anisotropy {
            return Err(AllocationError::NativeFailure);
        }
        self.texture_descriptor_set
            .ok_or(AllocationError::NativeFailure)?;
        let width = mips[0].width;
        let height = mips[0].height;
        let mip_count = u32::try_from(mips.len()).map_err(|_| AllocationError::NativeFailure)?;
        let total = mips
            .iter()
            .try_fold(0_u64, |sum, mip| sum.checked_add(mip.bytes.len() as u64))
            .ok_or(AllocationError::NativeFailure)?;
        let bucket = ez_gfx_hal::staging_bucket_size(total, ez_gfx_hal::DEFAULT_STAGING_POLICY)
            .map_err(|_| AllocationError::OutOfMemory)?;
        let upload_request = AllocationRequest::new(bucket, 4, MemoryClass::Upload, true, None)
            .map_err(|_| AllocationError::ZeroSize)?;
        let completed = self.completed_texture_transfer_value()?;
        for stale in self
            .texture_staging
            .trim(ez_gfx_hal::QueueKind::TextureTransfer, completed)
        {
            self.free(stale)?;
        }
        if self.allocator.is_none() {
            return Err(AllocationError::NativeFailure);
        }
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        let (image, allocation) = self.create_texture_image(
            &device,
            width,
            height,
            mip_count,
            format,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        )?;
        // SAFETY: the image is bound and the view initially selects only its coarsest mip. The
        // descriptor remains unpublished until that mip's transfer-to-graphics handoff completes.
        let view = match unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(texture_format_vk(format))
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: mip_count - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        } {
            Ok(view) => view,
            Err(error) => {
                self.destroy_unpublished_texture(&device, image, None, None, allocation);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let sampler_info = sampler_create_info(sampler_desc, mip_count);
        // SAFETY: `sampler_info` is initialized and lives through `create_sampler`, and anisotropy is requested only after the enabled-feature check above.
        let sampler = match unsafe { device.create_sampler(&sampler_info, None) } {
            Ok(sampler) => sampler,
            Err(error) => {
                self.destroy_unpublished_texture(&device, image, Some(view), None, allocation);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let upload = if let Some((_, upload)) =
            self.texture_staging
                .take(total, ez_gfx_hal::QueueKind::TextureTransfer, completed)
        {
            upload
        } else {
            match self.allocate(upload_request) {
                Ok(upload) => upload,
                Err(error) => {
                    self.destroy_unpublished_texture(
                        &device,
                        image,
                        Some(view),
                        Some(sampler),
                        allocation,
                    );
                    return Err(error);
                }
            }
        };
        let cancellation = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completions = match self.upload_texture_mips(
            upload,
            &TextureUpload {
                mips,
                image,
                total,
                cancellation: &cancellation,
            },
        ) {
            Ok(completions) => completions,
            Err(error) => {
                self.destroy_unpublished_texture(
                    &device,
                    image,
                    Some(view),
                    Some(sampler),
                    allocation,
                );
                return Err(error);
            }
        };
        let texture = NativeTexture {
            image,
            view,
            allocation,
            sampler,
            format,
            width,
            height,
            mip_count,
            resident_mips: 0,
            mip_completions: completions.iter().rev().map(|token| token.value).collect(),
            cancellation,
            binding,
            msaa: None,
        };
        Ok((texture, completions))
    }
    /// Creates an uninitialized single-mip color image for managed render-target use.
    ///
    /// The sampled image carries color-attachment, sampled, and transfer roles
    /// with no initial contents; the first render pass transitions and clears
    /// it. With `samples > 1` a second multisampled image renders the pass and
    /// resolves into the sampled image, which stays the only sampled, readback,
    /// and descriptor image. The returned texture reuses the texture record
    /// with inert transfer fields (`resident_mips` set, zero completion):
    /// route it only through render-target entry points, never through upload,
    /// publish, or region-update paths. The safe layer owns the true format;
    /// the stored texture format is the closest block-compatible value for
    /// record shape only.
    ///
    /// # Errors
    ///
    /// Returns an error for zero dimensions, an unsupported sample count or
    /// (non-color) format, excessive aggregate bytes, or native allocation failure.
    ///
    /// # Panics
    ///
    /// Panics if the allocator handle checked above is concurrently taken,
    /// which indicates corrupt context-owned state, never caller input.
    pub fn create_render_target(
        &mut self,
        format: ez_gfx_runtime::target::Format,
        width: u32,
        height: u32,
        binding: u32,
        samples: u8,
    ) -> Result<NativeTexture, AllocationError> {
        use ez_gfx_runtime::target::Format;
        let (vk_format, hal_format, bytes_per_texel) = match format {
            Format::Rgba8Unorm => (vk::Format::R8G8B8A8_UNORM, TextureFormat::Rgba8Unorm, 4),
            Format::Bgra8Srgb => (vk::Format::B8G8R8A8_SRGB, TextureFormat::Rgba8Srgb, 4),
            Format::Rgba16Float => (
                vk::Format::R16G16B16A16_SFLOAT,
                TextureFormat::Rgba8Unorm,
                8,
            ),
            _ => return Err(AllocationError::Unsupported),
        };
        let Some(sample_flags) = msaa::sample_count_flags(samples) else {
            return Err(AllocationError::Unsupported);
        };
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
        if self.allocator.is_none() {
            return Err(AllocationError::NativeFailure);
        }
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        let create = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `create` is initialized without dangling pointers and lives through
        // `create_image`; no allocation callbacks are supplied.
        let image = unsafe { device.create_image(&create, None) }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
        // SAFETY: the image is the undestroyed result of `create_image`, so requirements
        // may be queried before binding.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = match self
            .allocator
            .as_mut()
            .expect("allocator checked before image creation")
            .allocate(&AllocationCreateDesc {
                name: "ez-gfx-render-target",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: allocation failed before binding, publication, or submission.
                unsafe { device.destroy_image(image, None) };
                return Err(map_allocator(&error));
            }
        };
        // SAFETY: `allocation` was created from this `image`'s requirements.
        let bound =
            unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) };
        if let Err(error) = bound {
            // SAFETY: binding failed before publication or submission.
            unsafe { device.destroy_image(image, None) };
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator initialized")
                .free(allocation);
            return Err(map_allocation_vk(map_vk(error)));
        }
        // SAFETY: the image is bound; the single-mip view covers the whole target.
        let view = match unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk_format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        } {
            Ok(view) => view,
            Err(error) => {
                self.destroy_unpublished_texture(&device, image, None, None, allocation);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        // Render targets sample with fixed nearest filtering until the sampled-binding
        // slice assigns heap samplers; the sampler is never null.
        let (resolve_sampler, allocation) =
            self.create_resolve_sampler(&device, image, view, allocation)?;
        // Single-sample targets render directly into the sampled image; the
        // multisampled image below stays absent.
        let (msaa, allocation) = self.create_msaa_storage(
            vk_format,
            width,
            height,
            samples,
            sample_flags,
            (image, view, resolve_sampler, allocation),
        )?;
        Ok(NativeTexture {
            image,
            view,
            allocation,
            sampler: resolve_sampler,
            format: hal_format,
            width,
            height,
            mip_count: 1,
            resident_mips: 1,
            mip_completions: vec![0],
            cancellation: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            binding,
            msaa,
        })
    }
    /// Copies one validated tightly packed region into its native mip.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid region, unavailable staging/device state, or failed queue
    /// submission. Borrowed bytes are copied into reusable staging before this method returns.
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
        let size = region.bytes.len() as u64;
        let bucket = ez_gfx_hal::staging_bucket_size(size, ez_gfx_hal::DEFAULT_STAGING_POLICY)
            .map_err(|_| AllocationError::OutOfMemory)?;
        let completed = self.completed_texture_transfer_value()?;
        for stale in self
            .texture_staging
            .trim(ez_gfx_hal::QueueKind::TextureTransfer, completed)
        {
            self.free(stale)?;
        }
        let request = AllocationRequest::new(bucket, 4, MemoryClass::Upload, true, None)
            .map_err(|_| AllocationError::ZeroSize)?;
        let mut upload = if let Some((_, upload)) =
            self.texture_staging
                .take(size, ez_gfx_hal::QueueKind::TextureTransfer, completed)
        {
            upload
        } else {
            self.allocate(request)?
        };
        self.mapped_slice_mut(&mut upload)?
            .get_mut(..region.bytes.len())
            .ok_or(AllocationError::NativeFailure)?
            .copy_from_slice(region.bytes);
        self.flush(&mut upload, 0, size)?;

        let value = self.next_texture_value;
        let next = value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
        let completion = CompletionToken::new(QueueKind::TextureTransfer, value)
            .map_err(|_| AllocationError::NativeFailure)?;
        let copy = texture_region_copy(region)?;
        let submitted = self
            .texture_worker
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .submit(VulkanTransferJob {
                value,
                bytes: size,
                cancelled: Some(texture.cancellation.clone()),
                copy: VulkanTransferCopy::Texture {
                    source: upload.buffer,
                    destination: texture.image,
                    region: copy,
                    initialized: true,
                    stream_stage: 0,
                },
            });
        if let Err(error) = submitted {
            self.texture_staging
                .put(upload.allocation.size(), upload, None);
            return Err(error.to_allocation_error());
        }
        self.next_texture_value = next;
        self.texture_staging
            .put(upload.allocation.size(), upload, Some(completion));
        // Metadata uses image mip indices, including overwrites outside the current view.
        texture.mip_completions[region.mip_level as usize] = value;
        Ok(completion)
    }

    /// Publishes exactly the requested contiguous coarse mip range after transfer completion.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is invalid, its handoff has not completed, or creating the
    /// replacement native view fails.
    pub fn publish_texture_mips(
        &mut self,
        texture: &mut NativeTexture,
        resident_mips: u32,
    ) -> Result<(), AllocationError> {
        let (base_mip_level, level_count) = resident_mip_range(texture.mip_count, resident_mips)
            .ok_or(AllocationError::ZeroSize)?;
        if resident_mips == texture.resident_mips {
            return Ok(());
        }
        let first = usize::try_from(base_mip_level).map_err(|_| AllocationError::NativeFailure)?;
        let required = texture.mip_completions[first..]
            .iter()
            .copied()
            .max()
            .ok_or(AllocationError::NativeFailure)?;
        if self.completed_texture_transfer_value()? < required {
            return Err(AllocationError::NativeFailure);
        }
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        let descriptor_set = self
            .texture_descriptor_set
            .ok_or(AllocationError::NativeFailure)?;
        let view = if resident_mips == 1 && texture.resident_mips == 0 {
            texture.view
        } else {
            // SAFETY: the complete image allocation contains every selected level, and transfer
            // completion above establishes shader-readable layout and graphics-family ownership.
            unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(texture.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(texture_format_vk(texture.format))
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level,
                            level_count,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?
        };
        write_texture_descriptor(
            &device,
            descriptor_set,
            texture.binding,
            view,
            texture.sampler,
        );
        if view != texture.view {
            let old = core::mem::replace(&mut texture.view, view);
            self.defer_resource(DeferredResource::TextureView(old))?;
        }
        texture.resident_mips = resident_mips;
        Ok(())
    }

    /// Reports whether bindless descriptor rewrites can avoid every submitted frame.
    pub fn texture_descriptor_update_ready(&self) -> bool {
        self.in_flight_mask() == 0
    }

    /// Queries physical-device format features for render-target formats.
    ///
    /// Reports optimal-tiling color, sampled, and storage roles for RGBA8,
    /// BGRA sRGB, and RGBA16F, plus depth attachment support for D32 float.
    /// Multisample ceilings come from per-format image queries, not the role bits.
    ///
    /// # Errors
    ///
    /// Returns an error when no physical device is selected.
    pub fn probe_target_formats(
        &self,
    ) -> Result<ez_gfx_runtime::target::FormatCapabilities, AllocationError> {
        use ez_gfx_runtime::target::Format;
        let physical = self.physical_device.ok_or(AllocationError::NativeFailure)?;
        // Color formats attach for rendering; depth attaches for depth testing.
        // A failed sample query omits that format so resolution fails closed.
        let candidates = [
            (
                Format::Rgba8Unorm,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            (
                Format::Bgra8Srgb,
                vk::Format::B8G8R8A8_SRGB,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            (
                Format::Rgba16Float,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            (
                Format::Depth32Float,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
        ];
        let mut supports = Vec::with_capacity(candidates.len());
        for (format, vk_format, usage) in candidates {
            // SAFETY: the physical device belongs to this instance for the call.
            let features = unsafe {
                self.instance
                    .get_physical_device_format_properties(physical, vk_format)
            }
            .optimal_tiling_features;
            // A failed sample query keeps single-sample admission, matching the
            // Direct3D 12 floor; the role bits still gate the format.
            let max_samples =
                msaa::max_sample_count(&self.instance, physical, vk_format, usage).unwrap_or(1);
            if let Some(support) = msaa::support_for_target_format(format, features, max_samples) {
                supports.push(support);
            }
        }
        ez_gfx_runtime::target::FormatCapabilities::new(supports)
            .map_err(|_| AllocationError::NativeFailure)
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
    /// Returns an error when native completion cannot be queried.
    pub fn texture_retirement_ready(
        &self,
        completion: CompletionToken,
    ) -> Result<bool, AllocationError> {
        Ok(self.completed_texture_transfer_value()? >= completion.value
            && self.in_flight_mask() == 0)
    }

    /// Defers texture destruction until every referencing frame completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the texture cannot be queued for deferred destruction.
    pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
        self.defer_resource(DeferredResource::Texture(texture))
    }

    /// Copies a shader-readable image to host-visible memory and returns tightly packed RGBA8 pixels.
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
        let size = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|value| value.checked_mul(4))
            .ok_or(AllocationError::ZeroSize)?;
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        let queue = self.graphics_queue.ok_or(AllocationError::NativeFailure)?;
        let family = self
            .graphics_queue_family
            .ok_or(AllocationError::NativeFailure)?;
        // SAFETY: `family` belongs to this initialized device and create-info storage spans the call.
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
        }
        .map_err(|error| map_allocation_vk(map_vk(error)))?;
        let mut readback = match self.allocate(
            AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                .map_err(|_| AllocationError::ZeroSize)?,
        ) {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: no command buffer was allocated from this new pool.
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(error);
            }
        };
        // SAFETY: this pool is graphics-family local and exclusively owned by this call.
        let command = match unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(commands) => commands[0],
            Err(error) => {
                let _ = self.free(readback);
                // SAFETY: allocation failed before any command buffer became pending.
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        let mut submitted = false;
        let graphics_queue_lock = self.graphics_queue_lock.clone();
        let result = (|| {
            // SAFETY: `command` was just allocated from `pool` in the initial state, host access is serialized by `&mut self`, and the begin-info storage lives through `begin_command_buffer`.
            unsafe {
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_copy = vk::ImageMemoryBarrier::default()
                .image(texture.image)
                .subresource_range(range)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            // SAFETY: `command` is recording, `texture.image` is retained by the shared texture reference, and the `to_copy` barrier slice lives through `cmd_pipeline_barrier`.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    core::slice::from_ref(&to_copy),
                );
            };
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,

                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            // SAFETY: `command` is recording, the preceding barrier establishes transfer-source layout, the image and `readback.buffer` are retained through completion, and `region` lives through the call.
            unsafe {
                device.cmd_copy_image_to_buffer(
                    command,
                    texture.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    core::slice::from_ref(&region),
                );
            };
            let to_shader = vk::ImageMemoryBarrier::default()
                .image(texture.image)
                .subresource_range(range)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            // SAFETY: `command` is recording, `texture.image` is retained through execution, and `to_shader` lives through barrier recording, after which the command remains recording for `end_command_buffer`.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    core::slice::from_ref(&to_shader),
                );
                device.end_command_buffer(command)
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            let submit = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command));
            let _queue_guard = graphics_queue_lock.lock();
            // SAFETY: the closed command buffer and graphics queue belong to this live device.
            unsafe {
                device.queue_submit(queue, core::slice::from_ref(&submit), vk::Fence::null())
            }
            .map_err(|error| map_allocation_vk(map_vk(error)))?;
            submitted = true;
            // SAFETY: the graphics queue remains live and locked through this wait.
            unsafe { device.queue_wait_idle(queue) }
                .map_err(|error| map_allocation_vk(map_vk(error)))?;
            self.invalidate(&mut readback, 0, size)?;
            Ok(self.mapped_slice(&readback)?
                [..usize::try_from(size).map_err(|_| AllocationError::NativeFailure)?]
                .to_vec())
        })();
        if submitted && result.is_err() {
            // SAFETY: `submitted` means `queue_submit` succeeded; queue access is serialized by `&mut self`, and this idle wait precedes freeing the command and readback storage.
            let _ = unsafe { device.queue_wait_idle(queue) };
        }
        // SAFETY: `command` came from `pool`; unsubmitted paths never made it pending, submitted paths wait for completion, and the slice lives through `free_command_buffers`.
        unsafe { device.free_command_buffers(pool, &[command]) };
        // SAFETY: the only command buffer was freed above and no pool work remains pending.
        unsafe { device.destroy_command_pool(pool, None) };
        let freed = self.free(readback);
        match (result, freed) {
            (Ok(pixels), Ok(())) => Ok(pixels),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }
}

impl NativeContext {
    /// Returns the completed value of the independent texture transfer timeline.
    ///
    /// # Errors
    ///
    /// Returns an error when the texture worker failed or the device timeline is unavailable.
    pub fn completed_texture_transfer_value(&self) -> Result<u64, AllocationError> {
        if let Some(error) = self
            .texture_worker
            .as_ref()
            .and_then(ez_gfx_hal::TransferWorker::terminal_error)
        {
            return Err(error.to_allocation_error());
        }
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        let timeline = self
            .texture_timeline
            .ok_or(AllocationError::NativeFailure)?;

        // SAFETY: the timeline belongs to the retained live device.
        unsafe { device.get_semaphore_counter_value(timeline) }
            .map_err(|error| map_allocation_vk(map_vk(error)))
    }
}

#[path = "texture_msaa.rs"]
mod msaa;
#[cfg(test)]
#[path = "texture_tests.rs"]
mod texture_tests;
