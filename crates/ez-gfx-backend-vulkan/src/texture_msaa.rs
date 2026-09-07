//! Vulkan multisampled render-target storage, resolve samplers, and format support.

use super::{
    Allocation, AllocationCreateDesc, AllocationError, AllocationScheme, MemoryLocation,
    NativeContext, SamplerAddressMode, SamplerFilter, TextureSamplerDesc, map_allocation_vk,
    map_allocator, map_vk, sampler_create_info, vk,
};

/// Maps optimal-tiling feature bits onto render-target roles for one format.
///
/// Depth formats never report color or storage roles; sampling follows the
/// sampled-image bit on every format. A depth candidate without attachment
/// support is omitted so resolution fails with a diagnostic instead of
/// selecting an unusable format. The caller supplies the probed multisample
/// ceiling; declarations above it fail resolution exactly like before.
pub(super) fn support_for_target_format(
    format: ez_gfx_runtime::target::Format,
    features: vk::FormatFeatureFlags,
    max_samples: u8,
) -> Option<ez_gfx_runtime::target::FormatSupport> {
    use ez_gfx_core::capability::CompressionSupport;
    use ez_gfx_runtime::target::Format;
    if format == Format::Depth32Float
        && !features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    {
        return None;
    }
    let (color, sampled, storage) = match format {
        Format::Depth32Float => (
            false,
            features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
            false,
        ),
        _ => (
            features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
            features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
            features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
        ),
    };
    // Only the role bits vary per device; the ceiling comes from the image
    // format query below.
    Some(
        ez_gfx_runtime::target::FormatSupport::new(
            format,
            color,
            sampled,
            storage,
            max_samples,
            CompressionSupport::NONE,
        )
        .expect("probed sample counts are always valid"),
    )
}

/// Maps a validated sample count onto its Vulkan flag; other values are rejected by declaration validation before reaching here.
pub(super) fn sample_count_flags(samples: u8) -> Option<vk::SampleCountFlags> {
    match samples {
        1 => Some(vk::SampleCountFlags::TYPE_1),
        2 => Some(vk::SampleCountFlags::TYPE_2),
        4 => Some(vk::SampleCountFlags::TYPE_4),
        8 => Some(vk::SampleCountFlags::TYPE_8),
        _ => None,
    }
}

/// Queries the highest multisample count the device can attach for one format and usage.
///
/// A failed query omits the format so resolution fails with a diagnostic
/// instead of selecting an unusable count; single-sample stays admissible
/// whenever the image itself is creatable.
pub(super) fn max_sample_count(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Option<u8> {
    // SAFETY: the physical device belongs to this instance for the call.
    let properties = unsafe {
        instance.get_physical_device_image_format_properties(
            physical,
            format,
            vk::ImageType::TYPE_2D,
            vk::ImageTiling::OPTIMAL,
            usage,
            vk::ImageCreateFlags::empty(),
        )
    }
    .ok()?;
    let counts = properties.sample_counts;
    if counts.contains(vk::SampleCountFlags::TYPE_8) {
        Some(8)
    } else if counts.contains(vk::SampleCountFlags::TYPE_4) {
        Some(4)
    } else if counts.contains(vk::SampleCountFlags::TYPE_2) {
        Some(2)
    } else if counts.contains(vk::SampleCountFlags::TYPE_1) {
        Some(1)
    } else {
        None
    }
}

impl NativeContext {
    /// Creates multisampled render storage beside an unpublished resolve image.
    ///
    /// Single-sample targets need nothing and keep every resolve part; the
    /// returned allocation always survives for the caller to publish. Failures
    /// free the unpublished resolve parts before returning.
    ///
    /// # Errors
    ///
    /// Returns an error for native image, allocation, binding, or view failure.
    pub(super) fn create_msaa_storage(
        &mut self,
        vk_format: vk::Format,
        width: u32,
        height: u32,
        samples: u8,
        sample_flags: vk::SampleCountFlags,
        resolve: (vk::Image, vk::ImageView, vk::Sampler, Allocation),
    ) -> Result<(Option<crate::MsaaStorage>, Allocation), AllocationError> {
        let device = self
            .device
            .as_ref()
            .ok_or(AllocationError::NativeFailure)?
            .clone();
        let (image, view, resolve_sampler, allocation) = resolve;
        // Single-sample targets render directly into the sampled image; the
        // multisampled image below stays absent.
        let msaa = if samples == 1 {
            None
        } else {
            let create_msaa = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk_format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(sample_flags)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // SAFETY: `create_msaa` is initialized without dangling pointers and
            // lives through `create_image`; no allocation callbacks are supplied.
            let msaa_image = match unsafe { device.create_image(&create_msaa, None) } {
                Ok(image) => image,
                Err(error) => {
                    self.destroy_unpublished_texture(
                        &device,
                        image,
                        Some(view),
                        Some(resolve_sampler),
                        allocation,
                    );
                    return Err(map_allocation_vk(map_vk(error)));
                }
            };
            // SAFETY: the image is the undestroyed result of `create_image`, so
            // requirements may be queried before binding.
            let msaa_requirements = unsafe { device.get_image_memory_requirements(msaa_image) };
            let msaa_allocation = match self
                .allocator
                .as_mut()
                .expect("allocator initialized")
                .allocate(&AllocationCreateDesc {
                    name: "ez-gfx-render-target-msaa",
                    requirements: msaa_requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                }) {
                Ok(allocation) => allocation,
                Err(error) => {
                    // SAFETY: allocation failed before binding, publication, or submission.
                    unsafe { device.destroy_image(msaa_image, None) };
                    self.destroy_unpublished_texture(
                        &device,
                        image,
                        Some(view),
                        Some(resolve_sampler),
                        allocation,
                    );
                    return Err(map_allocator(&error));
                }
            };
            // SAFETY: `msaa_allocation` was created from this image's requirements.
            if let Err(error) = unsafe {
                device.bind_image_memory(
                    msaa_image,
                    msaa_allocation.memory(),
                    msaa_allocation.offset(),
                )
            } {
                // SAFETY: binding failed before publication or submission.
                unsafe { device.destroy_image(msaa_image, None) };
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator initialized")
                    .free(msaa_allocation);
                self.destroy_unpublished_texture(
                    &device,
                    image,
                    Some(view),
                    Some(resolve_sampler),
                    allocation,
                );
                return Err(map_allocation_vk(map_vk(error)));
            }
            // SAFETY: the image is bound; the single-mip view covers the whole target.
            let msaa_view = match unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(msaa_image)
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
                    self.destroy_unpublished_texture(
                        &device,
                        msaa_image,
                        None,
                        None,
                        msaa_allocation,
                    );
                    self.destroy_unpublished_texture(
                        &device,
                        image,
                        Some(view),
                        Some(resolve_sampler),
                        allocation,
                    );
                    return Err(map_allocation_vk(map_vk(error)));
                }
            };
            Some(crate::MsaaStorage {
                image: msaa_image,
                view: msaa_view,
                allocation: msaa_allocation,
                samples,
            })
        };
        Ok((msaa, allocation))
    }
    /// Creates the fixed nearest sampler for a render target's sampled image.
    ///
    /// The returned allocation always survives for the caller to publish;
    /// failures free the unpublished resolve parts before returning.
    ///
    /// # Errors
    ///
    /// Returns an error for native sampler creation failure.
    pub(super) fn create_resolve_sampler(
        &mut self,
        device: &ash::Device,
        image: vk::Image,
        view: vk::ImageView,
        allocation: Allocation,
    ) -> Result<(vk::Sampler, Allocation), AllocationError> {
        // Render targets sample with fixed nearest filtering until the sampled-binding
        // slice assigns heap samplers; the sampler is never null.
        // SAFETY: the descriptor is fully specified with valid filter and clamp modes.
        let resolve_sampler = match unsafe {
            device.create_sampler(
                &sampler_create_info(
                    TextureSamplerDesc {
                        min_filter: SamplerFilter::Nearest,
                        mag_filter: SamplerFilter::Nearest,
                        max_anisotropy: 1.0,
                        address_u: SamplerAddressMode::Clamp,
                        address_v: SamplerAddressMode::Clamp,
                        address_w: SamplerAddressMode::Clamp,
                    },
                    1,
                ),
                None,
            )
        } {
            Ok(resolve_sampler) => resolve_sampler,
            Err(error) => {
                self.destroy_unpublished_texture(device, image, Some(view), None, allocation);
                return Err(map_allocation_vk(map_vk(error)));
            }
        };
        Ok((resolve_sampler, allocation))
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;
    use ez_gfx_runtime::target::{Format, FormatSupport};

    #[test]
    fn feature_bits_select_color_sampled_and_storage_roles() {
        // Full optimal features admit every render-target role for RGBA8.
        let full = vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::STORAGE_IMAGE;
        let support = support_for_target_format(Format::Rgba8Unorm, full, 4).unwrap();
        assert_eq!(
            support,
            FormatSupport::new(
                Format::Rgba8Unorm,
                true,
                true,
                true,
                4,
                ez_gfx_core::capability::CompressionSupport::NONE
            )
            .unwrap()
        );
    }

    #[test]
    fn ceiling_above_declaration_admits_multisample_resolution() {
        use ez_gfx_runtime::target::{ClearValue, TargetDeclaration, TargetUsage};
        // A ceiling of 4 admits 1/2/4-sample declarations and rejects 8-sample ones.
        let full = vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::STORAGE_IMAGE;
        let capabilities = ez_gfx_runtime::target::FormatCapabilities::new(vec![
            support_for_target_format(Format::Rgba8Unorm, full, 4).unwrap(),
        ])
        .unwrap();
        for samples in [1, 2, 4] {
            let declaration = TargetDeclaration::new(
                "msaa",
                TargetUsage::Color,
                1.0,
                samples,
                vec![Format::Rgba8Unorm],
                ClearValue::None,
                true,
            )
            .unwrap();
            assert_eq!(
                capabilities.resolve(&declaration).unwrap(),
                Format::Rgba8Unorm
            );
        }
        let over = TargetDeclaration::new(
            "msaa",
            TargetUsage::Color,
            1.0,
            8,
            vec![Format::Rgba8Unorm],
            ClearValue::None,
            true,
        )
        .unwrap();
        assert!(
            capabilities
                .resolve(&over)
                .is_err_and(|error| error
                    == ez_gfx_runtime::target::TargetError::UnsupportedFormat)
        );
    }

    #[test]
    fn missing_bits_withhold_only_their_roles() {
        // A sampled-only format resolves for sampling but never as a color target.
        let sampled =
            support_for_target_format(Format::Rgba8Unorm, vk::FormatFeatureFlags::SAMPLED_IMAGE, 1)
                .unwrap();
        assert!(!sampled.color);
        assert!(sampled.sampled);
        assert!(!sampled.storage);
    }

    #[test]
    fn depth_support_never_reports_color_or_storage() {
        // Depth aspects carry no color/storage roles regardless of feature bits.
        let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE;
        let support = support_for_target_format(Format::Depth32Float, features, 1).unwrap();
        assert!(!support.color);
        assert!(!support.storage);
        assert!(support.sampled);
    }

    #[test]
    fn depth_without_attachment_support_is_omitted() {
        // Omitting the record makes resolution fail with UnsupportedFormat
        // instead of selecting an unusable depth format.
        assert!(
            support_for_target_format(
                Format::Depth32Float,
                vk::FormatFeatureFlags::SAMPLED_IMAGE,
                1
            )
            .is_none()
        );
    }
}
