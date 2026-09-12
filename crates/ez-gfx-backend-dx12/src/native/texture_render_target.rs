//! Direct3D 12 render-target creation, probe format support, and sample ceilings.

use super::{
    AllocationError, ID3D12DescriptorHeap, ImageMip, NativeContext, NativeTexture,
    TEXTURE_DESCRIPTOR_CAPACITY, TextureFormat, map_allocation_windows,
};

/// Maps D3D12 format-support bits onto render-target roles for one format.
///
/// Depth formats never report color or storage roles; sampling follows the
/// texture bit on every format. The caller supplies the probed multisample
/// ceiling; declarations above it fail resolution exactly like before.
pub(super) fn support_for_target_format(
    format: ez_gfx_runtime::target::Format,
    support: windows::Win32::Graphics::Direct3D12::D3D12_FORMAT_SUPPORT1,
    max_samples: u8,
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
    // Only the role bits vary per device; the ceiling comes from the quality query below.
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

/// Queries the highest multisample count with at least one quality level for one format.
///
/// Counts without quality levels stay excluded; single-sample rendering is
/// always available, so the floor is 1.
pub(super) fn max_sample_count(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
) -> u8 {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
    };
    let mut ceiling: u8 = 1;
    for count in [2_u8, 4, 8] {
        let mut data = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: format,
            SampleCount: u32::from(count),
            Flags: windows::Win32::Graphics::Direct3D12::D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
            NumQualityLevels: 0,
        };
        // SAFETY: `data` is a live aligned query record sized exactly, and the
        // device outlives the call.
        let supported = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                (&raw mut data).cast(),
                u32::try_from(core::mem::size_of_val(&data)).unwrap_or(u32::MAX),
            )
        }
        .is_ok()
            && data.NumQualityLevels > 0;
        if supported {
            ceiling = count;
        }
    }
    ceiling
}

impl NativeContext {
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
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        };
        let (dxgi, hal_format, bytes_per_texel): (_, TextureFormat, u64) = match format {
            Format::Rgba8Unorm => (DXGI_FORMAT_R8G8B8A8_UNORM, TextureFormat::Rgba8Unorm, 4),
            Format::Bgra8Srgb => (DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, TextureFormat::Rgba8Srgb, 4),
            Format::Rgba16Float => (DXGI_FORMAT_R16G16B16A16_FLOAT, TextureFormat::Rgba8Unorm, 8),
            _ => return Err(AllocationError::Unsupported),
        };
        if !matches!(samples, 1 | 2 | 4 | 8) {
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
        if bytes
            > u64::try_from(ez_gfx_texture_manager::texture::MAX_TEXTURE_BYTES).unwrap_or(u64::MAX)
        {
            return Err(AllocationError::OutOfMemory);
        }
        // A single mip bypasses block-alignment validation; dimensions stay logical.
        let mips = [ImageMip {
            width,
            height,
            bytes: &[],
        }];
        let (resource, allocation, _) =
            self.create_texture_resource(&mips, dxgi, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 1)?;
        // The RTV heap holds the resolve view first and the multisampled view
        // second; it drops with the texture record after GPU retirement.
        let descriptors = if samples == 1 { 1 } else { 2 };
        // SAFETY: the descriptor count covers the views created below and the
        // device outlives the returned heap through the texture record.
        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            self.device.CreateDescriptorHeap(
                &windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: descriptors,
                    Flags: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                },
            )
        }
        .map_err(|error| map_allocation_windows(&error))?;
        // SAFETY: the heap retains its descriptor slots for the calls below.
        let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let rtv_desc = windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC {
            Format: dxgi,
            ViewDimension: windows::Win32::Graphics::Direct3D12::D3D12_RTV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        // SAFETY: `rtv` addresses the heap's first slot and `desc` selects a
        // matching view of the retained resource through the call.
        unsafe {
            self.device
                .CreateRenderTargetView(&resource, Some(&raw const rtv_desc), rtv);
        }
        // Single-sample targets render directly into the resolve resource; the
        // multisampled storage below stays absent.
        let msaa = if samples == 1 {
            None
        } else {
            let (msaa_resource, msaa_allocation, _) = match self.create_texture_resource(
                &mips,
                dxgi,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                u32::from(samples),
            ) {
                Ok(created) => created,
                Err(error) => {
                    // The resolve resource never published; release it before failing.
                    // The allocator initialized the resolve image above, so it
                    // stays available for this release.
                    drop(resource);
                    if let Some(allocator) = self.allocator.as_mut() {
                        let _ = allocator.free(allocation);
                    }
                    return Err(error);
                }
            };
            // SAFETY: the heap type is a valid descriptor heap type and the
            // device outlives the call.
            let stride = unsafe {
                self.device.GetDescriptorHandleIncrementSize(
                    windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                )
            };
            // SAFETY: the heap retains two slots for multisampled targets and
            // outlives the handle through the texture record.
            let mut msaa_rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
            msaa_rtv.ptr = msaa_rtv.ptr.wrapping_add(stride as usize);
            let msaa_desc = windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC {
                Format: dxgi,
                ViewDimension:
                    windows::Win32::Graphics::Direct3D12::D3D12_RTV_DIMENSION_TEXTURE2DMS,
                ..Default::default()
            };
            // SAFETY: `msaa_rtv` addresses the heap's second slot and `desc`
            // selects a matching multisampled view through the call.
            unsafe {
                self.device.CreateRenderTargetView(
                    &msaa_resource,
                    Some(&raw const msaa_desc),
                    msaa_rtv,
                );
            }
            Some(super::super::MsaaStorage {
                resource: msaa_resource,
                allocation: msaa_allocation,
                rtv: msaa_rtv,
                format: dxgi,
                samples,
            })
        };
        Ok(NativeTexture {
            resource,
            allocation,
            format: hal_format,
            sampler_desc: None,
            width,
            height,
            mip_count: 1,
            resident_mips: 1,
            mip_completions: vec![0],
            cancellation: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            binding,
            rtv: Some((rtv_heap, rtv)),
            msaa,
        })
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
        let support = support_for_target_format(Format::Rgba8Unorm, full, 4).unwrap();
        assert!(support.color && support.sampled && support.storage);
        assert_eq!(support.max_samples, 4);
    }

    #[test]
    fn ceiling_above_declaration_admits_multisample_resolution() {
        use ez_gfx_runtime::target::{ClearValue, TargetDeclaration, TargetUsage};
        // A ceiling of 4 admits 1/2/4-sample declarations and rejects 8-sample ones.
        let full = D3D12_FORMAT_SUPPORT1_RENDER_TARGET
            | D3D12_FORMAT_SUPPORT1_TEXTURE2D
            | D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
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
    fn depth_support_never_reports_color() {
        // Depth aspects carry no color role regardless of support bits.
        let depth = support_for_target_format(
            Format::Depth32Float,
            D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL | D3D12_FORMAT_SUPPORT1_TEXTURE2D,
            1,
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
            support_for_target_format(Format::Depth32Float, D3D12_FORMAT_SUPPORT1_TEXTURE2D, 1)
                .is_none()
        );
    }
}
