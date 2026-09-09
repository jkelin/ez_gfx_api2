use super::*;
use crate::device::{cached_device_supports_surface, device_extensions};

fn extension(name: &CStr) -> vk::ExtensionProperties {
    let mut property = vk::ExtensionProperties::default();
    // Test names are shorter than Vulkan's fixed extension-name storage and include the NUL.
    for (destination, source) in property
        .extension_name
        .iter_mut()
        .zip(name.to_bytes_with_nul())
    {
        *destination = i8::try_from(*source).unwrap();
    }
    property
}

fn enabled_has(enabled: &[*const core::ffi::c_char], name: &CStr) -> bool {
    enabled.iter().any(|pointer| {
        // SAFETY: extension policy returns pointers to static NUL-terminated Vulkan names.
        (unsafe { CStr::from_ptr(*pointer) }) == name
    })
}

#[test]
fn device_policy_enables_advertised_swapchain_and_portability_subset() {
    let available = [
        extension(khr::swapchain::NAME),
        extension(khr::portability_subset::NAME),
    ];

    let (enabled, swapchain) = device_extensions(&available);

    assert!(enabled_has(&enabled, khr::swapchain::NAME));
    assert!(enabled_has(&enabled, khr::portability_subset::NAME));
    assert!(swapchain);

    let (enabled, swapchain) = device_extensions(&[extension(khr::portability_subset::NAME)]);
    assert!(!enabled_has(&enabled, khr::swapchain::NAME));
    assert!(enabled_has(&enabled, khr::portability_subset::NAME));
    assert!(!swapchain);
}

#[test]
fn cached_headless_device_requires_enabled_swapchain_for_later_surface() {
    assert!(cached_device_supports_surface(true, true));
    assert!(cached_device_supports_surface(false, false));
    assert!(!cached_device_supports_surface(true, false));
}

#[test]
fn texture_heap_layout_matches_slang_bindless_contract() {
    let [texture, sampler] = texture_descriptor_layout_bindings();

    assert_eq!(TEXTURE_DESCRIPTOR_SET, 1);
    assert_eq!(texture.binding, TEXTURE_DESCRIPTOR_BINDING);
    assert_eq!(texture.descriptor_type, vk::DescriptorType::SAMPLED_IMAGE);
    assert_eq!(texture.descriptor_count, TEXTURE_DESCRIPTOR_CAPACITY);
    assert_eq!(texture.stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(sampler.binding, SAMPLER_DESCRIPTOR_BINDING);
    assert_eq!(sampler.descriptor_type, vk::DescriptorType::SAMPLER);
    assert_eq!(sampler.descriptor_count, TEXTURE_DESCRIPTOR_CAPACITY);
    assert_eq!(sampler.stage_flags, vk::ShaderStageFlags::ALL);
}

#[test]
fn paired_texture_capacity_honors_every_update_after_bind_limit() {
    let mut limits = vk::PhysicalDeviceDescriptorIndexingProperties {
        max_update_after_bind_descriptors_in_all_pools: 2048,
        max_per_stage_descriptor_update_after_bind_samplers: 1024,
        max_per_stage_descriptor_update_after_bind_sampled_images: 1024,
        max_per_stage_update_after_bind_resources: 2048,
        max_descriptor_set_update_after_bind_samplers: 1024,
        max_descriptor_set_update_after_bind_sampled_images: 1024,
        ..Default::default()
    };

    assert_eq!(paired_texture_capacity(&limits), 1024);
    limits.max_per_stage_update_after_bind_resources = 2046;
    assert_eq!(paired_texture_capacity(&limits), 1023);
}

#[test]
fn texture_heap_rejects_missing_consumed_features_in_diagnostic_order() {
    let mut features = vk::PhysicalDeviceVulkan12Features {
        descriptor_indexing: vk::FALSE,
        descriptor_binding_partially_bound: vk::TRUE,
        descriptor_binding_sampled_image_update_after_bind: vk::FALSE,
        shader_sampled_image_array_non_uniform_indexing: vk::FALSE,
        ..Default::default()
    };

    assert_eq!(
        texture_heap_rejection(&features),
        Some("descriptor_binding_sampled_image_update_after_bind")
    );
    features.descriptor_binding_sampled_image_update_after_bind = vk::TRUE;
    assert_eq!(
        texture_heap_rejection(&features),
        Some("shader_sampled_image_array_non_uniform_indexing")
    );
    features.shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
    assert_eq!(texture_heap_rejection(&features), None);
    features.descriptor_binding_partially_bound = vk::FALSE;
    assert_eq!(
        texture_heap_rejection(&features),
        Some("descriptor_binding_partially_bound")
    );
}

#[test]
fn resource_access_selects_compatible_pipeline_stages() {
    let cases = [
        (
            ResourceAccess::IndexRead,
            vk::PipelineStageFlags::VERTEX_INPUT,
        ),
        (
            ResourceAccess::IndirectRead,
            vk::PipelineStageFlags::DRAW_INDIRECT,
        ),
        (
            ResourceAccess::ColorAttachmentWrite,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (
            ResourceAccess::DepthStencilWrite,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        ),
        (
            ResourceAccess::TransferRead,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (
            ResourceAccess::Present,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        ),
    ];

    for (access, expected) in cases {
        let (stage, _, _) = vulkan_state(ResourceState {
            queue: QueueKind::Graphics,
            stage: ShaderStage::Fragment,
            access,
        });
        assert_eq!(stage, expected, "{access:?}");
    }
}

#[test]
fn sampler_state_preserves_filter_address_and_mip_configuration() {
    let info = sampler_create_info(
        TextureSamplerDesc {
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Nearest,
            max_anisotropy: 16.0,
            address_u: SamplerAddressMode::Repeat,
            address_v: SamplerAddressMode::Clamp,
            address_w: SamplerAddressMode::Repeat,
        },
        5,
    );

    assert_eq!(info.min_filter, vk::Filter::LINEAR);
    assert_eq!(info.mag_filter, vk::Filter::NEAREST);
    assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
    assert_eq!(info.address_mode_u, vk::SamplerAddressMode::REPEAT);
    assert_eq!(info.address_mode_v, vk::SamplerAddressMode::CLAMP_TO_EDGE);
    assert_eq!(info.address_mode_w, vk::SamplerAddressMode::REPEAT);
    assert_eq!(info.anisotropy_enable, vk::TRUE);
    assert!((info.max_anisotropy - 16.0).abs() < f32::EPSILON);
    assert!((info.max_lod - 5.0).abs() < f32::EPSILON);
}
