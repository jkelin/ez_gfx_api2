use super::*;
use std::collections::BTreeSet;

fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
    vk::QueueFamilyProperties {
        queue_flags: flags,
        queue_count: count,
        ..Default::default()
    }
}

#[test]
fn draw_admission_requires_indirect_count_and_core_draw_features() {
    let supported = vk::PhysicalDeviceVulkan12Features {
        draw_indirect_count: vk::TRUE,
        ..Default::default()
    };
    assert_eq!(draw_feature_rejection(true, true, &supported), None);
    assert_eq!(
        draw_feature_rejection(false, true, &supported),
        Some("vertex_pipeline_stores_and_atomics")
    );
    assert_eq!(
        draw_feature_rejection(true, false, &supported),
        Some("multi_draw_indirect")
    );

    let missing_count = vk::PhysicalDeviceVulkan12Features::default();
    assert_eq!(
        draw_feature_rejection(true, true, &missing_count),
        Some("draw_indirect_count")
    );
}

#[test]
fn transfer_family_prefers_non_graphics_hardware_queue() {
    let families = [
        family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER, 2),
        family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 1),
    ];

    assert_eq!(select_transfer_family(&families, 0), 1);
}

#[test]
fn transfer_family_falls_back_when_specialized_queue_is_unavailable() {
    let families = [
        family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER, 1),
        family(vk::QueueFlags::TRANSFER, 0),
    ];

    assert_eq!(select_transfer_family(&families, 0), 0);
}

#[test]
fn enumeration_reports_unique_named_adapters() {
    // No surface is created, shown, or activated by this test.
    let adapters = NativeContext::enumerate_adapters().expect("Vulkan enumerates adapters");
    assert!(!adapters.is_empty());
    let mut identities = BTreeSet::new();
    for adapter in &adapters {
        assert_ne!(adapter.stable_id(), [0; 16]);
        assert!(!adapter.name().is_empty());
        assert!(!adapter.driver().is_empty());
        assert!(identities.insert(adapter.stable_id()));
    }
}

#[test]
fn explicit_selection_rejects_unknown_identity() {
    // No surface is created, shown, or activated by this test.
    let mut context = NativeContext::create(false, false).expect("Vulkan instance");
    assert_eq!(
        context.init_device_for_adapter(None, [0xA5; 16], false),
        Err(HalError::InvalidArgument)
    );
}

#[test]
fn explicit_selection_admits_enumerated_adapter() {
    // No surface is created, shown, or activated by this test.
    let adapters = NativeContext::enumerate_adapters().expect("Vulkan enumerates adapters");
    let wanted = adapters.first().expect("at least one adapter").stable_id();
    let mut context = NativeContext::create(false, false).expect("Vulkan instance");
    let admitted = context
        .init_device_for_adapter(None, wanted, true)
        .expect("enumerated adapter initializes");
    assert_eq!(admitted.stable_id(), wanted);
}
