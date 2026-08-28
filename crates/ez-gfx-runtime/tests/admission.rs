use ez_gfx_core::{
    Backend,
    capability::{AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport},
};
use ez_gfx_runtime::{AdapterCatalog, RuntimeError};

fn adapter(id: u8, class: AdapterClass, sampled: u32) -> AdapterInfo {
    AdapterInfo::new(
        Backend::Vulkan,
        [id; 16],
        "adapter",
        "driver",
        class,
        AdapterCapabilities {
            bindless_sampled_textures: sampled,
            bindless_storage_resources: 1024,
            bindless_samplers: 256,
            max_indirect_draw_count: 65_535,
            shader_model: 0x0605,
            timeline_synchronization: true,
            resource_aliasing: true,
            dynamic_rendering: true,
            presentation: true,
            compression: CompressionSupport::BC,
        },
    )
    .unwrap()
}

#[test]
fn catalog_requires_unique_stable_identity() {
    let duplicate = adapter(1, AdapterClass::Discrete, 4096);
    assert_eq!(
        AdapterCatalog::new(vec![duplicate.clone(), duplicate]),
        Err(RuntimeError::DuplicateAdapterIdentity)
    );
}

#[test]
fn explicit_and_default_selection_admit_before_device_creation() {
    let unsupported = adapter(1, AdapterClass::Discrete, 64);
    let supported = adapter(2, AdapterClass::Integrated, 4096);
    let catalog = AdapterCatalog::new(vec![unsupported.clone(), supported.clone()]).unwrap();

    assert_eq!(
        catalog.select(unsupported.stable_id(), false),
        Err(RuntimeError::UnsupportedAdapter)
    );
    assert_eq!(
        catalog.select_default(false).unwrap().info().stable_id(),
        supported.stable_id()
    );
    assert_eq!(
        catalog.select([9; 16], false),
        Err(RuntimeError::AdapterNotFound)
    );
}
