use ez_gfx_core::{
    Backend, ResourceAccess, ResourceKind, SemanticError, SemanticGraph, SemanticId,
    SemanticResource, TargetBinding, TargetLayout,
    capability::{
        AdapterCapabilities, AdapterClass, AdapterInfo, CapabilityError, CompressionSupport,
        SemanticProfile, select_default_adapter,
    },
};

fn resource(name: &str) -> SemanticResource {
    SemanticResource::new(name, ResourceKind::SampledTexture, ResourceAccess::Read, 1).unwrap()
}

#[test]
fn semantic_names_and_ids_are_stable_and_validated() {
    let first = SemanticId::from_name("material.albedo").unwrap();
    assert_eq!(first, SemanticId::from_name("material.albedo").unwrap());
    assert_ne!(first, SemanticId::from_name("material.normal").unwrap());

    for invalid in ["", ".leading", "trailing.", "two..dots", "white space", "é"] {
        assert!(matches!(
            SemanticId::from_name(invalid),
            Err(SemanticError::InvalidName)
        ));
    }
}

#[test]
fn semantic_graph_rejects_collisions_and_layout_drift() {
    let graph = SemanticGraph::new(vec![resource("material.albedo")]).unwrap();
    let id = graph.resources()[0].id();

    assert_eq!(
        SemanticGraph::new(vec![
            resource("material.albedo"),
            resource("material.albedo")
        ]),
        Err(SemanticError::DuplicateName)
    );

    let valid =
        TargetLayout::new(Backend::Vulkan, vec![TargetBinding::descriptor(id, 0, 4)]).unwrap();
    graph.validate_layout(&valid).unwrap();

    let missing = TargetLayout::new(Backend::Dx12, Vec::new()).unwrap();
    assert_eq!(
        graph.validate_layout(&missing),
        Err(SemanticError::MissingTargetBinding(id))
    );

    let duplicate = TargetLayout::new(
        Backend::Metal,
        vec![
            TargetBinding::argument(id, 1),
            TargetBinding::argument(id, 2),
        ],
    );
    assert_eq!(duplicate, Err(SemanticError::DuplicateTargetBinding(id)));
}

fn capabilities(compression: CompressionSupport) -> AdapterCapabilities {
    AdapterCapabilities {
        bindless_sampled_textures: 8192,
        bindless_storage_resources: 2048,
        bindless_samplers: 1024,
        max_indirect_draw_count: 65_535,
        shader_model: 0x0605,
        timeline_synchronization: true,
        resource_aliasing: true,
        dynamic_rendering: true,
        presentation: true,
        compression,
    }
}

#[test]
fn capability_floor_reports_every_missing_requirement() {
    let mut caps = capabilities(CompressionSupport::NONE);
    caps.bindless_sampled_textures = 2;
    caps.timeline_synchronization = false;
    caps.shader_model = 0x0604;

    let errors = SemanticProfile::V1.admit(&caps).unwrap_err();
    assert!(errors.contains(&CapabilityError::Limit {
        name: "bindless_sampled_textures",
        required: 1024,
        available: 2
    }));
    assert!(errors.contains(&CapabilityError::Limit {
        name: "shader_model",
        required: 0x0605,
        available: 0x0604
    }));
    assert!(errors.contains(&CapabilityError::MissingFeature("timeline_synchronization")));
    assert!(errors.contains(&CapabilityError::MissingCompression));
}

#[test]
fn sampled_texture_capacity_admits_exact_canonical_limit() {
    let mut caps = capabilities(CompressionSupport::BC);
    caps.bindless_sampled_textures = 1024;
    assert!(SemanticProfile::V1.admit(&caps).is_ok());

    caps.bindless_samplers = 1023;
    assert!(SemanticProfile::V1.admit(&caps).is_err());

    caps.bindless_sampled_textures = 1023;
    assert!(SemanticProfile::V1.admit(&caps).is_err());
}

#[test]
fn deterministic_default_rejects_software_unless_opted_in() {
    let software = AdapterInfo::new(
        Backend::Vulkan,
        [2; 16],
        "software",
        "driver",
        AdapterClass::Software,
        capabilities(CompressionSupport::BC),
    )
    .unwrap();
    let discrete = AdapterInfo::new(
        Backend::Dx12,
        [1; 16],
        "discrete",
        "driver",
        AdapterClass::Discrete,
        capabilities(CompressionSupport::BC),
    )
    .unwrap();
    let integrated = AdapterInfo::new(
        Backend::Metal,
        [3; 16],
        "integrated",
        "driver",
        AdapterClass::Integrated,
        capabilities(CompressionSupport::ASTC),
    )
    .unwrap();

    assert_eq!(
        select_default_adapter(
            &[software.clone(), integrated.clone(), discrete.clone()],
            false
        )
        .unwrap()
        .stable_id(),
        discrete.stable_id()
    );
    assert!(select_default_adapter(&[software], false).is_err());
}
