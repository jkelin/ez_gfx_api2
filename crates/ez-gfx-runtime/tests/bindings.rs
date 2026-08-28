use ez_gfx_artifact::Stage;
use ez_gfx_core::Backend;
use ez_gfx_runtime::binding::{
    BindingError, BindingKind, PublicBinding, ReflectedBindings, ResourceIdentity,
};

const METADATA: &[u8] = br#"{
  "reflections": [
    {"target":"Spirv","entry":"main","stage":"Compute","reflection":{"parameters":[
      {"semantic_name":"instances","api_kind":"structured","binding_index":2,"binding_space":0},
      {"semantic_name":"draws","api_kind":"indirect","binding_index":3,"binding_space":0}
    ]}},
    {"target":"Dxil","entry":"main","stage":"Compute","reflection":{"parameters":[
      {"semantic_name":"instances","api_kind":"structured","binding_index":5,"binding_space":1},
      {"semantic_name":"draws","api_kind":"indirect","binding_index":6,"binding_space":1}
    ]}}
  ]
}"#;

#[test]
fn metadata_selects_exact_backend_entry_and_stage() {
    let bindings =
        ReflectedBindings::parse(METADATA, Backend::Dx12, "main", Stage::Compute).unwrap();
    let instance = bindings
        .requirements()
        .iter()
        .find(|requirement| requirement.name == "instances")
        .unwrap();
    assert_eq!(instance.kind, BindingKind::Structured);
    assert_eq!((instance.space, instance.binding), (1, 5));
}

#[test]
fn public_bindings_are_exact_unique_and_kind_checked() {
    let bindings =
        ReflectedBindings::parse(METADATA, Backend::Vulkan, "main", Stage::Compute).unwrap();
    let valid = [
        PublicBinding {
            name: "draws".into(),
            resource: ResourceIdentity::Indirect(9),
        },
        PublicBinding {
            name: "instances".into(),
            resource: ResourceIdentity::Structured(7),
        },
    ];
    assert!(bindings.validate(&valid).is_ok());
    assert_eq!(
        bindings.validate(&valid[..1]),
        Err(BindingError::Missing("instances".into()))
    );
    assert_eq!(
        bindings.validate(&[valid[0].clone(), valid[0].clone(), valid[1].clone()]),
        Err(BindingError::Duplicate("draws".into()))
    );
    assert_eq!(
        bindings.validate(&[
            PublicBinding {
                name: "draws".into(),
                resource: ResourceIdentity::Structured(9)
            },
            valid[1].clone()
        ]),
        Err(BindingError::KindMismatch("draws".into()))
    );
    assert_eq!(
        bindings.validate(&[
            valid[0].clone(),
            valid[1].clone(),
            PublicBinding {
                name: "extra".into(),
                resource: ResourceIdentity::Structured(1)
            }
        ]),
        Err(BindingError::Unknown("extra".into()))
    );
}

#[test]
fn malformed_or_ambiguous_metadata_fails_closed() {
    assert_eq!(
        ReflectedBindings::parse(b"{}", Backend::Vulkan, "main", Stage::Compute),
        Err(BindingError::MissingReflection)
    );
    assert!(matches!(
        ReflectedBindings::parse(b"not-json", Backend::Vulkan, "main", Stage::Compute),
        Err(BindingError::InvalidMetadata)
    ));
    let duplicate = br#"{"reflections":[{"target":"Spirv","entry":"main","stage":"Compute","reflection":{"parameters":[{"semantic_name":"x","api_kind":"structured","binding_index":0,"binding_space":0},{"semantic_name":"x","api_kind":"structured","binding_index":1,"binding_space":0}]}}]}"#;
    assert_eq!(
        ReflectedBindings::parse(duplicate, Backend::Vulkan, "main", Stage::Compute),
        Err(BindingError::Duplicate("x".into()))
    );
}

#[test]
fn dxil_register_namespaces_allow_srv_and_uav_at_the_same_index() {
    let dxil = br#"{"reflections":[{"target":"Dxil","entry":"main","stage":"Compute","reflection":{"parameters":[
        {"semantic_name":"input","api_kind":"structured","binding_index":0,"binding_space":0,"descriptor_count":1,"resource_access":"Read"},
        {"semantic_name":"output","api_kind":"structured","binding_index":0,"binding_space":0,"descriptor_count":1,"resource_access":"ReadWrite"}
    ]}}]}"#;
    let bindings = ReflectedBindings::parse(dxil, Backend::Dx12, "main", Stage::Compute).unwrap();
    assert_eq!(bindings.requirements().len(), 2);

    let spirv = std::str::from_utf8(dxil)
        .unwrap()
        .replace("\"Dxil\"", "\"Spirv\"");
    assert_eq!(
        ReflectedBindings::parse(spirv.as_bytes(), Backend::Vulkan, "main", Stage::Compute),
        Err(BindingError::DuplicatePhysicalSlot {
            space: 0,
            binding: 0
        }),
    );
}
