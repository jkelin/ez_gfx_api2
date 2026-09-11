//! Runtime integration and contract tests.

use ez_gfx_artifact::Stage;
use ez_gfx_core::{
    Backend,
    handle::{BufferHandle, CounterBufferHandle, LocalHandle, PackedHandle},
};
use ez_gfx_runtime::binding::{
    BindingError, BindingKind, PublicBinding, ReflectedBindings, ResourceIdentity,
};

const METADATA: &[u8] = br#"{
  "reflections": [
    {"target":"Spirv","entry":"main","stage":"Compute","reflection":{"parameters":[
      {"semantic_name":"instances","api_kind":"buffer","binding_index":2,"binding_space":0},
      {"semantic_name":"draws","api_kind":"counter_buffer","binding_index":3,"binding_space":0,"descriptor_count":2}
    ]}},
    {"target":"Dxil","entry":"main","stage":"Compute","reflection":{"parameters":[
      {"semantic_name":"instances","api_kind":"buffer","binding_index":5,"binding_space":1},
      {"semantic_name":"draws","api_kind":"counter_buffer","binding_index":6,"binding_space":1,"descriptor_count":2}
    ]}}
  ]
}"#;

fn structured(slot: u32) -> BufferHandle {
    BufferHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(slot, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn indirect(slot: u32) -> CounterBufferHandle {
    CounterBufferHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(slot, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn metadata_selects_exact_backend_entry_and_stage() {
    let bindings =
        ReflectedBindings::parse(METADATA, Backend::Dx12, "main", Stage::Compute).unwrap();
    let instance = bindings
        .requirements()
        .iter()
        .find(|requirement| requirement.name == "instances")
        .unwrap();
    assert_eq!(instance.kind, BindingKind::Buffer);
    assert_eq!((instance.space, instance.binding), (1, 5));
}

#[test]
fn public_bindings_are_exact_unique_and_kind_checked() {
    let bindings =
        ReflectedBindings::parse(METADATA, Backend::Vulkan, "main", Stage::Compute).unwrap();
    let valid = [
        PublicBinding {
            name: "draws".into(),
            resource: ResourceIdentity::Counter(indirect(9)),
        },
        PublicBinding {
            name: "instances".into(),
            resource: ResourceIdentity::Buffer(structured(7)),
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
                resource: ResourceIdentity::Buffer(structured(9))
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
                resource: ResourceIdentity::Buffer(structured(1))
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
    let duplicate = br#"{"reflections":[{"target":"Spirv","entry":"main","stage":"Compute","reflection":{"parameters":[{"semantic_name":"x","api_kind":"buffer","binding_index":0,"binding_space":0},{"semantic_name":"x","api_kind":"buffer","binding_index":1,"binding_space":0}]}}]}"#;
    assert_eq!(
        ReflectedBindings::parse(duplicate, Backend::Vulkan, "main", Stage::Compute),
        Err(BindingError::Duplicate("x".into()))
    );
}

#[test]
fn dxil_register_namespaces_allow_srv_and_uav_at_the_same_index() {
    let dxil = br#"{"reflections":[{"target":"Dxil","entry":"main","stage":"Compute","reflection":{"parameters":[
        {"semantic_name":"input","api_kind":"buffer","binding_index":0,"binding_space":0,"descriptor_count":1,"resource_access":"Read"},
        {"semantic_name":"output","api_kind":"buffer","binding_index":0,"binding_space":0,"descriptor_count":1,"resource_access":"ReadWrite"}
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

#[test]
fn merge_rejects_counter_descriptor_tail_overlap() {
    let metadata = br#"{"reflections":[
        {"target":"Spirv","entry":"vertexmain","stage":"Vertex","reflection":{"parameters":[
            {"semantic_name":"draws","api_kind":"counter_buffer","binding_index":0,"binding_space":0,"descriptor_count":2,"resource_access":"ReadWrite"}
        ]}},
        {"target":"Spirv","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[
            {"semantic_name":"values","api_kind":"buffer","binding_index":1,"binding_space":0,"descriptor_count":1,"resource_access":"Read"}
        ]}}
    ]}"#;
    let vertex =
        ReflectedBindings::parse(metadata, Backend::Vulkan, "vertexmain", Stage::Vertex).unwrap();
    let fragment =
        ReflectedBindings::parse(metadata, Backend::Vulkan, "fragmentmain", Stage::Fragment)
            .unwrap();

    assert_eq!(
        vertex.merge(&fragment),
        Err(BindingError::DuplicatePhysicalSlot {
            space: 0,
            binding: 1,
        })
    );
}

#[test]
fn pipeline_layout_reads_canonical_texture_heap_and_depth_contract() {
    let metadata = br#"{"reflections":[{"target":"Metallib","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[],"texture_heap":{"binding_space":1,"binding_index":6,"capacity":1024,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1},"depth_required":true}}]}"#;
    let layout = ez_gfx_runtime::binding::PipelineLayout::parse(
        metadata,
        Backend::Metal,
        "fragmentmain",
        Stage::Fragment,
    )
    .unwrap();
    assert!(layout.depth_required());
    assert_eq!(
        layout.texture_heap(),
        Some(&ez_gfx_runtime::binding::TextureHeapLayout {
            space: 1,
            binding: 6,
            capacity: 1024,
            argument_stride: 2,
            texture_argument_offset: 0,
            sampler_argument_offset: 1,
        })
    );

    let oversized = br#"{"reflections":[{"target":"Metallib","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[],"texture_heap":{"binding_space":1,"binding_index":6,"capacity":1025,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1}}}]}"#;
    assert_eq!(
        ez_gfx_runtime::binding::PipelineLayout::parse(
            oversized,
            Backend::Metal,
            "fragmentmain",
            Stage::Fragment,
        ),
        Err(BindingError::InvalidMetadata)
    );
}

#[test]
fn graphics_stage_layouts_merge_only_identical_texture_heaps() {
    let matching = br#"{"reflections":[
        {"target":"Metallib","entry":"vertexmain","stage":"Vertex","reflection":{"parameters":[],"texture_heap":{"binding_space":1,"binding_index":6,"capacity":1024,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1}}},
        {"target":"Metallib","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[],"texture_heap":{"binding_space":1,"binding_index":6,"capacity":1024,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1},"depth_required":true}}
    ]}"#;
    let reflections = ez_gfx_runtime::binding::validate_stage_reflections(
        matching,
        Backend::Metal,
        &[
            (Stage::Vertex, "vertexmain"),
            (Stage::Fragment, "fragmentmain"),
        ],
    )
    .unwrap();
    let merged = reflections[0]
        .pipeline_layout()
        .merge(reflections[1].pipeline_layout())
        .unwrap();
    assert_eq!(
        merged.texture_heap(),
        reflections[0].pipeline_layout().texture_heap()
    );
    assert!(merged.depth_required());

    let conflicting = std::str::from_utf8(matching).unwrap().replacen(
        "\"binding_index\":6",
        "\"binding_index\":5",
        1,
    );
    assert_eq!(
        ez_gfx_runtime::binding::validate_stage_reflections(
            conflicting.as_bytes(),
            Backend::Metal,
            &[
                (Stage::Vertex, "vertexmain"),
                (Stage::Fragment, "fragmentmain"),
            ],
        ),
        Err(BindingError::ConflictingStageLayout)
    );
}

#[test]
fn dispatch_stage_reflections_require_nonzero_workgroup_size() {
    for stage in [Stage::Compute, Stage::Task, Stage::Mesh] {
        let valid = format!(
            r#"{{"reflections":[{{"target":"Metallib","entry":"main","stage":"{stage:?}","reflection":{{"parameters":[],"workgroup_size":[8,2,1]}}}}]}}"#
        );
        let reflections = ez_gfx_runtime::binding::validate_stage_reflections(
            valid.as_bytes(),
            Backend::Metal,
            &[(stage, "main")],
        )
        .unwrap();
        assert_eq!(reflections[0].workgroup_size(), Some([8, 2, 1]));

        for invalid_workgroup in ["null", "[8,0,1]"] {
            let invalid = format!(
                r#"{{"reflections":[{{"target":"Metallib","entry":"main","stage":"{stage:?}","reflection":{{"parameters":[],"workgroup_size":{invalid_workgroup}}}}}]}}"#
            );
            assert_eq!(
                ez_gfx_runtime::binding::validate_stage_reflections(
                    invalid.as_bytes(),
                    Backend::Metal,
                    &[(stage, "main")],
                ),
                Err(BindingError::InvalidMetadata)
            );
        }
    }

    let vertex_with_workgroup = br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Vertex","reflection":{"parameters":[],"workgroup_size":[1,1,1]}}]}"#;
    assert_eq!(
        ez_gfx_runtime::binding::validate_stage_reflections(
            vertex_with_workgroup,
            Backend::Metal,
            &[(Stage::Vertex, "main")],
        ),
        Err(BindingError::InvalidMetadata)
    );
}

#[test]
fn mesh_stage_fold_validates_order_compatibility_and_retains_stage_layouts() {
    let metadata = br#"{"reflections":[
        {"target":"Metallib","entry":"taskmain","stage":"Task","reflection":{"parameters":[{"semantic_name":"draws","api_kind":"buffer","binding_index":0,"binding_space":0}],"workgroup_size":[1,1,1],"texture_heap":{"binding_space":1,"binding_index":6,"capacity":1024,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1}}},
        {"target":"Metallib","entry":"meshmain","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"draws","api_kind":"buffer","binding_index":0,"binding_space":0}],"workgroup_size":[8,1,1]}},
        {"target":"Metallib","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[],"texture_heap":{"binding_space":1,"binding_index":6,"capacity":1024,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1}}}
    ]}"#;
    let stages = [
        (Stage::Task, "taskmain"),
        (Stage::Mesh, "meshmain"),
        (Stage::Fragment, "fragmentmain"),
    ];
    let reflections =
        ez_gfx_runtime::binding::validate_stage_reflections(metadata, Backend::Metal, &stages)
            .unwrap();
    assert_eq!(reflections.len(), 3);
    assert!(!reflections.spilled());
    assert!(reflections[0].pipeline_layout().texture_heap().is_some());
    assert!(reflections[1].pipeline_layout().texture_heap().is_none());
    assert!(reflections[2].pipeline_layout().texture_heap().is_some());

    assert_eq!(
        ez_gfx_runtime::binding::validate_stage_reflections(
            metadata,
            Backend::Metal,
            &[
                (Stage::Mesh, "meshmain"),
                (Stage::Task, "taskmain"),
                (Stage::Fragment, "fragmentmain"),
            ],
        ),
        Err(BindingError::InvalidMetadata)
    );

    let conflicting = std::str::from_utf8(metadata).unwrap().replacen(
        "\"binding_index\":0",
        "\"binding_index\":2",
        1,
    );
    assert_eq!(
        ez_gfx_runtime::binding::validate_stage_reflections(
            conflicting.as_bytes(),
            Backend::Metal,
            &stages,
        ),
        Err(BindingError::ConflictingStageBinding("draws".into()))
    );

    let ambiguous = br#"{"reflections":[
        {"target":"Metallib","entry":"meshmain","stage":"Mesh","reflection":{"parameters":[],"workgroup_size":[8,1,1]}},
        {"target":"Metallib","entry":"meshmain","stage":"Mesh","reflection":{"parameters":[],"workgroup_size":[8,1,1]}}
    ]}"#;
    assert_eq!(
        ez_gfx_runtime::binding::validate_stage_reflections(
            ambiguous,
            Backend::Metal,
            &[(Stage::Mesh, "meshmain")],
        ),
        Err(BindingError::AmbiguousReflection)
    );
}

#[test]
fn stage_layout_identity_canonicalizes_adjacent_descriptor_partitions() {
    let identity = |metadata: &[u8], stage| {
        ez_gfx_runtime::binding::validate_stage_reflections(
            metadata,
            Backend::Metal,
            &[(stage, "main")],
        )
        .unwrap()[0]
            .physical_layout_identity()
    };
    let ranged = br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"all","api_kind":"buffer","binding_index":3,"binding_space":1,"descriptor_count":2,"resource_access":"Read"}],"workgroup_size":[1,1,1]}}]}"#;
    let split = br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"later","api_kind":"buffer","binding_index":4,"binding_space":1,"resource_access":"Read"},{"semantic_name":"earlier","api_kind":"buffer","binding_index":3,"binding_space":1,"resource_access":"Read"}],"workgroup_size":[1,1,1]}}]}"#;

    assert_eq!(identity(ranged, Stage::Mesh), identity(split, Stage::Mesh));
}

#[test]
fn stage_layout_identity_distinguishes_consumed_descriptor_shape() {
    let identity = |metadata: &[u8], stage| {
        ez_gfx_runtime::binding::validate_stage_reflections(
            metadata,
            Backend::Metal,
            &[(stage, "main")],
        )
        .unwrap()[0]
            .physical_layout_identity()
    };
    let base = br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"all","api_kind":"buffer","binding_index":3,"binding_space":1,"descriptor_count":2,"resource_access":"Read"}],"workgroup_size":[1,1,1]}}]}"#;
    let changed = [
        br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"all","api_kind":"buffer","binding_index":3,"binding_space":1,"descriptor_count":2,"resource_access":"ReadWrite"}],"workgroup_size":[1,1,1]}}]}"#.as_slice(),
        br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"all","api_kind":"buffer","binding_index":3,"binding_space":2,"descriptor_count":2,"resource_access":"Read"}],"workgroup_size":[1,1,1]}}]}"#,
        br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"first","api_kind":"buffer","binding_index":3,"binding_space":1,"resource_access":"Read"},{"semantic_name":"second","api_kind":"buffer","binding_index":5,"binding_space":1,"resource_access":"Read"}],"workgroup_size":[1,1,1]}}]}"#,
        br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"all","api_kind":"buffer","binding_index":3,"binding_space":1,"descriptor_count":2,"resource_access":"Read"}],"workgroup_size":[1,1,1],"texture_heap":{"binding_space":2,"binding_index":8,"capacity":16,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1}}}]}"#,
    ];
    let base = identity(base, Stage::Mesh);

    for metadata in changed {
        assert_ne!(base, identity(metadata, Stage::Mesh));
    }

    let fragment = br#"{"reflections":[{"target":"Metallib","entry":"main","stage":"Fragment","reflection":{"parameters":[{"semantic_name":"all","api_kind":"buffer","binding_index":3,"binding_space":1,"descriptor_count":2,"resource_access":"Read"}]}}]}"#;
    assert_ne!(base, identity(fragment, Stage::Fragment));
}
