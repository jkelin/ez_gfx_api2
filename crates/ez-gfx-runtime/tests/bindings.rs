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
fn compute_reflection_requires_nonzero_workgroup_size() {
    let valid = br#"{"reflections":[{"target":"Metallib","entry":"computemain","stage":"Compute","reflection":{"parameters":[],"workgroup_size":[8,2,1]}}]}"#;
    let reflections = ez_gfx_runtime::binding::validate_stage_reflections(
        valid,
        Backend::Metal,
        &[(Stage::Compute, "computemain")],
    )
    .unwrap();
    assert_eq!(reflections[0].workgroup_size(), Some([8, 2, 1]));

    for invalid in [
        br#"{"reflections":[{"target":"Metallib","entry":"computemain","stage":"Compute","reflection":{"parameters":[]}}]}"#.as_slice(),
        br#"{"reflections":[{"target":"Metallib","entry":"computemain","stage":"Compute","reflection":{"parameters":[],"workgroup_size":[8,0,1]}}]}"#.as_slice(),
    ] {
        assert_eq!(
            ez_gfx_runtime::binding::validate_stage_reflections(
                invalid,
                Backend::Metal,
                &[(Stage::Compute, "computemain")],
            ),
            Err(BindingError::InvalidMetadata)
        );
    }
}
