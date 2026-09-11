//! Runtime integration and contract tests.

use ez_gfx_artifact::{
    AppleArchitecture, ApplePlatform, Artifact, CompatibilityVersion, MetalCompatibility,
    Provenance, Stage, Target, TargetCompatibility, TargetVariant,
};
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_runtime::shader::{MetalEnvironment, RuntimeShader, ShaderLoadError};

fn compatibility(target: Target) -> TargetCompatibility {
    match target {
        Target::Metallib => TargetCompatibility::MetalLibrary {
            metal: MetalCompatibility {
                platform: ApplePlatform::MacOs,
                architecture: AppleArchitecture::X86_64,
                minimum_os: CompatibilityVersion::new(14, 0),
                sdk: CompatibilityVersion::new(15, 0),
                language: CompatibilityVersion::new(3, 0),
                library: CompatibilityVersion::new(1, 0),
                toolchain: "apple-clang-16".into(),
            },
        },
        _ => TargetCompatibility::portable(target).unwrap(),
    }
}

const METAL_ENVIRONMENT: MetalEnvironment = MetalEnvironment {
    platform: ApplePlatform::MacOs,
    architecture: AppleArchitecture::X86_64,
    os: CompatibilityVersion::new(15, 0),
    max_language: CompatibilityVersion::new(3, 0),
    max_library: CompatibilityVersion::new(1, 0),
};

fn load(bytes: &[u8], backend: Backend) -> Result<RuntimeShader, ShaderLoadError> {
    RuntimeShader::load_for_environment(
        bytes,
        backend,
        SemanticProfile::V1,
        Stage::Compute,
        "main",
        (backend == Backend::Metal).then_some(&METAL_ENVIRONMENT),
    )
}

fn metadata(stages: &[Stage]) -> Vec<u8> {
    let reflections = [Target::Spirv, Target::Dxil, Target::Metallib]
        .into_iter()
        .flat_map(|target| {
            stages.iter().map(move |stage| {
                let workgroup_size = matches!(stage, Stage::Compute | Stage::Task | Stage::Mesh)
                    .then_some([1, 1, 1]);
                serde_json::json!({
                    "target": format!("{target:?}"),
                    "entry": format!("{stage:?}").replace("Compute", "main"),
                    "stage": format!("{stage:?}"),
                    "reflection": {
                        "parameters":[],
                        "workgroup_size": workgroup_size
                    }
                })
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&serde_json::json!({
        "semantic_abi": 1,
        "reflections": reflections
    }))
    .unwrap()
}

fn artifact(metal: Target) -> Vec<u8> {
    let variants = [
        (Target::Spirv, b"spirv".to_vec()),
        (Target::Dxil, b"dxil".to_vec()),
        (metal, b"metal".to_vec()),
    ]
    .into_iter()
    .map(|(target, bytes)| {
        TargetVariant::new(
            target,
            Stage::Compute,
            "main",
            "ez-gfx-v1",
            compatibility(target),
            bytes,
        )
        .unwrap()
    })
    .collect();
    Artifact::new(
        metadata(&[Stage::Compute]),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap()
}
#[test]
fn partial_artifact_loads_each_requested_backend_and_rejects_missing_backends() {
    for (target, available_backend) in [
        (Target::Spirv, Backend::Vulkan),
        (Target::Dxil, Backend::Dx12),
        (Target::Metallib, Backend::Metal),
    ] {
        let bytes = Artifact::new(
            metadata(&[Stage::Compute]),
            Provenance::new("slangc", "2026.16", vec![], "host"),
            vec![
                TargetVariant::new(
                    target,
                    Stage::Compute,
                    "main",
                    "ez-gfx-v1",
                    compatibility(target),
                    vec![target as u8],
                )
                .unwrap(),
            ],
        )
        .unwrap()
        .encode()
        .unwrap();

        assert!(Artifact::decode(&bytes).is_ok());
        assert!(load(&bytes, available_backend).is_ok());
        for missing_backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal]
            .into_iter()
            .filter(|backend| *backend != available_backend)
        {
            assert_eq!(
                load(&bytes, missing_backend),
                Err(ShaderLoadError::MissingProduct {
                    stage: Stage::Compute
                })
            );
        }
    }
}

fn artifact_with_metadata(metadata: Vec<u8>) -> Vec<u8> {
    let variants = [Target::Spirv, Target::Dxil, Target::Metallib]
        .into_iter()
        .map(|target| {
            TargetVariant::new(
                target,
                Stage::Compute,
                "main",
                "ez-gfx-v1",
                compatibility(target),
                vec![target as u8],
            )
            .unwrap()
        })
        .collect();
    Artifact::new(
        metadata,
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap()
}

#[test]
fn every_backend_rejects_reflection_before_product_exposure() {
    for (backend, target) in [
        (Backend::Vulkan, "Spirv"),
        (Backend::Dx12, "Dxil"),
        (Backend::Metal, "Metallib"),
    ] {
        assert_eq!(
            load(
                &artifact_with_metadata(br#"{"reflections":[]}"#.to_vec()),
                backend,
            ),
            Err(ShaderLoadError::Reflection(
                ez_gfx_runtime::binding::BindingError::MissingReflection
            ))
        );

        let malformed = serde_json::to_vec(&serde_json::json!({"reflections":[{
            "target": target,
            "entry": "main",
            "stage": "Compute",
            "reflection": {
                "parameters":[{
                    "semantic_name":"resource",
                    "api_kind":"unsupported",
                    "binding_index":0,
                    "binding_space":0
                }],
                "workgroup_size":[1,1,1]
            }
        }]}))
        .unwrap();
        assert_eq!(
            load(&artifact_with_metadata(malformed), backend),
            Err(ShaderLoadError::Reflection(
                ez_gfx_runtime::binding::BindingError::InvalidMetadata
            ))
        );

        let invalid_heap = serde_json::to_vec(&serde_json::json!({"reflections":[{
            "target": target,
            "entry": "main",
            "stage": "Compute",
            "reflection": {
                "parameters":[],
                "texture_heap":{
                    "binding_space":0,
                    "binding_index":0,
                    "capacity":0,
                    "argument_stride":2,
                    "texture_argument_offset":0,
                    "sampler_argument_offset":1
                }
                ,"workgroup_size":[1,1,1]
            }
        }]}))
        .unwrap();
        assert_eq!(
            load(&artifact_with_metadata(invalid_heap), backend),
            Err(ShaderLoadError::Reflection(
                ez_gfx_runtime::binding::BindingError::InvalidMetadata
            ))
        );
    }
}

#[test]
fn metal_selection_is_compatible_and_deterministic() {
    let metal = |minimum_os, bytes| {
        TargetVariant::new(
            Target::Metallib,
            Stage::Compute,
            "main",
            "ez-gfx-v1",
            TargetCompatibility::MetalLibrary {
                metal: MetalCompatibility {
                    platform: ApplePlatform::MacOs,
                    architecture: AppleArchitecture::X86_64,
                    minimum_os,
                    sdk: CompatibilityVersion::new(15, 0),
                    language: CompatibilityVersion::new(3, 0),
                    library: CompatibilityVersion::new(1, 0),
                    toolchain: "apple-clang-16".into(),
                },
            },
            vec![bytes],
        )
        .unwrap()
    };
    let bytes = Artifact::new(
        metadata(&[Stage::Compute]),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        vec![
            TargetVariant::new(
                Target::Spirv,
                Stage::Compute,
                "main",
                "ez-gfx-v1",
                compatibility(Target::Spirv),
                vec![1],
            )
            .unwrap(),
            TargetVariant::new(
                Target::Dxil,
                Stage::Compute,
                "main",
                "ez-gfx-v1",
                compatibility(Target::Dxil),
                vec![2],
            )
            .unwrap(),
            metal(CompatibilityVersion::new(14, 0), 3),
            metal(CompatibilityVersion::new(15, 0), 4),
        ],
    )
    .unwrap()
    .encode()
    .unwrap();

    let selected = load(&bytes, Backend::Metal).unwrap();
    assert_eq!(selected.product(Stage::Compute), Some([4].as_slice()));

    let incompatible = MetalEnvironment {
        architecture: AppleArchitecture::Aarch64,
        ..METAL_ENVIRONMENT
    };
    assert_eq!(
        RuntimeShader::load_for_environment(
            &bytes,
            Backend::Metal,
            SemanticProfile::V1,
            Stage::Compute,
            "main",
            Some(&incompatible),
        ),
        Err(ShaderLoadError::MissingProduct {
            stage: Stage::Compute
        })
    );
}

#[test]
fn selects_backend_profile_and_every_available_stage() {
    let shader = RuntimeShader::load(
        &artifact(Target::Metallib),
        Backend::Dx12,
        SemanticProfile::V1,
        Stage::Compute,
        "main",
    )
    .unwrap();
    assert_eq!(shader.product(Stage::Compute).unwrap(), b"dxil");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(shader.metadata()).unwrap()["semantic_abi"],
        1
    );
}

#[test]
fn metal_runtime_requires_offline_metallib_and_never_uses_msl() {
    assert_eq!(
        RuntimeShader::load(
            &artifact(Target::Msl),
            Backend::Metal,
            SemanticProfile::V1,
            Stage::Compute,
            "main",
        ),
        Err(ShaderLoadError::MissingProduct {
            stage: Stage::Compute
        })
    );
}

#[test]
fn malformed_and_missing_backend_coverage_fail_without_fallback() {
    assert!(matches!(
        RuntimeShader::load(
            b"bad",
            Backend::Vulkan,
            SemanticProfile::V1,
            Stage::Compute,
            "main",
        ),
        Err(ShaderLoadError::Artifact(_))
    ));

    let variants = [Target::Spirv, Target::Dxil, Target::Metallib]
        .into_iter()
        .map(|target| {
            TargetVariant::new(
                target,
                Stage::Compute,
                "main",
                "other-profile",
                compatibility(target),
                vec![target as u8],
            )
            .unwrap()
        })
        .collect();
    let bytes = Artifact::new(
        metadata(&[Stage::Compute]),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap();
    assert_eq!(
        RuntimeShader::load(
            &bytes,
            Backend::Vulkan,
            SemanticProfile::V1,
            Stage::Compute,
            "main",
        ),
        Err(ShaderLoadError::MissingProduct {
            stage: Stage::Compute
        })
    );
}

#[test]
fn named_entry_selection_is_exact_and_stage_typed() {
    let entries = [
        (Stage::Compute, "first", 1_u8),
        (Stage::Compute, "second", 2_u8),
        (Stage::Vertex, "vertexmain", 3_u8),
    ];
    let variants = entries
        .into_iter()
        .map(|(stage, entry, byte)| {
            TargetVariant::new(
                Target::Spirv,
                stage,
                entry,
                "ez-gfx-v1",
                compatibility(Target::Spirv),
                vec![byte],
            )
            .unwrap()
        })
        .collect();
    let reflections = entries.map(|(stage, entry, _)| {
        serde_json::json!({
            "target": "Spirv",
            "entry": entry,
            "stage": format!("{stage:?}"),
            "reflection": {
                "parameters": [],
                "workgroup_size": (stage == Stage::Compute).then_some([1, 1, 1])
            }
        })
    });
    let bytes = Artifact::new(
        serde_json::to_vec(&serde_json::json!({
            "semantic_abi": 1,
            "reflections": reflections
        }))
        .unwrap(),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap();

    for (entry, byte) in [("first", 1_u8), ("second", 2_u8)] {
        let shader = RuntimeShader::load(
            &bytes,
            Backend::Vulkan,
            SemanticProfile::V1,
            Stage::Compute,
            entry,
        )
        .unwrap();
        assert_eq!(shader.product(Stage::Compute), Some([byte].as_slice()));
        assert_eq!(shader.shader_product().2, entry);
    }
    assert_eq!(
        RuntimeShader::load(
            &bytes,
            Backend::Vulkan,
            SemanticProfile::V1,
            Stage::Compute,
            "missing",
        ),
        Err(ShaderLoadError::UnknownEntryPoint)
    );
    assert_eq!(
        RuntimeShader::load(
            &bytes,
            Backend::Vulkan,
            SemanticProfile::V1,
            Stage::Fragment,
            "vertexmain",
        ),
        Err(ShaderLoadError::WrongStage)
    );
}

#[test]
fn task_and_mesh_products_expose_validated_workgroup_sizes() {
    let stages = [
        (Stage::Task, "taskmain", 1_u8),
        (Stage::Mesh, "meshmain", 2),
    ];
    let variants = stages
        .into_iter()
        .map(|(stage, entry, byte)| {
            TargetVariant::new(
                Target::Spirv,
                stage,
                entry,
                "ez-gfx-v1",
                compatibility(Target::Spirv),
                vec![byte],
            )
            .unwrap()
        })
        .collect();
    let reflections = stages.map(|(stage, entry, _)| {
        serde_json::json!({
            "target": "Spirv",
            "entry": entry,
            "stage": format!("{stage:?}"),
            "reflection": {"parameters": [], "workgroup_size": [4, 2, 1]}
        })
    });
    let bytes = Artifact::new(
        serde_json::to_vec(&serde_json::json!({"reflections": reflections})).unwrap(),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap();

    for (stage, entry, byte) in stages {
        let shader =
            RuntimeShader::load(&bytes, Backend::Vulkan, SemanticProfile::V1, stage, entry)
                .unwrap();
        assert_eq!(shader.product(stage), Some([byte].as_slice()));
        assert_eq!(shader.workgroup_size().unwrap(), [4, 2, 1]);
    }
    for (requested, entry) in [
        (Stage::Mesh, "taskmain"),
        (Stage::Task, "meshmain"),
        (Stage::Fragment, "meshmain"),
    ] {
        assert_eq!(
            RuntimeShader::load(
                &bytes,
                Backend::Vulkan,
                SemanticProfile::V1,
                requested,
                entry,
            ),
            Err(ShaderLoadError::WrongStage)
        );
    }
    assert_eq!(
        RuntimeShader::load(
            &bytes,
            Backend::Vulkan,
            SemanticProfile::V1,
            Stage::Mesh,
            "mesh",
        ),
        Err(ShaderLoadError::UnknownEntryPoint)
    );
}
