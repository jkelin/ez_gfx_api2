//! Runtime integration and contract tests.

use ez_gfx_artifact::{Artifact, Provenance, Stage, Target, TargetVariant};
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_runtime::shader::{RuntimeShader, ShaderLoadError, ShaderRequest};

fn artifact(metal: Target) -> Vec<u8> {
    let variants = [
        (Target::Spirv, b"spirv".to_vec()),
        (Target::Dxil, b"dxil".to_vec()),
        (metal, b"metal".to_vec()),
    ]
    .into_iter()
    .map(|(target, bytes)| {
        TargetVariant::new(target, Stage::Compute, "main", "ez-gfx-v1", bytes).unwrap()
    })
    .collect();
    Artifact::new(
        b"{\"semantic_abi\":1}".to_vec(),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap()
}

#[test]
fn selects_exact_backend_profile_entry_and_stage() {
    let shader = RuntimeShader::load(
        &artifact(Target::Metallib),
        Backend::Dx12,
        SemanticProfile::V1,
        &[ShaderRequest::new("main", Stage::Compute).unwrap()],
    )
    .unwrap();
    assert_eq!(shader.product("main", Stage::Compute).unwrap(), b"dxil");
    assert_eq!(shader.metadata(), b"{\"semantic_abi\":1}");
}

#[test]
fn metal_runtime_requires_offline_metallib_and_never_uses_msl() {
    assert_eq!(
        RuntimeShader::load(
            &artifact(Target::Msl),
            Backend::Metal,
            SemanticProfile::V1,
            &[ShaderRequest::new("main", Stage::Compute).unwrap()]
        ),
        Err(ShaderLoadError::MissingProduct {
            entry: "main".into(),
            stage: Stage::Compute
        })
    );
}

#[test]
fn malformed_or_missing_requests_fail_without_fallback() {
    assert!(matches!(
        RuntimeShader::load(
            b"bad",
            Backend::Vulkan,
            SemanticProfile::V1,
            &[ShaderRequest::new("main", Stage::Compute).unwrap()]
        ),
        Err(ShaderLoadError::Artifact(_))
    ));
    assert_eq!(
        ShaderRequest::new("", Stage::Compute),
        Err(ShaderLoadError::InvalidRequest)
    );
    assert_eq!(
        RuntimeShader::load(
            &artifact(Target::Metallib),
            Backend::Vulkan,
            SemanticProfile::V1,
            &[ShaderRequest::new("other", Stage::Compute).unwrap()]
        ),
        Err(ShaderLoadError::MissingProduct {
            entry: "other".into(),
            stage: Stage::Compute
        })
    );
}

#[test]
fn pipeline_stage_pairing_requires_exact_vertex_fragment_or_compute_products() {
    let variants = [Target::Spirv, Target::Dxil, Target::Metallib]
        .into_iter()
        .flat_map(|target| {
            [Stage::Vertex, Stage::Fragment]
                .into_iter()
                .map(move |stage| {
                    TargetVariant::new(
                        target,
                        stage,
                        "main",
                        "ez-gfx-v1",
                        vec![target as u8 + stage as u8 + 1],
                    )
                    .unwrap()
                })
        })
        .collect();
    let graphics_artifact = Artifact::new(
        b"{\"semantic_abi\":1}".to_vec(),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap();
    let graphics = RuntimeShader::load(
        &graphics_artifact,
        Backend::Vulkan,
        SemanticProfile::V1,
        &[
            ShaderRequest::new("main", Stage::Vertex).unwrap(),
            ShaderRequest::new("main", Stage::Fragment).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(graphics.graphics_pair().unwrap().0.1, Stage::Vertex);
    assert!(graphics.compute_product().is_err());

    let compute_artifact = artifact(Target::Metallib);
    let compute = RuntimeShader::load(
        &compute_artifact,
        Backend::Vulkan,
        SemanticProfile::V1,
        &[ShaderRequest::new("main", Stage::Compute).unwrap()],
    )
    .unwrap();
    assert_eq!(compute.compute_product().unwrap().1, Stage::Compute);
    assert!(compute.graphics_pair().is_err());
}

#[test]
fn one_artifact_can_select_graphics_and_compute_pipelines() {
    let variants = [Target::Spirv, Target::Dxil, Target::Metallib]
        .into_iter()
        .flat_map(|target| {
            [Stage::Vertex, Stage::Fragment, Stage::Compute]
                .into_iter()
                .map(move |stage| {
                    TargetVariant::new(
                        target,
                        stage,
                        format!("{stage:?}"),
                        "ez-gfx-v1",
                        vec![target as u8 + stage as u8 + 1],
                    )
                    .unwrap()
                })
        })
        .collect();
    let artifact = Artifact::new(
        b"{\"semantic_abi\":1}".to_vec(),
        Provenance::new("slangc", "2026.16", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap();
    let shader = RuntimeShader::load(
        &artifact,
        Backend::Vulkan,
        SemanticProfile::V1,
        &[
            ShaderRequest::new("Vertex", Stage::Vertex).unwrap(),
            ShaderRequest::new("Fragment", Stage::Fragment).unwrap(),
            ShaderRequest::new("Compute", Stage::Compute).unwrap(),
        ],
    )
    .unwrap();

    assert!(shader.graphics_pair().is_ok());
    assert_eq!(shader.compute_product().unwrap().1, Stage::Compute);
}
