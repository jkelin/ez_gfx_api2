//! Owned shader compilation tests.

use ez_gfx_artifact::Artifact;
use ez_gfx_compiler::{CompilerError, EasyGraphicsCompiler, Target};
use std::fs;

const ALL_TARGETS: &[Target] = &[Target::Spirv, Target::Dxil, Target::Metal];

fn write_shared_root(root: &std::path::Path) {
    fs::write(
        root.join("ez_gfx_api.slang"),
        include_str!("../../../ez_gfx_api.slang"),
    )
    .unwrap();
}

fn assert_stage_products(artifact: &Artifact, expected: &[(ez_gfx_artifact::Stage, &str)]) {
    assert_eq!(artifact.variants.len(), expected.len() * ALL_TARGETS.len());
    for (stage, entry_point) in expected {
        let products: Vec<_> = artifact
            .variants
            .iter()
            .filter(|variant| variant.stage == *stage && variant.entry_point == *entry_point)
            .collect();
        assert_eq!(products.len(), ALL_TARGETS.len());
        assert_eq!(
            products
                .iter()
                .map(|variant| variant.target)
                .collect::<std::collections::BTreeSet<_>>(),
            if cfg!(target_os = "macos") {
                std::collections::BTreeSet::from([
                    ez_gfx_artifact::Target::Spirv,
                    ez_gfx_artifact::Target::Dxil,
                    ez_gfx_artifact::Target::Metallib,
                ])
            } else {
                std::collections::BTreeSet::from([
                    ez_gfx_artifact::Target::Spirv,
                    ez_gfx_artifact::Target::Dxil,
                    ez_gfx_artifact::Target::Msl,
                ])
            }
        );
        let spirv = products
            .iter()
            .find(|variant| variant.target == ez_gfx_artifact::Target::Spirv)
            .unwrap();
        assert!(
            spirv
                .bytes
                .windows(b"SPV_EXT_mesh_shader".len())
                .any(|window| window == b"SPV_EXT_mesh_shader")
        );
        assert!(
            !spirv
                .bytes
                .windows(b"SPV_NV_mesh_shader".len())
                .any(|window| window == b"SPV_NV_mesh_shader")
        );
    }

    let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata).unwrap();
    for reflection in metadata["reflections"].as_array().unwrap() {
        let expected_profile = match reflection["target"].as_str().unwrap() {
            "Spirv" => "spirv_1_5",
            "Dxil" => "sm_6_5",
            "Msl" | "Metallib" => "metal_3_0",
            target => panic!("unexpected target {target}"),
        };
        assert_eq!(reflection["profile"], expected_profile);
        assert_eq!(
            reflection["reflection"]["workgroup_size"],
            serde_json::json!([1, 1, 1])
        );
    }
}

#[test]
fn missing_shader_reports_its_path_before_native_compiler_use() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("missing.slang");

    let error = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap_err();

    assert!(matches!(error, CompilerError::SourceRead { path, .. } if path == source));
}

#[test]
fn directory_with_slang_extension_is_rejected_as_a_non_file() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::create_dir(&source).unwrap();

    assert!(matches!(
        EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true),
        Err(CompilerError::InvalidRequest("source file"))
    ));
}

#[test]
fn empty_and_duplicate_target_lists_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        "[shader(\"compute\")] [numthreads(1,1,1)] void main() {}",
    )
    .unwrap();

    assert!(matches!(
        EasyGraphicsCompiler::compile_shader(&source, &[], true),
        Err(CompilerError::InvalidRequest("targets"))
    ));
    assert!(matches!(
        EasyGraphicsCompiler::compile_shader(&source, &[Target::Spirv, Target::Spirv], true),
        Err(CompilerError::InvalidRequest("duplicate target"))
    ));
}
#[test]
fn non_lowercase_slang_source_extensions_are_rejected_before_native_compiler_use() {
    let root = tempfile::tempdir().unwrap();

    for extension in ["txt", "SLANG"] {
        let source = root.path().join(format!("shader.{extension}"));
        fs::write(
            &source,
            "[shader(\"compute\")] [numthreads(1,1,1)] void main() {}",
        )
        .unwrap();

        assert!(matches!(
            EasyGraphicsCompiler::compile_shader(&source, &[Target::Spirv], true),
            Err(CompilerError::InvalidRequest("source extension"))
        ));
    }
}

#[test]
fn target_order_does_not_change_artifact_bytes_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        "[shader(\"compute\")] [numthreads(1,1,1)] void main() {}",
    )
    .unwrap();

    let canonical = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap();
    let reversed = EasyGraphicsCompiler::compile_shader(
        &source,
        &[Target::Metal, Target::Dxil, Target::Spirv],
        true,
    )
    .unwrap();

    assert_eq!(canonical, reversed);
}

#[test]
fn compute_workgroup_size_is_serialized_for_every_target_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        "[shader(\"compute\")] [numthreads(8,2,1)] void main() {}",
    )
    .unwrap();

    let bytes = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap();
    let artifact = Artifact::decode(&bytes).unwrap();
    let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata).unwrap();

    for reflection in metadata["reflections"].as_array().unwrap() {
        assert_eq!(
            reflection["reflection"]["workgroup_size"],
            serde_json::json!([8, 2, 1])
        );
    }
}

#[test]
fn discovered_entries_compile_for_every_requested_target_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        r#"
            [shader("vertex")] float4 vertexmain() : SV_Position { return 0; }
            [shader("fragment")] float4 fragmentmain() : SV_Target { return 1; }
        "#,
    )
    .unwrap();

    let bytes = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap();
    let artifact = Artifact::decode(&bytes).unwrap();

    assert_eq!(artifact.variants.len(), 6);
}

#[test]
fn same_name_entry_points_resolve_by_stage_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(&source, include_str!("fixtures/same-name-stages.slang")).unwrap();

    let compiled = EasyGraphicsCompiler::compile_shader(&source, &[Target::Spirv], true).unwrap();
    let artifact = Artifact::decode(&compiled).unwrap();

    for stage in [
        ez_gfx_artifact::Stage::Vertex,
        ez_gfx_artifact::Stage::Fragment,
    ] {
        let products: Vec<_> = artifact
            .variants
            .iter()
            .filter(|variant| variant.stage == stage && variant.entry_point == "shared")
            .collect();
        assert_eq!(products.len(), 1);
        assert_eq!(products[0].target, ez_gfx_artifact::Target::Spirv);
    }

    assert_eq!(artifact.variants.len(), 2);
    let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata).unwrap();
    for stage in ["Vertex", "Fragment"] {
        let reflections: Vec<_> = metadata["reflections"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|reflection| reflection["stage"] == stage)
            .collect();
        assert_eq!(reflections.len(), 1);
        assert_eq!(reflections[0]["entry"], "shared");
        assert_eq!(reflections[0]["reflection"]["entry"], "shared");
        assert_eq!(reflections[0]["reflection"]["stage"], stage);
    }
}

#[test]
fn same_stage_entry_points_compile_and_round_trip_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        r#"
            [shader("compute")] [numthreads(1,1,1)] void first() {}
            [shader("compute")] [numthreads(1,1,1)] void second() {}
        "#,
    )
    .unwrap();

    let compiled = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap();
    let saved = compiled.save_shader();
    let loaded = EasyGraphicsCompiler::load_compiled_shader(&saved).unwrap();
    let artifact = Artifact::decode(&loaded).unwrap();

    assert_eq!(artifact.variants.len(), 6);
    assert_eq!(
        artifact
            .variants
            .iter()
            .map(|variant| variant.entry_point.as_str())
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from(["first", "second"])
    );
}

#[test]
fn mesh_entry_compiles_for_every_requested_target_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    write_shared_root(root.path());
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        r#"
            import ez_gfx_api;

            struct VertexOutput {
                float4 position : SV_Position;
                float3 color : COLOR;
            };

            [shader("mesh")]
            [numthreads(1, 1, 1)]
            [outputtopology("triangle")]
            void meshmain(
                OutputVertices<VertexOutput, 3> vertices,
                OutputIndices<uint3, 1> triangles)
            {
                SetMeshOutputCounts(3, 1);
                vertices[0] = { float4(-1, -1, 0, 1), float3(1, 0, 0) };
                vertices[1] = { float4(0, 1, 0, 1), float3(0, 1, 0) };
                vertices[2] = { float4(1, -1, 0, 1), float3(0, 0, 1) };
                triangles[0] = uint3(0, 1, 2);
            }
        "#,
    )
    .unwrap();

    let bytes = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap();
    let artifact = Artifact::decode(&bytes).unwrap();

    assert_stage_products(&artifact, &[(ez_gfx_artifact::Stage::Mesh, "meshmain")]);
}

#[test]
fn task_and_mesh_entries_compile_for_every_requested_target_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    write_shared_root(root.path());
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        r#"
            import ez_gfx_api;

            struct MeshPayload { uint vertex_count; float3 color; };
            struct VertexOutput {
                float4 position : SV_Position;
                float3 color : COLOR;
            };

            [shader("amplification")]
            [numthreads(1, 1, 1)]
            void taskmain()
            {
                MeshPayload payload;
                payload.vertex_count = 3;
                payload.color = float3(0.25, 0.5, 0.75);
                DispatchMesh(1, 1, 1, payload);
            }

            [shader("mesh")]
            [numthreads(1, 1, 1)]
            [outputtopology("triangle")]
            void meshmain(
                in payload MeshPayload payload,
                OutputVertices<VertexOutput, 3> vertices,
                OutputIndices<uint3, 1> triangles)
            {
                SetMeshOutputCounts(payload.vertex_count, 1);
                vertices[0] = { float4(-1, -1, 0, 1), payload.color };
                vertices[1] = { float4(0, 1, 0, 1), payload.color };
                vertices[2] = { float4(1, -1, 0, 1), payload.color };
                triangles[0] = uint3(0, 1, 2);
            }
        "#,
    )
    .unwrap();

    let bytes = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap();
    let artifact = Artifact::decode(&bytes).unwrap();

    assert_stage_products(
        &artifact,
        &[
            (ez_gfx_artifact::Stage::Task, "taskmain"),
            (ez_gfx_artifact::Stage::Mesh, "meshmain"),
        ],
    );
}

#[cfg(target_os = "macos")]
#[test]
fn apple_development_metal_compilation_emits_runtime_loadable_metallib() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        "[shader(\"compute\")] [numthreads(1,1,1)] void main() {}",
    )
    .unwrap();

    let compiled = EasyGraphicsCompiler::compile_shader(&source, &[Target::Metal], true).unwrap();
    let artifact = Artifact::decode(&compiled).unwrap();

    assert_eq!(artifact.variants.len(), 1);
    assert_eq!(
        artifact.variants[0].target,
        ez_gfx_artifact::Target::Metallib
    );
    assert!(matches!(
        artifact.variants[0].compatibility,
        ez_gfx_artifact::TargetCompatibility::MetalLibrary { .. }
    ));
}

#[test]
fn shared_buffer_semantics_compile_for_every_target_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    fs::write(
        root.path().join("ez_gfx_api.slang"),
        include_str!("../../../ez_gfx_api.slang"),
    )
    .unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        r#"
            import ez_gfx_api;
            struct Draw { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };
            [Buffer("value")] StructuredBuffer<uint> value;
            [CounterBuffer("draws")] CounterBuffer<Draw> draws;
            [shader("compute")] [numthreads(1,1,1)]
            void main(uint3 id : SV_DispatchThreadID) {
                draws.set_count(0);
                uint index = draws.add_count(value[0]);
                draws.set(index, Draw(3, 1, 0, 0, 0));
            }
        "#,
    )
    .unwrap();

    let bytes = EasyGraphicsCompiler::compile_shader(&source, ALL_TARGETS, true).unwrap();
    let artifact = Artifact::decode(&bytes).unwrap();
    let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata).unwrap();

    let reflections = metadata["reflections"].as_array().unwrap();
    assert_eq!(reflections.len(), ALL_TARGETS.len());
    let metal_target = if cfg!(target_os = "macos") {
        "Metallib"
    } else {
        "Msl"
    };
    for target in ["Spirv", "Dxil", metal_target] {
        let matching: Vec<_> = reflections
            .iter()
            .filter(|reflection| reflection["target"] == target)
            .collect();
        assert_eq!(matching.len(), 1);
        let parameters = matching[0]["reflection"]["parameters"].as_array().unwrap();
        let counters: Vec<_> = parameters
            .iter()
            .filter(|parameter| parameter["api_kind"] == "counter_buffer")
            .collect();
        assert_eq!(counters.len(), 1);
        assert_eq!(counters[0]["descriptor_count"], 2);
        assert_eq!(counters[0]["resource_access"], "ReadWrite");
    }
}
