//! Owned shader compilation tests.

use ez_gfx_artifact::Artifact;
use ez_gfx_compiler::{CompilerError, EasyGraphicsCompiler, Target};
use std::fs;

const ALL_TARGETS: &[Target] = &[Target::Spirv, Target::Dxil, Target::Metal];

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
