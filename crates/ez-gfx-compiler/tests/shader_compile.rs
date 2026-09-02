//! Owned shader compilation tests.

use ez_gfx_artifact::Artifact;
use ez_gfx_compiler::{CompilerError, Target, compile_shader};
use std::fs;

const ALL_TARGETS: &[Target] = &[Target::Spirv, Target::Dxil, Target::Metal];

#[test]
fn missing_shader_reports_its_path_before_native_compiler_use() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("missing.slang");

    let error = compile_shader(&source, ALL_TARGETS, true).unwrap_err();

    assert!(matches!(error, CompilerError::SourceRead { path, .. } if path == source));
}

#[test]
fn directory_with_slang_extension_is_rejected_as_a_non_file() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::create_dir(&source).unwrap();

    assert!(matches!(
        compile_shader(&source, ALL_TARGETS, true),
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
        compile_shader(&source, &[], true),
        Err(CompilerError::InvalidRequest("targets"))
    ));
    assert!(matches!(
        compile_shader(&source, &[Target::Spirv, Target::Spirv], true),
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
            compile_shader(&source, &[Target::Spirv], true),
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

    let canonical = compile_shader(&source, ALL_TARGETS, true).unwrap();
    let reversed =
        compile_shader(&source, &[Target::Metal, Target::Dxil, Target::Spirv], true).unwrap();

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

    let bytes = compile_shader(&source, ALL_TARGETS, true).unwrap();
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

    let bytes = compile_shader(&source, ALL_TARGETS, true).unwrap();
    let artifact = Artifact::decode(&bytes).unwrap();

    assert_eq!(artifact.variants.len(), 6);
}

#[test]
fn duplicate_declared_stage_is_rejected_when_slang_is_available() {
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

    let error = compile_shader(&source, ALL_TARGETS, true).unwrap_err();

    assert!(matches!(error, CompilerError::DuplicateStage(_)));
}
