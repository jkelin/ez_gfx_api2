//! Compiler request behavior tests.

use ez_gfx_compiler::{CompilerError, Target, compile_shader};
use std::fs;

const ALL_TARGETS: &[Target] = &[Target::Spirv, Target::Dxil, Target::Metal];

#[test]
fn source_without_declared_entries_is_rejected_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(&source, "float helper(float value) { return value; }").unwrap();

    let error = compile_shader(&source, ALL_TARGETS, true).unwrap_err();

    assert!(matches!(error, CompilerError::NoEntryPoints));
}

#[test]
fn selected_target_subset_is_emitted_for_every_discovered_stage_when_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    fs::write(
        &source,
        r#"
            [shader("vertex")]
            float4 vs() : SV_Position { return float4(0.0); }

            [shader("fragment")]
            float4 fs() : SV_Target { return float4(1.0); }
        "#,
    )
    .unwrap();

    let bytes = compile_shader(&source, &[Target::Spirv], true).unwrap();
    let artifact = ez_gfx_artifact::Artifact::decode(&bytes).unwrap();

    assert_eq!(artifact.variants.len(), 2);
    assert!(artifact.variants.iter().all(|variant| {
        variant.target == ez_gfx_artifact::Target::Spirv
            && matches!(
                variant.stage,
                ez_gfx_artifact::Stage::Vertex | ez_gfx_artifact::Stage::Fragment
            )
    }));
}
