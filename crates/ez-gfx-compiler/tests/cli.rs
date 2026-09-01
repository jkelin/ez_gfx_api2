//! Compiler CLI tests.

use std::{fs, process::Command};

#[test]
fn cli_compiles_source_to_explicit_output_when_slang_exists() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("shader.slang");
    let output = root.path().join("build").join("shader.ezgfxshader");
    fs::write(
        &source,
        "[shader(\"compute\")] [numthreads(1,1,1)] void main() {}",
    )
    .unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_ez-gfx-compile"))
        .arg(&source)
        .args(["--target", "spirv", "--target", "dxil", "--target", "metal"])
        .arg("--development")
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    ez_gfx_artifact::Artifact::decode(&fs::read(output).unwrap()).unwrap();
}

#[test]
fn cli_defaults_output_beside_source_when_slang_exists() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("named.shader.slang");
    fs::write(
        &source,
        "[shader(\"compute\")] [numthreads(1,1,1)] void main() {}",
    )
    .unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_ez-gfx-compile"))
        .arg(&source)
        .args(["--target", "spirv", "--target", "dxil", "--target", "metal"])
        .arg("--development")
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(root.path().join("named.shader.ezgfxshader").is_file());
}

#[test]
fn cli_help_describes_source_target_and_output_contract() {
    let result = Command::new(env!("CARGO_BIN_EXE_ez-gfx-compile"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("<SOURCE>"));
    assert!(stdout.contains("--target <TARGETS>"));
    assert!(stdout.contains("--development"));
    assert!(stdout.contains("--output <OUTPUT>"));
}

#[test]
fn cli_rejects_missing_targets_unknown_targets_and_extra_sources() {
    for arguments in [
        vec!["shader.slang"],
        vec!["shader.slang", "--target", "msl"],
        vec!["first.slang", "second.slang", "--target", "spirv"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_ez-gfx-compile"))
            .args(arguments)
            .output()
            .unwrap();
        assert!(!result.status.success());
    }
}

#[test]
fn cli_reports_source_read_context() {
    let result = Command::new(env!("CARGO_BIN_EXE_ez-gfx-compile"))
        .args(["does-not-exist.slang", "--target", "spirv"])
        .arg("--development")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("read shader source"));
}
