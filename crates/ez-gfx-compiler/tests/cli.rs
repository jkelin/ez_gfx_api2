//! Compiler integration tests.
use std::{fs, process::Command};

#[test]
fn cli_compiles_development_manifest_when_native_slang_exists() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = std::env::temp_dir().join(format!("ez-gfx-cli-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let source = root.join("shader.slang");
    fs::write(
        &source,
        "[shader(\"compute\")] [numthreads(1,1,1)] void main() {}",
    )
    .unwrap();
    let output = root.join("shader.ezshader");
    let manifest = root.join("manifest.json");
    let value = serde_json::json!({"source": source, "output": output, "development": true, "targets": [
        {"target":"spirv","stage":"compute","entry":"main","profile":"spirv_1_5"},
        {"target":"dxil","stage":"compute","entry":"main","profile":"sm_6_5"},
        {"target":"msl","stage":"compute","entry":"main","profile":"metal_3_0"}
    ]});
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let binary = env!("CARGO_BIN_EXE_ez-gfx-compile");
    let result = Command::new(binary).arg(&manifest).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let first_size = fs::metadata(&output).unwrap().len();
    let result = Command::new(binary).arg(&manifest).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::metadata(&output).unwrap().len(), first_size);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn cli_rejects_missing_manifest() {
    let result = Command::new(env!("CARGO_BIN_EXE_ez-gfx-compile"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage"));
}

#[test]
fn cli_rejects_invalid_manifest() {
    let root = std::env::temp_dir().join(format!("ez-gfx-cli-invalid-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let manifest = root.join("invalid.json");
    fs::write(&manifest, b"{").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ez-gfx-compile"))
        .arg(&manifest)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("invalid manifest"));
    let _ = fs::remove_dir_all(root);
}
