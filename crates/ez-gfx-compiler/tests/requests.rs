//! Compiler integration tests.
use ez_gfx_artifact::{Stage, Target};
use ez_gfx_compiler::{CompilationRequest, CompilerConfig, CompilerError, TargetRequest};
use std::path::PathBuf;

#[test]
fn request_requires_per_entry_backend_coverage() {
    let request = CompilationRequest::new(
        PathBuf::from("shader.slang"),
        PathBuf::from("out"),
        vec![TargetRequest::new(Target::Spirv, Stage::Vertex, "main", "spirv_1_5").unwrap()],
    );
    assert!(matches!(
        request.validate(),
        Err(CompilerError::MissingCoverage {
            target: Target::Dxil,
            ..
        })
    ));
}

#[test]
fn request_rejects_multiple_entry_points_for_one_stage() {
    let request = CompilationRequest::new(
        PathBuf::from("shader.slang"),
        PathBuf::from("out"),
        vec![
            TargetRequest::new(Target::Spirv, Stage::Vertex, "first", "spirv_1_5").unwrap(),
            TargetRequest::new(Target::Dxil, Stage::Vertex, "second", "sm_6_5").unwrap(),
            TargetRequest::new(Target::Msl, Stage::Vertex, "first", "metal_3_0").unwrap(),
        ],
    );

    assert!(matches!(
        request.validate(),
        Err(CompilerError::DuplicateStage(Stage::Vertex))
    ));
}

#[test]
fn release_request_requires_metallib_not_msl() {
    let request = CompilationRequest::new(
        PathBuf::from("shader.slang"),
        PathBuf::from("out"),
        vec![
            TargetRequest::new(Target::Spirv, Stage::Vertex, "main", "spirv_1_5").unwrap(),
            TargetRequest::new(Target::Dxil, Stage::Vertex, "main", "sm_6_5").unwrap(),
            TargetRequest::new(Target::Msl, Stage::Vertex, "main", "metal_3_0").unwrap(),
        ],
    );
    assert!(matches!(
        request.validate(),
        Err(CompilerError::MissingCoverage {
            target: Target::Metallib,
            ..
        })
    ));
}

#[test]
fn development_request_may_use_msl_intermediate() {
    let root = std::env::temp_dir().join(format!("ez-gfx-dev-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("shader.slang");
    std::fs::write(&source, "[shader(\"compute\")] void main() {}").unwrap();
    let mut request = CompilationRequest::new(
        source.clone(),
        root.join("out"),
        vec![
            TargetRequest::new(Target::Spirv, Stage::Vertex, "main", "spirv_1_5").unwrap(),
            TargetRequest::new(Target::Dxil, Stage::Vertex, "main", "sm_6_5").unwrap(),
            TargetRequest::new(Target::Msl, Stage::Vertex, "main", "metal_3_0").unwrap(),
        ],
    );
    request.release_complete = false;
    assert!(request.validate().is_ok());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn request_rejects_empty_entry_and_nul_options() {
    assert!(matches!(
        TargetRequest::new(Target::Spirv, Stage::Vertex, "", "spirv_1_5"),
        Err(CompilerError::InvalidRequest(_))
    ));
    let mut request =
        CompilationRequest::new(PathBuf::from("shader.slang"), PathBuf::from("out"), vec![]);
    request.defines.push("BAD\0DEFINE".into());
    assert!(matches!(
        request.validate(),
        Err(CompilerError::InvalidRequest(_))
    ));
}

#[test]
fn native_binding_validation_does_not_use_executable_path() {
    let config = CompilerConfig::new("");
    assert!(!matches!(
        config.validate_tool(),
        Err(CompilerError::Native(_))
    ));
}

#[cfg(not(target_os = "macos"))]
#[test]
fn missing_apple_tool_is_typed() {
    let result = ez_gfx_compiler::build_metallib(
        PathBuf::from("missing.metal"),
        PathBuf::from("out.metallib"),
    );
    assert!(matches!(result, Err(CompilerError::AppleToolNotFound(_))));
}

#[test]
fn in_process_compute_smoke_when_native_slang_is_available() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let root = std::env::temp_dir().join(format!("ez-gfx-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("smoke.slang");
    std::fs::write(
        &source,
        r#"[__AttributeUsage(_AttributeTargets.Var)]
struct StructuredBufferAttribute { string name; };

[StructuredBuffer("values")]
RWStructuredBuffer<uint> values;

[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id: SV_DispatchThreadID) { values[id.x] += 1; }
"#,
    )
    .unwrap();
    let mut request = CompilationRequest::new(
        source,
        root.join("out"),
        vec![
            TargetRequest::new(Target::Spirv, Stage::Compute, "main", "spirv_1_5").unwrap(),
            TargetRequest::new(Target::Dxil, Stage::Compute, "main", "sm_6_5").unwrap(),
            TargetRequest::new(Target::Msl, Stage::Compute, "main", "metal_3_0").unwrap(),
        ],
    );
    request.release_complete = false;
    let result = ez_gfx_compiler::compile(&CompilerConfig::new(""), &request);
    match result {
        Ok(artifact) => {
            assert_eq!(artifact.variants.len(), 3);
            assert!(
                artifact
                    .variants
                    .iter()
                    .all(|variant| !variant.bytes.is_empty())
            );
            assert!(artifact.metadata.len() > 32);
            let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata).unwrap();
            let spirv = metadata["reflections"]
                .as_array()
                .unwrap()
                .iter()
                .find(|reflection| reflection["target"] == "Spirv")
                .unwrap();
            let parameters = spirv["reflection"]["parameters"].as_array().unwrap();
            let binding = parameters
                .iter()
                .find(|parameter| parameter["semantic_name"] == "values")
                .unwrap();
            assert_eq!(binding["api_kind"], "structured");
            assert_eq!(binding["binding_index"], 0);
            let dxil = metadata["reflections"]
                .as_array()
                .unwrap()
                .iter()
                .find(|reflection| reflection["target"] == "Dxil")
                .unwrap();
            assert_eq!(dxil["profile"], "sm_6_5");
            let dxil_parameters = dxil["reflection"]["parameters"].as_array().unwrap();
            let dxil_binding = dxil_parameters
                .iter()
                .find(|parameter| parameter["semantic_name"] == "values")
                .unwrap();
            assert_eq!(dxil_binding["resource_access"], "ReadWrite");
            assert_eq!(binding["binding_space"], 0);
        }
        Err(CompilerError::NativeUnavailable) => {}
        Err(CompilerError::Native(message))
            if message.to_lowercase().contains("unavailable")
                || message.to_lowercase().contains("not found") => {}
        Err(error) => panic!("native smoke failed: {error}"),
    }
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(target_os = "macos")]
#[test]
fn concurrent_metallib_compiles_isolate_temporary_outputs() {
    if shader_slang::GlobalSession::new().is_none() {
        return;
    }
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("ez-gfx-concurrent-{}-{unique}", std::process::id()));
    let output = root.join("out");
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("shader.slang");
    std::fs::write(
        &source,
        r#"[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id: SV_DispatchThreadID) {}
"#,
    )
    .unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let threads = (0..2)
        .map(|_| {
            let source = source.clone();
            let output = output.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let request = CompilationRequest::new(
                    source,
                    output,
                    vec![
                        TargetRequest::new(Target::Spirv, Stage::Compute, "main", "spirv_1_5")
                            .unwrap(),
                        TargetRequest::new(Target::Dxil, Stage::Compute, "main", "sm_6_5").unwrap(),
                        TargetRequest::new(Target::Metallib, Stage::Compute, "main", "metal_3_0")
                            .unwrap(),
                    ],
                );
                barrier.wait();
                ez_gfx_compiler::compile(&CompilerConfig::new(""), &request)
                    .unwrap()
                    .execution_digest()
            })
        })
        .collect::<Vec<_>>();
    let digests = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(digests[0], digests[1]);
    assert_eq!(std::fs::read_dir(&output).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(root);
}
