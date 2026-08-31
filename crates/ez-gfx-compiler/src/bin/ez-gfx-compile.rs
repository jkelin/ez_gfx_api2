//! Compiler command-line integration.
use ez_gfx_artifact::{Stage, Target};
use ez_gfx_compiler::{CompilationRequest, CompilerConfig, TargetRequest};
use serde::Deserialize;
use std::{env, fs, path::PathBuf};

#[derive(Deserialize)]
struct Manifest {
    source: PathBuf,
    output: PathBuf,
    #[serde(default)]
    required_version: String,
    #[serde(default)]
    include_dirs: Vec<PathBuf>,
    #[serde(default)]
    defines: Vec<String>,
    #[serde(default)]
    semantic_metadata: serde_json::Value,
    #[serde(default)]
    toolchain: String,
    #[serde(default)]
    apple_toolchain: String,
    #[serde(default)]
    development: bool,
    targets: Vec<TargetSpec>,
}
#[derive(Deserialize)]
struct TargetSpec {
    target: String,
    stage: String,
    entry: String,
    profile: String,
}
fn target(value: &str) -> Result<Target, String> {
    match value {
        "spirv" => Ok(Target::Spirv),
        "dxil" => Ok(Target::Dxil),
        "msl" => Ok(Target::Msl),
        "metallib" => Ok(Target::Metallib),
        _ => Err(format!("unknown target `{value}`")),
    }
}
fn stage(value: &str) -> Result<Stage, String> {
    match value {
        "vertex" => Ok(Stage::Vertex),
        "fragment" => Ok(Stage::Fragment),
        "compute" => Ok(Stage::Compute),
        "geometry" => Ok(Stage::Geometry),
        "tess-control" => Ok(Stage::TessellationControl),
        "tess-eval" => Ok(Stage::TessellationEvaluation),
        _ => Err(format!("unknown stage `{value}`")),
    }
}
fn main() {
    if let Err(error) = run() {
        eprintln!("ez-gfx-compile: {error}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let manifest_path = env::args()
        .nth(1)
        .ok_or_else(|| "usage: ez-gfx-compile MANIFEST.json".to_owned())?;
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(&manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid manifest: {e}"))?;
    let targets = manifest
        .targets
        .into_iter()
        .map(|spec| {
            TargetRequest::new(
                target(&spec.target)?,
                stage(&spec.stage)?,
                spec.entry,
                spec.profile,
            )
            .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut request = CompilationRequest::new(
        manifest.source,
        manifest
            .output
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf(),
        targets,
    );
    request.include_dirs = manifest.include_dirs;
    request.defines = manifest.defines;
    request.semantic_metadata =
        serde_json::to_vec(&manifest.semantic_metadata).map_err(|e| e.to_string())?;
    request.toolchain = manifest.toolchain;
    request.apple_toolchain = manifest.apple_toolchain;
    request.release_complete = !manifest.development;
    let config = CompilerConfig::new(manifest.required_version);
    let artifact = ez_gfx_compiler::compile(&config, &request).map_err(|e| e.to_string())?;
    let bytes = artifact.encode().map_err(|e| e.to_string())?;
    let parent = manifest
        .output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let temp = parent.join(format!(".ezshader-{}.tmp", std::process::id()));
    fs::write(&temp, bytes).map_err(|e| e.to_string())?;
    if let Err(error) = fs::rename(&temp, &manifest.output) {
        let _ = fs::remove_file(&temp);
        return Err(format!("atomic output replacement failed: {error}"));
    }
    Ok(())
}
