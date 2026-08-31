//! Compiler command-line integration.
use anyhow::{Context, Result, bail};
use clap::Parser;
use ez_gfx_artifact::{Stage, Target};
use ez_gfx_compiler::{CompilationRequest, CompilerConfig, TargetRequest};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "ez-gfx-compile", about = "Compile an ez-gfx shader manifest")]
struct Cli {
    /// Shader compilation manifest.
    #[arg(value_name = "MANIFEST")]
    manifest: PathBuf,
}

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

fn target(value: &str) -> Result<Target> {
    match value {
        "spirv" => Ok(Target::Spirv),
        "dxil" => Ok(Target::Dxil),
        "msl" => Ok(Target::Msl),
        "metallib" => Ok(Target::Metallib),
        _ => bail!("unknown target `{value}`"),
    }
}

fn stage(value: &str) -> Result<Stage> {
    match value {
        "vertex" => Ok(Stage::Vertex),
        "fragment" => Ok(Stage::Fragment),
        "compute" => Ok(Stage::Compute),
        "geometry" => Ok(Stage::Geometry),
        "tess-control" => Ok(Stage::TessellationControl),
        "tess-eval" => Ok(Stage::TessellationEvaluation),
        _ => bail!("unknown stage `{value}`"),
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(&cli) {
        eprintln!("ez-gfx-compile: {error:#}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<()> {
    let manifest_bytes = fs::read(&cli.manifest)
        .with_context(|| format!("read shader manifest {}", cli.manifest.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).context("invalid manifest")?;
    let targets = manifest
        .targets
        .into_iter()
        .map(|spec| {
            let target = target(&spec.target)?;
            let stage = stage(&spec.stage)?;
            TargetRequest::new(target, stage, spec.entry, spec.profile)
                .context("validate shader target request")
        })
        .collect::<Result<Vec<_>>>()?;
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
        serde_json::to_vec(&manifest.semantic_metadata).context("serialize semantic metadata")?;
    request.toolchain = manifest.toolchain;
    request.apple_toolchain = manifest.apple_toolchain;
    request.release_complete = !manifest.development;

    let config = CompilerConfig::new(manifest.required_version);
    let artifact =
        ez_gfx_compiler::compile(&config, &request).context("compile shader manifest")?;
    let bytes = artifact.encode().context("encode shader artifact")?;
    let parent = manifest
        .output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create output directory {}", parent.display()))?;
    let temporary = parent.join(format!(".ezgfxshader-{}.tmp", std::process::id()));
    fs::write(&temporary, bytes)
        .with_context(|| format!("write temporary artifact {}", temporary.display()))?;
    if let Err(error) = fs::rename(&temporary, &manifest.output) {
        let _ = fs::remove_file(&temporary);
        return Err(error).with_context(|| {
            format!(
                "replace shader artifact {} atomically",
                manifest.output.display()
            )
        });
    }
    Ok(())
}
