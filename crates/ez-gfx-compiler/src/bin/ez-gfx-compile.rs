//! Compiler command-line integration.

use anyhow::{Context, Result, bail};
use clap::Parser;
use ez_gfx_compiler::{EasyGraphicsCompiler, Target};
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "ez-gfx-compile", about = "Compile a Slang shader source")]
struct Cli {
    /// Root Slang shader source.
    #[arg(value_name = "SOURCE")]
    source: PathBuf,
    /// Target family to compile. Repeat for multiple families.
    #[arg(short, long = "target", value_name = "TARGETS", required = true)]
    targets: Vec<Target>,
    /// Emit portable development outputs, including MSL instead of metallib.
    #[arg(long)]
    development: bool,
    /// Artifact path. Defaults to SOURCE with the `.ezgfxshader` extension.
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(&cli) {
        eprintln!("ez-gfx-compile: {error:#}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<()> {
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| cli.source.with_extension("ezgfxshader"));
    if output.extension().and_then(|extension| extension.to_str()) != Some("ezgfxshader") {
        bail!("output must use the .ezgfxshader extension");
    }
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create artifact directory {}", parent.display()))?;
    let compiled = EasyGraphicsCompiler::compile_shader(&cli.source, &cli.targets, cli.development)
        .with_context(|| format!("compile shader source {}", cli.source.display()))?;
    fs::write(&output, compiled.save_shader())
        .with_context(|| format!("write shader artifact {}", output.display()))?;
    Ok(())
}
