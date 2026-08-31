//! Compiles portable example shader artifacts into Cargo's private output directory.

use anyhow::Context as _;
use ez_gfx_compiler::{MetalOutput, compile, plan_manifest};
use std::{env, fs, path::Path};

const SHADERS: [(&str, &str); 6] = [
    ("01_triangle/01_triangle.json", "01_triangle.ezgfxshader"),
    (
        "02_textured_cube/02_textured_cube.json",
        "02_textured_cube.ezgfxshader",
    ),
    (
        "03_compute_structured/03_compute_structured.json",
        "03_compute_structured.ezgfxshader",
    ),
    ("04_imgui/04_imgui.json", "04_imgui.ezgfxshader"),
    ("05_helmet/05_helmet.json", "05_helmet.ezgfxshader"),
    (
        "06_sponza_ktx2/06_sponza_ktx2.json",
        "06_sponza_ktx2.ezgfxshader",
    ),
];

fn main() {
    if let Err(error) = build_shaders() {
        panic!("build example shaders: {error}");
    }
}

fn build_shaders() -> anyhow::Result<()> {
    let manifest_dir =
        env::var_os("CARGO_MANIFEST_DIR").context("CARGO_MANIFEST_DIR is missing")?;
    let examples = Path::new(&manifest_dir);
    let workspace = examples
        .parent()
        .context("examples package has no workspace parent")?;
    let out_dir = env::var_os("OUT_DIR").context("OUT_DIR is missing")?;
    let out_dir = Path::new(&out_dir);
    let metal = if env::var_os("CARGO_CFG_TARGET_VENDOR").as_deref() == Some("apple".as_ref()) {
        MetalOutput::Library
    } else {
        MetalOutput::Source
    };

    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_VENDOR");
    track_slang_files(workspace)?;
    for (manifest_name, artifact_name) in SHADERS {
        let manifest_path = examples.join(manifest_name);
        println!("cargo::rerun-if-changed={}", manifest_path.display());
        let manifest = fs::read(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let plan = plan_manifest(&manifest, workspace, out_dir, metal)
            .with_context(|| format!("plan {}", manifest_path.display()))?;
        let artifact = compile(&plan.config, &plan.request)
            .with_context(|| format!("compile {}", manifest_path.display()))?;
        let bytes = artifact
            .encode()
            .with_context(|| format!("encode {}", manifest_path.display()))?;
        let output = out_dir.join(artifact_name);
        fs::write(&output, bytes).with_context(|| format!("write {}", output.display()))?;
    }
    Ok(())
}

fn track_slang_files(directory: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("scan shader imports in {}", directory.display()))?
    {
        let entry = entry.context("read shader import entry")?;
        let path = entry.path();
        if path.is_dir() {
            let ignored = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| matches!(name, ".git" | "target"));
            if !ignored {
                track_slang_files(&path)?;
            }
        } else if path.extension().and_then(|value| value.to_str()) == Some("slang") {
            println!("cargo::rerun-if-changed={}", path.display());
        }
    }
    Ok(())
}
