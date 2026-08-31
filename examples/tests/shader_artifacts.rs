//! Build-script shader artifact contract tests.

use anyhow::Context as _;
use ez_gfx_artifact::{Artifact, Stage, Target};
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_runtime::shader::{RuntimeShader, ShaderLoadError};

const ARTIFACTS: [(&str, &[u8], bool); 6] = [
    (
        "triangle",
        include_bytes!(concat!(env!("OUT_DIR"), "/01_triangle.ezgfxshader")),
        false,
    ),
    (
        "cube",
        include_bytes!(concat!(env!("OUT_DIR"), "/02_textured_cube.ezgfxshader")),
        false,
    ),
    (
        "compute",
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/03_compute_structured.ezgfxshader"
        )),
        true,
    ),
    (
        "imgui",
        include_bytes!(concat!(env!("OUT_DIR"), "/04_imgui.ezgfxshader")),
        false,
    ),
    (
        "helmet",
        include_bytes!(concat!(env!("OUT_DIR"), "/05_helmet.ezgfxshader")),
        true,
    ),
    (
        "sponza",
        include_bytes!(concat!(env!("OUT_DIR"), "/06_sponza_ktx2.ezgfxshader")),
        true,
    ),
];

#[test]
fn generated_artifacts_select_every_stage_without_entry_requests() -> anyhow::Result<()> {
    for (name, bytes, has_compute) in ARTIFACTS {
        for backend in [Backend::Vulkan, Backend::Dx12] {
            let shader = RuntimeShader::load(bytes, backend, SemanticProfile::V1)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .with_context(|| format!("{name} {backend:?}"))?;
            assert!(shader.graphics_pair().is_ok(), "{name} {backend:?}");
            assert_eq!(
                shader.compute_product().is_ok(),
                has_compute,
                "{name} {backend:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn generated_metal_product_matches_the_build_host() -> anyhow::Result<()> {
    for (name, bytes, _) in ARTIFACTS {
        let artifact = Artifact::decode(bytes).with_context(|| format!("decode {name}"))?;
        #[cfg(target_vendor = "apple")]
        {
            assert!(
                artifact
                    .variants
                    .iter()
                    .any(|variant| variant.target == Target::Metallib)
            );
            assert!(RuntimeShader::load(bytes, Backend::Metal, SemanticProfile::V1).is_ok());
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            assert!(
                artifact
                    .variants
                    .iter()
                    .any(|variant| variant.target == Target::Msl)
            );
            assert!(
                !artifact
                    .variants
                    .iter()
                    .any(|variant| variant.target == Target::Metallib)
            );
            assert!(matches!(
                RuntimeShader::load(bytes, Backend::Metal, SemanticProfile::V1),
                Err(ShaderLoadError::MissingProduct {
                    stage: Stage::Vertex
                })
            ));
        }
        assert!(!name.is_empty());
    }
    Ok(())
}

#[test]
fn compiler_assigned_reflection_has_unique_physical_bindings_per_stage() -> anyhow::Result<()> {
    for (name, bytes, _) in ARTIFACTS {
        let artifact = Artifact::decode(bytes).with_context(|| format!("decode {name}"))?;
        let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata)
            .with_context(|| format!("decode {name} reflection metadata"))?;
        let reflections = metadata["reflections"]
            .as_array()
            .context("reflection metadata has no reflections array")?;
        for reflection in reflections {
            let parameters = reflection["reflection"]["parameters"]
                .as_array()
                .with_context(|| format!("{name} reflection has no parameters array"))?;
            let mut bindings = std::collections::BTreeSet::new();
            for parameter in parameters {
                if parameter["api_kind"].is_null() {
                    continue;
                }
                let key = (
                    parameter["category"].as_str(),
                    parameter["binding_space"].as_u64(),
                    parameter["binding_index"].as_u64(),
                );
                assert!(
                    bindings.insert(key),
                    "{name}: duplicate binding {key:?}: {reflection}"
                );
            }
        }
    }
    Ok(())
}
