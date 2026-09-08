//! Runtime in-memory shader compilation contract tests.

use ez_gfx_artifact::{Artifact, Stage, Target};
use ez_gfx_compiler::compile_shader;
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_runtime::shader::{RuntimeShader, ShaderLoadError};
use std::sync::LazyLock;

struct ArtifactCase {
    name: &'static str,
    bytes: Vec<u8>,
    has_compute: bool,
}

static ARTIFACTS: LazyLock<Result<Vec<ArtifactCase>, String>> =
    LazyLock::new(|| compile_artifacts().map_err(|error| format!("{error:#}")));

fn artifacts() -> anyhow::Result<&'static [ArtifactCase]> {
    ARTIFACTS
        .as_deref()
        .map_err(|error| anyhow::anyhow!("{error}"))
}

fn compile_artifacts() -> anyhow::Result<Vec<ArtifactCase>> {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow::anyhow!("examples package has no workspace parent"))?;
    let shaders: [(&str, &str, bool); 6] = [
        ("triangle", "examples/01_triangle/01_triangle.slang", false),
        (
            "cube",
            "examples/02_textured_cube/02_textured_cube.slang",
            false,
        ),
        (
            "compute",
            "examples/03_compute_structured/03_compute_structured.slang",
            true,
        ),
        ("imgui", "examples/04_imgui/04_imgui.slang", false),
        ("helmet", "examples/05_helmet/05_helmet.slang", true),
        (
            "sponza",
            "examples/06_sponza_ktx2/06_sponza_ktx2.slang",
            true,
        ),
    ];

    shaders
        .into_iter()
        .map(|(name, source, has_compute)| {
            let bytes = compile_shader(
                &workspace_root.join(source),
                &[
                    ez_gfx_compiler::Target::Spirv,
                    ez_gfx_compiler::Target::Dxil,
                    ez_gfx_compiler::Target::Metal,
                ],
                !cfg!(target_vendor = "apple"),
            )?;
            Ok(ArtifactCase {
                name,
                bytes,
                has_compute,
            })
        })
        .collect()
}

#[test]
fn runtime_compiled_artifacts_select_every_stage_without_entry_requests() -> anyhow::Result<()> {
    for case in artifacts()? {
        let (name, bytes, has_compute) = (case.name, case.bytes.as_slice(), case.has_compute);
        for backend in [Backend::Vulkan, Backend::Dx12] {
            let shader = RuntimeShader::load(bytes, backend, SemanticProfile::V1)
                .map_err(|error| anyhow::anyhow!("{error:?}"))?;
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
fn runtime_compiled_metal_product_matches_the_host() -> anyhow::Result<()> {
    for case in artifacts()? {
        let (name, bytes) = (case.name, case.bytes.as_slice());
        let artifact = Artifact::decode(bytes)?;
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
fn runtime_compiler_reflection_has_unique_physical_bindings_per_stage() -> anyhow::Result<()> {
    for case in artifacts()? {
        let (name, bytes) = (case.name, case.bytes.as_slice());
        let artifact = Artifact::decode(bytes)?;
        let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata)?;
        let reflections = metadata["reflections"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("{name}: reflections must be an array"))?;
        for reflection in reflections {
            let parameters = reflection["reflection"]["parameters"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("{name}: parameters must be an array"))?;
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

#[test]
fn sponza_fragment_marks_bindless_texture_and_sampler_selection_nonuniform() -> anyhow::Result<()> {
    const OP_DECORATE: u16 = 71;
    const NON_UNIFORM: u32 = 5300;

    let case = artifacts()?
        .iter()
        .find(|case| case.name == "sponza")
        .ok_or_else(|| anyhow::anyhow!("missing Sponza artifact"))?;
    let artifact = Artifact::decode(&case.bytes)?;
    let variant = artifact
        .variants
        .iter()
        .find(|variant| variant.target == Target::Spirv && variant.stage == Stage::Fragment)
        .ok_or_else(|| anyhow::anyhow!("missing Sponza fragment SPIR-V"))?;
    let bytes = variant.bytes.as_slice();
    assert!(bytes.len().is_multiple_of(4), "SPIR-V must contain words");

    let words = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte SPIR-V word")))
        .collect::<Vec<_>>();
    assert_eq!(words.first().copied(), Some(0x0723_0203));

    let mut decorations = 0;
    let mut offset = 5;
    while offset < words.len() {
        let word_count = (words[offset] >> 16) as usize;
        let opcode = words[offset] as u16;
        assert!(word_count != 0, "SPIR-V instruction must advance");
        if opcode == OP_DECORATE && word_count >= 3 && words[offset + 2] == NON_UNIFORM {
            decorations += 1;
        }
        offset += word_count;
    }

    assert_eq!(offset, words.len(), "SPIR-V instructions must be complete");
    assert!(
        decorations >= 2,
        "texture and sampler descriptor selections must remain nonuniform"
    );
    Ok(())
}

#[test]
fn sponza_vertex_reflection_preserves_portable_primitive_identity() -> anyhow::Result<()> {
    let case = artifacts()?
        .iter()
        .find(|case| case.name == "sponza")
        .ok_or_else(|| anyhow::anyhow!("missing Sponza artifact"))?;
    let artifact = Artifact::decode(&case.bytes)?;
    let metadata: serde_json::Value = serde_json::from_slice(&artifact.metadata)?;
    let reflections = metadata["reflections"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Sponza reflections must be an array"))?;

    let metal_target = if cfg!(target_vendor = "apple") {
        "Metallib"
    } else {
        "Msl"
    };
    for target in ["Spirv", "Dxil", metal_target] {
        let vertex = reflections
            .iter()
            .find(|reflection| reflection["target"] == target && reflection["stage"] == "Vertex")
            .ok_or_else(|| anyhow::anyhow!("missing Sponza {target} vertex reflection"))?;
        let parameters = vertex["reflection"]["parameters"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Sponza {target} parameters must be an array"))?;
        assert_eq!(
            parameters
                .iter()
                .filter(|parameter| parameter["semantic_name"] == "primitive_ids")
                .count(),
            1,
            "{target} vertex product must consume one primitive identity buffer"
        );
    }
    Ok(())
}
