//! Runtime integration and contract tests.

use ez_gfx_artifact::Stage;
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_runtime::shader::{RuntimeShader, ShaderRequest};

#[test]
fn every_example_artifact_selects_all_native_products() {
    let examples: [(&str, &[u8], bool); 6] = [
        (
            "triangle",
            include_bytes!("../../../examples/01_triangle/01_triangle.ezgfx"),
            false,
        ),
        (
            "cube",
            include_bytes!("../../../examples/02_textured_cube/02_textured_cube.ezgfx"),
            false,
        ),
        (
            "compute",
            include_bytes!("../../../examples/03_compute_structured/03_compute_structured.ezgfx"),
            true,
        ),
        (
            "imgui",
            include_bytes!("../../../examples/04_imgui/04_imgui.ezgfx"),
            false,
        ),
        (
            "helmet",
            include_bytes!("../../../examples/05_helmet/05_helmet.ezgfx"),
            true,
        ),
        (
            "sponza",
            include_bytes!("../../../examples/06_sponza_ktx2/06_sponza_ktx2.ezgfx"),
            true,
        ),
    ];
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        for (name, artifact, compute) in examples {
            let mut requests = vec![
                ShaderRequest::new("vertexmain", Stage::Vertex).unwrap(),
                ShaderRequest::new("fragmentmain", Stage::Fragment).unwrap(),
            ];
            if compute {
                requests.push(ShaderRequest::new("computemain", Stage::Compute).unwrap());
            }
            let shader = RuntimeShader::load(artifact, backend, SemanticProfile::V1, &requests)
                .unwrap_or_else(|error| panic!("{name} {backend:?}: {error:?}"));
            assert!(shader.graphics_pair().is_ok());
            assert_eq!(shader.compute_product().is_ok(), compute);
        }
    }
}
