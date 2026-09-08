//! Triangle using safe ez-gfx context, resource, and frame APIs.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use shared::*;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

fn run_example(config: ExampleConfig) -> shared::Result<Option<ProgramReport>> {
    run(config, move |context, _surface, _backend| {
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| anyhow::anyhow!("examples package has no workspace parent"))?;
        let shader_bytes = ez_gfx_compiler::compile_shader(
            &workspace_root.join("examples/01_triangle/01_triangle.slang"),
            &[
                ez_gfx_compiler::Target::Spirv,
                ez_gfx_compiler::Target::Dxil,
                ez_gfx_compiler::Target::Metal,
            ],
            !cfg!(target_vendor = "apple"),
        )?;
        let vertical = if cfg!(target_vendor = "apple") {
            -1.0
        } else {
            1.0
        };
        let positions = [
            [-0.5_f32, -0.5 * vertical, 0.0, 1.0],
            [0.5, -0.5 * vertical, 0.0, 1.0],
            [0.0, 0.5 * vertical, 0.0, 1.0],
        ];
        let indices = [0_u32, 1, 2];

        let indices = context.upload_indices(&indices)?;
        let first_index = indices.range()?.0;
        let positions_heap = context.create_vertex_heap("positions")?;
        let positions = positions_heap.upload(&positions)?;
        let shader = context.load_shader(&shader_bytes)?;

        Ok(move |window_frame: WindowFrame| {
            let mut frame = window_frame.frame;
            let indirect = frame.acquire_counted_buffer(1)?;
            indirect.write(
                &mut frame,
                0,
                &[DrawIndexedCommand {
                    index_count: 3,
                    instance_count: 1,
                    first_index,
                    vertex_offset: 0,
                    first_instance: 0,
                }],
            )?;
            frame.retain_vertex_allocation(&positions)?;
            frame.retain_index_allocation(&indices)?;
            frame.add_graphics(
                &shader,
                &indirect,
                &[],
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                &[],
            )?;
            Ok::<_, anyhow::Error>(frame)
        })
    })
}

fn main() {
    run_program("01_triangle", WIDTH, HEIGHT, "ez_gfx_api2", run_example);
}
