//! Triangle using safe ez-gfx context, resource, and frame APIs.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use shared::*;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("01_triangle", WIDTH, HEIGHT, "ez_gfx_api2")?;
    {
        let context = example.context();
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
        let indices = context.upload_indices(&[0_u32, 1, 2])?;
        let first_index = indices.range()?.0;
        let positions_heap = context.create_vertex_heap("positions")?;
        let _positions = positions_heap.upload(&positions)?;
        let shader = context.load_shader(&shader_bytes)?;

        while let Some(window_frame) = example.wait_for_next_frame()? {
            let mut frame = example.surface().begin_frame()?;
            let swapchain_target =
                frame.configure_swapchain(window_frame.size, Format::Bgra8Srgb)?;
            // Counter buffers are one-frame values: the first bound frame consumes them.
            let commands = [DrawIndexedCommand {
                index_count: 3,
                instance_count: 1,
                first_index,
                vertex_offset: 0,
                first_instance: 0,
            }];
            let indirect = example
                .context()
                .acquire_counter_buffer_from(commands.as_slice())?;
            frame.add_graphics(
                &shader,
                &indirect,
                &[],
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                &[],
            )?;
            example.handle_frame(frame, swapchain_target)?;
        }
    }
    example.close()?;
    Ok(())
}
