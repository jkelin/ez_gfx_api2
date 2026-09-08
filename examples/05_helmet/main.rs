//! Model rendering with compute-generated draws through the safe frame interface.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use glam::{DVec2, Mat4, Vec3};
use shared::{input::*, math::*, mesh::*, *};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScenePush {
    mvp: Mat4,
    primitive_count: u32,
    padding: [u32; 3],
}

fn run_example(config: ExampleConfig) -> shared::Result<Option<ProgramReport>> {
    run(config, move |context, _surface, backend| {
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| anyhow::anyhow!("examples package has no workspace parent"))?;
        let shader_bytes = ez_gfx_compiler::compile_shader(
            &workspace_root.join("examples/05_helmet/05_helmet.slang"),
            &[
                ez_gfx_compiler::Target::Spirv,
                ez_gfx_compiler::Target::Dxil,
                ez_gfx_compiler::Target::Metal,
            ],
            !cfg!(target_vendor = "apple"),
        )?;
        let mesh = load_geometry_glb(include_bytes!("helmet.glb"))?;
        let primitive_count = u32::try_from(mesh.primitives.len())?;
        let primitive_bytes = u64::from(primitive_count)
            .checked_mul(u64::try_from(std::mem::size_of::<BasicPrimitive>())?)
            .ok_or_else(|| anyhow::anyhow!("primitive records size overflow"))?;
        if primitive_bytes > 16 * 1024 * 1024 {
            anyhow::bail!("primitive records exceed ABI boundary");
        }
        let index_allocation = context.upload_indices(&mesh.indices)?;
        let (first_index, _) = index_allocation.range()?;
        let records = basic_primitives(&mesh, first_index)?;
        let positions_heap = context.create_vertex_heap("positions")?;
        let positions = positions_heap.upload(&mesh.positions)?;
        let normals_heap = context.create_vertex_heap("normals")?;
        let normals = normals_heap.upload(&mesh.normals)?;
        let shader = context.load_shader(&shader_bytes)?;
        let mut camera = OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0);
        let clip_y = shared::clip_y(backend);
        let target = Vec3::ZERO;
        let mut push = ScenePush {
            mvp: Mat4::IDENTITY,
            primitive_count,
            padding: [0; 3],
        };

        Ok(move |window_frame: WindowFrame| {
            let mut frame = window_frame.frame;
            let input = window_frame.input;
            let events = &window_frame.events;
            for &event in events {
                match event {
                    SceneInput::CursorMoved { x, y } => camera.cursor(DVec2::new(x, y)),
                    SceneInput::PrimaryButton(value) => camera.set_dragging(value),
                    SceneInput::ScrollLines(lines) => camera.zoom(lines),
                    _ => {}
                }
            }
            push.mvp = row_major(
                perspective(
                    60.0_f32.to_radians(),
                    input.width as f32 / input.height as f32,
                    0.1,
                    100.0,
                    clip_y,
                )? * camera.view(target)?,
            );
            let primitives = frame.acquire_buffer::<BasicPrimitive>(records.len())?;
            primitives.write(&mut frame, 0, &records)?;
            let indirect = frame.acquire_counted_buffer(primitive_count)?;
            indirect.publish_count(&mut frame, primitive_count)?;
            let bindings = [
                Binding::buffer("primitives", &primitives),
                Binding::counted_buffer("draw_commands", &indirect),
            ];
            frame.retain_vertex_allocation(&positions)?;
            frame.retain_vertex_allocation(&normals)?;
            frame.retain_index_allocation(&index_allocation)?;
            frame.add_compute(&shader, [primitive_count, 1, 1], &bindings, bytes_of(&push))?;
            frame.add_graphics(
                &shader,
                &indirect,
                &bindings,
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                bytes_of(&push),
            )?;
            Ok::<_, anyhow::Error>(frame)
        })
    })
}

fn main() {
    run_program("05_helmet", WIDTH, HEIGHT, "ez_gfx_api2", run_example);
}
