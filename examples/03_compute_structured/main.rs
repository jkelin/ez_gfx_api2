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

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("03_compute_structured", WIDTH, HEIGHT, "ez_gfx_api2")?;
    let native = example.native_surface()?;
    let backend = backend_config(native.platform, example.backend());
    let [width, height] = example.surface_size();
    let context = Context::new(ContextOptions {
        enable_debug: example.debug_enabled(),
        enable_validation: example.validation_enabled(),
        surface_platform: backend.platform,
        backend: backend.backend,
        texture_decode_workers: 0,
        adapter_selection: None,
    })?;
    let surface = context.create_surface(SurfaceOptions {
        window: native.window,
        display: native.display,
        platform: backend.platform,
        width,
        height,
        cache_presented_snapshots: true,
    })?;
    example.register_observations(&context)?;
    {
        let workspace_root = Example::workspace_root()?;
        let shader_bytes = ez_gfx_compiler::compile_shader(
            &workspace_root.join("examples/03_compute_structured/03_compute_structured.slang"),
            &[
                ez_gfx_compiler::Target::Spirv,
                ez_gfx_compiler::Target::Dxil,
                ez_gfx_compiler::Target::Metal,
            ],
            !cfg!(target_vendor = "apple"),
        )?;
        let mesh = load_geometry_glb(include_bytes!("../shared/assets/sponza.glb"))?;
        let primitive_count = u32::try_from(mesh.primitives.len())?;
        let records_size = std::mem::size_of::<BasicPrimitive>()
            .checked_mul(mesh.primitives.len())
            .ok_or_else(|| anyhow::anyhow!("primitive records size overflow"))?;
        if records_size > 16 * 1024 * 1024 {
            anyhow::bail!("primitive records exceed ABI boundary");
        }
        let index_allocation = context.upload_indices(&mesh.indices)?;
        let (first_index, _) = index_allocation.range()?;
        let records = basic_primitives(&mesh, first_index)?;
        let positions_heap = context.create_vertex_heap("positions")?;
        let _positions = positions_heap.upload(&mesh.positions)?;
        let normals_heap = context.create_vertex_heap("normals")?;
        let _normals = normals_heap.upload(&mesh.normals)?;
        let shader = context.load_shader(&shader_bytes)?;
        let mut camera = OrbitCamera::new((-30.0_f32).to_radians(), 52.0_f32.to_radians(), 2.2);
        let clip_y = shared::clip_y(backend.backend);
        let target = Vec3::new(0.0, 0.55, 0.0);
        let mut push = ScenePush {
            mvp: Mat4::IDENTITY,
            primitive_count,
            padding: [0; 3],
        };

        while let Some(window_frame) = example.wait_for_next_frame(&surface)? {
            let mut frame = surface.begin_frame()?;
            let swapchain_target =
                frame.configure_swapchain(window_frame.size, Format::Bgra8Srgb)?;
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
                    500.0,
                    clip_y,
                )? * camera.view(target)?,
            );
            // Buffers are one-frame values: the first bound frame consumes them.
            let primitives = context.acquire_buffer_from(records.as_slice())?;
            let indirect =
                context.acquire_counter_buffer::<DrawIndexedCommand>(primitive_count as usize)?;
            // Compute fills the draw commands; only the visible count is published up front.
            indirect.publish_count(primitive_count)?;
            let bindings = [
                Binding::buffer("primitives", &primitives),
                Binding::counter_buffer("draw_commands", &indirect),
            ];
            frame.add_compute(&shader, [primitive_count, 1, 1], &bindings, bytes_of(&push))?;
            frame.add_graphics(
                &shader,
                &indirect,
                &bindings,
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                bytes_of(&push),
            )?;
            example.handle_frame(frame, swapchain_target)?;
        }
    }
    drop(surface);
    context.close().map_err(|(_, error)| error)?;
    Ok(())
}
