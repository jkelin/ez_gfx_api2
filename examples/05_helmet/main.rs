//! Model rendering with compute-generated draws through the safe frame interface.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use glam::{Mat4, Vec3};
use shared::{math::*, mesh::*, *};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneParams {
    mvp: Mat4,
    primitive_count: u32,
    padding: [u32; 3],
}

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("05_helmet", WIDTH, HEIGHT, "ez_gfx_api2")?;

    let backend = backend_config(example.backend());

    let context = Context::new(ContextOptions {
        enable_debug: example.debug_enabled(),
        enable_validation: example.validation_enabled(),
        backend: backend.backend,
        texture_decode_workers: 0,
        adapter_selection: None,
    })?;
    let surface = context.create_surface_window(example.native_surface()?, true)?;
    example.register_observations(&context)?;
    let shader_bytes = ez_gfx_compiler::compile_shader(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/05_helmet/05_helmet.slang"
        )),
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
    let _positions = positions_heap.upload(&mesh.positions)?;
    let normals_heap = context.create_vertex_heap("normals")?;
    let _normals = normals_heap.upload(&mesh.normals)?;
    let shader = context.load_shader(&shader_bytes)?;
    let mut camera = OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0)
        .with_clip_y(shared::clip_y(backend.backend));
    let target = Vec3::ZERO;
    let mut params = SceneParams {
        mvp: Mat4::IDENTITY,
        primitive_count,
        padding: [0; 3],
    };

    while let Some(window_frame) = example.wait_for_next_frame(&surface)? {
        let mut frame = surface.begin_frame()?;
        let swapchain_target = frame.configure_swapchain(window_frame.size, Format::Bgra8Srgb)?;
        camera.handle_window_events(&window_frame.events);
        params.mvp = row_major(camera.projection(window_frame.size)? * camera.view(target)?);
        let primitives = context.acquire_buffer_from(records.as_slice())?;
        let indirect =
            context.acquire_counter_buffer::<DrawIndexedCommand>(primitive_count as usize)?;
        let params_buffer = context.acquire_value_buffer(params)?;
        frame.bind_buffer("params", &params_buffer)?;
        frame.bind_buffer("primitives", &primitives)?;
        frame.bind_buffer("draw_commands", &indirect)?;
        frame.execute_compute(&shader, [primitive_count, 1, 1])?;

        frame.execute_graphics(
            &shader,
            &indirect,
            DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
        )?;
        example.handle_frame(frame, swapchain_target)?;
    }
    Ok(())
}
