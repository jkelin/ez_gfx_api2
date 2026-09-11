//! Triangle using safe ez-gfx context, resource, and frame APIs.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use shared::*;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("01_triangle", WIDTH, HEIGHT, "ez_gfx_api2")?;

    let backend = backend_config(example.backend());

    let context = Context::new(ContextOptions {
        enable_debug: example.debug_enabled(),
        enable_validation: example.validation_enabled(),
        backend: backend.backend,
        texture_decode_workers: 0,
        adapter_selection: None,
    })?;
    let surface = context.create_surface_window(example.native_surface()?, false)?;
    example.register_observations(&context)?;
    let compiled_shader = ez_gfx_compiler::EasyGraphicsCompiler::compile_shader(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/01_triangle/01_triangle.slang"
        )),
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
    let vertex_shader = compiled_shader.load_vertex_shader(&context, "vertexmain")?;
    let fragment_shader = compiled_shader.load_fragment_shader(&context, "fragmentmain")?;

    while let Some(window_frame) = example.wait_for_next_frame(&context, &surface)? {
        let mut frame = surface.begin_frame()?;
        let swapchain_target = frame.configure_swapchain(
            window_frame.size,
            Format::Bgra8Srgb,
            ez_gfx::PresentationMode::Immediate,
        )?;
        // Counter buffers are one-frame values: the first bound frame consumes them.
        let commands = [DrawIndexedCommand {
            index_count: 3,
            instance_count: 1,
            first_index,
            vertex_offset: 0,
            first_instance: 0,
        }];
        let indirect = context.acquire_counter_buffer_from(commands.as_slice())?;
        frame.execute_graphics(
            &vertex_shader,
            &fragment_shader,
            &indirect,
            DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
        )?;
        example.handle_frame(&context, frame, swapchain_target)?;
    }
    Ok(())
}
