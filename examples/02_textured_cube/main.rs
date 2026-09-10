//! Textured cube using safe ez-gfx context, resource, and frame APIs.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use glam::{Mat4, Vec3};
use shared::{math::*, *};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneParams {
    mvp: Mat4,
    texture_id: u32,
    padding: [u32; 3],
}

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("02_textured_cube", WIDTH, HEIGHT, "ez_gfx_api2")?;

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
    let shader_bytes = ez_gfx_compiler::EasyGraphicsCompiler::compile_shader(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/02_textured_cube/02_textured_cube.slang"
        )),
        &[
            ez_gfx_compiler::Target::Spirv,
            ez_gfx_compiler::Target::Dxil,
            ez_gfx_compiler::Target::Metal,
        ],
        !cfg!(target_vendor = "apple"),
    )?;
    let positions: [[f32; 4]; 24] = [
        [-1., -1., 1., 1.],
        [1., -1., 1., 1.],
        [1., 1., 1., 1.],
        [-1., 1., 1., 1.],
        [1., -1., -1., 1.],
        [-1., -1., -1., 1.],
        [-1., 1., -1., 1.],
        [1., 1., -1., 1.],
        [-1., -1., -1., 1.],
        [-1., -1., 1., 1.],
        [-1., 1., 1., 1.],
        [-1., 1., -1., 1.],
        [1., -1., 1., 1.],
        [1., -1., -1., 1.],
        [1., 1., -1., 1.],
        [1., 1., 1., 1.],
        [-1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., -1., 1.],
        [-1., 1., -1., 1.],
        [-1., -1., -1., 1.],
        [1., -1., -1., 1.],
        [1., -1., 1., 1.],
        [-1., -1., 1., 1.],
    ];
    let indices = [
        0_u32, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16,
        17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
    ];
    let index_allocation = context.upload_indices(&indices)?;
    let (first_index, _) = index_allocation.range()?;
    let positions_heap = context.create_vertex_heap("positions")?;
    let _positions_handle = positions_heap.upload(&positions)?;
    let config = TextureConfig {
        width: 0,
        height: 0,
        mip_count: 0,
        destination: ez_gfx::TextureDestination::Auto,
        sampler: TextureSamplerDesc {
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            max_anisotropy: 1.0,
            address_u: SamplerAddressMode::Clamp,
            address_v: SamplerAddressMode::Clamp,
            address_w: SamplerAddressMode::Clamp,
        },
    };
    let texture = context.load_texture(
        TextureSource::Png,
        include_bytes!("cube.png"),
        true,
        &config,
    )?;
    context.wait_idle()?;
    let texture_id = texture.binding()?;
    let vertex_shader = shader_bytes.load_vertex_shader(&context, "vertexmain")?;
    let fragment_shader = shader_bytes.load_fragment_shader(&context, "fragmentmain")?;
    let index_count = indices.len() as u32;
    let mut camera = OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0)
        .with_clip_y(shared::clip_y(backend.backend));
    let mut params = SceneParams {
        mvp: Mat4::IDENTITY,
        texture_id,
        padding: [0; 3],
    };
    while let Some(window_frame) = example.wait_for_next_frame(&surface)? {
        let mut frame = surface.begin_frame()?;
        let swapchain_target = frame.configure_swapchain(
            window_frame.size,
            Format::Bgra8Srgb,
            ez_gfx::PresentationMode::Immediate,
        )?;
        camera.handle_window_events(&window_frame.events);
        params.mvp = row_major(camera.projection(window_frame.size)? * camera.view(Vec3::ZERO)?);
        // Counter buffers are one-frame values: the first bound frame consumes them.
        let commands = [DrawIndexedCommand {
            index_count,
            instance_count: 1,
            first_index,
            vertex_offset: 0,
            first_instance: 0,
        }];
        let indirect = context.acquire_counter_buffer_from(commands.as_slice())?;
        let params_buffer = context.acquire_value_buffer(params)?;
        frame.bind_buffer("params", &params_buffer)?;
        frame.execute_graphics(
            &vertex_shader,
            &fragment_shader,
            &indirect,
            DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap(),
        )?;
        example.handle_frame(frame, swapchain_target)?;
        example.update_title(&context);
    }
    Ok(())
}
