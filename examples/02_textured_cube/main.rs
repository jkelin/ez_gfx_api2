//! Textured cube using safe ez-gfx context, resource, and frame APIs.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use glam::{DVec2, Mat4, Vec3};
use shared::{input::*, math::*, *};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    mvp: Mat4,
    texture_id: u32,
    padding: [u32; 3],
}

fn run_example(config: ExampleConfig) -> shared::Result<Option<ProgramReport>> {
    run(config, move |context, _surface, backend| {
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| anyhow::anyhow!("examples package has no workspace parent"))?;
        let shader_bytes = ez_gfx_compiler::compile_shader(
            &workspace_root.join("examples/02_textured_cube/02_textured_cube.slang"),
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
            0_u32, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
        ];
        let index_bytes = byte_len(&indices)?;
        let positions_bytes = byte_len(&positions)?;
        create_index_heap(context, index_bytes)?;
        let index_allocation = upload_indices(context, &indices)?;
        let (first_index, _) = index_allocation.range()?;
        let positions_heap = create_vertex_heap(context, "positions", positions_bytes, 16)?;
        let positions_handle = upload_vertices(&positions_heap, &positions)?;
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
        let texture = load_texture(
            context,
            TextureSource::Png,
            include_bytes!("cube.png"),
            true,
            &config,
        )?;
        context.wait_idle()?;
        let texture_id = texture.binding()?;
        let shader = load_shader(context, &shader_bytes)?;
        let index_count = indices.len() as u32;
        let mut camera = OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0);
        let clip_y = shared::clip_y(backend);
        let mut push = Push {
            mvp: Mat4::IDENTITY,
            texture_id,
            padding: [0; 3],
        };

        Ok(
            move |context: &Context,
                  surface: &Surface,
                  input: FrameInput,
                  events: &[SceneInput]| {
                let mut frame = begin_frame(context, surface)?;
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
                    )? * camera.view(Vec3::ZERO)?,
                );
                let indirect = frame.acquire_indirect(1)?;
                indirect.write(
                    &mut frame,
                    0,
                    &[DrawIndexedCommand {
                        index_count,
                        instance_count: 1,
                        first_index,
                        vertex_offset: 0,
                        first_instance: 0,
                    }],
                )?;
                frame.retain_vertex_allocation(&positions_handle)?;
                frame.retain_index_allocation(&index_allocation)?;
                frame.retain_texture(&texture)?;
                frame.add_graphics(
                    &shader,
                    &indirect,
                    &[],
                    DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap(),
                    bytes_of(&push),
                )?;
                Ok::<_, anyhow::Error>(frame)
            },
        )
    })
}

fn main() {
    run_program(
        "02_textured_cube",
        WIDTH,
        HEIGHT,
        "ez_gfx_api2",
        run_example,
    );
}
