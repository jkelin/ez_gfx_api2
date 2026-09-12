//! `ImGui` using safe ez-gfx context, resource, and frame interfaces.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use imgui::{Condition, DrawCmd, Key, MouseButton, TextureId};
use shared::{input::*, *};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

const IDENTITY_INDEX_COUNT: usize = 65_536;
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ImGuiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    col: u32,
    padding: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ImGuiCommand {
    clip_rect: [f32; 4],
    texture_id: u32,
    idx_offset: u32,
    vtx_offset: u32,
    padding: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiParams {
    display_size: [f32; 2],
    vertical_sign: f32,
    padding: f32,
}

fn rebuild_draw_data(
    imgui: &mut imgui::Context,
    cpu_vertices: &mut Vec<ImGuiVertex>,
    cpu_indices: &mut Vec<u32>,
    cpu_commands: &mut Vec<ImGuiCommand>,
    draw_counts: &mut Vec<u32>,
) -> anyhow::Result<()> {
    cpu_vertices.clear();
    cpu_indices.clear();
    cpu_commands.clear();
    draw_counts.clear();
    let draw = imgui.render();
    for list in draw.draw_lists() {
        let vertex_base = cpu_vertices.len() as u32;
        let index_base = cpu_indices.len() as u32;
        cpu_vertices.extend(list.vtx_buffer().iter().map(|vertex| ImGuiVertex {
            pos: vertex.pos,
            uv: vertex.uv,
            col: u32::from(vertex.col[0])
                | (u32::from(vertex.col[1]) << 8)
                | (u32::from(vertex.col[2]) << 16)
                | (u32::from(vertex.col[3]) << 24),
            padding: [0; 3],
        }));
        cpu_indices.extend(list.idx_buffer().iter().map(|index| u32::from(*index)));
        for command in list.commands() {
            if let DrawCmd::Elements { count, cmd_params } = command {
                cpu_commands.push(ImGuiCommand {
                    clip_rect: cmd_params.clip_rect,
                    texture_id: cmd_params.texture_id.id() as u32,
                    idx_offset: index_base + cmd_params.idx_offset as u32,
                    vtx_offset: vertex_base + cmd_params.vtx_offset as u32,
                    padding: 0,
                });
                draw_counts.push(count as u32);
            }
        }
    }
    anyhow::ensure!(
        !cpu_vertices.is_empty() && !cpu_indices.is_empty() && !cpu_commands.is_empty(),
        "ImGui produced no drawable commands"
    );
    Ok(())
}

fn imgui_key(key: SceneKey) -> Option<Key> {
    Some(match key {
        SceneKey::Tab => Key::Tab,
        SceneKey::Left => Key::LeftArrow,
        SceneKey::Right => Key::RightArrow,
        SceneKey::Up => Key::UpArrow,
        SceneKey::Down => Key::DownArrow,
        SceneKey::PageUp => Key::PageUp,
        SceneKey::PageDown => Key::PageDown,
        SceneKey::Home => Key::Home,
        SceneKey::End => Key::End,
        SceneKey::Insert => Key::Insert,
        SceneKey::Delete => Key::Delete,
        SceneKey::Backspace => Key::Backspace,
        SceneKey::Space => Key::Space,
        SceneKey::Enter => Key::Enter,
        SceneKey::Escape => Key::Escape,
        SceneKey::Other => return None,
    })
}

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("04_imgui", WIDTH, HEIGHT, "ez_gfx_api2")?;

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
            "/04_imgui/04_imgui.slang"
        )),
        &[
            ez_gfx_compiler::Target::Spirv,
            ez_gfx_compiler::Target::Dxil,
            ez_gfx_compiler::Target::Metal,
        ],
        !cfg!(target_vendor = "apple"),
    )?;
    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);
    let identity = (0..IDENTITY_INDEX_COUNT as u32).collect::<Vec<_>>();
    let atlas = imgui.fonts().build_rgba32_texture();
    let config = TextureConfig {
        source: TextureSource::Rgba8 {
            width: atlas.width,
            height: atlas.height,
        },
        generate_mips: false,
        required_mips: 1,
        width: atlas.width,
        height: atlas.height,
        mip_count: 0,
        destination: ez_gfx::TextureDestination::Rgba8Unorm,
        sampler: TextureSamplerDesc {
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            max_anisotropy: 1.0,
            address_u: SamplerAddressMode::Clamp,
            address_v: SamplerAddressMode::Clamp,
            address_w: SamplerAddressMode::Clamp,
        },
    };
    // The atlas slot is immediately stable; it samples fallback until the upload publishes.
    let texture_id = context.load_texture(atlas.data, &config)?.binding()?;
    imgui.fonts().tex_id = TextureId::new(texture_id as usize);
    let identity_indices = context.upload_indices(&identity)?;
    let identity_start = identity_indices.range()?.0;
    let vertices_heap = context.create_vertex_heap("imgui_vertices")?;
    let indices_heap = context.create_vertex_heap("imgui_indices")?;
    let vertex_shader = shader_bytes.load_vertex_shader(&context, "vertexmain")?;
    let fragment_shader = shader_bytes.load_fragment_shader(&context, "fragmentmain")?;
    let mut vertices = None;
    let mut indices = None;
    let mut cpu_vertices = Vec::new();
    let mut cpu_indices = Vec::new();
    let mut cpu_commands = Vec::new();
    let mut uploaded_vertices = Vec::new();
    let mut uploaded_indices = Vec::new();
    let mut draw_counts = Vec::new();
    let mut params = UiParams {
        display_size: [640.0, 480.0],
        vertical_sign: if backend.backend == Backend::Vulkan {
            1.0
        } else {
            -1.0
        },
        padding: 0.0,
    };

    while let Some(window_frame) = example.wait_for_next_frame(&context, &surface)? {
        let input = window_frame.input;
        let events = &window_frame.events;
        for &event in events {
            let io = imgui.io_mut();
            match event {
                SceneInput::CursorMoved { x, y } => io.add_mouse_pos_event([x as f32, y as f32]),
                SceneInput::PrimaryButton(value) => {
                    io.add_mouse_button_event(MouseButton::Left, value)
                }
                SceneInput::ScrollLines(lines) => io.add_mouse_wheel_event([0.0, lines]),
                SceneInput::Character(value) => io.add_input_character(value),
                SceneInput::Key { key, pressed } => {
                    if let Some(key) = imgui_key(key) {
                        io.add_key_event(key, pressed);
                    }
                }
            }
        }
        let display_size = [input.width as f32, input.height as f32];
        {
            let io = imgui.io_mut();
            io.display_size = display_size;
            io.delta_time = input.delta_seconds.max(1.0 / 1000.0);
        }
        let ui = imgui.frame();
        ui.window("Dear ImGui Demo")
            .position([20.0, 20.0], Condition::Always)
            .size([550.0, 440.0], Condition::Always)
            .build(|| {});
        let mut open = true;
        ui.show_demo_window(&mut open);
        params.display_size = display_size;
        rebuild_draw_data(
            &mut imgui,
            &mut cpu_vertices,
            &mut cpu_indices,
            &mut cpu_commands,
            &mut draw_counts,
        )?;

        if cpu_vertices != uploaded_vertices {
            drop(vertices.take());
            vertices = Some(vertices_heap.upload(&cpu_vertices)?);
            anyhow::ensure!(
                vertices.as_ref().expect("just assigned").range()?.0 == 0,
                "dedicated ImGui vertex heap did not restart at zero"
            );
            std::mem::swap(&mut cpu_vertices, &mut uploaded_vertices);
        }
        if cpu_indices != uploaded_indices {
            drop(indices.take());
            indices = Some(indices_heap.upload(&cpu_indices)?);
            anyhow::ensure!(
                indices.as_ref().expect("just assigned").range()?.0 == 0,
                "dedicated ImGui index-data heap did not restart at zero"
            );
            std::mem::swap(&mut cpu_indices, &mut uploaded_indices);
        }
        let commands = context.acquire_buffer_from(cpu_commands.as_slice())?;
        let draws = draw_counts
            .iter()
            .copied()
            .enumerate()
            .map(|(index, count)| DrawIndexedCommand {
                index_count: count,
                instance_count: 1,
                first_index: identity_start,
                vertex_offset: 0,
                first_instance: index as u32,
            })
            .collect::<Vec<_>>();
        let indirect = context.acquire_counter_buffer_from(draws.as_slice())?;
        let params_buffer = context.acquire_value_buffer(params)?;
        let mut frame = surface.begin_frame()?;
        let swapchain_target = frame.configure_swapchain(
            window_frame.size,
            Format::Bgra8Srgb,
            ez_gfx::PresentationMode::Immediate,
        )?;
        frame.bind_buffer("params", &params_buffer)?;
        frame.bind_buffer("imgui_commands", &commands)?;
        frame.execute_graphics(
            &vertex_shader,
            &fragment_shader,
            &indirect,
            DynamicPipelineState::from_abi(0, 0, 0, 1).unwrap(),
        )?;
        example.handle_frame(&context, frame, swapchain_target)?;
    }
    Ok(())
}
