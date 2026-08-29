use std::ffi::CString;

use ez_gfx_ffi::{EzGfxDrawIndexedCommand, EzGfxDynamicState};
use imgui::{Condition, DrawCmd, Key, MouseButton, TextureId};

use super::{FrameInput, SceneInput, SceneKey, SceneState, common::*};

const IDENTITY_INDEX_COUNT: usize = 65_536;

#[repr(C)]
#[derive(Clone, Copy)]
struct ImGuiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    col: u32,
    padding: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ImGuiCommand {
    clip_rect: [f32; 4],
    texture_id: u32,
    idx_offset: u32,
    vtx_offset: u32,
    padding: u32,
}
#[repr(C)]
struct Push {
    display_size: [f32; 2],
    vertical_sign: f32,
    padding: f32,
}

pub struct ImGuiScene {
    imgui: imgui::Context,
    shader: Shader,
    texture: Texture,
    indices: IndexHeap,
    identity_start: u32,
    indirect: Indirect,
    vertices: Option<Structured>,
    draw_indices: Option<Structured>,
    commands: Option<Structured>,
    vertex_capacity: usize,
    index_capacity: usize,
    command_capacity: usize,
    cpu_vertices: Vec<ImGuiVertex>,
    cpu_indices: Vec<u32>,
    cpu_commands: Vec<ImGuiCommand>,
    draw_counts: Vec<u32>,
    push: Push,
    vertices_name: CString,
    indices_name: CString,
    commands_name: CString,
}

impl ImGuiScene {
    pub fn create(context: u64) -> Result<Self, String> {
        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        let atlas = imgui.fonts().build_rgba32_texture();
        let texture = Texture::load(
            context,
            atlas.data,
            1,
            atlas.width,
            atlas.height,
            false,
            1.0,
            false,
            "ImGui font atlas",
        )?;
        imgui.fonts().tex_id = TextureId::new(texture.binding as usize);
        let identity = (0..IDENTITY_INDEX_COUNT as u32).collect::<Vec<_>>();
        let (indices, identity_start) =
            IndexHeap::upload(context, &identity, "ImGui identity indices")?;
        Ok(Self {
            imgui,
            shader: Shader::load(context, include_bytes!("../04_imgui.ezgfx"), false)?,
            texture,
            indices,
            identity_start,
            indirect: Indirect::create(context, 256, "ImGui indirect draws")?,
            vertices: None,
            draw_indices: None,
            commands: None,
            vertex_capacity: 0,
            index_capacity: 0,
            command_capacity: 0,
            cpu_vertices: Vec::new(),
            cpu_indices: Vec::new(),
            cpu_commands: Vec::new(),
            draw_counts: Vec::new(),
            push: Push {
                display_size: [640.0, 480.0],
                vertical_sign: if cfg!(target_vendor = "apple") {
                    -1.0
                } else {
                    1.0
                },
                padding: 0.0,
            },
            vertices_name: CString::new("imgui_vertices").unwrap(),
            indices_name: CString::new("imgui_indices").unwrap(),
            commands_name: CString::new("imgui_commands").unwrap(),
        })
    }

    fn rebuild_draw_data(&mut self) -> Result<(), String> {
        self.cpu_vertices.clear();
        self.cpu_indices.clear();
        self.cpu_commands.clear();
        self.draw_counts.clear();
        let draw = self.imgui.render();
        for list in draw.draw_lists() {
            let vertex_base = self.cpu_vertices.len() as u32;
            let index_base = self.cpu_indices.len() as u32;
            self.cpu_vertices
                .extend(list.vtx_buffer().iter().map(|vertex| ImGuiVertex {
                    pos: vertex.pos,
                    uv: vertex.uv,
                    col: u32::from(vertex.col[0])
                        | (u32::from(vertex.col[1]) << 8)
                        | (u32::from(vertex.col[2]) << 16)
                        | (u32::from(vertex.col[3]) << 24),
                    padding: [0; 3],
                }));
            self.cpu_indices
                .extend(list.idx_buffer().iter().map(|index| u32::from(*index)));
            for command in list.commands() {
                if let DrawCmd::Elements { count, cmd_params } = command {
                    self.cpu_commands.push(ImGuiCommand {
                        clip_rect: cmd_params.clip_rect,
                        texture_id: cmd_params.texture_id.id() as u32,
                        idx_offset: index_base + cmd_params.idx_offset as u32,
                        vtx_offset: vertex_base + cmd_params.vtx_offset as u32,
                        padding: 0,
                    });
                    self.draw_counts.push(count as u32);
                }
            }
        }
        if self.cpu_vertices.is_empty()
            || self.cpu_indices.is_empty()
            || self.cpu_commands.is_empty()
        {
            return Err("ImGui produced no drawable commands".to_owned());
        }
        Ok(())
    }

    fn upload_dynamic<T>(
        context: u64,
        name: &str,
        values: &[T],
        buffer: &mut Option<Structured>,
        capacity: &mut usize,
    ) -> Result<(), String> {
        if values.len() > *capacity {
            if let Some(previous) = buffer.take() {
                previous.destroy(context);
            }
            *buffer = Some(Structured::upload(context, name, values)?);
            *capacity = values.len();
        } else {
            buffer
                .as_ref()
                .ok_or_else(|| format!("{name} buffer unavailable"))?
                .rewrite(context, values)?;
        }
        Ok(())
    }
}

impl SceneState for ImGuiScene {
    fn handle_input(&mut self, input: SceneInput) {
        let io = self.imgui.io_mut();
        match input {
            SceneInput::CursorMoved { x, y } => io.add_mouse_pos_event([x as f32, y as f32]),
            SceneInput::PrimaryButton(value) => io.add_mouse_button_event(MouseButton::Left, value),
            SceneInput::ScrollLines(lines) => io.add_mouse_wheel_event([0.0, lines]),
            SceneInput::Character(value) => io.add_input_character(value),
            SceneInput::Key { key, pressed } => {
                if let Some(key) = imgui_key(key) {
                    io.add_key_event(key, pressed);
                }
            }
        }
    }

    fn update(&mut self, frame: FrameInput) -> Result<(), String> {
        let display_size = [frame.width as f32, frame.height as f32];
        {
            let io = self.imgui.io_mut();
            io.display_size = display_size;
            io.delta_time = frame.delta_seconds.max(1.0 / 1000.0);
        }
        let ui = self.imgui.frame();
        ui.window("Dear ImGui Demo")
            .position([20.0, 20.0], Condition::Always)
            .size([550.0, 440.0], Condition::Always)
            .build(|| {});
        let mut open = true;
        ui.show_demo_window(&mut open);
        self.push.display_size = display_size;
        self.rebuild_draw_data()
    }

    fn record(&mut self, context: u64) -> Result<(), String> {
        Self::upload_dynamic(
            context,
            "imgui_vertices",
            &self.cpu_vertices,
            &mut self.vertices,
            &mut self.vertex_capacity,
        )?;
        Self::upload_dynamic(
            context,
            "imgui_indices",
            &self.cpu_indices,
            &mut self.draw_indices,
            &mut self.index_capacity,
        )?;
        Self::upload_dynamic(
            context,
            "imgui_commands",
            &self.cpu_commands,
            &mut self.commands,
            &mut self.command_capacity,
        )?;
        if self.cpu_commands.len() > self.indirect.capacity as usize {
            let replacement = Indirect::create(
                context,
                self.cpu_commands.len() as u32,
                "ImGui indirect draws",
            )?;
            let previous = std::mem::replace(&mut self.indirect, replacement);
            previous.destroy(context);
        }
        for (index, count) in self.draw_counts.iter().copied().enumerate() {
            self.indirect.write(
                context,
                index as u32,
                EzGfxDrawIndexedCommand {
                    index_count: count,
                    instance_count: 1,
                    first_index: self.identity_start,
                    vertex_offset: 0,
                    first_instance: index as u32,
                },
            )?;
        }
        self.indirect
            .set_count(context, self.cpu_commands.len() as u32)?;
        let bindings = [
            binding(&self.vertices_name, self.vertices.as_ref().unwrap().0, 0),
            binding(&self.indices_name, self.draw_indices.as_ref().unwrap().0, 0),
            binding(&self.commands_name, self.commands.as_ref().unwrap().0, 0),
        ];
        let dynamic = EzGfxDynamicState {
            cull_mode: 0,
            front_face: 0,
            primitive_type: 0,
            blend_mode: 1,
        };
        record_graphics(
            context,
            self.shader.0,
            self.indirect.handle,
            &bindings,
            Some(dynamic),
            bytes_of(&self.push),
        )
    }

    fn destroy(self: Box<Self>, context: u64) {
        if let Some(buffer) = self.commands {
            buffer.destroy(context);
        }
        if let Some(buffer) = self.draw_indices {
            buffer.destroy(context);
        }
        if let Some(buffer) = self.vertices {
            buffer.destroy(context);
        }
        self.indirect.destroy(context);
        self.indices.destroy(context);
        self.texture.destroy(context);
        self.shader.destroy(context);
    }
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
