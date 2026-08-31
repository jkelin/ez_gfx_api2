//! `ImGui` using safe ez-gfx context, resource, and frame APIs.
#![allow(
    clippy::borrow_as_ptr,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::ignored_unit_patterns,
    clippy::map_unwrap_or,
    clippy::match_overlapping_arm,
    clippy::needless_pass_by_value,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::type_complexity,
    clippy::unused_self,
    clippy::wildcard_imports,
    reason = "The inline renderer preserves fixed graphics ABI and callback contracts."
)]
mod renderer {
    use crate::shared;
    use crate::shared::{FrameInput, SceneInput, SceneKey};
    use anyhow::Context as _;
    use ez_gfx::{
        ContextHandle, DrawIndexedCommand, DynamicPipelineState, EzGfxResult, IndirectBufferHandle,
        PublicBinding, ResourceIdentity, SamplerAddressMode, SamplerFilter, ShaderHandle,
        StructuredBufferHandle, TextureConfig, TextureHandle, TextureSamplerDesc, TextureSource,
        acquire_indirect, acquire_structured, create_index_heap, destroy_index_heap,
        destroy_shader, load_shader, load_texture, release_indirect, release_structured,
        render_add_graphics, set_indirect_count, texture_binding, unload_texture, upload_indices,
        write_indirect, write_structured,
    };
    use imgui::{Condition, DrawCmd, Key, MouseButton, TextureId};

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
    struct Push {
        display_size: [f32; 2],
        vertical_sign: f32,
        padding: f32,
    }

    pub(super) struct ImGuiScene {
        imgui: imgui::Context,
        shader: ShaderHandle,
        texture: TextureHandle,
        identity_start: u32,
        indirect: IndirectBufferHandle,
        indirect_capacity: u32,
        vertices: Option<StructuredBufferHandle>,
        draw_indices: Option<StructuredBufferHandle>,
        commands: Option<StructuredBufferHandle>,
        vertex_capacity: usize,
        index_capacity: usize,
        command_capacity: usize,
        cpu_vertices: Vec<ImGuiVertex>,
        cpu_indices: Vec<u32>,
        cpu_commands: Vec<ImGuiCommand>,
        uploaded_vertices: Vec<ImGuiVertex>,
        uploaded_indices: Vec<u32>,
        uploaded_commands: Vec<ImGuiCommand>,
        uploaded_draw_counts: Vec<u32>,
        draw_counts: Vec<u32>,
        push: Push,
    }

    impl ImGuiScene {
        pub(super) fn create(context: ContextHandle) -> anyhow::Result<Self> {
            let mut imgui = imgui::Context::create();
            imgui.set_ini_filename(None);
            let identity = (0..IDENTITY_INDEX_COUNT as u32).collect::<Vec<_>>();
            let identity_bytes = shared::byte_len(&identity)?;
            let atlas = imgui.fonts().build_rgba32_texture();
            let config = TextureConfig {
                width: atlas.width,
                height: atlas.height,
                mip_count: 0,
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
                TextureSource::Rgba8 {
                    width: atlas.width,
                    height: atlas.height,
                },
                atlas.data,
                false,
                &config,
            )
            .map_err(|error| anyhow::anyhow!("{error:?}"))
            .context("load ImGui font atlas")?;
            let texture_id = match texture_binding(context, texture) {
                Ok(value) => value,
                Err(error) => {
                    unload_texture(context, texture);
                    return Err(anyhow::anyhow!("{error:?}").context("resolve ImGui font binding"));
                }
            };
            imgui.fonts().tex_id = TextureId::new(texture_id as usize);
            if let Err(error) = status(
                create_index_heap(context, identity_bytes),
                "create ImGui index heap",
            ) {
                unload_texture(context, texture);
                return Err(error);
            }
            let identity_start = match upload_indices(
                context,
                identity.len() as u32,
                shared::slice_bytes(&identity),
            ) {
                Ok(value) => value,
                Err(error) => {
                    destroy_index_heap(context);
                    unload_texture(context, texture);
                    return Err(
                        anyhow::anyhow!("{error:?}").context("upload ImGui identity indices")
                    );
                }
            };
            let indirect = match acquire_indirect(context, 256) {
                Ok(handle) => handle,
                Err(error) => {
                    destroy_index_heap(context);
                    unload_texture(context, texture);
                    return Err(
                        anyhow::anyhow!("{error:?}").context("acquire ImGui indirect draws")
                    );
                }
            };
            let shader = match load_shader(
                context,
                include_bytes!(concat!(env!("OUT_DIR"), "/04_imgui.ezgfxshader")),
            ) {
                Ok(value) => value,
                Err(error) => {
                    release_indirect(context, indirect);
                    destroy_index_heap(context);
                    unload_texture(context, texture);
                    return Err(anyhow::anyhow!("{error:?}").context("load ImGui artifact"));
                }
            };
            Ok(Self {
                imgui,
                shader,
                texture,
                identity_start,
                indirect,
                indirect_capacity: 256,
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
                uploaded_vertices: Vec::new(),
                uploaded_indices: Vec::new(),
                uploaded_commands: Vec::new(),
                uploaded_draw_counts: Vec::new(),
                push: Push {
                    display_size: [640.0, 480.0],
                    vertical_sign: if cfg!(target_vendor = "apple") {
                        -1.0
                    } else {
                        1.0
                    },
                    padding: 0.0,
                },
            })
        }

        fn rebuild_draw_data(&mut self) -> anyhow::Result<()> {
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
                anyhow::bail!("ImGui produced no drawable commands");
            }
            Ok(())
        }

        fn upload_dynamic<T: bytemuck::Pod>(
            context: ContextHandle,
            name: &str,
            values: &[T],
            buffer: &mut Option<StructuredBufferHandle>,
            capacity: &mut usize,
        ) -> anyhow::Result<Option<StructuredBufferHandle>> {
            if values.len() > *capacity {
                let replacement = acquire_structured(context, shared::byte_len(values)?)
                    .map_err(|error| anyhow::anyhow!("{error:?}"))
                    .with_context(|| format!("acquire {name}"))?;
                if let Err(error) = status(
                    write_structured(context, replacement, shared::slice_bytes(values)),
                    &format!("upload {name}"),
                ) {
                    release_structured(context, replacement);
                    return Err(error);
                }
                if let Some(previous) = buffer.replace(replacement) {
                    release_structured(context, previous);
                }
                *capacity = values.len();
                Ok(Some(replacement))
            } else {
                let handle = buffer.ok_or_else(|| anyhow::anyhow!("{name} buffer unavailable"))?;
                status(
                    write_structured(context, handle, shared::slice_bytes(values)),
                    &format!("rewrite {name}"),
                )?;
                Ok(None)
            }
        }
    }

    impl ImGuiScene {
        pub(super) fn handle_input(&mut self, input: SceneInput) {
            let io = self.imgui.io_mut();
            match input {
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
        pub(super) fn update(&mut self, frame: FrameInput) -> anyhow::Result<()> {
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
        pub(super) fn record(&mut self, context: ContextHandle) -> anyhow::Result<()> {
            if self.cpu_vertices != self.uploaded_vertices {
                Self::upload_dynamic(
                    context,
                    "imgui_vertices",
                    &self.cpu_vertices,
                    &mut self.vertices,
                    &mut self.vertex_capacity,
                )?;
                std::mem::swap(&mut self.cpu_vertices, &mut self.uploaded_vertices);
            }
            if self.cpu_indices != self.uploaded_indices {
                Self::upload_dynamic(
                    context,
                    "imgui_indices",
                    &self.cpu_indices,
                    &mut self.draw_indices,
                    &mut self.index_capacity,
                )?;
                std::mem::swap(&mut self.cpu_indices, &mut self.uploaded_indices);
            }
            if self.cpu_commands != self.uploaded_commands {
                Self::upload_dynamic(
                    context,
                    "imgui_commands",
                    &self.cpu_commands,
                    &mut self.commands,
                    &mut self.command_capacity,
                )?;
                std::mem::swap(&mut self.cpu_commands, &mut self.uploaded_commands);
            }
            if self.uploaded_commands.len() > self.indirect_capacity as usize {
                let replacement = acquire_indirect(context, self.uploaded_commands.len() as u32)
                    .map_err(|error| anyhow::anyhow!("{error:?}"))
                    .context("grow ImGui indirect draws")?;
                let previous = std::mem::replace(&mut self.indirect, replacement);
                release_indirect(context, previous);
                self.indirect_capacity = self.uploaded_commands.len() as u32;
                self.uploaded_draw_counts.clear();
            }
            if self.draw_counts != self.uploaded_draw_counts {
                for (index, count) in self.draw_counts.iter().copied().enumerate() {
                    status(
                        write_indirect(
                            context,
                            self.indirect,
                            index as u32,
                            DrawIndexedCommand {
                                index_count: count,
                                instance_count: 1,
                                first_index: self.identity_start,
                                vertex_offset: 0,
                                first_instance: index as u32,
                            },
                        ),
                        &format!("write ImGui draw {index}"),
                    )?;
                }
                status(
                    set_indirect_count(context, self.indirect, self.uploaded_commands.len() as u32),
                    "set ImGui draw count",
                )?;
                self.uploaded_draw_counts.clone_from(&self.draw_counts);
            }
            let bindings = [
                PublicBinding {
                    name: "imgui_vertices".to_owned(),
                    resource: ResourceIdentity::Structured(
                        self.vertices
                            .ok_or_else(|| anyhow::anyhow!("ImGui vertex buffer unavailable"))?,
                    ),
                },
                PublicBinding {
                    name: "imgui_indices".to_owned(),
                    resource: ResourceIdentity::Structured(
                        self.draw_indices
                            .ok_or_else(|| anyhow::anyhow!("ImGui index buffer unavailable"))?,
                    ),
                },
                PublicBinding {
                    name: "imgui_commands".to_owned(),
                    resource: ResourceIdentity::Structured(
                        self.commands
                            .ok_or_else(|| anyhow::anyhow!("ImGui command buffer unavailable"))?,
                    ),
                },
            ];
            status(
                render_add_graphics(
                    context,
                    self.shader,
                    self.indirect,
                    &bindings,
                    DynamicPipelineState::from_abi(0, 0, 0, 1).unwrap(),
                    shared::bytes_of(&self.push),
                ),
                "record ImGui graphics pipeline",
            )
        }
        pub(super) fn destroy(self, context: ContextHandle) {
            if let Some(value) = self.commands {
                release_structured(context, value);
            }
            if let Some(value) = self.draw_indices {
                release_structured(context, value);
            }
            if let Some(value) = self.vertices {
                release_structured(context, value);
            }
            release_indirect(context, self.indirect);
            destroy_index_heap(context);
            unload_texture(context, self.texture);
            destroy_shader(context, self.shader);
        }
    }

    fn status(result: EzGfxResult, operation: &str) -> anyhow::Result<()> {
        match result {
            EzGfxResult::Ok => Ok(()),
            error => Err(anyhow::anyhow!("{error:?}").context(operation.to_owned())),
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
}
#[path = "../shared/mod.rs"]
mod shared;

use anyhow::Context as _;
use ez_gfx::{
    Backend, ContextHandle, ContextOptions, EzGfxResult, SurfaceHandle, SurfaceOptions,
    SurfacePlatform, begin_render, create_context, create_surface, destroy_context,
    destroy_surface, finish_render, frame_readback, init_device, poll_diagnostic,
    poll_runtime_event, resize_surface, set_snapshot_cache, wait_idle,
};
use renderer::ImGuiScene as ExampleScene;
use shared::{
    FrameInput, LifecycleCallbacks, LifecycleConfig, NativePlatform, NativeSurface, SceneInput,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

struct Example {
    resources: Option<ExampleScene>,
    context: Option<ContextHandle>,
    surface: Option<SurfaceHandle>,
    benchmark: shared::BenchmarkRunner,
}

impl Example {
    fn new(benchmark: Option<shared::BenchmarkConfig>) -> Self {
        Self {
            resources: None,
            context: None,
            surface: None,
            benchmark: shared::BenchmarkRunner::new(benchmark),
        }
    }

    fn context(&self) -> ContextHandle {
        self.context.expect("example context is initialized")
    }

    fn surface(&self) -> SurfaceHandle {
        self.surface.expect("example surface is initialized")
    }
}

impl LifecycleCallbacks for Example {
    type Report = shared::ProgramReport;

    fn initialize(&mut self, native: NativeSurface, width: u32, height: u32) -> anyhow::Result<()> {
        let (backend, backend_name) = backend()?;
        let platform = match native.platform {
            NativePlatform::Win32 => SurfacePlatform::Win32,
            NativePlatform::MetalLayer => SurfacePlatform::MetalLayer,
        };
        let context = create_context(ContextOptions {
            enable_debug: shared::env_flag("EZ_GFX_EXAMPLE_DEBUG")?,
            enable_validation: shared::env_flag("EZ_GFX_EXAMPLE_VALIDATION")?,
            surface_platform: platform,
            backend,
        })
        .map_err(|error| anyhow::anyhow!("{error:?}"))
        .with_context(|| format!("create {backend_name} context"))?;
        self.context = Some(context);
        let surface = create_surface(
            context,
            SurfaceOptions {
                window: native.window,
                display: native.display,
                platform,
                width,
                height,
                cache_presented_snapshots: false,
            },
        )
        .map_err(|error| anyhow::anyhow!("{error:?}"))
        .with_context(|| format!("create {backend_name} surface"))?;
        self.surface = Some(surface);
        status(
            init_device(self.context(), self.surface()),
            &format!("initialize {backend_name} surface device"),
        )?;
        status(
            resize_surface(self.context(), self.surface(), width, height),
            &format!("initialize {backend_name} swapchain"),
        )?;
        self.resources = Some(ExampleScene::create(self.context())?);
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        status(
            resize_surface(self.context(), self.surface(), width, height),
            "resize presented surface",
        )
    }

    fn input(&mut self, input: SceneInput) {
        if let Some(resources) = &mut self.resources {
            resources.handle_input(input);
        }
    }

    fn render(
        &mut self,
        frame: FrameInput,
        terminal: bool,
        frame_index: u32,
    ) -> anyhow::Result<()> {
        self.benchmark.begin_frame(frame_index);
        let context = self.context();
        let surface = self.surface();
        if terminal {
            status(
                set_snapshot_cache(context, surface, true),
                "enable terminal snapshot cache",
            )?;
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("example resources are unavailable"))?;
        resources.update(frame)?;
        status(begin_render(context, surface), "begin presented frame")?;
        resources.record(context)?;
        status(finish_render(context), "submit and present example")?;
        self.benchmark.end_frame(frame_index.saturating_add(1));
        Ok(())
    }

    fn capture(
        &mut self,
        width: u32,
        height: u32,
        frames: u32,
    ) -> anyhow::Result<shared::ProgramReport> {
        let rgba8 = frame_readback(self.context())
            .map_err(|error| anyhow::anyhow!("{error:?}"))
            .context("read presented snapshot")?;
        let counts = shared::drain_bounded(
            4096,
            || {
                poll_runtime_event(self.context())
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(|error| anyhow::anyhow!("{error:?}"))
                    .context("poll runtime event")
            },
            || {
                poll_diagnostic(self.context())
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(|error| anyhow::anyhow!("{error:?}"))
                    .context("poll diagnostic")
            },
        )?;
        Ok(shared::ProgramReport {
            frame: shared::PresentedFrame {
                width,
                height,
                frames,
                rgba8,
                runtime_events: counts.runtime_events,
                diagnostics: counts.diagnostics,
                dropped_observations: counts.dropped,
            },
            benchmark: self.benchmark.report(),
        })
    }

    fn shutdown(&mut self) {
        let Some(context) = self.context.take() else {
            return;
        };
        let _ = wait_idle(context);
        if let Some(resources) = self.resources.take() {
            resources.destroy(context);
        }
        if let Some(surface) = self.surface.take() {
            destroy_surface(context, surface);
        }
        destroy_context(context);
    }
}

fn run_example_with_benchmark(
    frame_limit: Option<u32>,
    benchmark: Option<shared::BenchmarkConfig>,
) -> anyhow::Result<Option<shared::ProgramReport>> {
    shared::run(
        LifecycleConfig {
            width: WIDTH,
            height: HEIGHT,
            title: "ez_gfx_api2",
            frame_limit,
        },
        Example::new(benchmark),
    )
}

fn backend() -> anyhow::Result<(Backend, &'static str)> {
    match std::env::var("EZ_GFX_BACKEND").ok().as_deref() {
        #[cfg(target_vendor = "apple")]
        None | Some("metal") => Ok((Backend::Metal, "Metal")),
        #[cfg(not(target_vendor = "apple"))]
        None | Some("vulkan") => Ok((Backend::Vulkan, "Vulkan")),
        #[cfg(windows)]
        Some("dx12") => Ok((Backend::Dx12, "DX12")),
        Some(value) => Err(anyhow::anyhow!("unsupported EZ_GFX_BACKEND `{value}`")),
    }
}

fn status(result: EzGfxResult, operation: &str) -> anyhow::Result<()> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(anyhow::anyhow!("{error:?}").context(operation.to_owned())),
    }
}

fn main() {
    let backend = std::env::var("EZ_GFX_BACKEND").unwrap_or_else(|_| {
        if cfg!(target_vendor = "apple") {
            "metal".to_owned()
        } else {
            "vulkan".to_owned()
        }
    });
    shared::run_program("04_imgui", &backend, run_example_with_benchmark);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_selection_defaults_and_validates() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(backend(), Ok((Backend::Metal, "Metal")));
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(backend().unwrap(), (Backend::Vulkan, "Vulkan"));
    }
}
