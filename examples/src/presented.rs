use std::{ffi::CString, mem::size_of_val};

use crate::{Example, scenes::scene_data};

use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxBinding, EzGfxDiagnostic, EzGfxResult, EzGfxRuntimeRecord,
    EzGfxShaderEntry, EzGfxSurfaceDesc, EzGfxTextureDesc, ez_gfx_acquire_indirect,
    ez_gfx_begin_render, ez_gfx_context_create_backend, ez_gfx_context_destroy,
    ez_gfx_context_init_device, ez_gfx_context_wait_idle, ez_gfx_finish_render,
    ez_gfx_frame_readback, ez_gfx_index_heap_create, ez_gfx_index_heap_destroy,
    ez_gfx_indirect_release, ez_gfx_indirect_set_draw_count, ez_gfx_indirect_write_draw,
    ez_gfx_poll_diagnostic, ez_gfx_poll_runtime_event, ez_gfx_render_add_compute_pipeline,
    ez_gfx_render_add_vertex_pipeline, ez_gfx_shader_destroy, ez_gfx_shader_load_artifact,
    ez_gfx_structured_acquire, ez_gfx_structured_release, ez_gfx_structured_write,
    ez_gfx_surface_create, ez_gfx_surface_destroy, ez_gfx_surface_resize,
    ez_gfx_texture_get_binding, ez_gfx_texture_load, ez_gfx_texture_unload,
    ez_gfx_vertex_heap_create, ez_gfx_vertex_heap_destroy, ez_gfx_vertex_upload,
    ez_gfx_vertex_upload_indices,
};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
/// Missing input preserves Vulkan; only implemented native backend names are accepted.
fn parse_backend(value: Option<&str>) -> Result<(u8, &'static str), String> {
    match value {
        None | Some("vulkan") => Ok((1, "Vulkan")),
        #[cfg(windows)]
        Some("dx12") => Ok((2, "DX12")),
        Some(value) => Err(format!("unsupported EZ_GFX_BACKEND `{value}`")),
    }
}

#[derive(Debug)]
pub struct PresentedReport {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub rgba8: Vec<u8>,
    pub runtime_events: u32,
    pub diagnostics: u32,
    pub dropped_observations: u64,
}

/// The default is interactive; a positive limit produces a deterministic automation run.
pub fn run_example(
    example: Example,
    frame_limit: Option<u32>,
) -> Result<Option<PresentedReport>, String> {
    if frame_limit == Some(0) {
        return Err("frame limit must be positive".to_owned());
    }
    if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
        // SAFETY: this runs on the main thread before Vulkan or the event loop starts.
        unsafe { std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~") };
    }
    let event_loop = EventLoop::new().map_err(|error| error.to_string())?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = TriangleApp::new(example, frame_limit);
    event_loop
        .run_app(&mut app)
        .map_err(|error| error.to_string())?;
    app.cleanup();
    if let Some(error) = app.error {
        return Err(error);
    }
    Ok(app.report)
}

struct TriangleResources {
    shader: u64,
    positions: u64,
    colors: u64,
    indirect: u64,
    texture: u64,
    texture_id: u32,
    compute: bool,
    compute_groups: u32,
    position_name: CString,
    color_name: CString,
    heap_name: CString,
}

impl TriangleResources {
    // Scene boundaries reject empty or oversized arrays before native allocation.
    fn create(context: u64, example: Example) -> Result<Self, String> {
        let data = scene_data(example)?;
        if data.positions.is_empty()
            || data.positions.len() != data.colors.len()
            || data.indices.is_empty()
            || data.commands.is_empty()
        {
            return Err("scene produced incomplete geometry".to_owned());
        }
        let position_count = u32::try_from(data.positions.len())
            .map_err(|_| "position count exceeds ABI".to_owned())?;
        let color_count =
            u32::try_from(data.colors.len()).map_err(|_| "color count exceeds ABI".to_owned())?;
        let index_count =
            u32::try_from(data.indices.len()).map_err(|_| "index count exceeds ABI".to_owned())?;
        let command_count =
            u32::try_from(data.commands.len()).map_err(|_| "draw count exceeds ABI".to_owned())?;
        let artifact = scene_artifact();
        let vertex = CString::new("vertexmain").unwrap();
        let fragment = CString::new("fragmentmain").unwrap();
        let compute_entry = CString::new("computemain").unwrap();
        let mut entries = vec![
            EzGfxShaderEntry {
                entry: vertex.as_ptr(),
                stage: 1,
                _padding: [0; 7],
            },
            EzGfxShaderEntry {
                entry: fragment.as_ptr(),
                stage: 2,
                _padding: [0; 7],
            },
        ];
        if data.compute {
            entries.push(EzGfxShaderEntry {
                entry: compute_entry.as_ptr(),
                stage: 3,
                _padding: [0; 7],
            });
        }
        let mut shader = 0;
        status(
            ez_gfx_shader_load_artifact(
                artifact.as_ptr(),
                artifact.len(),
                entries.as_ptr(),
                entries.len(),
                &mut shader,
                context,
            ),
            "load example shader",
        )?;

        let position_name = CString::new("positions").unwrap();
        let color_name = CString::new("colors").unwrap();
        let heap_name = CString::new(format!("{example:?}-positions")).unwrap();
        let mut positions = 0;
        let mut colors = 0;
        status(
            ez_gfx_structured_acquire(
                position_count,
                16,
                position_name.as_ptr(),
                &mut positions,
                context,
            ),
            "acquire shader positions",
        )?;
        status(
            ez_gfx_structured_acquire(color_count, 16, color_name.as_ptr(), &mut colors, context),
            "acquire shader colors",
        )?;
        status(
            ez_gfx_structured_write(
                positions,
                data.positions.as_ptr().cast(),
                size_of_val(data.positions.as_slice()) as u64,
                context,
            ),
            "upload shader positions",
        )?;
        status(
            ez_gfx_structured_write(
                colors,
                data.colors.as_ptr().cast(),
                size_of_val(data.colors.as_slice()) as u64,
                context,
            ),
            "upload shader colors",
        )?;

        status(
            ez_gfx_vertex_heap_create(
                heap_name.as_ptr(),
                size_of_val(data.positions.as_slice()) as u64,
                16,
                context,
            ),
            "create mapped vertex heap",
        )?;
        let mut first_vertex = 0;
        status(
            ez_gfx_vertex_upload(
                heap_name.as_ptr(),
                data.positions.as_ptr().cast(),
                position_count,
                16,
                &mut first_vertex,
                context,
            ),
            "upload mapped vertices",
        )?;
        status(
            ez_gfx_index_heap_create(
                size_of_val(data.indices.as_slice()) as u64,
                heap_name.as_ptr(),
                context,
            ),
            "create index heap",
        )?;
        let mut first_index = 0;
        status(
            ez_gfx_vertex_upload_indices(
                data.indices.as_ptr().cast(),
                index_count,
                &mut first_index,
                context,
            ),
            "upload scene indices",
        )?;

        let mut indirect = 0;
        status(
            ez_gfx_acquire_indirect(command_count, heap_name.as_ptr(), &mut indirect, context),
            "acquire scene indirect",
        )?;
        for (slot, source) in data.commands.iter().enumerate() {
            let mut command = *source;
            command.first_index = command
                .first_index
                .checked_add(first_index)
                .ok_or_else(|| "first index overflow".to_owned())?;
            command.vertex_offset = command
                .vertex_offset
                .checked_add(first_vertex as i32)
                .ok_or_else(|| "vertex offset overflow".to_owned())?;
            status(
                ez_gfx_indirect_write_draw(indirect, slot as u32, &command, context),
                "write scene draw",
            )?;
        }
        status(
            ez_gfx_indirect_set_draw_count(indirect, command_count, context),
            "set scene draw count",
        )?;

        let (texture, texture_id) = match data.texture {
            Some(texture) => load_texture(
                context,
                &texture.bytes,
                texture.source_format,
                texture.width,
                texture.height,
            )?,
            None => (0, 0),
        };
        status(ez_gfx_context_wait_idle(context), "complete scene uploads")?;
        Ok(Self {
            shader,
            positions,
            colors,
            indirect,
            texture,
            texture_id,
            compute: data.compute,
            compute_groups: command_count,
            position_name,
            color_name,
            heap_name,
        })
    }

    // Compute mutates the color buffer before graphics consumes the same storage in submission order.
    fn render(&self, context: u64, surface: u64) -> Result<(), String> {
        status(
            ez_gfx_begin_render(surface, context),
            "begin presented frame",
        )?;
        let position = EzGfxBinding {
            name: self.position_name.as_ptr(),
            structured: self.positions,
            indirect: 0,
            render_target: 0,
        };
        let color = EzGfxBinding {
            name: self.color_name.as_ptr(),
            structured: self.colors,
            indirect: 0,
            render_target: 0,
        };
        let bindings = [position, color];
        if self.compute {
            status(
                ez_gfx_render_add_compute_pipeline(
                    self.shader,
                    self.compute_groups,
                    1,
                    1,
                    bindings.as_ptr(),
                    bindings.len() as u32,
                    core::ptr::null(),
                    0,
                    context,
                ),
                "enqueue example compute",
            )?;
        }
        #[repr(C)]
        struct Push {
            tint: [f32; 4],
            use_texture: u32,
            texture_id: u32,
            _padding: [u32; 2],
        }
        let push = Push {
            tint: [1.0; 4],
            use_texture: u32::from(self.texture != 0),
            texture_id: self.texture_id,
            _padding: [0; 2],
        };
        status(
            ez_gfx_render_add_vertex_pipeline(
                self.shader,
                self.indirect,
                bindings.as_ptr(),
                bindings.len() as u32,
                core::ptr::null(),
                (&push as *const Push).cast(),
                std::mem::size_of::<Push>() as u32,
                context,
            ),
            "enqueue example graphics",
        )?;
        status(ez_gfx_finish_render(context), "submit and present example")
    }

    // GPU idle is established by the caller, so release order cannot race queued work.
    fn destroy(self, context: u64) {
        if self.texture != 0 {
            ez_gfx_texture_unload(self.texture, context);
        }
        ez_gfx_indirect_release(self.indirect, context);
        ez_gfx_structured_release(self.colors, context);
        ez_gfx_structured_release(self.positions, context);
        ez_gfx_vertex_heap_destroy(self.heap_name.as_ptr(), context);
        ez_gfx_index_heap_destroy(context);
        ez_gfx_shader_destroy(self.shader, context);
    }
}

struct TriangleApp {
    example: Example,
    frame_limit: Option<u32>,
    window: Option<Window>,
    context: u64,
    surface: u64,
    width: u32,
    height: u32,
    resources: Option<TriangleResources>,
    frames: u32,
    report: Option<PresentedReport>,
    error: Option<String>,
}

impl TriangleApp {
    // Before resumed there are no native resources; cleanup therefore remains idempotent.
    fn new(example: Example, frame_limit: Option<u32>) -> Self {
        Self {
            example,
            frame_limit,
            window: None,
            context: 0,
            surface: 0,
            resources: None,
            frames: 0,
            width: WIDTH,
            height: HEIGHT,
            report: None,
            error: None,
        }
    }

    // OS callbacks retain errors and exit rather than unwinding across the application handler.
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: impl Into<String>) {
        self.error = Some(error.into());
        event_loop.exit();
    }

    // A two-call readback prevents writing through an undersized host buffer.
    fn capture(&self) -> Result<Vec<u8>, String> {
        let mut size = 0;
        status(
            ez_gfx_frame_readback(core::ptr::null_mut(), 0, &mut size, self.context),
            "query presented snapshot",
        )?;
        let mut bytes = vec![0; size];
        status(
            ez_gfx_frame_readback(bytes.as_mut_ptr(), bytes.len(), &mut size, self.context),
            "read presented snapshot",
        )?;
        bytes.truncate(size);
        Ok(bytes)
    }

    // Polling is bounded even if a producer misbehaves; empty queues terminate without reading uninitialized payloads.
    fn drain_observability(&self) -> Result<(u32, u32, u64), String> {
        let mut events = 0_u32;
        let mut diagnostics = 0_u32;
        let mut dropped = 0_u64;
        for _ in 0..4096 {
            let mut record = EzGfxRuntimeRecord {
                correlation_id: 0,
                resource: 0,
                backend: 0,
                phase: 0,
                status: 0,
                _padding: [0; 5],
            };
            let mut present = 0;
            let mut overflow = 0;
            status(
                ez_gfx_poll_runtime_event(&mut record, &mut present, &mut overflow, self.context),
                "poll runtime event",
            )?;
            dropped = dropped.saturating_add(overflow);
            if present == 0 {
                break;
            }
            events += 1;
        }
        for _ in 0..4096 {
            let record = EzGfxRuntimeRecord {
                correlation_id: 0,
                resource: 0,
                backend: 0,
                phase: 0,
                status: 0,
                _padding: [0; 5],
            };
            let mut diagnostic = EzGfxDiagnostic {
                record,
                level: 0,
                _padding: [0; 7],
            };
            let mut present = 0;
            let mut overflow = 0;
            status(
                ez_gfx_poll_diagnostic(&mut diagnostic, &mut present, &mut overflow, self.context),
                "poll diagnostic",
            )?;
            dropped = dropped.saturating_add(overflow);
            if present == 0 {
                break;
            }
            diagnostics += 1;
        }
        Ok((events, diagnostics, dropped))
    }

    // The externally owned window outlives surface destruction and is dropped only with this app.
    fn cleanup(&mut self) {
        if self.context == 0 {
            return;
        }
        let _ = ez_gfx_context_wait_idle(self.context);
        if let Some(resources) = self.resources.take() {
            resources.destroy(self.context);
        }
        if self.surface != 0 {
            ez_gfx_surface_destroy(self.surface, self.context);
            self.surface = 0;
        }
        ez_gfx_context_destroy(self.context);
        self.context = 0;
    }
}

impl ApplicationHandler for TriangleApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(format!("ez_gfx_api2 — {:?}", self.example))
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => return self.fail(event_loop, error.to_string()),
        };
        let raw = match window.window_handle() {
            Ok(handle) => handle.as_raw(),
            Err(error) => return self.fail(event_loop, error.to_string()),
        };
        let RawWindowHandle::Win32(handle) = raw else {
            return self.fail(event_loop, "presented example currently requires Win32");
        };
        let Some(instance) = handle.hinstance else {
            return self.fail(event_loop, "Win32 handle omitted its module instance");
        };
        let (backend, backend_name) = match std::env::var("EZ_GFX_BACKEND") {
            Ok(value) => match parse_backend(Some(&value)) {
                Ok(selection) => selection,
                Err(error) => return self.fail(event_loop, error),
            },
            Err(std::env::VarError::NotPresent) => parse_backend(None).unwrap(),
            Err(error) => return self.fail(event_loop, error.to_string()),
        };
        let context_desc = EzGfxBackendContextDesc {
            enable_debug: 1,
            enable_validation: 1,
            surface_platform: 0,
            backend,
        };
        if let Err(error) = status(
            ez_gfx_context_create_backend(&context_desc, &mut self.context),
            &format!("create {backend_name} context"),
        ) {
            return self.fail(event_loop, error);
        }
        let surface_desc = EzGfxSurfaceDesc {
            window: handle.hwnd.get() as *mut _,
            display: instance.get() as *mut _,
            platform: 0,
            width: WIDTH,
            height: HEIGHT,
            cache_presented_snapshots: 1,
        };
        let initialized = status(
            ez_gfx_surface_create(&surface_desc, &mut self.surface, self.context),
            "create Win32 Vulkan surface",
        )
        .and_then(|_| {
            status(
                ez_gfx_context_init_device(self.surface, self.context),
                "initialize Vulkan surface device",
            )
        })
        .and_then(|_| {
            status(
                ez_gfx_surface_resize(self.surface, WIDTH, HEIGHT, self.context),
                "initialize swapchain",
            )
        })
        .and_then(|_| {
            TriangleResources::create(self.context, self.example)
                .map(|resources| self.resources = Some(resources))
        });
        if let Err(error) = initialized {
            return self.fail(event_loop, error);
        }

        self.window = Some(window);
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|window| window.id() != id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                self.width = size.width;
                self.height = size.height;
                if let Err(error) = status(
                    ez_gfx_surface_resize(self.surface, size.width, size.height, self.context),
                    "resize Vulkan surface",
                ) {
                    self.fail(event_loop, error);
                }
            }
            WindowEvent::RedrawRequested => {
                let result = self
                    .resources
                    .as_ref()
                    .ok_or_else(|| "example resources are unavailable".to_owned())
                    .and_then(|resources| resources.render(self.context, self.surface));
                if let Err(error) = result {
                    return self.fail(event_loop, error);
                }
                self.frames += 1;
                if self.frame_limit.is_some_and(|limit| self.frames >= limit) {
                    match self.capture().and_then(|rgba8| {
                        self.drain_observability()
                            .map(|observations| (rgba8, observations))
                    }) {
                        Ok((rgba8, (runtime_events, diagnostics, dropped_observations))) => {
                            self.report = Some(PresentedReport {
                                width: self.width,
                                height: self.height,
                                frames: self.frames,
                                rgba8,
                                runtime_events,
                                diagnostics,
                                dropped_observations,
                            })
                        }
                        Err(error) => self.error = Some(error),
                    }
                    event_loop.exit();
                } else {
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.cleanup();
    }
}

// Build tooling produces this validated multi-target artifact; runtime examples never load Slang.
fn scene_artifact() -> &'static [u8] {
    include_bytes!("../assets/scene.ezgfx")
}

// Encoded source formats carry dimensions internally; raw atlas data supplies exact dimensions.
fn load_texture(
    context: u64,
    bytes: &[u8],
    source_format: u8,
    width: u32,
    height: u32,
) -> Result<(u64, u32), String> {
    let label = CString::new("example texture").unwrap();
    let desc = EzGfxTextureDesc {
        source_format,
        destination_format: 0,
        width,
        height,
        mip_count: 0,
        generate_mips: 1,
        min_filter: 1,
        mag_filter: 1,
        max_anisotropy: 1.0,
        address_mode_u: 0,
        address_mode_v: 0,
        address_mode_w: 0,
        debug_label: label.as_ptr(),
    };
    let mut texture = 0;
    status(
        ez_gfx_texture_load(bytes.as_ptr(), bytes.len(), &desc, &mut texture, context),
        "load example texture",
    )?;
    status(
        ez_gfx_context_wait_idle(context),
        "complete example texture upload",
    )?;
    let mut binding = 0;
    status(
        ez_gfx_texture_get_binding(texture, &mut binding, context),
        "resolve example texture binding",
    )?;
    Ok((texture, binding))
}

// Every ABI error remains visible to the example caller.
fn status(result: EzGfxResult, operation: &str) -> Result<(), String> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(format!("{operation}: {error:?}")),
    }
}
#[cfg(test)]
mod tests {
    use super::parse_backend;

    #[test]
    fn backend_selection_defaults_and_validates() {
        assert_eq!(parse_backend(None), Ok((1, "Vulkan")));
        assert_eq!(parse_backend(Some("vulkan")), Ok((1, "Vulkan")));
        #[cfg(windows)]
        assert_eq!(parse_backend(Some("dx12")), Ok((2, "DX12")));
        assert!(parse_backend(Some("unknown")).is_err());
    }
}
