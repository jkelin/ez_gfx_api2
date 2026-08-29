mod host;
mod scenes;

use std::time::Instant;

use crate::host::HostSurface;
use crate::scenes::{FrameInput, ImGuiScene, SceneInput, SceneKey, SceneState};

use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxDiagnostic, EzGfxResult, EzGfxRuntimeRecord, ez_gfx_begin_render,
    ez_gfx_context_create_backend, ez_gfx_context_destroy, ez_gfx_context_init_device,
    ez_gfx_context_wait_idle, ez_gfx_finish_render, ez_gfx_frame_readback, ez_gfx_poll_diagnostic,
    ez_gfx_poll_runtime_event, ez_gfx_surface_create, ez_gfx_surface_destroy,
    ez_gfx_surface_resize,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event::{ElementState, MouseButton, MouseScrollDelta},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
/// Missing input preserves Vulkan; only implemented native backend names are accepted.
/// Selects the host-compatible backend; Metal is the only native macOS presentation path.
fn parse_backend(value: Option<&str>) -> Result<(u8, &'static str), String> {
    match value {
        #[cfg(target_vendor = "apple")]
        None | Some("metal") => Ok((3, "Metal")),
        #[cfg(not(target_vendor = "apple"))]
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
pub fn run_example(frame_limit: Option<u32>) -> Result<Option<PresentedReport>, String> {
    if frame_limit == Some(0) {
        return Err("frame limit must be positive".to_owned());
    }
    if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
        // SAFETY: this runs on the main thread before Vulkan or the event loop starts.
        unsafe { std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~") };
    }
    let event_loop = EventLoop::new().map_err(|error| error.to_string())?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(frame_limit);
    event_loop
        .run_app(&mut app)
        .map_err(|error| error.to_string())?;
    app.cleanup();
    if let Some(error) = app.error {
        return Err(error);
    }
    Ok(app.report)
}

struct App {
    frame_limit: Option<u32>,
    window: Option<Window>,
    host: Option<HostSurface>,
    resources: Option<ImGuiScene>,
    context: u64,
    surface: u64,
    width: u32,
    height: u32,
    frames: u32,
    last_frame: Instant,
    report: Option<PresentedReport>,
    error: Option<String>,
}

impl App {
    // Before resumed there are no native resources; cleanup therefore remains idempotent.
    fn new(frame_limit: Option<u32>) -> Self {
        Self {
            host: None,
            frame_limit,
            window: None,
            context: 0,
            surface: 0,
            resources: None,
            frames: 0,
            last_frame: Instant::now(),
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
            Box::new(resources).destroy(self.context);
        }
        if self.surface != 0 {
            ez_gfx_surface_destroy(self.surface, self.context);
            self.surface = 0;
        }
        ez_gfx_context_destroy(self.context);
        self.context = 0;
    }
}

fn scene_key(key: &Key) -> SceneKey {
    match key {
        Key::Named(NamedKey::Tab) => SceneKey::Tab,
        Key::Named(NamedKey::ArrowLeft) => SceneKey::Left,
        Key::Named(NamedKey::ArrowRight) => SceneKey::Right,
        Key::Named(NamedKey::ArrowUp) => SceneKey::Up,
        Key::Named(NamedKey::ArrowDown) => SceneKey::Down,
        Key::Named(NamedKey::PageUp) => SceneKey::PageUp,
        Key::Named(NamedKey::PageDown) => SceneKey::PageDown,
        Key::Named(NamedKey::Home) => SceneKey::Home,
        Key::Named(NamedKey::End) => SceneKey::End,
        Key::Named(NamedKey::Insert) => SceneKey::Insert,
        Key::Named(NamedKey::Delete) => SceneKey::Delete,
        Key::Named(NamedKey::Backspace) => SceneKey::Backspace,
        Key::Named(NamedKey::Space) => SceneKey::Space,
        Key::Named(NamedKey::Enter) => SceneKey::Enter,
        Key::Named(NamedKey::Escape) => SceneKey::Escape,
        _ => SceneKey::Other,
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("ez_gfx_api2".to_owned())
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => return self.fail(event_loop, error.to_string()),
        };
        let host = match HostSurface::attach(&window, WIDTH, HEIGHT) {
            Ok(host) => host,
            Err(error) => return self.fail(event_loop, error),
        };
        let (backend, backend_name) = match std::env::var("EZ_GFX_BACKEND") {
            Ok(value) => match parse_backend(Some(&value)) {
                Ok(selection) => selection,
                Err(error) => return self.fail(event_loop, error),
            },
            Err(std::env::VarError::NotPresent) => parse_backend(None).unwrap(),
            Err(error) => return self.fail(event_loop, error.to_string()),
        };
        #[cfg(target_vendor = "apple")]
        if backend != 3 {
            return self.fail(event_loop, "macOS example host requires Metal backend");
        }
        let context_desc = EzGfxBackendContextDesc {
            enable_debug: 1,
            enable_validation: 1,
            surface_platform: host.desc.platform,
            backend,
        };
        if let Err(error) = status(
            ez_gfx_context_create_backend(&context_desc, &mut self.context),
            &format!("create {backend_name} context"),
        ) {
            return self.fail(event_loop, error);
        }
        let initialized = status(
            ez_gfx_surface_create(&host.desc, &mut self.surface, self.context),
            &format!("create {backend_name} surface"),
        )
        .and_then(|_| {
            status(
                ez_gfx_context_init_device(self.surface, self.context),
                &format!("initialize {backend_name} surface device"),
            )
        })
        .and_then(|_| {
            status(
                ez_gfx_surface_resize(self.surface, WIDTH, HEIGHT, self.context),
                &format!("initialize {backend_name} swapchain"),
            )
        })
        .and_then(|_| {
            ImGuiScene::create(self.context).map(|resources| self.resources = Some(resources))
        });
        if let Err(error) = initialized {
            return self.fail(event_loop, error);
        }
        self.host = Some(host);
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
                if let Some(host) = &self.host {
                    host.resize(size.width, size.height);
                }
                if let Err(error) = status(
                    ez_gfx_surface_resize(self.surface, size.width, size.height, self.context),
                    "resize presented surface",
                ) {
                    self.fail(event_loop, error);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(resources) = &mut self.resources {
                    resources.handle_input(SceneInput::CursorMoved {
                        x: position.x,
                        y: position.y,
                    });
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(resources) = &mut self.resources {
                    resources
                        .handle_input(SceneInput::PrimaryButton(state == ElementState::Pressed));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 24.0,
                };
                if let Some(resources) = &mut self.resources {
                    resources.handle_input(SceneInput::ScrollLines(lines));
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(resources) = &mut self.resources {
                    resources.handle_input(SceneInput::Key {
                        key: scene_key(&event.logical_key),
                        pressed: event.state == ElementState::Pressed,
                    });
                    if event.state == ElementState::Pressed
                        && let Key::Character(text) = &event.logical_key
                    {
                        for character in text.chars() {
                            resources.handle_input(SceneInput::Character(character));
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let delta_seconds = self.last_frame.elapsed().as_secs_f32();
                self.last_frame = Instant::now();
                let result = self
                    .resources
                    .as_mut()
                    .ok_or_else(|| "example resources are unavailable".to_owned())
                    .and_then(|resources| {
                        resources.update(FrameInput {
                            width: self.width,
                            height: self.height,
                            delta_seconds,
                        })?;
                        status(
                            ez_gfx_begin_render(self.surface, self.context),
                            "begin presented frame",
                        )?;
                        resources.record(self.context)?;
                        status(
                            ez_gfx_finish_render(self.context),
                            "submit and present example",
                        )
                    });
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

// Every ABI error remains visible to the example caller.
fn status(result: EzGfxResult, operation: &str) -> Result<(), String> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(format!("{operation}: {error:?}")),
    }
}
fn main() {
    let frame_limit = match std::env::var("EZ_GFX_EXAMPLE_MAX_FRAMES") {
        Ok(value) => {
            let value = value.parse::<u32>().unwrap_or_else(|_| {
                eprintln!("EZ_GFX_EXAMPLE_MAX_FRAMES must be a positive integer");
                std::process::exit(2)
            });
            if value == 0 {
                eprintln!("EZ_GFX_EXAMPLE_MAX_FRAMES must be positive");
                std::process::exit(2);
            }
            Some(value)
        }
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => {
            eprintln!("EZ_GFX_EXAMPLE_MAX_FRAMES: {error}");
            std::process::exit(2);
        }
    };
    let report = run_example(frame_limit).unwrap_or_else(|error| {
        eprintln!("example failed: {error}");
        std::process::exit(1)
    });
    if let Some(report) = report {
        if let Some(path) = std::env::var_os("EZ_GFX_EXAMPLE_SNAPSHOT") {
            let update = std::env::var("EZ_GFX_UPDATE_SNAPSHOTS").ok().as_deref() == Some("1");
            if update {
                image::save_buffer_with_format(
                    &path,
                    &report.rgba8,
                    report.width,
                    report.height,
                    image::ColorType::Rgba8,
                    image::ImageFormat::Png,
                )
                .unwrap_or_else(|e| panic!("update snapshot: {e}"));
            } else {
                let expected = image::open(&path)
                    .unwrap_or_else(|e| panic!("open snapshot: {e}"))
                    .into_rgba8();
                assert_eq!(expected.dimensions(), (report.width, report.height));
                assert_eq!(expected.into_raw(), report.rgba8);
            }
        }
        if std::env::var_os("EZ_GFX_EXAMPLE_REPORT").is_some() {
            println!(
                "ez-gfx-snapshot {} {} {} {} {} {} {}",
                report.width,
                report.height,
                report.frames,
                blake3::hash(&report.rgba8),
                report.runtime_events,
                report.diagnostics,
                report.dropped_observations
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_backend;

    #[test]
    fn backend_selection_defaults_and_validates() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(parse_backend(None), Ok((3, "Metal")));
        #[cfg(target_vendor = "apple")]
        assert_eq!(parse_backend(Some("metal")), Ok((3, "Metal")));
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(parse_backend(None), Ok((1, "Vulkan")));
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(parse_backend(Some("vulkan")), Ok((1, "Vulkan")));
        #[cfg(windows)]
        assert_eq!(parse_backend(Some("dx12")), Ok((2, "DX12")));
        assert!(parse_backend(Some("unknown")).is_err());
    }
}
