use super::{
    BenchmarkConfig, BenchmarkRunner, Error, FrameInput, HostSurface, NativeSurface,
    PresentedFrame, ProgramReport, Result, SceneInput, backend_config, dispatch_window_input,
    drain_bounded,
};
use ez_gfx::{
    Backend, Context, ContextOptions, Frame, Surface, SurfaceOptions, create_context,
    create_surface,
};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// Window and automation settings for one example.
#[derive(Clone, Copy, Debug)]
pub struct ExampleConfig {
    pub width: u32,
    pub height: u32,
    pub title: &'static str,
    pub frame_limit: Option<u32>,
    pub benchmark: Option<BenchmarkConfig>,
    pub(crate) backend: Backend,
    pub(crate) visible: bool,
    pub(crate) debug: bool,
    pub(crate) validation: bool,
}

/// Owns the native host, graphics lifetime, frame pacing, input, and automation.
pub struct Example<H> {
    config: ExampleConfig,
    handler: H,
    benchmark: BenchmarkRunner,
}

impl<H> Example<H> {
    pub fn new(config: ExampleConfig, handler: H) -> Self {
        Self {
            benchmark: BenchmarkRunner::new(config.benchmark),
            config,
            handler,
        }
    }
}

impl<H> Example<H> {
    /// Consumes one fully recorded frame for submission and presentation.
    pub fn handle_frame(&mut self, frame: Frame) -> Result<()> {
        frame.finish().map_err(Error::from)
    }
}

/// Creates graphics state after the native window resumes, then runs the host.
pub fn run<S, H, E>(config: ExampleConfig, setup: S) -> Result<Option<ProgramReport>>
where
    S: FnOnce(&Context, &Surface, Backend) -> std::result::Result<H, E>,
    H: FnMut(&Context, &Surface, FrameInput, &[SceneInput]) -> std::result::Result<Frame, E>,
    E: std::fmt::Display,
{
    if config.frame_limit == Some(0) {
        return Err(Error::message("frame limit must be positive"));
    }
    if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
        // SAFETY: this executes before the event loop or graphics initialization begins.
        unsafe { std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~") };
    }

    let visible = config.visible;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(config, setup, visible);
    let run_result = event_loop.run_app(&mut app).map_err(Error::from);
    app.finish(run_result)
}

struct Graphics<H> {
    context: Context,
    surface: Surface,
    example: Example<H>,
}

struct App<S, H> {
    config: ExampleConfig,
    setup: Option<S>,
    graphics: Option<Graphics<H>>,
    visible: bool,
    window: Option<Window>,
    host: Option<HostSurface>,
    width: u32,
    height: u32,
    frames: u32,
    last_frame: Instant,
    pending_input: Vec<SceneInput>,
    report: Option<ProgramReport>,
    error: Option<Error>,
}

impl<S, H> App<S, H> {
    fn new(config: ExampleConfig, setup: S, visible: bool) -> Self {
        Self {
            width: config.width,
            height: config.height,
            config,
            setup: Some(setup),
            graphics: None,
            visible,
            window: None,
            host: None,
            frames: 0,
            last_frame: Instant::now(),
            pending_input: Vec::new(),
            report: None,
            error: None,
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
        self.error = Some(error);
        event_loop.exit();
    }

    fn finish(mut self, run_result: Result<()>) -> Result<Option<ProgramReport>> {
        run_result?;
        self.error.map_or(Ok(self.report.take()), Err)
    }
}

impl<S, H, E> App<S, H>
where
    S: FnOnce(&Context, &Surface, Backend) -> std::result::Result<H, E>,
    H: FnMut(&Context, &Surface, FrameInput, &[SceneInput]) -> std::result::Result<Frame, E>,
    E: std::fmt::Display,
{
    fn initialize(&mut self, native: NativeSurface) -> Result<()> {
        let backend = backend_config(native.platform, self.config.backend);
        let context = create_context(ContextOptions {
            enable_debug: self.config.debug,
            enable_validation: self.config.validation,
            surface_platform: backend.platform,
            backend: backend.backend,
            texture_decode_workers: 0,
            adapter_selection: None,
        })?;
        let surface = create_surface(
            &context,
            SurfaceOptions {
                window: native.window,
                display: native.display,
                platform: backend.platform,
                width: self.width,
                height: self.height,
                cache_presented_snapshots: false,
            },
        )?;
        let setup = self
            .setup
            .take()
            .ok_or_else(|| Error::message("example setup already consumed"))?;
        let handler = setup(&context, &surface, backend.backend).map_err(Error::callback)?;
        self.graphics = Some(Graphics {
            context,
            surface,
            example: Example::new(self.config, handler),
        });
        Ok(())
    }

    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let terminal = self
            .config
            .frame_limit
            .is_some_and(|limit| self.frames.saturating_add(1) >= limit);
        let input = FrameInput {
            width: self.width,
            height: self.height,
            delta_seconds: self.last_frame.elapsed().as_secs_f32(),
        };
        self.last_frame = Instant::now();

        let result = (|| {
            let graphics = self
                .graphics
                .as_mut()
                .ok_or_else(|| Error::message("graphics state is unavailable"))?;
            if terminal {
                graphics.surface.set_snapshot_cache(true)?;
            }
            graphics.example.benchmark.begin_frame(self.frames);
            let frame = (graphics.example.handler)(
                &graphics.context,
                &graphics.surface,
                input,
                &self.pending_input,
            )
            .map_err(Error::callback)?;
            graphics.example.handle_frame(frame)?;
            graphics
                .example
                .benchmark
                .end_frame(self.frames.saturating_add(1));
            Ok(())
        })();
        self.pending_input.clear();
        if let Err(error) = result {
            return self.fail(event_loop, error);
        }

        self.frames = self.frames.saturating_add(1);
        if terminal {
            match self.capture() {
                Ok(report) => self.report = Some(report),
                Err(error) => self.error = Some(error),
            }
            event_loop.exit();
        } else if self.visible {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn capture(&mut self) -> Result<ProgramReport> {
        let graphics = self
            .graphics
            .as_ref()
            .ok_or_else(|| Error::message("graphics state is unavailable"))?;
        let rgba8 = graphics.context.frame_readback()?;
        let counts = drain_bounded(
            4096,
            || {
                graphics
                    .context
                    .poll_runtime_event()
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(Error::from)
            },
            || {
                graphics
                    .context
                    .poll_diagnostic()
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(Error::from)
            },
        )?;
        Ok(ProgramReport {
            frame: PresentedFrame {
                width: self.width,
                height: self.height,
                frames: self.frames,
                rgba8,
                runtime_events: counts.runtime_events,
                diagnostics: counts.diagnostics,
                dropped_observations: counts.dropped,
            },
            benchmark: graphics.example.benchmark.report(),
        })
    }
}

impl<S, H, E> ApplicationHandler for App<S, H>
where
    S: FnOnce(&Context, &Surface, Backend) -> std::result::Result<H, E>,
    H: FnMut(&Context, &Surface, FrameInput, &[SceneInput]) -> std::result::Result<Frame, E>,
    E: std::fmt::Display,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height))
            .with_visible(self.visible)
            .with_active(self.visible);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => return self.fail(event_loop, error.into()),
        };
        let host = match HostSurface::attach(&window, self.config.width, self.config.height) {
            Ok(host) => host,
            Err(error) => return self.fail(event_loop, error),
        };
        if let Err(error) = self.initialize(host.descriptor()) {
            return self.fail(event_loop, error);
        }
        self.host = Some(host);
        self.window = Some(window);
        if self.visible {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Hidden windows may never receive redraw events, so polling drives them directly.
        if !self.visible && self.window.is_some() && !event_loop.exiting() {
            self.render_frame(event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|window| window.id() != id) {
            return;
        }
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                self.width = size.width;
                self.height = size.height;
                if let Some(host) = &self.host {
                    host.resize(size.width, size.height);
                }
                if let Some(graphics) = &mut self.graphics
                    && let Err(error) = graphics.surface.resize(size.width, size.height)
                {
                    self.fail(event_loop, error.into());
                }
            }
            WindowEvent::RedrawRequested if self.visible => self.render_frame(event_loop),
            _ => {
                dispatch_window_input(&event, |input| self.pending_input.push(input));
            }
        }
    }
}
