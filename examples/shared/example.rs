use super::{
    BenchmarkConfig, BenchmarkRunner, Error, FrameInput, HostSurface, NativeSurface,
    PresentedFrame, ProgramReport, Result, SceneInput, backend_config, dispatch_window_input,
};
use ez_gfx::{Backend, Context, ContextOptions, Event, Frame, Surface, SurfaceOptions};
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    platform::pump_events::{EventLoopExtPumpEvents, PumpStatus},
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

/// Host input and timing for one pending frame.
pub struct WindowFrame {
    pub size: [u32; 2],
    pub input: FrameInput,
    pub events: Vec<SceneInput>,
}

#[derive(Default)]
struct Observations {
    rgba8: Vec<u8>,
    runtime_events: u32,
    diagnostics: u32,
    dropped: u64,
}

struct HostState {
    config: ExampleConfig,
    window: Option<Window>,
    host: Option<HostSurface>,
    width: u32,
    height: u32,
    pending_resize: Option<(u32, u32)>,
    pending_input: Vec<SceneInput>,
    redraw_ready: bool,
    closed: bool,
    error: Option<Error>,
}

impl HostState {
    fn new(config: ExampleConfig) -> Self {
        Self {
            width: config.width,
            height: config.height,
            config,
            window: None,
            host: None,
            pending_resize: None,
            pending_input: Vec::new(),
            redraw_ready: false,
            closed: false,
            error: None,
        }
    }

    fn fail(&mut self, error: impl Into<Error>) {
        self.error = Some(error.into());
        self.closed = true;
    }

    fn descriptor(&self) -> Result<NativeSurface> {
        self.host
            .as_ref()
            .map(HostSurface::descriptor)
            .ok_or_else(|| Error::message("native host was not resumed"))
    }

    fn initialize_window(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height))
            .with_visible(self.config.visible)
            .with_active(self.config.visible);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => return self.fail(error),
        };
        let host = match HostSurface::attach(&window, self.config.width, self.config.height) {
            Ok(host) => host,
            Err(error) => return self.fail(error),
        };
        self.host = Some(host);
        self.window = Some(window);
    }

    fn window_event(&mut self, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|window| window.id() != id) {
            return;
        }
        match &event {
            WindowEvent::CloseRequested => self.closed = true,
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                self.width = size.width;
                self.height = size.height;
                self.pending_resize = Some((size.width, size.height));
                if let Some(host) = &self.host {
                    host.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.redraw_ready = true,
            _ => {
                dispatch_window_input(&event, |input| self.pending_input.push(input));
            }
        }
    }
}

/// Owns event pumping, the native host, graphics lifetime, pacing, and automation.
pub struct Example {
    event_loop: Option<EventLoop<()>>,
    state: HostState,
    context: Option<Context>,
    surface: Option<Surface>,
    backend: Backend,
    observations: Rc<RefCell<Observations>>,
    benchmark: BenchmarkRunner,
    frames: u32,
    last_frame: Instant,
    report: Option<ProgramReport>,
}

impl Example {
    /// Creates the native host and atomic context/surface pair.
    pub fn new(config: ExampleConfig) -> Result<Self> {
        if config.frame_limit == Some(0) {
            return Err(Error::message("frame limit must be positive"));
        }
        if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
            // SAFETY: no event loop, worker, or graphics context exists yet.
            unsafe { std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~") };
        }

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let observations = Rc::new(RefCell::new(Observations::default()));
        let mut example = Self {
            event_loop: Some(event_loop),
            state: HostState::new(config),
            context: None,
            surface: None,
            backend: config.backend,
            observations: Rc::clone(&observations),
            benchmark: BenchmarkRunner::new(config.benchmark),
            frames: 0,
            last_frame: Instant::now(),
            report: None,
        };
        while example.state.window.is_none() && !example.state.closed {
            example.pump_once()?;
        }
        if let Some(error) = example.state.error.take() {
            return Err(error);
        }

        let native = example.state.descriptor()?;
        let backend = backend_config(native.platform, config.backend);
        let context = Context::new(ContextOptions {
            enable_debug: config.debug,
            enable_validation: config.validation,
            surface_platform: backend.platform,
            backend: backend.backend,
            texture_decode_workers: 0,
            adapter_selection: None,
        })?;
        let surface = context.create_surface(SurfaceOptions {
            window: native.window,
            display: native.display,
            platform: backend.platform,
            width: example.state.width,
            height: example.state.height,
            cache_presented_snapshots: false,
        })?;
        let callback_observations = Rc::clone(&observations);
        context.register_callback(move |event| {
            let mut observations = callback_observations.borrow_mut();
            match event {
                Event::Upload(_) => {}
                Event::Runtime(_) => {
                    observations.runtime_events = observations.runtime_events.saturating_add(1);
                }
                Event::Diagnostic { .. } => {
                    observations.diagnostics = observations.diagnostics.saturating_add(1);
                }
                Event::ObservationsDropped(count) => {
                    observations.dropped = observations.dropped.saturating_add(count);
                }
                Event::Readback { bytes, .. } | Event::Snapshot(bytes) => {
                    observations.rgba8.clear_from_slice(bytes);
                }
                _ => {}
            }
        })?;
        example.context = Some(context);
        example.surface = Some(surface);
        Ok(example)
    }

    fn pump_once(&mut self) -> Result<()> {
        let mut event_loop = self
            .event_loop
            .take()
            .ok_or_else(|| Error::message("event loop is already pumping"))?;
        let status = event_loop.pump_app_events(Some(Duration::from_millis(100)), self);
        self.event_loop = Some(event_loop);
        if let PumpStatus::Exit(code) = status {
            return Err(Error::message(format!("event loop exited: {code}")));
        }
        Ok(())
    }

    /// Pumps until host input is ready, or returns `None` after completion.
    pub fn wait_for_next_frame(&mut self) -> Result<Option<WindowFrame>> {
        if self.state.closed
            || self
                .state
                .config
                .frame_limit
                .is_some_and(|limit| self.frames >= limit)
        {
            return Ok(None);
        }
        if self.state.config.visible
            && let Some(window) = &self.state.window
        {
            window.request_redraw();
        }

        self.state.redraw_ready = false;
        while !self.state.redraw_ready && !self.state.closed {
            self.pump_once()?;
        }
        if let Some(error) = self.state.error.take() {
            return Err(error);
        }
        if self.state.closed {
            return Ok(None);
        }
        if let Some((width, height)) = self.state.pending_resize.take() {
            self.surface().resize(width, height)?;
        }

        self.benchmark.begin_frame(self.frames);
        let input = FrameInput {
            width: self.state.width,
            height: self.state.height,
            delta_seconds: self.last_frame.elapsed().as_secs_f32(),
        };
        self.last_frame = Instant::now();
        Ok(Some(WindowFrame {
            size: [self.state.width, self.state.height],
            input,
            events: std::mem::take(&mut self.state.pending_input),
        }))
    }

    /// Consumes the pending frame and configures terminal swapchain readback.
    pub fn handle_frame(
        &mut self,
        mut frame: Frame,
        swapchain_target: ez_gfx::RenderTarget,
    ) -> Result<()> {
        if swapchain_target.extent()? != (self.state.width, self.state.height) {
            return Err(Error::message(
                "swapchain target extent does not match host size",
            ));
        }
        let terminal = self
            .state
            .config
            .frame_limit
            .is_some_and(|limit| self.frames.saturating_add(1) >= limit);
        let _readback = terminal
            .then(|| swapchain_target.prepare_readback(&mut frame))
            .transpose()?;
        drop(swapchain_target);
        frame.finish()?;
        self.frames = self.frames.saturating_add(1);
        self.benchmark.end_frame(self.frames);
        if self
            .state
            .config
            .frame_limit
            .is_some_and(|limit| self.frames >= limit)
        {
            let observations = self.observations.borrow();
            self.report = Some(ProgramReport {
                frame: PresentedFrame {
                    width: self.state.width,
                    height: self.state.height,
                    frames: self.frames,
                    rgba8: observations.rgba8.clone(),
                    runtime_events: observations.runtime_events,
                    diagnostics: observations.diagnostics,
                    dropped_observations: observations.dropped,
                },
                benchmark: self.benchmark.report(),
            });
        }
        Ok(())
    }

    pub fn context(&self) -> &Context {
        self.context.as_ref().expect("context exists until close")
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn surface(&self) -> &Surface {
        self.surface.as_ref().expect("surface exists until close")
    }

    /// Drops caller resources first, then deterministically tears down the surface and context.
    pub fn close(mut self) -> Result<Option<ProgramReport>> {
        self.surface.take();
        let context = self.context.take().expect("context exists until close");
        context.close().map_err(|(_, error)| error)?;
        Ok(self.report.take())
    }
}

impl ApplicationHandler for Example {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.state.initialize_window(event_loop);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Hidden automation cannot depend on compositor redraw delivery.
        if !self.state.config.visible {
            self.state.redraw_ready = true;
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        self.state.window_event(id, event);
    }
}

trait ClearFromSlice {
    fn clear_from_slice(&mut self, source: &[u8]);
}

impl ClearFromSlice for Vec<u8> {
    fn clear_from_slice(&mut self, source: &[u8]) {
        self.clear();
        self.extend_from_slice(source);
    }
}
