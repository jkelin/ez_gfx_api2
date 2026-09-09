use super::{
    BenchmarkRunner, Error, FrameInput, HostSurface, NativeSurface, PresentedFrame, ProgramReport,
    Result, SceneInput, dispatch_window_input, publish_snapshot,
};
use ez_gfx::{Backend, Context, Event, Frame, Surface};
use std::{
    cell::RefCell,
    ffi::OsString,
    io::Write,
    path::Path,
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
    title: &'static str,
    frame_limit: Option<u32>,
    visible: bool,
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
    fn new(
        title: &'static str,
        width: u32,
        height: u32,
        frame_limit: Option<u32>,
        visible: bool,
    ) -> Self {
        Self {
            title,
            frame_limit,
            visible,
            width,
            height,
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
            .with_title(self.title)
            .with_inner_size(PhysicalSize::new(self.width, self.height))
            .with_visible(self.visible)
            .with_active(self.visible);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => return self.fail(error),
        };
        let host = match HostSurface::attach(&window, self.width, self.height) {
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

/// Owns process options, event pumping, the native host, pacing, and automation.
pub struct Example {
    identity: &'static str,
    backend_name: &'static str,
    snapshot: Option<OsString>,
    update_snapshots: bool,
    report_stdout: bool,
    event_loop: Option<EventLoop<()>>,
    state: HostState,
    backend: Backend,
    debug_enabled: bool,
    validation_enabled: bool,
    observations: Rc<RefCell<Observations>>,
    benchmark: BenchmarkRunner,
    frames: u32,
    last_frame: Instant,
    report: Option<ProgramReport>,
}

impl Example {
    /// Parses process options, then creates the event loop and native window host.
    pub fn new(
        identity: &'static str,
        width: u32,
        height: u32,
        title: &'static str,
    ) -> Result<Self> {
        let options = super::program_options().unwrap_or_else(|error| super::exit_config(error));
        if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
            // SAFETY: no event loop, worker, or graphics context exists yet.
            unsafe { std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~") };
        }

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut example = Self {
            identity,
            backend_name: super::host::backend_name_for(options.backend),
            snapshot: options.snapshot,
            update_snapshots: options.update_snapshots,
            report_stdout: options.report,
            event_loop: Some(event_loop),
            state: HostState::new(title, width, height, options.frame_limit, options.visible),
            backend: options.backend,
            debug_enabled: options.debug,
            validation_enabled: options.validation,
            observations: Rc::new(RefCell::new(Observations::default())),
            benchmark: BenchmarkRunner::new(options.benchmark),
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

        example.state.descriptor()?;
        Ok(example)
    }

    /// Returns the validated native handles for explicit surface creation.
    pub fn native_surface(&self) -> Result<NativeSurface> {
        self.state.descriptor()
    }

    /// Returns whether graphics debug behavior was requested.
    pub const fn debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    /// Returns whether graphics validation was requested.
    pub const fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// Returns the current nonzero native surface size.
    pub const fn surface_size(&self) -> [u32; 2] {
        [self.state.width, self.state.height]
    }

    /// Registers the observations used by snapshot, report, and benchmark automation.
    pub fn register_observations(&self, context: &Context) -> Result<()> {
        let callback_observations = Rc::clone(&self.observations);
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
        Ok(())
    }

    /// Returns the parent directory of the examples package.
    pub fn workspace_root() -> Result<&'static Path> {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| Error::message("examples package has no workspace parent"))
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
    pub fn wait_for_next_frame(&mut self, surface: &Surface) -> Result<Option<WindowFrame>> {
        if self.state.closed
            || self
                .state
                .frame_limit
                .is_some_and(|limit| self.frames >= limit)
        {
            return Ok(None);
        }
        if self.state.visible
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
            surface.resize(width, height)?;
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

    pub fn backend(&self) -> Backend {
        self.backend
    }

    fn publish_report(&mut self) -> Result<()> {
        // An early return has no terminal report to publish.
        let Some(report) = self.report.take() else {
            return Ok(());
        };
        let frame = report.frame;
        publish_snapshot(
            self.snapshot.take(),
            self.update_snapshots,
            frame.width,
            frame.height,
            frame.frames,
            &frame.rgba8,
            frame.runtime_events,
            frame.diagnostics,
            frame.dropped_observations,
            self.report_stdout,
        )?;
        if let Some(benchmark) = report.benchmark {
            let frame_time_ns = benchmark.elapsed_ns as f64 / f64::from(benchmark.measured_frames);
            let fps = 1_000_000_000.0 / frame_time_ns;
            writeln!(
                std::io::stdout().lock(),
                "{{\"benchmark\":\"{}\",\"backend\":\"{}\",\"warmup_frames\":{},\"measured_frames\":{},\"elapsed_ns\":{},\"frame_time_ns\":{frame_time_ns:.3},\"fps\":{fps:.3}}}",
                self.identity,
                self.backend_name,
                benchmark.warmup_frames,
                benchmark.measured_frames,
                benchmark.elapsed_ns,
            )
            .map_err(Error::ReportOutput)?;
        }
        Ok(())
    }
}

const fn should_exit_after_publication_failure(is_panicking: bool) -> bool {
    // An active unwind must preserve its original panic instead of terminating the process here.
    !is_panicking
}

impl Drop for Example {
    fn drop(&mut self) {
        if let Err(error) = self.publish_report() {
            // Drop cannot report a secondary stderr failure without risking recursive failure.
            let _ = writeln!(
                std::io::stderr().lock(),
                "example publication failed: {error}"
            );
            if should_exit_after_publication_failure(std::thread::panicking()) {
                std::process::exit(1);
            }
        }
    }
}

impl ApplicationHandler for Example {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.state.initialize_window(event_loop);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Hidden automation cannot depend on compositor redraw delivery.
        if !self.state.visible {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_failure_exit_preserves_active_unwind() {
        assert!(should_exit_after_publication_failure(false));
        assert!(!should_exit_after_publication_failure(true));
    }

    #[test]
    #[should_panic(expected = "sentinel unwind")]
    fn publication_failure_does_not_double_panic_during_unwind() {
        let directory = tempfile::tempdir().unwrap();
        let missing_snapshot = directory.path().join("missing.png");
        let observations = Rc::new(RefCell::new(Observations::default()));
        let _example = Example {
            identity: "drop_regression",
            backend_name: "vulkan",
            snapshot: Some(missing_snapshot.into_os_string()),
            update_snapshots: false,
            report_stdout: false,
            event_loop: None,
            state: HostState::new("drop regression", 1, 1, Some(1), false),
            backend: Backend::Vulkan,
            debug_enabled: false,
            validation_enabled: false,
            observations,
            benchmark: BenchmarkRunner::new(None),
            frames: 1,
            last_frame: Instant::now(),
            report: Some(ProgramReport {
                frame: PresentedFrame {
                    width: 1,
                    height: 1,
                    frames: 1,
                    rgba8: vec![0, 0, 0, 255],
                    runtime_events: 0,
                    diagnostics: 0,
                    dropped_observations: 0,
                },
                benchmark: None,
            }),
        };

        panic!("sentinel unwind");
    }
}
