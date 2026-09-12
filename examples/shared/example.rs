use super::{
    BenchmarkRunner, Error, FrameInput, HostSurface, PresentedFrame, ProgramReport, Result,
    SceneInput, dispatch_window_input, publish_snapshot,
};
use ez_gfx::{Backend, Context, Event, Frame, Surface};
use std::{
    cell::RefCell,
    ffi::OsString,
    io::Write,
    rc::Rc,
    sync::Arc,
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
#[derive(Clone, Copy)]
struct FrameTiming {
    frame: u32,
    host_wait_ns: u128,
    record_ns: u128,
    submit_present_ns: u128,
}

const TITLE_REFRESH_INTERVAL: Duration = Duration::from_millis(200);

fn title_refresh_due(last: Option<Instant>, now: Instant, force: bool) -> bool {
    force || last.is_none_or(|last| now.saturating_duration_since(last) >= TITLE_REFRESH_INTERVAL)
}

/// Render-thread result consumed by the window-thread host after joining.
pub(crate) struct ThreadedRenderOutput {
    report: Option<ProgramReport>,
    frame_timings: Vec<FrameTiming>,
}

/// One lock-free telemetry publication after a completed threaded frame.
pub(crate) struct ThreadedFrameUpdate {
    pub(crate) frames: u32,
    pub(crate) fps: f32,
    pub(crate) diagnostics: Option<Option<ez_gfx::ResourceDiagnostics>>,
}

/// Sendable automation configuration moved into the graphics thread.
pub(crate) struct ThreadedRenderConfig {
    frame_limit: Option<u32>,
    benchmark: BenchmarkRunner,
    frame_timings_enabled: bool,
}

/// Render-side automation state. No window-thread acknowledgement is required.
pub(crate) struct ThreadedRenderHost {
    frame_limit: Option<u32>,
    observations: Rc<RefCell<Observations>>,
    benchmark: BenchmarkRunner,
    frame_timings_enabled: bool,
    frame_timings: Vec<FrameTiming>,
    pending_frame_started: Option<Instant>,
    frames: u32,
    last_present: Option<Instant>,
    last_diagnostics: Option<Instant>,
    fps: f32,
    report: Option<ProgramReport>,
}

fn register_observations(context: &Context, observations: Rc<RefCell<Observations>>) -> Result<()> {
    context.register_callback(move |event| {
        let mut observations = observations.borrow_mut();
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

impl ThreadedRenderConfig {
    /// Creates render-side state and registers its thread-local observations.
    pub(crate) fn start(self, context: &Context) -> Result<ThreadedRenderHost> {
        let observations = Rc::new(RefCell::new(Observations::default()));
        register_observations(context, Rc::clone(&observations))?;
        Ok(ThreadedRenderHost {
            frame_limit: self.frame_limit,
            observations,
            benchmark: self.benchmark,
            frame_timings_enabled: self.frame_timings_enabled,
            frame_timings: Vec::new(),
            pending_frame_started: None,
            frames: 0,
            last_present: None,
            last_diagnostics: None,
            fps: 0.0,
            report: None,
        })
    }
}

impl ThreadedRenderHost {
    /// Returns false after the exact configured terminal frame.
    pub(crate) fn should_render(&self) -> bool {
        self.frame_limit.is_none_or(|limit| self.frames < limit)
    }

    /// Starts render-loop timing after startup compilation and resource creation.
    pub(crate) fn begin_frame(&mut self) {
        self.benchmark.begin_frame(self.frames);
        self.pending_frame_started = Some(Instant::now());
    }

    /// Finishes, optionally captures, and publishes one render-thread frame.
    pub(crate) fn finish_frame(
        &mut self,
        context: &Context,
        mut frame: Frame,
        swapchain_target: ez_gfx::RenderTarget,
        size: [u32; 2],
    ) -> Result<ThreadedFrameUpdate> {
        let frame_started = self
            .pending_frame_started
            .take()
            .ok_or_else(|| Error::message("threaded frame timing began without a pending frame"))?;
        let record_ns = frame_started.elapsed().as_nanos();
        let submit_present_started = Instant::now();
        if swapchain_target.extent()? != (size[0], size[1]) {
            return Err(Error::message(
                "swapchain target extent does not match render-thread size",
            ));
        }
        let terminal = self
            .frame_limit
            .is_some_and(|limit| self.frames.saturating_add(1) >= limit);
        let _readback = terminal
            .then(|| swapchain_target.prepare_readback(&mut frame))
            .transpose()?;
        drop(swapchain_target);
        frame.finish()?;
        let submit_present_ns = submit_present_started.elapsed().as_nanos();
        self.frames = self.frames.saturating_add(1);
        self.benchmark.end_frame(self.frames);

        let presented_at = Instant::now();
        if let Some(previous) = self.last_present {
            self.fps = smoothed_fps(self.fps, presented_at.duration_since(previous));
        }
        self.last_present = Some(presented_at);
        if self.frame_timings_enabled {
            self.frame_timings.push(FrameTiming {
                frame: self.frames,
                host_wait_ns: 0,
                record_ns,
                submit_present_ns,
            });
        }

        let refresh = title_refresh_due(self.last_diagnostics, presented_at, terminal);
        let diagnostics = refresh.then(|| context.resource_diagnostics().ok());
        if refresh {
            self.last_diagnostics = Some(presented_at);
        }
        if terminal {
            let observations = self.observations.borrow();
            self.report = Some(ProgramReport {
                frame: PresentedFrame {
                    width: size[0],
                    height: size[1],
                    frames: self.frames,
                    rgba8: observations.rgba8.clone(),
                    runtime_events: observations.runtime_events,
                    diagnostics: observations.diagnostics,
                    dropped_observations: observations.dropped,
                },
                benchmark: self.benchmark.report(),
            });
        }
        Ok(ThreadedFrameUpdate {
            frames: self.frames,
            fps: self.fps,
            diagnostics,
        })
    }

    /// Returns completed report data after the graphics thread stops.
    pub(crate) fn finish(self) -> ThreadedRenderOutput {
        ThreadedRenderOutput {
            report: self.report,
            frame_timings: self.frame_timings,
        }
    }
}

struct HostState {
    title: &'static str,
    frame_limit: Option<u32>,
    visible: bool,
    window: Option<Arc<Window>>,
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
            // Undefined Vulkan extents need the toolkit's requested size before the first frame.
            pending_resize: Some((width, height)),
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
    fn record_resize(&mut self, size: PhysicalSize<u32>) {
        // Platform transitions may report one zero dimension. The graphics API
        // represents every unavailable drawable as 0x0.
        let (width, height) = if size.width == 0 || size.height == 0 {
            (0, 0)
        } else {
            (size.width, size.height)
        };
        self.width = width;
        self.height = height;
        self.pending_resize = Some((width, height));
        self.redraw_ready = true;
    }

    fn apply_pending_resize(
        &mut self,
        resize: impl FnOnce(u32, u32) -> ez_gfx::Result<()>,
    ) -> Result<bool> {
        let drawable_ready = self.width > 0 && self.height > 0;
        let Some((width, height)) = self.pending_resize.take() else {
            return Ok(drawable_ready);
        };
        match resize(width, height) {
            Ok(()) if drawable_ready => Ok(true),
            Err(ez_gfx::Error::NotReady) if !drawable_ready => Ok(false),
            Ok(()) => Err(Error::message(
                "minimized surface resize unexpectedly succeeded",
            )),
            Err(error) => Err(error.into()),
        }
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
            Ok(window) => Arc::new(window),
            Err(error) => return self.fail(error),
        };
        self.window = Some(window);
    }

    fn window_event(&mut self, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|window| window.id() != id) {
            return;
        }
        match &event {
            WindowEvent::CloseRequested => self.closed = true,
            WindowEvent::Resized(size) => self.record_resize(*size),
            WindowEvent::RedrawRequested if self.width > 0 && self.height > 0 => {
                self.redraw_ready = true;
            }
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
    strict_all: bool,
    snapshot: Option<OsString>,
    update_snapshots: bool,
    report_stdout: bool,
    event_loop: Option<EventLoop<()>>,
    state: HostState,
    backend: Backend,
    debug_enabled: bool,
    validation_enabled: bool,
    resize_after_first_frame: bool,
    observations: Rc<RefCell<Observations>>,
    benchmark: BenchmarkRunner,
    frame_timings_enabled: bool,
    frame_timings: Vec<FrameTiming>,
    pending_frame_started: Option<Instant>,
    pending_host_wait_ns: u128,
    frames: u32,
    last_frame: Instant,
    last_present: Option<Instant>,
    // Exponential moving average of presented frames per second for the title.
    fps: f32,
    // Reused title buffer so per-frame title updates never allocate.
    title_text: String,
    last_title_refresh: Option<Instant>,
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
            strict_all: options.strict_all,
            snapshot: options.snapshot,
            update_snapshots: options.update_snapshots,
            report_stdout: options.report,
            event_loop: Some(event_loop),
            state: HostState::new(title, width, height, options.frame_limit, options.visible),
            backend: options.backend,
            debug_enabled: options.debug,
            validation_enabled: options.validation,
            resize_after_first_frame: options.resize_after_first_frame,
            observations: Rc::new(RefCell::new(Observations::default())),
            benchmark: BenchmarkRunner::new(options.benchmark),
            frame_timings_enabled: options.frame_timings,
            frame_timings: Vec::new(),
            pending_frame_started: None,
            pending_host_wait_ns: 0,
            frames: 0,
            last_frame: Instant::now(),
            last_present: None,
            fps: 0.0,
            title_text: String::new(),
            last_title_refresh: None,
            report: None,
        };
        while example.state.window.is_none() && !example.state.closed {
            example.pump_once()?;
        }
        if let Some(error) = example.state.error.take() {
            return Err(error);
        }

        example
            .state
            .window
            .as_ref()
            .ok_or_else(|| Error::message("native host was not resumed"))?;
        // Startup-only hint; stderr keeps snapshot/benchmark stdout machine-parseable.
        super::observability::print_frame_title_legend();
        Ok(example)
    }

    /// Returns the live host window used for surface creation.
    pub fn window(&self) -> Result<&Window> {
        self.state
            .window
            .as_deref()
            .ok_or_else(|| Error::message("native host was not resumed"))
    }
    /// Returns an owned clone of the live host used for surface creation.
    pub fn native_surface(&self) -> Result<HostSurface> {
        let window = self
            .state
            .window
            .as_ref()
            .ok_or_else(|| Error::message("native host was not resumed"))?;
        Ok(HostSurface::attach(Arc::clone(window))?)
    }

    /// Returns whether graphics debug behavior was requested.
    pub const fn debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    /// Returns whether graphics validation was requested.
    pub const fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// Returns the current native drawable size, or `[0, 0]` while minimized.
    pub(crate) const fn surface_size(&self) -> [u32; 2] {
        [self.state.width, self.state.height]
    }

    /// Pumps one window-event batch without using frame completion as pacing.
    pub(crate) fn pump_window_events(&mut self) -> Result<bool> {
        if self.state.closed {
            return Ok(false);
        }
        self.pump_once()?;
        if let Some(error) = self.state.error.take() {
            return Err(error);
        }
        Ok(!self.state.closed)
    }

    /// Transfers benchmark and frame-timing ownership to a graphics thread.
    pub(crate) fn take_threaded_render_config(&mut self) -> ThreadedRenderConfig {
        ThreadedRenderConfig {
            frame_limit: self.state.frame_limit,
            benchmark: std::mem::replace(&mut self.benchmark, BenchmarkRunner::new(None)),
            frame_timings_enabled: std::mem::replace(&mut self.frame_timings_enabled, false),
        }
    }

    /// Installs joined render-thread results for ordinary drop-time publication.
    pub(crate) fn accept_threaded_render_output(&mut self, output: ThreadedRenderOutput) {
        self.report = output.report;
        self.frame_timings = output.frame_timings;
        self.frame_timings_enabled = !self.frame_timings.is_empty();
    }

    /// Registers the observations used by snapshot, report, and benchmark automation.
    pub fn register_observations(&self, context: &Context) -> Result<()> {
        register_observations(context, Rc::clone(&self.observations))
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
    pub fn wait_for_next_frame(
        &mut self,
        _context: &Context,
        surface: &Surface,
    ) -> Result<Option<WindowFrame>> {
        if self.state.closed
            || self
                .state
                .frame_limit
                .is_some_and(|limit| self.frames >= limit)
        {
            return Ok(None);
        }

        let host_wait_started = Instant::now();
        loop {
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
            if !self
                .state
                .apply_pending_resize(|width, height| surface.resize(width, height))?
            {
                continue;
            }

            self.benchmark.begin_frame(self.frames);
            let delta_seconds = self.last_frame.elapsed().as_secs_f32();
            let input = FrameInput {
                width: self.state.width,
                height: self.state.height,
                delta_seconds,
            };
            self.last_frame = Instant::now();
            self.pending_host_wait_ns = host_wait_started.elapsed().as_nanos();
            self.pending_frame_started = Some(Instant::now());
            return Ok(Some(WindowFrame {
                size: [self.state.width, self.state.height],
                input,
                events: std::mem::take(&mut self.state.pending_input),
            }));
        }
    }
    /// Refreshes the window title with externally sampled FPS and diagnostics.
    ///
    /// This is safe to call from the window thread while rendering proceeds on
    /// another thread. The title buffer reuses capacity after initial growth.
    pub(crate) fn update_title(
        &mut self,
        fps: f32,
        diagnostics: Option<&ez_gfx::ResourceDiagnostics>,
    ) {
        // A missing window can only occur during host teardown; retain the
        // previous title instead of turning telemetry into a shutdown error.
        let Some(window) = self.state.window.as_ref() else {
            return;
        };
        self.title_text.clear();
        super::observability::push_frame_title(
            &mut self.title_text,
            self.identity,
            self.backend_name,
            fps,
            diagnostics,
        );
        window.set_title(&self.title_text);
    }

    fn refresh_title(&mut self, context: &Context, force: bool) {
        let now = Instant::now();
        if !title_refresh_due(self.last_title_refresh, now, force) {
            return;
        }
        // Query and native title mutation share one 5 Hz gate; the terminal frame is forced.
        let diagnostics = context.resource_diagnostics().ok();
        self.update_title(self.fps, diagnostics.as_ref());
        self.last_title_refresh = Some(now);
    }

    /// Consumes the pending frame, presents it, then refreshes the window title.
    ///
    /// The title reflects the FPS sample of this completed presentation, so it
    /// runs only after `finish` succeeds; failed frames keep the previous title.
    pub fn handle_frame(
        &mut self,
        context: &Context,
        mut frame: Frame,
        swapchain_target: ez_gfx::RenderTarget,
    ) -> Result<()> {
        let frame_started = self
            .pending_frame_started
            .take()
            .ok_or_else(|| Error::message("frame timing began without a pending host frame"))?;
        let record_ns = frame_started.elapsed().as_nanos();
        let submit_present_started = Instant::now();
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
        let submit_present_ns = submit_present_started.elapsed().as_nanos();
        let presented_at = Instant::now();
        if let Some(previous) = self.last_present {
            self.fps = smoothed_fps(self.fps, presented_at.duration_since(previous));
        }
        self.last_present = Some(presented_at);
        self.frames = self.frames.saturating_add(1);
        // Successful presentation only: `finish` already returned, and `?`
        // above skips this on failure, so a dropped frame never paints a
        // title for a presentation that did not happen.
        self.refresh_title(context, terminal);
        if self.frame_timings_enabled {
            self.frame_timings.push(FrameTiming {
                frame: self.frames,
                host_wait_ns: self.pending_host_wait_ns,
                record_ns,
                submit_present_ns,
            });
        }
        self.benchmark.end_frame(self.frames);
        if self.resize_after_first_frame && self.frames == 1 {
            // Regression automation publishes a maximized-scale extent directly; the hidden host
            // window is never shown, activated, or asked to change state.
            self.state.record_resize(PhysicalSize::new(1920, 1080));
        }
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

    /// Submits one hidden allocation-probe frame without terminal readback or reporting.
    ///
    /// # Errors
    /// Returns an error for a visible host, missing pending frame, mismatched extent, or failed submission.
    pub fn handle_allocation_frame(
        &mut self,
        frame: Frame,
        swapchain_target: ez_gfx::RenderTarget,
    ) -> Result<()> {
        if self.state.visible {
            return Err(Error::message("allocation probe requires a hidden host"));
        }
        self.pending_frame_started
            .take()
            .ok_or_else(|| Error::message("frame timing began without a pending host frame"))?;
        if swapchain_target.extent()? != (self.state.width, self.state.height) {
            return Err(Error::message(
                "swapchain target extent does not match host size",
            ));
        }
        drop(swapchain_target);
        frame.finish()?;
        self.frames = self.frames.saturating_add(1);
        Ok(())
    }

    /// Returns whether the native host was created hidden.
    pub const fn is_hidden(&self) -> bool {
        !self.state.visible
    }

    /// Returns the stable backend name used by reports.
    pub const fn backend_name(&self) -> &'static str {
        self.backend_name
    }
    /// Returns whether the probe must require whole-window zero allocations.
    pub const fn strict_all(&self) -> bool {
        self.strict_all
    }
    /// Returns whether this run compares or updates a snapshot.
    pub const fn snapshot_enabled(&self) -> bool {
        self.snapshot.is_some()
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
        if self.frame_timings_enabled {
            let mut stdout = std::io::stdout().lock();
            for timing in &self.frame_timings {
                writeln!(
                    stdout,
                    "ez-gfx-frame-timing {} {} {} {} {} {}",
                    self.identity,
                    self.backend_name,
                    timing.frame,
                    timing.host_wait_ns,
                    timing.record_ns,
                    timing.submit_present_ns
                )
                .map_err(Error::ReportOutput)?;
            }
        }
        Ok(())
    }
}

fn smoothed_fps(previous: f32, elapsed: Duration) -> f32 {
    // Timer granularity can produce a zero interval; retain the last valid sample.
    let seconds = elapsed.as_secs_f32();
    if seconds == 0.0 {
        previous
    } else {
        let instant = 1.0 / seconds;
        if previous == 0.0 {
            instant
        } else {
            previous.mul_add(0.9, instant * 0.1)
        }
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
        // The host owns continuous pacing under ControlFlow::Poll. Redraw delivery may be
        // compositor-paced, so it is input/invalidating information rather than a frame clock.
        // Minimized surfaces still wait for a nonzero resize.
        if self.state.width > 0 && self.state.height > 0 {
            self.state.redraw_ready = true;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        self.state.window_event(id, event);
        event_loop.set_control_flow(if self.state.width == 0 || self.state.height == 0 {
            ControlFlow::Wait
        } else {
            ControlFlow::Poll
        });
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
    fn initial_extent_is_forwarded_once_before_hidden_frames() {
        let mut state = HostState::new("hidden", 640, 480, None, false);
        let mut forwarded = Vec::new();

        assert!(
            state
                .apply_pending_resize(|width, height| {
                    forwarded.push((width, height));
                    Ok(())
                })
                .unwrap()
        );
        assert_eq!(forwarded, [(640, 480)]);
        assert_eq!(state.pending_resize, None);
    }

    #[test]
    fn minimized_resize_waits_until_nonzero_restore() {
        let mut state = HostState::new("resize", 640, 480, None, false);

        state.record_resize(PhysicalSize::new(0, 480));
        assert_eq!((state.width, state.height), (0, 0));
        assert!(state.redraw_ready);
        assert!(
            !state
                .apply_pending_resize(|width, height| {
                    assert_eq!((width, height), (0, 0));
                    Err(ez_gfx::Error::NotReady)
                })
                .unwrap()
        );
        assert_eq!(state.pending_resize, None);

        state.record_resize(PhysicalSize::new(800, 600));
        assert!(
            state
                .apply_pending_resize(|width, height| {
                    assert_eq!((width, height), (800, 600));
                    Ok(())
                })
                .unwrap()
        );
        assert_eq!((state.width, state.height), (800, 600));
        assert_eq!(state.pending_resize, None);
    }
    #[test]
    fn minimized_resize_propagates_unexpected_graphics_errors() {
        let mut state = HostState::new("resize", 640, 480, None, false);
        state.record_resize(PhysicalSize::new(0, 0));

        assert!(matches!(
            state.apply_pending_resize(|_, _| Err(ez_gfx::Error::NativeFailure)),
            Err(Error::Graphics(ez_gfx::Error::NativeFailure))
        ));
    }

    #[test]
    fn publication_failure_exit_preserves_active_unwind() {
        assert!(should_exit_after_publication_failure(false));
        assert!(!should_exit_after_publication_failure(true));
    }

    #[test]
    fn fps_smoothing_uses_completed_presentation_intervals() {
        assert_eq!(smoothed_fps(0.0, Duration::ZERO), 0.0);
        assert_eq!(smoothed_fps(0.0, Duration::from_millis(10)), 100.0);
        assert_eq!(smoothed_fps(60.0, Duration::from_millis(10)), 64.0);
    }

    #[test]
    fn title_refresh_runs_first_then_at_five_hertz_and_when_forced() {
        let started = Instant::now();
        assert!(title_refresh_due(None, started, false));
        assert!(!title_refresh_due(
            Some(started),
            started + Duration::from_millis(199),
            false
        ));
        assert!(title_refresh_due(
            Some(started),
            started + Duration::from_millis(200),
            false
        ));
        assert!(title_refresh_due(Some(started), started, true));
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
            strict_all: false,
            snapshot: Some(missing_snapshot.into_os_string()),
            update_snapshots: false,
            report_stdout: false,
            event_loop: None,
            state: HostState::new("drop regression", 1, 1, Some(1), false),
            backend: Backend::Vulkan,
            debug_enabled: false,
            validation_enabled: false,
            resize_after_first_frame: false,
            observations,
            benchmark: BenchmarkRunner::new(None),
            frame_timings_enabled: false,
            frame_timings: Vec::new(),
            pending_frame_started: None,
            pending_host_wait_ns: 0,
            frames: 1,
            last_frame: Instant::now(),
            last_present: None,
            fps: 0.0,
            title_text: String::new(),
            last_title_refresh: None,
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
