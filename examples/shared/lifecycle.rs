use super::{FrameInput, HostSurface, NativeSurface, SceneInput, dispatch_window_input, env_flag};
use anyhow::Context as _;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

pub struct LifecycleConfig {
    pub width: u32,
    pub height: u32,
    pub title: &'static str,
    pub frame_limit: Option<u32>,
}

pub trait LifecycleCallbacks {
    type Report;
    fn initialize(&mut self, surface: NativeSurface, width: u32, height: u32)
    -> anyhow::Result<()>;
    fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()>;
    fn input(&mut self, input: SceneInput);
    fn render(&mut self, frame: FrameInput, terminal: bool, frame_index: u32)
    -> anyhow::Result<()>;
    fn capture(&mut self, width: u32, height: u32, frames: u32) -> anyhow::Result<Self::Report>;
    fn shutdown(&mut self);
}

pub fn run<C: LifecycleCallbacks>(
    config: LifecycleConfig,
    callbacks: C,
) -> anyhow::Result<Option<C::Report>> {
    if config.frame_limit == Some(0) {
        anyhow::bail!("frame limit must be positive");
    }
    if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
        // SAFETY: this executes before the event loop or graphics initialization begins.
        unsafe { std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~") };
    }
    let visible = !env_flag("EZ_GFX_EXAMPLE_HIDDEN")?;
    let event_loop = EventLoop::new().context("create event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(config, callbacks, visible);
    let run_result = event_loop.run_app(&mut app).context("run event loop");
    finish_app(app, run_result)
}

fn finish_app<C: LifecycleCallbacks>(
    mut app: App<C>,
    run_result: anyhow::Result<()>,
) -> anyhow::Result<Option<C::Report>> {
    app.callbacks.shutdown();
    run_result?;
    app.error.map_or(Ok(app.report), Err)
}

struct App<C: LifecycleCallbacks> {
    config: LifecycleConfig,
    callbacks: C,
    visible: bool,
    window: Option<Window>,
    host: Option<HostSurface>,
    width: u32,
    height: u32,
    frames: u32,
    last_frame: Instant,
    report: Option<C::Report>,
    error: Option<anyhow::Error>,
}

impl<C: LifecycleCallbacks> App<C> {
    fn new(config: LifecycleConfig, callbacks: C, visible: bool) -> Self {
        Self {
            width: config.width,
            height: config.height,
            config,
            callbacks,
            visible,
            window: None,
            host: None,
            frames: 0,
            last_frame: Instant::now(),
            report: None,
            error: None,
        }
    }
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        self.error = Some(error);
        event_loop.exit();
    }
    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let terminal = self
            .config
            .frame_limit
            .is_some_and(|limit| self.frames.saturating_add(1) >= limit);
        let frame = FrameInput {
            width: self.width,
            height: self.height,
            delta_seconds: self.last_frame.elapsed().as_secs_f32(),
        };
        self.last_frame = Instant::now();
        if let Err(error) = self.callbacks.render(frame, terminal, self.frames) {
            return self.fail(event_loop, error);
        }
        self.frames += 1;
        if terminal {
            match self.callbacks.capture(self.width, self.height, self.frames) {
                Ok(report) => self.report = Some(report),
                Err(error) => self.error = Some(error),
            }
            event_loop.exit();
        } else {
            self.window.as_ref().unwrap().request_redraw();
        }
    }
}

impl<C: LifecycleCallbacks> ApplicationHandler for App<C> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height))
            .with_visible(self.visible);
        let window = match event_loop
            .create_window(attributes)
            .context("create example window")
        {
            Ok(window) => window,
            Err(error) => return self.fail(event_loop, error),
        };
        let host = match HostSurface::attach(&window, self.config.width, self.config.height) {
            Ok(host) => host,
            Err(error) => return self.fail(event_loop, error),
        };
        if let Err(error) =
            self.callbacks
                .initialize(host.descriptor(), self.config.width, self.config.height)
        {
            return self.fail(event_loop, error);
        }
        self.host = Some(host);
        self.window = Some(window);
        if self.visible {
            self.window.as_ref().unwrap().request_redraw();
        } else {
            // Hidden automation windows are not guaranteed to receive a platform redraw event.
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
                if let Err(error) = self.callbacks.resize(size.width, size.height) {
                    self.fail(event_loop, error);
                }
            }
            WindowEvent::RedrawRequested => self.render_frame(event_loop),
            _ => {
                dispatch_window_input(&event, |input| self.callbacks.input(input));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Callbacks {
        shutdowns: u32,
    }

    impl LifecycleCallbacks for Callbacks {
        type Report = ();

        fn initialize(
            &mut self,
            _surface: NativeSurface,
            _width: u32,
            _height: u32,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn resize(&mut self, _width: u32, _height: u32) -> anyhow::Result<()> {
            Ok(())
        }

        fn input(&mut self, _input: SceneInput) {}

        fn render(
            &mut self,
            _frame: FrameInput,
            _terminal: bool,
            _frame_index: u32,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn capture(&mut self, _width: u32, _height: u32, _frames: u32) -> anyhow::Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) {
            self.shutdowns += 1;
            assert_eq!(self.shutdowns, 1);
        }
    }

    fn app() -> App<Callbacks> {
        App::new(
            LifecycleConfig {
                width: 1,
                height: 1,
                title: "test",
                frame_limit: Some(1),
            },
            Callbacks { shutdowns: 0 },
            false,
        )
    }

    #[test]
    fn shutdown_runs_once_on_normal_and_event_loop_errors() {
        assert!(finish_app(app(), Ok(())).is_ok());
        let error = finish_app(app(), Err(anyhow::anyhow!("event loop failed"))).unwrap_err();
        assert_eq!(error.to_string(), "event loop failed");
    }
}
