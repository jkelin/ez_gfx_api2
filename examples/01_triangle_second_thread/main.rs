//! Triangle with window events and graphics running on separate threads.
#[path = "../shared/mod.rs"]
mod shared;

#[cfg(windows)]
use ez_gfx::*;
#[cfg(windows)]
use shared::*;
#[cfg(windows)]
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
};

#[cfg(windows)]
const WIDTH: u32 = 640;
#[cfg(windows)]
const HEIGHT: u32 = 480;

#[cfg(windows)]
struct RenderState {
    stop: AtomicBool,
    surface_size: AtomicU64,
    frames: AtomicU32,
    telemetry_frame: AtomicU32,
    fps: AtomicU32,
    diagnostics_valid: AtomicBool,
    pending_textures: AtomicU32,
    pending_texture_bytes: AtomicU64,
    pending_vertex_uploads: AtomicU32,
    pending_vertex_bytes: AtomicU64,
    pending_index_uploads: AtomicU32,
    pending_index_bytes: AtomicU64,
    staging_buckets: AtomicU32,
    staging_bytes: AtomicU64,
    pipeline_entries: AtomicU32,
    readback_bytes: AtomicU64,
}

#[cfg(windows)]
impl RenderState {
    fn new(size: [u32; 2]) -> Self {
        Self {
            stop: AtomicBool::new(false),
            surface_size: AtomicU64::new(Self::pack_size(size)),
            frames: AtomicU32::new(0),
            fps: AtomicU32::new(0.0_f32.to_bits()),
            telemetry_frame: AtomicU32::new(0),
            diagnostics_valid: AtomicBool::new(false),
            pending_textures: AtomicU32::new(0),
            pending_texture_bytes: AtomicU64::new(0),
            pending_vertex_uploads: AtomicU32::new(0),
            pending_vertex_bytes: AtomicU64::new(0),
            pending_index_uploads: AtomicU32::new(0),
            pending_index_bytes: AtomicU64::new(0),
            staging_buckets: AtomicU32::new(0),
            staging_bytes: AtomicU64::new(0),
            pipeline_entries: AtomicU32::new(0),
            readback_bytes: AtomicU64::new(0),
        }
    }

    const fn pack_size([width, height]: [u32; 2]) -> u64 {
        (width as u64) << 32 | height as u64
    }

    fn set_surface_size(&self, size: [u32; 2]) {
        self.surface_size
            .store(Self::pack_size(size), Ordering::Relaxed);
    }

    fn surface_size(&self) -> [u32; 2] {
        let packed = self.surface_size.load(Ordering::Relaxed);
        [(packed >> 32) as u32, packed as u32]
    }

    fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn publish_frame(&self, update: ThreadedFrameUpdate) {
        self.fps.store(update.fps.to_bits(), Ordering::Relaxed);
        if let Some(diagnostics) = update.diagnostics {
            if let Some(diagnostics) = diagnostics {
                self.pending_textures
                    .store(diagnostics.pending_textures, Ordering::Relaxed);
                self.pending_texture_bytes
                    .store(diagnostics.pending_texture_bytes, Ordering::Relaxed);
                self.pending_vertex_uploads
                    .store(diagnostics.pending_vertex_uploads, Ordering::Relaxed);
                self.pending_vertex_bytes
                    .store(diagnostics.pending_vertex_bytes, Ordering::Relaxed);
                self.pending_index_uploads
                    .store(diagnostics.pending_index_uploads, Ordering::Relaxed);
                self.pending_index_bytes
                    .store(diagnostics.pending_index_bytes, Ordering::Relaxed);
                self.staging_buckets
                    .store(diagnostics.staging_buckets, Ordering::Relaxed);
                self.staging_bytes
                    .store(diagnostics.staging_bytes, Ordering::Relaxed);
                self.pipeline_entries
                    .store(diagnostics.pipeline_entries, Ordering::Relaxed);
                self.readback_bytes
                    .store(diagnostics.readback_bytes, Ordering::Relaxed);
                self.diagnostics_valid.store(true, Ordering::Relaxed);
            } else {
                self.diagnostics_valid.store(false, Ordering::Relaxed);
            }
            self.telemetry_frame.store(update.frames, Ordering::Release);
        }
        // Publishing the frame last lets the window thread acquire one complete sample.
        self.frames.store(update.frames, Ordering::Release);
    }

    fn telemetry(&self) -> (u32, u32, f32, Option<ResourceDiagnostics>) {
        let frames = self.frames.load(Ordering::Acquire);
        let telemetry_frame = self.telemetry_frame.load(Ordering::Acquire);
        let fps = f32::from_bits(self.fps.load(Ordering::Relaxed));
        let diagnostics =
            self.diagnostics_valid
                .load(Ordering::Relaxed)
                .then(|| ResourceDiagnostics {
                    pending_textures: self.pending_textures.load(Ordering::Relaxed),
                    pending_texture_bytes: self.pending_texture_bytes.load(Ordering::Relaxed),
                    pending_vertex_uploads: self.pending_vertex_uploads.load(Ordering::Relaxed),
                    pending_vertex_bytes: self.pending_vertex_bytes.load(Ordering::Relaxed),
                    pending_index_uploads: self.pending_index_uploads.load(Ordering::Relaxed),
                    pending_index_bytes: self.pending_index_bytes.load(Ordering::Relaxed),
                    staging_buckets: self.staging_buckets.load(Ordering::Relaxed),
                    staging_bytes: self.staging_bytes.load(Ordering::Relaxed),
                    pipeline_entries: self.pipeline_entries.load(Ordering::Relaxed),
                    readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
                });
        (frames, telemetry_frame, fps, diagnostics)
    }
}

#[cfg(windows)]
fn should_apply_title(applied_frame: u32, telemetry_frame: u32) -> bool {
    // Frame zero denotes no telemetry when both values retain their initial state.
    applied_frame != telemetry_frame
}

#[cfg(not(windows))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "01_triangle_second_thread requires Windows: Linux raw window handles are not Send, and Metal surface creation is main-thread constrained"
    )
}

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    let mut example = Example::new("01_triangle_second_thread", WIDTH, HEIGHT, "ez_gfx_api2")?;
    let backend = backend_config(example.backend());
    let debug_enabled = example.debug_enabled();
    let validation_enabled = example.validation_enabled();
    let host = example.native_surface()?;
    let state = Arc::new(RenderState::new(example.surface_size()));
    let render_state = Arc::clone(&state);
    let render_config = example.take_threaded_render_config();

    let graphics_thread = std::thread::Builder::new()
        .name("triangle-gpu".to_owned())
        .spawn(move || -> anyhow::Result<ThreadedRenderOutput> {
            let context = Context::new(ContextOptions {
                enable_debug: debug_enabled,
                enable_validation: validation_enabled,
                backend: backend.backend,
                texture_decode_workers: 0,
                adapter_selection: None,
            })?;
            let surface = context.create_surface_window(host, false)?;
            render(context, surface, &render_state, render_config)
        })?;

    let mut title_frame = 0;
    let window_result = (|| -> shared::Result<()> {
        loop {
            state.set_surface_size(example.surface_size());
            let (_, telemetry_frame, fps, diagnostics) = state.telemetry();
            if should_apply_title(title_frame, telemetry_frame) {
                example.update_title(fps, diagnostics.as_ref());
                title_frame = telemetry_frame;
            }
            if graphics_thread.is_finished() || !example.pump_window_events()? {
                break;
            }
        }
        Ok(())
    })();
    state.request_stop();
    let render_result = graphics_thread
        .join()
        .map_err(|_| anyhow::anyhow!("graphics thread panicked"))
        .and_then(|result| result);
    // Concurrent failures prefer render failure; window failure is otherwise preserved.
    let render_output = match render_result {
        Ok(output) => {
            window_result?;
            output
        }
        Err(error) => return Err(error),
    };
    let (_, telemetry_frame, fps, diagnostics) = state.telemetry();
    if should_apply_title(title_frame, telemetry_frame) {
        example.update_title(fps, diagnostics.as_ref());
    }
    example.accept_threaded_render_output(render_output);
    Ok(())
}

#[cfg(windows)]
fn render(
    context: Context,
    surface: Surface,
    state: &RenderState,
    config: ThreadedRenderConfig,
) -> anyhow::Result<ThreadedRenderOutput> {
    let compiled_shader = ez_gfx_compiler::EasyGraphicsCompiler::compile_shader(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/01_triangle_second_thread/01_triangle_second_thread.slang"
        )),
        &[
            ez_gfx_compiler::Target::Spirv,
            ez_gfx_compiler::Target::Dxil,
            ez_gfx_compiler::Target::Metal,
        ],
        !cfg!(target_vendor = "apple"),
    )?;
    let vertical = if cfg!(target_vendor = "apple") {
        -1.0
    } else {
        1.0
    };
    let positions = [
        [-0.5_f32, -0.5 * vertical, 0.0, 1.0],
        [0.5, -0.5 * vertical, 0.0, 1.0],
        [0.0, 0.5 * vertical, 0.0, 1.0],
    ];
    let indices = context.upload_indices(&[0_u32, 1, 2])?;
    let first_index = indices.range()?.0;
    let positions_heap = context.create_vertex_heap("positions")?;
    let _positions = positions_heap.upload(&positions)?;
    let vertex_shader = compiled_shader.load_vertex_shader(&context, "vertexmain")?;
    let fragment_shader = compiled_shader.load_fragment_shader(&context, "fragmentmain")?;
    let mut configured_size = [0, 0];

    let mut automation = config.start(&context)?;
    while !state.should_stop() && automation.should_render() {
        let size = state.surface_size();
        if size != configured_size {
            let [width, height] = size;
            match surface.resize(width, height) {
                Ok(()) if width > 0 && height > 0 => {}
                Err(ez_gfx::Error::NotReady) if width == 0 || height == 0 => {}
                Ok(()) => anyhow::bail!("minimized surface resize unexpectedly succeeded"),
                Err(error) => return Err(error.into()),
            }
            configured_size = size;
        }
        if size[0] == 0 || size[1] == 0 {
            // A minimized window has no drawable. Yield until the window thread publishes a size.
            std::thread::yield_now();
            continue;
        }

        automation.begin_frame();
        let mut frame = surface.begin_frame()?;
        let swapchain_target =
            frame.configure_swapchain(size, Format::Bgra8Srgb, PresentationMode::Immediate)?;
        let commands = [DrawIndexedCommand {
            index_count: 3,
            instance_count: 1,
            first_index,
            vertex_offset: 0,
            first_instance: 0,
        }];
        let indirect = context.acquire_counter_buffer_from(commands.as_slice())?;
        frame.execute_graphics(
            &vertex_shader,
            &fragment_shader,
            &indirect,
            DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
        )?;
        let update = automation.finish_frame(&context, frame, swapchain_target, size)?;
        state.publish_frame(update);
    }

    Ok(automation.finish())
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn title_frame_is_applied_once_and_final_sample_is_not_missed() {
        let telemetry_frame = 7;
        let mut applied_frame = 0;
        let mut applications = 0;
        if should_apply_title(applied_frame, telemetry_frame) {
            applications += 1;
            applied_frame = telemetry_frame;
        }
        if should_apply_title(applied_frame, telemetry_frame) {
            applications += 1;
        }
        assert_eq!(applications, 1);

        assert!(should_apply_title(0, telemetry_frame));
    }
}
