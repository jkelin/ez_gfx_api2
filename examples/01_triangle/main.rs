//! Triangle using safe ez-gfx context, resource, and frame APIs.
mod renderer {
    use ez_gfx::*;

    use crate::shared::{input::*, *};

    pub(super) struct Triangle {
        shader: ShaderHandle,
        first_index: u32,
        _positions_heap: VertexHeapHandle,
        _positions: VertexAllocationHandle,
        _indices: IndexAllocationHandle,
    }

    impl Triangle {
        pub(super) fn create(context: ContextHandle) -> anyhow::Result<Self> {
            let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .ok_or_else(|| anyhow::anyhow!("examples package has no workspace parent"))?;
            let shader_bytes = ez_gfx_compiler::compile_shader(
                &workspace_root.join("examples/01_triangle/01_triangle.slang"),
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
            let indices = [0_u32, 1, 2];

            let index_bytes = byte_len(&indices)?;
            let positions_bytes = byte_len(&positions)?;
            create_index_heap(context, index_bytes)?;
            let indices_handle = upload_indices(context, &indices)?;
            let (first_index, _) = index_allocation_range(context, indices_handle)?;
            let positions_heap = create_vertex_heap(context, "positions", positions_bytes, 16)?;
            let positions_handle = upload_vertices(context, positions_heap, &positions)?;

            let shader = load_shader(context, &shader_bytes)?;

            Ok(Self {
                shader,
                first_index,
                _positions_heap: positions_heap,
                _positions: positions_handle,
                _indices: indices_handle,
            })
        }
    }

    impl Triangle {
        pub(super) fn handle_input(&mut self, _input: SceneInput) {}

        pub(super) fn update(&mut self, _frame: FrameInput) {}

        pub(super) fn record(&mut self, context: ContextHandle) -> anyhow::Result<()> {
            let indirect = acquire_indirect(context, 1)?;
            write_indirect(
                context,
                indirect,
                0,
                &[DrawIndexedCommand {
                    index_count: 3,
                    instance_count: 1,
                    first_index: self.first_index,
                    vertex_offset: 0,
                    first_instance: 0,
                }],
            )?;
            let state = DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap();
            render_add_graphics(context, self.shader, indirect, &[], state, &[])?;
            Ok(())
        }
    }
}

#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use renderer::Triangle as ExampleScene;
use shared::*;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

struct Example {
    resources: Option<ExampleScene>,
    context: Option<ContextHandle>,
    surface: Option<SurfaceHandle>,
    benchmark: BenchmarkRunner,
}

impl Example {
    fn new(benchmark: Option<BenchmarkConfig>) -> Self {
        Self {
            resources: None,
            context: None,
            surface: None,
            benchmark: BenchmarkRunner::new(benchmark),
        }
    }

    fn context_handle(&self) -> ContextHandle {
        self.context.expect("example context is initialized")
    }

    fn surface(&self) -> SurfaceHandle {
        self.surface.expect("example surface is initialized")
    }
}

impl LifecycleCallbacks for Example {
    type Report = ProgramReport;

    fn initialize(&mut self, native: NativeSurface, width: u32, height: u32) -> anyhow::Result<()> {
        let config = backend_config(native.platform)?;
        let backend = config.backend;

        let platform = config.platform;
        let context = create_context(ContextOptions {
            enable_debug: env_flag("EZ_GFX_EXAMPLE_DEBUG")?,
            enable_validation: env_flag("EZ_GFX_EXAMPLE_VALIDATION")?,
            surface_platform: platform,
            backend,
            texture_decode_workers: 0,
            adapter_selection: None,
        })?;
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
        )?;
        self.surface = Some(surface);
        init_device(self.context_handle(), self.surface())?;
        resize_surface(self.context_handle(), self.surface(), width, height)?;
        self.resources = Some(ExampleScene::create(self.context_handle())?);
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        resize_surface(self.context_handle(), self.surface(), width, height)?;
        Ok(())
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
        let context = self.context_handle();
        let surface = self.surface();
        if terminal {
            set_snapshot_cache(context, surface, true)?;
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("example resources are unavailable"))?;
        resources.update(frame);
        begin_render(context, surface)?;
        resources.record(context)?;
        finish_render(context)?;
        self.benchmark.end_frame(frame_index.saturating_add(1));
        Ok(())
    }

    fn capture(&mut self, width: u32, height: u32, frames: u32) -> anyhow::Result<ProgramReport> {
        let rgba8 = frame_readback(self.context_handle())?;
        let counts = drain_bounded(
            4096,
            || {
                poll_runtime_event(self.context_handle())
                    .map(|(record, dropped)| (record.is_some(), dropped))
            },
            || {
                poll_diagnostic(self.context_handle())
                    .map(|(record, dropped)| (record.is_some(), dropped))
            },
        )?;
        Ok(ProgramReport {
            frame: PresentedFrame {
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
        if let Some(context) = self.context.take() {
            let _ = destroy_context(context);
        }
    }
}

fn run_example_with_benchmark(
    frame_limit: Option<u32>,
    benchmark: Option<BenchmarkConfig>,
) -> anyhow::Result<Option<ProgramReport>> {
    run(
        LifecycleConfig {
            width: WIDTH,
            height: HEIGHT,
            title: "ez_gfx_api2",
            frame_limit,
        },
        Example::new(benchmark),
    )
}

fn main() {
    let backend = backend_name().unwrap_or_else(|error| {
        eprintln!("{error:#}");
        std::process::exit(2);
    });
    run_program("01_triangle", backend, run_example_with_benchmark);
}
