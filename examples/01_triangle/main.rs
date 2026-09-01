//! Triangle using safe ez-gfx context, resource, and frame APIs.
mod renderer {
    use anyhow::Context as _;
    use ez_gfx::*;

    use crate::shared::{input::*, *};

    pub(super) struct Triangle {
        shader: ShaderHandle,
        indirect: IndirectBufferHandle,
        bindings: [PublicBinding; 1],
    }

    impl Triangle {
        pub(super) fn create(context: ContextHandle) -> anyhow::Result<Self> {
            let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .context("examples package has no workspace parent")?;
            let shader_bytes = ez_gfx_compiler::compile_shader(
                &workspace_root.join("examples/01_triangle/01_triangle.slang"),
                &[
                    ez_gfx_compiler::Target::Spirv,
                    ez_gfx_compiler::Target::Dxil,
                    ez_gfx_compiler::Target::Metal,
                ],
                !cfg!(target_vendor = "apple"),
            )
            .context("compile triangle shader")?;
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
            status(create_index_heap(context, index_bytes), "create index heap")?;
            let first_index = upload_indices(context, indices.len() as u32, slice_bytes(&indices))
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("upload indices")?;
            let positions_handle = acquire_structured(context, positions_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire positions")?;
            status(
                write_structured(context, positions_handle, slice_bytes(&positions)),
                "upload positions",
            )?;

            let indirect = acquire_indirect(context, 1)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire triangle indirect buffer")?;
            status(
                write_indirect(
                    context,
                    indirect,
                    0,
                    DrawIndexedCommand {
                        index_count: 3,
                        instance_count: 1,
                        first_index,
                        vertex_offset: 0,
                        first_instance: 0,
                    },
                ),
                "write triangle draw",
            )?;
            status(
                set_indirect_count(context, indirect, 1),
                "set triangle draw count",
            )?;

            let shader = load_shader(context, &shader_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("load triangle artifact")?;

            Ok(Self {
                shader,
                indirect,
                bindings: [PublicBinding {
                    name: "positions".to_owned(),
                    resource: ResourceIdentity::Structured(positions_handle),
                }],
            })
        }
    }

    impl Triangle {
        pub(super) fn handle_input(&mut self, _input: SceneInput) {}

        pub(super) fn update(&mut self, _frame: FrameInput) {}

        pub(super) fn record(&mut self, context: ContextHandle) -> anyhow::Result<()> {
            let bindings = &self.bindings;
            let state = DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap();
            status(
                render_add_graphics(context, self.shader, self.indirect, bindings, state, &[]),
                "record triangle graphics pipeline",
            )
        }
    }

    fn status(result: EzGfxResult, operation: &str) -> anyhow::Result<()> {
        match result {
            EzGfxResult::Ok => Ok(()),
            error => Err(anyhow::anyhow!("{error:?}").context(operation.to_owned())),
        }
    }
}

#[path = "../shared/mod.rs"]
mod shared;

use anyhow::Context as _;
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

    fn context(&self) -> ContextHandle {
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
        let backend_name = config.name;
        let platform = config.platform;
        let context = create_context(ContextOptions {
            enable_debug: env_flag("EZ_GFX_EXAMPLE_DEBUG")?,
            enable_validation: env_flag("EZ_GFX_EXAMPLE_VALIDATION")?,
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
        resources.update(frame);
        status(begin_render(context, surface), "begin presented frame")?;
        resources.record(context)?;
        status(finish_render(context), "submit and present example")?;
        self.benchmark.end_frame(frame_index.saturating_add(1));
        Ok(())
    }

    fn capture(&mut self, width: u32, height: u32, frames: u32) -> anyhow::Result<ProgramReport> {
        let rgba8 = frame_readback(self.context())
            .map_err(|error| anyhow::anyhow!("{error:?}"))
            .context("read presented snapshot")?;
        let counts = drain_bounded(
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

fn status(result: EzGfxResult, operation: &str) -> anyhow::Result<()> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(anyhow::anyhow!("{error:?}").context(operation.to_owned())),
    }
}
