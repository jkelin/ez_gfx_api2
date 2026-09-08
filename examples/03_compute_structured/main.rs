//! Structured compute using safe ez-gfx context, resource, and frame APIs.
mod renderer {
    use crate::shared::{input::*, math::*, mesh::*, *};
    use ez_gfx::*;
    use glam::{DVec2, Mat4, Vec3};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScenePush {
        mvp: Mat4,
        primitive_count: u32,
        padding: [u32; 3],
    }

    pub(super) struct ModelScene {
        shader: ShaderHandle,
        camera: OrbitCamera,
        clip_y: ClipY,
        target: Vec3,
        near: f32,
        far: f32,
        push: ScenePush,
        primitive_count: u32,
        records: Vec<BasicPrimitive>,
        _positions_heap: VertexHeapHandle,
        _normals_heap: VertexHeapHandle,
        _positions: VertexAllocationHandle,
        _normals: VertexAllocationHandle,
        _indices: IndexAllocationHandle,
    }

    impl ModelScene {
        pub(super) fn create(context: ContextHandle, clip_y: ClipY) -> anyhow::Result<Self> {
            let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .ok_or_else(|| anyhow::anyhow!("examples package has no workspace parent"))?;
            let shader_bytes = ez_gfx_compiler::compile_shader(
                &workspace_root.join("examples/03_compute_structured/03_compute_structured.slang"),
                &[
                    ez_gfx_compiler::Target::Spirv,
                    ez_gfx_compiler::Target::Dxil,
                    ez_gfx_compiler::Target::Metal,
                ],
                !cfg!(target_vendor = "apple"),
            )?;
            let mesh = load_geometry_glb(include_bytes!("../shared/assets/sponza.glb"))?;
            let primitive_count = u32::try_from(mesh.primitives.len())?;
            let records_size = std::mem::size_of::<BasicPrimitive>()
                .checked_mul(mesh.primitives.len())
                .ok_or_else(|| anyhow::anyhow!("primitive records size overflow"))?;
            if records_size > 16 * 1024 * 1024 {
                anyhow::bail!("primitive records exceed ABI boundary");
            }
            let index_bytes = byte_len(&mesh.indices)?;
            let positions_bytes = byte_len(&mesh.positions)?;
            let normals_bytes = byte_len(&mesh.normals)?;
            create_index_heap(context, index_bytes)?;
            let index_allocation = upload_indices(context, &mesh.indices)?;
            let (first_index, _) = index_allocation_range(context, index_allocation)?;
            let records = basic_primitives(&mesh, first_index)?;
            let positions_heap = create_vertex_heap(context, "positions", positions_bytes, 16)?;
            let positions = upload_vertices(context, positions_heap, &mesh.positions)?;
            let normals_heap = create_vertex_heap(context, "normals", normals_bytes, 16)?;
            let normals = upload_vertices(context, normals_heap, &mesh.normals)?;
            let shader = load_shader(context, &shader_bytes)?;
            Ok(Self {
                shader,
                camera: OrbitCamera::new((-30.0_f32).to_radians(), 52.0_f32.to_radians(), 2.2),
                clip_y,
                target: Vec3::new(0.0, 0.55, 0.0),
                near: 0.1,
                far: 500.0,
                push: ScenePush {
                    mvp: Mat4::IDENTITY,
                    primitive_count,
                    padding: [0; 3],
                },
                primitive_count,
                records,
                _positions_heap: positions_heap,
                _normals_heap: normals_heap,
                _positions: positions,
                _normals: normals,
                _indices: index_allocation,
            })
        }

        pub(super) fn handle_input(&mut self, input: SceneInput) {
            match input {
                SceneInput::CursorMoved { x, y } => self.camera.cursor(DVec2::new(x, y)),
                SceneInput::PrimaryButton(value) => self.camera.set_dragging(value),
                SceneInput::ScrollLines(lines) => self.camera.zoom(lines),
                _ => {}
            }
        }

        pub(super) fn update(&mut self, frame: FrameInput) -> anyhow::Result<()> {
            self.push.mvp = row_major(
                perspective(
                    60.0_f32.to_radians(),
                    frame.width as f32 / frame.height as f32,
                    self.near,
                    self.far,
                    self.clip_y,
                )? * self.camera.view(self.target)?,
            );
            Ok(())
        }

        pub(super) fn record(&mut self, context: ContextHandle) -> anyhow::Result<()> {
            let primitives = acquire_structured::<BasicPrimitive>(context, self.records.len())?;
            write_structured(context, primitives, 0, &self.records)?;
            let indirect = acquire_indirect(context, self.primitive_count)?;
            publish_compute_indirect_count(context, indirect, self.primitive_count)?;
            let bindings = [
                PublicBinding {
                    name: "primitives".to_owned(),
                    resource: ResourceIdentity::Structured(primitives),
                },
                PublicBinding {
                    name: "draw_commands".to_owned(),
                    resource: ResourceIdentity::Indirect(indirect),
                },
            ];
            render_add_compute(
                context,
                self.shader,
                [self.primitive_count, 1, 1],
                &bindings,
                bytes_of(&self.push),
            )?;
            render_add_graphics(
                context,
                self.shader,
                indirect,
                &bindings,
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                bytes_of(&self.push),
            )?;
            Ok(())
        }
    }
}

#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use renderer::ModelScene as ExampleScene;
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
        })
        .map_err(|error| anyhow::anyhow!("{error:?}"))?;
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
        .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        self.surface = Some(surface);
        init_device(self.context_handle(), self.surface())?;
        resize_surface(self.context_handle(), self.surface(), width, height)?;
        self.resources = Some(ExampleScene::create(
            self.context_handle(),
            shared::clip_y(backend),
        )?);
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
        resources.update(frame)?;
        begin_render(context, surface)?;
        resources.record(context)?;
        finish_render(context)?;
        self.benchmark.end_frame(frame_index.saturating_add(1));
        Ok(())
    }

    fn capture(&mut self, width: u32, height: u32, frames: u32) -> anyhow::Result<ProgramReport> {
        let rgba8 =
            frame_readback(self.context_handle()).map_err(|error| anyhow::anyhow!("{error:?}"))?;
        let counts = drain_bounded(
            4096,
            || {
                poll_runtime_event(self.context_handle())
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(|error| anyhow::anyhow!("{error:?}"))
            },
            || {
                poll_diagnostic(self.context_handle())
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(|error| anyhow::anyhow!("{error:?}"))
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
    run_program("03_compute_structured", backend, run_example_with_benchmark);
}
