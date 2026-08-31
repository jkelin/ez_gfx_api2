//! Helmet model using safe ez-gfx context, resource, and frame APIs.
#![allow(
    clippy::borrow_as_ptr,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::ignored_unit_patterns,
    clippy::map_unwrap_or,
    clippy::match_overlapping_arm,
    clippy::needless_pass_by_value,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::type_complexity,
    clippy::unused_self,
    clippy::wildcard_imports,
    reason = "The inline renderer preserves fixed graphics ABI and callback contracts."
)]
mod renderer {
    use crate::shared;
    use crate::shared::{FrameInput, SceneInput};
    use crate::shared::{
        math::{Mat4, OrbitCamera, mul, perspective},
        mesh::{BasicPrimitive, basic_primitives, load_geometry_glb},
    };
    use ez_gfx::{
        DynamicPipelineState, EzGfxResult, PublicBinding, ResourceIdentity, ShaderRequest, Stage,
        acquire_indirect, acquire_structured, create_index_heap, destroy_index_heap,
        destroy_shader, load_shader, release_indirect, release_structured, render_add_compute,
        render_add_graphics, set_indirect_count, upload_indices, write_structured,
    };

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScenePush {
        mvp: Mat4,
        primitive_count: u32,
        padding: [u32; 3],
    }

    pub(super) struct ModelScene {
        shader: u64,
        positions: u64,
        normals: u64,
        primitives: u64,
        indirect: u64,
        camera: OrbitCamera,
        target: [f32; 3],
        near: f32,
        far: f32,
        push: ScenePush,
        primitive_count: u32,
        bindings: [PublicBinding; 4],
    }

    impl ModelScene {
        #[allow(
            clippy::too_many_lines,
            reason = "The example keeps every safe resource acquisition, binding, and failure cleanup visible in one linear flow."
        )]
        pub(super) fn create(context: u64) -> Result<Self, String> {
            let mesh = load_geometry_glb(include_bytes!("helmet.glb"))?;
            let primitive_count = u32::try_from(mesh.primitives.len())
                .map_err(|_| "primitive count exceeds ABI".to_owned())?;
            let primitive_bytes = u64::from(primitive_count)
                .checked_mul(
                    u64::try_from(std::mem::size_of::<BasicPrimitive>())
                        .map_err(|_| "primitive record size exceeds ABI".to_owned())?,
                )
                .ok_or_else(|| "primitive records size overflow".to_owned())?;
            if primitive_bytes > 16 * 1024 * 1024 {
                return Err("primitive records exceed ABI boundary".to_owned());
            }
            let index_bytes = shared::byte_len(&mesh.indices)?;
            let positions_bytes = shared::byte_len(&mesh.positions)?;
            let normals_bytes = shared::byte_len(&mesh.normals)?;
            status(
                create_index_heap(context, index_bytes),
                "create model index heap",
            )?;
            let first_index = match upload_indices(
                context,
                mesh.indices.len() as u32,
                shared::slice_bytes(&mesh.indices),
            ) {
                Ok(value) => value,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(format!("upload model indices: {error:?}"));
                }
            };
            let records = match basic_primitives(&mesh, first_index) {
                Ok(records) => records,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(error);
                }
            };
            let positions = match acquire_structured(context, positions_bytes) {
                Ok(handle) => handle.get(),
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(format!("acquire positions: {error:?}"));
                }
            };
            if let Err(error) = status(
                write_structured(context, positions, shared::slice_bytes(&mesh.positions)),
                "upload positions",
            ) {
                release_structured(context, positions);
                destroy_index_heap(context);
                return Err(error);
            }
            let normals = match acquire_structured(context, normals_bytes) {
                Ok(handle) => handle.get(),
                Err(error) => {
                    release_structured(context, positions);
                    destroy_index_heap(context);
                    return Err(format!("acquire normals: {error:?}"));
                }
            };
            if let Err(error) = status(
                write_structured(context, normals, shared::slice_bytes(&mesh.normals)),
                "upload normals",
            ) {
                release_structured(context, normals);
                release_structured(context, positions);
                destroy_index_heap(context);
                return Err(error);
            }
            let primitives = match acquire_structured(context, primitive_bytes) {
                Ok(handle) => handle.get(),
                Err(error) => {
                    release_structured(context, normals);
                    release_structured(context, positions);
                    destroy_index_heap(context);
                    return Err(format!("acquire primitives: {error:?}"));
                }
            };
            if let Err(error) = status(
                write_structured(context, primitives, shared::slice_bytes(&records)),
                "upload primitives",
            ) {
                release_structured(context, primitives);
                release_structured(context, normals);
                release_structured(context, positions);
                destroy_index_heap(context);
                return Err(error);
            }
            let indirect = match acquire_indirect(context, primitive_count) {
                Ok(handle) => handle.get(),
                Err(error) => {
                    release_structured(context, primitives);
                    release_structured(context, normals);
                    release_structured(context, positions);
                    destroy_index_heap(context);
                    return Err(format!("acquire compute draw commands: {error:?}"));
                }
            };
            if let Err(error) = status(
                set_indirect_count(context, indirect, primitive_count),
                "set compute draw count",
            ) {
                release_indirect(context, indirect);
                release_structured(context, primitives);
                release_structured(context, normals);
                release_structured(context, positions);
                destroy_index_heap(context);
                return Err(error);
            }
            let requests = [
                ShaderRequest::new("vertexmain", Stage::Vertex).unwrap(),
                ShaderRequest::new("fragmentmain", Stage::Fragment).unwrap(),
                ShaderRequest::new("computemain", Stage::Compute).unwrap(),
            ];
            let shader = match load_shader(context, include_bytes!("05_helmet.ezgfx"), &requests) {
                Ok(value) => value.get(),
                Err(error) => {
                    release_indirect(context, indirect);
                    release_structured(context, primitives);
                    release_structured(context, normals);
                    release_structured(context, positions);
                    destroy_index_heap(context);
                    return Err(format!("load helmet artifact: {error:?}"));
                }
            };
            Ok(Self {
                shader,
                positions,
                normals,
                primitives,
                indirect,
                camera: OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0),
                target: [0.0, 0.0, 0.0],
                near: 0.1,
                far: 100.0,
                push: ScenePush {
                    mvp: [0.0; 16],
                    primitive_count,
                    padding: [0; 3],
                },
                primitive_count,
                bindings: [
                    PublicBinding {
                        name: "positions".to_owned(),
                        resource: ResourceIdentity::Structured(positions),
                    },
                    PublicBinding {
                        name: "normals".to_owned(),
                        resource: ResourceIdentity::Structured(normals),
                    },
                    PublicBinding {
                        name: "primitives".to_owned(),
                        resource: ResourceIdentity::Structured(primitives),
                    },
                    PublicBinding {
                        name: "draw_commands".to_owned(),
                        resource: ResourceIdentity::Indirect(indirect),
                    },
                ],
            })
        }
    }

    impl ModelScene {
        pub(super) fn handle_input(&mut self, input: SceneInput) {
            match input {
                SceneInput::CursorMoved { x, y } => self.camera.cursor([x, y]),
                SceneInput::PrimaryButton(value) => self.camera.set_dragging(value),
                SceneInput::ScrollLines(lines) => self.camera.zoom(lines),
                _ => {}
            }
        }
        pub(super) fn update(&mut self, frame: FrameInput) -> Result<(), String> {
            self.push.mvp = mul(
                perspective(
                    60.0_f32.to_radians(),
                    frame.width as f32 / frame.height as f32,
                    self.near,
                    self.far,
                )?,
                self.camera.view(self.target)?,
            );
            Ok(())
        }
        pub(super) fn record(&mut self, context: u64) -> Result<(), String> {
            let bindings = &self.bindings;
            status(
                render_add_compute(
                    context,
                    self.shader,
                    [self.primitive_count, 1, 1],
                    bindings,
                    shared::bytes_of(&self.push),
                ),
                "record compute pipeline",
            )?;
            status(
                render_add_graphics(
                    context,
                    self.shader,
                    self.indirect,
                    bindings,
                    DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                    shared::bytes_of(&self.push),
                ),
                "record model graphics pipeline",
            )
        }
        pub(super) fn destroy(self, context: u64) {
            release_indirect(context, self.indirect);
            release_structured(context, self.primitives);
            release_structured(context, self.normals);
            release_structured(context, self.positions);
            destroy_index_heap(context);
            destroy_shader(context, self.shader);
        }
    }

    fn status(result: EzGfxResult, operation: &str) -> Result<(), String> {
        match result {
            EzGfxResult::Ok => Ok(()),
            error => Err(format!("{operation}: {error:?}")),
        }
    }
}
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::{
    Backend, ContextOptions, EzGfxResult, SurfaceOptions, SurfacePlatform, begin_render,
    create_context, create_surface, destroy_context, destroy_surface, finish_render,
    frame_readback, init_device, poll_diagnostic, poll_runtime_event, resize_surface,
    set_snapshot_cache, wait_idle,
};
use renderer::ModelScene as ExampleScene;
use shared::{
    FrameInput, LifecycleCallbacks, LifecycleConfig, NativePlatform, NativeSurface, SceneInput,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

struct Example {
    resources: Option<ExampleScene>,
    context: u64,
    surface: u64,
    benchmark: shared::BenchmarkRunner,
}

impl Example {
    const fn new(benchmark: Option<shared::BenchmarkConfig>) -> Self {
        Self {
            resources: None,
            context: 0,
            surface: 0,
            benchmark: shared::BenchmarkRunner::new(benchmark),
        }
    }
}

impl LifecycleCallbacks for Example {
    type Report = shared::ProgramReport;

    fn initialize(&mut self, native: NativeSurface, width: u32, height: u32) -> Result<(), String> {
        let (backend, backend_name) = backend()?;
        let platform = match native.platform {
            NativePlatform::Win32 => SurfacePlatform::Win32,
            NativePlatform::MetalLayer => SurfacePlatform::MetalLayer,
        };
        self.context = create_context(ContextOptions {
            enable_debug: shared::env_flag("EZ_GFX_EXAMPLE_DEBUG")?,
            enable_validation: shared::env_flag("EZ_GFX_EXAMPLE_VALIDATION")?,
            surface_platform: platform,
            backend,
        })
        .map_err(|error| format!("create {backend_name} context: {error:?}"))?
        .get();
        self.surface = create_surface(
            self.context,
            SurfaceOptions {
                window: native.window,
                display: native.display,
                platform,
                width,
                height,
                cache_presented_snapshots: false,
            },
        )
        .map_err(|error| format!("create {backend_name} surface: {error:?}"))?
        .get();
        status(
            init_device(self.context, self.surface),
            &format!("initialize {backend_name} surface device"),
        )?;
        status(
            resize_surface(self.context, self.surface, width, height),
            &format!("initialize {backend_name} swapchain"),
        )?;
        self.resources = Some(ExampleScene::create(self.context)?);
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        status(
            resize_surface(self.context, self.surface, width, height),
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
    ) -> Result<(), String> {
        self.benchmark.begin_frame(frame_index);
        if terminal {
            status(
                set_snapshot_cache(self.context, self.surface, true),
                "enable terminal snapshot cache",
            )?;
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| "example resources are unavailable".to_owned())?;
        resources.update(frame)?;
        status(
            begin_render(self.context, self.surface),
            "begin presented frame",
        )?;
        resources.record(self.context)?;
        status(finish_render(self.context), "submit and present example")?;
        self.benchmark.end_frame(frame_index.saturating_add(1));
        Ok(())
    }

    fn capture(
        &mut self,
        width: u32,
        height: u32,
        frames: u32,
    ) -> Result<shared::ProgramReport, String> {
        let rgba8 = frame_readback(self.context)
            .map_err(|error| format!("read presented snapshot: {error:?}"))?;
        let counts = shared::drain_bounded(
            4096,
            || {
                poll_runtime_event(self.context)
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(|error| format!("poll runtime event: {error:?}"))
            },
            || {
                poll_diagnostic(self.context)
                    .map(|(record, dropped)| (record.is_some(), dropped))
                    .map_err(|error| format!("poll diagnostic: {error:?}"))
            },
        )?;
        Ok(shared::ProgramReport {
            frame: shared::PresentedFrame {
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
        if self.context == 0 {
            return;
        }
        let _ = wait_idle(self.context);
        if let Some(resources) = self.resources.take() {
            resources.destroy(self.context);
        }
        if self.surface != 0 {
            destroy_surface(self.context, self.surface);
            self.surface = 0;
        }
        destroy_context(self.context);
        self.context = 0;
    }
}

fn run_example_with_benchmark(
    frame_limit: Option<u32>,
    benchmark: Option<shared::BenchmarkConfig>,
) -> Result<Option<shared::ProgramReport>, String> {
    shared::run(
        LifecycleConfig {
            width: WIDTH,
            height: HEIGHT,
            title: "ez_gfx_api2",
            frame_limit,
        },
        Example::new(benchmark),
    )
}

fn backend() -> Result<(Backend, &'static str), String> {
    match std::env::var("EZ_GFX_BACKEND").ok().as_deref() {
        #[cfg(target_vendor = "apple")]
        None | Some("metal") => Ok((Backend::Metal, "Metal")),
        #[cfg(not(target_vendor = "apple"))]
        None | Some("vulkan") => Ok((Backend::Vulkan, "Vulkan")),
        #[cfg(windows)]
        Some("dx12") => Ok((Backend::Dx12, "DX12")),
        Some(value) => Err(format!("unsupported EZ_GFX_BACKEND `{value}`")),
    }
}

fn status(result: EzGfxResult, operation: &str) -> Result<(), String> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(format!("{operation}: {error:?}")),
    }
}

fn main() {
    let backend = std::env::var("EZ_GFX_BACKEND").unwrap_or_else(|_| {
        if cfg!(target_vendor = "apple") {
            "metal".to_owned()
        } else {
            "vulkan".to_owned()
        }
    });
    shared::run_program("05_helmet", &backend, run_example_with_benchmark);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_selection_defaults_and_validates() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(backend(), Ok((Backend::Metal, "Metal")));
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(backend(), Ok((Backend::Vulkan, "Vulkan")));
    }
}
