//! Triangle using safe ez-gfx context, resource, and frame APIs.
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
    use crate::shared::{FrameInput, SceneInput};
    use ez_gfx::{
        ContextHandle, DrawIndexedCommand, DynamicPipelineState, EzGfxResult, IndirectBufferHandle,
        PublicBinding, ResourceIdentity, ShaderHandle, StructuredBufferHandle, acquire_indirect,
        acquire_structured, create_index_heap, destroy_index_heap, destroy_shader, load_shader,
        release_indirect, release_structured, render_add_graphics, set_indirect_count,
        upload_indices, write_indirect, write_structured,
    };

    use crate::shared;

    pub(super) struct Triangle {
        shader: ShaderHandle,
        positions: StructuredBufferHandle,
        indirect: IndirectBufferHandle,
        bindings: [PublicBinding; 1],
    }

    impl Triangle {
        pub(super) fn create(context: ContextHandle) -> anyhow::Result<Self> {
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

            let index_bytes = shared::byte_len(&indices)?;
            let positions_bytes = shared::byte_len(&positions)?;
            status(create_index_heap(context, index_bytes), "create index heap")?;
            let first_index = match upload_indices(
                context,
                indices.len() as u32,
                shared::slice_bytes(&indices),
            ) {
                Ok(first_index) => first_index,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("upload indices"));
                }
            };
            let positions_handle = match acquire_structured(context, positions_bytes) {
                Ok(handle) => handle,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("acquire positions"));
                }
            };
            if let Err(error) = status(
                write_structured(context, positions_handle, shared::slice_bytes(&positions)),
                "upload positions",
            ) {
                release_structured(context, positions_handle);
                destroy_index_heap(context);
                return Err(error);
            }

            let indirect = match acquire_indirect(context, 1) {
                Ok(handle) => handle,
                Err(error) => {
                    release_structured(context, positions_handle);
                    destroy_index_heap(context);
                    return Err(
                        anyhow::anyhow!("{error:?}").context("acquire triangle indirect buffer")
                    );
                }
            };
            let draw_result = status(
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
            )
            .and_then(|()| {
                status(
                    set_indirect_count(context, indirect, 1),
                    "set triangle draw count",
                )
            });
            if let Err(error) = draw_result {
                release_indirect(context, indirect);
                release_structured(context, positions_handle);
                destroy_index_heap(context);
                return Err(error);
            }

            let shader = match load_shader(
                context,
                include_bytes!(concat!(env!("OUT_DIR"), "/01_triangle.ezgfxshader")),
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    release_indirect(context, indirect);
                    release_structured(context, positions_handle);
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("load triangle artifact"));
                }
            };

            Ok(Self {
                shader,
                positions: positions_handle,
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

        pub(super) fn destroy(self, context: ContextHandle) {
            release_indirect(context, self.indirect);
            release_structured(context, self.positions);
            destroy_index_heap(context);
            destroy_shader(context, self.shader);
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
use ez_gfx::{
    Backend, ContextHandle, ContextOptions, EzGfxResult, SurfaceHandle, SurfaceOptions,
    SurfacePlatform, begin_render, create_context, create_surface, destroy_context,
    destroy_surface, finish_render, frame_readback, init_device, poll_diagnostic,
    poll_runtime_event, resize_surface, set_snapshot_cache, wait_idle,
};
use renderer::Triangle as ExampleScene;
use shared::{
    FrameInput, LifecycleCallbacks, LifecycleConfig, NativePlatform, NativeSurface, SceneInput,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

struct Example {
    resources: Option<ExampleScene>,
    context: Option<ContextHandle>,
    surface: Option<SurfaceHandle>,
    benchmark: shared::BenchmarkRunner,
}

impl Example {
    fn new(benchmark: Option<shared::BenchmarkConfig>) -> Self {
        Self {
            resources: None,
            context: None,
            surface: None,
            benchmark: shared::BenchmarkRunner::new(benchmark),
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
    type Report = shared::ProgramReport;

    fn initialize(&mut self, native: NativeSurface, width: u32, height: u32) -> anyhow::Result<()> {
        let (backend, backend_name) = backend()?;
        let platform = match native.platform {
            NativePlatform::Win32 => SurfacePlatform::Win32,
            NativePlatform::MetalLayer => SurfacePlatform::MetalLayer,
        };
        let context = create_context(ContextOptions {
            enable_debug: shared::env_flag("EZ_GFX_EXAMPLE_DEBUG")?,
            enable_validation: shared::env_flag("EZ_GFX_EXAMPLE_VALIDATION")?,
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

    fn capture(
        &mut self,
        width: u32,
        height: u32,
        frames: u32,
    ) -> anyhow::Result<shared::ProgramReport> {
        let rgba8 = frame_readback(self.context())
            .map_err(|error| anyhow::anyhow!("{error:?}"))
            .context("read presented snapshot")?;
        let counts = shared::drain_bounded(
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
        let Some(context) = self.context.take() else {
            return;
        };
        let _ = wait_idle(context);
        if let Some(resources) = self.resources.take() {
            resources.destroy(context);
        }
        if let Some(surface) = self.surface.take() {
            destroy_surface(context, surface);
        }
        destroy_context(context);
    }
}

fn run_example_with_benchmark(
    frame_limit: Option<u32>,
    benchmark: Option<shared::BenchmarkConfig>,
) -> anyhow::Result<Option<shared::ProgramReport>> {
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

fn backend() -> anyhow::Result<(Backend, &'static str)> {
    match std::env::var("EZ_GFX_BACKEND").ok().as_deref() {
        #[cfg(target_vendor = "apple")]
        None | Some("metal") => Ok((Backend::Metal, "Metal")),
        #[cfg(not(target_vendor = "apple"))]
        None | Some("vulkan") => Ok((Backend::Vulkan, "Vulkan")),
        #[cfg(windows)]
        Some("dx12") => Ok((Backend::Dx12, "DX12")),
        Some(value) => Err(anyhow::anyhow!("unsupported EZ_GFX_BACKEND `{value}`")),
    }
}

fn status(result: EzGfxResult, operation: &str) -> anyhow::Result<()> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(anyhow::anyhow!("{error:?}").context(operation.to_owned())),
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
    shared::run_program("01_triangle", &backend, run_example_with_benchmark);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_selection_defaults_and_validates() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(backend(), Ok((Backend::Metal, "Metal")));
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(backend().unwrap(), (Backend::Vulkan, "Vulkan"));
    }
}
