//! Textured cube using safe ez-gfx context, resource, and frame APIs.
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
    use crate::shared::math::{ClipY, OrbitCamera, perspective, row_major};
    use crate::shared::{FrameInput, SceneInput};
    use ez_gfx::{
        ContextHandle, DrawIndexedCommand, DynamicPipelineState, EzGfxResult, IndirectBufferHandle,
        PublicBinding, ResourceIdentity, SamplerAddressMode, SamplerFilter, ShaderHandle,
        StructuredBufferHandle, TextureConfig, TextureHandle, TextureSamplerDesc, TextureSource,
        acquire_indirect, acquire_structured, create_index_heap, destroy_index_heap,
        destroy_shader, load_shader, load_texture, release_indirect, release_structured,
        render_add_graphics, set_indirect_count, texture_binding, unload_texture, upload_indices,
        write_indirect, write_structured,
    };
    use glam::{DVec2, Mat4, Vec3};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Push {
        mvp: Mat4,
        texture_id: u32,
        padding: [u32; 3],
    }

    pub(super) struct TexturedCube {
        shader: ShaderHandle,
        positions: StructuredBufferHandle,
        indirect: IndirectBufferHandle,
        texture: TextureHandle,
        camera: OrbitCamera,
        clip_y: ClipY,
        push: Push,
        bindings: [PublicBinding; 1],
    }

    impl TexturedCube {
        #[allow(
            clippy::too_many_lines,
            reason = "The example keeps every safe resource acquisition and failure cleanup visible in one linear flow."
        )]
        pub(super) fn create(context: ContextHandle, clip_y: ClipY) -> anyhow::Result<Self> {
            let positions: [[f32; 4]; 24] = [
                [-1., -1., 1., 1.],
                [1., -1., 1., 1.],
                [1., 1., 1., 1.],
                [-1., 1., 1., 1.],
                [1., -1., -1., 1.],
                [-1., -1., -1., 1.],
                [-1., 1., -1., 1.],
                [1., 1., -1., 1.],
                [-1., -1., -1., 1.],
                [-1., -1., 1., 1.],
                [-1., 1., 1., 1.],
                [-1., 1., -1., 1.],
                [1., -1., 1., 1.],
                [1., -1., -1., 1.],
                [1., 1., -1., 1.],
                [1., 1., 1., 1.],
                [-1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., -1., 1.],
                [-1., 1., -1., 1.],
                [-1., -1., -1., 1.],
                [1., -1., -1., 1.],
                [1., -1., 1., 1.],
                [-1., -1., 1., 1.],
            ];
            let indices = [
                0_u32, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15,
                12, 16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
            ];
            let index_bytes = shared::byte_len(&indices)?;
            let positions_bytes = shared::byte_len(&positions)?;
            status(
                create_index_heap(context, index_bytes),
                "create cube index heap",
            )?;
            let first_index = match upload_indices(
                context,
                indices.len() as u32,
                shared::slice_bytes(&indices),
            ) {
                Ok(value) => value,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("upload cube indices"));
                }
            };
            let positions_handle = match acquire_structured(context, positions_bytes) {
                Ok(handle) => handle,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("acquire cube positions"));
                }
            };
            if let Err(error) = status(
                write_structured(context, positions_handle, shared::slice_bytes(&positions)),
                "upload cube positions",
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
                        anyhow::anyhow!("{error:?}").context("acquire cube indirect buffer")
                    );
                }
            };
            let draw_result = status(
                write_indirect(
                    context,
                    indirect,
                    0,
                    DrawIndexedCommand {
                        index_count: indices.len() as u32,
                        instance_count: 1,
                        first_index,
                        vertex_offset: 0,
                        first_instance: 0,
                    },
                ),
                "write cube draw",
            )
            .and_then(|()| {
                status(
                    set_indirect_count(context, indirect, 1),
                    "set cube draw count",
                )
            });
            if let Err(error) = draw_result {
                release_indirect(context, indirect);
                release_structured(context, positions_handle);
                destroy_index_heap(context);
                return Err(error);
            }
            let config = TextureConfig {
                width: 0,
                height: 0,
                mip_count: 0,
                sampler: TextureSamplerDesc {
                    min_filter: SamplerFilter::Linear,
                    mag_filter: SamplerFilter::Linear,
                    max_anisotropy: 1.0,
                    address_u: SamplerAddressMode::Clamp,
                    address_v: SamplerAddressMode::Clamp,
                    address_w: SamplerAddressMode::Clamp,
                },
            };
            let texture = match load_texture(
                context,
                TextureSource::Png,
                include_bytes!("cube.png"),
                true,
                &config,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    release_indirect(context, indirect);
                    release_structured(context, positions_handle);
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("load cube texture"));
                }
            };
            let texture_id = match texture_binding(context, texture) {
                Ok(value) => value,
                Err(error) => {
                    unload_texture(context, texture);
                    release_indirect(context, indirect);
                    release_structured(context, positions_handle);
                    destroy_index_heap(context);
                    return Err(
                        anyhow::anyhow!("{error:?}").context("resolve cube texture binding")
                    );
                }
            };
            let shader = match load_shader(
                context,
                include_bytes!(concat!(env!("OUT_DIR"), "/02_textured_cube.ezgfxshader")),
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    unload_texture(context, texture);
                    release_indirect(context, indirect);
                    release_structured(context, positions_handle);
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("load cube artifact"));
                }
            };
            Ok(Self {
                shader,
                positions: positions_handle,
                indirect,
                texture,
                camera: OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0),
                clip_y,
                push: Push {
                    mvp: Mat4::IDENTITY,
                    texture_id,
                    padding: [0; 3],
                },
                bindings: [PublicBinding {
                    name: "positions".to_owned(),
                    resource: ResourceIdentity::Structured(positions_handle),
                }],
            })
        }
    }

    impl TexturedCube {
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
                    0.1,
                    100.0,
                    self.clip_y,
                )? * self.camera.view(Vec3::ZERO)?,
            );
            Ok(())
        }
        pub(super) fn record(&mut self, context: ContextHandle) -> anyhow::Result<()> {
            let bindings = &self.bindings;
            status(
                render_add_graphics(
                    context,
                    self.shader,
                    self.indirect,
                    bindings,
                    DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap(),
                    shared::bytes_of(&self.push),
                ),
                "record cube graphics pipeline",
            )
        }
        pub(super) fn destroy(self, context: ContextHandle) {
            unload_texture(context, self.texture);
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
use renderer::TexturedCube as ExampleScene;
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
        self.resources = Some(ExampleScene::create(self.context(), clip_y(backend))?);
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
        resources.update(frame)?;
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

fn clip_y(backend: Backend) -> shared::math::ClipY {
    // Every supported backend has an explicit clip-Y convention; no fallback can hide a new backend.
    match backend {
        Backend::Vulkan => shared::math::ClipY::Vulkan,
        Backend::Dx12 => shared::math::ClipY::Dx12,
        Backend::Metal => shared::math::ClipY::Metal,
    }
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
    shared::run_program("02_textured_cube", &backend, run_example_with_benchmark);
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
