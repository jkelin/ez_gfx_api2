//! Textured Sponza using safe ez-gfx context, resource, and frame APIs.
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
        math::{ClipY, OrbitCamera, perspective, row_major},
        mesh::load_textured_glb,
    };
    use anyhow::Context as _;
    use ez_gfx::{
        ContextHandle, DynamicPipelineState, EzGfxResult, IndirectBufferHandle, PublicBinding,
        ResourceIdentity, SamplerAddressMode, SamplerFilter, ShaderHandle, StructuredBufferHandle,
        TextureConfig, TextureHandle, TextureSamplerDesc, TextureSource, acquire_indirect,
        acquire_structured, create_index_heap, destroy_index_heap, destroy_shader, load_shader,
        load_texture, release_indirect, release_structured, render_add_compute,
        render_add_graphics, set_indirect_count, texture_binding, unload_texture, upload_indices,
        write_structured,
    };
    use glam::{DVec2, Mat4, Vec3};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct PrimitiveTextured {
        first_index: u32,
        index_count: u32,
        vertex_offset: u32,
        normal_offset: u32,
        uv_offset: u32,
        texture_id: u32,
        padding: [u32; 2],
        transform: Mat4,
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScenePush {
        mvp: Mat4,
        primitive_count: u32,
        padding: [u32; 3],
    }
    pub(super) struct ModelScene {
        shader: ShaderHandle,
        positions: StructuredBufferHandle,
        normals: StructuredBufferHandle,
        uvs: StructuredBufferHandle,
        primitives: StructuredBufferHandle,
        indirect: IndirectBufferHandle,
        textures: Vec<TextureHandle>,
        camera: OrbitCamera,
        clip_y: ClipY,
        target: Vec3,
        near: f32,
        far: f32,
        push: ScenePush,
        primitive_count: u32,
        bindings: [PublicBinding; 5],
    }

    impl ModelScene {
        #[allow(
            clippy::too_many_lines,
            reason = "The example keeps every safe resource acquisition and failure cleanup visible in one linear flow."
        )]
        pub(super) fn create(context: ContextHandle, clip_y: ClipY) -> anyhow::Result<Self> {
            let mesh = load_textured_glb(include_bytes!("../shared/assets/sponza.glb"))?;
            let primitive_count =
                u32::try_from(mesh.primitives.len()).context("primitive count exceeds ABI")?;
            let primitive_bytes = u64::from(primitive_count)
                .checked_mul(
                    u64::try_from(std::mem::size_of::<PrimitiveTextured>())
                        .context("primitive record size exceeds ABI")?,
                )
                .ok_or_else(|| anyhow::anyhow!("primitive records size overflow"))?;
            if primitive_bytes > 16 * 1024 * 1024 {
                anyhow::bail!("primitive records exceed ABI boundary");
            }
            let index_bytes = shared::byte_len(&mesh.indices)?;
            let positions_bytes = shared::byte_len(&mesh.positions)?;
            let normals_bytes = shared::byte_len(&mesh.normals)?;
            let uvs_bytes = shared::byte_len(&mesh.uvs)?;
            status(
                create_index_heap(context, index_bytes),
                "create Sponza index heap",
            )?;
            let first_index = match upload_indices(
                context,
                mesh.indices.len() as u32,
                shared::slice_bytes(&mesh.indices),
            ) {
                Ok(value) => value,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("upload Sponza indices"));
                }
            };
            let positions = match acquire_structured(context, positions_bytes) {
                Ok(handle) => handle,
                Err(error) => {
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("acquire positions"));
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
                Ok(handle) => handle,
                Err(error) => {
                    release_structured(context, positions);
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("acquire normals"));
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
            let uvs = match acquire_structured(context, uvs_bytes) {
                Ok(handle) => handle,
                Err(error) => {
                    release_structured(context, normals);
                    release_structured(context, positions);
                    destroy_index_heap(context);
                    return Err(anyhow::anyhow!("{error:?}").context("acquire uvs"));
                }
            };
            if let Err(error) = status(
                write_structured(context, uvs, shared::slice_bytes(&mesh.uvs)),
                "upload uvs",
            ) {
                release_structured(context, uvs);
                release_structured(context, normals);
                release_structured(context, positions);
                destroy_index_heap(context);
                return Err(error);
            }
            let cleanup_base = |textures: Vec<TextureHandle>| {
                for texture in textures {
                    unload_texture(context, texture);
                }
                release_structured(context, uvs);
                release_structured(context, normals);
                release_structured(context, positions);
                destroy_index_heap(context);
            };
            let repeat_sampler = TextureSamplerDesc {
                min_filter: SamplerFilter::Linear,
                mag_filter: SamplerFilter::Linear,
                max_anisotropy: 1.0,
                address_u: SamplerAddressMode::Repeat,
                address_v: SamplerAddressMode::Repeat,
                address_w: SamplerAddressMode::Repeat,
            };
            let fallback_config = TextureConfig {
                width: 1,
                height: 1,
                mip_count: 0,
                sampler: repeat_sampler,
            };
            let fallback = match load_texture(
                context,
                TextureSource::Rgba8 {
                    width: 1,
                    height: 1,
                },
                &[255, 255, 255, 255],
                false,
                &fallback_config,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    cleanup_base(Vec::new());
                    return Err(anyhow::anyhow!("{error:?}").context("load Sponza fallback"));
                }
            };
            let fallback_binding = match texture_binding(context, fallback) {
                Ok(binding) => binding,
                Err(error) => {
                    cleanup_base(vec![fallback]);
                    return Err(anyhow::anyhow!("{error:?}").context("resolve fallback binding"));
                }
            };
            let mut textures = vec![fallback];
            let mut image_bindings = Vec::with_capacity(mesh.images.len());
            for image in &mesh.images {
                if image.mime_type != "image/ktx2" {
                    cleanup_base(textures);
                    anyhow::bail!("Sponza base-color image is not KTX2");
                }
                let config = TextureConfig {
                    width: 0,
                    height: 0,
                    mip_count: 0,
                    sampler: TextureSamplerDesc {
                        max_anisotropy: 16.0,
                        ..repeat_sampler
                    },
                };
                let texture =
                    match load_texture(context, TextureSource::Ktx2, &image.bytes, true, &config) {
                        Ok(value) => value,
                        Err(error) => {
                            cleanup_base(textures);
                            return Err(anyhow::anyhow!("{error:?}").context("load Sponza KTX2"));
                        }
                    };
                let binding = match texture_binding(context, texture) {
                    Ok(value) => value,
                    Err(error) => {
                        unload_texture(context, texture);
                        cleanup_base(textures);
                        return Err(
                            anyhow::anyhow!("{error:?}").context("resolve Sponza texture binding")
                        );
                    }
                };
                image_bindings.push(binding);
                textures.push(texture);
            }
            let records = mesh
                .primitives
                .iter()
                .map(|primitive| PrimitiveTextured {
                    first_index: primitive.first_index + first_index,
                    index_count: primitive.index_count,
                    vertex_offset: primitive.vertex_offset,
                    normal_offset: primitive.normal_offset,
                    uv_offset: primitive.uv_offset,
                    texture_id: primitive
                        .image
                        .and_then(|index| image_bindings.get(index).copied())
                        .unwrap_or(fallback_binding),
                    padding: [0; 2],
                    transform: row_major(primitive.transform),
                })
                .collect::<Vec<_>>();
            let primitives = match acquire_structured(context, primitive_bytes) {
                Ok(handle) => handle,
                Err(error) => {
                    cleanup_base(textures);
                    return Err(anyhow::anyhow!("{error:?}").context("acquire primitives"));
                }
            };
            if let Err(error) = status(
                write_structured(context, primitives, shared::slice_bytes(&records)),
                "upload primitives",
            ) {
                release_structured(context, primitives);
                for texture in textures {
                    unload_texture(context, texture);
                }
                release_structured(context, uvs);
                release_structured(context, normals);
                release_structured(context, positions);
                destroy_index_heap(context);
                return Err(error);
            }
            let indirect = match acquire_indirect(context, primitive_count) {
                Ok(handle) => handle,
                Err(error) => {
                    release_structured(context, primitives);
                    cleanup_base(textures);
                    return Err(
                        anyhow::anyhow!("{error:?}").context("acquire Sponza indirect commands")
                    );
                }
            };
            if let Err(error) = status(
                set_indirect_count(context, indirect, primitive_count),
                "set Sponza draw count",
            ) {
                release_indirect(context, indirect);
                release_structured(context, primitives);
                cleanup_base(textures);
                return Err(error);
            }
            let shader = match load_shader(
                context,
                include_bytes!(concat!(env!("OUT_DIR"), "/06_sponza_ktx2.ezgfxshader")),
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    release_indirect(context, indirect);
                    release_structured(context, primitives);
                    cleanup_base(textures);
                    return Err(anyhow::anyhow!("{error:?}").context("load Sponza artifact"));
                }
            };
            Ok(Self {
                shader,
                positions,
                normals,
                uvs,
                primitives,
                indirect,
                textures,
                camera: OrbitCamera::new(90.0_f32.to_radians(), 8.0_f32.to_radians(), 0.45),
                clip_y,
                target: Vec3::new(0.0, -0.32, 0.0),
                near: 0.02,
                far: 100.0,
                push: ScenePush {
                    mvp: Mat4::IDENTITY,
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
                    PublicBinding {
                        name: "uvs".to_owned(),
                        resource: ResourceIdentity::Structured(uvs),
                    },
                ],
            })
        }
    }
    impl ModelScene {
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
            let bindings = &self.bindings;
            status(
                render_add_compute(
                    context,
                    self.shader,
                    [self.primitive_count, 1, 1],
                    bindings,
                    shared::bytes_of(&self.push),
                ),
                "record Sponza compute pipeline",
            )?;
            status(
                render_add_graphics(
                    context,
                    self.shader,
                    self.indirect,
                    bindings,
                    DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap(),
                    shared::bytes_of(&self.push),
                ),
                "record Sponza graphics pipeline",
            )
        }
        pub(super) fn destroy(self, context: ContextHandle) {
            for texture in self.textures {
                unload_texture(context, texture);
            }
            release_indirect(context, self.indirect);
            release_structured(context, self.primitives);
            release_structured(context, self.uvs);
            release_structured(context, self.normals);
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
use renderer::ModelScene as ExampleScene;
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
    shared::run_program("06_sponza_ktx2", &backend, run_example_with_benchmark);
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
