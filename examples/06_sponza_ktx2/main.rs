//! Textured Sponza using safe ez-gfx context, resource, and frame APIs.
mod renderer {
    use crate::shared::{input::*, math::*, mesh::*, *};
    use anyhow::Context as _;
    use ez_gfx::*;
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
        indirect: IndirectBufferHandle,
        camera: OrbitCamera,
        clip_y: ClipY,
        target: Vec3,
        near: f32,
        far: f32,
        push: ScenePush,
        primitive_count: u32,
        bindings: [PublicBinding; 6],
    }

    impl ModelScene {
        pub(super) fn create(context: ContextHandle, clip_y: ClipY) -> anyhow::Result<Self> {
            let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .context("examples package has no workspace parent")?;
            let shader_bytes = ez_gfx_compiler::compile_shader(
                &workspace_root.join("examples/06_sponza_ktx2/06_sponza_ktx2.slang"),
                &[
                    ez_gfx_compiler::Target::Spirv,
                    ez_gfx_compiler::Target::Dxil,
                    ez_gfx_compiler::Target::Metal,
                ],
                !cfg!(target_vendor = "apple"),
            )
            .context("compile Sponza shader")?;
            let mesh = load_textured_glb(include_bytes!("../shared/assets/sponza.glb"))?;
            let primitive_count =
                u32::try_from(mesh.primitives.len()).context("primitive count exceeds ABI")?;
            let primitive_ids = primitive_ids(&mesh.primitives, mesh.positions.len())?;
            let primitive_bytes = u64::from(primitive_count)
                .checked_mul(
                    u64::try_from(std::mem::size_of::<PrimitiveTextured>())
                        .context("primitive record size exceeds ABI")?,
                )
                .ok_or_else(|| anyhow::anyhow!("primitive records size overflow"))?;
            if primitive_bytes > 16 * 1024 * 1024 {
                anyhow::bail!("primitive records exceed ABI boundary");
            }
            let index_bytes = byte_len(&mesh.indices)?;
            let positions_bytes = byte_len(&mesh.positions)?;
            let normals_bytes = byte_len(&mesh.normals)?;
            let uvs_bytes = byte_len(&mesh.uvs)?;
            let primitive_ids_bytes = byte_len(&primitive_ids)?;
            status(
                create_index_heap(context, index_bytes),
                "create Sponza index heap",
            )?;
            let first_index = upload_indices(
                context,
                mesh.indices.len() as u32,
                slice_bytes(&mesh.indices),
            )
            .map_err(|error| anyhow::anyhow!("{error:?}"))
            .context("upload Sponza indices")?;
            let positions = acquire_structured(context, positions_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire positions")?;
            status(
                write_structured(context, positions, slice_bytes(&mesh.positions)),
                "upload positions",
            )?;
            let normals = acquire_structured(context, normals_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire normals")?;
            status(
                write_structured(context, normals, slice_bytes(&mesh.normals)),
                "upload normals",
            )?;
            let uvs = acquire_structured(context, uvs_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire uvs")?;
            status(
                write_structured(context, uvs, slice_bytes(&mesh.uvs)),
                "upload uvs",
            )?;
            let primitive_ids_buffer = acquire_structured(context, primitive_ids_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire primitive IDs")?;
            status(
                write_structured(context, primitive_ids_buffer, slice_bytes(&primitive_ids)),
                "upload primitive IDs",
            )?;
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
                destination: ez_gfx::TextureDestination::Rgba8Unorm,
                sampler: repeat_sampler,
            };
            let fallback = load_texture(
                context,
                TextureSource::Rgba8 {
                    width: 1,
                    height: 1,
                },
                &[255, 255, 255, 255],
                false,
                &fallback_config,
            )
            .map_err(|error| anyhow::anyhow!("{error:?}"))
            .context("load Sponza fallback")?;
            status(wait_idle(context), "wait for Sponza fallback")?;
            let fallback_binding = texture_binding(context, fallback)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("resolve fallback binding")?;
            let mut textures = vec![fallback];
            let mut image_bindings = Vec::with_capacity(mesh.images.len());
            for image in &mesh.images {
                if image.mime_type != "image/ktx2" {
                    anyhow::bail!("Sponza base-color image is not KTX2");
                }
                let config = TextureConfig {
                    width: 0,
                    height: 0,
                    mip_count: 0,
                    destination: ez_gfx::TextureDestination::Rgba8Unorm,
                    sampler: TextureSamplerDesc {
                        max_anisotropy: 16.0,
                        ..repeat_sampler
                    },
                };
                let texture =
                    match load_texture(context, TextureSource::Ktx2, &image.bytes, true, &config) {
                        Ok(value) => value,
                        Err(error) => {
                            return Err(anyhow::anyhow!("{error:?}").context("load Sponza KTX2"));
                        }
                    };
                status(wait_idle(context), "wait for Sponza texture")?;
                let binding = texture_binding(context, texture)
                    .map_err(|error| anyhow::anyhow!("{error:?}"))
                    .context("resolve Sponza texture binding")?;
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
            let primitives = acquire_structured(context, primitive_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire primitives")?;
            status(
                write_structured(context, primitives, slice_bytes(&records)),
                "upload primitives",
            )?;
            let indirect = acquire_indirect(context, primitive_count)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("acquire Sponza indirect commands")?;
            status(
                set_indirect_count(context, indirect, primitive_count),
                "set Sponza draw count",
            )?;
            let shader = load_shader(context, &shader_bytes)
                .map_err(|error| anyhow::anyhow!("{error:?}"))
                .context("load Sponza artifact")?;
            Ok(Self {
                shader,
                indirect,
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
                    PublicBinding {
                        name: "primitive_ids".to_owned(),
                        resource: ResourceIdentity::Structured(primitive_ids_buffer),
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
                    bytes_of(&self.push),
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
                    bytes_of(&self.push),
                ),
                "record Sponza graphics pipeline",
            )
        }
    }
    fn primitive_ids(
        primitives: &[PrimitiveData],
        vertex_count: usize,
    ) -> anyhow::Result<Vec<u32>> {
        if primitives.is_empty() || vertex_count == 0 {
            anyhow::bail!("primitive identity requires nonempty primitives and vertices");
        }

        let mut ids = vec![u32::MAX; vertex_count];
        for (index, primitive) in primitives.iter().enumerate() {
            let start =
                usize::try_from(primitive.vertex_offset).context("vertex offset exceeds ABI")?;
            let end = match primitives.get(index + 1) {
                Some(next) => {
                    usize::try_from(next.vertex_offset).context("vertex offset exceeds ABI")?
                }
                None => vertex_count,
            };
            // The loader appends one contiguous vertex range per primitive; gaps or empty ranges
            // would make the shader's vertex-to-primitive lookup ambiguous.
            if (index == 0 && start != 0) || start >= end || end > vertex_count {
                anyhow::bail!("primitive vertex ranges are not contiguous");
            }
            ids[start..end].fill(u32::try_from(index).context("primitive index exceeds ABI")?);
        }
        if ids.iter().any(|id| *id == u32::MAX) {
            anyhow::bail!("primitive vertex ranges do not cover the mesh");
        }
        Ok(ids)
    }

    fn status(result: EzGfxResult, operation: &str) -> anyhow::Result<()> {
        match result {
            EzGfxResult::Ok => Ok(()),
            error => Err(anyhow::anyhow!("{error:?}").context(operation.to_owned())),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn primitive(vertex_offset: u32) -> PrimitiveData {
            PrimitiveData {
                first_index: 0,
                index_count: 3,
                vertex_offset,
                normal_offset: vertex_offset,
                uv_offset: vertex_offset,
                transform: Mat4::IDENTITY,
                image: None,
            }
        }

        #[test]
        fn primitive_ids_cover_contiguous_vertex_ranges() {
            assert_eq!(
                primitive_ids(&[primitive(0), primitive(2)], 5).unwrap(),
                [0, 0, 1, 1, 1]
            );
        }

        #[test]
        fn primitive_ids_reject_empty_gapped_reversed_and_out_of_bounds_ranges() {
            for (primitives, vertices) in [
                (vec![], 0),
                (vec![primitive(0)], 0),
                (vec![primitive(1)], 2),
                (vec![primitive(0), primitive(0)], 2),
                (vec![primitive(2), primitive(1)], 3),
                (vec![primitive(0), primitive(3)], 2),
            ] {
                assert!(primitive_ids(&primitives, vertices).is_err());
            }
        }
    }
}

#[path = "../shared/mod.rs"]
mod shared;

use anyhow::Context as _;
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
            texture_decode_workers: 0,
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
        self.resources = Some(ExampleScene::create(
            self.context(),
            shared::clip_y(backend),
        )?);
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

fn status(result: EzGfxResult, operation: &str) -> anyhow::Result<()> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(anyhow::anyhow!("{error:?}").context(operation.to_owned())),
    }
}

fn main() {
    let backend = backend_name().unwrap_or_else(|error| {
        eprintln!("{error:#}");
        std::process::exit(2);
    });
    run_program("06_sponza_ktx2", backend, run_example_with_benchmark);
}
