//! Textured Sponza using safe ez-gfx context, resource, and frame interfaces.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use glam::{Mat4, Vec3};
use shared::{math::*, mesh::*, *};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

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
struct SceneParams {
    mvp: Mat4,
    primitive_count: u32,
    padding: [u32; 3],
}

fn primitive_ids(primitives: &[PrimitiveData], vertex_count: usize) -> anyhow::Result<Vec<u32>> {
    if primitives.is_empty() || vertex_count == 0 {
        anyhow::bail!("primitive identity requires nonempty primitives and vertices");
    }

    let mut ids = vec![u32::MAX; vertex_count];
    for (index, primitive) in primitives.iter().enumerate() {
        let start = usize::try_from(primitive.vertex_offset)?;
        let end = match primitives.get(index + 1) {
            Some(next) => usize::try_from(next.vertex_offset)?,
            None => vertex_count,
        };
        // The loader appends one contiguous vertex range per primitive; gaps or empty ranges
        // would make the shader's vertex-to-primitive lookup ambiguous.
        if (index == 0 && start != 0) || start >= end || end > vertex_count {
            anyhow::bail!("primitive vertex ranges are not contiguous");
        }
        ids[start..end].fill(u32::try_from(index)?);
    }
    if ids.iter().any(|id| *id == u32::MAX) {
        anyhow::bail!("primitive vertex ranges do not cover the mesh");
    }
    Ok(ids)
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

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("06_sponza_ktx2", WIDTH, HEIGHT, "ez_gfx_api2")?;

    let backend = backend_config(example.backend());

    let context = Context::new(ContextOptions {
        enable_debug: example.debug_enabled(),
        enable_validation: example.validation_enabled(),
        backend: backend.backend,
        texture_decode_workers: 0,
        adapter_selection: None,
    })?;
    let surface = context.create_surface_window(example.native_surface()?, false)?;
    example.register_observations(&context)?;
    let shader_bytes = ez_gfx_compiler::EasyGraphicsCompiler::compile_shader(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/06_sponza_ktx2/06_sponza_ktx2.slang"
        )),
        &[
            ez_gfx_compiler::Target::Spirv,
            ez_gfx_compiler::Target::Dxil,
            ez_gfx_compiler::Target::Metal,
        ],
        !cfg!(target_vendor = "apple"),
    )?;
    let mesh = load_textured_glb(include_bytes!("../shared/assets/sponza.glb"))?;
    let primitive_count = u32::try_from(mesh.primitives.len())?;
    let primitive_ids = primitive_ids(&mesh.primitives, mesh.positions.len())?;
    let primitive_bytes = u64::from(primitive_count)
        .checked_mul(u64::try_from(std::mem::size_of::<PrimitiveTextured>())?)
        .ok_or_else(|| anyhow::anyhow!("primitive records size overflow"))?;
    if primitive_bytes > 16 * 1024 * 1024 {
        anyhow::bail!("primitive records exceed ABI boundary");
    }
    let index_allocation = context.upload_indices(&mesh.indices)?;
    let (first_index, _) = index_allocation.range()?;
    let positions_heap = context.create_vertex_heap("positions")?;
    let _positions = positions_heap.upload(&mesh.positions)?;
    let normals_heap = context.create_vertex_heap("normals")?;
    let _normals = normals_heap.upload(&mesh.normals)?;
    let uvs_heap = context.create_vertex_heap("uvs")?;
    let _uvs = uvs_heap.upload(&mesh.uvs)?;
    let primitive_ids_heap = context.create_vertex_heap("primitive_ids")?;
    let _primitive_ids_buffer = primitive_ids_heap.upload(&primitive_ids)?;
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
    let fallback = context.load_texture(
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[255, 255, 255, 255],
        false,
        &fallback_config,
    )?;
    context.wait_idle()?;
    let fallback_binding = fallback.binding()?;
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
        let texture = match context.load_texture(TextureSource::Ktx2, &image.bytes, true, &config) {
            Ok(value) => value,
            Err(error) => {
                return Err(anyhow::anyhow!("{error:?}"));
            }
        };
        context.wait_idle()?;
        let binding = texture.binding()?;
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
    let compute_shader = shader_bytes.load_compute_shader(&context, "computemain")?;
    let vertex_shader = shader_bytes.load_vertex_shader(&context, "vertexmain")?;
    let fragment_shader = shader_bytes.load_fragment_shader(&context, "fragmentmain")?;
    let mut camera = OrbitCamera::new(90.0_f32.to_radians(), 8.0_f32.to_radians(), 0.45)
        .with_frustum(60.0_f32.to_radians(), 0.02, 100.0)
        .with_clip_y(shared::clip_y(backend.backend));
    let target = Vec3::new(0.0, -0.32, 0.0);
    let mut params = SceneParams {
        mvp: Mat4::IDENTITY,
        primitive_count,
        padding: [0; 3],
    };

    while let Some(window_frame) = example.wait_for_next_frame(&surface)? {
        let mut frame = surface.begin_frame()?;
        let swapchain_target = frame.configure_swapchain(window_frame.size, Format::Bgra8Srgb)?;
        camera.handle_window_events(&window_frame.events);
        params.mvp = row_major(camera.projection(window_frame.size)? * camera.view(target)?);
        let primitives = context.acquire_buffer_from(records.as_slice())?;
        let indirect =
            context.acquire_counter_buffer::<DrawIndexedCommand>(primitive_count as usize)?;
        let params_buffer = context.acquire_value_buffer(params)?;
        frame.bind_buffer("params", &params_buffer)?;
        frame.bind_buffer("primitives", &primitives)?;
        frame.bind_buffer("draw_commands", &indirect)?;
        frame.execute_compute(&compute_shader, [primitive_count, 1, 1])?;

        frame.execute_graphics(
            &vertex_shader,
            &fragment_shader,
            &indirect,
            DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap(),
        )?;
        example.handle_frame(frame, swapchain_target)?;
        example.update_title(&context);
    }
    Ok(())
}
