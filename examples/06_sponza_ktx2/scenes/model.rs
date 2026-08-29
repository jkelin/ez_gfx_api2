use std::ffi::CString;

use ez_gfx_ffi::EzGfxDynamicState;

use super::{
    FrameInput, SceneInput, SceneState,
    common::*,
    math::{Mat4, OrbitCamera, mul, perspective},
    mesh::load_glb,
};

#[repr(C)]
#[derive(Clone, Copy)]
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
#[derive(Clone, Copy)]
struct ScenePush {
    mvp: Mat4,
    primitive_count: u32,
    padding: [u32; 3],
}

pub struct ModelScene {
    shader: Shader,
    positions: Structured,
    normals: Structured,
    uvs: Structured,
    primitives: Structured,
    indirect: Indirect,
    indices: IndexHeap,
    textures: Vec<Texture>,
    camera: OrbitCamera,
    target: [f32; 3],
    near: f32,
    far: f32,
    push: ScenePush,
    primitive_count: u32,
    positions_name: CString,
    normals_name: CString,
    uvs_name: CString,
    primitives_name: CString,
    indirect_name: CString,
}

impl ModelScene {
    pub fn create(context: u64) -> Result<Self, String> {
        let bytes = include_bytes!("../../shared/assets/sponza.glb").as_slice();
        let camera = OrbitCamera::new(90.0_f32.to_radians(), 8.0_f32.to_radians(), 0.45);
        let target = [0.0, -0.32, 0.0];
        let near = 0.02;
        let far = 100.0;
        let artifact = include_bytes!("../06_sponza_ktx2.ezgfx").as_slice();
        let mesh = load_glb(bytes)?;
        let primitive_count = u32::try_from(mesh.primitives.len())
            .map_err(|_| "primitive count exceeds ABI".to_owned())?;
        let (indices, first_index) = IndexHeap::upload(context, &mesh.indices, "model indices")?;
        let positions = Structured::upload(context, "positions", &mesh.positions)?;
        let normals = Structured::upload(context, "normals", &mesh.normals)?;
        let uvs = Structured::upload(context, "uvs", &mesh.uvs)?;
        let mut textures = Vec::new();
        let fallback = Texture::load(
            context,
            &[255, 255, 255, 255],
            1,
            1,
            1,
            false,
            1.0,
            true,
            "Sponza fallback",
        )?;
        let fallback_binding = fallback.binding;
        textures.push(fallback);
        let mut image_bindings = Vec::with_capacity(mesh.images.len());
        for image in &mesh.images {
            if image.mime_type != "image/ktx2" {
                return Err("Sponza base-color image is not KTX2".to_owned());
            }
            let texture = Texture::load(
                context,
                &image.bytes,
                6,
                0,
                0,
                true,
                16.0,
                true,
                "Sponza KTX2",
            )?;
            image_bindings.push(texture.binding);
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
                transform: primitive.transform,
            })
            .collect::<Vec<_>>();
        let primitive_bytes = records.len() * std::mem::size_of::<PrimitiveTextured>();
        let primitives = Structured::upload(context, "primitives", &records)?;
        if primitive_bytes > 16 * 1024 * 1024 {
            return Err("primitive records exceed ABI boundary".to_owned());
        }
        let indirect = Indirect::create(context, primitive_count, "compute draw commands")?;
        indirect.set_count(context, primitive_count)?;
        Ok(Self {
            shader: Shader::load(context, artifact, true)?,
            positions,
            normals,
            uvs,
            primitives,
            indirect,
            indices,
            textures,
            camera,
            target,
            near,
            far,
            push: ScenePush {
                mvp: [0.0; 16],
                primitive_count,
                padding: [0; 3],
            },
            primitive_count,
            positions_name: CString::new("positions").unwrap(),
            normals_name: CString::new("normals").unwrap(),
            uvs_name: CString::new("uvs").unwrap(),
            primitives_name: CString::new("primitives").unwrap(),
            indirect_name: CString::new("draw_commands").unwrap(),
        })
    }
}

impl SceneState for ModelScene {
    fn handle_input(&mut self, input: SceneInput) {
        match input {
            SceneInput::CursorMoved { x, y } => self.camera.cursor([x, y]),
            SceneInput::PrimaryButton(value) => self.camera.set_dragging(value),
            SceneInput::ScrollLines(lines) => self.camera.zoom(lines),
            _ => {}
        }
    }

    fn update(&mut self, frame: FrameInput) -> Result<(), String> {
        let projection = perspective(
            60.0_f32.to_radians(),
            frame.width as f32 / frame.height as f32,
            self.near,
            self.far,
        )?;
        self.push.mvp = mul(projection, self.camera.view(self.target)?);
        Ok(())
    }

    fn record(&mut self, context: u64) -> Result<(), String> {
        let compute_bindings = [
            binding(&self.positions_name, self.positions.0, 0),
            binding(&self.normals_name, self.normals.0, 0),
            binding(&self.primitives_name, self.primitives.0, 0),
            binding(&self.indirect_name, 0, self.indirect.handle),
            binding(&self.uvs_name, self.uvs.0, 0),
        ];
        record_compute(
            context,
            self.shader.0,
            self.primitive_count,
            &compute_bindings,
            bytes_of(&self.push),
        )?;
        let graphics_bindings = [
            binding(&self.positions_name, self.positions.0, 0),
            binding(&self.normals_name, self.normals.0, 0),
            binding(&self.primitives_name, self.primitives.0, 0),
            binding(&self.indirect_name, 0, self.indirect.handle),
            binding(&self.uvs_name, self.uvs.0, 0),
        ];
        let dynamic = Some(EzGfxDynamicState {
            cull_mode: 2,
            front_face: 0,
            primitive_type: 0,
            blend_mode: 0,
        });
        record_graphics(
            context,
            self.shader.0,
            self.indirect.handle,
            &graphics_bindings,
            dynamic,
            bytes_of(&self.push),
        )
    }

    fn destroy(self: Box<Self>, context: u64) {
        for texture in self.textures {
            texture.destroy(context);
        }
        self.indirect.destroy(context);
        self.primitives.destroy(context);
        self.uvs.destroy(context);
        self.normals.destroy(context);
        self.positions.destroy(context);
        self.indices.destroy(context);
        self.shader.destroy(context);
    }
}
