use std::ffi::CString;

use ez_gfx_ffi::{EzGfxDrawIndexedCommand, EzGfxDynamicState};

use super::{
    FrameInput, SceneInput, SceneState,
    common::*,
    math::{Mat4, OrbitCamera, identity, mul, perspective},
};

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    mvp: Mat4,
    texture_id: u32,
    padding: [u32; 3],
}

pub struct TexturedCube {
    shader: Shader,
    positions: Structured,
    indirect: Indirect,
    indices: IndexHeap,
    texture: Texture,
    camera: OrbitCamera,
    push: Push,
    positions_name: CString,
}

impl TexturedCube {
    pub fn create(context: u64) -> Result<Self, String> {
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
            0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16,
            17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
        ];
        let (index_heap, first_index) = IndexHeap::upload(context, &indices, "cube indices")?;
        let positions_buffer = Structured::upload(context, "positions", &positions)?;
        let indirect = Indirect::create(context, 1, "cube draw")?;
        indirect.write(
            context,
            0,
            EzGfxDrawIndexedCommand {
                index_count: indices.len() as u32,
                instance_count: 1,
                first_index,
                vertex_offset: 0,
                first_instance: 0,
            },
        )?;
        indirect.set_count(context, 1)?;
        let texture = Texture::load(
            context,
            include_bytes!("../cube.png"),
            4,
            0,
            0,
            true,
            1.0,
            false,
            "cube texture",
        )?;
        Ok(Self {
            shader: Shader::load(context, include_bytes!("../02_textured_cube.ezgfx"), false)?,
            positions: positions_buffer,
            indirect,
            indices: index_heap,
            push: Push {
                mvp: identity(),
                texture_id: texture.binding,
                padding: [0; 3],
            },
            texture,
            camera: OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0),
            positions_name: CString::new("positions").unwrap(),
        })
    }
}

impl SceneState for TexturedCube {
    fn handle_input(&mut self, input: SceneInput) {
        match input {
            SceneInput::CursorMoved { x, y } => self.camera.cursor([x, y]),
            SceneInput::PrimaryButton(value) => self.camera.set_dragging(value),
            SceneInput::ScrollLines(lines) => self.camera.zoom(lines),
            _ => {}
        }
    }
    fn update(&mut self, frame: FrameInput) -> Result<(), String> {
        let aspect = frame.width as f32 / frame.height as f32;
        self.push.mvp = mul(
            perspective(60.0_f32.to_radians(), aspect, 0.1, 100.0)?,
            mul(self.camera.view([0.0; 3])?, identity()),
        );
        Ok(())
    }
    fn record(&mut self, context: u64) -> Result<(), String> {
        let bindings = [binding(&self.positions_name, self.positions.0, 0)];
        let dynamic = EzGfxDynamicState {
            cull_mode: 2,
            front_face: 0,
            primitive_type: 0,
            blend_mode: 0,
        };
        record_graphics(
            context,
            self.shader.0,
            self.indirect.handle,
            &bindings,
            Some(dynamic),
            bytes_of(&self.push),
        )
    }
    fn destroy(self: Box<Self>, context: u64) {
        self.texture.destroy(context);
        self.indirect.destroy(context);
        self.positions.destroy(context);
        self.indices.destroy(context);
        self.shader.destroy(context);
    }
}
