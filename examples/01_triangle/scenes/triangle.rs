use std::ffi::CString;

use ez_gfx_ffi::EzGfxDrawIndexedCommand;

use super::{
    FrameInput, SceneInput, SceneState,
    common::{IndexHeap, Indirect, Shader, Structured, binding, record_graphics},
};

pub struct Triangle {
    shader: Shader,
    positions: Structured,
    indirect: Indirect,
    indices: IndexHeap,
    positions_name: CString,
}

impl Triangle {
    pub fn create(context: u64) -> Result<Self, String> {
        // Metal and Vulkan map raw clip-space Y oppositely when no projection matrix is present.
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
        let (indices, first_index) = IndexHeap::upload(context, &[0, 1, 2], "triangle indices")?;
        let structured = Structured::upload(context, "positions", &positions)?;
        let indirect = Indirect::create(context, 1, "triangle draw")?;
        indirect.write(
            context,
            0,
            EzGfxDrawIndexedCommand {
                index_count: 3,
                instance_count: 1,
                first_index,
                vertex_offset: 0,
                first_instance: 0,
            },
        )?;
        indirect.set_count(context, 1)?;
        Ok(Self {
            shader: Shader::load(context, include_bytes!("../01_triangle.ezgfx"), false)?,
            positions: structured,
            indirect,
            indices,
            positions_name: CString::new("positions").unwrap(),
        })
    }
}

impl SceneState for Triangle {
    fn handle_input(&mut self, _input: SceneInput) {}
    fn update(&mut self, _frame: FrameInput) -> Result<(), String> {
        Ok(())
    }
    fn record(&mut self, context: u64) -> Result<(), String> {
        let bindings = [binding(&self.positions_name, self.positions.0, 0)];
        record_graphics(
            context,
            self.shader.0,
            self.indirect.handle,
            &bindings,
            None,
            &[],
        )
    }
    fn destroy(self: Box<Self>, context: u64) {
        self.indirect.destroy(context);
        self.positions.destroy(context);
        self.indices.destroy(context);
        self.shader.destroy(context);
    }
}
