#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct DrawIndexedCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedIndirectBuffer {
    commands: Vec<DrawIndexedCommand>,
    draw_count: u32,
}

impl IndexedIndirectBuffer {
    pub fn new(capacity: u32) -> Result<Self, IndirectError> {
        if capacity == 0 {
            return Err(IndirectError::InvalidCapacity);
        }
        Ok(Self {
            commands: vec![DrawIndexedCommand::default(); capacity as usize],
            draw_count: 0,
        })
    }

    pub fn write(&mut self, index: u32, command: DrawIndexedCommand) -> Result<(), IndirectError> {
        let destination = self
            .commands
            .get_mut(index as usize)
            .ok_or(IndirectError::OutOfBounds)?;
        *destination = command;
        Ok(())
    }

    pub fn set_draw_count(&mut self, count: u32) -> Result<(), IndirectError> {
        if count as usize > self.commands.len() {
            return Err(IndirectError::OutOfBounds);
        }
        self.draw_count = count;
        Ok(())
    }

    pub const fn draw_count(&self) -> u32 {
        self.draw_count
    }
    pub fn commands(&self) -> &[DrawIndexedCommand] {
        &self.commands[..self.draw_count as usize]
    }
    /// Compute producers may fill the standard command storage, but must publish draw count separately.
    pub fn commands_mut(&mut self) -> &mut [DrawIndexedCommand] {
        &mut self.commands
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}
impl Viewport {
    /// Coordinates and extents must be finite; depth remains normalized and ordered.
    pub fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<Self, IndirectError> {
        if ![x, y, width, height, min_depth, max_depth]
            .iter()
            .all(|value| value.is_finite())
            || width <= 0.0
            || height <= 0.0
            || min_depth < 0.0
            || max_depth > 1.0
            || min_depth > max_depth
        {
            return Err(IndirectError::InvalidViewport);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Scissor {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}
impl Scissor {
    /// Empty extents and signed-end overflow are rejected before backend conversion.
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Result<Self, IndirectError> {
        if width == 0
            || height == 0
            || width > i32::MAX as u32
            || height > i32::MAX as u32
            || x.checked_add(width as i32).is_none()
            || y.checked_add(height as i32).is_none()
        {
            return Err(IndirectError::InvalidScissor);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicView {
    pub viewport: Viewport,
    pub scissor: Scissor,
}
impl DynamicView {
    pub const fn new(viewport: Viewport, scissor: Scissor) -> Result<Self, IndirectError> {
        Ok(Self { viewport, scissor })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawBatch {
    pub first: u32,
    pub count: u32,
    pub state: DynamicView,
}

/// Only consecutive equal state is grouped, preserving indirect command order and compute-filled indices.
pub fn group_dynamic_views(states: &[DynamicView]) -> Vec<DrawBatch> {
    let mut batches: Vec<DrawBatch> = Vec::new();
    for (index, state) in states.iter().copied().enumerate() {
        match batches.last_mut() {
            Some(batch) if batch.state == state => batch.count += 1,
            _ => batches.push(DrawBatch {
                first: index as u32,
                count: 1,
                state,
            }),
        }
    }
    batches
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectError {
    InvalidCapacity,
    OutOfBounds,
    InvalidViewport,
    InvalidScissor,
}
