#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// Parameters for one indexed indirect draw.
pub struct DrawIndexedCommand {
    /// Number of indices drawn per instance.
    pub index_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Offset of the first index in the bound index buffer.
    pub first_index: u32,
    /// Signed offset added to each fetched vertex index.
    pub vertex_offset: i32,
    /// Base instance passed to the draw.
    pub first_instance: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Fixed-capacity storage for indexed indirect draw commands.
pub struct IndexedIndirectBuffer {
    /// Allocated command slots, including unpublished entries.
    commands: Vec<DrawIndexedCommand>,
    /// Number of command slots exposed for drawing.
    draw_count: u32,
}

impl IndexedIndirectBuffer {
    /// Allocates `capacity` command slots with no draws published.
    ///
    /// # Errors
    ///
    /// Returns `IndirectError::InvalidCapacity` if `capacity` is zero.
    pub fn new(capacity: u32) -> Result<Self, IndirectError> {
        if capacity == 0 {
            return Err(IndirectError::InvalidCapacity);
        }
        Ok(Self {
            commands: vec![DrawIndexedCommand::default(); capacity as usize],
            draw_count: 0,
        })
    }

    /// Replaces a contiguous command range and publishes its prefix.
    ///
    /// The active count advances to `max(previous_count, start_index + commands.len())`.
    /// An empty batch is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`IndirectError::OutOfBounds`] when the checked element range
    /// exceeds the allocated command capacity.
    pub fn write_batch(
        &mut self,
        start_index: u32,
        commands: &[DrawIndexedCommand],
    ) -> Result<(), IndirectError> {
        if commands.is_empty() {
            return Ok(());
        }
        let count = u32::try_from(commands.len()).map_err(|_| IndirectError::OutOfBounds)?;
        let end = start_index
            .checked_add(count)
            .ok_or(IndirectError::OutOfBounds)?;
        let start = usize::try_from(start_index).map_err(|_| IndirectError::OutOfBounds)?;
        let end_index = usize::try_from(end).map_err(|_| IndirectError::OutOfBounds)?;
        let destination = self
            .commands
            .get_mut(start..end_index)
            .ok_or(IndirectError::OutOfBounds)?;

        destination.copy_from_slice(commands);
        self.draw_count = self.draw_count.max(end);
        Ok(())
    }

    /// Publishes the active count written by a GPU producer.
    ///
    /// CPU writes publish their range through [`Self::write_batch`]. This
    /// explicit path exists only because GPU command generation cannot update
    /// the CPU-side publication metadata.
    ///
    /// # Errors
    ///
    /// Returns [`IndirectError::OutOfBounds`] if `count` exceeds capacity.
    pub fn publish_generated_count(&mut self, count: u32) -> Result<(), IndirectError> {
        if count as usize > self.commands.len() {
            return Err(IndirectError::OutOfBounds);
        }
        self.draw_count = count;
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.draw_count = 0;
    }

    /// Returns the number of commands currently published for drawing.
    pub const fn draw_count(&self) -> u32 {
        self.draw_count
    }
    /// Returns the published prefix of the command storage.
    pub fn commands(&self) -> &[DrawIndexedCommand] {
        &self.commands[..self.draw_count as usize]
    }
    /// Compute producers may fill the standard command storage, but must publish draw count separately.
    pub fn commands_mut(&mut self) -> &mut [DrawIndexedCommand] {
        &mut self.commands
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Floating-point viewport bounds and depth range.
pub struct Viewport {
    /// Horizontal origin in framebuffer coordinates.
    pub x: f32,
    /// Vertical origin in framebuffer coordinates.
    pub y: f32,
    /// Positive horizontal extent in framebuffer coordinates.
    pub width: f32,
    /// Positive vertical extent in framebuffer coordinates.
    pub height: f32,
    /// Lower normalized depth bound.
    pub min_depth: f32,
    /// Upper normalized depth bound.
    pub max_depth: f32,
}
impl Viewport {
    /// Coordinates and extents must be finite; depth remains normalized and ordered.
    ///
    /// # Errors
    ///
    /// Returns `IndirectError::InvalidViewport` if any value is non-finite, an extent is non-positive, or the depth range is outside `0.0..=1.0` or reversed.
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
/// Integer scissor rectangle in framebuffer coordinates.
pub struct Scissor {
    /// Horizontal origin of the scissor rectangle.
    pub x: i32,
    /// Vertical origin of the scissor rectangle.
    pub y: i32,
    /// Nonzero horizontal extent of the scissor rectangle.
    pub width: u32,
    /// Nonzero vertical extent of the scissor rectangle.
    pub height: u32,
}
impl Scissor {
    /// Empty extents and signed-end overflow are rejected before backend conversion.
    ///
    /// # Errors
    ///
    /// Returns `IndirectError::InvalidScissor` if an extent is zero or exceeds `i32::MAX`, or if adding an extent to its origin overflows `i32`.
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Result<Self, IndirectError> {
        let width_i32 = i32::try_from(width).map_err(|_| IndirectError::InvalidScissor)?;
        let height_i32 = i32::try_from(height).map_err(|_| IndirectError::InvalidScissor)?;
        if width == 0
            || height == 0
            || x.checked_add(width_i32).is_none()
            || y.checked_add(height_i32).is_none()
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
/// Dynamic viewport and scissor state for a draw.
pub struct DynamicView {
    /// Viewport applied during rasterization.
    pub viewport: Viewport,
    /// Scissor rectangle applied during rasterization.
    pub scissor: Scissor,
}
impl DynamicView {
    /// # Errors
    ///
    /// This constructor currently cannot fail, but retains a `Result` for API symmetry.
    pub const fn new(viewport: Viewport, scissor: Scissor) -> Result<Self, IndirectError> {
        Ok(Self { viewport, scissor })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Consecutive indirect draws sharing one dynamic view.
pub struct DrawBatch {
    /// Index of the first indirect command in the batch.
    pub first: u32,
    /// Number of consecutive indirect commands in the batch.
    pub count: u32,
    /// Dynamic view shared by every draw in the batch.
    pub state: DynamicView,
}

/// Only consecutive equal state is grouped, preserving indirect command order and compute-filled indices.
///
/// # Panics
///
/// The caller must provide no more states than the u32 indirect index space.
pub fn group_dynamic_views(states: &[DynamicView]) -> Vec<DrawBatch> {
    let mut batches: Vec<DrawBatch> = Vec::new();
    for (index, state) in states.iter().copied().enumerate() {
        match batches.last_mut() {
            Some(batch) if batch.state == state => batch.count += 1,
            _ => batches.push(DrawBatch {
                first: u32::try_from(index).expect("validated index fits u32"),
                count: 1,
                state,
            }),
        }
    }
    batches
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Validation failures for indirect draws and dynamic raster state.
pub enum IndirectError {
    /// Command storage capacity was zero.
    InvalidCapacity,
    /// A command index or published draw count exceeded capacity.
    OutOfBounds,
    /// Viewport coordinates, extents, or depth bounds were invalid.
    InvalidViewport,
    /// Scissor extents were empty, unrepresentable, or overflowed their origin.
    InvalidScissor,
}
