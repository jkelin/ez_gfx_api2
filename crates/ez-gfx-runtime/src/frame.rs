use crate::{
    graph::CompiledGraph,
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer, IndirectError},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameState {
    Idle,
    Recording,
    Submitted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameSubmission {
    pub commands: Vec<DrawIndexedCommand>,
}

pub struct FrameRecorder {
    state: FrameState,
    indirect: IndexedIndirectBuffer,
    graph_enqueued: bool,
}
impl FrameRecorder {
    pub fn new(indirect_capacity: u32) -> Result<Self, FrameError> {
        Ok(Self {
            state: FrameState::Idle,
            indirect: IndexedIndirectBuffer::new(indirect_capacity).map_err(map_indirect)?,
            graph_enqueued: false,
        })
    }

    pub const fn state(&self) -> FrameState {
        self.state
    }

    pub fn begin(&mut self) -> Result<(), FrameError> {
        if self.state != FrameState::Idle {
            return Err(FrameError::AlreadyRecording);
        }
        self.state = FrameState::Recording;
        self.graph_enqueued = false;
        self.indirect.set_draw_count(0).map_err(map_indirect)
    }

    pub fn write_indirect(
        &mut self,
        index: u32,
        command: DrawIndexedCommand,
    ) -> Result<(), FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.indirect.write(index, command).map_err(map_indirect)?;
        self.indirect
            .set_draw_count(
                self.indirect.draw_count().max(
                    index
                        .checked_add(1)
                        .ok_or(FrameError::IndirectOutOfBounds)?,
                ),
            )
            .map_err(map_indirect)
    }

    /// The compiled graph is consumed because its ordering/barrier decision belongs to exactly one frame.
    pub fn enqueue(&mut self, _graph: CompiledGraph) -> Result<(), FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        if self.graph_enqueued {
            return Err(FrameError::GraphAlreadyEnqueued);
        }
        self.graph_enqueued = true;
        Ok(())
    }

    /// Marks backend-native work as the frame workload; repeated pipeline additions remain one submission.
    pub fn mark_work_enqueued(&mut self) -> Result<(), FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.graph_enqueued = true;
        Ok(())
    }

    pub fn submit(&mut self) -> Result<FrameSubmission, FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        if !self.graph_enqueued {
            return Err(FrameError::MissingGraph);
        }
        self.state = FrameState::Submitted;
        Ok(FrameSubmission {
            commands: self.indirect.commands().to_vec(),
        })
    }

    pub fn finish(&mut self) -> Result<(), FrameError> {
        if self.state != FrameState::Submitted {
            return Err(FrameError::NotSubmitted);
        }
        self.state = FrameState::Idle;
        Ok(())
    }
}

fn map_indirect(error: IndirectError) -> FrameError {
    match error {
        IndirectError::OutOfBounds => FrameError::IndirectOutOfBounds,
        _ => FrameError::InvalidCapacity,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameError {
    InvalidCapacity,
    AlreadyRecording,
    NotRecording,
    GraphAlreadyEnqueued,
    MissingGraph,
    IndirectOutOfBounds,
    NotSubmitted,
}
