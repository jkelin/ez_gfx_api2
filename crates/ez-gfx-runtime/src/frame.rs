use crate::{
    binding::{PipelineLayout, PublicBinding, ReflectedBindings},
    graph::{CompiledGraph, FrameGraph, GraphError, NodeDesc, NodeId, ResourceDesc, ResourceId},
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer, IndirectError},
};
use ez_gfx_hal::{CompletionToken, DynamicPipelineState, ResourceState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameState {
    Idle,
    Recording,
    Submitted,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutableNode {
    Graphics {
        shader: u64,
        indirect: u64,
        draw_count: u32,
        bindings: Vec<PublicBinding>,
        layout: ReflectedBindings,
        pipeline_layout: PipelineLayout,
        state: DynamicPipelineState,
        push_constants: Vec<u8>,
    },
    Compute {
        shader: u64,
        groups: [u32; 3],
        bindings: Vec<PublicBinding>,
        layout: ReflectedBindings,
        push_constants: Vec<u8>,
    },
    TextureReadback {
        texture: u64,
    },
    Present {
        surface: u64,
    },
}

pub struct FrameSubmission {
    pub graph: CompiledGraph,
    pub nodes: Vec<ExecutableNode>,
    pub commands: Vec<DrawIndexedCommand>,
}

pub struct FrameRecorder {
    state: FrameState,
    indirect: IndexedIndirectBuffer,
    graph: FrameGraph,
    nodes: Vec<ExecutableNode>,
}
impl FrameRecorder {
    pub fn new(indirect_capacity: u32) -> Result<Self, FrameError> {
        Ok(Self {
            state: FrameState::Idle,
            indirect: IndexedIndirectBuffer::new(indirect_capacity).map_err(map_indirect)?,
            graph: FrameGraph::new(),
            nodes: Vec::new(),
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
        self.graph = FrameGraph::new();
        self.nodes.clear();
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

    pub fn add_resource(&mut self, desc: ResourceDesc) -> Result<ResourceId, FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.graph.add_resource(desc).map_err(FrameError::Graph)
    }

    pub fn set_resource_ready(
        &mut self,
        resource: ResourceId,
        completion: CompletionToken,
    ) -> Result<(), FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.graph
            .set_resource_ready(resource, completion)
            .map_err(FrameError::Graph)
    }

    pub fn set_resource_initial_state(
        &mut self,
        resource: ResourceId,
        state: ResourceState,
    ) -> Result<(), FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.graph
            .set_resource_initial_state(resource, state)
            .map_err(FrameError::Graph)
    }

    /// Capacity is reserved before graph mutation, so a failed record never leaves an unpaired node.
    pub fn record_node(
        &mut self,
        node: NodeDesc,
        payload: ExecutableNode,
    ) -> Result<NodeId, FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.nodes
            .try_reserve(1)
            .map_err(|_| FrameError::CapacityExhausted)?;
        let id = self.graph.add_node(node).map_err(FrameError::Graph)?;
        if id.index() as usize != self.nodes.len() {
            return Err(FrameError::NodePayloadMismatch);
        }
        self.nodes.push(payload);
        Ok(id)
    }

    pub fn submit(&mut self) -> Result<FrameSubmission, FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        if self.nodes.is_empty() {
            return Err(FrameError::MissingGraph);
        }
        let graph = self.graph.compile().map_err(FrameError::Graph)?;
        if graph.order().len() != self.nodes.len() {
            return Err(FrameError::NodePayloadMismatch);
        }
        self.state = FrameState::Submitted;
        Ok(FrameSubmission {
            graph,
            nodes: core::mem::take(&mut self.nodes),
            commands: self.indirect.commands().to_vec(),
        })
    }

    pub fn finish(&mut self) -> Result<(), FrameError> {
        if self.state != FrameState::Submitted {
            return Err(FrameError::NotSubmitted);
        }
        self.state = FrameState::Idle;
        self.graph = FrameGraph::new();
        Ok(())
    }

    pub fn abort(&mut self) {
        self.state = FrameState::Idle;
        self.graph = FrameGraph::new();
        self.nodes.clear();
    }
}

fn map_indirect(error: IndirectError) -> FrameError {
    match error {
        IndirectError::OutOfBounds => FrameError::IndirectOutOfBounds,
        _ => FrameError::InvalidCapacity,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameError {
    InvalidCapacity,
    AlreadyRecording,
    NotRecording,
    MissingGraph,
    IndirectOutOfBounds,
    NotSubmitted,
    CapacityExhausted,
    NodePayloadMismatch,
    Graph(GraphError),
}
