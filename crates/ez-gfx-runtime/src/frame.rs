use crate::{
    binding::{PipelineLayout, PublicBinding, ReflectedBindings},
    graph::{CompiledGraph, FrameGraph, GraphError, NodeDesc, NodeId, ResourceDesc, ResourceId},
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer, IndirectError},
};
use ez_gfx_core::handle::{
    IndirectBufferHandle, RenderTargetHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use ez_gfx_hal::{CompletionToken, DynamicPipelineState, ResourceState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Recording lifecycle state for a frame.
pub enum FrameState {
    /// No frame is being recorded or awaiting completion.
    Idle,
    /// A frame is accepting graph nodes, resources, and draw commands.
    Recording,
    /// The recorded frame has been submitted and awaits completion.
    Submitted,
}

#[derive(Clone, Debug, PartialEq)]
/// Backend work associated with one compiled graph node.
pub enum ExecutableNode {
    /// A graphics draw node with shader, indirect, and dynamic state.
    Graphics {
        /// Shader handle.
        shader: ShaderHandle,
        /// Indirect buffer handle.
        indirect: IndirectBufferHandle,
        /// Number of draws.
        draw_count: u32,
        /// Reflected resource bindings.
        bindings: Vec<PublicBinding>,
        /// Reflected shader layout.
        layout: ReflectedBindings,
        /// Backend pipeline layout.
        pipeline_layout: PipelineLayout,
        /// Dynamic pipeline state.
        state: DynamicPipelineState,
        /// Push-constant bytes.
        push_constants: Vec<u8>,
    },
    /// A compute dispatch node.
    Compute {
        /// Shader handle.
        shader: ShaderHandle,
        /// Workgroup counts.
        groups: [u32; 3],
        /// Reflected resource bindings.
        bindings: Vec<PublicBinding>,
        /// Reflected shader layout.
        layout: ReflectedBindings,
        /// Push-constant bytes.
        push_constants: Vec<u8>,
    },
    /// A texture readback node.
    TextureReadback {
        /// Texture handle.
        texture: TextureHandle,
    },
    /// A managed render-target readback node.
    RenderTargetReadback {
        /// Render-target handle.
        target: RenderTargetHandle,
    },
    /// A presentation node.
    Present {
        /// Surface handle.
        surface: SurfaceHandle,
    },
}

/// Compiled frame data ready for backend execution.
pub struct FrameSubmission {
    /// Compiled dependency graph and execution order.
    pub graph: CompiledGraph,
    /// Executable payloads indexed by graph node ID.
    pub nodes: Vec<ExecutableNode>,
    /// Indexed draw commands captured for the frame.
    pub commands: Vec<DrawIndexedCommand>,
}

/// Records frame graph work and indexed draw commands through submission.
pub struct FrameRecorder {
    /// Current recording lifecycle state.
    state: FrameState,
    /// Indexed draw-command storage for the current frame.
    indirect: IndexedIndirectBuffer,
    /// Dependency graph under construction.
    graph: FrameGraph,
    /// Executable payloads aligned with graph node IDs.
    nodes: Vec<ExecutableNode>,
}
impl FrameRecorder {
    /// Creates an idle recorder with the requested indirect draw capacity.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::InvalidCapacity` or `FrameError::IndirectOutOfBounds` if the indirect buffer rejects the requested capacity.
    pub fn new(indirect_capacity: u32) -> Result<Self, FrameError> {
        Ok(Self {
            state: FrameState::Idle,
            indirect: IndexedIndirectBuffer::new(indirect_capacity).map_err(map_indirect)?,
            graph: FrameGraph::new(),
            nodes: Vec::new(),
        })
    }

    /// Returns the current recording lifecycle state.
    pub const fn state(&self) -> FrameState {
        self.state
    }

    /// Starts a fresh recording and clears prior graph and draw data.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::AlreadyRecording` unless the recorder is idle.
    pub fn begin(&mut self) -> Result<(), FrameError> {
        if self.state != FrameState::Idle {
            return Err(FrameError::AlreadyRecording);
        }
        self.state = FrameState::Recording;
        self.graph = FrameGraph::new();
        self.nodes.clear();
        self.indirect.reset();
        Ok(())
    }

    /// Writes an indexed draw command and extends the active draw range.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotRecording` when no recording is active, or an
    /// indirect-buffer error if the index is invalid.
    pub fn write_indirect(
        &mut self,
        index: u32,
        command: DrawIndexedCommand,
    ) -> Result<(), FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.indirect
            .write_batch(index, core::slice::from_ref(&command))
            .map_err(map_indirect)
    }

    /// Adds a resource description to the frame graph being recorded.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotRecording` when no recording is active, or `FrameError::Graph` if the resource cannot be added to the graph.
    pub fn add_resource(&mut self, desc: ResourceDesc) -> Result<ResourceId, FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.graph.add_resource(desc).map_err(FrameError::Graph)
    }

    /// Associates an external completion token with a graph resource.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotRecording` when no recording is active, or `FrameError::Graph` if the graph rejects the resource completion token.
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

    /// Sets the resource state assumed before graph execution.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotRecording` when no recording is active, or `FrameError::Graph` if the graph rejects the resource's initial state.
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
    ///
    /// # Errors
    ///
    /// Returns an error when no recording is active, node capacity cannot be reserved, the graph rejects the node, or the node ID does not match the payload index.
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

    /// Compiles the recorded graph and transfers its executable frame data.
    ///
    /// # Errors
    ///
    /// Returns an error when no recording is active, no nodes were recorded, graph compilation fails, or compiled nodes do not match their payloads.
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

    /// Marks a submitted frame complete and returns the recorder to idle.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotSubmitted` unless the recorder is in the submitted state.
    pub fn finish(&mut self) -> Result<(), FrameError> {
        if self.state != FrameState::Submitted {
            return Err(FrameError::NotSubmitted);
        }
        self.state = FrameState::Idle;
        self.graph = FrameGraph::new();
        Ok(())
    }

    /// Discards recorded work and returns the recorder to idle.
    pub fn abort(&mut self) {
        self.state = FrameState::Idle;
        self.graph = FrameGraph::new();
        self.nodes.clear();
    }
}

/// Translates indirect-buffer failures into frame recording failures.
fn map_indirect(error: IndirectError) -> FrameError {
    match error {
        IndirectError::OutOfBounds => FrameError::IndirectOutOfBounds,
        _ => FrameError::InvalidCapacity,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Failure reported while configuring or recording a frame.
pub enum FrameError {
    /// The requested indirect draw capacity is unsupported.
    InvalidCapacity,
    /// Recording was requested while another recording is active.
    AlreadyRecording,
    /// The operation requires an active recording.
    NotRecording,
    /// Submission requires at least one recorded graph node.
    MissingGraph,
    /// An indirect draw index exceeds the configured capacity.
    IndirectOutOfBounds,
    /// Completion was requested before frame submission.
    NotSubmitted,
    /// Memory could not be reserved for another executable node.
    CapacityExhausted,
    /// Executable payloads do not align with compiled graph nodes.
    NodePayloadMismatch,
    /// Frame graph construction or compilation failed.
    Graph(GraphError),
}
