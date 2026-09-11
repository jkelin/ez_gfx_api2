use crate::{
    binding::{PipelineLayout, ReflectedBindings, ResourceIdentity},
    graph::{
        CompiledGraph, FRAME_WORKSPACE_BYTE_LIMIT, FrameGraph, GraphError, GraphTemplateCache,
        GraphTemplateCacheStats, GraphWorkspace, NodeDesc, NodeId, ResourceDesc, ResourceId,
    },
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer, IndirectError},
    render::{RenderWorkspace, build_execution_plan_into},
};
use ez_gfx_core::handle::{
    CounterBufferHandle, RenderTargetHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use ez_gfx_hal::{
    CompletionToken, DynamicPipelineState, ExecutionAction, FrameExecutionPlan, ResourceState,
};
use std::sync::atomic::{AtomicU64, Ordering};


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
    /// A graphics draw node with stage shaders, counter buffer, and dynamic state.
    Graphics {
        /// Vertex shader handle.
        vertex_shader: ShaderHandle,
        /// Fragment shader handle.
        fragment_shader: ShaderHandle,
        /// Counter buffer handle.
        counter: CounterBufferHandle,
        /// Maximum number of indirect draws.
        draw_capacity: u32,
        /// Range of snapshotted resources in the submission binding arena.
        bindings: core::ops::Range<usize>,
        /// Reflected shader layout.
        layout: ReflectedBindings,
        /// Backend pipeline layout.
        pipeline_layout: PipelineLayout,
        /// Dynamic pipeline state.
        state: DynamicPipelineState,
    },
    /// A compute dispatch node.
    Compute {
        /// Shader handle.
        shader: ShaderHandle,
        /// Workgroup counts.
        groups: [u32; 3],
        /// Range of snapshotted resources in the submission binding arena.
        bindings: core::ops::Range<usize>,
        /// Reflected shader layout.
        layout: ReflectedBindings,
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

static NEXT_RECORDER_ID: AtomicU64 = AtomicU64::new(1);

/// Compiled frame data ready for backend execution.
pub struct FrameSubmission {
    /// Compiled dependency graph and execution order.
    pub graph: CompiledGraph,
    /// Immutable backend execution plan derived from `graph`.
    pub plan: FrameExecutionPlan,
    /// Executable payloads indexed by graph node ID.
    pub nodes: Vec<ExecutableNode>,
    /// Indexed draw commands captured for the frame.
    pub commands: Vec<DrawIndexedCommand>,
    /// Resource snapshots stored in reflected requirement order for each bound node.
    pub binding_resources: Vec<ResourceIdentity>,
    /// Recorder identity; private so consumers cannot construct recycle tokens.
    recorder_id: u64,
    /// Monotonic submission generation owned by `recorder_id`.
    generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Retained CPU storage owned by one frame recorder.
pub struct FrameWorkspaceStats {
    /// Current retained capacity in bytes.
    pub retained_bytes: usize,
    /// Largest retained capacity observed after a successful compile.
    pub high_water_bytes: usize,
    /// Hard ceiling for retained frame-workspace capacity.
    pub byte_limit: usize,
    /// Stable-graph template cache telemetry.
    pub graph_cache: GraphTemplateCacheStats,
}

struct FrameWorkspace {
    graph: CompiledGraph,
    graph_scratch: GraphWorkspace,
    graph_cache: GraphTemplateCache,
    plan: FrameExecutionPlan,
    render_scratch: RenderWorkspace,
    commands: Vec<DrawIndexedCommand>,
    lowering_scratch_bytes: usize,
    in_flight_bytes: usize,
    high_water_bytes: usize,
}

impl Default for FrameWorkspace {
    fn default() -> Self {
        Self {
            graph: CompiledGraph::default(),
            graph_cache: GraphTemplateCache::default(),
            graph_scratch: GraphWorkspace::default(),
            plan: FrameExecutionPlan {
                actions: Vec::new(),
            },
            render_scratch: RenderWorkspace::default(),
            commands: Vec::new(),
            lowering_scratch_bytes: 0,
            in_flight_bytes: 0,
            high_water_bytes: 0,
        }
    }
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
    /// Reusable resource snapshot arena referenced by executable node ranges.
    binding_resources: Vec<ResourceIdentity>,
    /// Unique owner stamped into every submission recycle token.
    recorder_id: u64,
    /// Last generation issued by this recorder.
    submission_generation: u64,
    /// Typed reusable storage spanning compilation through execution lowering.
    workspace: FrameWorkspace,
}
impl FrameRecorder {
    /// Creates an idle recorder with the requested counter-buffer capacity.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::InvalidCapacity` for an unsupported indirect capacity,
    /// or `FrameError::CapacityExhausted` if recorder identities are exhausted.
    pub fn new(indirect_capacity: u32) -> Result<Self, FrameError> {
        let recorder_id = NEXT_RECORDER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| FrameError::CapacityExhausted)?;
        Ok(Self {
            state: FrameState::Idle,
            indirect: IndexedIndirectBuffer::new(indirect_capacity).map_err(map_indirect)?,
            graph: FrameGraph::new(),
            nodes: Vec::new(),
            binding_resources: Vec::new(),
            workspace: FrameWorkspace::default(),
            recorder_id,
            submission_generation: 0,
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
        self.graph.clear();
        self.nodes.clear();
        self.indirect.reset();
        Ok(())
    }

    /// Writes an indexed draw command and extends the active draw range.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotRecording` when no recording is active, or a
    /// counter-buffer error if the index is invalid.
    pub fn write_counter(
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

    /// Sets the state carried into this frame by a persistent history resource.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotRecording` when no recording is active, or `FrameError::Graph`
    /// when the resource is unknown or is not persistent history.
    pub fn set_history_state(
        &mut self,
        resource: ResourceId,
        state: ResourceState,
    ) -> Result<(), FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.graph
            .set_history_state(resource, state)
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

    /// Records a node whose reflected resources are snapshotted into the reusable binding arena.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::record_node`], or `CapacityExhausted` when binding
    /// storage cannot be reserved. Resources must be supplied in reflected requirement order.
    pub fn record_bound_node(
        &mut self,
        node: NodeDesc,
        resources: impl ExactSizeIterator<Item = ResourceIdentity>,
        payload: impl FnOnce(core::ops::Range<usize>) -> ExecutableNode,
    ) -> Result<NodeId, FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        self.nodes
            .try_reserve(1)
            .map_err(|_| FrameError::CapacityExhausted)?;
        self.binding_resources
            .try_reserve(resources.len())
            .map_err(|_| FrameError::CapacityExhausted)?;
        let id = self.graph.add_node(node).map_err(FrameError::Graph)?;
        if id.index() as usize != self.nodes.len() {
            return Err(FrameError::NodePayloadMismatch);
        }
        let start = self.binding_resources.len();
        self.binding_resources.extend(resources);
        let range = start..self.binding_resources.len();
        self.nodes.push(payload(range));
        Ok(id)
    }

    /// Compiles the recorded graph and transfers its reusable buffers to a submission.
    ///
    /// Return the submission to [`Self::finish`] after backend encoding so the next
    /// frame can reuse every retained capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when recording state, graph validity, payload alignment, or
    /// the bounded frame-workspace capacity is invalid.
    pub fn submit(&mut self) -> Result<FrameSubmission, FrameError> {
        if self.state != FrameState::Recording {
            return Err(FrameError::NotRecording);
        }
        if self.nodes.is_empty() {
            return Err(FrameError::MissingGraph);
        }
        self.graph
            .compile_cached_into(
                &mut self.workspace.graph,
                &mut self.workspace.graph_scratch,
                &mut self.workspace.graph_cache,
            )
            .map_err(FrameError::Graph)?;
        if self.workspace.graph.order().len() != self.nodes.len() {
            return Err(FrameError::NodePayloadMismatch);
        }
        build_execution_plan_into(
            &self.workspace.graph,
            self.nodes.len(),
            &mut self.workspace.plan,
            &mut self.workspace.render_scratch,
        )
        .map_err(|_| FrameError::NodePayloadMismatch)?;
        self.workspace.commands.clear();
        self.workspace
            .commands
            .try_reserve(self.indirect.commands().len())
            .map_err(|_| FrameError::CapacityExhausted)?;
        self.workspace
            .commands
            .extend_from_slice(self.indirect.commands());

        let retained = self.retained_bytes();
        if retained > FRAME_WORKSPACE_BYTE_LIMIT {
            return Err(FrameError::CapacityExhausted);
        }
        let generation = self
            .submission_generation
            .checked_add(1)
            .ok_or(FrameError::CapacityExhausted)?;
        self.submission_generation = generation;
        self.workspace.high_water_bytes = self.workspace.high_water_bytes.max(retained);
        self.state = FrameState::Submitted;
        let submission = FrameSubmission {
            graph: core::mem::take(&mut self.workspace.graph),
            plan: core::mem::replace(
                &mut self.workspace.plan,
                FrameExecutionPlan {
                    actions: Vec::new(),
                },
            ),
            nodes: core::mem::take(&mut self.nodes),
            commands: core::mem::take(&mut self.workspace.commands),
            binding_resources: core::mem::take(&mut self.binding_resources),
            recorder_id: self.recorder_id,
            generation,
        };
        self.workspace.in_flight_bytes = submission_retained_bytes(&submission);
        Ok(submission)
    }

    /// Recycles this recorder's completed submission and returns to idle.
    ///
    /// # Errors
    ///
    /// Returns `FrameError::NotSubmitted` unless the recorder is submitted,
    /// `FrameError::SubmissionMismatch` for a foreign or stale token, or
    /// `FrameError::CapacityExhausted` before oversized capacities are adopted.
    pub fn finish(&mut self, mut submission: FrameSubmission) -> Result<(), FrameError> {
        if self.state != FrameState::Submitted {
            return Err(FrameError::NotSubmitted);
        }
        let retained = validate_submission_recycle(
            self.recorder_id,
            self.submission_generation,
            self.retained_bytes()
                .saturating_sub(self.workspace.in_flight_bytes),
            submission.recorder_id,
            submission.generation,
            submission_retained_bytes(&submission),
        )?;
        submission.nodes.clear();
        submission.commands.clear();
        submission.binding_resources.clear();
        self.workspace.graph = submission.graph;
        self.workspace.plan = submission.plan;
        self.nodes = submission.nodes;
        self.workspace.commands = submission.commands;
        self.binding_resources = submission.binding_resources;
        self.graph.clear();
        self.workspace.in_flight_bytes = 0;
        self.state = FrameState::Idle;
        self.workspace.high_water_bytes = self.workspace.high_water_bytes.max(retained);
        Ok(())
    }

    /// Discards recorded work and returns the recorder to idle.
    ///
    /// Capacity remains reusable unless one rejected recording grew beyond the
    /// workspace ceiling, in which case it is released so later frames can recover.
    pub fn abort(&mut self) {
        self.state = FrameState::Idle;
        self.workspace.in_flight_bytes = 0;
        self.graph.clear();
        self.nodes.clear();
        self.binding_resources.clear();
        // A graph can exceed the retained-byte ceiling while it is still being
        // recorded. Do not let that one rejected input poison every later submit.
        if self.retained_bytes() > FRAME_WORKSPACE_BYTE_LIMIT {
            self.graph = FrameGraph::new();
            self.nodes = Vec::new();
            let graph_cache = core::mem::take(&mut self.workspace.graph_cache);
            let high_water_bytes = self.workspace.high_water_bytes;
            self.workspace = FrameWorkspace {
                graph_cache,
                high_water_bytes,
                ..FrameWorkspace::default()
            };
        }
    }
    /// Invalidates stable graph templates after an incompatible runtime or owner change.
    ///
    /// Existing submissions remain valid because templates contain CPU-only copied structure.
    pub fn invalidate_graph_templates(&mut self) {
        self.workspace.graph_cache.invalidate();
    }

    /// Reports bounded retained frame-workspace capacity.
    pub fn workspace_stats(&self) -> FrameWorkspaceStats {
        FrameWorkspaceStats {
            retained_bytes: self.retained_bytes(),
            high_water_bytes: self.workspace.high_water_bytes,
            byte_limit: FRAME_WORKSPACE_BYTE_LIMIT,
            graph_cache: self.workspace.graph_cache.stats(),
        }
    }

    /// Accounts reusable backend-lowering scratch against the frame-workspace ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError::CapacityExhausted`] when the combined retained capacity exceeds
    /// the workspace limit. The previous accounting remains intact on failure.
    pub fn set_lowering_scratch_bytes(&mut self, bytes: usize) -> Result<(), FrameError> {
        let previous = self.workspace.lowering_scratch_bytes;
        self.workspace.lowering_scratch_bytes = bytes;
        let retained = self.retained_bytes();
        if retained > FRAME_WORKSPACE_BYTE_LIMIT {
            self.workspace.lowering_scratch_bytes = previous;
            return Err(FrameError::CapacityExhausted);
        }
        self.workspace.high_water_bytes = self.workspace.high_water_bytes.max(retained);
        Ok(())
    }

    fn retained_bytes(&self) -> usize {
        self.graph
            .retained_bytes()
            .saturating_add(
                self.nodes
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ExecutableNode>()),
            )
            .saturating_add(
                self.binding_resources
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ResourceIdentity>()),
            )
            .saturating_add(self.workspace.graph.retained_bytes())
            .saturating_add(self.workspace.graph_scratch.retained_bytes())
            .saturating_add(plan_retained_bytes(&self.workspace.plan))
            .saturating_add(self.workspace.graph_cache.retained_bytes())
            .saturating_add(self.workspace.render_scratch.retained_bytes())
            .saturating_add(
                self.workspace
                    .commands
                    .capacity()
                    .saturating_mul(core::mem::size_of::<DrawIndexedCommand>()),
            )
            .saturating_add(self.workspace.lowering_scratch_bytes)
            .saturating_add(self.workspace.in_flight_bytes)
    }
}

fn plan_retained_bytes(plan: &FrameExecutionPlan) -> usize {
    plan.actions
        .capacity()
        .saturating_mul(core::mem::size_of::<ExecutionAction>())
        .saturating_add(
            plan.actions
                .iter()
                .map(|action| match action {
                    ExecutionAction::BeginPass(pass) => pass
                        .nodes
                        .capacity()
                        .saturating_add(pass.colors.capacity())
                        .saturating_mul(core::mem::size_of::<u32>()),
                    _ => 0,
                })
                .sum::<usize>(),
        )
}

fn submission_retained_bytes(submission: &FrameSubmission) -> usize {
    submission
        .graph
        .retained_bytes()
        .saturating_add(plan_retained_bytes(&submission.plan))
        .saturating_add(
            submission
                .nodes
                .capacity()
                .saturating_mul(core::mem::size_of::<ExecutableNode>()),
        )
        .saturating_add(
            submission
                .binding_resources
                .capacity()
                .saturating_mul(core::mem::size_of::<ResourceIdentity>()),
        )
        .saturating_add(
            submission
                .commands
                .capacity()
                .saturating_mul(core::mem::size_of::<DrawIndexedCommand>()),
        )
}

// Foreign/stale tokens fail before capacity accounting; saturating addition
// converts arithmetic overflow into the same bounded-capacity rejection.
fn validate_submission_recycle(
    recorder_id: u64,
    generation: u64,
    retained_bytes: usize,
    submission_recorder_id: u64,
    submission_generation: u64,
    submission_bytes: usize,
) -> Result<usize, FrameError> {
    if submission_recorder_id != recorder_id || submission_generation != generation {
        return Err(FrameError::SubmissionMismatch);
    }
    let retained = retained_bytes.saturating_add(submission_bytes);
    if retained > FRAME_WORKSPACE_BYTE_LIMIT {
        return Err(FrameError::CapacityExhausted);
    }
    Ok(retained)
}

#[cfg(test)]
mod submission_tests {
    use super::{FRAME_WORKSPACE_BYTE_LIMIT, FrameError, validate_submission_recycle};

    #[test]
    fn recycle_metadata_rejects_foreign_stale_and_oversized_submissions() {
        assert_eq!(
            validate_submission_recycle(1, 2, 0, 9, 2, 0),
            Err(FrameError::SubmissionMismatch)
        );
        assert_eq!(
            validate_submission_recycle(1, 2, 0, 1, 1, 0),
            Err(FrameError::SubmissionMismatch)
        );
        assert_eq!(
            validate_submission_recycle(1, 2, 1, 1, 2, FRAME_WORKSPACE_BYTE_LIMIT),
            Err(FrameError::CapacityExhausted)
        );
        assert_eq!(
            validate_submission_recycle(1, 2, 1, 1, 2, 2),
            Ok(3)
        );
    }

    #[test]
    fn lowering_scratch_is_bounded_and_reported() {
        let mut recorder = super::FrameRecorder::new(1).unwrap();
        let baseline = recorder.workspace_stats().retained_bytes;
        recorder.set_lowering_scratch_bytes(4096).unwrap();
        assert_eq!(recorder.workspace_stats().retained_bytes, baseline + 4096);
        assert_eq!(
            recorder.set_lowering_scratch_bytes(FRAME_WORKSPACE_BYTE_LIMIT),
            Err(FrameError::CapacityExhausted)
        );
        assert_eq!(recorder.workspace_stats().retained_bytes, baseline + 4096);
    }
}

/// Translates counter-buffer failures into frame recording failures.
fn map_indirect(error: IndirectError) -> FrameError {
    match error {
        IndirectError::OutOfBounds => FrameError::CounterOutOfBounds,
        _ => FrameError::InvalidCapacity,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Failure reported while configuring or recording a frame.
pub enum FrameError {
    /// The requested counter-buffer capacity is unsupported.
    InvalidCapacity,
    /// Recording was requested while another recording is active.
    AlreadyRecording,
    /// The operation requires an active recording.
    NotRecording,
    /// Submission requires at least one recorded graph node.
    MissingGraph,
    /// A counter-buffer draw index exceeds the configured capacity.
    CounterOutOfBounds,
    /// Completion was requested before frame submission.
    NotSubmitted,
    /// A completion token belongs to another recorder or submission generation.
    SubmissionMismatch,
    /// Memory could not be reserved for another executable node.
    CapacityExhausted,
    /// Executable payloads do not align with compiled graph nodes.
    NodePayloadMismatch,
    /// Frame graph construction or compilation failed.
    Graph(GraphError),
}
