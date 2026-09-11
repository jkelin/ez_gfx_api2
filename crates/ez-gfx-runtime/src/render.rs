pub use ez_gfx_hal::{
    BlendMode, CullMode, DynamicPipelineState, FrontFace, PrimitiveTopology, RenderStateError,
};

use ez_gfx_hal::{
    AttachmentLoadOp, AttachmentStoreOp, ExecutionAction, ExecutionBarrier, ExecutionPass,
    ExecutionRange, ExecutionWait, FrameExecutionBackend, FrameExecutionPlan, ImageSubresources,
};

use crate::graph::{CompiledGraph, LoadOp, ResourceRange, StoreOp};

#[derive(Default)]
pub(crate) struct RenderWorkspace {
    pass_starts: Vec<Option<ExecutionPass>>,
    pass_ends: Vec<bool>,
    reusable_pass_vectors: Vec<(Vec<u32>, Vec<u32>)>,
}

impl RenderWorkspace {
    pub(crate) fn retained_bytes(&self) -> usize {
        let starts = self
            .pass_starts
            .capacity()
            .saturating_mul(core::mem::size_of::<Option<ExecutionPass>>());
        let ends = self
            .pass_ends
            .capacity()
            .saturating_mul(core::mem::size_of::<bool>());
        let reusable = self
            .reusable_pass_vectors
            .capacity()
            .saturating_mul(core::mem::size_of::<(Vec<u32>, Vec<u32>)>())
            .saturating_add(
                self.reusable_pass_vectors
                    .iter()
                    .map(|(nodes, colors)| {
                        nodes
                            .capacity()
                            .saturating_mul(core::mem::size_of::<u32>())
                            .saturating_add(
                                colors
                                    .capacity()
                                    .saturating_mul(core::mem::size_of::<u32>()),
                            )
                    })
                    .sum::<usize>(),
            );
        starts.saturating_add(ends).saturating_add(reusable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Errors that can prevent a compiled render graph from becoming or executing a frame plan.
pub enum ExecutionError<E> {
    /// A graph node has no corresponding payload.
    MissingPayload {
        /// Missing node index.
        node: u32,
    },
    /// More payloads were supplied than the graph's ordered nodes require.
    UnexpectedPayloads,
    /// A compiled image transition contains an invalid mip or layer range.
    InvalidCompiledRange,
    /// The rendering backend rejected the frame plan or its payloads.
    Backend(E),
}

/// Converts a compiled graph into one immutable command plan before native recording begins.
///
/// # Errors
///
/// Returns an error if a graph node lacks a payload, extra payloads are supplied, or a compiled image transition has an invalid mip or layer range.
pub fn build_execution_plan(
    graph: &CompiledGraph,
    payload_count: usize,
) -> Result<FrameExecutionPlan, ExecutionError<core::convert::Infallible>> {
    let mut plan = FrameExecutionPlan {
        actions: Vec::new(),
    };
    let mut workspace = RenderWorkspace::default();
    build_execution_plan_into(graph, payload_count, &mut plan, &mut workspace)?;
    Ok(plan)
}

pub(crate) fn build_execution_plan_into(
    graph: &CompiledGraph,
    payload_count: usize,
    plan: &mut FrameExecutionPlan,
    workspace: &mut RenderWorkspace,
) -> Result<(), ExecutionError<core::convert::Infallible>> {
    if let Some(node) = graph
        .order()
        .iter()
        .find(|node| node.index() as usize >= payload_count)
    {
        return Err(ExecutionError::MissingPayload { node: node.index() });
    }
    if payload_count != graph.order().len() {
        return Err(ExecutionError::UnexpectedPayloads);
    }

    for action in plan.actions.drain(..) {
        if let ExecutionAction::BeginPass(pass) = action {
            workspace
                .reusable_pass_vectors
                .push((pass.nodes, pass.colors));
        }
    }
    workspace.pass_starts.clear();
    workspace.pass_starts.resize_with(payload_count, || None);
    workspace.pass_ends.clear();
    workspace.pass_ends.resize(payload_count, false);

    for pass in graph.passes() {
        let (Some(first), Some(last)) = (pass.nodes.first(), pass.nodes.last()) else {
            continue;
        };
        let (mut nodes, mut colors) = workspace.reusable_pass_vectors.pop().unwrap_or_default();
        nodes.clear();
        colors.clear();
        nodes.extend(pass.nodes.iter().map(|node| node.index()));
        colors.extend(pass.info.colors().iter().map(|resource| resource.index()));
        workspace.pass_starts[first.index() as usize] = Some(ExecutionPass {
            nodes,
            colors,
            depth: pass.info.depth().map(super::graph::ResourceId::index),
            area: pass.info.area(),
            samples: pass.info.samples(),
            load: match pass.info.load() {
                LoadOp::Load => AttachmentLoadOp::Load,
                LoadOp::Clear => AttachmentLoadOp::Clear,
                LoadOp::Discard => AttachmentLoadOp::Discard,
            },
            store: match pass.info.store() {
                StoreOp::Store => AttachmentStoreOp::Store,
                StoreOp::Discard => AttachmentStoreOp::Discard,
            },
        });
        workspace.pass_ends[last.index() as usize] = true;
    }

    for node in graph.order() {
        plan.actions.extend(
            graph
                .waits()
                .iter()
                .filter(|wait| wait.node == *node)
                .map(|wait| {
                    ExecutionAction::Wait(ExecutionWait {
                        node: node.index(),
                        source: wait.source.map(super::graph::NodeId::index),
                        external: wait.external,
                    })
                }),
        );
        for transition in graph
            .transitions()
            .iter()
            .filter(|transition| transition.node == *node)
        {
            let range = match transition.range {
                ResourceRange::Buffer(range) => ExecutionRange::Buffer(range),
                ResourceRange::Image(range) => ExecutionRange::Image(
                    ImageSubresources::new(
                        range.first_mip,
                        range.mip_count,
                        range.first_layer,
                        range.layer_count,
                    )
                    .map_err(|_| ExecutionError::InvalidCompiledRange)?,
                ),
            };
            plan.actions
                .push(ExecutionAction::Barrier(ExecutionBarrier {
                    node: node.index(),
                    resource: transition.resource.index(),
                    range,
                    before: transition.before,
                    after: transition.after,
                }));
        }
        if let Some(pass) = workspace.pass_starts[node.index() as usize].take() {
            plan.actions.push(ExecutionAction::BeginPass(pass));
        }
        plan.actions
            .push(ExecutionAction::ExecuteNode(node.index()));
        if workspace.pass_ends[node.index() as usize] {
            plan.actions.push(ExecutionAction::EndPass);
        }
    }
    Ok(())
}

/// Preflights the whole plan, then gives it to the backend as one atomic frame submission.
///
/// # Errors
///
/// Returns an error if plan construction finds missing or extra payloads or an invalid compiled image range, or if the backend rejects execution.
pub fn execute_compiled_graph<B, P>(
    graph: &CompiledGraph,
    payloads: &[P],
    backend: &mut B,
) -> Result<(), ExecutionError<B::Error>>
where
    B: FrameExecutionBackend<P>,
{
    let plan = build_execution_plan(graph, payloads.len()).map_err(|error| match error {
        ExecutionError::MissingPayload { node } => ExecutionError::MissingPayload { node },
        ExecutionError::UnexpectedPayloads => ExecutionError::UnexpectedPayloads,
        ExecutionError::InvalidCompiledRange => ExecutionError::InvalidCompiledRange,
        ExecutionError::Backend(never) => match never {},
    })?;
    backend
        .execute(&plan, payloads)
        .map_err(ExecutionError::Backend)
}
