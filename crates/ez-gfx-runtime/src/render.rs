pub use ez_gfx_hal::{
    BlendMode, CullMode, DynamicPipelineState, FrontFace, PrimitiveTopology, RenderStateError,
};

use std::collections::BTreeMap;

use ez_gfx_hal::{
    AttachmentLoadOp, AttachmentStoreOp, ExecutionAction, ExecutionBarrier, ExecutionPass,
    ExecutionRange, ExecutionWait, FrameExecutionBackend, FrameExecutionPlan, ImageSubresources,
};

use crate::graph::{CompiledGraph, LoadOp, ResourceRange, StoreOp};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionError<E> {
    MissingPayload { node: u32 },
    UnexpectedPayloads,
    InvalidCompiledRange,
    Backend(E),
}

/// Converts a compiled graph into one immutable command plan before native recording begins.
pub fn build_execution_plan(
    graph: &CompiledGraph,
    payload_count: usize,
) -> Result<FrameExecutionPlan, ExecutionError<core::convert::Infallible>> {
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

    let mut pass_starts = BTreeMap::new();
    let mut pass_ends = BTreeMap::new();
    for pass in graph.passes() {
        let (Some(first), Some(last)) = (pass.nodes.first(), pass.nodes.last()) else {
            continue;
        };
        pass_starts.insert(
            first.index(),
            ExecutionPass {
                nodes: pass.nodes.iter().map(|node| node.index()).collect(),
                colors: pass
                    .info
                    .colors()
                    .iter()
                    .map(|resource| resource.index())
                    .collect(),
                depth: pass.info.depth().map(|resource| resource.index()),
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
            },
        );
        pass_ends.insert(last.index(), ());
    }

    let mut actions = Vec::new();
    for node in graph.order() {
        actions.extend(
            graph
                .waits()
                .iter()
                .filter(|wait| wait.node == *node)
                .map(|wait| {
                    ExecutionAction::Wait(ExecutionWait {
                        node: node.index(),
                        source: wait.source.map(|source| source.index()),
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
            actions.push(ExecutionAction::Barrier(ExecutionBarrier {
                node: node.index(),
                resource: transition.resource.index(),
                range,
                before: transition.before,
                after: transition.after,
            }));
        }
        if let Some(pass) = pass_starts.remove(&node.index()) {
            actions.push(ExecutionAction::BeginPass(pass));
        }
        actions.push(ExecutionAction::ExecuteNode(node.index()));
        if pass_ends.contains_key(&node.index()) {
            actions.push(ExecutionAction::EndPass);
        }
    }
    Ok(FrameExecutionPlan { actions })
}

/// Preflights the whole plan, then gives it to the backend as one atomic frame submission.
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
