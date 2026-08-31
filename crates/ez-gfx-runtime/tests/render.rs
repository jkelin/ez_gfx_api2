use ez_gfx_runtime::render::{CullMode, DynamicPipelineState, PrimitiveTopology};

#[test]
fn dynamic_pipeline_state_validates_every_discriminant() {
    let state = DynamicPipelineState::from_abi(2, 1, 0, 1).unwrap();
    assert_eq!(state.cull, CullMode::Back);
    assert_eq!(state.topology, PrimitiveTopology::TriangleList);

    for values in [[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 6, 0], [0, 0, 0, 2]] {
        assert!(
            DynamicPipelineState::from_abi(values[0], values[1], values[2], values[3]).is_err()
        );
    }
}

use ez_gfx_hal::{
    AttachmentLoadOp, BufferRange, ExecutionAction, FrameExecutionBackend, FrameExecutionPlan,
    QueueKind, ResourceAccess, ResourceState, ShaderStage,
};
use ez_gfx_runtime::{
    graph::{
        Access, FrameGraph, ImageRange, LoadOp, NodeDesc, PassInfo, ResourceDesc, ResourceLifetime,
        StoreOp,
    },
    render::{ExecutionError, build_execution_plan, execute_compiled_graph},
    target::Format,
};

#[derive(Default)]
struct TraceBackend {
    trace: Vec<String>,
}

impl FrameExecutionBackend<&'static str> for TraceBackend {
    type Error = &'static str;

    fn execute(
        &mut self,
        plan: &FrameExecutionPlan,
        payloads: &[&'static str],
    ) -> Result<(), Self::Error> {
        for action in &plan.actions {
            match action {
                ExecutionAction::Wait(wait) => self.trace.push(format!("wait:{}", wait.node)),
                ExecutionAction::Barrier(barrier) => {
                    self.trace.push(format!("barrier:{}", barrier.node));
                }
                ExecutionAction::BeginPass(pass) => {
                    self.trace.push(format!("begin:{:?}", pass.nodes));
                }
                ExecutionAction::ExecuteNode(node) => {
                    self.trace
                        .push(format!("node:{node}:{}", payloads[*node as usize]));
                }
                ExecutionAction::EndPass => self.trace.push("end".into()),
            }
        }
        self.trace.push("submit".into());
        Ok(())
    }
}

#[test]
fn executor_consumes_compiled_order_barriers_and_coalesced_passes() {
    let mut graph = FrameGraph::new();
    let buffer = graph
        .add_resource(ResourceDesc::buffer(64, 4, ResourceLifetime::External).unwrap())
        .unwrap();
    let write = ResourceState::new(
        QueueKind::Compute,
        ShaderStage::Compute,
        ResourceAccess::StorageWrite,
    )
    .unwrap();
    let read = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::AllGraphics,
        ResourceAccess::StorageRead,
    )
    .unwrap();
    graph
        .add_node(
            NodeDesc::new("compute", QueueKind::Compute).access(Access::buffer(
                buffer,
                BufferRange::new(0, 64).unwrap(),
                write,
            )),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("draw-a", QueueKind::Graphics).access(Access::buffer(
                buffer,
                BufferRange::new(0, 64).unwrap(),
                read,
            )),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("draw-b", QueueKind::Graphics).access(Access::buffer(
                buffer,
                BufferRange::new(0, 64).unwrap(),
                read,
            )),
        )
        .unwrap();

    let compiled = graph.compile().unwrap();
    let mut backend = TraceBackend::default();
    execute_compiled_graph(&compiled, &["compute", "draw-a", "draw-b"], &mut backend).unwrap();

    assert_eq!(
        backend.trace,
        [
            "barrier:0",
            "node:0:compute",
            "wait:1",
            "barrier:1",
            "node:1:draw-a",
            "wait:2",
            "node:2:draw-b",
            "submit",
        ]
    );
}

#[test]
fn executor_rejects_missing_payload_before_backend_work() {
    let mut graph = FrameGraph::new();
    graph
        .add_node(NodeDesc::new("node", QueueKind::Graphics))
        .unwrap();
    let compiled = graph.compile().unwrap();
    let mut backend = TraceBackend::default();

    assert_eq!(
        execute_compiled_graph::<_, &'static str>(&compiled, &[], &mut backend),
        Err(ExecutionError::MissingPayload { node: 0 })
    );
    assert!(backend.trace.is_empty());
}

#[test]
fn compatible_graphics_nodes_execute_inside_one_pass() {
    let mut graph = FrameGraph::new();
    let surface = graph
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                1,
                1,
                Format::Bgra8Srgb,
                1,
                ResourceLifetime::External,
            )
            .unwrap(),
        )
        .unwrap();
    let state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    )
    .unwrap();
    for (name, load) in [("first", LoadOp::Clear), ("second", LoadOp::Load)] {
        graph
            .add_node(
                NodeDesc::new(name, QueueKind::Graphics)
                    .access(Access::image(
                        surface,
                        ImageRange::all(1, 1).unwrap(),
                        state,
                    ))
                    .pass(
                        PassInfo::new(vec![surface], None, [0, 0, 64, 64], 1, load, StoreOp::Store)
                            .unwrap(),
                    ),
            )
            .unwrap();
    }
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.passes().len(), 1);
    let plan = build_execution_plan(&compiled, 2).unwrap();
    assert!(matches!(
        plan.actions.first(),
        Some(ExecutionAction::Barrier(_))
    ));
    assert!(plan.actions.iter().any(|action| matches!(
        action,
        ExecutionAction::BeginPass(pass) if pass.load == AttachmentLoadOp::Clear
    )));
    let mut backend = TraceBackend::default();
    execute_compiled_graph(&compiled, &["first", "second"], &mut backend).unwrap();
    assert_eq!(
        backend.trace,
        [
            "barrier:0",
            "begin:[0, 1]",
            "node:0:first",
            "node:1:second",
            "end",
            "submit",
        ]
    );
}

#[test]
fn reordered_nodes_resolve_payloads_by_node_id() {
    let mut graph = FrameGraph::new();
    let first = graph
        .add_node(NodeDesc::new("first", QueueKind::Compute))
        .unwrap();
    let second = graph
        .add_node(NodeDesc::new("second", QueueKind::Compute))
        .unwrap();
    graph.add_dependency(second, first).unwrap();
    let compiled = graph.compile().unwrap();
    let mut backend = TraceBackend::default();
    execute_compiled_graph(&compiled, &["payload-0", "payload-1"], &mut backend).unwrap();
    assert_eq!(
        backend.trace,
        ["node:1:payload-1", "node:0:payload-0", "submit"]
    );
}

#[test]
fn external_readiness_wait_precedes_resource_transition_and_node() {
    let mut graph = FrameGraph::new();
    let buffer = graph
        .add_resource(ResourceDesc::buffer(16, 4, ResourceLifetime::External).unwrap())
        .unwrap();
    graph
        .set_resource_ready(
            buffer,
            ez_gfx_hal::CompletionToken::new(QueueKind::Transfer, 7).unwrap(),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("readback", QueueKind::Transfer).access(Access::buffer(
                buffer,
                BufferRange::new(0, 16).unwrap(),
                ResourceState::new(
                    QueueKind::Transfer,
                    ShaderStage::None,
                    ResourceAccess::TransferRead,
                )
                .unwrap(),
            )),
        )
        .unwrap();
    let compiled = graph.compile().unwrap();
    let mut backend = TraceBackend::default();
    execute_compiled_graph(&compiled, &["readback"], &mut backend).unwrap();
    assert_eq!(
        backend.trace,
        ["wait:0", "barrier:0", "node:0:readback", "submit"]
    );
}

#[derive(Default)]
struct FailingBackend {
    calls: usize,
}

impl FrameExecutionBackend<&'static str> for FailingBackend {
    type Error = &'static str;

    fn execute(
        &mut self,
        _plan: &FrameExecutionPlan,
        _payloads: &[&'static str],
    ) -> Result<(), Self::Error> {
        self.calls += 1;
        Err("record failed")
    }
}

#[test]
fn backend_failure_occurs_at_the_single_atomic_execution_boundary() {
    let mut graph = FrameGraph::new();
    graph
        .add_node(NodeDesc::new("node", QueueKind::Compute))
        .unwrap();
    let compiled = graph.compile().unwrap();
    let mut backend = FailingBackend::default();
    assert_eq!(
        execute_compiled_graph(&compiled, &["payload"], &mut backend),
        Err(ExecutionError::Backend("record failed"))
    );
    assert_eq!(backend.calls, 1);
}
