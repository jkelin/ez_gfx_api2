use ez_gfx_hal::{BufferRange, QueueKind, ResourceAccess, ResourceState};
use ez_gfx_runtime::{
    frame::{FrameError, FrameRecorder, FrameState},
    graph::{Access, FrameGraph, NodeDesc, ResourceDesc, ResourceLifetime},
    indirect::DrawIndexedCommand,
};

#[test]
fn frame_requires_ordered_begin_record_enqueue_submit_finish() {
    let mut frame = FrameRecorder::new(4).unwrap();
    assert_eq!(frame.submit(), Err(FrameError::NotRecording));
    frame.begin().unwrap();
    assert_eq!(frame.begin(), Err(FrameError::AlreadyRecording));
    frame
        .write_indirect(
            0,
            DrawIndexedCommand {
                index_count: 3,
                instance_count: 1,
                ..Default::default()
            },
        )
        .unwrap();

    let mut graph = FrameGraph::new();
    let resource = graph
        .add_resource(ResourceDesc::buffer(20, 4, ResourceLifetime::External).unwrap())
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("indirect", QueueKind::Graphics).access(Access::buffer(
                resource,
                BufferRange::new(0, 20).unwrap(),
                ResourceState::new(
                    QueueKind::Graphics,
                    ez_gfx_hal::ShaderStage::Vertex,
                    ResourceAccess::IndirectRead,
                )
                .unwrap(),
            )),
        )
        .unwrap();
    frame.enqueue(graph.compile().unwrap()).unwrap();
    let submission = frame.submit().unwrap();
    assert_eq!(submission.commands.len(), 1);
    assert_eq!(frame.state(), FrameState::Submitted);
    frame.finish().unwrap();
    assert_eq!(frame.state(), FrameState::Idle);
}

#[test]
fn indirect_bounds_and_missing_graph_are_rejected() {
    let mut frame = FrameRecorder::new(1).unwrap();
    frame.begin().unwrap();
    assert_eq!(
        frame.write_indirect(1, DrawIndexedCommand::default()),
        Err(FrameError::IndirectOutOfBounds)
    );
    assert_eq!(frame.submit(), Err(FrameError::MissingGraph));
}

#[test]
fn native_pipeline_work_is_a_submitable_graph_workload() {
    let mut frame = FrameRecorder::new(1).unwrap();
    frame.begin().unwrap();
    frame.mark_work_enqueued().unwrap();
    assert!(frame.submit().is_ok());
}
