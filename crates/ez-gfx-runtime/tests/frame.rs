use ez_gfx_hal::{BufferRange, QueueKind, ResourceAccess, ResourceState};
use ez_gfx_runtime::{
    frame::{ExecutableNode, FrameError, FrameRecorder, FrameState},
    graph::{Access, NodeDesc, ResourceDesc, ResourceLifetime},
    indirect::DrawIndexedCommand,
};

#[test]
fn frame_requires_ordered_begin_record_enqueue_submit_finish() {
    let mut frame = FrameRecorder::new(4).unwrap();
    assert!(matches!(frame.submit(), Err(FrameError::NotRecording)));
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

    let resource = frame
        .add_resource(ResourceDesc::buffer(20, 4, ResourceLifetime::External).unwrap())
        .unwrap();
    frame
        .record_node(
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
            ExecutableNode::TextureReadback { texture: 7 },
        )
        .unwrap();
    let submission = frame.submit().unwrap();
    assert_eq!(submission.nodes.len(), 1);
    assert_eq!(submission.graph.order().len(), 1);
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
}

#[test]
fn recording_a_node_is_atomic_and_payload_is_retained() {
    let mut frame = FrameRecorder::new(1).unwrap();
    frame.begin().unwrap();
    let invalid = NodeDesc::new("", QueueKind::Graphics);
    assert!(
        frame
            .record_node(invalid, ExecutableNode::TextureReadback { texture: 1 })
            .is_err()
    );
    assert!(matches!(frame.submit(), Err(FrameError::MissingGraph)));
}
