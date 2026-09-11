//! Runtime integration and contract tests.

use ez_gfx_core::handle::{LocalHandle, PackedHandle, TextureHandle};
use ez_gfx_hal::{
    BufferRange, CompletionToken, ExecutionAction, QueueKind, ResourceAccess, ResourceState,
    ShaderStage,
};
use ez_gfx_runtime::{
    frame::{ExecutableNode, FrameError, FrameRecorder, FrameState},
    graph::{
        Access, Format, ImageRange, LoadOp, NodeDesc, PassInfo, ResourceDesc, ResourceLifetime,
        StoreOp,
    },
    indirect::DrawIndexedCommand,
};

fn texture(slot: u32) -> TextureHandle {
    TextureHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(slot, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn frame_requires_ordered_begin_record_enqueue_submit_finish() {
    let mut frame = FrameRecorder::new(4).unwrap();
    assert!(matches!(frame.submit(), Err(FrameError::NotRecording)));
    frame.begin().unwrap();
    assert_eq!(frame.begin(), Err(FrameError::AlreadyRecording));
    frame
        .write_counter(
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
            ExecutableNode::TextureReadback {
                texture: texture(7),
            },
        )
        .unwrap();
    let submission = frame.submit().unwrap();
    assert_eq!(submission.nodes.len(), 1);
    assert_eq!(submission.graph.order().len(), 1);
    assert_eq!(frame.state(), FrameState::Submitted);
    frame.finish(submission).unwrap();
    assert_eq!(frame.state(), FrameState::Idle);
}

#[test]
fn frame_workspace_reuses_capacity_after_submission_recycling() {
    let mut frame = FrameRecorder::new(1).unwrap();

    let run_frame = |frame: &mut FrameRecorder| {
        frame.begin().unwrap();
        let resource = frame
            .add_resource(ResourceDesc::buffer(4, 4, ResourceLifetime::External).unwrap())
            .unwrap();
        frame
            .record_node(
                NodeDesc::new("read", QueueKind::Graphics).access(Access::buffer(
                    resource,
                    BufferRange::new(0, 4).unwrap(),
                    ResourceState::new(
                        QueueKind::Graphics,
                        ez_gfx_hal::ShaderStage::Vertex,
                        ResourceAccess::SampledRead,
                    )
                    .unwrap(),
                )),
                ExecutableNode::TextureReadback {
                    texture: texture(7),
                },
            )
            .unwrap();
        let submission = frame.submit().unwrap();
        frame.finish(submission).unwrap();
    };

    run_frame(&mut frame);
    run_frame(&mut frame);
    let warmed = frame.workspace_stats();
    run_frame(&mut frame);

    assert_eq!(
        frame.workspace_stats().retained_bytes,
        warmed.retained_bytes
    );
    assert_eq!(
        frame.workspace_stats().high_water_bytes,
        warmed.high_water_bytes
    );
    assert!(warmed.retained_bytes <= warmed.byte_limit);
}

#[test]
fn recycled_render_pass_vectors_do_not_accumulate_prior_nodes() {
    let mut frame = FrameRecorder::new(1).unwrap();

    for _ in 0..3 {
        frame.begin().unwrap();
        let color = frame
            .add_resource(
                ResourceDesc::image(
                    4,
                    4,
                    1,
                    1,
                    Format::Rgba8Unorm,
                    1,
                    ResourceLifetime::External,
                )
                .unwrap(),
            )
            .unwrap();
        let attachment = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::Fragment,
            ResourceAccess::ColorAttachmentWrite,
        )
        .unwrap();
        frame
            .record_node(
                NodeDesc::new("pass", QueueKind::Graphics)
                    .access(Access::image(
                        color,
                        ImageRange::all(1, 1).unwrap(),
                        attachment,
                    ))
                    .pass(
                        PassInfo::new(
                            vec![color],
                            None,
                            [0, 0, 4, 4],
                            1,
                            LoadOp::Clear,
                            StoreOp::Store,
                        )
                        .unwrap(),
                    ),
                ExecutableNode::TextureReadback {
                    texture: texture(7),
                },
            )
            .unwrap();

        let submission = frame.submit().unwrap();
        let pass = submission
            .plan
            .actions
            .iter()
            .find_map(|action| match action {
                ExecutionAction::BeginPass(pass) => Some(pass),
                _ => None,
            })
            .unwrap();
        assert_eq!(pass.nodes, [0]);
        assert_eq!(pass.colors, [color.index()]);
        frame.finish(submission).unwrap();
    }
}

#[test]
fn counter_bounds_and_missing_graph_are_rejected() {
    let mut frame = FrameRecorder::new(1).unwrap();
    frame.begin().unwrap();
    assert_eq!(
        frame.write_counter(1, DrawIndexedCommand::default()),
        Err(FrameError::CounterOutOfBounds)
    );
}

#[test]
fn recording_a_node_is_atomic_and_payload_is_retained() {
    let mut frame = FrameRecorder::new(1).unwrap();
    frame.begin().unwrap();
    let invalid = NodeDesc::new("", QueueKind::Graphics);
    assert!(
        frame
            .record_node(
                invalid,
                ExecutableNode::TextureReadback {
                    texture: texture(1),
                },
            )
            .is_err()
    );
    assert!(matches!(frame.submit(), Err(FrameError::MissingGraph)));
}

fn cached_submission(
    frame: &mut FrameRecorder,
    size: u64,
    name: &str,
    ready: u64,
    initial: ResourceState,
) -> ez_gfx_runtime::frame::FrameSubmission {
    frame.begin().unwrap();
    let resource = frame
        .add_resource(ResourceDesc::buffer(size, 4, ResourceLifetime::External).unwrap())
        .unwrap();
    frame
        .set_resource_ready(
            resource,
            CompletionToken {
                queue: QueueKind::Transfer,
                value: ready,
            },
        )
        .unwrap();
    frame.set_resource_initial_state(resource, initial).unwrap();
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .unwrap();
    frame
        .record_node(
            NodeDesc::new(name, QueueKind::Graphics).access(Access::buffer(
                resource,
                BufferRange::new(0, size).unwrap(),
                sampled,
            )),
            ExecutableNode::TextureReadback {
                texture: texture(u32::try_from(ready).unwrap()),
            },
        )
        .unwrap();
    frame.submit().unwrap()
}
#[test]
fn frame_rejects_submission_from_another_recorder() {
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .unwrap();
    let mut first = FrameRecorder::new(1).unwrap();
    let mut second = FrameRecorder::new(1).unwrap();
    let first_submission = cached_submission(&mut first, 4, "first", 1, sampled);
    let second_submission = cached_submission(&mut second, 4, "second", 2, sampled);

    assert_eq!(
        first.finish(second_submission),
        Err(FrameError::SubmissionMismatch)
    );
    assert_eq!(first.state(), FrameState::Submitted);
    first.finish(first_submission).unwrap();
    second.abort();
}

#[test]
fn graph_template_hit_recomputes_dynamic_waits_and_transitions() {
    let mut frame = FrameRecorder::new(1).unwrap();
    let transfer = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::None,
        ResourceAccess::TransferRead,
    )
    .unwrap();
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .unwrap();

    let first = cached_submission(&mut frame, 4, "first name", 1, transfer);
    assert_eq!(first.graph.transitions().len(), 1);
    assert_eq!(first.graph.waits()[0].external.unwrap().value, 1);
    frame.finish(first).unwrap();

    let second = cached_submission(&mut frame, 4, "renamed node", 9, sampled);
    assert!(second.graph.transitions().is_empty());
    assert_eq!(second.graph.waits()[0].external.unwrap().value, 9);
    assert_eq!(
        second.nodes[0],
        ExecutableNode::TextureReadback {
            texture: texture(9),
        },
    );
    frame.finish(second).unwrap();

    let stats = frame.workspace_stats().graph_cache;
    assert_eq!((stats.hits, stats.misses, stats.compiles), (1, 1, 1));
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes <= stats.high_water_bytes);
    assert_eq!(stats.schema, 1);
}

#[test]
fn graph_template_cache_evicts_lru_and_invalidates_generation() {
    let mut frame = FrameRecorder::new(1).unwrap();
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .unwrap();
    for size in 1..=8 {
        let submission = cached_submission(&mut frame, size * 4, "shape", size, sampled);
        frame.finish(submission).unwrap();
    }

    let hit = cached_submission(&mut frame, 4, "shape", 20, sampled);
    frame.finish(hit).unwrap();
    let ninth = cached_submission(&mut frame, 36, "shape", 21, sampled);
    frame.finish(ninth).unwrap();
    let evicted = cached_submission(&mut frame, 8, "shape", 22, sampled);
    frame.finish(evicted).unwrap();
    let before = frame.workspace_stats().graph_cache;
    assert_eq!(before.entries, 8);
    assert_eq!(before.evictions, 2);
    assert_eq!((before.hits, before.misses, before.compiles), (1, 10, 10));
    assert_eq!(before.entry_limit, 8);
    assert_eq!(before.byte_limit, 8 * 1024 * 1024);
    assert!(before.retained_bytes <= before.byte_limit);
    assert!(before.high_water_bytes <= before.byte_limit);

    frame.invalidate_graph_templates();
    let invalidated = frame.workspace_stats().graph_cache;
    assert_eq!(invalidated.entries, 0);
    assert_eq!(invalidated.invalidations, 1);
    assert_eq!(invalidated.generation, before.generation + 1);
    assert!(invalidated.retained_bytes < before.retained_bytes);
    assert!(invalidated.retained_bytes <= invalidated.byte_limit);
    assert_eq!(invalidated.entry_limit, 8);
}

#[test]
fn graph_template_hit_recomputes_history_derived_transitions() {
    let mut frame = FrameRecorder::new(1).unwrap();
    let transfer = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::None,
        ResourceAccess::TransferRead,
    )
    .unwrap();
    let sampled = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .unwrap();

    for (index, carried) in [transfer, sampled].into_iter().enumerate() {
        frame.begin().unwrap();
        let resource = frame
            .add_resource(ResourceDesc::buffer(4, 4, ResourceLifetime::PersistentHistory).unwrap())
            .unwrap();
        frame.set_history_state(resource, carried).unwrap();
        frame
            .record_node(
                NodeDesc::new("history", QueueKind::Graphics).access(Access::buffer(
                    resource,
                    BufferRange::new(0, 4).unwrap(),
                    sampled,
                )),
                ExecutableNode::TextureReadback {
                    texture: texture(7),
                },
            )
            .unwrap();
        let submission = frame.submit().unwrap();
        assert_eq!(
            submission.graph.transitions().len(),
            usize::from(index == 0)
        );
        assert_eq!(submission.graph.history_state(resource), Some(sampled));
        frame.finish(submission).unwrap();
    }

    let stats = frame.workspace_stats().graph_cache;
    assert_eq!((stats.hits, stats.misses, stats.compiles), (1, 1, 1));
}
