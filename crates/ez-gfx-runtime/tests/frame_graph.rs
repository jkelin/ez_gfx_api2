//! Runtime integration and contract tests.

use ez_gfx_hal::{
    BufferRange, CompletionToken, QueueKind, ResourceAccess, ResourceState, ShaderStage,
};
use ez_gfx_runtime::graph::*;

fn state(queue: QueueKind, stage: ShaderStage, access: ResourceAccess) -> ResourceState {
    ResourceState::new(queue, stage, access).unwrap()
}

#[test]
fn overlapping_ranges_emit_raw_war_waw_but_disjoint_ranges_do_not() {
    let mut graph = FrameGraph::new();
    let buffer = graph
        .add_resource(ResourceDesc::buffer(1024, 16, ResourceLifetime::Transient).unwrap())
        .unwrap();
    let write = state(
        QueueKind::Compute,
        ShaderStage::Compute,
        ResourceAccess::StorageWrite,
    );
    let read = state(
        QueueKind::Compute,
        ShaderStage::Compute,
        ResourceAccess::StorageRead,
    );
    let n0 = graph
        .add_node(
            NodeDesc::new("write", QueueKind::Compute).access(Access::buffer(
                buffer,
                BufferRange::new(0, 128).unwrap(),
                write,
            )),
        )
        .unwrap();
    let n1 = graph
        .add_node(
            NodeDesc::new("read", QueueKind::Compute).access(Access::buffer(
                buffer,
                BufferRange::new(64, 64).unwrap(),
                read,
            )),
        )
        .unwrap();
    let n2 = graph
        .add_node(
            NodeDesc::new("disjoint", QueueKind::Compute).access(Access::buffer(
                buffer,
                BufferRange::new(256, 64).unwrap(),
                read,
            )),
        )
        .unwrap();
    let n3 = graph
        .add_node(
            NodeDesc::new("rewrite", QueueKind::Compute).access(Access::buffer(
                buffer,
                BufferRange::new(64, 32).unwrap(),
                write,
            )),
        )
        .unwrap();

    let compiled = graph.compile().unwrap();
    assert!(
        compiled
            .hazards()
            .contains(&HazardEdge::new(n0, n1, buffer, HazardKind::Raw))
    );
    assert!(
        compiled
            .hazards()
            .contains(&HazardEdge::new(n1, n3, buffer, HazardKind::War))
    );
    assert!(
        compiled
            .hazards()
            .contains(&HazardEdge::new(n0, n3, buffer, HazardKind::Waw))
    );
    assert!(!compiled.hazards().iter().any(|edge| edge.to == n2));
    assert_eq!(compiled.order(), &[n0, n1, n2, n3]);
}

#[test]
fn image_subresources_feedback_and_cycles_fail_closed() {
    let mut graph = FrameGraph::new();
    let image = graph
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                4,
                2,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let read = state(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    );
    let write = state(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    );
    let access = ImageRange::new(0, 1, 0, 1).unwrap();
    assert_eq!(
        graph.add_node(
            NodeDesc::new("feedback", QueueKind::Graphics)
                .access(Access::image(image, access, read))
                .access(Access::image(image, access, write))
        ),
        Err(GraphError::Feedback { resource: image })
    );

    let a = graph
        .add_node(
            NodeDesc::new("a", QueueKind::Graphics).access(Access::image(image, access, write)),
        )
        .unwrap();
    let b = graph
        .add_node(
            NodeDesc::new("b", QueueKind::Graphics)
                .access(Access::image(image, access, read))
                .depends_on(a),
        )
        .unwrap();
    graph.add_dependency(b, a).unwrap();
    assert!(matches!(graph.compile(), Err(GraphError::Cycle { .. })));
}

#[test]
fn cross_queue_and_external_readiness_waits_are_explicit_and_history_survives() {
    let mut graph = FrameGraph::new();
    let history = graph
        .add_resource(
            ResourceDesc::image(
                32,
                32,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::PersistentHistory,
            )
            .unwrap(),
        )
        .unwrap();
    let ready = CompletionToken::new(QueueKind::Transfer, 9).unwrap();
    graph.set_resource_ready(history, ready).unwrap();
    let previous = state(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    );
    graph.set_history_state(history, previous).unwrap();
    let writer = graph
        .add_node(
            NodeDesc::new("compute", QueueKind::Compute).access(Access::image(
                history,
                ImageRange::all(1, 1).unwrap(),
                state(
                    QueueKind::Compute,
                    ShaderStage::Compute,
                    ResourceAccess::StorageWrite,
                ),
            )),
        )
        .unwrap();
    let reader = graph
        .add_node(
            NodeDesc::new("sample", QueueKind::Graphics).access(Access::image(
                history,
                ImageRange::all(1, 1).unwrap(),
                previous,
            )),
        )
        .unwrap();

    let compiled = graph.compile().unwrap();
    assert!(
        compiled
            .waits()
            .iter()
            .any(|wait| wait.node == writer && wait.external == Some(ready))
    );
    assert!(
        compiled
            .waits()
            .iter()
            .any(|wait| wait.node == reader && wait.source == Some(writer))
    );
    assert_eq!(compiled.history_state(history), Some(previous));
}

#[test]
fn first_use_load_rejects_transient_attachments() {
    for depth in [false, true] {
        let mut graph = FrameGraph::new();
        let format = if depth {
            Format::Depth32Float
        } else {
            Format::Rgba8Unorm
        };
        let attachment = graph
            .add_resource(
                ResourceDesc::image(16, 16, 1, 1, format, 1, ResourceLifetime::Transient).unwrap(),
            )
            .unwrap();
        let pass = PassInfo::new(
            (!depth).then_some(attachment).into_iter().collect(),
            depth.then_some(attachment),
            [0, 0, 16, 16],
            1,
            LoadOp::Load,
            StoreOp::Store,
        )
        .unwrap();
        let access = if depth {
            ResourceAccess::DepthStencilWrite
        } else {
            ResourceAccess::ColorAttachmentWrite
        };
        graph
            .add_node(
                NodeDesc::new("invalid-load", QueueKind::Graphics)
                    .pass(pass)
                    .access(Access::image(
                        attachment,
                        ImageRange::all(1, 1).unwrap(),
                        state(QueueKind::Graphics, ShaderStage::Fragment, access),
                    )),
            )
            .unwrap();

        assert!(matches!(graph.compile(), Err(GraphError::InvalidPass)));
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "This integration test covers the complete pass-merge contract."
)]
fn store_then_load_passes_merge_but_repeated_clear_and_transitions_do_not() {
    let write = state(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    );

    let mut compatible = FrameGraph::new();
    let attachment = compatible
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let first = PassInfo::new(
        vec![attachment],
        None,
        [0, 0, 64, 64],
        1,
        LoadOp::Clear,
        StoreOp::Store,
    )
    .unwrap();
    let second = PassInfo::new(
        vec![attachment],
        None,
        [0, 0, 64, 64],
        1,
        LoadOp::Load,
        StoreOp::Store,
    )
    .unwrap();
    compatible
        .add_node(
            NodeDesc::new("first", QueueKind::Graphics)
                .pass(first)
                .access(Access::image(
                    attachment,
                    ImageRange::all(1, 1).unwrap(),
                    write,
                )),
        )
        .unwrap();
    compatible
        .add_node(
            NodeDesc::new("second", QueueKind::Graphics)
                .pass(second)
                .access(Access::image(
                    attachment,
                    ImageRange::all(1, 1).unwrap(),
                    write,
                )),
        )
        .unwrap();
    assert_eq!(compatible.compile().unwrap().passes()[0].nodes.len(), 2);

    let mut repeated_clear = FrameGraph::new();
    let attachment = repeated_clear
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let clear = PassInfo::new(
        vec![attachment],
        None,
        [0, 0, 64, 64],
        1,
        LoadOp::Clear,
        StoreOp::Store,
    )
    .unwrap();
    repeated_clear
        .add_node(
            NodeDesc::new("first", QueueKind::Graphics)
                .pass(clear.clone())
                .access(Access::image(
                    attachment,
                    ImageRange::all(1, 1).unwrap(),
                    write,
                )),
        )
        .unwrap();
    repeated_clear
        .add_node(
            NodeDesc::new("second", QueueKind::Graphics)
                .pass(clear)
                .access(Access::image(
                    attachment,
                    ImageRange::all(1, 1).unwrap(),
                    write,
                )),
        )
        .unwrap();
    assert_eq!(repeated_clear.compile().unwrap().passes().len(), 2);

    let mut transition = FrameGraph::new();
    let attachment = transition
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let transition_buffer = transition
        .add_resource(ResourceDesc::buffer(16, 4, ResourceLifetime::External).unwrap())
        .unwrap();
    let storage_read = state(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::StorageRead,
    );
    let first = PassInfo::new(
        vec![attachment],
        None,
        [0, 0, 64, 64],
        1,
        LoadOp::Clear,
        StoreOp::Store,
    )
    .unwrap();
    let second = PassInfo::new(
        vec![attachment],
        None,
        [0, 0, 64, 64],
        1,
        LoadOp::Load,
        StoreOp::Store,
    )
    .unwrap();
    transition
        .add_node(
            NodeDesc::new("first", QueueKind::Graphics)
                .pass(first)
                .access(Access::image(
                    attachment,
                    ImageRange::all(1, 1).unwrap(),
                    write,
                )),
        )
        .unwrap();
    transition
        .add_node(
            NodeDesc::new("second", QueueKind::Graphics)
                .pass(second)
                .access(Access::image(
                    attachment,
                    ImageRange::all(1, 1).unwrap(),
                    write,
                ))
                .access(Access::buffer(
                    transition_buffer,
                    BufferRange::new(0, 16).unwrap(),
                    storage_read,
                )),
        )
        .unwrap();
    assert_eq!(transition.compile().unwrap().passes().len(), 2);
}

#[test]
fn alias_plan_has_no_backend_allocation_estimates() {
    let mut graph = FrameGraph::new();
    let a = graph
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                1,
                1,
                Format::Rgba16Float,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let b = graph
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                1,
                1,
                Format::Rgba16Float,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let write = state(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::ColorAttachmentWrite,
    );
    graph
        .add_node(
            NodeDesc::new("first", QueueKind::Graphics).access(Access::image(
                a,
                ImageRange::all(1, 1).unwrap(),
                write,
            )),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("second", QueueKind::Graphics).access(Access::image(
                b,
                ImageRange::all(1, 1).unwrap(),
                write,
            )),
        )
        .unwrap();

    assert_eq!(
        graph.compile().unwrap().alias(a).unwrap().slot,
        graph.compile().unwrap().alias(b).unwrap().slot
    );
    assert_eq!(
        core::mem::size_of::<AliasAssignment>(),
        core::mem::size_of::<u32>()
    );
}

#[test]
fn mixed_subresource_states_emit_each_required_transition() {
    let mut graph = FrameGraph::new();
    let image = graph
        .add_resource(
            ResourceDesc::image(
                64,
                64,
                2,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let write = state(
        QueueKind::Compute,
        ShaderStage::Compute,
        ResourceAccess::StorageWrite,
    );
    let read = state(
        QueueKind::Compute,
        ShaderStage::Compute,
        ResourceAccess::StorageRead,
    );
    let read_write = state(
        QueueKind::Compute,
        ShaderStage::Compute,
        ResourceAccess::StorageReadWrite,
    );
    graph
        .add_node(
            NodeDesc::new("mip-zero", QueueKind::Compute).access(Access::image(
                image,
                ImageRange::new(0, 1, 0, 1).unwrap(),
                write,
            )),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("mip-one", QueueKind::Compute).access(Access::image(
                image,
                ImageRange::new(1, 1, 0, 1).unwrap(),
                read,
            )),
        )
        .unwrap();
    let mixed = graph
        .add_node(
            NodeDesc::new("all", QueueKind::Compute).access(Access::image(
                image,
                ImageRange::all(2, 1).unwrap(),
                read_write,
            )),
        )
        .unwrap();

    let transitions: Vec<_> = graph
        .compile()
        .unwrap()
        .transitions()
        .iter()
        .filter(|transition| transition.node == mixed)
        .copied()
        .collect();
    assert_eq!(transitions.len(), 2);
    assert!(transitions.iter().any(|transition| transition.range
        == ResourceRange::Image(ImageRange::new(0, 1, 0, 1).unwrap())
        && transition.before == Some(write)));
    assert!(transitions.iter().any(|transition| transition.range
        == ResourceRange::Image(ImageRange::new(1, 1, 0, 1).unwrap())
        && transition.before == Some(read)));
}

#[test]
fn independent_cross_queue_transients_do_not_alias() {
    let mut graph = FrameGraph::new();
    let a = graph
        .add_resource(ResourceDesc::buffer(256, 16, ResourceLifetime::Transient).unwrap())
        .unwrap();
    let b = graph
        .add_resource(ResourceDesc::buffer(256, 16, ResourceLifetime::Transient).unwrap())
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("graphics", QueueKind::Graphics).access(Access::buffer(
                a,
                BufferRange::new(0, 256).unwrap(),
                state(
                    QueueKind::Graphics,
                    ShaderStage::Vertex,
                    ResourceAccess::StorageWrite,
                ),
            )),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("compute", QueueKind::Compute).access(Access::buffer(
                b,
                BufferRange::new(0, 256).unwrap(),
                state(
                    QueueKind::Compute,
                    ShaderStage::Compute,
                    ResourceAccess::StorageWrite,
                ),
            )),
        )
        .unwrap();

    let compiled = graph.compile().unwrap();
    assert_ne!(
        compiled.alias(a).unwrap().slot,
        compiled.alias(b).unwrap().slot
    );
}

#[test]
fn explicitly_ordered_cross_queue_transients_may_share_a_logical_slot() {
    let mut graph = FrameGraph::new();
    let a = graph
        .add_resource(ResourceDesc::buffer(256, 16, ResourceLifetime::Transient).unwrap())
        .unwrap();
    let b = graph
        .add_resource(ResourceDesc::buffer(256, 16, ResourceLifetime::Transient).unwrap())
        .unwrap();
    let first = graph
        .add_node(
            NodeDesc::new("graphics", QueueKind::Graphics).access(Access::buffer(
                a,
                BufferRange::new(0, 256).unwrap(),
                state(
                    QueueKind::Graphics,
                    ShaderStage::Vertex,
                    ResourceAccess::StorageWrite,
                ),
            )),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("compute", QueueKind::Compute)
                .depends_on(first)
                .access(Access::buffer(
                    b,
                    BufferRange::new(0, 256).unwrap(),
                    state(
                        QueueKind::Compute,
                        ShaderStage::Compute,
                        ResourceAccess::StorageWrite,
                    ),
                )),
        )
        .unwrap();

    let compiled = graph.compile().unwrap();
    assert_eq!(
        compiled.alias(a).unwrap().slot,
        compiled.alias(b).unwrap().slot
    );
    assert!(
        compiled
            .waits()
            .iter()
            .any(|wait| wait.source == Some(first))
    );
}

#[test]
fn explicit_order_orients_hazards_in_scheduled_direction() {
    let mut graph = FrameGraph::new();
    let buffer = graph
        .add_resource(ResourceDesc::buffer(64, 4, ResourceLifetime::External).unwrap())
        .unwrap();
    let reader = graph
        .add_node(
            NodeDesc::new("reader", QueueKind::Compute).access(Access::buffer(
                buffer,
                BufferRange::new(0, 64).unwrap(),
                state(
                    QueueKind::Compute,
                    ShaderStage::Compute,
                    ResourceAccess::StorageRead,
                ),
            )),
        )
        .unwrap();
    let writer = graph
        .add_node(
            NodeDesc::new("writer", QueueKind::Compute).access(Access::buffer(
                buffer,
                BufferRange::new(0, 64).unwrap(),
                state(
                    QueueKind::Compute,
                    ShaderStage::Compute,
                    ResourceAccess::StorageWrite,
                ),
            )),
        )
        .unwrap();
    graph.add_dependency(writer, reader).unwrap();

    let compiled = graph.compile().unwrap();

    assert_eq!(compiled.order(), &[writer, reader]);
    assert_eq!(
        compiled.hazards(),
        &[HazardEdge::new(writer, reader, buffer, HazardKind::Raw)]
    );
}

#[test]
fn transient_attachment_load_requires_the_loaded_subresource_to_survive() {
    let mut graph = FrameGraph::new();
    let image = graph
        .add_resource(
            ResourceDesc::image(
                16,
                16,
                2,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("other mip", QueueKind::Compute).access(Access::image(
                image,
                ImageRange::new(1, 1, 0, 1).unwrap(),
                state(
                    QueueKind::Compute,
                    ShaderStage::Compute,
                    ResourceAccess::StorageWrite,
                ),
            )),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("load mip zero", QueueKind::Graphics)
                .access(Access::image(
                    image,
                    ImageRange::new(0, 1, 0, 1).unwrap(),
                    state(
                        QueueKind::Graphics,
                        ShaderStage::Fragment,
                        ResourceAccess::ColorAttachmentWrite,
                    ),
                ))
                .pass(
                    PassInfo::new(
                        vec![image],
                        None,
                        [0, 0, 16, 16],
                        1,
                        LoadOp::Load,
                        StoreOp::Store,
                    )
                    .unwrap(),
                ),
        )
        .unwrap();

    assert!(matches!(graph.compile(), Err(GraphError::InvalidPass)));
}

#[test]
fn discarded_transient_attachment_cannot_be_loaded_later() {
    let mut graph = FrameGraph::new();
    let image = graph
        .add_resource(
            ResourceDesc::image(
                16,
                16,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .unwrap(),
        )
        .unwrap();
    let attachment = Access::image(
        image,
        ImageRange::all(1, 1).unwrap(),
        state(
            QueueKind::Graphics,
            ShaderStage::Fragment,
            ResourceAccess::ColorAttachmentWrite,
        ),
    );
    graph
        .add_node(
            NodeDesc::new("discard", QueueKind::Graphics)
                .access(attachment)
                .pass(
                    PassInfo::new(
                        vec![image],
                        None,
                        [0, 0, 16, 16],
                        1,
                        LoadOp::Clear,
                        StoreOp::Discard,
                    )
                    .unwrap(),
                ),
        )
        .unwrap();
    graph
        .add_node(
            NodeDesc::new("load discarded", QueueKind::Graphics)
                .access(attachment)
                .pass(
                    PassInfo::new(
                        vec![image],
                        None,
                        [0, 0, 16, 16],
                        1,
                        LoadOp::Load,
                        StoreOp::Store,
                    )
                    .unwrap(),
                ),
        )
        .unwrap();

    assert!(matches!(graph.compile(), Err(GraphError::InvalidPass)));
}

#[test]
fn image_descriptions_reject_mips_beyond_the_terminal_texel() {
    assert!(matches!(
        ResourceDesc::image(
            1,
            1,
            2,
            1,
            Format::Rgba8Unorm,
            1,
            ResourceLifetime::Transient,
        ),
        Err(GraphError::InvalidResource)
    ));
}
