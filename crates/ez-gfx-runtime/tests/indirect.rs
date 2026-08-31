//! Runtime integration and contract tests.

use ez_gfx_runtime::indirect::*;

fn view(x: i32) -> DynamicView {
    DynamicView::new(
        Viewport::new(0.0, 0.0, 640.0, 480.0, 0.0, 1.0).unwrap(),
        Scissor::new(x, 0, 100, 100).unwrap(),
    )
    .unwrap()
}

#[test]
fn command_buffer_validates_capacity_and_standard_layout() {
    assert_eq!(core::mem::size_of::<DrawIndexedCommand>(), 20);
    let mut buffer = IndexedIndirectBuffer::new(2).unwrap();
    buffer
        .write(
            0,
            DrawIndexedCommand {
                index_count: 3,
                instance_count: 1,
                first_index: 0,
                vertex_offset: -1,
                first_instance: 0,
            },
        )
        .unwrap();
    assert_eq!(
        buffer.write(2, DrawIndexedCommand::default()),
        Err(IndirectError::OutOfBounds)
    );
    buffer.set_draw_count(1).unwrap();
    assert_eq!(buffer.draw_count(), 1);
    assert_eq!(buffer.set_draw_count(3), Err(IndirectError::OutOfBounds));
}

#[test]
fn invalid_viewports_and_scissors_fail() {
    assert_eq!(
        Viewport::new(0.0, 0.0, f32::NAN, 1.0, 0.0, 1.0),
        Err(IndirectError::InvalidViewport)
    );
    assert_eq!(Scissor::new(0, 0, 0, 1), Err(IndirectError::InvalidScissor));
}

#[test]
fn consecutive_equal_dynamic_state_forms_batches() {
    let views = [view(0), view(0), view(10), view(10), view(0)];
    assert_eq!(
        group_dynamic_views(&views),
        vec![
            DrawBatch {
                first: 0,
                count: 2,
                state: views[0]
            },
            DrawBatch {
                first: 2,
                count: 2,
                state: views[2]
            },
            DrawBatch {
                first: 4,
                count: 1,
                state: views[4]
            },
        ]
    );
}
