//! Lossless upload-event queue transition contracts.

use ez_gfx_core::handle::{LocalHandle, PackedHandle, TextureHandle};
use ez_gfx_runtime::upload::{UploadEvent, UploadEventQueue, UploadResource, UploadStatus};

// Child slots are capped by the packed-handle layout; queue depth is independent of that cap.
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
fn upload_events_are_lossless_and_fifo() {
    let mut queue = UploadEventQueue::new();
    for sequence in 0..10_000 {
        queue.push(UploadEvent {
            resource: UploadResource::Texture(texture(sequence % 1_000)),
            status: UploadStatus::SourceStaged,
        });
    }

    for sequence in 0..10_000 {
        assert_eq!(
            queue.pop(),
            Some(UploadEvent {
                resource: UploadResource::Texture(texture(sequence % 1_000)),
                status: UploadStatus::SourceStaged,
            })
        );
    }
    assert_eq!(queue.pop(), None);
}

#[test]
fn upload_event_transitions_preserve_terminal_outcomes() {
    let resource = UploadResource::Texture(texture(4));
    let mut queue = UploadEventQueue::new();
    queue.push(UploadEvent {
        resource,
        status: UploadStatus::SourceStaged,
    });
    queue.push(UploadEvent {
        resource,
        status: UploadStatus::DeviceReady,
    });
    queue.push(UploadEvent {
        resource,
        status: UploadStatus::Cancelled,
    });

    assert_eq!(queue.pop().unwrap().status, UploadStatus::SourceStaged);
    assert_eq!(queue.pop().unwrap().status, UploadStatus::DeviceReady);
    assert_eq!(queue.pop().unwrap().status, UploadStatus::Cancelled);
}
