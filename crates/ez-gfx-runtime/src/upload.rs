//! Lossless context upload progress events.

use std::collections::VecDeque;

use ez_gfx_core::handle::{IndexAllocationHandle, TextureHandle, VertexAllocationHandle};

use crate::observability::RuntimeStatus;

/// Typed resource associated with one upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UploadResource {
    /// Asynchronously decoded and uploaded texture.
    Texture(TextureHandle),
    /// Range within a named vertex heap.
    Vertex(VertexAllocationHandle),
    /// Range within the global index heap.
    Index(IndexAllocationHandle),
}

/// Observable upload transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UploadStatus {
    /// Caller input has been copied into runtime-owned memory and may be released.
    SourceStaged,
    /// Device-local data and publication prerequisites are complete; rendering may consume it.
    DeviceReady,
    /// Upload ended with a terminal error.
    Failed(RuntimeStatus),
    /// Upload was cancelled before readiness.
    Cancelled,
}

/// One typed upload transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UploadEvent {
    /// Resource whose upload changed state.
    pub resource: UploadResource,
    /// New progress or terminal state.
    pub status: UploadStatus,
}

/// Context-owned, lossless FIFO drained by the application once per frame.
#[derive(Debug, Default)]
pub struct UploadEventQueue {
    events: VecDeque<UploadEvent>,
}

impl UploadEventQueue {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one event without an artificial admission limit.
    pub fn push(&mut self, event: UploadEvent) {
        self.events.push_back(event);
    }

    /// Removes the oldest pending event.
    pub fn pop(&mut self) -> Option<UploadEvent> {
        self.events.pop_front()
    }
}
