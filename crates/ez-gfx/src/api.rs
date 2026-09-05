/// Result of a safe ez-gfx operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EzGfxResult {
    /// Operation completed successfully.
    Ok = 0,
    /// An argument violates the operation contract.
    InvalidArgument = 1,
    /// A context or resource handle is invalid or stale.
    InvalidContext = 2,
    /// The native graphics backend failed.
    NativeFailure = 3,
    /// Completion or output is not yet available.
    NotReady = 4,
    /// The requested capability is unavailable.
    Unsupported = 5,
    /// The graphics device was lost.
    DeviceLost = 6,
    /// A bounded asynchronous queue has no available capacity.
    QueueFull = 7,
    /// An asynchronous operation was cancelled before completion.
    Cancelled = 8,
}
