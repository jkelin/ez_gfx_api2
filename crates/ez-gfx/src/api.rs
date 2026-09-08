use ez_gfx_core::capability::CapabilityError;
use ez_gfx_runtime::LifecycleError;

/// Error returned by the safe Rust facade.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An argument violates the operation contract.
    #[error("invalid argument")]
    InvalidArgument,
    /// A context or resource handle is invalid or stale.
    #[error("invalid or stale context/resource handle")]
    InvalidContext,
    /// The native graphics backend failed.
    #[error("native graphics backend failure")]
    NativeFailure,
    /// Completion or output is not yet available.
    #[error("operation is not ready")]
    NotReady,
    /// The requested capability is unavailable.
    #[error("unsupported operation or capability")]
    Unsupported,
    /// The graphics device was lost.
    #[error("graphics device lost")]
    DeviceLost,
    /// Asynchronous scheduling or staging capacity is unavailable.
    #[error("asynchronous scheduling capacity unavailable")]
    QueueFull,
    /// An asynchronous operation was cancelled before completion.
    #[error("asynchronous operation cancelled")]
    Cancelled,
    /// Preserves a lifecycle or handle-validation cause.
    #[error(transparent)]
    Lifecycle(#[from] LifecycleError),
    /// Preserves an adapter capability cause.
    #[error(transparent)]
    Capability(#[from] CapabilityError),
}

/// Result returned by the safe Rust facade.
pub type Result<T> = std::result::Result<T, Error>;
