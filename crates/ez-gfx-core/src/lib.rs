//! Shared resource, capability, and semantic types for the graphics API.
#![forbid(unsafe_code)]

/// Normalized adapter capability limits and feature support.
pub mod capability;
/// Opaque local and packed resource handle types.
pub mod handle;
/// Backend-independent resource and access semantics.
pub mod semantic;

pub use semantic::{
    Backend, ResourceAccess, ResourceKind, SemanticError, SemanticGraph, SemanticId,
    SemanticResource, TargetBinding, TargetLayout,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Coarse status returned by API operations.
pub enum Status {
    /// An argument failed validation.
    InvalidArgument,
    /// The referenced context is invalid.
    InvalidContext,
    /// A native operation failed.
    NativeFailure,
    /// The requested operation is not ready.
    NotReady,
    /// The requested feature is unsupported.
    Unsupported,
    /// The device was lost.
    DeviceLost,
}

/// Result type for operations that return only a status on failure.
pub type Result<T> = core::result::Result<T, Status>;
