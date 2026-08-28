#![forbid(unsafe_code)]

pub mod capability;
pub mod handle;
pub mod semantic;

pub use semantic::{
    Backend, ResourceAccess, ResourceKind, SemanticError, SemanticGraph, SemanticId,
    SemanticResource, TargetBinding, TargetLayout,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    InvalidArgument,
    InvalidContext,
    NativeFailure,
    NotReady,
    Unsupported,
    DeviceLost,
}

pub type Result<T> = core::result::Result<T, Status>;
