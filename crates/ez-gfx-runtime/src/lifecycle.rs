use core::fmt;
use std::{
    sync::atomic::{AtomicU8, Ordering},
    thread::{self, ThreadId},
};

use ez_gfx_core::handle::{
    ContextHandle, GenerationalArena, HandleError, HandleParts, LocalHandle, PackedHandle,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// Classifies resources tracked by a graphics context.
pub enum ResourceKind {
    /// Native presentation surface.
    Surface = 1,
    /// Compiled shader resource.
    Shader = 2,
    /// Indirect command resource.
    Indirect = 3,
    /// Structured buffer resource.
    Structured = 4,
    /// Texture resource.
    Texture = 5,
    /// Render-target resource.
    RenderTarget = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// Indicates whether a graphics context can accept work.
pub enum ContextHealth {
    /// The context can accept work.
    Healthy = 0,
    /// The context is permanently unavailable.
    Lost = 1,
}

#[derive(Debug)]
/// Tracks context ownership, thread affinity, health, and resource identities.
pub struct ContextIdentity {
    /// Local handle used as the owner of packed resource handles.
    owner: LocalHandle,
    /// Thread permitted to access the context.
    creator: ThreadId,
    /// Atomically stored context health state.
    health: AtomicU8,
    /// Generational storage mapping child handles to resource kinds.
    resources: GenerationalArena<ResourceKind>,
}

impl ContextIdentity {
    /// Owner fields must fit the existing 20/20 context packing before any resource is inserted.
    ///
    /// # Errors
    ///
    /// Returns an error if the owner cannot be encoded as a context handle.
    pub fn new(owner: LocalHandle) -> Result<Self, LifecycleError> {
        PackedHandle::context(owner).map_err(LifecycleError::Handle)?;
        Ok(Self {
            owner,
            creator: thread::current().id(),
            health: AtomicU8::new(ContextHealth::Healthy as u8),
            resources: GenerationalArena::new(),
        })
    }

    /// # Panics
    ///
    /// Panics only if the validated owner cannot be repacked as a context handle.
    pub fn context_handle(&self) -> ContextHandle {
        ContextHandle::from_packed(
            PackedHandle::context(self.owner).expect("owner was validated at construction"),
        )
        .expect("context packing preserves context shape")
    }
    /// Returns the current context health with acquire ordering.
    pub fn health(&self) -> ContextHealth {
        match self.health.load(Ordering::Acquire) {
            0 => ContextHealth::Healthy,
            _ => ContextHealth::Lost,
        }
    }

    /// The first loss transition succeeds; later reports are observable but never repeat fan-out.
    ///
    /// # Errors
    ///
    /// Returns an error if the context has already been marked lost.
    pub fn mark_lost(&self) -> Result<(), LifecycleError> {
        self.health
            .compare_exchange(
                ContextHealth::Healthy as u8,
                ContextHealth::Lost as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| LifecycleError::AlreadyLost)
    }

    /// Loss takes precedence over affinity so no thread can begin work after a fatal native result.
    ///
    /// # Errors
    ///
    /// Returns an error if the context is lost or the current thread is not the creator thread.
    pub fn check_thread_and_health(&self) -> Result<(), LifecycleError> {
        if self.health() == ContextHealth::Lost {
            return Err(LifecycleError::DeviceLost);
        }
        if thread::current().id() != self.creator {
            return Err(LifecycleError::WrongThread);
        }
        Ok(())
    }

    /// Identity capacity is limited by the packed 12-bit child slot and generation fields.
    ///
    /// # Errors
    ///
    /// Returns an error if the context is lost, the current thread is not the creator thread, arena insertion fails, or the child handle cannot be packed.
    pub fn insert(&mut self, kind: ResourceKind) -> Result<PackedHandle, LifecycleError> {
        self.check_thread_and_health()?;
        let local = self
            .resources
            .insert(kind)
            .map_err(LifecycleError::Handle)?;
        match PackedHandle::child(self.owner, local) {
            Ok(handle) => Ok(handle),
            Err(error) => {
                let _ = self.resources.remove(local);
                Err(LifecycleError::Handle(error))
            }
        }
    }

    /// Owner, packed form, generation, and kind are all checked before typed resource lookup.
    ///
    /// # Errors
    ///
    /// Returns an error if the context is lost, the current thread is not the creator thread, the handle is invalid, is not a resource handle, has the wrong owner, is stale, or has the wrong resource kind.
    pub fn resolve(
        &self,
        handle: PackedHandle,
        expected: ResourceKind,
    ) -> Result<(), LifecycleError> {
        self.check_thread_and_health()?;
        let (owner, child) = child_parts(handle)?;
        if owner != self.owner {
            return Err(LifecycleError::WrongOwner);
        }
        let actual = self
            .resources
            .get(child)
            .map_err(|_| LifecycleError::StaleHandle)?;
        if *actual != expected {
            return Err(LifecycleError::WrongResourceKind);
        }
        Ok(())
    }

    /// Null/stale/wrong-kind destroys fail closed; generation advances before the slot is reused.
    ///
    /// # Errors
    ///
    /// Returns an error if resource resolution fails or the resolved child handle is stale when removed.
    pub fn remove(
        &mut self,
        handle: PackedHandle,
        expected: ResourceKind,
    ) -> Result<(), LifecycleError> {
        self.resolve(handle, expected)?;
        let (_, child) = child_parts(handle)?;
        self.resources
            .remove(child)
            .map_err(|_| LifecycleError::StaleHandle)?;
        Ok(())
    }

    /// Invalidates every tracked resource handle.
    pub fn invalidate_resources(&mut self) {
        let _ = self.resources.clear();
    }
}

/// Extracts owner and child handles from a packed resource handle.
///
/// # Errors
///
/// Returns an error if the packed handle is invalid or is a context handle rather than a resource handle.
fn child_parts(handle: PackedHandle) -> Result<(LocalHandle, LocalHandle), LifecycleError> {
    match handle.parts().map_err(LifecycleError::Handle)? {
        HandleParts::Child { owner, child } => Ok((owner, child)),
        HandleParts::Context(_) => Err(LifecycleError::ExpectedResource),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Reports lifecycle and handle-validation failures.
pub enum LifecycleError {
    /// Packed-handle encoding or arena operation failed.
    Handle(HandleError),
    /// The resource belongs to another context.
    WrongOwner,
    /// The resource kind differs from the expected kind.
    WrongResourceKind,
    /// The resource handle no longer identifies a live allocation.
    StaleHandle,
    /// A context handle was supplied where a resource handle was required.
    ExpectedResource,
    /// The operation ran on a thread other than the context creator.
    WrongThread,
    /// The context cannot accept work because the device was lost.
    DeviceLost,
    /// The context had already transitioned to the lost state.
    AlreadyLost,
}

impl fmt::Display for LifecycleError {
    /// Formats the lifecycle failure using its debug representation.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LifecycleError {}
