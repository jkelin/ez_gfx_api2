use core::fmt;
use std::{
    sync::atomic::{AtomicU8, Ordering},
    thread::{self, ThreadId},
};

use ez_gfx_core::handle::{GenerationalArena, HandleError, HandleParts, LocalHandle, PackedHandle};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ResourceKind {
    Surface = 1,
    Shader = 2,
    Indirect = 3,
    Structured = 4,
    Texture = 5,
    RenderTarget = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ContextHealth {
    Healthy = 0,
    Lost = 1,
}

#[derive(Debug)]
pub struct ContextIdentity {
    owner: LocalHandle,
    creator: ThreadId,
    health: AtomicU8,
    resources: GenerationalArena<ResourceKind>,
}

impl ContextIdentity {
    /// Owner fields must fit the existing 20/20 context packing before any resource is inserted.
    pub fn new(owner: LocalHandle) -> Result<Self, LifecycleError> {
        PackedHandle::context(owner).map_err(LifecycleError::Handle)?;
        Ok(Self {
            owner,
            creator: thread::current().id(),
            health: AtomicU8::new(ContextHealth::Healthy as u8),
            resources: GenerationalArena::new(),
        })
    }

    pub fn context_handle(&self) -> PackedHandle {
        PackedHandle::context(self.owner).expect("owner was validated at construction")
    }

    pub fn health(&self) -> ContextHealth {
        match self.health.load(Ordering::Acquire) {
            0 => ContextHealth::Healthy,
            _ => ContextHealth::Lost,
        }
    }

    /// The first loss transition succeeds; later reports are observable but never repeat fan-out.
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

    pub fn invalidate_resources(&mut self) {
        let _ = self.resources.clear();
    }
}

fn child_parts(handle: PackedHandle) -> Result<(LocalHandle, LocalHandle), LifecycleError> {
    match handle.parts().map_err(LifecycleError::Handle)? {
        HandleParts::Child { owner, child } => Ok((owner, child)),
        HandleParts::Context(_) => Err(LifecycleError::ExpectedResource),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleError {
    Handle(HandleError),
    WrongOwner,
    WrongResourceKind,
    StaleHandle,
    ExpectedResource,
    WrongThread,
    DeviceLost,
    AlreadyLost,
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LifecycleError {}
