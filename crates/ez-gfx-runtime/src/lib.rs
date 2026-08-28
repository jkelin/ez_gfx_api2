#![forbid(unsafe_code)]

mod api;
pub mod binding;
pub mod cache;
pub mod descriptor;
pub mod frame;
pub mod geometry;
pub mod graph;
pub mod indirect;
mod lifecycle;
pub mod observability;
pub mod render;
pub mod shader;
pub mod target;
pub mod texture;

pub use api::{ContextOptions, PublicApiError, SurfaceOptions, SurfacePlatform, SurfaceState};
pub use lifecycle::{ContextHealth, ContextIdentity, LifecycleError, ResourceKind};

use core::fmt;
use std::collections::BTreeSet;

use ez_gfx_core::capability::{AdapterClass, AdapterInfo, SemanticProfile, select_default_adapter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterCatalog {
    adapters: Vec<AdapterInfo>,
}

impl AdapterCatalog {
    /// Stable identity must be unique across the complete cross-backend enumeration.
    pub fn new(adapters: Vec<AdapterInfo>) -> Result<Self, RuntimeError> {
        let mut identities = BTreeSet::new();
        for adapter in &adapters {
            if !identities.insert(adapter.stable_id()) {
                return Err(RuntimeError::DuplicateAdapterIdentity);
            }
        }
        Ok(Self { adapters })
    }

    pub fn adapters(&self) -> &[AdapterInfo] {
        &self.adapters
    }

    /// Explicit selection still rejects software unless the caller opts in and always applies admission.
    pub fn select(
        &self,
        stable_id: [u8; 16],
        allow_software: bool,
    ) -> Result<AdmittedAdapter<'_>, RuntimeError> {
        let info = self
            .adapters
            .iter()
            .find(|adapter| adapter.stable_id() == stable_id)
            .ok_or(RuntimeError::AdapterNotFound)?;
        admit(info, allow_software)
    }

    pub fn select_default(
        &self,
        allow_software: bool,
    ) -> Result<AdmittedAdapter<'_>, RuntimeError> {
        let info = select_default_adapter(&self.adapters, allow_software)
            .map_err(|_| RuntimeError::NoAdmittedAdapter)?;
        admit(info, allow_software)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmittedAdapter<'a> {
    info: &'a AdapterInfo,
    profile: SemanticProfile,
}

impl<'a> AdmittedAdapter<'a> {
    pub const fn info(self) -> &'a AdapterInfo {
        self.info
    }
    pub const fn profile(self) -> SemanticProfile {
        self.profile
    }
}

fn admit(info: &AdapterInfo, allow_software: bool) -> Result<AdmittedAdapter<'_>, RuntimeError> {
    if info.class() == AdapterClass::Software && !allow_software {
        return Err(RuntimeError::SoftwareAdapterNotAllowed);
    }
    SemanticProfile::V1
        .admit(info.capabilities())
        .map_err(|_| RuntimeError::UnsupportedAdapter)?;
    Ok(AdmittedAdapter {
        info,
        profile: SemanticProfile::V1,
    })
}

/// A context can only be assembled from an admitted adapter and a real backend-owned device.
pub struct Context<D> {
    adapter_id: [u8; 16],
    profile: SemanticProfile,
    device: D,
}

impl<D> Context<D> {
    pub fn new(adapter: AdmittedAdapter<'_>, device: D) -> Self {
        Self {
            adapter_id: adapter.info.stable_id(),
            profile: adapter.profile,
            device,
        }
    }

    pub const fn adapter_id(&self) -> [u8; 16] {
        self.adapter_id
    }
    pub const fn profile(&self) -> SemanticProfile {
        self.profile
    }
    pub const fn device(&self) -> &D {
        &self.device
    }
    pub fn into_device(self) -> D {
        self.device
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    DuplicateAdapterIdentity,
    AdapterNotFound,
    SoftwareAdapterNotAllowed,
    UnsupportedAdapter,
    NoAdmittedAdapter,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for RuntimeError {}
