//! Runtime resource and frame execution APIs.

#![forbid(unsafe_code)]

mod api;
/// Resource binding and bind-group APIs.
pub mod binding;
/// Reusable runtime resource caches.
pub mod cache;
/// Resource descriptor types and validation.
pub mod descriptor;
/// Frame recording and execution APIs.
pub mod frame;
/// Geometry buffers and draw layout APIs.
pub mod geometry;
/// Render graph construction and scheduling.
pub mod graph;
/// Indirect draw and dispatch command APIs.
pub mod indirect;
mod lifecycle;
/// Runtime diagnostics, metrics, and tracing.
pub mod observability;
/// Render command and pipeline APIs.
pub mod render;
/// Shader modules and pipeline shader configuration.
pub mod shader;
/// Render-target and surface output APIs.
pub mod target;
/// Texture creation, views, and sampling APIs.
pub mod texture;

pub use api::{
    AdapterSelection, ContextOptions, PublicApiError, SurfaceOptions, SurfacePlatform, SurfaceState,
};
pub use lifecycle::{ContextHealth, ContextIdentity, LifecycleError, ResourceKind};

use core::fmt;
use std::collections::BTreeSet;

use ez_gfx_core::capability::{
    AdapterClass, AdapterInfo, CapabilityError, SemanticProfile, select_default_adapter,
};

#[derive(Clone, Debug, Eq, PartialEq)]
/// Validated adapters available for runtime selection.
pub struct AdapterCatalog {
    /// Adapters with unique stable identities.
    adapters: Vec<AdapterInfo>,
}

impl AdapterCatalog {
    /// Stable identity must be unique across the complete cross-backend enumeration.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::DuplicateAdapterIdentity` when two adapters have the same stable identity.
    pub fn new(adapters: Vec<AdapterInfo>) -> Result<Self, RuntimeError> {
        let mut identities = BTreeSet::new();
        for adapter in &adapters {
            if !identities.insert(adapter.stable_id()) {
                return Err(RuntimeError::DuplicateAdapterIdentity);
            }
        }
        Ok(Self { adapters })
    }

    /// Returns all enumerated adapters in catalog order.
    pub fn adapters(&self) -> &[AdapterInfo] {
        &self.adapters
    }

    /// Explicit selection still rejects software unless the caller opts in and always applies admission.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::AdapterNotFound` if no adapter has the requested stable identity, or an admission error if the matched adapter is disallowed software or does not satisfy the required semantic profile.
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

    /// Selects and admits the preferred adapter under the software policy.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::NoAdmittedAdapter` if default selection finds no eligible adapter, or an admission error if the selected adapter is disallowed software or does not satisfy the required semantic profile.
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
/// An adapter admitted for the runtime semantic profile.
pub struct AdmittedAdapter<'a> {
    /// Capabilities and identity of the admitted adapter.
    info: &'a AdapterInfo,
    /// Semantic profile satisfied by the adapter.
    profile: SemanticProfile,
}

impl<'a> AdmittedAdapter<'a> {
    /// Returns the admitted adapter information.
    pub const fn info(self) -> &'a AdapterInfo {
        self.info
    }
    /// Returns the semantic profile satisfied by the adapter.
    pub const fn profile(self) -> SemanticProfile {
        self.profile
    }
}

/// Verifies that an adapter satisfies software policy and profile requirements.
///
/// # Errors
///
/// Returns `RuntimeError::SoftwareAdapterNotAllowed` when software adapters are disallowed, or `RuntimeError::UnsupportedAdapter` when the adapter does not satisfy the required semantic profile.
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
    /// Stable identity of the adapter backing the context.
    adapter_id: [u8; 16],
    /// Semantic profile enforced for the context.
    profile: SemanticProfile,
    /// Backend-owned device used for runtime execution.
    device: D,
}

impl<D> Context<D> {
    /// Creates a context from an admitted adapter and backend device.
    pub fn new(adapter: AdmittedAdapter<'_>, device: D) -> Self {
        Self {
            adapter_id: adapter.info.stable_id(),
            profile: adapter.profile,
            device,
        }
    }

    /// Returns the stable identity of the backing adapter.
    pub const fn adapter_id(&self) -> [u8; 16] {
        self.adapter_id
    }
    /// Returns the context's semantic profile.
    pub const fn profile(&self) -> SemanticProfile {
        self.profile
    }
    /// Borrows the backend-owned device.
    pub const fn device(&self) -> &D {
        &self.device
    }
    /// Consumes the context and returns its backend-owned device.
    pub fn into_device(self) -> D {
        self.device
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Failures encountered while cataloging or admitting adapters.
pub enum RuntimeError {
    /// Two enumerated adapters reported the same stable identity.
    DuplicateAdapterIdentity,
    /// No adapter matched the requested stable identity.
    AdapterNotFound,
    /// A software adapter was requested without explicit permission.
    SoftwareAdapterNotAllowed,
    /// The adapter does not satisfy the required semantic profile.
    UnsupportedAdapter,
    /// No enumerated adapter passed selection and admission.
    NoAdmittedAdapter,
}

impl fmt::Display for RuntimeError {
    /// Writes the diagnostic name of the runtime error.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for RuntimeError {}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Per-adapter admission diagnosis for rejection reporting.
pub struct AdapterReport {
    /// Enumerated adapter this report diagnoses.
    adapter: AdapterInfo,
    /// Profile requirements the adapter fails. Empty when admitted.
    errors: Vec<CapabilityError>,
    /// Whether software policy alone rejects the adapter.
    software_rejected: bool,
}

impl AdapterReport {
    /// Returns the diagnosed adapter.
    pub const fn adapter(&self) -> &AdapterInfo {
        &self.adapter
    }
    /// Returns the unmet profile requirements. Empty when admitted.
    pub fn errors(&self) -> &[CapabilityError] {
        &self.errors
    }
    /// Returns whether software policy alone rejects the adapter.
    pub const fn software_rejected(&self) -> bool {
        self.software_rejected
    }
    /// Returns whether the adapter passes both policy and profile admission.
    pub const fn admitted(&self) -> bool {
        self.errors.is_empty() && !self.software_rejected
    }
}

/// Diagnoses one enumerated adapter without ranking or cataloging it.
///
/// Software policy is reported separately from profile errors so callers can
/// distinguish "retry with `allow_software`" from "hardware cannot qualify".
/// An all-zero stable identity is reported as rejected: it can never enter a
/// catalog, so it must never read as admitted here.
#[must_use]
pub fn admission_report(info: &AdapterInfo, allow_software: bool) -> AdapterReport {
    let software_rejected = info.class() == AdapterClass::Software && !allow_software;
    let errors = match SemanticProfile::V1.admit(info.capabilities()) {
        Ok(()) => Vec::new(),
        Err(errors) => errors,
    };
    AdapterReport {
        adapter: info.clone(),
        errors,
        software_rejected,
    }
}

#[cfg(test)]
mod adapter_tests {
    use super::*;
    use ez_gfx_core::Backend;
    use ez_gfx_core::capability::{AdapterCapabilities, CompressionSupport};

    fn capable() -> AdapterCapabilities {
        AdapterCapabilities {
            bindless_sampled_textures: 1024,
            bindless_storage_resources: 1024,
            bindless_samplers: 1024,
            max_indirect_draw_count: 65_535,
            shader_model: 0x0605,
            timeline_synchronization: true,
            resource_aliasing: true,
            dynamic_rendering: true,
            presentation: true,
            compression: CompressionSupport::BC,
        }
    }

    fn info(
        backend: Backend,
        id: u8,
        class: AdapterClass,
        caps: AdapterCapabilities,
    ) -> AdapterInfo {
        AdapterInfo::new(
            backend,
            [id; 16],
            format!("adapter-{id}"),
            format!("driver-{id}"),
            class,
            caps,
        )
        .expect("synthetic adapter identity is valid")
    }

    #[test]
    fn adapter_selection_defaults_to_first_fit() {
        let options =
            ContextOptions::new_for_backend(0, 0, 0, Backend::Vulkan).expect("valid options");
        assert_eq!(options.adapter_selection, None);
    }

    #[test]
    fn with_adapter_records_stable_identity_and_policy() {
        let options =
            ContextOptions::new_for_backend(0, 0, 0, Backend::Vulkan).expect("valid options");
        let selected = options.with_adapter([7; 16], true);
        assert_eq!(
            selected.adapter_selection,
            Some(AdapterSelection::new([7; 16], true))
        );
    }

    #[test]
    fn explicit_selection_bypasses_ranking_but_keeps_admission() {
        let weak = AdapterInfo::new(
            Backend::Vulkan,
            [9; 16],
            "weak",
            "driver",
            AdapterClass::Discrete,
            AdapterCapabilities {
                bindless_sampled_textures: 0,
                ..capable()
            },
        )
        .expect("synthetic adapter identity is valid");
        let catalog = AdapterCatalog::new(vec![
            info(Backend::Vulkan, 1, AdapterClass::Discrete, capable()),
            info(Backend::Dx12, 2, AdapterClass::Integrated, capable()),
            weak,
        ])
        .expect("unique stable identities");
        // Lower-ranked but admitted adapter is selectable explicitly.
        assert_eq!(
            catalog
                .select([2; 16], false)
                .expect("integrated adapter is admitted")
                .info()
                .stable_id(),
            [2; 16]
        );
        // Unknown identity fails before admission.
        assert_eq!(
            catalog.select([0xFF; 16], false).unwrap_err(),
            RuntimeError::AdapterNotFound
        );
        // Explicit selection never bypasses admission.
        assert_eq!(
            catalog.select([9; 16], false).unwrap_err(),
            RuntimeError::UnsupportedAdapter
        );
    }

    #[test]
    fn software_selection_requires_explicit_opt_in() {
        let catalog = AdapterCatalog::new(vec![info(
            Backend::Vulkan,
            3,
            AdapterClass::Software,
            capable(),
        )])
        .expect("unique stable identities");
        assert_eq!(
            catalog.select([3; 16], false).unwrap_err(),
            RuntimeError::SoftwareAdapterNotAllowed
        );
        assert_eq!(
            catalog
                .select([3; 16], true)
                .expect("opted-in software adapter is admitted")
                .info()
                .stable_id(),
            [3; 16]
        );
    }

    #[test]
    fn admission_report_separates_policy_from_profile() {
        let capable_software = info(Backend::Vulkan, 4, AdapterClass::Software, capable());
        let report = admission_report(&capable_software, false);
        assert!(report.errors().is_empty());
        assert!(report.software_rejected());
        assert!(!report.admitted());
        assert!(admission_report(&capable_software, true).admitted());

        let weak = info(
            Backend::Dx12,
            5,
            AdapterClass::Discrete,
            AdapterCapabilities {
                bindless_sampled_textures: 0,
                ..capable()
            },
        );
        let report = admission_report(&weak, false);
        assert!(!report.errors().is_empty());
        assert!(!report.software_rejected());
        assert!(!report.admitted());

        let strong = info(Backend::Vulkan, 6, AdapterClass::Discrete, capable());
        let report = admission_report(&strong, false);
        assert!(report.admitted());
        assert_eq!(report.adapter().stable_id(), [6; 16]);
    }
}
