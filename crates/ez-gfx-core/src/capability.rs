use core::fmt;

use crate::Backend;

/// Version of the normalized capability profile schema.
pub const CAPABILITY_PROFILE_SCHEMA_VERSION: u32 = 1;
/// Maximum supported bindless sampled textures in the semantic profile.
pub const MAX_BINDLESS_SAMPLED_TEXTURES: u32 = 1024;

/// Bit flags describing compressed texture support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct CompressionSupport(u8);

impl CompressionSupport {
    /// No compressed texture formats.
    pub const NONE: Self = Self(0);
    /// Block compression support.
    pub const BC: Self = Self(1 << 0);
    /// Adaptive scalable texture compression support.
    pub const ASTC: Self = Self(1 << 1);

    #[must_use]
    /// Returns the union of two compression support sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns whether any format in `other` is supported.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Returns the encoded compression flags.
    pub const fn bits(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Hardware limits and feature flags exposed through the normalized adapter profile.
#[allow(
    clippy::struct_excessive_bools,
    reason = "These independent feature flags are the stable cross-backend capability model."
)]
pub struct AdapterCapabilities {
    /// Maximum number of sampled textures addressable through bindless access.
    pub bindless_sampled_textures: u32,
    /// Maximum number of storage resources addressable through bindless access.
    pub bindless_storage_resources: u32,
    /// Maximum number of samplers addressable through bindless access.
    pub bindless_samplers: u32,
    /// Maximum number of indirect draws supported by one submission.
    pub max_indirect_draw_count: u32,
    /// Backend-native shader capability normalized as `(major << 8) | minor`.
    pub shader_model: u32,
    /// Whether timeline synchronization is available.
    pub timeline_synchronization: bool,
    /// Whether resources may alias the same allocation.
    pub resource_aliasing: bool,
    /// Whether dynamic rendering is available.
    pub dynamic_rendering: bool,
    /// Whether presentation to a surface is available.
    pub presentation: bool,
    /// Texture compression formats supported by the adapter.
    pub compression: CompressionSupport,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Semantic profile versions understood by the runtime.
pub enum SemanticProfile {
    /// Version one of the semantic profile.
    V1,
}

impl SemanticProfile {
    /// Returns the schema version used to encode this profile.
    pub const fn schema_version(self) -> u32 {
        CAPABILITY_PROFILE_SCHEMA_VERSION
    }

    /// Admission accumulates every missing requirement for one deterministic diagnosis.
    ///
    /// # Errors
    ///
    /// Returns every unmet numeric limit, feature flag, or compression requirement.
    pub fn admit(self, capabilities: &AdapterCapabilities) -> Result<(), Vec<CapabilityError>> {
        let mut errors = Vec::new();
        require_limit(
            &mut errors,
            "bindless_sampled_textures",
            MAX_BINDLESS_SAMPLED_TEXTURES,
            capabilities.bindless_sampled_textures,
        );
        require_limit(
            &mut errors,
            "bindless_storage_resources",
            1024,
            capabilities.bindless_storage_resources,
        );
        require_limit(
            &mut errors,
            "bindless_samplers",
            MAX_BINDLESS_SAMPLED_TEXTURES,
            capabilities.bindless_samplers,
        );
        require_limit(
            &mut errors,
            "max_indirect_draw_count",
            65_535,
            capabilities.max_indirect_draw_count,
        );
        require_limit(
            &mut errors,
            "shader_model",
            0x0605,
            capabilities.shader_model,
        );
        require_feature(
            &mut errors,
            "timeline_synchronization",
            capabilities.timeline_synchronization,
        );
        require_feature(
            &mut errors,
            "resource_aliasing",
            capabilities.resource_aliasing,
        );
        require_feature(
            &mut errors,
            "dynamic_rendering",
            capabilities.dynamic_rendering,
        );
        require_feature(&mut errors, "presentation", capabilities.presentation);
        if !capabilities
            .compression
            .intersects(CompressionSupport::BC.union(CompressionSupport::ASTC))
        {
            errors.push(CapabilityError::MissingCompression);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Failure reported when an adapter does not satisfy the semantic profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityError {
    /// A numeric capability is below its required minimum.
    Limit {
        /// Name of the capability.
        name: &'static str,
        /// Minimum value required by the profile.
        required: u32,
        /// Value reported by the adapter.
        available: u32,
    },
    /// A required boolean feature is unavailable.
    MissingFeature(&'static str),
    /// Neither supported block nor adaptive compression is available.
    MissingCompression,
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CapabilityError {}

/// Stable ordering categories used when selecting a default adapter.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum AdapterClass {
    /// Software renderer.
    Software = 0,
    /// Adapter with no stronger classification.
    Other = 1,
    /// Integrated GPU.
    Integrated = 2,
    /// Discrete GPU.
    Discrete = 3,
}

/// Adapter identity and normalized capabilities used by admission and selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterInfo {
    backend: Backend,
    stable_id: [u8; 16],
    name: String,
    driver: String,
    class: AdapterClass,
    capabilities: AdapterCapabilities,
}

impl AdapterInfo {
    /// Stable IDs, names, and driver identities must be nonzero/nonempty before entering cache keys.
    ///
    /// # Errors
    ///
    /// Returns [`AdapterError::UnstableIdentity`] for an all-zero ID or
    /// [`AdapterError::InvalidText`] when the name or driver is empty.
    pub fn new(
        backend: Backend,
        stable_id: [u8; 16],
        name: impl Into<String>,
        driver: impl Into<String>,
        class: AdapterClass,
        capabilities: AdapterCapabilities,
    ) -> Result<Self, AdapterError> {
        let name = name.into();
        let driver = driver.into();
        if stable_id == [0; 16] {
            return Err(AdapterError::UnstableIdentity);
        }
        if name.is_empty() || driver.is_empty() {
            return Err(AdapterError::InvalidText);
        }

        Ok(Self {
            backend,
            stable_id,
            name,
            driver,
            class,
            capabilities,
        })
    }

    /// Backend which owns this adapter.
    pub const fn backend(&self) -> Backend {
        self.backend
    }
    /// Stable identity suitable for cache keys.
    pub const fn stable_id(&self) -> [u8; 16] {
        self.stable_id
    }
    /// Human-readable adapter name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Human-readable driver name.
    pub fn driver(&self) -> &str {
        &self.driver
    }
    /// Hardware classification used for deterministic ranking.
    pub const fn class(&self) -> AdapterClass {
        self.class
    }
    /// Capabilities reported by this adapter.
    pub const fn capabilities(&self) -> &AdapterCapabilities {
        &self.capabilities
    }
}

/// Failure constructing or selecting an adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterError {
    /// The supplied stable identity is all zeroes.
    UnstableIdentity,
    /// Adapter name or driver text is empty.
    InvalidText,
    /// No adapter passed admission and selection policy.
    NoAdmittedAdapter,
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for AdapterError {}

/// Ranking is class-first, then backend and stable identity; enumeration order never matters.
///
/// # Errors
///
/// Returns [`AdapterError::NoAdmittedAdapter`] when no candidate is admitted
/// (including when software adapters are excluded).
pub fn select_default_adapter(
    adapters: &[AdapterInfo],
    allow_software: bool,
) -> Result<&AdapterInfo, AdapterError> {
    adapters
        .iter()
        .filter(|adapter| allow_software || adapter.class != AdapterClass::Software)
        .filter(|adapter| SemanticProfile::V1.admit(&adapter.capabilities).is_ok())
        .max_by_key(|adapter| {
            (
                adapter.class,
                backend_rank(adapter.backend),
                reverse_id(adapter.stable_id),
            )
        })
        .ok_or(AdapterError::NoAdmittedAdapter)
}

fn backend_rank(backend: Backend) -> u8 {
    match backend {
        Backend::Vulkan => 3,
        Backend::Dx12 => 2,
        Backend::Metal => 1,
    }
}

fn reverse_id(mut id: [u8; 16]) -> [u8; 16] {
    for byte in &mut id {
        *byte = !*byte;
    }
    id
}

fn require_limit(
    errors: &mut Vec<CapabilityError>,
    name: &'static str,
    required: u32,
    available: u32,
) {
    if available < required {
        errors.push(CapabilityError::Limit {
            name,
            required,
            available,
        });
    }
}

fn require_feature(errors: &mut Vec<CapabilityError>, name: &'static str, available: bool) {
    if !available {
        errors.push(CapabilityError::MissingFeature(name));
    }
}
