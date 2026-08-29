use core::fmt;

use crate::Backend;

pub const CAPABILITY_PROFILE_SCHEMA_VERSION: u32 = 1;
pub const MAX_BINDLESS_SAMPLED_TEXTURES: u32 = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct CompressionSupport(u8);

impl CompressionSupport {
    pub const NONE: Self = Self(0);
    pub const BC: Self = Self(1 << 0);
    pub const ASTC: Self = Self(1 << 1);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub const fn bits(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterCapabilities {
    pub bindless_sampled_textures: u32,
    pub bindless_storage_resources: u32,
    pub bindless_samplers: u32,
    pub max_indirect_draw_count: u32,
    /// Backend-native shader capability normalized as `(major << 8) | minor`.
    pub shader_model: u32,
    pub timeline_synchronization: bool,
    pub resource_aliasing: bool,
    pub dynamic_rendering: bool,
    pub presentation: bool,
    pub compression: CompressionSupport,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticProfile {
    V1,
}

impl SemanticProfile {
    pub const fn schema_version(self) -> u32 {
        CAPABILITY_PROFILE_SCHEMA_VERSION
    }

    /// Admission accumulates every missing requirement for one deterministic diagnosis.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityError {
    Limit {
        name: &'static str,
        required: u32,
        available: u32,
    },
    MissingFeature(&'static str),
    MissingCompression,
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CapabilityError {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum AdapterClass {
    Software = 0,
    Other = 1,
    Integrated = 2,
    Discrete = 3,
}

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

    pub const fn backend(&self) -> Backend {
        self.backend
    }
    pub const fn stable_id(&self) -> [u8; 16] {
        self.stable_id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn driver(&self) -> &str {
        &self.driver
    }
    pub const fn class(&self) -> AdapterClass {
        self.class
    }
    pub const fn capabilities(&self) -> &AdapterCapabilities {
        &self.capabilities
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterError {
    UnstableIdentity,
    InvalidText,
    NoAdmittedAdapter,
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for AdapterError {}

/// Ranking is class-first, then backend and stable identity; enumeration order never matters.
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
