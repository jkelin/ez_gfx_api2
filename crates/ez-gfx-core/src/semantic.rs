use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

/// ABI version for canonical semantic identifiers and layouts.
pub const SEMANTIC_ABI_VERSION: u32 = 1;
/// Maximum UTF-8 byte length of a semantic resource name.
pub const MAX_SEMANTIC_NAME_BYTES: usize = 255;

/// Graphics backends supported by the canonical semantic model.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Backend {
    /// Vulkan resource namespace.
    Vulkan = 1,
    /// Direct3D 12 resource namespace.
    Dx12 = 2,
    /// Metal resource namespace.
    Metal = 3,
}

/// Stable 128-bit identifier derived from a validated semantic name.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SemanticId([u8; 16]);

impl SemanticId {
    /// Names use ASCII dot-separated identifiers; empty segments and names over 255 bytes fail.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticError::InvalidName`] when the name violates the
    /// canonical ASCII dot-separated format or length limit.
    pub fn from_name(name: &str) -> Result<Self, SemanticError> {
        validate_name(name)?;
        let digest = blake3::hash(name.as_bytes());
        let mut id = [0; 16];
        id.copy_from_slice(&digest.as_bytes()[..16]);
        Ok(Self(id))
    }

    /// Creates an identifier from its canonical 16-byte representation.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Returns the identifier bytes.
    pub const fn bytes(self) -> [u8; 16] {
        self.0
    }
}

/// Resource categories understood by the semantic graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ResourceKind {
    /// Constant buffer resource.
    ConstantBuffer = 1,
    /// Structured buffer resource.
    StructuredBuffer = 2,
    /// Read-only sampled texture.
    SampledTexture = 3,
    /// Writable storage texture.
    StorageTexture = 4,
    /// Sampler state.
    Sampler = 5,
    /// Render target texture.
    RenderTarget = 6,
}

/// Access mode required by a semantic resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ResourceAccess {
    /// Read-only access.
    Read = 1,
    /// Write-only access.
    Write = 2,
    /// Read and write access.
    ReadWrite = 3,
}

/// Canonical resource declaration used by a semantic graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticResource {
    id: SemanticId,
    name: String,
    kind: ResourceKind,
    access: ResourceAccess,
    array_count: u32,
}

impl SemanticResource {
    /// Zero-length arrays and writable samplers are invalid at the canonical boundary.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticError::InvalidName`], [`SemanticError::ZeroArrayCount`],
    /// or [`SemanticError::InvalidAccess`] for invalid resource declarations.
    pub fn new(
        name: impl Into<String>,
        kind: ResourceKind,
        access: ResourceAccess,
        array_count: u32,
    ) -> Result<Self, SemanticError> {
        let name = name.into();
        let id = SemanticId::from_name(&name)?;
        if array_count == 0 {
            return Err(SemanticError::ZeroArrayCount);
        }
        if kind == ResourceKind::Sampler && access != ResourceAccess::Read {
            return Err(SemanticError::InvalidAccess);
        }

        Ok(Self {
            id,
            name,
            kind,
            access,
            array_count,
        })
    }

    /// Returns the stable semantic identifier.
    pub const fn id(&self) -> SemanticId {
        self.id
    }

    /// Returns the canonical resource name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the resource category.
    pub const fn kind(&self) -> ResourceKind {
        self.kind
    }

    /// Returns the required access mode.
    pub const fn access(&self) -> ResourceAccess {
        self.access
    }

    /// Returns the number of array elements.
    pub const fn array_count(&self) -> u32 {
        self.array_count
    }
}

/// Sorted, collision-checked collection of canonical resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticGraph {
    resources: Vec<SemanticResource>,
}

impl SemanticGraph {
    /// Duplicate names fail, while an improbable digest collision is reported separately and closed.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticError::DuplicateName`] for repeated names or
    /// [`SemanticError::IdCollision`] for distinct names sharing an ID.
    pub fn new(mut resources: Vec<SemanticResource>) -> Result<Self, SemanticError> {
        resources.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        let mut names = BTreeSet::new();
        let mut ids = BTreeMap::new();
        for resource in &resources {
            if !names.insert(resource.name.clone()) {
                return Err(SemanticError::DuplicateName);
            }
            if let Some(existing) = ids.insert(resource.id, resource.name.clone())
                && existing != resource.name
            {
                return Err(SemanticError::IdCollision(resource.id));
            }
        }

        Ok(Self { resources })
    }

    /// Returns resources in canonical name order.
    pub fn resources(&self) -> &[SemanticResource] {
        &self.resources
    }

    /// Every canonical resource must occur exactly once; target-only bindings are rejected too.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticError::MissingTargetBinding`] or
    /// [`SemanticError::UnknownTargetBinding`] when the layout differs from the graph.
    pub fn validate_layout(&self, layout: &TargetLayout) -> Result<(), SemanticError> {
        let expected: BTreeSet<_> = self.resources.iter().map(SemanticResource::id).collect();
        let actual: BTreeSet<_> = layout
            .bindings
            .iter()
            .map(TargetBinding::semantic_id)
            .collect();
        if let Some(missing) = expected.difference(&actual).next() {
            return Err(SemanticError::MissingTargetBinding(*missing));
        }
        if let Some(extra) = actual.difference(&expected).next() {
            return Err(SemanticError::UnknownTargetBinding(*extra));
        }

        Ok(())
    }
}

/// Backend-specific physical location of a semantic resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalBinding {
    /// Descriptor-set/space and binding location.
    Descriptor {
        /// Descriptor set or register space.
        space: u32,
        /// Binding number within the space.
        /// Binding number within the descriptor space.
        binding: u32,
    },
    /// Argument-buffer index.
    Argument {
        /// Argument-buffer index.
        index: u32,
    },
}

/// Maps one semantic identifier to a backend physical location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetBinding {
    semantic_id: SemanticId,
    physical: PhysicalBinding,
}

impl TargetBinding {
    /// Creates a descriptor binding.
    pub const fn descriptor(semantic_id: SemanticId, space: u32, binding: u32) -> Self {
        Self {
            semantic_id,
            physical: PhysicalBinding::Descriptor { space, binding },
        }
    }

    /// Creates an argument-buffer binding.
    pub const fn argument(semantic_id: SemanticId, index: u32) -> Self {
        Self {
            semantic_id,
            physical: PhysicalBinding::Argument { index },
        }
    }

    /// Returns the semantic identifier.
    pub const fn semantic_id(&self) -> SemanticId {
        self.semantic_id
    }

    /// Returns the physical backend location.
    pub const fn physical(self) -> PhysicalBinding {
        self.physical
    }
}

/// Complete backend binding layout for a shader interface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetLayout {
    backend: Backend,
    bindings: Vec<TargetBinding>,
}

impl TargetLayout {
    /// Semantic IDs may not repeat even when a target exposes multiple physical namespaces.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticError::DuplicateTargetBinding`] when an identifier repeats.
    pub fn new(backend: Backend, mut bindings: Vec<TargetBinding>) -> Result<Self, SemanticError> {
        bindings.sort_unstable_by_key(TargetBinding::semantic_id);
        for pair in bindings.windows(2) {
            if pair[0].semantic_id == pair[1].semantic_id {
                return Err(SemanticError::DuplicateTargetBinding(pair[0].semantic_id));
            }
        }

        Ok(Self { backend, bindings })
    }
    /// Returns the backend represented by this layout.
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns bindings in canonical identifier order.
    pub fn bindings(&self) -> &[TargetBinding] {
        &self.bindings
    }

    /// Finds a binding by semantic identifier.
    pub fn find(&self, id: SemanticId) -> Option<&TargetBinding> {
        self.bindings
            .binary_search_by_key(&id, TargetBinding::semantic_id)
            .ok()
            .map(|index| &self.bindings[index])
    }
}

/// Validation failures for semantic names, resources, and layouts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticError {
    /// The resource name is empty, too long, or not dot-separated ASCII.
    InvalidName,
    /// An array declaration has zero elements.
    ZeroArrayCount,
    /// A sampler requests writable access.
    InvalidAccess,
    /// Two resources have the same canonical name.
    DuplicateName,
    /// Two distinct names produced the same identifier.
    IdCollision(SemanticId),
    /// A layout repeats one semantic identifier.
    DuplicateTargetBinding(SemanticId),
    /// A graph resource is absent from the layout.
    MissingTargetBinding(SemanticId),
    /// A layout contains an unknown resource.
    UnknownTargetBinding(SemanticId),
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for SemanticError {}

fn validate_name(name: &str) -> Result<(), SemanticError> {
    if name.is_empty() || name.len() > MAX_SEMANTIC_NAME_BYTES {
        return Err(SemanticError::InvalidName);
    }

    let valid = name.split('.').all(|part| {
        !part.is_empty()
            && part.as_bytes()[0].is_ascii_alphabetic()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    });
    if !valid {
        return Err(SemanticError::InvalidName);
    }

    Ok(())
}
