use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

pub const SEMANTIC_ABI_VERSION: u32 = 1;
pub const MAX_SEMANTIC_NAME_BYTES: usize = 255;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Backend {
    Vulkan = 1,
    Dx12 = 2,
    Metal = 3,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SemanticId([u8; 16]);

impl SemanticId {
    /// Names use ASCII dot-separated identifiers; empty segments and names over 255 bytes fail.
    pub fn from_name(name: &str) -> Result<Self, SemanticError> {
        validate_name(name)?;
        let digest = blake3::hash(name.as_bytes());
        let mut id = [0; 16];
        id.copy_from_slice(&digest.as_bytes()[..16]);
        Ok(Self(id))
    }

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 16] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ResourceKind {
    ConstantBuffer = 1,
    StructuredBuffer = 2,
    SampledTexture = 3,
    StorageTexture = 4,
    Sampler = 5,
    RenderTarget = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ResourceAccess {
    Read = 1,
    Write = 2,
    ReadWrite = 3,
}

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

    pub const fn id(&self) -> SemanticId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn kind(&self) -> ResourceKind {
        self.kind
    }

    pub const fn access(&self) -> ResourceAccess {
        self.access
    }

    pub const fn array_count(&self) -> u32 {
        self.array_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticGraph {
    resources: Vec<SemanticResource>,
}

impl SemanticGraph {
    /// Duplicate names fail, while an improbable digest collision is reported separately and closed.
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

    pub fn resources(&self) -> &[SemanticResource] {
        &self.resources
    }

    /// Every canonical resource must occur exactly once; target-only bindings are rejected too.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalBinding {
    Descriptor { space: u32, binding: u32 },
    Argument { index: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetBinding {
    semantic_id: SemanticId,
    physical: PhysicalBinding,
}

impl TargetBinding {
    pub const fn descriptor(semantic_id: SemanticId, space: u32, binding: u32) -> Self {
        Self {
            semantic_id,
            physical: PhysicalBinding::Descriptor { space, binding },
        }
    }

    pub const fn argument(semantic_id: SemanticId, index: u32) -> Self {
        Self {
            semantic_id,
            physical: PhysicalBinding::Argument { index },
        }
    }

    pub const fn semantic_id(&self) -> SemanticId {
        self.semantic_id
    }

    pub const fn physical(self) -> PhysicalBinding {
        self.physical
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetLayout {
    backend: Backend,
    bindings: Vec<TargetBinding>,
}

impl TargetLayout {
    /// Semantic IDs may not repeat even when a target exposes multiple physical namespaces.
    pub fn new(backend: Backend, mut bindings: Vec<TargetBinding>) -> Result<Self, SemanticError> {
        bindings.sort_unstable_by_key(TargetBinding::semantic_id);
        for pair in bindings.windows(2) {
            if pair[0].semantic_id == pair[1].semantic_id {
                return Err(SemanticError::DuplicateTargetBinding(pair[0].semantic_id));
            }
        }

        Ok(Self { backend, bindings })
    }

    pub const fn backend(&self) -> Backend {
        self.backend
    }

    pub fn bindings(&self) -> &[TargetBinding] {
        &self.bindings
    }

    pub fn find(&self, id: SemanticId) -> Option<&TargetBinding> {
        self.bindings
            .binary_search_by_key(&id, TargetBinding::semantic_id)
            .ok()
            .map(|index| &self.bindings[index])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticError {
    InvalidName,
    ZeroArrayCount,
    InvalidAccess,
    DuplicateName,
    IdCollision(SemanticId),
    DuplicateTargetBinding(SemanticId),
    MissingTargetBinding(SemanticId),
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
