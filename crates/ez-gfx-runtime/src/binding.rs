use std::collections::{BTreeMap, BTreeSet};

use ez_gfx_artifact::Stage;
use ez_gfx_core::Backend;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingKind {
    Structured,
    Indirect,
    RenderTarget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResourceIdentity {
    Structured(u64),
    Indirect(u64),
    RenderTarget(u64),
}

impl ResourceIdentity {
    pub const fn kind(&self) -> BindingKind {
        match self {
            Self::Structured(_) => BindingKind::Structured,
            Self::Indirect(_) => BindingKind::Indirect,
            Self::RenderTarget(_) => BindingKind::RenderTarget,
        }
    }

    pub const fn handle(&self) -> u64 {
        match *self {
            Self::Structured(handle) | Self::Indirect(handle) | Self::RenderTarget(handle) => {
                handle
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicBinding {
    pub name: String,
    pub resource: ResourceIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingRequirement {
    pub name: String,
    pub kind: BindingKind,
    pub space: u32,
    pub binding: u32,
    pub descriptor_count: u32,
    pub writable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReflectedBindings {
    backend: Backend,
    requirements: Vec<BindingRequirement>,
}

impl ReflectedBindings {
    /// Metadata must contain exactly one reflection for the requested target/entry/stage; DXIL keeps SRV and UAV register namespaces distinct.
    pub fn parse(
        metadata: &[u8],
        backend: Backend,
        entry: &str,
        stage: Stage,
    ) -> Result<Self, BindingError> {
        if entry.is_empty() || entry.len() > 1024 || entry.as_bytes().contains(&0) {
            return Err(BindingError::InvalidMetadata);
        }
        let envelope: MetadataEnvelope =
            serde_json::from_slice(metadata).map_err(|_| BindingError::InvalidMetadata)?;
        let target = match backend {
            Backend::Vulkan => "Spirv",
            Backend::Dx12 => "Dxil",
            Backend::Metal => "Metallib",
        };
        let stage = stage_name(stage);
        let mut matches = envelope.reflections.into_iter().filter(|reflection| {
            reflection.target == target && reflection.entry == entry && reflection.stage == stage
        });
        let reflection = matches.next().ok_or(BindingError::MissingReflection)?;
        if matches.next().is_some() {
            return Err(BindingError::AmbiguousReflection);
        }

        let mut names = BTreeSet::new();
        let mut slots = BTreeSet::new();
        let mut requirements = Vec::with_capacity(reflection.reflection.parameters.len());
        for parameter in reflection.reflection.parameters {
            let Some(name) = parameter.semantic_name else {
                continue;
            };
            let Some(kind) = parameter.api_kind.and_then(parse_kind) else {
                continue;
            };
            if name.is_empty() || name.len() > 255 || name.as_bytes().contains(&0) {
                return Err(BindingError::InvalidMetadata);
            }
            if !names.insert(name.clone()) {
                return Err(BindingError::Duplicate(name));
            }
            if parameter.descriptor_count == 0 || parameter.descriptor_count > 2 {
                return Err(BindingError::InvalidMetadata);
            }
            let writable = parameter.resource_access != "Read";
            let namespace = if backend == Backend::Dx12 {
                u8::from(writable)
            } else {
                0
            };
            for binding in parameter.binding_index
                ..parameter
                    .binding_index
                    .checked_add(parameter.descriptor_count)
                    .ok_or(BindingError::InvalidMetadata)?
            {
                if !slots.insert((parameter.binding_space, binding, namespace)) {
                    return Err(BindingError::DuplicatePhysicalSlot {
                        space: parameter.binding_space,
                        binding,
                    });
                }
            }
            requirements.push(BindingRequirement {
                name,
                kind,
                space: parameter.binding_space,
                binding: parameter.binding_index,
                descriptor_count: parameter.descriptor_count,
                writable,
            });
        }
        requirements.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        Ok(Self {
            backend,
            requirements,
        })
    }

    pub fn requirements(&self) -> &[BindingRequirement] {
        &self.requirements
    }

    /// Public bindings are order-independent, but must match every reflected named resource exactly once and with the correct opaque-handle kind.
    pub fn validate(&self, bindings: &[PublicBinding]) -> Result<(), BindingError> {
        let expected = self
            .requirements
            .iter()
            .map(|requirement| (requirement.name.as_str(), requirement.kind))
            .collect::<BTreeMap<_, _>>();
        let mut actual = BTreeMap::new();
        for binding in bindings {
            if actual
                .insert(binding.name.as_str(), binding.resource.kind())
                .is_some()
            {
                return Err(BindingError::Duplicate(binding.name.clone()));
            }
        }
        for (&name, &kind) in &expected {
            match actual.get(name) {
                None => return Err(BindingError::Missing(name.into())),
                Some(actual_kind) if *actual_kind != kind => {
                    return Err(BindingError::KindMismatch(name.into()));
                }
                Some(_) => {}
            }
        }
        if let Some(name) = actual.keys().find(|name| !expected.contains_key(**name)) {
            return Err(BindingError::Unknown((*name).into()));
        }
        Ok(())
    }

    /// Stage layouts merge only when repeated semantic names preserve kind and physical location.
    pub fn merge(&self, other: &Self) -> Result<Self, BindingError> {
        if self.backend != other.backend {
            return Err(BindingError::InvalidMetadata);
        }
        let mut requirements = self.requirements.clone();
        for requirement in &other.requirements {
            if let Some(existing) = requirements
                .iter()
                .find(|existing| existing.name == requirement.name)
            {
                if existing != requirement {
                    return Err(BindingError::ConflictingStageBinding(
                        requirement.name.clone(),
                    ));
                }
            } else if requirements.iter().any(|existing| {
                existing.space == requirement.space
                    && existing.binding == requirement.binding
                    && (self.backend != Backend::Dx12 || existing.writable == requirement.writable)
            }) {
                return Err(BindingError::DuplicatePhysicalSlot {
                    space: requirement.space,
                    binding: requirement.binding,
                });
            } else {
                requirements.push(requirement.clone());
            }
        }
        requirements.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        Ok(Self {
            backend: self.backend,
            requirements,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BindingError {
    InvalidMetadata,
    MissingReflection,
    AmbiguousReflection,
    Duplicate(String),
    DuplicatePhysicalSlot { space: u32, binding: u32 },
    ConflictingStageBinding(String),
    Missing(String),
    Unknown(String),
    KindMismatch(String),
}

#[derive(Deserialize)]
struct MetadataEnvelope {
    #[serde(default)]
    reflections: Vec<TargetReflection>,
}

#[derive(Deserialize)]
struct TargetReflection {
    target: String,
    entry: String,
    stage: String,
    reflection: Reflection,
}

#[derive(Deserialize)]
struct Reflection {
    #[serde(default)]
    parameters: Vec<Parameter>,
}

#[derive(Deserialize)]
struct Parameter {
    #[serde(default)]
    semantic_name: Option<String>,
    #[serde(default)]
    api_kind: Option<String>,
    binding_index: u32,
    #[serde(default = "one")]
    descriptor_count: u32,
    #[serde(default)]
    resource_access: String,
    binding_space: u32,
}
const fn one() -> u32 {
    1
}

fn parse_kind(value: String) -> Option<BindingKind> {
    match value.as_str() {
        "structured" => Some(BindingKind::Structured),
        "indirect" => Some(BindingKind::Indirect),
        "render_target" => Some(BindingKind::RenderTarget),
        _ => None,
    }
}

const fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "Vertex",
        Stage::Fragment => "Fragment",
        Stage::Compute => "Compute",
        Stage::Geometry => "Geometry",
        Stage::TessellationControl => "TessellationControl",
        Stage::TessellationEvaluation => "TessellationEvaluation",
    }
}
