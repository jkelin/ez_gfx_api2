use std::collections::{BTreeMap, BTreeSet};

use ez_gfx_artifact::Stage;
use ez_gfx_core::{
    Backend,
    capability::MAX_BINDLESS_SAMPLED_TEXTURES,
    handle::{IndirectBufferHandle, RenderTargetHandle, StructuredBufferHandle},
};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Classifies the GPU resource represented by a binding.
pub enum BindingKind {
    /// A structured buffer resource.
    Structured,
    /// A buffer used for indirect GPU commands.
    Indirect,
    /// A render-target resource.
    RenderTarget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Associates a typed runtime handle with its GPU resource kind.
pub enum ResourceIdentity {
    /// A structured buffer handle.
    Structured(StructuredBufferHandle),
    /// An indirect-command buffer handle.
    Indirect(IndirectBufferHandle),
    /// A render-target handle.
    RenderTarget(RenderTargetHandle),
}

impl ResourceIdentity {
    /// Returns the GPU resource kind associated with this identity.
    pub const fn kind(&self) -> BindingKind {
        match self {
            Self::Structured(_) => BindingKind::Structured,
            Self::Indirect(_) => BindingKind::Indirect,
            Self::RenderTarget(_) => BindingKind::RenderTarget,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Names a GPU resource supplied for shader binding.
pub struct PublicBinding {
    /// Shader-visible semantic name.
    pub name: String,
    /// Opaque identity of the supplied GPU resource.
    pub resource: ResourceIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Describes one named resource required by shader reflection.
pub struct BindingRequirement {
    /// Shader-visible semantic name.
    pub name: String,
    /// Required GPU resource kind.
    pub kind: BindingKind,
    /// Descriptor space containing the resource.
    pub space: u32,
    /// First descriptor index occupied by the resource.
    pub binding: u32,
    /// Number of consecutive descriptors occupied by the resource.
    pub descriptor_count: u32,
    /// Whether the shader may write to the resource.
    pub writable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Stores validated named resource requirements for one backend.
pub struct ReflectedBindings {
    /// Graphics API targeted by the reflected metadata.
    backend: Backend,
    /// Named resource requirements sorted by semantic name.
    requirements: Vec<BindingRequirement>,
}

/// Maximum number of sampled textures supported by the bindless heap.
pub const MAX_TEXTURE_HEAP_CAPACITY: u32 = MAX_BINDLESS_SAMPLED_TEXTURES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes the reflected bindless texture heap and argument encoding.
pub struct TextureHeapLayout {
    /// Descriptor space containing the texture heap.
    pub space: u32,
    /// Descriptor index of the texture heap.
    pub binding: u32,
    /// Maximum number of sampled textures addressable by the heap.
    pub capacity: u32,
    /// Byte stride between encoded texture arguments.
    pub argument_stride: u32,
    /// Byte offset of the texture reference within each argument.
    pub texture_argument_offset: u32,
    /// Byte offset of the sampler reference within each argument.
    pub sampler_argument_offset: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Captures reflected texture-heap and depth-attachment requirements.
pub struct PipelineLayout {
    /// Bindless texture-heap layout when declared by the shader.
    texture_heap: Option<TextureHeapLayout>,
    /// Whether rendering requires a depth attachment.
    depth_required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Carries one fully parsed and semantically validated stage reflection.
pub struct ValidatedStageReflection {
    stage: Stage,
    bindings: ReflectedBindings,
    pipeline_layout: PipelineLayout,
}

impl ValidatedStageReflection {
    /// Returns the reflected stage.
    pub const fn stage(&self) -> Stage {
        self.stage
    }

    /// Returns validated named resource bindings.
    pub const fn bindings(&self) -> &ReflectedBindings {
        &self.bindings
    }

    /// Returns the validated pipeline layout.
    pub const fn pipeline_layout(&self) -> &PipelineLayout {
        &self.pipeline_layout
    }
}

impl PipelineLayout {
    /// Parses and validates the pipeline layout for a backend entry point and stage.
    ///
    /// # Errors
    ///
    /// Returns an error if the entry point is invalid, matching reflection is missing or ambiguous, metadata cannot be parsed, or the reflected texture-heap layout is invalid.
    pub fn parse(
        metadata: &[u8],
        backend: Backend,
        entry: &str,
        stage: Stage,
    ) -> Result<Self, BindingError> {
        let reflection = matching_reflection(metadata, backend, entry, stage)?;
        Self::from_reflection(&reflection)
    }

    fn from_reflection(reflection: &TargetReflection) -> Result<Self, BindingError> {
        let texture_heap =
            reflection
                .reflection
                .texture_heap
                .as_ref()
                .map(|heap| TextureHeapLayout {
                    space: heap.binding_space,
                    binding: heap.binding_index,
                    capacity: heap.capacity,
                    argument_stride: heap.argument_stride,
                    texture_argument_offset: heap.texture_argument_offset,
                    sampler_argument_offset: heap.sampler_argument_offset,
                });
        if let Some(heap) = texture_heap
            && (heap.capacity == 0
                || heap.capacity > MAX_TEXTURE_HEAP_CAPACITY
                || heap.argument_stride == 0
                || heap.texture_argument_offset >= heap.argument_stride
                || heap.sampler_argument_offset >= heap.argument_stride
                || heap.texture_argument_offset == heap.sampler_argument_offset)
        {
            return Err(BindingError::InvalidMetadata);
        }
        Ok(Self {
            texture_heap,
            depth_required: reflection.reflection.depth_required,
        })
    }

    /// Returns the bindless texture-heap layout when present.
    pub const fn texture_heap(&self) -> Option<&TextureHeapLayout> {
        self.texture_heap.as_ref()
    }

    /// Reports whether the pipeline requires a depth attachment.
    pub const fn depth_required(&self) -> bool {
        self.depth_required
    }
}

impl ReflectedBindings {
    /// Metadata must contain exactly one reflection for the requested target/entry/stage; DXIL keeps SRV and UAV register namespaces distinct.
    ///
    /// # Errors
    ///
    /// Returns an error if reflection metadata is invalid, missing, or ambiguous, or if named requirements contain duplicate names, invalid descriptor counts or ranges, or overlapping physical slots.
    pub fn parse(
        metadata: &[u8],
        backend: Backend,
        entry: &str,
        stage: Stage,
    ) -> Result<Self, BindingError> {
        let reflection = matching_reflection(metadata, backend, entry, stage)?;
        Self::from_reflection(&reflection, backend)
    }

    fn from_reflection(
        reflection: &TargetReflection,
        backend: Backend,
    ) -> Result<Self, BindingError> {
        let mut names = BTreeSet::new();
        let mut slots = BTreeSet::new();
        let mut requirements = Vec::with_capacity(reflection.reflection.parameters.len());
        for parameter in &reflection.reflection.parameters {
            let classified = match (&parameter.semantic_name, &parameter.api_kind) {
                (None, None) => continue,
                (Some(name), Some(kind)) => (name, kind),
                _ => return Err(BindingError::InvalidMetadata),
            };
            let (name, kind) = classified;
            let Some(kind) = parse_kind(kind) else {
                return Err(BindingError::InvalidMetadata);
            };
            if name.is_empty() || name.len() > 255 || name.as_bytes().contains(&0) {
                return Err(BindingError::InvalidMetadata);
            }
            if !names.insert(name.clone()) {
                return Err(BindingError::Duplicate(name.clone()));
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
                name: name.clone(),
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

    /// Returns reflected resource requirements sorted by semantic name.
    pub fn requirements(&self) -> &[BindingRequirement] {
        &self.requirements
    }

    /// Public bindings are order-independent, but must match every reflected named resource exactly once and with the correct opaque-handle kind.
    ///
    /// # Errors
    ///
    /// Returns an error if supplied bindings contain duplicate or unknown names, omit a required name, or use the wrong resource kind.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the backends differ, a repeated name has conflicting requirements, or distinct requirements occupy the same physical slot.
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

/// Parses metadata once and validates exactly one reflection for every selected stage.
///
/// # Errors
///
/// Returns an error for malformed metadata, absent or ambiguous stage reflection, invalid
/// bindings, invalid texture heaps, or conflicting graphics-stage resource layouts.
pub fn validate_stage_reflections(
    metadata: &[u8],
    backend: Backend,
    stages: &[(Stage, &str)],
) -> Result<Vec<ValidatedStageReflection>, BindingError> {
    let envelope: MetadataEnvelope =
        serde_json::from_slice(metadata).map_err(|_| BindingError::InvalidMetadata)?;
    let target = target_name(backend);
    let mut validated = Vec::with_capacity(stages.len());
    for &(stage, entry) in stages {
        if entry.is_empty() || entry.len() > 1024 || entry.as_bytes().contains(&0) {
            return Err(BindingError::InvalidMetadata);
        }
        let stage_name = stage_name(stage);
        let mut matches = envelope.reflections.iter().filter(|reflection| {
            reflection.target == target
                && reflection.entry == entry
                && reflection.stage == stage_name
        });
        let reflection = matches.next().ok_or(BindingError::MissingReflection)?;
        if matches.next().is_some() {
            return Err(BindingError::AmbiguousReflection);
        }
        validated.push(ValidatedStageReflection {
            stage,
            bindings: ReflectedBindings::from_reflection(reflection, backend)?,
            pipeline_layout: PipelineLayout::from_reflection(reflection)?,
        });
    }

    let vertex = validated.iter().find(|value| value.stage == Stage::Vertex);
    let fragment = validated
        .iter()
        .find(|value| value.stage == Stage::Fragment);
    if let (Some(vertex), Some(fragment)) = (vertex, fragment) {
        vertex.bindings.merge(&fragment.bindings)?;
    }
    Ok(validated)
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Runtime binding validation errors.
pub enum BindingError {
    /// Reflection metadata is invalid.
    InvalidMetadata,
    /// Shader reflection is absent.
    MissingReflection,
    /// Reflection data is ambiguous.
    AmbiguousReflection,
    /// A logical binding is duplicated.
    Duplicate(String),
    /// A physical heap slot is duplicated.
    DuplicatePhysicalSlot {
        /// Descriptor space.
        space: u32,
        /// Descriptor binding.
        binding: u32,
    },
    /// A semantic name has incompatible requirements across shader stages.
    ConflictingStageBinding(String),
    /// A semantic name required by reflection was not supplied.
    Missing(String),
    /// A supplied semantic name is not present in reflection.
    Unknown(String),
    /// A supplied resource has the wrong GPU resource kind.
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
    #[serde(default)]
    texture_heap: Option<TextureHeapMetadata>,
    #[serde(default)]
    depth_required: bool,
}

#[derive(Deserialize)]
struct TextureHeapMetadata {
    binding_space: u32,
    binding_index: u32,
    capacity: u32,
    argument_stride: u32,
    texture_argument_offset: u32,
    sampler_argument_offset: u32,
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

/// Maps a reflected API resource tag to its supported binding kind.
fn parse_kind(value: &str) -> Option<BindingKind> {
    match value {
        "structured" => Some(BindingKind::Structured),
        "indirect" => Some(BindingKind::Indirect),
        "render_target" => Some(BindingKind::RenderTarget),
        _ => None,
    }
}

/// Finds the unique reflection matching the backend, entry point, and shader stage.
///
/// # Errors
///
/// Returns an error if the entry point is invalid, the metadata is not valid JSON, or matching reflection is missing or ambiguous.
fn matching_reflection(
    metadata: &[u8],
    backend: Backend,
    entry: &str,
    stage: Stage,
) -> Result<TargetReflection, BindingError> {
    if entry.is_empty() || entry.len() > 1024 || entry.as_bytes().contains(&0) {
        return Err(BindingError::InvalidMetadata);
    }
    let envelope: MetadataEnvelope =
        serde_json::from_slice(metadata).map_err(|_| BindingError::InvalidMetadata)?;
    let target = target_name(backend);
    let stage = stage_name(stage);
    let mut matches = envelope.reflections.into_iter().filter(|reflection| {
        reflection.target == target && reflection.entry == entry && reflection.stage == stage
    });
    let reflection = matches.next().ok_or(BindingError::MissingReflection)?;
    if matches.next().is_some() {
        return Err(BindingError::AmbiguousReflection);
    }
    Ok(reflection)
}

const fn target_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Vulkan => "Spirv",
        Backend::Dx12 => "Dxil",
        Backend::Metal => "Metallib",
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
