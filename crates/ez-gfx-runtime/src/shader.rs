use crate::binding::{BindingError, ReflectedBindings};
use ez_gfx_artifact::{Artifact, ArtifactError, Stage, Target};
use ez_gfx_core::{Backend, capability::SemanticProfile};

pub type ShaderProduct<'a> = (usize, Stage, &'a str);
pub type GraphicsPair<'a> = (ShaderProduct<'a>, ShaderProduct<'a>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderRequest {
    entry: String,
    stage: Stage,
}
impl ShaderRequest {
    /// Entry names are bounded UTF-8 without NUL because they enter native pipeline creation unchanged.
    pub fn new(entry: impl Into<String>, stage: Stage) -> Result<Self, ShaderLoadError> {
        let entry = entry.into();
        if entry.is_empty() || entry.len() > 16 * 1024 || entry.as_bytes().contains(&0) {
            return Err(ShaderLoadError::InvalidRequest);
        }
        Ok(Self { entry, stage })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeShader {
    artifact: Artifact,
    selected: Vec<(String, Stage, usize)>,
    backend: Backend,
}

impl RuntimeShader {
    /// Decodes a bounded artifact and selects exact native products; no source, MSL, Slang, or JIT fallback exists.
    pub fn load(
        bytes: &[u8],
        backend: Backend,
        profile: SemanticProfile,
        requests: &[ShaderRequest],
    ) -> Result<Self, ShaderLoadError> {
        if requests.is_empty() {
            return Err(ShaderLoadError::InvalidRequest);
        }
        let artifact = Artifact::decode(bytes).map_err(ShaderLoadError::Artifact)?;
        let target = match backend {
            Backend::Vulkan => Target::Spirv,
            Backend::Dx12 => Target::Dxil,
            Backend::Metal => Target::Metallib,
        };
        let profile = profile_label(profile);
        let mut selected = Vec::with_capacity(requests.len());
        for request in requests {
            if selected
                .iter()
                .any(|(entry, stage, _)| entry == &request.entry && *stage == request.stage)
            {
                return Err(ShaderLoadError::DuplicateRequest);
            }
            let index = artifact
                .variants
                .iter()
                .position(|variant| {
                    variant.target == target
                        && variant.stage == request.stage
                        && variant.entry_point == request.entry
                        && variant.profile == profile
                })
                .ok_or_else(|| ShaderLoadError::MissingProduct {
                    entry: request.entry.clone(),
                    stage: request.stage,
                })?;
            selected.push((request.entry.clone(), request.stage, index));
        }
        Ok(Self {
            artifact,
            selected,
            backend,
        })
    }

    pub fn metadata(&self) -> &[u8] {
        &self.artifact.metadata
    }
    pub fn execution_digest(&self) -> [u8; 32] {
        self.artifact.execution_digest()
    }
    pub fn product(&self, entry: &str, stage: Stage) -> Option<&[u8]> {
        self.selected
            .iter()
            .find(|(selected, selected_stage, _)| selected == entry && *selected_stage == stage)
            .map(|(_, _, index)| self.artifact.variants[*index].bytes.as_slice())
    }
    pub fn products(&self) -> impl ExactSizeIterator<Item = (Stage, &[u8])> {
        self.selected
            .iter()
            .map(|(_, stage, index)| (*stage, self.artifact.variants[*index].bytes.as_slice()))
    }
    /// Resolves the target-native physical layout for one selected stage; absent compiler reflection is rejected rather than treated as no bindings.
    pub fn bindings(&self, entry: &str, stage: Stage) -> Result<ReflectedBindings, BindingError> {
        if !self
            .selected
            .iter()
            .any(|(selected, selected_stage, _)| selected == entry && *selected_stage == stage)
        {
            return Err(BindingError::MissingReflection);
        }
        ReflectedBindings::parse(&self.artifact.metadata, self.backend, entry, stage)
    }

    /// Graphics pipelines require exactly one selected vertex product and one selected fragment product.
    pub fn graphics_pair(&self) -> Result<GraphicsPair<'_>, ShaderLoadError> {
        let mut vertex = None;
        let mut fragment = None;
        for (product_index, (entry, stage, _)) in self.selected.iter().enumerate() {
            match stage {
                Stage::Vertex
                    if vertex
                        .replace((product_index, *stage, entry.as_str()))
                        .is_some() =>
                {
                    return Err(ShaderLoadError::InvalidStagePairing);
                }
                Stage::Fragment
                    if fragment
                        .replace((product_index, *stage, entry.as_str()))
                        .is_some() =>
                {
                    return Err(ShaderLoadError::InvalidStagePairing);
                }
                _ => {}
            }
        }
        vertex
            .zip(fragment)
            .ok_or(ShaderLoadError::InvalidStagePairing)
    }

    /// Compute lookup accepts mixed artifacts but still requires exactly one selected compute product.
    pub fn compute_product(&self) -> Result<(usize, Stage, &str), ShaderLoadError> {
        let mut compute = None;
        for (product_index, (entry, stage, _)) in self.selected.iter().enumerate() {
            if *stage != Stage::Compute {
                continue;
            }
            if compute
                .replace((product_index, *stage, entry.as_str()))
                .is_some()
            {
                return Err(ShaderLoadError::InvalidStagePairing);
            }
        }
        compute.ok_or(ShaderLoadError::InvalidStagePairing)
    }
}

fn profile_label(profile: SemanticProfile) -> &'static str {
    match profile {
        SemanticProfile::V1 => "ez-gfx-v1",
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderLoadError {
    InvalidRequest,
    DuplicateRequest,
    Artifact(ArtifactError),
    MissingProduct { entry: String, stage: Stage },
    InvalidStagePairing,
}
