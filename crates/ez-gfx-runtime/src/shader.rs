use crate::binding::{BindingError, ReflectedBindings};
use ez_gfx_artifact::{Artifact, ArtifactError, Stage, Target};
use ez_gfx_core::{Backend, capability::SemanticProfile};

/// Identifies a selected shader by selection index, stage, and entry point.
pub type ShaderProduct<'a> = (usize, Stage, &'a str);
/// Pairs the selected vertex and fragment shaders for graphics pipeline creation.
pub type GraphicsPair<'a> = (ShaderProduct<'a>, ShaderProduct<'a>);

#[derive(Clone, Debug, Eq, PartialEq)]
/// Specifies an entry point and stage to select from a shader artifact.
pub struct ShaderRequest {
    /// Names the shader entry point to select.
    entry: String,
    /// Specifies the shader stage to select.
    stage: Stage,
}
impl ShaderRequest {
    /// Entry names are bounded UTF-8 without NUL because they enter native pipeline creation unchanged.
    ///
    /// # Errors
    ///
    /// Returns `ShaderLoadError::InvalidRequest` if the entry point is empty, exceeds 16 KiB, or contains a NUL byte.
    pub fn new(entry: impl Into<String>, stage: Stage) -> Result<Self, ShaderLoadError> {
        let entry = entry.into();
        if entry.is_empty() || entry.len() > 16 * 1024 || entry.as_bytes().contains(&0) {
            return Err(ShaderLoadError::InvalidRequest);
        }
        Ok(Self { entry, stage })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Holds a decoded artifact and the native shaders selected for one backend.
pub struct RuntimeShader {
    /// Contains the decoded shader artifact.
    artifact: Artifact,
    /// Records requested entry points, stages, and matching artifact indices.
    selected: Vec<(String, Stage, usize)>,
    /// Identifies the backend targeted by the selected native shaders.
    backend: Backend,
}

impl RuntimeShader {
    /// Decodes a bounded artifact and selects exact native products; no source, MSL, Slang, or JIT fallback exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the request list is empty, artifact decoding fails, a request is duplicated, or no artifact variant matches a request.
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

    /// Returns the artifact metadata used for shader reflection.
    pub fn metadata(&self) -> &[u8] {
        &self.artifact.metadata
    }
    /// Computes the artifact digest over execution-relevant contents.
    pub fn execution_digest(&self) -> [u8; 32] {
        self.artifact.execution_digest()
    }
    /// Returns the selected native shader bytes for an entry point and stage.
    pub fn product(&self, entry: &str, stage: Stage) -> Option<&[u8]> {
        self.selected
            .iter()
            .find(|(selected, selected_stage, _)| selected == entry && *selected_stage == stage)
            .map(|(_, _, index)| self.artifact.variants[*index].bytes.as_slice())
    }
    /// Iterates over selected stages and native shader bytes in request order.
    pub fn products(&self) -> impl ExactSizeIterator<Item = (Stage, &[u8])> {
        self.selected
            .iter()
            .map(|(_, stage, index)| (*stage, self.artifact.variants[*index].bytes.as_slice()))
    }
    /// Resolves the target-native physical layout for one selected stage; absent compiler reflection is rejected rather than treated as no bindings.
    ///
    /// # Errors
    ///
    /// Returns `BindingError::MissingReflection` if the shader was not selected, or an error from parsing its reflected bindings.
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
    ///
    /// # Errors
    ///
    /// Returns `ShaderLoadError::InvalidStagePairing` unless exactly one vertex shader and one fragment shader are selected.
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
    ///
    /// # Errors
    ///
    /// Returns `ShaderLoadError::InvalidStagePairing` unless exactly one compute shader is selected.
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

/// Maps a semantic profile to its artifact profile label.
fn profile_label(profile: SemanticProfile) -> &'static str {
    match profile {
        SemanticProfile::V1 => "ez-gfx-v1",
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Reports request validation, artifact decoding, and shader selection failures.
pub enum ShaderLoadError {
    /// The request list or an entry-point name is invalid.
    InvalidRequest,
    /// The same entry point and stage were requested more than once.
    DuplicateRequest,
    /// The shader artifact could not be decoded or validated.
    Artifact(ArtifactError),
    /// No compiled shader matches the requested target, profile, entry point, and stage.
    /// A required compiled product is absent.
    MissingProduct {
        /// Shader entry point.
        entry: String,
        /// Shader stage.
        stage: Stage,
    },
    /// The selected shaders do not form the required graphics or compute stage set.
    InvalidStagePairing,
}
