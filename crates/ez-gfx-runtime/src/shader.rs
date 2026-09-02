use crate::binding::{
    BindingError, PipelineLayout, ReflectedBindings, ValidatedStageReflection,
    validate_stage_reflections,
};
use ez_gfx_artifact::{
    AppleArchitecture, ApplePlatform, Artifact, ArtifactError, CompatibilityVersion,
    MetalCompatibility, Stage, Target, TargetCompatibility,
};
use ez_gfx_core::{Backend, capability::SemanticProfile};
use std::collections::BTreeSet;

/// Identifies a selected shader by selection index, stage, and its artifact-owned entry point.
pub type ShaderProduct<'a> = (usize, Stage, &'a str);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes the Metal runtime compatibility boundary used for metallib admission.
pub struct MetalEnvironment {
    /// Running Apple platform.
    pub platform: ApplePlatform,
    /// Running CPU architecture.
    pub architecture: AppleArchitecture,
    /// Running operating-system version.
    pub os: CompatibilityVersion,
    /// Highest Metal language version accepted by this runtime.
    pub max_language: CompatibilityVersion,
    /// Highest metallib contract version accepted by this runtime.
    pub max_library: CompatibilityVersion,
}

impl MetalEnvironment {
    fn admits(self, compatibility: &MetalCompatibility) -> bool {
        compatibility.platform == self.platform
            && compatibility.architecture == self.architecture
            && compatibility.minimum_os <= self.os
            && compatibility.sdk >= compatibility.minimum_os
            && compatibility.language <= self.max_language
            && compatibility.library <= self.max_library
    }
}
/// Pairs the selected vertex and fragment shaders for graphics pipeline creation.
pub type GraphicsPair<'a> = (ShaderProduct<'a>, ShaderProduct<'a>);

#[derive(Clone, Debug, Eq, PartialEq)]
/// Holds a decoded artifact and the native shaders selected for one backend.
pub struct RuntimeShader {
    /// Contains the decoded shader artifact.
    artifact: Artifact,
    /// Records selected stages and matching artifact indices.
    selected: Vec<(Stage, usize)>,
    /// Fully parsed reflection products for selected stages.
    reflections: Vec<ValidatedStageReflection>,
}

impl RuntimeShader {
    /// Decodes an artifact and selects every compatible stage for the requested backend.
    ///
    /// Artifacts may contain a target subset. Selection fails closed when the
    /// requested backend has no product for any declared stage.
    ///
    /// Metal loads use the running macOS identity. Tests and embedding layers that already own a
    /// platform identity can call [`Self::load_for_environment`] explicitly.
    ///
    /// # Errors
    ///
    /// Returns an error before product exposure when decoding, a requested
    /// product is missing, compatibility selection fails, or reflection is invalid.
    pub fn load(
        bytes: &[u8],
        backend: Backend,
        profile: SemanticProfile,
    ) -> Result<Self, ShaderLoadError> {
        let metal = host_metal_environment();
        Self::load_for_environment(bytes, backend, profile, metal.as_ref())
    }

    /// Decodes an artifact and selects products compatible with an explicit Metal environment.
    ///
    /// # Errors
    ///
    /// Returns an error before product exposure when decoding, a requested product is missing,
    /// compatibility selection fails, or reflection validation fails.
    pub fn load_for_environment(
        bytes: &[u8],
        backend: Backend,
        profile: SemanticProfile,
        metal: Option<&MetalEnvironment>,
    ) -> Result<Self, ShaderLoadError> {
        let artifact = Artifact::decode(bytes).map_err(ShaderLoadError::Artifact)?;
        let target = match backend {
            Backend::Vulkan => Target::Spirv,
            Backend::Dx12 => Target::Dxil,
            Backend::Metal => Target::Metallib,
        };
        let profile = profile_label(profile);
        let stages = artifact
            .variants
            .iter()
            .map(|variant| variant.stage)
            .collect::<BTreeSet<_>>();
        let mut selected = Vec::with_capacity(stages.len());
        for stage in stages {
            let index = artifact
                .variants
                .iter()
                .enumerate()
                .filter(|(_, variant)| {
                    variant.target == target && variant.stage == stage && variant.profile == profile
                })
                .filter(|(_, variant)| match (&variant.compatibility, backend) {
                    (TargetCompatibility::Vulkan { .. }, Backend::Vulkan)
                    | (TargetCompatibility::Dx12 { .. }, Backend::Dx12) => true,
                    (
                        TargetCompatibility::MetalLibrary {
                            metal: compatibility,
                        },
                        Backend::Metal,
                    ) => metal.is_some_and(|environment| environment.admits(compatibility)),
                    _ => false,
                })
                .max_by_key(|(_, variant)| match &variant.compatibility {
                    TargetCompatibility::MetalLibrary {
                        metal: compatibility,
                    } => (compatibility.minimum_os, compatibility.sdk),
                    _ => (
                        CompatibilityVersion::new(0, 0),
                        CompatibilityVersion::new(0, 0),
                    ),
                })
                .map(|(index, _)| index)
                .ok_or(ShaderLoadError::MissingProduct { stage })?;
            selected.push((stage, index));
        }
        let reflection_keys = selected
            .iter()
            .map(|(stage, index)| (*stage, artifact.variants[*index].entry_point.as_str()))
            .collect::<Vec<_>>();
        let reflections = validate_stage_reflections(&artifact.metadata, backend, &reflection_keys)
            .map_err(ShaderLoadError::Reflection)?;
        Ok(Self {
            artifact,
            selected,
            reflections,
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

    /// Returns the selected native shader bytes for a stage.
    pub fn product(&self, stage: Stage) -> Option<&[u8]> {
        self.selected
            .iter()
            .find(|(selected_stage, _)| *selected_stage == stage)
            .map(|(_, index)| self.artifact.variants[*index].bytes.as_slice())
    }

    /// Iterates over selected stages and native shader bytes in stage order.
    pub fn products(&self) -> impl ExactSizeIterator<Item = (Stage, &[u8])> {
        self.selected
            .iter()
            .map(|(stage, index)| (*stage, self.artifact.variants[*index].bytes.as_slice()))
    }

    /// Returns the prevalidated physical bindings for one selected stage.
    ///
    /// # Errors
    ///
    /// Returns `BindingError::MissingReflection` if the stage was not selected.
    pub fn bindings(&self, stage: Stage) -> Result<ReflectedBindings, BindingError> {
        self.reflections
            .iter()
            .find(|reflection| reflection.stage() == stage)
            .map(|reflection| reflection.bindings().clone())
            .ok_or(BindingError::MissingReflection)
    }

    /// Returns the prevalidated pipeline layout for one selected stage.
    ///
    /// # Errors
    ///
    /// Returns `BindingError::MissingReflection` if the stage was not selected.
    pub fn pipeline_layout(&self, stage: Stage) -> Result<PipelineLayout, BindingError> {
        self.reflections
            .iter()
            .find(|reflection| reflection.stage() == stage)
            .map(|reflection| *reflection.pipeline_layout())
            .ok_or(BindingError::MissingReflection)
    }

    /// Returns the validated compute thread-group dimensions.
    ///
    /// # Errors
    ///
    /// Returns `BindingError::MissingReflection` if no compute stage was selected.
    pub fn compute_workgroup_size(&self) -> Result<[u32; 3], BindingError> {
        self.reflections
            .iter()
            .find(|reflection| reflection.stage() == Stage::Compute)
            .and_then(ValidatedStageReflection::workgroup_size)
            .ok_or(BindingError::MissingReflection)
    }

    /// Graphics pipelines require exactly one selected vertex product and one selected fragment product.
    ///
    /// # Errors
    ///
    /// Returns `ShaderLoadError::InvalidStagePairing` unless exactly one vertex shader and one fragment shader are selected.
    pub fn graphics_pair(&self) -> Result<GraphicsPair<'_>, ShaderLoadError> {
        let vertex = self.shader_product(Stage::Vertex);
        let fragment = self.shader_product(Stage::Fragment);
        vertex
            .zip(fragment)
            .ok_or(ShaderLoadError::InvalidStagePairing)
    }

    /// Compute lookup accepts mixed artifacts but requires a selected compute product.
    ///
    /// # Errors
    ///
    /// Returns `ShaderLoadError::InvalidStagePairing` unless a compute shader is selected.
    pub fn compute_product(&self) -> Result<ShaderProduct<'_>, ShaderLoadError> {
        self.shader_product(Stage::Compute)
            .ok_or(ShaderLoadError::InvalidStagePairing)
    }

    fn shader_product(&self, stage: Stage) -> Option<ShaderProduct<'_>> {
        self.selected
            .iter()
            .enumerate()
            .find(|(_, (selected_stage, _))| *selected_stage == stage)
            .map(|(product_index, (_, artifact_index))| {
                (
                    product_index,
                    stage,
                    self.artifact.variants[*artifact_index].entry_point.as_str(),
                )
            })
    }
}

/// Maps a semantic profile to its artifact profile label.
const fn profile_label(profile: SemanticProfile) -> &'static str {
    match profile {
        SemanticProfile::V1 => "ez-gfx-v1",
    }
}

#[cfg(target_os = "macos")]
static HOST_METAL_ENVIRONMENT: std::sync::LazyLock<Option<MetalEnvironment>> =
    std::sync::LazyLock::new(detect_host_metal_environment);

#[cfg(target_os = "macos")]
fn host_metal_environment() -> Option<MetalEnvironment> {
    *HOST_METAL_ENVIRONMENT
}

#[cfg(target_os = "macos")]
fn detect_host_metal_environment() -> Option<MetalEnvironment> {
    // A missing or malformed host version fails closed instead of guessing compatibility.
    let version = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|version| {
            let mut components = version.trim().split('.');
            Some(CompatibilityVersion::new(
                components.next()?.parse().ok()?,
                components.next().unwrap_or("0").parse().ok()?,
            ))
        })?;
    let architecture = match std::env::consts::ARCH {
        "aarch64" => AppleArchitecture::Aarch64,
        "x86_64" => AppleArchitecture::X86_64,
        _ => return None,
    };
    Some(MetalEnvironment {
        platform: ApplePlatform::MacOs,
        architecture,
        os: version,
        max_language: CompatibilityVersion::new(3, 0),
        max_library: CompatibilityVersion::new(1, 0),
    })
}

#[cfg(not(target_os = "macos"))]
const fn host_metal_environment() -> Option<MetalEnvironment> {
    None
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Reports artifact decoding and shader selection failures.
pub enum ShaderLoadError {
    /// The shader artifact could not be decoded or validated.
    Artifact(ArtifactError),
    /// A required compiled product is absent.
    MissingProduct {
        /// Shader stage.
        stage: Stage,
    },
    /// Required reflection was missing, ambiguous, malformed, or semantically invalid.
    Reflection(BindingError),
    /// The selected shaders do not form the required graphics or compute stage set.
    InvalidStagePairing,
}
