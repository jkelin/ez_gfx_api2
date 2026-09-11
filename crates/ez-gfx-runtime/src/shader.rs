use crate::binding::{
    BindingError, PipelineLayout, ReflectedBindings, ValidatedStageReflection,
    validate_stage_reflections,
};
use ez_gfx_artifact::{
    AppleArchitecture, ApplePlatform, Artifact, ArtifactError, CompatibilityVersion,
    MetalCompatibility, Stage, Target, TargetCompatibility,
};
use ez_gfx_core::{Backend, capability::SemanticProfile};

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
#[derive(Clone, Debug, Eq, PartialEq)]
/// Holds a decoded artifact and one named native shader selected for one backend and stage.
pub struct RuntimeShader {
    /// Contains the decoded shader artifact.
    artifact: Artifact,
    /// Selected stage and matching artifact index.
    selected: (Stage, usize),
    /// Fully parsed reflection product for the selected entry point.
    reflection: ValidatedStageReflection,
}

impl RuntimeShader {
    /// Decodes an artifact and selects one named entry point for the requested stage and backend.
    ///
    /// # Errors
    ///
    /// Returns an error before product exposure when decoding, the name is absent, the name belongs
    /// to another stage, the backend product is missing, compatibility selection fails, or
    /// reflection is invalid.
    pub fn load(
        bytes: &[u8],
        backend: Backend,
        profile: SemanticProfile,
        stage: Stage,
        entry_point: &str,
    ) -> Result<Self, ShaderLoadError> {
        let metal = host_metal_environment();
        Self::load_for_environment(bytes, backend, profile, stage, entry_point, metal.as_ref())
    }

    /// Decodes an artifact and selects one named entry point for an explicit Metal environment.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::load`].
    pub fn load_for_environment(
        bytes: &[u8],
        backend: Backend,
        profile: SemanticProfile,
        stage: Stage,
        entry_point: &str,
        metal: Option<&MetalEnvironment>,
    ) -> Result<Self, ShaderLoadError> {
        let artifact = Artifact::decode(bytes).map_err(ShaderLoadError::Artifact)?;
        if entry_point.is_empty() || entry_point.as_bytes().contains(&0) {
            return Err(ShaderLoadError::UnknownEntryPoint);
        }
        let mut named = artifact
            .variants
            .iter()
            .filter(|variant| variant.entry_point == entry_point);
        if !named.clone().any(|variant| variant.stage == stage) {
            return Err(if named.next().is_some() {
                ShaderLoadError::WrongStage
            } else {
                ShaderLoadError::UnknownEntryPoint
            });
        }
        let target = match backend {
            Backend::Vulkan => Target::Spirv,
            Backend::Dx12 => Target::Dxil,
            Backend::Metal => Target::Metallib,
        };
        let profile = profile_label(profile);
        let index = artifact
            .variants
            .iter()
            .enumerate()
            .filter(|(_, variant)| {
                variant.target == target
                    && variant.stage == stage
                    && variant.entry_point == entry_point
                    && variant.profile == profile
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
        let mut reflections =
            validate_stage_reflections(&artifact.metadata, backend, &[(stage, entry_point)])
                .map_err(ShaderLoadError::Reflection)?;
        let reflection = reflections
            .pop()
            .ok_or(ShaderLoadError::Reflection(BindingError::MissingReflection))?;
        Ok(Self {
            artifact,
            selected: (stage, index),
            reflection,
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

    /// Returns the selected native shader bytes when `stage` matches this shader.
    pub fn product(&self, stage: Stage) -> Option<&[u8]> {
        (self.selected.0 == stage).then(|| self.artifact.variants[self.selected.1].bytes.as_slice())
    }

    /// Iterates over the single selected native shader.
    pub fn products(&self) -> impl ExactSizeIterator<Item = (Stage, &[u8])> {
        std::iter::once((
            self.selected.0,
            self.artifact.variants[self.selected.1].bytes.as_slice(),
        ))
    }

    /// Returns the prevalidated physical bindings for one selected stage.
    ///
    /// # Errors
    ///
    /// Returns `BindingError::MissingReflection` if the stage was not selected.
    pub fn bindings(&self, stage: Stage) -> Result<ReflectedBindings, BindingError> {
        (self.reflection.stage() == stage)
            .then(|| self.reflection.bindings().clone())
            .ok_or(BindingError::MissingReflection)
    }

    /// Returns the prevalidated pipeline layout for one selected stage.
    ///
    /// # Errors
    ///
    /// Returns `BindingError::MissingReflection` if the stage was not selected.
    pub fn pipeline_layout(&self, stage: Stage) -> Result<PipelineLayout, BindingError> {
        (self.reflection.stage() == stage)
            .then(|| *self.reflection.pipeline_layout())
            .ok_or(BindingError::MissingReflection)
    }

    /// Returns the selected stage's canonical physical binding and texture-heap identity.
    pub const fn physical_layout_identity(&self) -> crate::binding::StageLayoutIdentity {
        self.reflection.physical_layout_identity()
    }

    /// Returns validated dispatch thread-group dimensions.
    ///
    /// # Errors
    ///
    /// Returns `BindingError::MissingReflection` if no compute, task, or mesh stage was selected.
    pub fn workgroup_size(&self) -> Result<[u32; 3], BindingError> {
        matches!(
            self.reflection.stage(),
            Stage::Compute | Stage::Task | Stage::Mesh
        )
        .then(|| self.reflection.workgroup_size())
        .flatten()
        .ok_or(BindingError::MissingReflection)
    }

    /// Returns the selected product identity.
    pub fn shader_product(&self) -> ShaderProduct<'_> {
        (
            0,
            self.selected.0,
            self.artifact.variants[self.selected.1].entry_point.as_str(),
        )
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
    /// No artifact entry point has the requested name.
    UnknownEntryPoint,
    /// The requested name exists, but not for the requested stage.
    WrongStage,
    /// A required compiled product is absent.
    MissingProduct {
        /// Shader stage.
        stage: Stage,
    },
    /// Required reflection was missing, ambiguous, malformed, or semantically invalid.
    Reflection(BindingError),
}
