//! Shader compilation request validation and artifact production.

#![forbid(unsafe_code)]
use core::fmt;
use ez_gfx_artifact::{
    AppleArchitecture, ApplePlatform, Artifact, ArtifactError, CompatibilityVersion,
    MetalCompatibility, Provenance, Stage, Target, TargetCompatibility, TargetVariant,
};
use ez_gfx_core::{Backend, SemanticError, SemanticGraph, TargetLayout};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

type CanonicalParameter = (String, Option<String>, Option<String>);
type CanonicalParameters = BTreeMap<(String, Stage), Vec<CanonicalParameter>>;

struct InvocationDirectory(tempfile::TempDir);

impl InvocationDirectory {
    /// Creates a uniquely named invocation directory that is removed on drop.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the parent or temporary directory cannot be created.
    fn create(parent: &Path) -> Result<Self, CompilerError> {
        fs::create_dir_all(parent).map_err(CompilerError::Io)?;
        tempfile::Builder::new()
            .prefix(".ez-gfx-compile-")
            .tempdir_in(parent)
            .map(Self)
            .map_err(CompilerError::Io)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Errors reported while checking backend layout coverage and semantic compatibility.
pub enum CompilationContractError {
    /// The target layout repeats a backend already supplied.
    DuplicateBackend(Backend),
    /// No target layout was supplied for the indicated backend.
    MissingBackend(Backend),
    /// A target layout violates semantic graph requirements.
    Semantic(SemanticError),
}
impl fmt::Display for CompilationContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CompilationContractError {}
/// Verifies that Vulkan, DX12, and Metal each have one semantically valid target layout.
///
/// # Errors
///
/// Returns an error for duplicate backends, missing Vulkan, DX12, or Metal layouts, or a layout rejected by the semantic graph.
pub fn validate_target_layouts(
    graph: &SemanticGraph,
    layouts: &[TargetLayout],
) -> Result<(), CompilationContractError> {
    let mut set = BTreeSet::new();
    for layout in layouts {
        if !set.insert(layout.backend()) {
            return Err(CompilationContractError::DuplicateBackend(layout.backend()));
        }
        graph
            .validate_layout(layout)
            .map_err(CompilationContractError::Semantic)?;
    }
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        if !set.contains(&backend) {
            return Err(CompilationContractError::MissingBackend(backend));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
/// Settings for Slang version validation and aggregate compiled-output size.
pub struct CompilerConfig {
    /// Tool build-tag fragment required for compilation.
    pub required_version: String,
    /// Maximum combined byte size permitted for compiled outputs.
    pub max_output_bytes: usize,
}
impl CompilerConfig {
    /// Creates compiler settings with the required Slang build-tag fragment and a 64 MiB output limit.
    pub fn new(required_version: impl Into<String>) -> Self {
        Self {
            required_version: required_version.into(),
            max_output_bytes: 64 * 1024 * 1024,
        }
    }
    /// Opens a Slang global session and returns its build tag after version validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the native Slang session is unavailable or its build tag does not contain the required version.
    pub fn validate_tool(&self) -> Result<String, CompilerError> {
        let session = shader_slang::GlobalSession::new().ok_or(CompilerError::NativeUnavailable)?;
        let version = session.build_tag_string().to_owned();
        if !self.required_version.is_empty() && !version.contains(&self.required_version) {
            return Err(CompilerError::VersionMismatch {
                expected: self.required_version.clone(),
                found: version,
            });
        }
        Ok(version)
    }
}

#[derive(Clone, Debug)]
/// Identifies one shader compilation by target format, stage, entry point, and Slang profile.
pub struct TargetRequest {
    /// Binary format to produce for this shader.
    pub target: Target,
    /// Pipeline stage implemented by the entry point.
    pub stage: Stage,
    /// Source entry point to compile.
    pub entry_point: String,
    /// Slang profile used to compile the entry point.
    pub profile: String,
}
impl TargetRequest {
    /// Creates a target request after validating the entry-point and profile strings.
    ///
    /// # Errors
    ///
    /// Returns an error if the entry point or profile is empty or contains a NUL character.
    pub fn new(
        target: Target,
        stage: Stage,
        entry_point: impl Into<String>,
        profile: impl Into<String>,
    ) -> Result<Self, CompilerError> {
        let entry_point = entry_point.into();
        let profile = profile.into();
        if entry_point.is_empty()
            || entry_point.contains('\0')
            || profile.is_empty()
            || profile.contains('\0')
        {
            return Err(CompilerError::InvalidRequest("entry/profile"));
        }
        Ok(Self {
            target,
            stage,
            entry_point,
            profile,
        })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Selects whether a build emits portable MSL source or an Apple metallib.
pub enum MetalOutput {
    /// Produces MSL without requiring Apple tools.
    Source,
    /// Produces a metallib and requires the Apple `xcrun` toolchain.
    Library,
}

#[derive(Clone, Debug)]
/// Compiler configuration and request derived from one shader manifest.
pub struct ShaderBuildPlan {
    /// Compiler version and output-limit configuration.
    pub config: CompilerConfig,
    /// Fully resolved compilation request.
    pub request: CompilationRequest,
}

#[derive(Deserialize)]
struct ShaderManifest {
    source: PathBuf,
    output: PathBuf,
    #[serde(default)]
    required_version: String,
    #[serde(default)]
    include_dirs: Vec<PathBuf>,
    #[serde(default)]
    defines: Vec<String>,
    #[serde(default)]
    semantic_metadata: serde_json::Value,
    #[serde(default)]
    toolchain: String,
    #[serde(default)]
    apple_toolchain: String,
    targets: Vec<ManifestTarget>,
}

#[derive(Deserialize)]
struct ManifestTarget {
    target: String,
    stage: String,
    entry: String,
    profile: String,
}

/// Parses a shader manifest into a host-specific compilation plan.
///
/// Relative source and include paths are resolved from `workspace_root`; the
/// manifest output path is validated but replaced by `output_dir`.
///
/// # Errors
///
/// Returns an error for malformed JSON, missing output names, unknown target or
/// stage names, or invalid target requests.
pub fn plan_manifest(
    manifest: &[u8],
    workspace_root: &Path,
    output_dir: &Path,
    metal: MetalOutput,
) -> Result<ShaderBuildPlan, CompilerError> {
    let manifest: ShaderManifest = serde_json::from_slice(manifest)
        .map_err(|_| CompilerError::InvalidRequest("manifest JSON"))?;
    if manifest.output.file_name().is_none() {
        return Err(CompilerError::InvalidRequest("manifest output"));
    }

    let targets = manifest
        .targets
        .into_iter()
        .map(|target| {
            let format = match target.target.as_str() {
                "spirv" => Target::Spirv,
                "dxil" => Target::Dxil,
                "msl" | "metallib" => match metal {
                    MetalOutput::Source => Target::Msl,
                    MetalOutput::Library => Target::Metallib,
                },
                _ => return Err(CompilerError::InvalidRequest("manifest target")),
            };
            let stage = match target.stage.as_str() {
                "vertex" => Stage::Vertex,
                "fragment" => Stage::Fragment,
                "compute" => Stage::Compute,
                "geometry" => Stage::Geometry,
                "tess-control" => Stage::TessellationControl,
                "tess-eval" => Stage::TessellationEvaluation,
                _ => return Err(CompilerError::InvalidRequest("manifest stage")),
            };
            TargetRequest::new(format, stage, target.entry, target.profile)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut request = CompilationRequest::new(
        resolve_manifest_path(workspace_root, &manifest.source),
        output_dir.to_path_buf(),
        targets,
    );
    request.include_dirs = manifest
        .include_dirs
        .iter()
        .map(|path| resolve_manifest_path(workspace_root, path))
        .collect();
    request.defines = manifest.defines;
    request.semantic_metadata = serde_json::to_vec(&manifest.semantic_metadata)
        .map_err(|_| CompilerError::InvalidRequest("metadata JSON"))?;
    request.toolchain = manifest.toolchain;
    request.apple_toolchain = manifest.apple_toolchain;
    request.release_complete = metal == MetalOutput::Library;

    Ok(ShaderBuildPlan {
        config: CompilerConfig::new(manifest.required_version),
        request,
    })
}

fn resolve_manifest_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else if path == Path::new(".") {
        root.to_path_buf()
    } else {
        root.join(path)
    }
}

#[derive(Clone, Debug)]
/// Inputs, target matrix, metadata, and provenance used to compile one shader source file.
pub struct CompilationRequest {
    /// Shader source file to compile.
    pub source: PathBuf,
    /// Directory used for temporary and generated outputs.
    pub output_dir: PathBuf,
    /// Requested target, stage, entry-point, and profile combinations.
    pub targets: Vec<TargetRequest>,
    /// Additional directories searched for imported shader sources.
    pub include_dirs: Vec<PathBuf>,
    /// Preprocessor definitions passed to Slang.
    pub defines: Vec<String>,
    /// JSON-encoded semantic metadata embedded in the artifact.
    pub semantic_metadata: Vec<u8>,
    /// Toolchain description recorded in artifact provenance.
    pub toolchain: String,
    /// Apple Metal toolchain description recorded in artifact provenance.
    pub apple_toolchain: String,
    /// Whether Metal coverage requires compiled metallib output.
    pub release_complete: bool,
}
impl CompilationRequest {
    /// Creates a compilation request with default metadata, no includes or defines, and required metallib coverage.
    pub fn new(source: PathBuf, output_dir: PathBuf, targets: Vec<TargetRequest>) -> Self {
        Self {
            source,
            output_dir,
            targets,
            include_dirs: vec![],
            defines: vec![],
            semantic_metadata: br"{}".to_vec(),
            toolchain: "unknown".into(),
            apple_toolchain: "unknown".into(),
            release_complete: true,
        }
    }
    /// Checks paths, defines, target uniqueness and coverage, and semantic metadata JSON.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid defines or include paths, duplicate variants, missing target coverage, an empty target list, an invalid source or output path, or invalid metadata JSON.
    pub fn validate(&self) -> Result<(), CompilerError> {
        if self.defines.iter().any(|v| {
            let (key, _) = v.split_once('=').unwrap_or((v.as_str(), ""));
            key.is_empty() || key.contains('\0') || v.contains('\0')
        }) {
            return Err(CompilerError::InvalidRequest("invalid define"));
        }
        if self
            .include_dirs
            .iter()
            .any(|path| path.to_string_lossy().contains('\0'))
        {
            return Err(CompilerError::InvalidRequest("invalid include path"));
        }
        let mut exact = BTreeSet::new();
        for t in &self.targets {
            if !exact.insert((
                t.target,
                t.stage,
                t.entry_point.as_str(),
                t.profile.as_str(),
            )) {
                return Err(CompilerError::InvalidRequest("duplicate variant"));
            }
        }
        let mut stages = BTreeMap::new();
        for target in &self.targets {
            if let Some(entry) = stages.insert(target.stage, target.entry_point.as_str())
                && entry != target.entry_point
            {
                return Err(CompilerError::DuplicateStage(target.stage));
            }
        }
        for (&stage, &entry) in &stages {
            for target in [Target::Spirv, Target::Dxil] {
                if !self
                    .targets
                    .iter()
                    .any(|t| t.entry_point == entry && t.stage == stage && t.target == target)
                {
                    return Err(CompilerError::MissingCoverage {
                        entry: entry.into(),
                        stage,
                        target,
                    });
                }
            }
            let metal = self.targets.iter().any(|t| {
                t.entry_point == entry
                    && t.stage == stage
                    && if self.release_complete {
                        t.target == Target::Metallib
                    } else {
                        matches!(t.target, Target::Msl | Target::Metallib)
                    }
            });
            if !metal {
                return Err(CompilerError::MissingCoverage {
                    entry: entry.into(),
                    stage,
                    target: Target::Metallib,
                });
            }
        }
        if self.targets.is_empty()
            || !self.source.is_file()
            || self.output_dir.as_os_str().is_empty()
        {
            return Err(CompilerError::InvalidRequest("source/output"));
        }
        if serde_json::from_slice::<serde_json::Value>(&self.semantic_metadata).is_err() {
            return Err(CompilerError::InvalidRequest("metadata JSON"));
        }
        Ok(())
    }
}

/// Compiles all requested shader targets and packages binaries, reflection metadata, and provenance into an artifact.
///
/// # Errors
///
/// Returns an error if request validation, Slang setup or compilation, Metal tool invocation, file I/O, output-limit enforcement, metadata serialization, or artifact construction fails.
pub fn compile(
    config: &CompilerConfig,
    request: &CompilationRequest,
) -> Result<Artifact, CompilerError> {
    use shader_slang::Downcast;
    request.validate()?;
    let version = config.validate_tool()?;
    let invocation_dir = request
        .targets
        .iter()
        .any(|target| target.target == Target::Metallib)
        .then(|| InvocationDirectory::create(&request.output_dir))
        .transpose()?;
    let module_name = request
        .source
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or(CompilerError::InvalidRequest("source name"))?;
    let search = std::ffi::CString::new(
        request
            .source
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_string_lossy()
            .as_ref(),
    )
    .map_err(|_| CompilerError::InvalidRequest("search path"))?;
    let global = shader_slang::GlobalSession::new().ok_or(CompilerError::NativeUnavailable)?;
    let mut options = shader_slang::CompilerOptions::default()
        .optimization(shader_slang::OptimizationLevel::High)
        .matrix_layout_row(true);
    for include in &request.include_dirs {
        options = options.include(include.to_string_lossy().as_ref());
    }
    for define in &request.defines {
        let (key, value) = define.split_once('=').unwrap_or((define.as_str(), ""));
        options = options.macro_define(key, value);
    }
    let target_descs: Vec<_> = request
        .targets
        .iter()
        .map(|t| {
            let format = match t.target {
                Target::Spirv => shader_slang::CompileTarget::Spirv,
                Target::Dxil => shader_slang::CompileTarget::Dxil,
                Target::Msl | Target::Metallib => shader_slang::CompileTarget::Metal,
            };
            shader_slang::TargetDesc::default()
                .format(format)
                .profile(global.find_profile(&t.profile))
                .options(&options)
        })
        .collect();
    let paths = [search.as_ptr()];
    let session_desc = shader_slang::SessionDesc::default()
        .targets(&target_descs)
        .search_paths(&paths)
        .options(&options);
    let session = global
        .create_session(&session_desc)
        .ok_or(CompilerError::NativeUnavailable)?;
    let module = session
        .load_module(module_name)
        .map_err(|e| CompilerError::Native(e.to_string()))?;
    let entries: Vec<_> = request
        .targets
        .iter()
        .map(|t| {
            module
                .find_entry_point_by_name(&t.entry_point)
                .ok_or_else(|| {
                    CompilerError::Native(format!("entry point missing: {}", t.entry_point))
                })
        })
        .collect::<Result<_, _>>()?;
    let mut components = vec![module.downcast().clone()];
    components.extend(entries.iter().map(|entry| entry.downcast().clone()));
    let linked = session
        .create_composite_component_type(&components)
        .and_then(|p| p.link())
        .map_err(|e| CompilerError::Native(e.to_string()))?;
    let (variants, reflections) = compile_targets(
        &linked,
        request,
        invocation_dir.as_ref(),
        config.max_output_bytes,
    )?;
    let semantic: serde_json::Value = serde_json::from_slice(&request.semantic_metadata)
        .map_err(|_| CompilerError::InvalidRequest("metadata JSON"))?;
    let metadata =
        serde_json::to_vec(&serde_json::json!({"semantic": semantic, "reflections": reflections}))
            .map_err(|e| CompilerError::Native(e.to_string()))?;
    Artifact::new(
        metadata,
        Provenance::new(
            "shader-slang",
            version,
            request.defines.clone(),
            format!(
                "{};apple-metal={}",
                request.toolchain, request.apple_toolchain
            ),
        ),
        variants,
    )
    .map_err(CompilerError::Artifact)
}

type ReflectedParameter = (
    String,
    String,
    String,
    String,
    String,
    Option<String>,
    Option<String>,
    u32,
    u32,
    Option<u32>,
    bool,
);

fn collect_parameters<'a>(
    parameters: impl Iterator<Item = &'a shader_slang::reflection::VariableLayout>,
) -> Vec<ReflectedParameter> {
    let mut parameters: Vec<_> = parameters
        .filter_map(|parameter| {
            let name = parameter.name()?.to_owned();
            let layout = parameter.type_layout();
            let variable = parameter.variable()?;
            let mut api_attribute = None;
            let mut texture_heap_capacity = None;
            let mut depth_required = false;
            for attribute in variable.user_attributes() {
                match attribute.name() {
                    "StructuredBuffer" => api_attribute = Some(("structured", attribute)),
                    "IndirectBuffer" => api_attribute = Some(("indirect", attribute)),
                    "ColorTarget" | "DepthTarget" => {
                        api_attribute = Some(("render_target", attribute));
                    }
                    "BindlessTextureHeap" if attribute.argument_count() == 1 => {
                        texture_heap_capacity = Some(
                            attribute
                                .argument_value_int(0)
                                .and_then(|value| u32::try_from(value).ok())
                                .unwrap_or(0),
                        );
                    }
                    "DepthPipeline" if attribute.argument_count() == 0 => {
                        depth_required = true;
                    }
                    _ => {}
                }
            }
            let (api_kind, semantic_name) = match api_attribute {
                Some((kind, attribute)) if attribute.argument_count() == 1 => (
                    Some(kind.to_owned()),
                    attribute.argument_value_string(0).map(str::to_owned),
                ),
                _ => (None, None),
            };
            let shape = format!("{:?}", layout.resource_shape());
            let resource_access = match layout.resource_access() {
                Some(shader_slang::ResourceAccess::Read) => "Read",
                Some(shader_slang::ResourceAccess::ReadWrite) => "ReadWrite",
                Some(shader_slang::ResourceAccess::RasterOrdered) => "RasterOrdered",
                Some(shader_slang::ResourceAccess::Append) => "Append",
                Some(shader_slang::ResourceAccess::Consume) => "Consume",
                Some(shader_slang::ResourceAccess::Write) => "Write",
                Some(shader_slang::ResourceAccess::Feedback) => "Feedback",
                Some(shader_slang::ResourceAccess::Unknown) => "Unknown",
                None | Some(shader_slang::ResourceAccess::None) => "None",
            };
            Some((
                name,
                format!("{:?}", layout.kind()),
                format!("{:?}", layout.parameter_category()),
                shape,
                resource_access.to_owned(),
                semantic_name,
                api_kind,
                parameter.binding_index(),
                parameter.binding_space(),
                texture_heap_capacity,
                depth_required,
            ))
        })
        .collect();
    parameters.sort();
    parameters
}

fn parse_compatibility_version(value: &str) -> Result<CompatibilityVersion, CompilerError> {
    // Profiles may prefix the numeric version (`metal_3_0`); absent minor versions mean `.0`.
    let value = value.trim();
    let mut components = value.split(['.', '_']);
    let major = components
        .find_map(|component| component.parse::<u16>().ok())
        .ok_or(CompilerError::InvalidRequest("compatibility version"))?;
    let minor = components
        .find_map(|component| component.parse::<u16>().ok())
        .unwrap_or(0);
    Ok(CompatibilityVersion::new(major, minor))
}

fn apple_tool_output(args: &[&str]) -> Result<String, CompilerError> {
    let output = Command::new("xcrun")
        .args(args)
        .output()
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => CompilerError::AppleToolNotFound("xcrun".into()),
            _ => CompilerError::Io(error),
        })?;
    if !output.status.success() {
        return Err(CompilerError::ToolFailed(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| CompilerError::InvalidRequest("Apple tool output"))
}

fn target_compatibility(
    target: &TargetRequest,
    request: &CompilationRequest,
) -> Result<TargetCompatibility, CompilerError> {
    if target.target != Target::Metallib {
        return TargetCompatibility::portable(target.target).map_err(CompilerError::Artifact);
    }
    let architecture = match std::env::consts::ARCH {
        "aarch64" => AppleArchitecture::Aarch64,
        "x86_64" => AppleArchitecture::X86_64,
        _ => return Err(CompilerError::InvalidRequest("Apple architecture")),
    };
    let sdk = parse_compatibility_version(&apple_tool_output(&[
        "--sdk",
        "macosx",
        "--show-sdk-version",
    ])?)?;
    let minimum_os = std::env::var("MACOSX_DEPLOYMENT_TARGET")
        .ok()
        .map(|value| parse_compatibility_version(&value))
        .transpose()?
        .unwrap_or(sdk);
    let language = parse_compatibility_version(&target.profile)?;
    let toolchain = if request.apple_toolchain.is_empty() || request.apple_toolchain == "unknown" {
        apple_tool_output(&["metal", "--version"])?
    } else {
        request.apple_toolchain.clone()
    };
    Ok(TargetCompatibility::MetalLibrary {
        metal: MetalCompatibility {
            platform: ApplePlatform::MacOs,
            architecture,
            minimum_os,
            sdk,
            language,
            library: CompatibilityVersion::new(1, 0),
            toolchain,
        },
    })
}

fn compile_targets(
    linked: &shader_slang::ComponentType,
    request: &CompilationRequest,
    invocation_dir: Option<&InvocationDirectory>,
    max_output_bytes: usize,
) -> Result<(Vec<TargetVariant>, Vec<serde_json::Value>), CompilerError> {
    let mut canonical_parameters = CanonicalParameters::new();
    let mut total_output = 0usize;
    let mut variants = Vec::new();
    let mut reflections = Vec::new();
    for (index, target) in request.targets.iter().enumerate() {
        let layout = linked
            .layout(
                i64::try_from(index)
                    .map_err(|_| CompilerError::InvalidRequest("too many targets"))?,
            )
            .map_err(|e| CompilerError::Native(e.to_string()))?;
        let reflected_entry = layout
            .entry_points()
            .find(|entry| entry.name() == target.entry_point)
            .ok_or_else(|| {
                CompilerError::Native(format!(
                    "entry point reflection missing: {}",
                    target.entry_point
                ))
            })?;
        let stage_matches = match target.stage {
            Stage::Vertex => reflected_entry.stage() == shader_slang::Stage::Vertex,
            Stage::Fragment => reflected_entry.stage() == shader_slang::Stage::Fragment,
            Stage::Compute => reflected_entry.stage() == shader_slang::Stage::Compute,
            Stage::Geometry => reflected_entry.stage() == shader_slang::Stage::Geometry,
            Stage::TessellationControl => reflected_entry.stage() == shader_slang::Stage::Hull,
            Stage::TessellationEvaluation => reflected_entry.stage() == shader_slang::Stage::Domain,
        };
        if !stage_matches {
            return Err(CompilerError::Native(format!(
                "stage mismatch: {}",
                target.entry_point
            )));
        }
        let mut parameters: Vec<_> =
            collect_parameters(layout.parameters().chain(reflected_entry.parameters()));
        parameters.sort();
        let canonical_view = parameters
            .iter()
            .filter(|parameter| parameter.6.is_some())
            .map(|parameter| {
                (
                    parameter.0.clone(),
                    parameter.5.clone(),
                    parameter.6.clone(),
                )
            })
            .collect::<Vec<_>>();
        let key = (target.entry_point.clone(), target.stage);
        if let Some(canonical) = canonical_parameters.get(&key) {
            if canonical != &canonical_view {
                return Err(CompilerError::Native(format!(
                    "reflection parameter mismatch: {}",
                    target.entry_point
                )));
            }
        } else {
            canonical_parameters.insert(key, canonical_view);
        }
        let mut heaps = parameters
            .iter()
            .filter_map(|parameter| parameter.9.map(|capacity| (parameter, capacity)));
        if let Some((_, capacity)) = heaps.clone().next()
            && (capacity == 0 || capacity > 1024)
        {
            return Err(CompilerError::Native(format!(
                "invalid bindless texture heap capacity: {}",
                target.entry_point
            )));
        }
        let texture_heap = heaps.next().map(|(parameter, capacity)| {
            serde_json::json!({
                "binding_space": parameter.8,
                "binding_index": parameter.7,
                "capacity": capacity,
                "argument_stride": 2,
                "texture_argument_offset": 0,
                "sampler_argument_offset": 1,
            })
        });
        if heaps.next().is_some() {
            return Err(CompilerError::Native(format!(
                "multiple bindless texture heaps: {}",
                target.entry_point
            )));
        }
        let depth_required = parameters.iter().any(|parameter| parameter.10);
        let reflection = serde_json::json!({"entry": target.entry_point, "stage": format!("{:?}", target.stage), "profile": target.profile, "parameters": parameters.iter().map(|(name,kind,category,shape,access,semantic_name,api_kind,binding_index,binding_space,_,_)| serde_json::json!({"name":name,"kind":kind,"category":category,"resource_shape":shape,"resource_access":access,"semantic_name":semantic_name,"api_kind":api_kind,"binding_index":binding_index,"binding_space":binding_space,"descriptor_count":1})).collect::<Vec<_>>(), "texture_heap": texture_heap, "depth_required": depth_required});
        reflections.push(serde_json::json!({"target": format!("{:?}", target.target), "entry": target.entry_point, "stage": format!("{:?}", target.stage), "profile": target.profile, "reflection": reflection}));
        let blob = linked
            .entry_point_code(
                i64::try_from(index)
                    .map_err(|_| CompilerError::InvalidRequest("too many targets"))?,
                i64::try_from(index)
                    .map_err(|_| CompilerError::InvalidRequest("too many targets"))?,
            )
            .map_err(|e| CompilerError::Native(e.to_string()))?;
        let mut bytes = blob.as_slice().to_vec();
        if target.target == Target::Metallib {
            if !cfg!(target_os = "macos") {
                return Err(CompilerError::AppleToolNotFound(
                    "xcrun (macOS only)".into(),
                ));
            }
            let invocation_dir =
                invocation_dir.ok_or(CompilerError::InvalidRequest("Metal output directory"))?;
            let stem = format!("{}-{index}", target.entry_point);
            let msl = invocation_dir.0.path().join(format!("{stem}.metal"));
            let metallib = invocation_dir.0.path().join(format!("{stem}.metallib"));
            fs::write(&msl, &bytes).map_err(CompilerError::Io)?;
            let result = build_metallib(msl.clone(), metallib.clone());
            let _ = fs::remove_file(&msl);
            result?;
            bytes = fs::read(metallib).map_err(CompilerError::Io)?;
        }
        total_output = total_output
            .checked_add(bytes.len())
            .ok_or(CompilerError::OutputLimit)?;
        if bytes.is_empty() || total_output > max_output_bytes {
            return Err(CompilerError::OutputLimit);
        }
        variants.push(
            TargetVariant::new(
                target.target,
                target.stage,
                &target.entry_point,
                "ez-gfx-v1",
                target_compatibility(target, request)?,
                bytes,
            )
            .map_err(CompilerError::Artifact)?,
        );
    }
    Ok((variants, reflections))
}

/// Compiles Metal source to AIR and links it into a metallib using the macOS `xcrun` toolchain.
///
/// # Errors
///
/// Returns an error if either Apple compiler command cannot start or exits unsuccessfully.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Callers transfer owned paths used for platform tool invocation."
)]
pub fn build_metallib(metal: PathBuf, output: PathBuf) -> Result<(), CompilerError> {
    let ir = output.with_extension("air");
    let mut c = Command::new("xcrun");
    c.args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal)
        .arg("-o")
        .arg(&ir);
    if let Err(error) = run_apple(&mut c) {
        let _ = fs::remove_file(&ir);
        return Err(error);
    }

    let mut l = Command::new("xcrun");
    l.args(["-sdk", "macosx", "metal", "-o"])
        .arg(&output)
        .arg(&ir);
    let result = run_apple(&mut l);
    let _ = fs::remove_file(&ir);
    result
}
///
/// # Errors
///
/// Returns an error if the command cannot start or exits unsuccessfully.
fn run_apple(c: &mut Command) -> Result<(), CompilerError> {
    let o = c.output().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            CompilerError::AppleToolNotFound("xcrun".into())
        } else {
            CompilerError::Io(e)
        }
    })?;
    if !o.status.success() {
        return Err(CompilerError::ToolFailed(
            String::from_utf8_lossy(&o.stderr).into(),
        ));
    }
    Ok(())
}
#[derive(Debug)]
/// Failures from request validation, shader compilation, Apple tooling, output limits, I/O, or artifact creation.
pub enum CompilerError {
    /// The compilation request is malformed or incomplete.
    InvalidRequest(&'static str),
    /// One stage names more than one logical entry point.
    DuplicateStage(Stage),
    /// A required compilation target was not requested.
    MissingTarget(Target),
    /// An entry point and stage lack output for a required target.
    MissingCoverage {
        /// Entry-point name.
        entry: String,
        /// Shader stage.
        stage: Stage,
        /// Compilation target.
        target: Target,
    },
    /// The compiler's expected and observed versions differ.
    VersionMismatch {
        /// Expected toolchain version.
        expected: String,
        /// Observed toolchain version.
        found: String,
    },
    /// The native compiler backend is unavailable.
    NativeUnavailable,
    /// A native compiler operation failed.
    Native(String),
    /// The Apple shader toolchain is unavailable.
    AppleToolNotFound(String),
    /// An external compiler tool failed.
    ToolFailed(String),
    /// Compiled output is empty, overflows its size total, or exceeds the configured limit.
    OutputLimit,
    /// File or directory access failed.
    Io(std::io::Error),
    /// Artifact construction or validation failed.
    Artifact(ArtifactError),
}
impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CompilerError {}

#[cfg(test)]
mod compatibility_tests {
    use super::*;

    #[test]
    fn compatibility_version_accepts_profiles_and_rejects_missing_numbers() {
        assert_eq!(
            parse_compatibility_version("metal_3_0").unwrap(),
            CompatibilityVersion::new(3, 0)
        );
        assert_eq!(
            parse_compatibility_version("15.2").unwrap(),
            CompatibilityVersion::new(15, 2)
        );
        assert_eq!(
            parse_compatibility_version("15").unwrap(),
            CompatibilityVersion::new(15, 0)
        );
        assert!(parse_compatibility_version("metal").is_err());
    }
}
