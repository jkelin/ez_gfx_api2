//! Shader compilation request validation and artifact production.

#![forbid(unsafe_code)]
mod error;

pub use error::CompilerError;

use core::fmt;
pub use ez_gfx_artifact::CompiledShader;
use ez_gfx_artifact::{
    AppleArchitecture, ApplePlatform, Artifact, CompatibilityVersion, MetalCompatibility,
    Provenance, Stage, Target as ArtifactTarget, TargetCompatibility, TargetVariant,
};
use ez_gfx_core::{
    Backend, SemanticError, SemanticGraph, TargetLayout, capability::MAX_BINDLESS_SAMPLED_TEXTURES,
};
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// Backend-independent shader output family.
pub enum Target {
    /// SPIR-V 1.5.
    Spirv,
    /// DirectX IL Shader Model 6.5.
    Dxil,
    /// Metal 3.0, emitted as metallib on Apple hosts and as development MSL elsewhere.
    Metal,
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Spirv => "spirv",
            Self::Dxil => "dxil",
            Self::Metal => "metal",
        })
    }
}

impl std::str::FromStr for Target {
    type Err = CompilerError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "spirv" => Ok(Self::Spirv),
            "dxil" => Ok(Self::Dxil),
            "metal" => Ok(Self::Metal),
            _ => Err(CompilerError::InvalidRequest("target")),
        }
    }
}

#[derive(Clone, Debug)]
struct CompilerConfig {
    max_output_bytes: usize,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            max_output_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
struct TargetRequest {
    target: ArtifactTarget,
    stage: Stage,
    entry_point: String,
    profile: &'static str,
}

#[derive(Clone, Debug)]
struct DiscoveredEntry {
    name: String,
    stage: Stage,
}

/// Stateless offline Slang compiler and validated artifact loader.
#[derive(Clone, Copy, Debug, Default)]
pub struct EasyGraphicsCompiler;

impl EasyGraphicsCompiler {
    /// Compiles every Slang entry point into one validated multi-target artifact.
    ///
    /// # Errors
    ///
    /// Returns an error if the source or target list is invalid, Slang reflection or compilation
    /// fails, Apple tooling is unavailable, or artifact encoding or validation fails.
    pub fn compile_shader(
        source: &Path,
        targets: &[Target],
        development: bool,
    ) -> Result<CompiledShader, CompilerError> {
        let bytes = compile_shader_bytes(source, targets, development)?;
        CompiledShader::load(&bytes).map_err(CompilerError::ArtifactValidation)
    }

    /// Validates serialized `.ezgfxshader` bytes without invoking Slang.
    ///
    /// # Errors
    ///
    /// Returns an artifact validation error for malformed or incompatible bytes.
    pub fn load_compiled_shader(bytes: &[u8]) -> Result<CompiledShader, CompilerError> {
        CompiledShader::load(bytes).map_err(CompilerError::ArtifactValidation)
    }
}

fn compile_shader_bytes(
    source: &Path,
    targets: &[Target],
    development: bool,
) -> Result<Vec<u8>, CompilerError> {
    // Metadata distinguishes regular files from directories before platform-specific
    // open behavior can obscure the invalid source kind.
    let source_error = |source_error| CompilerError::SourceRead {
        path: source.to_path_buf(),
        source: source_error,
    };
    let metadata = fs::metadata(source).map_err(source_error)?;
    if !metadata.is_file() {
        return Err(CompilerError::InvalidRequest("source file"));
    }
    fs::File::open(source).map_err(source_error)?;
    // Slang's module loader resolves `<stem>.slang`; reject other extensions so
    // the file opened above and the module loaded below share one lowercase convention.
    if source.extension().and_then(|extension| extension.to_str()) != Some("slang") {
        return Err(CompilerError::InvalidRequest("source extension"));
    }
    let targets = normalize_targets(targets)?;

    let output = tempfile::tempdir().map_err(CompilerError::TemporaryOutputCreate)?;
    let entries = discover_entries(source, targets[0], development)?;
    let target_requests = plan_target_requests(&entries, &targets, development);
    let request = CompilationRequest::new(
        source.to_path_buf(),
        output.path().to_path_buf(),
        target_requests,
    );
    let artifact = compile_request(&CompilerConfig::default(), &request)?;
    let bytes = artifact.encode().map_err(CompilerError::ArtifactEncoding)?;
    Artifact::decode(&bytes).map_err(CompilerError::ArtifactValidation)?;

    Ok(bytes)
}

fn normalize_targets(targets: &[Target]) -> Result<Vec<Target>, CompilerError> {
    if targets.is_empty() {
        return Err(CompilerError::InvalidRequest("targets"));
    }
    let mut unique = BTreeSet::new();
    if targets.iter().any(|target| !unique.insert(*target)) {
        return Err(CompilerError::InvalidRequest("duplicate target"));
    }
    Ok(unique.into_iter().collect())
}

fn plan_target_requests(
    entries: &[DiscoveredEntry],
    targets: &[Target],
    development: bool,
) -> Vec<TargetRequest> {
    let mut requests = Vec::with_capacity(entries.len() * targets.len());
    for target in targets {
        let (artifact_target, profile) = match target {
            Target::Spirv => (ArtifactTarget::Spirv, "spirv_1_5"),
            Target::Dxil => (ArtifactTarget::Dxil, "sm_6_5"),
            Target::Metal => (metal_artifact_target(development), "metal_3_0"),
        };
        requests.extend(entries.iter().map(|entry| TargetRequest {
            target: artifact_target,
            stage: entry.stage,
            entry_point: entry.name.clone(),
            profile,
        }));
    }
    requests
}

#[derive(Clone, Debug)]
struct CompilationRequest {
    source: PathBuf,
    output_dir: PathBuf,
    targets: Vec<TargetRequest>,
}

impl CompilationRequest {
    fn new(source: PathBuf, output_dir: PathBuf, targets: Vec<TargetRequest>) -> Self {
        Self {
            source,
            output_dir,
            targets,
        }
    }

    fn validate(&self) -> Result<(), CompilerError> {
        if self.targets.is_empty()
            || !self.source.is_file()
            || self.output_dir.as_os_str().is_empty()
        {
            return Err(CompilerError::InvalidRequest("source/output"));
        }
        Ok(())
    }
}
fn module_name(source: &Path) -> Result<&str, CompilerError> {
    source
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or(CompilerError::InvalidRequest("source name"))
}

fn source_search_path(source: &Path) -> Result<std::ffi::CString, CompilerError> {
    // Slang receives this path as UTF-8 C text, so unrepresentable or interior-NUL
    // parents must fail instead of naming a different module search directory.
    let parent = source
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_str()
        .ok_or(CompilerError::InvalidRequest("search path"))?;
    std::ffi::CString::new(parent).map_err(|_| CompilerError::InvalidRequest("search path"))
}

fn compile_target(target: ArtifactTarget) -> shader_slang::CompileTarget {
    match target {
        ArtifactTarget::Spirv => shader_slang::CompileTarget::Spirv,
        ArtifactTarget::Dxil => shader_slang::CompileTarget::Dxil,
        ArtifactTarget::Msl | ArtifactTarget::Metallib => shader_slang::CompileTarget::Metal,
    }
}

fn metal_artifact_target(development: bool) -> ArtifactTarget {
    if development && !cfg!(target_os = "macos") {
        ArtifactTarget::Msl
    } else {
        ArtifactTarget::Metallib
    }
}

fn discovery_target(target: Target, development: bool) -> (ArtifactTarget, &'static str) {
    match target {
        Target::Spirv => (ArtifactTarget::Spirv, "spirv_1_5"),
        Target::Dxil => (ArtifactTarget::Dxil, "sm_6_5"),
        Target::Metal => (metal_artifact_target(development), "metal_3_0"),
    }
}

fn artifact_stage(stage: shader_slang::Stage) -> Result<Stage, CompilerError> {
    match stage {
        shader_slang::Stage::Vertex => Ok(Stage::Vertex),
        shader_slang::Stage::Fragment => Ok(Stage::Fragment),
        shader_slang::Stage::Compute => Ok(Stage::Compute),
        shader_slang::Stage::Geometry => Ok(Stage::Geometry),
        shader_slang::Stage::Hull => Ok(Stage::TessellationControl),
        shader_slang::Stage::Domain => Ok(Stage::TessellationEvaluation),
        unsupported => Err(CompilerError::UnsupportedStage(format!("{unsupported:?}"))),
    }
}

fn discover_entries(
    source: &Path,
    target: Target,
    development: bool,
) -> Result<Vec<DiscoveredEntry>, CompilerError> {
    use shader_slang::Downcast;
    let module_name = module_name(source)?;
    let search = source_search_path(source)?;

    let global = shader_slang::GlobalSession::new().ok_or(CompilerError::NativeUnavailable)?;
    let options = shader_slang::CompilerOptions::default()
        .optimization(shader_slang::OptimizationLevel::High)
        .matrix_layout_row(true);
    let (artifact_target, profile) = discovery_target(target, development);
    let target_desc = shader_slang::TargetDesc::default()
        .format(compile_target(artifact_target))
        .profile(global.find_profile(profile))
        .options(&options);
    let paths = [search.as_ptr()];
    let target_descs = [target_desc];
    let session_desc = shader_slang::SessionDesc::default()
        .targets(&target_descs)
        .search_paths(&paths)
        .options(&options);
    let session = global
        .create_session(&session_desc)
        .ok_or(CompilerError::NativeUnavailable)?;
    let module = session
        .load_module(module_name)
        .map_err(|error| CompilerError::Native(error.to_string()))?;
    let module_entries: Vec<_> = module.entry_points().collect();
    if module_entries.is_empty() {
        return Err(CompilerError::NoEntryPoints);
    }
    let mut components = Vec::with_capacity(module_entries.len() + 1);
    components.push(module.downcast().clone());
    components.extend(module_entries.iter().map(|entry| entry.downcast().clone()));
    let linked = session
        .create_composite_component_type(&components)
        .and_then(|program| program.link())
        .map_err(|error| CompilerError::Native(error.to_string()))?;
    let layout = linked
        .layout(0)
        .map_err(|error| CompilerError::Native(error.to_string()))?;
    let mut entries = Vec::with_capacity(module_entries.len());
    for entry in layout.entry_points() {
        let stage = artifact_stage(entry.stage())?;
        entries.push(DiscoveredEntry {
            name: entry.name().to_owned(),
            stage,
        });
    }
    if entries.len() != module_entries.len() {
        return Err(CompilerError::Native(
            "entry point reflection count mismatch".into(),
        ));
    }
    Ok(entries)
}

fn compile_request(
    config: &CompilerConfig,
    request: &CompilationRequest,
) -> Result<Artifact, CompilerError> {
    use shader_slang::Downcast;
    request.validate()?;
    let invocation_dir = request
        .targets
        .iter()
        .any(|target| target.target == ArtifactTarget::Metallib)
        .then(|| InvocationDirectory::create(&request.output_dir))
        .transpose()?;
    let module_name = module_name(&request.source)?;
    let search = source_search_path(&request.source)?;
    let global = shader_slang::GlobalSession::new().ok_or(CompilerError::NativeUnavailable)?;
    let version = global.build_tag_string().to_owned();
    let options = shader_slang::CompilerOptions::default()
        .optimization(shader_slang::OptimizationLevel::High)
        .matrix_layout_row(true);
    let target_descs: Vec<_> = request
        .targets
        .iter()
        .map(|target| {
            shader_slang::TargetDesc::default()
                .format(compile_target(target.target))
                .profile(global.find_profile(target.profile))
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
        .map_err(|error| CompilerError::Native(error.to_string()))?;
    let entries: Vec<_> = request
        .targets
        .iter()
        .map(|target| {
            module
                .find_entry_point_by_name(&target.entry_point)
                .ok_or_else(|| {
                    CompilerError::Native(format!("entry point missing: {}", target.entry_point))
                })
        })
        .collect::<Result<_, _>>()?;
    let mut components = Vec::with_capacity(entries.len() + 1);
    components.push(module.downcast().clone());
    components.extend(entries.iter().map(|entry| entry.downcast().clone()));
    let linked = session
        .create_composite_component_type(&components)
        .and_then(|program| program.link())
        .map_err(|error| CompilerError::Native(error.to_string()))?;
    let (variants, reflections) = compile_targets(
        &linked,
        request,
        invocation_dir.as_ref(),
        config.max_output_bytes,
    )?;
    let metadata =
        serde_json::to_vec(&serde_json::json!({"semantic": {}, "reflections": reflections}))
            .map_err(|error| CompilerError::Native(error.to_string()))?;
    Artifact::new(
        metadata,
        Provenance::new("shader-slang", version, vec![], "fixed profiles"),
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
    shader_slang::ParameterCategory,
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
                    "Buffer" => api_attribute = Some(("buffer", attribute)),
                    "VertexHeap" => api_attribute = Some(("vertex_heap", attribute)),
                    "CounterBuffer" => api_attribute = Some(("counter_buffer", attribute)),
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
                layout.parameter_category(),
            ))
        })
        .collect();
    parameters.sort_by(|left, right| left.0.cmp(&right.0));
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

fn parse_deployment_version(value: &str) -> Result<CompatibilityVersion, CompilerError> {
    // Deployment targets are numeric dotted versions, unlike profile labels that may be prefixed.
    let mut components = value.trim().split('.');
    let major = components
        .next()
        .filter(|component| !component.is_empty())
        .ok_or(CompilerError::InvalidRequest("deployment target"))?
        .parse::<u16>()
        .map_err(|_| CompilerError::InvalidRequest("deployment target"))?;
    let minor = match components.next() {
        None => 0,
        Some(component) if !component.is_empty() => component
            .parse::<u16>()
            .map_err(|_| CompilerError::InvalidRequest("deployment target"))?,
        Some(_) => return Err(CompilerError::InvalidRequest("deployment target")),
    };
    if components.next().is_some() {
        return Err(CompilerError::InvalidRequest("deployment target"));
    }
    Ok(CompatibilityVersion::new(major, minor))
}

fn metal_minimum_os(
    deployment_target: Option<&str>,
    host_os: CompatibilityVersion,
) -> Result<CompatibilityVersion, CompilerError> {
    // An explicit deployment target is authoritative; otherwise use the host OS,
    // never the SDK version, because SDKs commonly support older deployment targets.
    deployment_target
        .map(parse_deployment_version)
        .transpose()
        .map(|value| value.unwrap_or(host_os))
}

fn host_macos_version() -> Result<CompatibilityVersion, CompilerError> {
    let output = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .map_err(CompilerError::Io)?;
    if !output.status.success() {
        return Err(CompilerError::ToolFailed(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    parse_compatibility_version(
        std::str::from_utf8(&output.stdout)
            .map_err(|_| CompilerError::InvalidRequest("macOS version"))?,
    )
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

fn target_compatibility(target: &TargetRequest) -> Result<TargetCompatibility, CompilerError> {
    if target.target != ArtifactTarget::Metallib {
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
    let minimum_os = metal_minimum_os(
        std::env::var("MACOSX_DEPLOYMENT_TARGET").ok().as_deref(),
        host_macos_version()?,
    )?;
    let language = parse_compatibility_version(target.profile)?;
    let toolchain = apple_tool_output(&["metal", "--version"])?;
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

fn reflected_workgroup_size(
    stage: Stage,
    reflected: [u64; 3],
    entry: &str,
) -> Result<Option<[u32; 3]>, CompilerError> {
    if stage != Stage::Compute {
        return Ok(None);
    }
    let dimension = |value| {
        u32::try_from(value)
            .map_err(|_| CompilerError::Native(format!("invalid compute workgroup size: {entry}")))
    };
    let [x, y, z] = reflected;
    let size = [dimension(x)?, dimension(y)?, dimension(z)?];
    if size.contains(&0) {
        return Err(CompilerError::Native(format!(
            "invalid compute workgroup size: {entry}"
        )));
    }
    Ok(Some(size))
}

fn select_texture_heap<T>(
    heaps: impl IntoIterator<Item = Result<Option<(T, u32)>, CompilerError>>,
    entry: &str,
) -> Result<Option<(T, u32)>, CompilerError> {
    let mut selected = None;
    for heap in heaps {
        let Some(heap) = heap? else {
            continue;
        };
        if selected.is_some() {
            return Err(CompilerError::Native(format!(
                "multiple bindless texture heaps: {entry}"
            )));
        }
        selected = Some(heap);
    }
    if selected
        .as_ref()
        .is_some_and(|(_, capacity)| *capacity == 0 || *capacity > MAX_BINDLESS_SAMPLED_TEXTURES)
    {
        return Err(CompilerError::Native(format!(
            "invalid bindless texture heap capacity: {entry}"
        )));
    }
    Ok(selected)
}

fn descriptor_count(api_kind: Option<&str>) -> u32 {
    if api_kind == Some("counter_buffer") {
        2
    } else {
        1
    }
}

fn reflected_resource_access<'a>(api_kind: Option<&str>, access: &'a str) -> &'a str {
    if api_kind == Some("counter_buffer") {
        "ReadWrite"
    } else {
        access
    }
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
        let workgroup_size = reflected_workgroup_size(
            target.stage,
            reflected_entry.compute_thread_group_size(),
            &target.entry_point,
        )?;
        let entry_metadata = linked
            .entry_point_metadata(
                i64::try_from(index)
                    .map_err(|_| CompilerError::InvalidRequest("too many targets"))?,
                i64::try_from(index)
                    .map_err(|_| CompilerError::InvalidRequest("too many targets"))?,
            )
            .map_err(|error| CompilerError::Native(error.to_string()))?;
        let parameters: Vec<_> =
            collect_parameters(layout.parameters().chain(reflected_entry.parameters()));
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
        let used_heaps = parameters
            .iter()
            .filter_map(|parameter| parameter.9.map(|capacity| (parameter, capacity)))
            .map(|(parameter, capacity)| {
                entry_metadata
                    .is_parameter_location_used(
                        parameter.11,
                        u64::from(parameter.8),
                        u64::from(parameter.7),
                    )
                    .ok_or_else(|| {
                        CompilerError::Native(format!(
                            "texture heap usage reflection failed: {}",
                            target.entry_point
                        ))
                    })
                    .map(|used| used.then_some((parameter, capacity)))
            });
        let texture_heap =
            select_texture_heap(used_heaps, &target.entry_point)?.map(|(parameter, capacity)| {
                serde_json::json!({
                    "binding_space": parameter.8,
                    "binding_index": parameter.7,
                    "capacity": capacity,
                    "argument_stride": 2,
                    "texture_argument_offset": 0,
                    "sampler_argument_offset": 1,
                })
            });
        let depth_required = parameters.iter().any(|parameter| parameter.10);
        let reflection = serde_json::json!({"entry": target.entry_point, "stage": format!("{:?}", target.stage), "profile": target.profile, "parameters": parameters.iter().map(|(name,kind,category,shape,access,semantic_name,api_kind,binding_index,binding_space,_,_,_)| serde_json::json!({"name":name,"kind":kind,"category":category,"resource_shape":shape,"resource_access":reflected_resource_access(api_kind.as_deref(), access),"semantic_name":semantic_name,"api_kind":api_kind,"binding_index":binding_index,"binding_space":binding_space,"descriptor_count":descriptor_count(api_kind.as_deref())})).collect::<Vec<_>>(), "texture_heap": texture_heap, "depth_required": depth_required, "workgroup_size": workgroup_size});
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
        if target.target == ArtifactTarget::Metallib {
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
                target_compatibility(target)?,
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

    #[test]
    fn reflected_workgroup_size_validates_each_fixed_dimension() {
        assert_eq!(
            reflected_workgroup_size(Stage::Compute, [8, 2, 1], "main").unwrap(),
            Some([8, 2, 1])
        );
        assert!(reflected_workgroup_size(Stage::Compute, [1, 0, 1], "main").is_err());
        assert!(reflected_workgroup_size(Stage::Compute, [1, u64::MAX, 1], "main").is_err());
        assert_eq!(
            reflected_workgroup_size(Stage::Vertex, [0, 0, 0], "main").unwrap(),
            None
        );
    }

    #[test]
    fn texture_heap_selection_rejects_ambiguity_and_invalid_capacity() {
        let none = std::iter::empty::<Result<Option<(u8, u32)>, CompilerError>>();
        assert_eq!(select_texture_heap(none, "main").unwrap(), None);
        assert_eq!(
            select_texture_heap([Ok(None), Ok(Some((7_u8, 1024)))], "main").unwrap(),
            Some((7, 1024))
        );
        assert!(select_texture_heap([Ok(Some((1_u8, 1))), Ok(Some((2, 1)))], "main").is_err());
        assert!(select_texture_heap([Ok(Some((1_u8, 0)))], "main").is_err());
        assert!(select_texture_heap([Ok(Some((1_u8, 1025)))], "main").is_err());
        assert!(matches!(
            select_texture_heap(
                [
                    Ok(Some((1_u8, 1))),
                    Err(CompilerError::InvalidRequest("reflection")),
                ],
                "main"
            ),
            Err(CompilerError::InvalidRequest("reflection"))
        ));
    }

    #[test]
    fn apple_development_builds_runtime_loadable_metallib() {
        let expected = if cfg!(target_os = "macos") {
            ArtifactTarget::Metallib
        } else {
            ArtifactTarget::Msl
        };

        assert_eq!(metal_artifact_target(true), expected);
        assert_eq!(metal_artifact_target(false), ArtifactTarget::Metallib);
    }

    #[test]
    fn bare_relative_source_searches_the_current_directory() {
        assert_eq!(
            source_search_path(Path::new("shader.slang"))
                .unwrap()
                .as_c_str(),
            c"."
        );
    }

    #[test]
    fn source_search_path_rejects_embedded_nul() {
        assert!(matches!(
            source_search_path(Path::new("bad\0dir/shader.slang")),
            Err(CompilerError::InvalidRequest("search path"))
        ));
    }

    #[test]
    fn metal_minimum_os_rejects_malformed_deployment_target() {
        for value in ["macos", "14.foo", "14.3.2", "prefix_14"] {
            assert!(
                metal_minimum_os(Some(value), CompatibilityVersion::new(15, 7)).is_err(),
                "{value} must be rejected"
            );
        }
    }

    #[test]
    fn metal_minimum_os_prefers_explicit_deployment_target() {
        assert_eq!(
            metal_minimum_os(Some("14.3"), CompatibilityVersion::new(26, 2)).unwrap(),
            CompatibilityVersion::new(14, 3)
        );
    }

    #[test]
    fn metal_minimum_os_uses_host_when_deployment_target_is_absent() {
        assert_eq!(
            metal_minimum_os(None, CompatibilityVersion::new(15, 7)).unwrap(),
            CompatibilityVersion::new(15, 7)
        );
    }

    #[cfg(windows)]
    #[test]
    fn source_search_path_rejects_non_utf8_parent() {
        use std::ffi::OsString;
        use std::os::windows::ffi::OsStringExt;

        let parent = PathBuf::from(OsString::from_wide(&[
            u16::from(b'b'),
            u16::from(b'a'),
            u16::from(b'd'),
            0xD800,
        ]));
        assert!(matches!(
            source_search_path(&parent.join("shader.slang")),
            Err(CompilerError::InvalidRequest("search path"))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn source_search_path_rejects_non_utf8_parent() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let parent = PathBuf::from(OsString::from_vec(b"bad\xFF".to_vec()));
        assert!(matches!(
            source_search_path(&parent.join("shader.slang")),
            Err(CompilerError::InvalidRequest("search path"))
        ));
    }
}
