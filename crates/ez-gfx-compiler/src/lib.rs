#![forbid(unsafe_code)]
use core::fmt;
use ez_gfx_artifact::{Artifact, ArtifactError, Provenance, Stage, Target, TargetVariant};
use ez_gfx_core::{Backend, SemanticError, SemanticGraph, TargetLayout};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    process::Command,
};

type CanonicalParameter = (String, Option<String>, Option<String>);
type CanonicalParameters = BTreeMap<(String, Stage), Vec<CanonicalParameter>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompilationContractError {
    DuplicateBackend(Backend),
    MissingBackend(Backend),
    Semantic(SemanticError),
}
impl fmt::Display for CompilationContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CompilationContractError {}
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
pub struct CompilerConfig {
    pub required_version: String,
    pub max_output_bytes: usize,
}
impl CompilerConfig {
    pub fn new(required_version: impl Into<String>) -> Self {
        Self {
            required_version: required_version.into(),
            max_output_bytes: 64 * 1024 * 1024,
        }
    }
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
pub struct TargetRequest {
    pub target: Target,
    pub stage: Stage,
    pub entry_point: String,
    pub profile: String,
}
impl TargetRequest {
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

#[derive(Clone, Debug)]
pub struct CompilationRequest {
    pub source: PathBuf,
    pub output_dir: PathBuf,
    pub targets: Vec<TargetRequest>,
    pub include_dirs: Vec<PathBuf>,
    pub defines: Vec<String>,
    pub semantic_metadata: Vec<u8>,
    pub toolchain: String,
    pub apple_toolchain: String,
    pub release_complete: bool,
}
impl CompilationRequest {
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
        let mut logical = BTreeSet::new();
        for t in &self.targets {
            logical.insert((t.entry_point.as_str(), t.stage));
        }
        for &(entry, stage) in &logical {
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

pub fn compile(
    config: &CompilerConfig,
    request: &CompilationRequest,
) -> Result<Artifact, CompilerError> {
    use shader_slang::Downcast;
    request.validate()?;
    let version = config.validate_tool()?;
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
    let mut canonical_parameters = CanonicalParameters::new();
    components.extend(entries.iter().map(|entry| entry.downcast().clone()));
    let linked = session
        .create_composite_component_type(&components)
        .and_then(|p| p.link())
        .map_err(|e| CompilerError::Native(e.to_string()))?;
    let mut total_output = 0usize;
    let mut variants = Vec::new();
    let mut reflections = Vec::new();
    for (index, target) in request.targets.iter().enumerate() {
        let layout = linked
            .layout(index as i64)
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
        let mut parameters: Vec<_> = layout
            .parameters()
            .chain(reflected_entry.parameters())
            .filter_map(|parameter| {
                let name = parameter.name()?.to_owned();
                let layout = parameter.type_layout();
                let variable = parameter.variable()?;
                let attribute =
                    variable
                        .user_attributes()
                        .find_map(|attribute| match attribute.name() {
                            "StructuredBuffer" => Some(("structured", attribute)),
                            "IndirectBuffer" => Some(("indirect", attribute)),
                            "ColorTarget" | "DepthTarget" => Some(("render_target", attribute)),
                            _ => None,
                        });
                let (api_kind, semantic_name) = match attribute {
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
                ))
            })
            .collect();
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
        let reflection = serde_json::json!({"entry": target.entry_point, "stage": format!("{:?}", target.stage), "profile": target.profile, "parameters": parameters.iter().map(|(name,kind,category,shape,access,semantic_name,api_kind,binding_index,binding_space)| serde_json::json!({"name":name,"kind":kind,"category":category,"resource_shape":shape,"resource_access":access,"semantic_name":semantic_name,"api_kind":api_kind,"binding_index":binding_index,"binding_space":binding_space,"descriptor_count":if api_kind.as_deref() == Some("indirect") { 2 } else { 1 }})).collect::<Vec<_>>()});
        reflections.push(serde_json::json!({"target": format!("{:?}", target.target), "entry": target.entry_point, "stage": format!("{:?}", target.stage), "profile": target.profile, "reflection": reflection}));
        let blob = linked
            .entry_point_code(index as i64, index as i64)
            .map_err(|e| CompilerError::Native(e.to_string()))?;
        let mut bytes = blob.as_slice().to_vec();
        if target.target == Target::Metallib {
            if !cfg!(target_os = "macos") {
                return Err(CompilerError::AppleToolNotFound(
                    "xcrun (macOS only)".into(),
                ));
            }
            fs::create_dir_all(&request.output_dir).map_err(CompilerError::Io)?;
            let stem = format!("{}-{}", target.entry_point, index);
            let msl = request.output_dir.join(format!("{stem}.tmp.metal"));
            let metallib = request.output_dir.join(format!("{stem}.metallib"));
            fs::write(&msl, &bytes).map_err(CompilerError::Io)?;
            let result = build_metallib(msl.clone(), metallib.clone());
            let _ = fs::remove_file(&msl);
            result?;
            bytes = fs::read(metallib).map_err(CompilerError::Io)?;
        }
        total_output = total_output
            .checked_add(bytes.len())
            .ok_or(CompilerError::OutputLimit)?;
        if bytes.is_empty() || total_output > config.max_output_bytes {
            return Err(CompilerError::OutputLimit);
        }
        variants.push(
            TargetVariant::new(
                target.target,
                target.stage,
                &target.entry_point,
                "ez-gfx-v1",
                bytes,
            )
            .map_err(CompilerError::Artifact)?,
        );
    }
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
pub enum CompilerError {
    InvalidRequest(&'static str),
    MissingTarget(Target),
    MissingCoverage {
        entry: String,
        stage: Stage,
        target: Target,
    },
    NativeUnavailable,
    Native(String),
    AppleToolNotFound(String),
    ToolFailed(String),
    VersionMismatch {
        expected: String,
        found: String,
    },
    OutputLimit,
    Io(std::io::Error),
    Artifact(ArtifactError),
}
impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CompilerError {}
