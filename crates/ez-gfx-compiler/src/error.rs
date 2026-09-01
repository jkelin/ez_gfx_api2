use core::fmt;
use ez_gfx_artifact::{ArtifactError, Stage};
use std::path::PathBuf;

#[derive(Debug)]
/// Failures from request validation, shader compilation, Apple tooling, output limits, I/O, or artifact creation.
pub enum CompilerError {
    /// The compilation request is malformed or incomplete.
    InvalidRequest(&'static str),
    /// A shader source could not be opened.
    SourceRead {
        /// Shader source path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// An isolated compiler output directory could not be created.
    TemporaryOutputCreate(std::io::Error),
    /// The compiled artifact could not be encoded.
    ArtifactEncoding(ArtifactError),
    /// Encoded artifact bytes failed validation.
    ArtifactValidation(ArtifactError),
    /// One stage declares more than one entry point.
    DuplicateStage(Stage),
    /// The shader declares no entry points.
    NoEntryPoints,
    /// A declared Slang stage is unsupported by the artifact contract.
    UnsupportedStage(String),
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
        match self {
            Self::InvalidRequest(field) => write!(f, "invalid compilation request: {field}"),
            Self::SourceRead { path, .. } => write!(f, "read shader source {}", path.display()),
            Self::TemporaryOutputCreate(_) => f.write_str("create temporary compiler output"),
            Self::ArtifactEncoding(_) => f.write_str("encode shader artifact"),
            Self::ArtifactValidation(_) => f.write_str("validate encoded shader artifact"),
            Self::DuplicateStage(stage) => {
                write!(f, "multiple entry points declare stage {stage:?}")
            }
            Self::NoEntryPoints => f.write_str("shader declares no entry points"),
            Self::UnsupportedStage(stage) => write!(f, "unsupported shader stage {stage}"),
            Self::NativeUnavailable => f.write_str("native Slang compiler is unavailable"),
            Self::Native(message) => write!(f, "Slang compiler failed: {message}"),
            Self::AppleToolNotFound(tool) => write!(f, "Apple shader tool unavailable: {tool}"),
            Self::ToolFailed(message) => write!(f, "external shader tool failed: {message}"),
            Self::OutputLimit => f.write_str("compiled shader output exceeds its limit"),
            Self::Io(_) => f.write_str("shader compiler I/O failed"),
            Self::Artifact(_) => f.write_str("shader artifact construction failed"),
        }
    }
}

impl std::error::Error for CompilerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SourceRead { source, .. }
            | Self::TemporaryOutputCreate(source)
            | Self::Io(source) => Some(source),
            Self::ArtifactEncoding(source)
            | Self::ArtifactValidation(source)
            | Self::Artifact(source) => Some(source),
            Self::InvalidRequest(_)
            | Self::DuplicateStage(_)
            | Self::NoEntryPoints
            | Self::UnsupportedStage(_)
            | Self::NativeUnavailable
            | Self::Native(_)
            | Self::AppleToolNotFound(_)
            | Self::ToolFailed(_)
            | Self::OutputLimit => None,
        }
    }
}

impl From<std::io::Error> for CompilerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ArtifactError> for CompilerError {
    fn from(value: ArtifactError) -> Self {
        Self::Artifact(value)
    }
}
