//! Validated, deterministic shader artifact containers.
#![forbid(unsafe_code)]

use rkyv::{Archive, Deserialize, Serialize, rancor::Error as RkyvError, util::AlignedVec};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

/// Maximum encoded artifact size accepted by the container format.
pub const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"EZSHDR03";
/// Current framed rkyv shader artifact format version.
pub const ARTIFACT_FORMAT_VERSION: u32 = 3;
const HEADER_BYTES: usize = 56;
const MAX_STRING: usize = 16 * 1024;
const MAX_VARIANTS: usize = 64;

/// Shader execution stages represented by an artifact variant.
#[derive(Archive, Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum Stage {
    /// Vertex shader.
    Vertex = 1,
    /// Fragment shader.
    Fragment = 2,
    /// Compute shader.
    Compute = 3,
    /// Geometry shader.
    Geometry = 4,
    /// Tessellation control shader.
    TessellationControl = 5,
    /// Tessellation evaluation shader.
    TessellationEvaluation = 6,
}

/// Shader products supported by shader artifacts.
#[derive(Archive, Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum Target {
    /// SPIR-V binary product.
    Spirv = 1,
    /// DXIL binary product.
    Dxil = 2,
    /// Portable Metal Shading Language source.
    Msl = 3,
    /// Compiled Metal library binary.
    Metallib = 4,
}

/// A deterministic two-component compatibility version.
#[derive(Archive, Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CompatibilityVersion {
    /// Major version.
    pub major: u16,
    /// Minor version.
    pub minor: u16,
}

impl CompatibilityVersion {
    /// Creates a compatibility version.
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

/// Apple operating-system family targeted by a metallib.
#[derive(Archive, Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum ApplePlatform {
    /// macOS.
    MacOs = 1,
    /// iOS.
    Ios = 2,
    /// tvOS.
    TvOs = 3,
    /// visionOS.
    VisionOs = 4,
}

/// CPU architecture targeted by an offline metallib.
#[derive(Archive, Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum AppleArchitecture {
    /// Apple Silicon.
    Aarch64 = 1,
    /// Intel 64-bit.
    X86_64 = 2,
}

/// Compatibility identity required to admit an offline metallib.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MetalCompatibility {
    /// Apple platform family.
    pub platform: ApplePlatform,
    /// CPU architecture.
    pub architecture: AppleArchitecture,
    /// Oldest supported operating-system version.
    pub minimum_os: CompatibilityVersion,
    /// SDK used to build the library.
    pub sdk: CompatibilityVersion,
    /// Metal language version used by the compiler.
    pub language: CompatibilityVersion,
    /// Metallib format contract version.
    pub library: CompatibilityVersion,
    /// Deterministic Apple compiler/toolchain identity.
    pub toolchain: String,
}

/// Target-specific binary compatibility metadata.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum TargetCompatibility {
    /// SPIR-V version required by Vulkan.
    Vulkan {
        /// SPIR-V compatibility version.
        spirv: CompatibilityVersion,
    },
    /// Shader Model version required by DX12.
    Dx12 {
        /// DXIL Shader Model compatibility version.
        shader_model: CompatibilityVersion,
    },
    /// Metal source language version. Shipping runtimes never select this product.
    MetalSource {
        /// Metal language version.
        language: CompatibilityVersion,
    },
    /// Offline Apple metallib compatibility identity.
    MetalLibrary {
        /// Apple platform, architecture, OS, SDK, language, library, and toolchain contract.
        metal: MetalCompatibility,
    },
}

impl TargetCompatibility {
    /// Returns canonical compatibility metadata for a portable target profile.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError::InvalidCompatibility`] for `Metallib`, whose compatibility must
    /// identify a concrete Apple build environment.
    pub const fn portable(target: Target) -> Result<Self, ArtifactError> {
        match target {
            Target::Spirv => Ok(Self::Vulkan {
                spirv: CompatibilityVersion::new(1, 5),
            }),
            Target::Dxil => Ok(Self::Dx12 {
                shader_model: CompatibilityVersion::new(6, 5),
            }),
            Target::Msl => Ok(Self::MetalSource {
                language: CompatibilityVersion::new(3, 0),
            }),
            Target::Metallib => Err(ArtifactError::InvalidCompatibility),
        }
    }
}

/// One compiled shader variant and its target metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetVariant {
    /// Binary target.
    pub target: Target,
    /// Shader stage.
    pub stage: Stage,
    /// Entry-point name.
    pub entry_point: String,
    /// Target profile name.
    pub profile: String,
    /// Target-specific binary compatibility contract.
    pub compatibility: TargetCompatibility,
    /// Compiled variant bytes.
    pub bytes: Vec<u8>,
}

impl TargetVariant {
    /// Validates and constructs a shader variant.
    ///
    /// # Errors
    ///
    /// Returns an error when entry-point/profile text is empty, oversized, or
    /// contains NUL, or when the binary is empty.
    pub fn new(
        target: Target,
        stage: Stage,
        entry_point: impl Into<String>,
        profile: impl Into<String>,
        compatibility: TargetCompatibility,
        bytes: Vec<u8>,
    ) -> Result<Self, ArtifactError> {
        let entry_point = entry_point.into();
        let profile = profile.into();
        if entry_point.is_empty()
            || entry_point.len() > MAX_STRING
            || entry_point.as_bytes().contains(&0)
        {
            return Err(ArtifactError::InvalidEntryPoint);
        }
        if profile.is_empty() || profile.len() > MAX_STRING || profile.as_bytes().contains(&0) {
            return Err(ArtifactError::InvalidProfile);
        }
        if bytes.is_empty() {
            return Err(ArtifactError::EmptyVariant);
        }
        validate_compatibility(target, &compatibility)?;
        Ok(Self {
            target,
            stage,
            entry_point,
            profile,
            compatibility,
            bytes,
        })
    }
}
/// Compiler identity and options used to produce shader variants.
#[derive(Archive, Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct Provenance {
    /// Compiler name.
    pub compiler: String,
    /// Compiler version.
    pub compiler_version: String,
    /// Compiler options.
    pub options: Vec<String>,
    /// Toolchain identity.
    pub toolchain: String,
}

impl Provenance {
    /// Creates provenance metadata without performing validation.
    pub fn new(
        compiler: impl Into<String>,
        version: impl Into<String>,
        options: Vec<String>,
        toolchain: impl Into<String>,
    ) -> Self {
        Self {
            compiler: compiler.into(),
            compiler_version: version.into(),
            options,
            toolchain: toolchain.into(),
        }
    }
    fn validate(&self) -> Result<(), ArtifactError> {
        for value in [&self.compiler, &self.compiler_version, &self.toolchain] {
            if value.is_empty() || value.len() > MAX_STRING || value.as_bytes().contains(&0) {
                return Err(ArtifactError::InvalidProvenance);
            }
        }
        if self.options.len() > 128
            || self
                .options
                .iter()
                .any(|v| v.len() > MAX_STRING || v.as_bytes().contains(&0))
        {
            return Err(ArtifactError::InvalidProvenance);
        }
        Ok(())
    }
}

#[derive(Archive, Serialize, Deserialize)]
struct ArtifactPayload {
    metadata: Vec<u8>,
    provenance: Provenance,
    stages: Vec<StagePayload>,
}

#[derive(Archive, Serialize, Deserialize)]
struct StagePayload {
    stage: Stage,
    entry_point: String,
    products: Vec<TargetProduct>,
}

#[derive(Archive, Serialize, Deserialize)]
struct TargetProduct {
    target: Target,
    profile: String,
    compatibility: CompatibilityPayload,
    bytes: Vec<u8>,
}

#[derive(Archive, Serialize, Deserialize, Clone)]
enum CompatibilityPayload {
    Vulkan {
        spirv: CompatibilityVersion,
    },
    Dx12 {
        shader_model: CompatibilityVersion,
    },
    MetalSource {
        language: CompatibilityVersion,
    },
    MetalLibrary {
        platform: ApplePlatform,
        architecture: AppleArchitecture,
        minimum_os: CompatibilityVersion,
        sdk: CompatibilityVersion,
        language: CompatibilityVersion,
        library: CompatibilityVersion,
        toolchain: String,
    },
}

impl From<&TargetCompatibility> for CompatibilityPayload {
    fn from(value: &TargetCompatibility) -> Self {
        match value {
            TargetCompatibility::Vulkan { spirv } => Self::Vulkan { spirv: *spirv },
            TargetCompatibility::Dx12 { shader_model } => Self::Dx12 {
                shader_model: *shader_model,
            },
            TargetCompatibility::MetalSource { language } => Self::MetalSource {
                language: *language,
            },
            TargetCompatibility::MetalLibrary { metal } => Self::MetalLibrary {
                platform: metal.platform,
                architecture: metal.architecture,
                minimum_os: metal.minimum_os,
                sdk: metal.sdk,
                language: metal.language,
                library: metal.library,
                toolchain: metal.toolchain.clone(),
            },
        }
    }
}

impl CompatibilityPayload {
    fn into_public(self) -> TargetCompatibility {
        match self {
            Self::Vulkan { spirv } => TargetCompatibility::Vulkan { spirv },
            Self::Dx12 { shader_model } => TargetCompatibility::Dx12 { shader_model },
            Self::MetalSource { language } => TargetCompatibility::MetalSource { language },
            Self::MetalLibrary {
                platform,
                architecture,
                minimum_os,
                sdk,
                language,
                library,
                toolchain,
            } => TargetCompatibility::MetalLibrary {
                metal: MetalCompatibility {
                    platform,
                    architecture,
                    minimum_os,
                    sdk,
                    language,
                    library,
                    toolchain,
                },
            },
        }
    }
}

/// Complete validated shader artifact with metadata and target variants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Artifact {
    /// UTF-8 metadata associated with the compiled artifact.
    pub metadata: Vec<u8>,
    /// Compiler provenance for the artifact.
    pub provenance: Provenance,
    /// Target variants with the same nonempty concrete target-product set for every stage.
    pub variants: Vec<TargetVariant>,
    digest: [u8; 32],
}

impl Artifact {
    /// Validates metadata, provenance, uniqueness, stage entry points, and uniform selected target
    /// products across every declared stage.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid metadata/provenance, invalid variant count, duplicate variants
    /// or stages, inconsistent stage target coverage, or archive serialization failure.
    pub fn new(
        metadata: Vec<u8>,
        provenance: Provenance,
        variants: Vec<TargetVariant>,
    ) -> Result<Self, ArtifactError> {
        validate_payload(&metadata, &provenance, &variants)?;
        let payload = serialize_payload(&metadata, &provenance, &variants)?;
        let digest = *blake3::hash(payload.as_slice()).as_bytes();

        Ok(Self {
            metadata,
            provenance,
            variants,
            digest,
        })
    }

    /// Returns the digest of the archived execution payload.
    pub fn execution_digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Encodes the artifact as a fixed frame containing one checked rkyv payload.
    ///
    /// # Errors
    ///
    /// Returns an error if current public fields violate semantic invariants, serialization fails,
    /// or the framed artifact exceeds [`MAX_ARTIFACT_BYTES`].
    pub fn encode(&self) -> Result<Vec<u8>, ArtifactError> {
        validate_payload(&self.metadata, &self.provenance, &self.variants)?;
        let payload = serialize_payload(&self.metadata, &self.provenance, &self.variants)?;
        let total = HEADER_BYTES
            .checked_add(payload.len())
            .ok_or(ArtifactError::TooLarge)?;
        if total > MAX_ARTIFACT_BYTES {
            return Err(ArtifactError::TooLarge);
        }
        let payload_len = u64::try_from(payload.len()).map_err(|_| ArtifactError::TooLarge)?;
        let digest = blake3::hash(payload.as_slice());
        let mut output = Vec::with_capacity(total);
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        output.extend_from_slice(&payload_len.to_le_bytes());
        output.extend_from_slice(digest.as_bytes());
        output.extend_from_slice(payload.as_slice());
        Ok(output)
    }

    /// Decodes the fixed frame, verifies its digest, byte-checks the rkyv payload, and validates all
    /// semantic invariants before returning owned data.
    ///
    /// # Errors
    ///
    /// Returns an error for truncated, malformed, oversized, incompatible, corrupted, structurally
    /// invalid, or semantically invalid input.
    pub fn decode(input: &[u8]) -> Result<Self, ArtifactError> {
        if input.len() > MAX_ARTIFACT_BYTES {
            return Err(ArtifactError::TooLarge);
        }
        if input.len() < HEADER_BYTES {
            return Err(ArtifactError::Truncated);
        }
        if &input[..8] != MAGIC {
            return Err(ArtifactError::InvalidHeader);
        }
        let version = u32::from_le_bytes(
            input[8..12]
                .try_into()
                .map_err(|_| ArtifactError::Truncated)?,
        );
        if version != ARTIFACT_FORMAT_VERSION {
            return Err(ArtifactError::UnsupportedVersion(version));
        }
        if input[12..16] != [0; 4] {
            return Err(ArtifactError::InvalidHeader);
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            input[16..24]
                .try_into()
                .map_err(|_| ArtifactError::Truncated)?,
        ))
        .map_err(|_| ArtifactError::TooLarge)?;
        let expected = HEADER_BYTES
            .checked_add(payload_len)
            .ok_or(ArtifactError::TooLarge)?;
        if expected != input.len() {
            return Err(if expected > input.len() {
                ArtifactError::Truncated
            } else {
                ArtifactError::InvalidHeader
            });
        }
        let payload = &input[HEADER_BYTES..];
        let actual_digest = blake3::hash(payload);
        if actual_digest.as_bytes() != &input[24..56] {
            return Err(ArtifactError::DigestMismatch);
        }

        // External byte slices carry no alignment guarantee. Copying the already-bounded payload
        // into aligned storage lets bytecheck validate every archived pointer. Semantic collection
        // and string ceilings are then checked on archived views before owned allocation.
        let mut aligned = AlignedVec::<16>::with_capacity(payload.len());
        aligned.extend_from_slice(payload);
        let archived = rkyv::access::<ArchivedArtifactPayload, RkyvError>(aligned.as_slice())
            .map_err(|_| ArtifactError::InvalidArchive)?;
        validate_archived_payload(archived)?;
        let payload = rkyv::deserialize::<ArtifactPayload, RkyvError>(archived)
            .map_err(|_| ArtifactError::InvalidArchive)?;
        let variants = expand_stages(&payload.stages)?;
        validate_payload(&payload.metadata, &payload.provenance, &variants)?;

        Ok(Self {
            metadata: payload.metadata,
            provenance: payload.provenance,
            variants,
            digest: *actual_digest.as_bytes(),
        })
    }
}

fn validate_archived_payload(payload: &ArchivedArtifactPayload) -> Result<(), ArtifactError> {
    let metadata = payload.metadata.as_slice();
    if metadata.is_empty()
        || metadata.len() > MAX_STRING * 16
        || std::str::from_utf8(metadata).is_err()
    {
        return Err(ArtifactError::InvalidMetadata);
    }

    let provenance = &payload.provenance;
    for value in [
        provenance.compiler.as_str(),
        provenance.compiler_version.as_str(),
        provenance.toolchain.as_str(),
    ] {
        if value.is_empty() || value.len() > MAX_STRING || value.as_bytes().contains(&0) {
            return Err(ArtifactError::InvalidProvenance);
        }
    }
    if provenance.options.len() > 128
        || provenance
            .options
            .iter()
            .any(|value| value.len() > MAX_STRING || value.as_bytes().contains(&0))
    {
        return Err(ArtifactError::InvalidProvenance);
    }

    if payload.stages.is_empty() || payload.stages.len() > MAX_VARIANTS {
        return Err(ArtifactError::InvalidVariantCount);
    }
    let mut variant_count = 0usize;
    for stage in payload.stages.iter() {
        let entry = stage.entry_point.as_str();
        if entry.is_empty() || entry.len() > MAX_STRING || entry.as_bytes().contains(&0) {
            return Err(ArtifactError::InvalidEntryPoint);
        }
        if stage.products.is_empty() {
            return Err(ArtifactError::InvalidVariantCount);
        }
        variant_count = variant_count
            .checked_add(stage.products.len())
            .filter(|count| *count <= MAX_VARIANTS)
            .ok_or(ArtifactError::InvalidVariantCount)?;
        for product in stage.products.iter() {
            let profile = product.profile.as_str();
            if profile.is_empty() || profile.len() > MAX_STRING || profile.as_bytes().contains(&0) {
                return Err(ArtifactError::InvalidProfile);
            }
            if let ArchivedCompatibilityPayload::MetalLibrary { toolchain, .. } =
                &product.compatibility
                && (toolchain.is_empty() || toolchain.len() > MAX_STRING)
            {
                return Err(ArtifactError::InvalidCompatibility);
            }
            if product.bytes.is_empty() {
                return Err(ArtifactError::EmptyVariant);
            }
        }
    }
    Ok(())
}

fn serialize_payload(
    metadata: &[u8],
    provenance: &Provenance,
    variants: &[TargetVariant],
) -> Result<AlignedVec<16>, ArtifactError> {
    let mut stages = BTreeMap::<Stage, StagePayload>::new();
    for variant in variants {
        let stage = stages.entry(variant.stage).or_insert_with(|| StagePayload {
            stage: variant.stage,
            entry_point: variant.entry_point.clone(),
            products: Vec::new(),
        });
        stage.products.push(TargetProduct {
            target: variant.target,
            profile: variant.profile.clone(),
            compatibility: CompatibilityPayload::from(&variant.compatibility),
            bytes: variant.bytes.clone(),
        });
    }
    let payload = ArtifactPayload {
        metadata: metadata.to_vec(),
        provenance: provenance.clone(),
        stages: stages.into_values().collect(),
    };
    rkyv::to_bytes::<RkyvError>(&payload).map_err(|_| ArtifactError::Serialization)
}
fn expand_stages(stages: &[StagePayload]) -> Result<Vec<TargetVariant>, ArtifactError> {
    let mut seen = BTreeSet::new();
    let mut variants = Vec::new();
    for stage in stages {
        if !seen.insert(stage.stage) {
            return Err(ArtifactError::DuplicateStage(stage.stage));
        }
        if stage.products.is_empty() {
            return Err(ArtifactError::InvalidVariantCount);
        }
        variants
            .len()
            .checked_add(stage.products.len())
            .filter(|&count| count <= MAX_VARIANTS)
            .ok_or(ArtifactError::InvalidVariantCount)?;
        for product in &stage.products {
            variants.push(TargetVariant {
                target: product.target,
                stage: stage.stage,
                entry_point: stage.entry_point.clone(),
                profile: product.profile.clone(),
                compatibility: product.compatibility.clone().into_public(),
                bytes: product.bytes.clone(),
            });
        }
    }
    Ok(variants)
}

fn validate_compatibility(
    target: Target,
    compatibility: &TargetCompatibility,
) -> Result<(), ArtifactError> {
    let matches_target = matches!(
        (target, compatibility),
        (Target::Spirv, TargetCompatibility::Vulkan { .. })
            | (Target::Dxil, TargetCompatibility::Dx12 { .. })
            | (Target::Msl, TargetCompatibility::MetalSource { .. })
            | (Target::Metallib, TargetCompatibility::MetalLibrary { .. })
    );
    if !matches_target {
        return Err(ArtifactError::InvalidCompatibility);
    }
    if let TargetCompatibility::MetalLibrary { metal } = compatibility
        && (metal.toolchain.is_empty()
            || metal.toolchain.len() > MAX_STRING
            || metal.toolchain.as_bytes().contains(&0)
            || metal.minimum_os.major == 0
            || metal.sdk.major == 0
            || metal.language.major == 0
            || metal.library.major == 0)
    {
        return Err(ArtifactError::InvalidCompatibility);
    }
    Ok(())
}

fn validate_payload(
    metadata: &[u8],
    provenance: &Provenance,
    variants: &[TargetVariant],
) -> Result<(), ArtifactError> {
    if metadata.is_empty()
        || std::str::from_utf8(metadata).is_err()
        || metadata.len() > MAX_STRING * 16
    {
        return Err(ArtifactError::InvalidMetadata);
    }
    provenance.validate()?;
    if variants.is_empty() || variants.len() > MAX_VARIANTS {
        return Err(ArtifactError::InvalidVariantCount);
    }

    let mut seen = BTreeSet::new();
    let mut stages = BTreeMap::new();
    let mut selected_targets = 0_u8;
    for variant in variants {
        validate_variant(variant)?;
        validate_compatibility(variant.target, &variant.compatibility)?;
        let target_bit = 1_u8 << (variant.target as u8 - 1);
        selected_targets |= target_bit;
        if let Some((entry, targets)) = stages.get_mut(&variant.stage) {
            if *entry != variant.entry_point.as_str() {
                return Err(ArtifactError::DuplicateStage(variant.stage));
            }
            *targets |= target_bit;
        } else {
            stages.insert(variant.stage, (variant.entry_point.as_str(), target_bit));
        }
        if !seen.insert((
            variant.target,
            variant.stage,
            variant.profile.as_str(),
            &variant.compatibility,
        )) {
            return Err(ArtifactError::DuplicateVariant);
        }
    }
    for (&stage, (_, targets)) in &stages {
        if *targets != selected_targets {
            return Err(ArtifactError::InconsistentTargetCoverage { stage });
        }
    }
    Ok(())
}

fn validate_variant(variant: &TargetVariant) -> Result<(), ArtifactError> {
    if variant.entry_point.is_empty()
        || variant.entry_point.len() > MAX_STRING
        || variant.entry_point.as_bytes().contains(&0)
    {
        return Err(ArtifactError::InvalidEntryPoint);
    }
    if variant.profile.is_empty()
        || variant.profile.len() > MAX_STRING
        || variant.profile.as_bytes().contains(&0)
    {
        return Err(ArtifactError::InvalidProfile);
    }
    if variant.bytes.is_empty() {
        return Err(ArtifactError::EmptyVariant);
    }
    Ok(())
}
/// Errors encountered while encoding, decoding, or validating an artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    /// Input ended before the complete container was available.
    Truncated,
    /// The container exceeds the configured size limit.
    TooLarge,
    /// The magic, reserved bytes, or payload length is invalid.
    InvalidHeader,
    /// The frame uses an unsupported artifact format version.
    UnsupportedVersion(u32),
    /// The rkyv payload could not be serialized.
    Serialization,
    /// Bytecheck rejected the archived payload or it could not be deserialized.
    InvalidArchive,
    /// Metadata is empty, invalid UTF-8, or oversized.
    InvalidMetadata,
    /// Provenance text or options are invalid.
    InvalidProvenance,
    /// An entry point is empty, oversized, or contains NUL.
    InvalidEntryPoint,
    /// A profile is empty, oversized, or contains NUL.
    InvalidProfile,
    /// Target compatibility metadata is absent, malformed, or mismatched.
    InvalidCompatibility,
    /// The artifact has too few or too many variants.
    InvalidVariantCount,
    /// A variant contains no compiled bytes.
    EmptyVariant,
    /// Two variants have the same target identity.
    DuplicateVariant,
    /// A stage names more than one logical entry point.
    DuplicateStage(Stage),
    /// A stage does not contain the artifact's exact selected concrete target-product set.
    InconsistentTargetCoverage {
        /// Stage whose concrete target-product set differs from the artifact union.
        stage: Stage,
    },
    /// The encoded digest does not match its content.
    DigestMismatch,
}
impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ArtifactError {}

#[cfg(test)]
mod preflight_tests {
    use super::*;

    fn product() -> TargetProduct {
        TargetProduct {
            target: Target::Spirv,
            profile: "ez-gfx-v1".into(),
            compatibility: CompatibilityPayload::from(
                &TargetCompatibility::portable(Target::Spirv).unwrap(),
            ),
            bytes: vec![1],
        }
    }

    fn payload() -> ArtifactPayload {
        ArtifactPayload {
            metadata: br#"{"reflections":[]}"#.to_vec(),
            provenance: Provenance::new("compiler", "1", vec![], "toolchain"),
            stages: vec![StagePayload {
                stage: Stage::Compute,
                entry_point: "main".into(),
                products: vec![product()],
            }],
        }
    }

    fn framed(payload: &ArtifactPayload) -> Vec<u8> {
        let payload = rkyv::to_bytes::<RkyvError>(payload).unwrap();
        let digest = blake3::hash(payload.as_slice());
        let mut bytes = Vec::with_capacity(HEADER_BYTES + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(digest.as_bytes());
        bytes.extend_from_slice(payload.as_slice());
        bytes
    }

    fn assert_valid_archive_rejected_before_owned_decode(
        payload: &ArtifactPayload,
        expected: ArtifactError,
    ) {
        let archived = rkyv::to_bytes::<RkyvError>(payload).unwrap();
        let archived =
            rkyv::access::<ArchivedArtifactPayload, RkyvError>(archived.as_slice()).unwrap();
        assert_eq!(validate_archived_payload(archived), Err(expected.clone()));

        // The malicious archive is structurally valid and can be decoded into owned values. The
        // boundary must reject its archived view before that allocation path is reached.
        assert!(rkyv::deserialize::<ArtifactPayload, RkyvError>(archived).is_ok());
        assert_eq!(Artifact::decode(&framed(payload)), Err(expected));
    }

    #[test]
    fn archived_preflight_rejects_oversized_metadata_and_strings() {
        let mut oversized_metadata = payload();
        oversized_metadata.metadata = vec![b'x'; MAX_STRING * 16 + 1];
        assert_valid_archive_rejected_before_owned_decode(
            &oversized_metadata,
            ArtifactError::InvalidMetadata,
        );

        let mut oversized_entry = payload();
        oversized_entry.stages[0].entry_point = "x".repeat(MAX_STRING + 1);
        assert_valid_archive_rejected_before_owned_decode(
            &oversized_entry,
            ArtifactError::InvalidEntryPoint,
        );

        let mut oversized_profile = payload();
        oversized_profile.stages[0].products[0].profile = "x".repeat(MAX_STRING + 1);
        assert_valid_archive_rejected_before_owned_decode(
            &oversized_profile,
            ArtifactError::InvalidProfile,
        );
    }

    #[test]
    fn archived_preflight_rejects_oversized_provenance_and_variant_counts() {
        let mut oversized_options = payload();
        oversized_options.provenance.options = vec!["x".into(); 129];
        assert_valid_archive_rejected_before_owned_decode(
            &oversized_options,
            ArtifactError::InvalidProvenance,
        );

        let mut oversized_variants = payload();
        oversized_variants.stages[0].products = (0..=MAX_VARIANTS).map(|_| product()).collect();
        assert_valid_archive_rejected_before_owned_decode(
            &oversized_variants,
            ArtifactError::InvalidVariantCount,
        );
    }
}
