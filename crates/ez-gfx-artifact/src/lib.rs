#![forbid(unsafe_code)]

use std::{collections::BTreeSet, fmt};

pub const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"EZSHDR01";
const VERSION: u16 = 1;
const HEADER: usize = 52;
const RECORD: usize = 16;
const MAX_SECTIONS: usize = 128;
const MAX_STRING: usize = 16 * 1024;
const MAX_VARIANTS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum Stage {
    Vertex = 1,
    Fragment = 2,
    Compute = 3,
    Geometry = 4,
    TessellationControl = 5,
    TessellationEvaluation = 6,
}

impl Stage {
    fn from_byte(value: u8) -> Result<Self, ArtifactError> {
        match value {
            1 => Ok(Self::Vertex),
            2 => Ok(Self::Fragment),
            3 => Ok(Self::Compute),
            4 => Ok(Self::Geometry),
            5 => Ok(Self::TessellationControl),
            6 => Ok(Self::TessellationEvaluation),
            _ => Err(ArtifactError::InvalidStage),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum Target {
    Spirv = 1,
    Dxil = 2,
    Msl = 3,
    Metallib = 4,
}

impl Target {
    fn from_byte(value: u8) -> Result<Self, ArtifactError> {
        match value {
            1 => Ok(Self::Spirv),
            2 => Ok(Self::Dxil),
            3 => Ok(Self::Msl),
            4 => Ok(Self::Metallib),
            _ => Err(ArtifactError::InvalidTarget),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetVariant {
    pub target: Target,
    pub stage: Stage,
    pub entry_point: String,
    pub profile: String,
    pub bytes: Vec<u8>,
}

impl TargetVariant {
    pub fn new(
        target: Target,
        stage: Stage,
        entry_point: impl Into<String>,
        profile: impl Into<String>,
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
        Ok(Self {
            target,
            stage,
            entry_point,
            profile,
            bytes,
        })
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Provenance {
    pub compiler: String,
    pub compiler_version: String,
    pub options: Vec<String>,
    pub toolchain: String,
}

impl Provenance {
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Artifact {
    pub metadata: Vec<u8>,
    pub provenance: Provenance,
    pub variants: Vec<TargetVariant>,
    digest: [u8; 32],
}

impl Artifact {
    pub fn new(
        metadata: Vec<u8>,
        provenance: Provenance,
        variants: Vec<TargetVariant>,
    ) -> Result<Self, ArtifactError> {
        if metadata.is_empty()
            || std::str::from_utf8(&metadata).is_err()
            || metadata.len() > MAX_STRING * 16
        {
            return Err(ArtifactError::InvalidMetadata);
        }
        provenance.validate()?;
        if variants.is_empty() || variants.len() > MAX_VARIANTS {
            return Err(ArtifactError::InvalidVariantCount);
        }
        let mut seen = BTreeSet::new();
        let mut logical = BTreeSet::new();
        for variant in &variants {
            let key = (
                variant.target,
                variant.entry_point.as_str(),
                variant.stage,
                variant.profile.as_str(),
            );
            if !seen.insert(key) {
                return Err(ArtifactError::DuplicateVariant);
            }
            logical.insert((variant.entry_point.as_str(), variant.stage));
        }
        for &(entry, stage) in &logical {
            for target in [Target::Spirv, Target::Dxil] {
                if !variants
                    .iter()
                    .any(|v| v.entry_point == entry && v.stage == stage && v.target == target)
                {
                    return Err(ArtifactError::MissingCoverage {
                        entry: entry.to_owned(),
                        stage,
                        target,
                    });
                }
            }
            if !variants.iter().any(|v| {
                v.entry_point == entry
                    && v.stage == stage
                    && matches!(v.target, Target::Msl | Target::Metallib)
            }) {
                return Err(ArtifactError::MissingCoverage {
                    entry: entry.to_owned(),
                    stage,
                    target: Target::Metallib,
                });
            }
        }
        let provenance_bytes = encode_provenance(&provenance)?;
        let digest = digest_content(&metadata, &provenance_bytes, &variants);
        Ok(Self {
            metadata,
            provenance,
            variants,
            digest,
        })
    }

    pub fn execution_digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn encode(&self) -> Result<Vec<u8>, ArtifactError> {
        let provenance = encode_provenance(&self.provenance)?;
        let digest = digest_content(&self.metadata, &provenance, &self.variants);
        let mut payloads = vec![self.metadata.clone(), provenance];
        payloads.extend(self.variants.iter().map(encode_variant));
        let sections = payloads.len();
        let payload_start = HEADER
            .checked_add(
                RECORD
                    .checked_mul(sections)
                    .ok_or(ArtifactError::TooLarge)?,
            )
            .ok_or(ArtifactError::TooLarge)?;
        let total = payload_start
            .checked_add(payloads.iter().map(Vec::len).sum())
            .ok_or(ArtifactError::TooLarge)?;
        if total > MAX_ARTIFACT_BYTES {
            return Err(ArtifactError::TooLarge);
        }
        let mut output = Vec::with_capacity(total);
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(&VERSION.to_le_bytes());
        output.extend_from_slice(&(sections as u16).to_le_bytes());
        output.extend_from_slice(&(total as u64).to_le_bytes());
        output.extend_from_slice(&digest);
        let mut offset = payload_start;
        for payload in &payloads {
            output.extend_from_slice(&(offset as u64).to_le_bytes());
            output.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            offset += payload.len();
        }
        for payload in payloads {
            output.extend_from_slice(&payload);
        }
        Ok(output)
    }

    pub fn decode(input: &[u8]) -> Result<Self, ArtifactError> {
        if input.len() > MAX_ARTIFACT_BYTES {
            return Err(ArtifactError::TooLarge);
        }
        if input.len() < HEADER {
            return Err(ArtifactError::Truncated);
        }
        if &input[..8] != MAGIC || u16::from_le_bytes(input[8..10].try_into().unwrap()) != VERSION {
            return Err(ArtifactError::InvalidHeader);
        }
        let count = u16::from_le_bytes(input[10..12].try_into().unwrap()) as usize;
        let total = u64::from_le_bytes(input[12..20].try_into().unwrap()) as usize;
        if !(3..=MAX_SECTIONS).contains(&count) {
            return Err(ArtifactError::InvalidHeader);
        }
        if total != input.len() {
            return Err(if total > input.len() {
                ArtifactError::Truncated
            } else {
                ArtifactError::InvalidHeader
            });
        }
        let table_end = HEADER
            .checked_add(count * RECORD)
            .ok_or(ArtifactError::TooLarge)?;
        if table_end > input.len() {
            return Err(ArtifactError::Truncated);
        }
        let mut sections = Vec::with_capacity(count);
        let mut previous = table_end;
        for i in 0..count {
            let at = HEADER + i * RECORD;
            let offset = u64::from_le_bytes(input[at..at + 8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(input[at + 8..at + 16].try_into().unwrap()) as usize;
            let end = offset
                .checked_add(len)
                .ok_or(ArtifactError::InvalidSection)?;
            if offset != previous || end > input.len() {
                return Err(ArtifactError::InvalidSection);
            }
            previous = end;
            sections.push(&input[offset..end]);
        }
        if previous != input.len() {
            return Err(ArtifactError::InvalidSection);
        }
        let metadata = sections[0].to_vec();
        let provenance = decode_provenance(sections[1])?;
        let mut variants = Vec::new();
        for section in &sections[2..] {
            variants.push(decode_variant(section)?);
        }
        let artifact = Self::new(metadata, provenance, variants)?;
        let mut digest = [0; 32];
        digest.copy_from_slice(&input[20..52]);
        if digest != artifact.digest {
            return Err(ArtifactError::DigestMismatch);
        }
        Ok(artifact)
    }
}

fn digest_content(metadata: &[u8], provenance: &[u8], variants: &[TargetVariant]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ez-gfx-execution-content-v1");
    hasher.update(&(metadata.len() as u64).to_le_bytes());
    hasher.update(metadata);
    hasher.update(&(provenance.len() as u64).to_le_bytes());
    hasher.update(provenance);
    for variant in variants {
        hasher.update(&[variant.target as u8, variant.stage as u8]);
        hasher.update(&(variant.entry_point.len() as u32).to_le_bytes());
        hasher.update(variant.entry_point.as_bytes());
        hasher.update(&(variant.profile.len() as u32).to_le_bytes());
        hasher.update(variant.profile.as_bytes());
        hasher.update(&(variant.bytes.len() as u64).to_le_bytes());
        hasher.update(&variant.bytes);
    }
    *hasher.finalize().as_bytes()
}
fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}
fn get_string<'a>(input: &'a [u8], at: &mut usize) -> Result<&'a str, ArtifactError> {
    if *at + 4 > input.len() {
        return Err(ArtifactError::Truncated);
    }
    let len = u32::from_le_bytes(input[*at..*at + 4].try_into().unwrap()) as usize;
    *at += 4;
    if len > MAX_STRING || *at + len > input.len() {
        return Err(ArtifactError::InvalidProvenance);
    }
    let value = std::str::from_utf8(&input[*at..*at + len])
        .map_err(|_| ArtifactError::InvalidProvenance)?;
    *at += len;
    Ok(value)
}
fn encode_provenance(value: &Provenance) -> Result<Vec<u8>, ArtifactError> {
    value.validate()?;
    let mut out = Vec::new();
    put_string(&mut out, &value.compiler);
    put_string(&mut out, &value.compiler_version);
    put_string(&mut out, &value.toolchain);
    out.extend_from_slice(&(value.options.len() as u16).to_le_bytes());
    for option in &value.options {
        put_string(&mut out, option);
    }
    Ok(out)
}
fn decode_provenance(input: &[u8]) -> Result<Provenance, ArtifactError> {
    let mut at = 0;
    let compiler = get_string(input, &mut at)?.to_owned();
    let version = get_string(input, &mut at)?.to_owned();
    let toolchain = get_string(input, &mut at)?.to_owned();
    if at + 2 > input.len() {
        return Err(ArtifactError::Truncated);
    }
    let count = u16::from_le_bytes(input[at..at + 2].try_into().unwrap()) as usize;
    at += 2;
    if count > 128 {
        return Err(ArtifactError::InvalidProvenance);
    }
    let mut options = Vec::with_capacity(count);
    for _ in 0..count {
        options.push(get_string(input, &mut at)?.to_owned());
    }
    if at != input.len() {
        return Err(ArtifactError::InvalidProvenance);
    }
    let result = Provenance::new(compiler, version, options, toolchain);
    result.validate()?;
    Ok(result)
}
fn encode_variant(value: &TargetVariant) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(value.target as u8);
    out.push(value.stage as u8);
    put_string(&mut out, &value.entry_point);
    put_string(&mut out, &value.profile);
    out.extend_from_slice(&value.bytes);
    out
}
fn decode_variant(input: &[u8]) -> Result<TargetVariant, ArtifactError> {
    if input.len() < 2 {
        return Err(ArtifactError::Truncated);
    }
    let target = Target::from_byte(input[0])?;
    let stage = Stage::from_byte(input[1])?;
    let mut at = 2;
    let entry = get_string(input, &mut at)?.to_owned();
    let profile = get_string(input, &mut at)?.to_owned();
    TargetVariant::new(target, stage, entry, profile, input[at..].to_vec())
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    Truncated,
    TooLarge,
    InvalidHeader,
    InvalidSection,
    InvalidMetadata,
    InvalidProvenance,
    InvalidTarget,
    InvalidEntryPoint,
    InvalidProfile,
    InvalidStage,
    InvalidVariantCount,
    EmptyVariant,
    DuplicateVariant,
    MissingCoverage {
        entry: String,
        stage: Stage,
        target: Target,
    },
    DigestMismatch,
}
impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ArtifactError {}
