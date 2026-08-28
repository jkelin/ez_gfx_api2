use std::collections::BTreeMap;

use ez_gfx_core::capability::CompressionSupport;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Format {
    Rgba8Unorm = 1,
    Bgra8Srgb = 2,
    Rgba16Float = 3,
    Depth32Float = 4,
    Bc7Unorm = 5,
    Astc4x4Unorm = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetUsage {
    Color,
    Depth,
    Storage,
    Sampled,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadOp {
    Load,
    Clear,
    Discard,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearValue {
    None,
    Color([f32; 4]),
    DepthStencil { depth: f32, stencil: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetDeclaration {
    name: String,
    usage: TargetUsage,
    relative_scale: f32,
    samples: u8,
    candidates: Vec<Format>,
    clear: ClearValue,
    sampleable: bool,
}

impl TargetDeclaration {
    /// Scale and clear numbers must be finite; candidates are ordered, nonempty, and unique.
    pub fn new(
        name: impl Into<String>,
        usage: TargetUsage,
        relative_scale: f32,
        samples: u8,
        candidates: Vec<Format>,
        clear: ClearValue,
        sampleable: bool,
    ) -> Result<Self, TargetError> {
        let name = name.into();
        if name.is_empty() || name.len() > 255 {
            return Err(TargetError::InvalidName);
        }
        if !relative_scale.is_finite() || relative_scale <= 0.0 {
            return Err(TargetError::InvalidScale);
        }
        if !matches!(samples, 1 | 2 | 4 | 8) {
            return Err(TargetError::InvalidSamples);
        }
        if candidates.is_empty() {
            return Err(TargetError::NoCandidates);
        }
        let mut sorted = candidates.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != candidates.len() {
            return Err(TargetError::DuplicateCandidate);
        }
        validate_clear(usage, clear)?;
        Ok(Self {
            name,
            usage,
            relative_scale,
            samples,
            candidates,
            clear,
            sampleable,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub const fn usage(&self) -> TargetUsage {
        self.usage
    }
    pub const fn relative_scale(&self) -> f32 {
        self.relative_scale
    }
    pub const fn samples(&self) -> u8 {
        self.samples
    }
    pub fn candidates(&self) -> &[Format] {
        &self.candidates
    }
    pub const fn clear(&self) -> ClearValue {
        self.clear
    }
    pub const fn sampleable(&self) -> bool {
        self.sampleable
    }
}

fn validate_clear(usage: TargetUsage, clear: ClearValue) -> Result<(), TargetError> {
    match clear {
        ClearValue::Color(values) if !values.iter().all(|value| value.is_finite()) => {
            Err(TargetError::InvalidClear)
        }
        ClearValue::DepthStencil { depth, .. }
            if !depth.is_finite() || !(0.0..=1.0).contains(&depth) =>
        {
            Err(TargetError::InvalidClear)
        }
        ClearValue::Color(_) if usage == TargetUsage::Color => Ok(()),
        ClearValue::DepthStencil { .. } if usage == TargetUsage::Depth => Ok(()),
        ClearValue::None => Ok(()),
        ClearValue::Color(_) | ClearValue::DepthStencil { .. } => {
            Err(TargetError::ClearTypeMismatch)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormatSupport {
    pub format: Format,
    pub color: bool,
    pub sampled: bool,
    pub storage: bool,
    pub max_samples: u8,
    pub compression: CompressionSupport,
}

impl FormatSupport {
    /// Sample limits are the same closed set used by target declarations.
    pub fn new(
        format: Format,
        color: bool,
        sampled: bool,
        storage: bool,
        max_samples: u8,
        compression: CompressionSupport,
    ) -> Result<Self, TargetError> {
        if !matches!(max_samples, 1 | 2 | 4 | 8) {
            return Err(TargetError::InvalidSamples);
        }
        Ok(Self {
            format,
            color,
            sampled,
            storage,
            max_samples,
            compression,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FormatCapabilities {
    formats: BTreeMap<Format, FormatSupport>,
}

impl FormatCapabilities {
    /// Duplicate probe records fail rather than allowing order-dependent capability results.
    pub fn new(formats: Vec<FormatSupport>) -> Result<Self, TargetError> {
        let mut result = BTreeMap::new();
        for support in formats {
            if result.insert(support.format, support).is_some() {
                return Err(TargetError::DuplicateSupport);
            }
        }
        Ok(Self { formats: result })
    }

    pub fn resolve(&self, declaration: &TargetDeclaration) -> Result<Format, TargetError> {
        self.resolve_with_compression(
            declaration,
            CompressionSupport::BC.union(CompressionSupport::ASTC),
        )
    }

    /// Candidate order is authoritative; compressed candidates also require the admitted device family.
    pub fn resolve_with_compression(
        &self,
        declaration: &TargetDeclaration,
        compression: CompressionSupport,
    ) -> Result<Format, TargetError> {
        declaration
            .candidates
            .iter()
            .copied()
            .find(|candidate| {
                let Some(support) = self.formats.get(candidate) else {
                    return false;
                };
                let usage = match declaration.usage {
                    TargetUsage::Color => support.color,
                    TargetUsage::Depth => *candidate == Format::Depth32Float,
                    TargetUsage::Storage => support.storage,
                    TargetUsage::Sampled => support.sampled,
                };
                let sampleable = !declaration.sampleable || support.sampled;
                let compressed = support.compression == CompressionSupport::NONE
                    || support.compression.intersects(compression);
                usage && sampleable && declaration.samples <= support.max_samples && compressed
            })
            .ok_or(TargetError::UnsupportedFormat)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetError {
    InvalidName,
    InvalidScale,
    InvalidSamples,
    NoCandidates,
    DuplicateCandidate,
    InvalidClear,
    ClearTypeMismatch,
    DuplicateSupport,
    UnsupportedFormat,
}
