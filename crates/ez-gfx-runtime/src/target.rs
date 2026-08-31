use std::collections::BTreeMap;

use ez_gfx_core::capability::CompressionSupport;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
/// Pixel and depth formats available for render targets and textures.
pub enum Format {
    /// Four-channel 8-bit normalized RGBA format.
    Rgba8Unorm = 1,
    /// Four-channel 8-bit sRGB BGRA format.
    Bgra8Srgb = 2,
    /// Four-channel 16-bit floating-point RGBA format.
    Rgba16Float = 3,
    /// 32-bit floating-point depth format.
    Depth32Float = 4,
    /// BC7-compressed 8-bit normalized RGBA format.
    Bc7Unorm = 5,
    /// ASTC-compressed 4-by-4 texel 8-bit normalized RGBA format.
    Astc4x4Unorm = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Intended access mode for a target.
pub enum TargetUsage {
    /// Supports color-rendering output.
    Color,
    /// Supports depth-rendering output.
    Depth,
    /// Supports storage access.
    Storage,
    /// Supports texture sampling.
    Sampled,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Action applied to existing target contents when a pass begins.
pub enum LoadOp {
    /// Preserves the existing contents.
    Load,
    /// Replaces the contents with the configured clear color or depth.
    Clear,
    /// Leaves the previous contents undefined.
    Discard,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Action applied to target contents when a pass ends.
pub enum StoreOp {
    /// Preserves the rendered contents for later use.
    Store,
    /// Leaves the rendered contents undefined.
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Data used to clear a color or depth-stencil target.
pub enum ClearValue {
    /// No clear value.
    None,
    /// A color clear value.
    Color([f32; 4]),
    /// A depth and stencil clear value.
    DepthStencil {
        /// Depth clear value.
        depth: f32,
        /// Stencil clear value.
        stencil: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
/// Validated requirements for allocating and using a render target.
pub struct TargetDeclaration {
    /// Human-readable target name.
    name: String,
    /// Required access mode.
    usage: TargetUsage,
    /// Scale factor relative to the reference dimensions.
    relative_scale: f32,
    /// Requested multisample count.
    samples: u8,
    /// Acceptable formats in preference order.
    candidates: Vec<Format>,
    /// Data supplied when clearing the target.
    clear: ClearValue,
    /// Whether the target must support texture sampling.
    sampleable: bool,
}

impl TargetDeclaration {
    /// Scale and clear numbers must be finite; candidates are ordered, nonempty, and unique.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is empty or exceeds 255 bytes, the scale is non-finite or non-positive, the sample count is unsupported, candidates are empty or duplicated, or the clear value is invalid or incompatible with the usage.
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

    /// Returns the target's human-readable name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns the required access mode.
    pub const fn usage(&self) -> TargetUsage {
        self.usage
    }
    /// Returns the scale factor relative to the reference dimensions.
    pub const fn relative_scale(&self) -> f32 {
        self.relative_scale
    }
    /// Returns the requested multisample count.
    pub const fn samples(&self) -> u8 {
        self.samples
    }
    /// Returns acceptable formats in preference order.
    pub fn candidates(&self) -> &[Format] {
        &self.candidates
    }
    /// Returns the configured clear data.
    pub const fn clear(&self) -> ClearValue {
        self.clear
    }
    /// Reports whether texture sampling support is required.
    pub const fn sampleable(&self) -> bool {
        self.sampleable
    }
}

/// Checks that clear data is finite, in range, and compatible with the target usage.
///
/// # Errors
///
/// Returns an error if a color clear contains a non-finite component, a depth clear is non-finite or outside 0.0..=1.0, or the clear type is incompatible with the target usage.
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
/// Device capabilities relevant to one texture format.
pub struct FormatSupport {
    /// Format described by these capabilities.
    pub format: Format,
    /// Whether color-rendering output is supported.
    pub color: bool,
    /// Whether texture sampling is supported.
    pub sampled: bool,
    /// Whether storage access is supported.
    pub storage: bool,
    /// Highest supported multisample count.
    pub max_samples: u8,
    /// Compression family required by the format.
    pub compression: CompressionSupport,
}

impl FormatSupport {
    /// Sample limits are the same closed set used by target declarations.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_samples` is not 1, 2, 4, or 8.
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
/// Device format capabilities indexed by texture format.
pub struct FormatCapabilities {
    /// Per-format capabilities used during target resolution.
    formats: BTreeMap<Format, FormatSupport>,
}

impl FormatCapabilities {
    /// Duplicate probe records fail rather than allowing order-dependent capability results.
    ///
    /// # Errors
    ///
    /// Returns an error if more than one capability record specifies the same format.
    pub fn new(formats: Vec<FormatSupport>) -> Result<Self, TargetError> {
        let mut result = BTreeMap::new();
        for support in formats {
            if result.insert(support.format, support).is_some() {
                return Err(TargetError::DuplicateSupport);
            }
        }
        Ok(Self { formats: result })
    }

    /// Selects the first compatible candidate while admitting BC and ASTC compression.
    ///
    /// # Errors
    ///
    /// Returns an error if no candidate format satisfies the declaration's usage, sampling, sample-count, and compression requirements.
    pub fn resolve(&self, declaration: &TargetDeclaration) -> Result<Format, TargetError> {
        self.resolve_with_compression(
            declaration,
            CompressionSupport::BC.union(CompressionSupport::ASTC),
        )
    }

    /// Candidate order is authoritative; compressed candidates also require the admitted device family.
    ///
    /// # Errors
    ///
    /// Returns an error if no candidate format satisfies the declaration's usage, sampling, sample-count, and admitted compression requirements.
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
/// Failure reported while validating or resolving a target.
pub enum TargetError {
    /// The target name is empty or exceeds 255 bytes.
    InvalidName,
    /// The relative scale is non-finite or not positive.
    InvalidScale,
    /// The sample count is not 1, 2, 4, or 8.
    InvalidSamples,
    /// No candidate formats were provided.
    NoCandidates,
    /// A candidate format appears more than once.
    DuplicateCandidate,
    /// Clear data contains a non-finite or out-of-range component.
    InvalidClear,
    /// The clear data does not match the target usage.
    ClearTypeMismatch,
    /// Capabilities contain more than one record for a format.
    DuplicateSupport,
    /// No candidate format satisfies the declared requirements.
    UnsupportedFormat,
}
