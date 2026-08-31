//! Runtime integration and contract tests.

use ez_gfx_core::capability::CompressionSupport;
use ez_gfx_runtime::target::*;

#[test]
fn declarations_validate_scale_samples_candidates_and_clear_type() {
    assert_eq!(
        TargetDeclaration::new(
            "color",
            TargetUsage::Color,
            0.0,
            1,
            vec![Format::Rgba8Unorm],
            ClearValue::Color([0.0; 4]),
            true
        ),
        Err(TargetError::InvalidScale)
    );
    assert_eq!(
        TargetDeclaration::new(
            "color",
            TargetUsage::Color,
            1.0,
            3,
            vec![Format::Rgba8Unorm],
            ClearValue::Color([0.0; 4]),
            true
        ),
        Err(TargetError::InvalidSamples)
    );
    assert_eq!(
        TargetDeclaration::new(
            "depth",
            TargetUsage::Depth,
            1.0,
            1,
            vec![Format::Depth32Float],
            ClearValue::Color([0.0; 4]),
            false
        ),
        Err(TargetError::ClearTypeMismatch)
    );
    assert_eq!(
        TargetDeclaration::new(
            "color",
            TargetUsage::Color,
            1.0,
            1,
            vec![Format::Rgba8Unorm],
            ClearValue::Color([f32::NAN, 0.0, 0.0, 0.0]),
            true
        ),
        Err(TargetError::InvalidClear)
    );
}

#[test]
fn format_resolution_follows_candidate_order_and_required_usage() {
    let declaration = TargetDeclaration::new(
        "hdr",
        TargetUsage::Color,
        1.0,
        4,
        vec![Format::Rgba16Float, Format::Rgba8Unorm],
        ClearValue::Color([0.0, 0.0, 0.0, 1.0]),
        true,
    )
    .unwrap();
    let formats = FormatCapabilities::new(vec![
        FormatSupport::new(
            Format::Rgba16Float,
            false,
            true,
            false,
            1,
            CompressionSupport::NONE,
        )
        .unwrap(),
        FormatSupport::new(
            Format::Rgba8Unorm,
            true,
            true,
            false,
            4,
            CompressionSupport::NONE,
        )
        .unwrap(),
    ])
    .unwrap();
    assert_eq!(formats.resolve(&declaration), Ok(Format::Rgba8Unorm));
}

#[test]
fn compressed_formats_require_device_family_support() {
    let declaration = TargetDeclaration::new(
        "texture",
        TargetUsage::Sampled,
        1.0,
        1,
        vec![Format::Bc7Unorm],
        ClearValue::None,
        true,
    )
    .unwrap();
    let formats = FormatCapabilities::new(vec![
        FormatSupport::new(
            Format::Bc7Unorm,
            false,
            true,
            false,
            1,
            CompressionSupport::BC,
        )
        .unwrap(),
    ])
    .unwrap();
    assert_eq!(
        formats.resolve_with_compression(&declaration, CompressionSupport::ASTC),
        Err(TargetError::UnsupportedFormat)
    );
    assert_eq!(
        formats.resolve_with_compression(&declaration, CompressionSupport::BC),
        Ok(Format::Bc7Unorm)
    );
}
