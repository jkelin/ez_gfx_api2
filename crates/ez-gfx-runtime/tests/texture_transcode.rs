//! Feature-gated native texture transcode contracts.
#![cfg(any(feature = "basis", feature = "ktx2"))]

#[cfg(feature = "ktx2")]
use ez_gfx_runtime::texture::TextureError;
use ez_gfx_runtime::texture::{TextureDecoder, TextureSource};

#[cfg(all(feature = "ktx2", not(feature = "basis")))]
#[test]
fn universal_ktx2_requires_native_feature() {
    let source = include_bytes!("fixtures/alpha_simple_basis.ktx2");
    assert_eq!(
        TextureDecoder::decode(TextureSource::Ktx2, source),
        Err(TextureError::Unsupported)
    );
}

#[cfg(all(feature = "ktx2", feature = "basis"))]
#[test]
fn universal_ktx2_rejects_oversized_dimensions_before_transcoding() {
    let mut source = include_bytes!("fixtures/alpha_simple_basis.ktx2").to_vec();
    for dimension in [0_u32, 1 << 30] {
        source[20..24].copy_from_slice(&dimension.to_le_bytes());
        assert!(matches!(
            TextureDecoder::decode(TextureSource::Ktx2, &source),
            Err(TextureError::InvalidData | TextureError::TooLarge)
        ));
    }
}

#[cfg(feature = "basis")]
#[test]
fn standalone_explicit_target_does_not_follow_auto_policy() {
    use ez_gfx_core::capability::CompressionSupport;
    use ez_gfx_hal::TextureFormat;
    use ez_gfx_runtime::texture::TextureDestination;
    let source = include_bytes!("fixtures/rust-logo-etc.basis");
    for (destination, format) in [
        (TextureDestination::Bc1Srgb, TextureFormat::Bc1Srgb),
        (TextureDestination::Bc3Unorm, TextureFormat::Bc3Unorm),
        (TextureDestination::Bc7Srgb, TextureFormat::Bc7Srgb),
        (
            TextureDestination::Astc4x4Unorm,
            TextureFormat::Astc4x4Unorm,
        ),
        (TextureDestination::Rgba8Srgb, TextureFormat::Rgba8Srgb),
    ] {
        let decoded = TextureDecoder::decode_for_destination(
            TextureSource::Basis,
            source,
            CompressionSupport::BC.union(CompressionSupport::ASTC),
            destination,
        )
        .unwrap();
        assert_eq!(decoded.format, format);
        assert_eq!(
            decoded.mips[0].bytes.len() as u64,
            format.level_bytes(64, 64).unwrap()
        );
    }
}

#[cfg(all(feature = "basis", feature = "ktx2"))]
#[test]
fn uastc_containers_preserve_srgb_and_explicit_linear_override() {
    use ez_gfx_core::capability::CompressionSupport;
    use ez_gfx_hal::TextureFormat;
    use ez_gfx_runtime::texture::TextureDestination;
    // Generated from examples/02_textured_cube/cube.png, resized to 32x32 with Triangle filtering,
    // by basisu_c_sys 0.9.0: UASTC LDR4x4, sRGB defaults, no Zstd; same encoded source in both containers.
    let ktx = include_bytes!("fixtures/cube-uastc-srgb.ktx2");
    let basis = include_bytes!("fixtures/cube-uastc-srgb.basis");
    for (source, bytes) in [
        (TextureSource::Ktx2, ktx.as_slice()),
        (TextureSource::Basis, basis.as_slice()),
    ] {
        let automatic =
            TextureDecoder::decode_with_support(source, bytes, CompressionSupport::BC).unwrap();
        assert_eq!(automatic.format, TextureFormat::Bc7Srgb);
        let linear = TextureDecoder::decode_for_destination(
            source,
            bytes,
            CompressionSupport::BC,
            TextureDestination::Bc7Unorm,
        )
        .unwrap();
        assert_eq!(linear.format, TextureFormat::Bc7Unorm);
        assert_eq!(automatic.mips, linear.mips);
    }
    let ktx_rgba = TextureDecoder::decode(TextureSource::Ktx2, ktx).unwrap();
    let basis_rgba = TextureDecoder::decode(TextureSource::Basis, basis).unwrap();
    assert_eq!(ktx_rgba.mips, basis_rgba.mips);
}

#[cfg(all(feature = "basis", feature = "ktx2"))]
#[test]
fn data_channel_auto_retains_transfer_metadata_without_changing_pixels() {
    use ez_gfx_core::capability::CompressionSupport;
    use ez_gfx_hal::TextureFormat;
    let mut source = include_bytes!("fixtures/alpha_simple_basis.ktx2").to_vec();
    let dfd = u32::from_le_bytes(source[48..52].try_into().unwrap()) as usize;
    source[dfd + 31] = 3;
    source[dfd + 47] = 4;
    source[dfd + 14] = 1;
    let linear =
        TextureDecoder::decode_with_support(TextureSource::Ktx2, &source, CompressionSupport::BC)
            .unwrap();
    source[dfd + 14] = 2;
    let srgb =
        TextureDecoder::decode_with_support(TextureSource::Ktx2, &source, CompressionSupport::BC)
            .unwrap();
    assert_eq!(linear.format, TextureFormat::Rgba8Unorm);
    assert_eq!(srgb.format, TextureFormat::Rgba8Srgb);
    assert_eq!(linear.mips, srgb.mips);
}

#[cfg(all(feature = "basis", feature = "ktx2"))]
#[test]
fn uastc_data_channel_layouts_preserve_logical_rg_values() {
    use ez_gfx_core::capability::CompressionSupport;
    let original = include_bytes!("fixtures/cube-uastc-srgb.ktx2");
    let reference = TextureDecoder::decode(TextureSource::Ktx2, original).unwrap();
    let dfd = u32::from_le_bytes(original[48..52].try_into().unwrap()) as usize;
    for channel in [4_u8, 5, 6] {
        let mut source = original.to_vec();
        source[dfd + 31] = channel;
        let decoded = TextureDecoder::decode_with_support(
            TextureSource::Ktx2,
            &source,
            CompressionSupport::BC,
        )
        .unwrap();
        for (mip, reference) in decoded.mips.iter().zip(&reference.mips) {
            for (pixel, prior) in mip
                .bytes
                .chunks_exact(4)
                .zip(reference.bytes.chunks_exact(4))
            {
                let green = match channel {
                    4 => 0,
                    5 => prior[3],
                    _ => prior[1],
                };
                assert_eq!(pixel, [prior[0], green, 0, 255]);
            }
        }
    }
}

#[cfg(feature = "ktx2")]
#[test]
fn direct_ktx2_validates_geometry_and_exact_levels() {
    // Minimal direct RGBA8 KTX2: two tightly packed levels and an empty DFD section.
    let mut source = vec![0_u8; 152];
    source[..12].copy_from_slice(&[
        0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB, 13, 10, 26, 10,
    ]);
    for (offset, value) in [
        (12, 37_u32),
        (16, 1),
        (20, 2),
        (24, 2),
        (36, 1),
        (40, 2),
        (48, 128),
        (52, 4),
        (128, 4),
    ] {
        source[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (80, 132_u64),
        (88, 16),
        (96, 16),
        (104, 148),
        (112, 4),
        (120, 4),
    ] {
        source[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    source[132..148].fill(127);
    source[148..].copy_from_slice(&[1, 2, 3, 255]);
    let decoded = TextureDecoder::decode(TextureSource::Ktx2, &source).unwrap();
    assert_eq!(decoded.mips[1].bytes, [1, 2, 3, 255]);
    for (offset, value) in [(20, 0_u32), (20, 1), (24, 1), (112, 3)] {
        let mut malformed = source.clone();
        malformed[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(matches!(
            TextureDecoder::decode(TextureSource::Ktx2, &malformed),
            Err(TextureError::InvalidData | TextureError::TooLarge)
        ));
    }
    let mut malformed = source;
    malformed[20..24].copy_from_slice(&1_u32.to_le_bytes());
    malformed[24..28].copy_from_slice(&1_u32.to_le_bytes());
    assert_eq!(
        TextureDecoder::decode(TextureSource::Ktx2, &malformed),
        Err(TextureError::InvalidData)
    );
}
