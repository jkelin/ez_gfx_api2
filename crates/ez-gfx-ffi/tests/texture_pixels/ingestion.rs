use super::*;

pub(super) fn direct(
    context: ContextHandle,
    quad: &Quad,
    format: TextureFormat,
    config: &TextureConfig,
    bytes: &[u8],
    expected: &[u8],
) {
    // Raw carries native format metadata; no custom decoder participates in this admission.
    let texture = load_texture(
        context,
        TextureSource::Raw {
            format,
            width: config.width,
            height: config.height,
            mip_count: 1,
        },
        bytes,
        false,
        config,
    )
    .unwrap();
    await_texture(context, texture);
    quad.draw(texture, true);
    assert_eq!(
        frame_readback(context).unwrap(),
        expected,
        "{format:?} raw ingestion pixels"
    );
    unload_texture(context, texture);

    if format == TextureFormat::Bc1Unorm {
        // Minimal legacy DDS: one 8x4 DXT1 level, no array, volume, or cubemap flags.
        let mut words = [0_u32; 31];
        words[0] = 124;
        words[1] = 0x1 | 0x2 | 0x4 | 0x1000;
        words[2] = config.height;
        words[3] = config.width;
        words[18] = 32;
        words[19] = 0x4;
        words[20] = u32::from_le_bytes(*b"DXT1");
        words[26] = 0x1000;
        let mut dds = Vec::from(*b"DDS ");
        dds.extend(words.into_iter().flat_map(u32::to_le_bytes));
        dds.extend_from_slice(bytes);
        let texture = load_texture(context, TextureSource::Dds, &dds, false, config).unwrap();
        await_texture(context, texture);
        quad.draw(texture, true);
        assert_eq!(
            frame_readback(context).unwrap(),
            expected,
            "DDS ingestion pixels"
        );
        unload_texture(context, texture);
    }
}

#[cfg(feature = "basis")]
pub(super) fn universal(context: ContextHandle, quad: &Quad) {
    // Both standalone payload encodings must reach native sampling through canonical dispatch.
    for bytes in [
        include_bytes!("../../../ez-gfx-runtime/tests/fixtures/rust-logo-etc.basis").as_slice(),
        include_bytes!("../../../ez-gfx-runtime/tests/fixtures/cube-uastc-srgb.basis").as_slice(),
    ] {
        universal_case(context, quad, TextureSource::Basis, bytes);
    }
    #[cfg(feature = "ktx2")]
    for bytes in [
        include_bytes!("../../../ez-gfx-runtime/tests/fixtures/alpha_simple_basis.ktx2").as_slice(),
        include_bytes!("../../../ez-gfx-runtime/tests/fixtures/cube-uastc-srgb.ktx2").as_slice(),
    ] {
        universal_case(context, quad, TextureSource::Ktx2, bytes);
    }
}

#[cfg(feature = "basis")]
fn universal_case(context: ContextHandle, quad: &Quad, source: TextureSource, bytes: &[u8]) {
    use ez_gfx_core::capability::CompressionSupport;
    use ez_gfx_runtime::texture::TextureDecoder;

    for (rgba, destinations) in [
        (
            TextureDestination::Rgba8Unorm,
            [
                TextureDestination::Bc7Unorm,
                TextureDestination::Astc4x4Unorm,
            ],
        ),
        (
            TextureDestination::Rgba8Srgb,
            [TextureDestination::Bc7Srgb, TextureDestination::Astc4x4Srgb],
        ),
    ] {
        // Decode independently to RGBA, then sample that reference with identical UV and transfer
        // semantics. Comparison tolerates lossy block encoding, never orientation or blank output.
        let decoded =
            TextureDecoder::decode_for_destination(source, bytes, CompressionSupport::NONE, rgba)
                .unwrap();
        let mut config = TextureConfig {
            width: decoded.width,
            height: decoded.height,
            mip_count: decoded.mip_count,
            destination: rgba,
            sampler: TextureSamplerDesc {
                min_filter: SamplerFilter::Nearest,
                mag_filter: SamplerFilter::Nearest,
                max_anisotropy: 1.0,
                address_u: SamplerAddressMode::Clamp,
                address_v: SamplerAddressMode::Clamp,
                address_w: SamplerAddressMode::Clamp,
            },
        };
        let rgba_bytes: Vec<u8> = decoded
            .mips
            .iter()
            .flat_map(|mip| mip.bytes.iter().copied())
            .collect();
        let reference = load_texture(
            context,
            TextureSource::Raw {
                format: decoded.format,
                width: decoded.width,
                height: decoded.height,
                mip_count: decoded.mip_count,
            },
            &rgba_bytes,
            false,
            &config,
        )
        .unwrap();
        await_texture(context, reference);
        // Initial readiness exposes only the coarse tail; image comparison requires every mip.
        assert_eq!(wait_idle(context), EzGfxResult::Ok);
        assert_eq!(
            texture_residency(context, reference),
            Ok((decoded.mip_count, decoded.mip_count))
        );
        quad.draw(reference, true);
        let expected = frame_readback(context).unwrap();
        unload_texture(context, reference);

        for destination in destinations {
            config.destination = destination;
            let texture = load_texture(context, source, bytes, false, &config).unwrap();
            await_texture(context, texture);
            assert_eq!(wait_idle(context), EzGfxResult::Ok);
            assert_eq!(
                texture_residency(context, texture),
                Ok((decoded.mip_count, decoded.mip_count))
            );
            quad.draw(texture, true);
            let actual = frame_readback(context).unwrap();
            assert_eq!(actual.len(), expected.len());
            let mut error = [0_u64; 4];
            for (actual, expected) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
                for channel in 0..4 {
                    error[channel] += u64::from(actual[channel].abs_diff(expected[channel]));
                }
            }
            // Average per-channel error is bounded to 8/255 over the entire sampled image.
            // This covers lossy BC7/ASTC differences while rejecting wrong colors and alpha.
            for (channel, error) in error.into_iter().enumerate() {
                assert!(
                    error <= 8 * 64 * 64,
                    "{source:?} {destination:?} channel {channel} total error {error}"
                );
            }
            quad.draw(texture, false);
            unload_texture(context, texture);
            assert_stale(context, texture, &[0; 16]);
            assert_eq!(wait_idle(context), EzGfxResult::Ok);
        }
    }
}
#[cfg(target_vendor = "apple")]
#[test]
fn metal_minified_sampling_selects_published_mips() {
    // A 256px texture on the 64px hidden surface minifies 4:1, forcing mip
    // selection. mip0 is white and coarser levels are black: without a mip
    // filter every fragment would sample the white base level.
    let artifact = compile_shader(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/02_textured_cube/02_textured_cube.slang"),
        &[Target::Metal],
        false,
    )
    .expect("compile cube shader artifact");
    let native = TestContext::create_with_validation(3, false);
    let context = ContextHandle::from_raw(native.context).unwrap();
    let surface = SurfaceHandle::from_raw(native.surface).unwrap();
    let quad = Quad::create(context, surface, &artifact);
    let white = [255_u8, 255, 255, 255].repeat(256 * 256);
    let black = [0_u8, 0, 0, 255].repeat(128 * 128);
    let mut bytes = white.clone();
    bytes.extend_from_slice(&black);
    bytes.extend_from_slice(&[0_u8, 0, 0, 255].repeat(64 * 64));
    let config = TextureConfig {
        width: 256,
        height: 256,
        mip_count: 3,
        destination: TextureDestination::Rgba8Unorm,
        sampler: TextureSamplerDesc {
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            max_anisotropy: 1.0,
            address_u: SamplerAddressMode::Clamp,
            address_v: SamplerAddressMode::Clamp,
            address_w: SamplerAddressMode::Clamp,
        },
    };
    let texture = load_texture(
        context,
        TextureSource::Raw {
            format: TextureFormat::Rgba8Unorm,
            width: 256,
            height: 256,
            mip_count: 3,
        },
        &bytes,
        false,
        &config,
    )
    .expect("admit minification chain");
    await_texture(context, texture);
    // `await` only guarantees the coarse view: force the full chain resident so
    // the draw truly minifies across mip levels instead of sampling one level.
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
    assert_eq!(set_texture_residency(context, texture, 3), EzGfxResult::Ok);
    assert_eq!(texture_residency(context, texture), Ok((3, 3)));
    quad.draw(texture, true);
    let sampled = frame_readback(context).unwrap();
    assert_eq!(sampled.len(), 64 * 64 * 4);
    let mean: f64 = sampled
        .chunks_exact(4)
        .map(|pixel| f64::from(pixel[0]))
        .sum::<f64>()
        / (64.0 * 64.0);
    // Nearest mip selection at 4:1 minification reads a black coarse level.
    assert!(mean < 64.0, "minified mean {mean} sampled the base level");
    // A demoted RGBA view exposes a smaller extent: direct capture waits for
    // the full chain, then returns the white base level byte-for-byte.
    assert_eq!(set_texture_residency(context, texture, 1), EzGfxResult::Ok);
    assert_eq!(
        frame_enqueue_readback(context, texture),
        EzGfxResult::NotReady
    );
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
    assert_eq!(set_texture_residency(context, texture, 3), EzGfxResult::Ok);
    assert_eq!(quad.readback_direct(texture), white);
    // A submitted frame gates publication, but its completion must unblock
    // polling on its own: the shared path reaps settled slots before consulting
    // the gate, so no wait_idle round-trip is required here.
    quad.draw(texture, false);
    let pending = load_texture(
        context,
        TextureSource::Rgba8 {
            width: 64,
            height: 64,
        },
        &[128_u8, 128, 128, 255].repeat(64 * 64),
        false,
        &TextureConfig {
            width: 64,
            height: 64,
            mip_count: 0,
            destination: TextureDestination::Rgba8Unorm,
            sampler: config.sampler,
        },
    )
    .expect("admit reap probe");
    // Settled-slot reaping must unblock this with polling alone.
    await_texture(context, pending);
    unload_texture(context, pending);
    unload_texture(context, texture);
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
}
