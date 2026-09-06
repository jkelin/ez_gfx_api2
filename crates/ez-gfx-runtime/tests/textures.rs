//! Runtime integration and contract tests.

use ez_gfx_core::capability::CompressionSupport;
use ez_gfx_hal::{CompletionToken, QueueKind, TextureFormat};
use ez_gfx_runtime::texture::{
    DecodedMip, DecodedTexture, TextureDecoder, TextureDestination, TextureError, TextureEvent,
    TextureRegistry, TextureSource, TextureUploadTelemetry, coarse_to_fine_mip_levels,
    generate_mips, register_texture_decoder, unregister_texture_decoder,
};
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};
use std::sync::Arc;

#[path = "textures/dds.rs"]
mod dds;

#[test]
fn progressive_submission_orders_terminal_mip_before_finer_levels() {
    assert_eq!(
        coarse_to_fine_mip_levels(0).collect::<Vec<_>>(),
        Vec::<u32>::new()
    );
    assert_eq!(coarse_to_fine_mip_levels(1).collect::<Vec<_>>(), [0]);
    assert_eq!(
        coarse_to_fine_mip_levels(4).collect::<Vec<_>>(),
        [3, 2, 1, 0]
    );
}

#[test]
fn retired_texture_slots_are_not_reused_until_explicit_release() {
    let mut registry = TextureRegistry::new(2, 2).unwrap();
    let first = registry.begin_upload().unwrap();
    let second = registry.begin_upload().unwrap();

    registry.retire(first).unwrap();
    assert_eq!(
        registry.begin_upload(),
        Err(TextureError::CapacityExceeded),
        "retirement must keep the descriptor binding unavailable"
    );
    registry.release_retired(first).unwrap();

    let reused = registry.begin_upload().unwrap();
    assert_eq!(registry.reserved_binding(reused), Ok(0));
    assert_ne!(reused, first);
    assert_eq!(registry.reserved_binding(second), Ok(1));
}

#[test]
fn upload_telemetry_snapshots_are_atomic_and_saturating() {
    let telemetry = TextureUploadTelemetry::default();
    telemetry.record_decode(u64::MAX);
    telemetry.record_decode(1);
    telemetry.record_staging_bytes(17);
    telemetry.record_queue_latency(23);
    telemetry.record_handoff_latency(29);

    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.decode_microseconds, u64::MAX);
    assert_eq!(snapshot.staging_bytes, 17);
    assert_eq!(snapshot.queue_latency_microseconds, 23);
    assert_eq!(snapshot.handoff_latency_microseconds, 29);
}

#[test]
fn raw_and_png_decode_to_bounded_rgba8() {
    let raw = TextureDecoder::decode(
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1, 2, 3, 4],
    )
    .unwrap();
    assert_eq!(
        raw,
        DecodedTexture {
            width: 1,
            height: 1,
            mip_count: 1,
            format: TextureFormat::Rgba8Unorm,
            mips: vec![DecodedMip {
                width: 1,
                height: 1,
                bytes: vec![1, 2, 3, 4]
            }]
        }
    );
    assert_eq!(
        TextureDecoder::decode(
            TextureSource::Rgba8 {
                width: 2,
                height: 1
            },
            &[0; 4]
        ),
        Err(TextureError::InvalidData)
    );

    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&[9, 8, 7, 6], 1, 1, ExtendedColorType::Rgba8)
        .unwrap();
    assert_eq!(
        TextureDecoder::decode(TextureSource::Png, &png)
            .unwrap()
            .mips[0]
            .bytes,
        vec![9, 8, 7, 6]
    );
    let rgb = TextureDecoder::decode(
        TextureSource::Rgb8 {
            width: 1,
            height: 1,
        },
        &[5, 6, 7],
    )
    .unwrap();
    assert_eq!(rgb.mips[0].bytes, vec![5, 6, 7, 255]);
}

#[test]
fn generated_mips_box_filter_odd_edges_and_stop_at_one_pixel() {
    let base = DecodedTexture {
        width: 3,
        height: 1,
        mip_count: 1,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![DecodedMip {
            width: 3,
            height: 1,
            bytes: vec![0, 0, 0, 0, 10, 20, 30, 40, 100, 120, 140, 160],
        }],
    };
    let generated = generate_mips(base).unwrap();
    assert_eq!(
        (
            generated.mip_count,
            generated.mips[1].width,
            generated.mips[1].height
        ),
        (2, 1, 1)
    );
    // Area-weighted coverage averages all three texels, not just the first two.
    assert_eq!(generated.mips[1].bytes, vec![37, 47, 57, 66]);

    let single = TextureDecoder::decode(
        TextureSource::Rgba8 {
            width: 1,
            height: 1,
        },
        &[1, 2, 3, 4],
    )
    .unwrap();
    assert_eq!(generate_mips(single.clone()).unwrap(), single);
}

#[test]
fn mip_validation_accepts_a_complete_terminal_chain() {
    let complete = DecodedTexture {
        width: 4,
        height: 2,
        mip_count: 3,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![
            DecodedMip {
                width: 4,
                height: 2,
                bytes: vec![0; 32],
            },
            DecodedMip {
                width: 2,
                height: 1,
                bytes: vec![0; 8],
            },
            DecodedMip {
                width: 1,
                height: 1,
                bytes: vec![0; 4],
            },
        ],
    };

    assert_eq!(generate_mips(complete.clone()), Ok(complete));
}

#[test]
fn mip_validation_rejects_levels_after_the_terminal_texel() {
    let repeated_terminal = DecodedTexture {
        width: 1,
        height: 1,
        mip_count: 2,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![
            DecodedMip {
                width: 1,
                height: 1,
                bytes: vec![0; 4],
            },
            DecodedMip {
                width: 1,
                height: 1,
                bytes: vec![0; 4],
            },
        ],
    };

    assert_eq!(
        generate_mips(repeated_terminal),
        Err(TextureError::InvalidData)
    );
}

#[test]
fn generated_mips_cover_trailing_odd_rows_and_columns() {
    // 3x3 with a bright last row and column: every texel contributes.
    let mut base_bytes = Vec::with_capacity(3 * 3 * 4);
    for y in 0..3 {
        for x in 0..3 {
            let value = if x == 2 || y == 2 { 255 } else { 0 };
            base_bytes.extend_from_slice(&[value, value, value, 255]);
        }
    }
    let generated = generate_mips(DecodedTexture {
        width: 3,
        height: 3,
        mip_count: 1,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![DecodedMip {
            width: 3,
            height: 3,
            bytes: base_bytes,
        }],
    })
    .unwrap();
    assert_eq!(generated.mip_count, 2);
    // Five bright texels of nine, rounded to nearest: 1275 / 9 = 142.
    assert_eq!(generated.mips[1].bytes, vec![142, 142, 142, 255]);

    // 5x5 with a bright last row and column exercises every chain stage.
    let mut wide_bytes = Vec::with_capacity(5 * 5 * 4);
    for y in 0..5 {
        for x in 0..5 {
            let value = if x == 4 || y == 4 { 255 } else { 0 };
            wide_bytes.extend_from_slice(&[value, value, value, 255]);
        }
    }
    let wide = generate_mips(DecodedTexture {
        width: 5,
        height: 5,
        mip_count: 1,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![DecodedMip {
            width: 5,
            height: 5,
            bytes: wide_bytes,
        }],
    })
    .unwrap();
    assert_eq!(wide.mip_count, 3);
    assert_eq!((wide.mips[1].width, wide.mips[1].height), (2, 2));
    // Side windows each see two bright texels of six: 510 / 6 = 85; the
    // bottom-right 3x3 window sees five of nine: 1275 / 9 = 142.
    assert_eq!(
        wide.mips[1].bytes,
        vec![
            0, 0, 0, 255, 85, 85, 85, 255, 85, 85, 85, 255, 142, 142, 142, 255
        ]
    );
    // The terminal level averages the full 2x2 above: 312 / 4 = 78.
    assert_eq!(wide.mips[2].bytes, vec![78, 78, 78, 255]);
}

#[test]
fn generated_mips_filter_srgb_in_linear_light_with_linear_alpha() {
    // Two black and two white texels: linear mean 0.5 encodes to 188 sRGB.
    let generated = generate_mips(DecodedTexture {
        width: 2,
        height: 2,
        mip_count: 1,
        format: TextureFormat::Rgba8Srgb,
        mips: vec![DecodedMip {
            width: 2,
            height: 2,
            bytes: vec![
                0, 0, 0, 0, 0, 0, 0, 85, 255, 255, 255, 170, 255, 255, 255, 255,
            ],
        }],
    })
    .unwrap();
    // Alpha stays a straight linear mean: (0 + 85 + 170 + 255) / 4 = 127.
    assert_eq!(generated.mips[1].bytes, vec![188, 188, 188, 127]);
}

#[test]
fn generated_mips_cover_degenerate_single_texel_axes() {
    // 1x3 averages all three rows: (10 + 20 + 30) / 3 = 20.
    let generated = generate_mips(DecodedTexture {
        width: 1,
        height: 3,
        mip_count: 1,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![DecodedMip {
            width: 1,
            height: 3,
            bytes: vec![10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255],
        }],
    })
    .unwrap();
    assert_eq!((generated.mip_count, generated.mips[1].width), (2, 1));
    assert_eq!(generated.mips[1].bytes, vec![20, 0, 0, 255]);
}

#[test]
fn generated_mips_reject_compressed_input_and_over_budget_chains() {
    let compressed = DecodedTexture {
        width: 4,
        height: 4,
        mip_count: 1,
        format: TextureFormat::Bc1Unorm,
        mips: vec![DecodedMip {
            width: 4,
            height: 4,
            bytes: vec![0; 8],
        }],
    };
    assert_eq!(generate_mips(compressed), Err(TextureError::Unsupported));

    // A 3600x3600 base fits the input budget, but its generated chain exceeds
    // 64 MiB and must fail before any level allocation.
    let wide = DecodedTexture {
        width: 3600,
        height: 3600,
        mip_count: 1,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![DecodedMip {
            width: 3600,
            height: 3600,
            bytes: vec![0; 3600 * 3600 * 4],
        }],
    };
    assert_eq!(generate_mips(wide), Err(TextureError::TooLarge));
}

#[cfg(feature = "ktx2")]
#[test]
fn malformed_ktx2_is_rejected_without_panics() {
    assert_eq!(
        TextureDecoder::decode(TextureSource::Ktx2, b"bad"),
        Err(TextureError::InvalidData)
    );
}

#[cfg(feature = "basis")]
#[test]
fn standalone_basis_decodes_complete_fixture_chain() {
    let bytes = include_bytes!("fixtures/rust-logo-etc.basis");
    let decoded =
        TextureDecoder::decode_with_support(TextureSource::Basis, bytes, CompressionSupport::BC)
            .unwrap();

    assert_eq!(
        (decoded.width, decoded.height, decoded.mip_count),
        (64, 64, 7)
    );
    assert_eq!(decoded.format, TextureFormat::Bc3Unorm);
    assert!(decoded.mips.iter().all(|mip| {
        mip.bytes.len() as u64 == decoded.format.level_bytes(mip.width, mip.height).unwrap()
    }));
}

#[cfg(not(feature = "basis"))]
#[test]
fn standalone_basis_is_unsupported_without_feature() {
    assert_eq!(
        TextureDecoder::decode(TextureSource::Basis, b"basis"),
        Err(TextureError::Unsupported)
    );
}
#[test]
fn custom_decoder_registration_validates_output_and_unregisters_cleanly() {
    const FORMAT: u8 = 201;
    unregister_texture_decoder(FORMAT).ok();
    register_texture_decoder(
        FORMAT,
        Arc::new(|bytes, _| {
            Ok(DecodedTexture {
                width: 1,
                height: 1,
                mip_count: 1,
                format: TextureFormat::Rgba8Unorm,
                mips: vec![DecodedMip {
                    width: 1,
                    height: 1,
                    bytes: bytes.to_vec(),
                }],
            })
        }),
    )
    .unwrap();

    assert_eq!(
        TextureDecoder::decode(TextureSource::Custom(FORMAT), &[1, 2, 3, 4])
            .unwrap()
            .mips[0]
            .bytes,
        [1, 2, 3, 4]
    );
    assert_eq!(
        TextureDecoder::decode(TextureSource::Custom(FORMAT), &[1]),
        Err(TextureError::InvalidData)
    );
    unregister_texture_decoder(FORMAT).unwrap();
    assert_eq!(
        TextureDecoder::decode(TextureSource::Custom(FORMAT), &[1, 2, 3, 4]),
        Err(TextureError::Unsupported)
    );
}

#[test]
fn prepared_custom_decode_retains_callback_after_unregister() {
    const FORMAT: u8 = 202;
    unregister_texture_decoder(FORMAT).ok();
    register_texture_decoder(
        FORMAT,
        Arc::new(|bytes, _| {
            Ok(DecodedTexture {
                width: 1,
                height: 1,
                mip_count: 1,
                format: TextureFormat::Rgba8Unorm,
                mips: vec![DecodedMip {
                    width: 1,
                    height: 1,
                    bytes: bytes.to_vec(),
                }],
            })
        }),
    )
    .unwrap();
    let prepared = TextureDecoder::prepare(
        TextureSource::Custom(FORMAT),
        CompressionSupport::NONE,
        TextureDestination::Auto,
    )
    .unwrap();
    unregister_texture_decoder(FORMAT).unwrap();

    assert_eq!(prepared.decode(&[1, 2, 3, 4]).unwrap().width, 1);
}

#[test]
fn custom_compressed_output_requires_backend_admission() {
    const FORMAT: u8 = 203;
    unregister_texture_decoder(FORMAT).ok();
    register_texture_decoder(
        FORMAT,
        Arc::new(|_, _| {
            Ok(DecodedTexture {
                width: 4,
                height: 4,
                mip_count: 1,
                format: TextureFormat::Bc1Unorm,
                mips: vec![DecodedMip {
                    width: 4,
                    height: 4,
                    bytes: vec![0; 8],
                }],
            })
        }),
    )
    .unwrap();

    assert_eq!(
        TextureDecoder::decode_with_support(
            TextureSource::Custom(FORMAT),
            &[1],
            CompressionSupport::NONE,
        ),
        Err(TextureError::Unsupported)
    );
    unregister_texture_decoder(FORMAT).unwrap();
}

#[cfg(all(feature = "ktx2", feature = "basis"))]
#[test]
fn basis_transcoding_selects_native_bc_when_supported() {
    let bytes = include_bytes!("fixtures/alpha_simple_basis.ktx2");
    let decoded =
        TextureDecoder::decode_with_support(TextureSource::Ktx2, bytes, CompressionSupport::BC)
            .unwrap();

    assert!(matches!(
        decoded.format,
        TextureFormat::Bc3Unorm | TextureFormat::Bc3Srgb
    ));
    assert!(decoded.mips.iter().all(|mip| {
        mip.bytes.len() as u64 == decoded.format.level_bytes(mip.width, mip.height).unwrap()
    }));
}

#[cfg(all(feature = "ktx2", feature = "basis"))]
#[test]
fn basis_destination_is_explicit_and_capability_checked() {
    let bytes = include_bytes!("fixtures/alpha_simple_basis.ktx2");
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Ktx2,
            bytes,
            CompressionSupport::NONE,
            TextureDestination::Bc7Unorm,
        ),
        Err(TextureError::Unsupported)
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Ktx2,
            bytes,
            CompressionSupport::BC,
            TextureDestination::Rgba8Srgb,
        )
        .unwrap()
        .format,
        TextureFormat::Rgba8Srgb
    );
}

#[cfg(all(feature = "ktx2", feature = "basis"))]
#[test]
fn ktx2_basislz_transcodes_every_mip_to_rgba8() {
    let bytes = include_bytes!("fixtures/alpha_simple_basis.ktx2");
    let header = ktx2::Reader::new(bytes.as_slice()).unwrap().header();
    let decoded = TextureDecoder::decode(TextureSource::Ktx2, bytes).unwrap();
    assert_eq!(
        (decoded.width, decoded.height, decoded.mip_count),
        (
            header.pixel_width,
            header.pixel_height,
            header.level_count.max(1)
        )
    );
    assert_eq!(decoded.mips.len(), decoded.mip_count as usize);
    assert!(
        decoded
            .mips
            .iter()
            .all(|mip| mip.bytes.len() == mip.width as usize * mip.height as usize * 4)
    );
}

#[cfg(all(feature = "ktx2", feature = "basis"))]
#[test]
fn ktx2_honors_every_explicit_native_target() {
    let bytes = include_bytes!("fixtures/alpha_simple_basis.ktx2");
    for (destination, expected) in [
        (TextureDestination::Bc1Unorm, TextureFormat::Bc1Unorm),
        (TextureDestination::Bc3Srgb, TextureFormat::Bc3Srgb),
        (TextureDestination::Bc7Unorm, TextureFormat::Bc7Unorm),
        (TextureDestination::Astc4x4Srgb, TextureFormat::Astc4x4Srgb),
        (TextureDestination::Rgba8Srgb, TextureFormat::Rgba8Srgb),
    ] {
        let texture = TextureDecoder::decode_for_destination(
            TextureSource::Ktx2,
            bytes,
            CompressionSupport::BC.union(CompressionSupport::ASTC),
            destination,
        )
        .unwrap();
        assert_eq!(texture.format, expected);
        for mip in texture.mips {
            assert_eq!(
                mip.bytes.len() as u64,
                expected.level_bytes(mip.width, mip.height).unwrap()
            );
        }
    }
}

#[cfg(all(feature = "ktx2", feature = "basis"))]
#[test]
fn etc1s_data_channels_decode_to_canonical_rgba_pixels() {
    let original = include_bytes!("fixtures/alpha_simple_basis.ktx2");
    let reference = TextureDecoder::decode(TextureSource::Ktx2, original).unwrap();
    let dfd = u32::from_le_bytes(original[48..52].try_into().unwrap()) as usize;
    for second_channel in [4_u8, 15] {
        let mut source = original.to_vec();
        source[dfd + 31] = 3; // ETC1S RRR primary plane.
        source[dfd + 47] = second_channel; // GGG means RG; otherwise a single R data channel.
        source[dfd + 14] = 1; // Linear data, not an sRGB color.
        for destination in [TextureDestination::Auto, TextureDestination::Rgba8Unorm] {
            let decoded = TextureDecoder::decode_for_destination(
                TextureSource::Ktx2,
                &source,
                CompressionSupport::BC,
                destination,
            )
            .unwrap();
            assert_eq!(decoded.format, TextureFormat::Rgba8Unorm);
            for (actual, prior) in decoded.mips.iter().zip(&reference.mips) {
                for (pixel, encoded) in actual
                    .bytes
                    .chunks_exact(4)
                    .zip(prior.bytes.chunks_exact(4))
                {
                    assert_eq!(
                        pixel,
                        [
                            encoded[0],
                            if second_channel == 4 { encoded[3] } else { 0 },
                            0,
                            255
                        ]
                    );
                }
            }
        }
        assert_eq!(
            TextureDecoder::decode_for_destination(
                TextureSource::Ktx2,
                &source,
                CompressionSupport::BC,
                TextureDestination::Bc3Unorm,
            ),
            Err(TextureError::Unsupported)
        );
    }
}

#[cfg(not(feature = "ktx2"))]
#[test]
fn disabled_ktx2_is_explicitly_unsupported() {
    assert_eq!(
        TextureDecoder::decode(TextureSource::Ktx2, b"disabled"),
        Err(TextureError::Unsupported)
    );
}

#[test]
fn polling_reports_not_ready_until_transfer_tokens_reach_ready() {
    let mut registry = TextureRegistry::new(2, 4).unwrap();
    let texture = registry.begin_upload().unwrap();
    let ready = CompletionToken::new(QueueKind::Transfer, 3).unwrap();
    registry.mark_submitted(texture, ready).unwrap();
    registry
        .mark_mips_submitted(
            texture,
            2,
            CompletionToken::new(QueueKind::Transfer, 5).unwrap(),
        )
        .unwrap();
    assert_eq!(registry.binding_index(texture), Err(TextureError::NotReady));
    assert_eq!(registry.poll(QueueKind::Transfer, 2).unwrap(), 0);
    assert_eq!(registry.binding_index(texture), Err(TextureError::NotReady));
    assert_eq!(registry.poll(QueueKind::Transfer, 3).unwrap(), 1);
    assert_eq!(registry.binding_index(texture), Ok(0));
    assert_eq!(
        registry.drain_events(),
        vec![TextureEvent::Resident {
            texture,
            binding: 0,
            resident_mips: 1
        }]
    );
    assert_eq!(registry.poll(QueueKind::Transfer, 4).unwrap(), 0);
    assert_eq!(registry.resident_mips(texture), Ok(1));
    assert_eq!(registry.poll(QueueKind::Transfer, 5).unwrap(), 1);
    assert_eq!(registry.resident_mips(texture), Ok(2));
    assert_eq!(
        registry.drain_events(),
        vec![TextureEvent::Resident {
            texture,
            binding: 0,
            resident_mips: 2
        }]
    );
    registry.unload(texture).unwrap();
    assert_eq!(
        registry.drain_events(),
        vec![TextureEvent::Unloaded { texture }]
    );
    assert_eq!(registry.binding_index(texture), Err(TextureError::NotFound));
}

#[test]
fn cancel_upload_reuses_binding_without_emitting_public_events() {
    let mut registry = TextureRegistry::new(1, 1).unwrap();
    let failed = registry.begin_upload().unwrap();
    registry.cancel_upload(failed).unwrap();
    assert!(registry.drain_events().is_empty());

    let replacement = registry.begin_upload().unwrap();
    assert_ne!(replacement, failed);
    assert_eq!(registry.binding_index(failed), Err(TextureError::NotFound));
    assert_eq!(
        registry.cancel_upload(failed),
        Err(TextureError::InvalidState)
    );
    assert_eq!(registry.begin_upload(), Err(TextureError::CapacityExceeded));
}

#[test]
fn clear_invalidates_every_slot_and_rebuilds_the_free_list() {
    let mut registry = TextureRegistry::new(3, 3).unwrap();
    let resident = registry.begin_upload().unwrap();
    let allocated = registry.begin_upload().unwrap();
    let vacant = registry.begin_upload().unwrap();
    registry.cancel_upload(vacant).unwrap();
    let ready = CompletionToken::new(QueueKind::Transfer, 1).unwrap();
    registry.mark_submitted(resident, ready).unwrap();
    registry.poll(QueueKind::Transfer, 1).unwrap();

    registry.clear().unwrap();

    assert!(registry.drain_events().is_empty());
    for stale in [resident, allocated, vacant] {
        assert_eq!(registry.binding_index(stale), Err(TextureError::NotFound));
    }
    let replacements = [
        registry.begin_upload().unwrap(),
        registry.begin_upload().unwrap(),
        registry.begin_upload().unwrap(),
    ];
    assert_eq!(
        replacements
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert_eq!(registry.begin_upload(), Err(TextureError::CapacityExceeded));
}

#[test]
fn raw_ingestion_preserves_and_matches_exactly() {
    let level = [0x66_u8; 16];
    let source = TextureSource::Raw {
        format: TextureFormat::Bc7Unorm,
        width: 4,
        height: 4,
        mip_count: 1,
    };
    let decoded = TextureDecoder::decode_for_destination(
        source,
        &level,
        CompressionSupport::BC,
        TextureDestination::Auto,
    )
    .unwrap();
    assert_eq!(decoded.format, TextureFormat::Bc7Unorm);
    assert_eq!(decoded.mips[0].bytes, level.to_vec());
    // A zero count cannot bound the chain and is rejected outright.
    let defaulted = TextureSource::Raw {
        format: TextureFormat::Bc7Unorm,
        width: 4,
        height: 4,
        mip_count: 0,
    };
    assert_eq!(
        TextureDecoder::decode_for_destination(
            defaulted,
            &level,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            source,
            &level,
            CompressionSupport::BC,
            TextureDestination::Bc3Unorm,
        ),
        Err(TextureError::Unsupported)
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            source,
            &level[..15],
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    let mut trailing = level.to_vec();
    trailing.push(0);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            source,
            &trailing,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            source,
            &level,
            CompressionSupport::NONE,
            TextureDestination::Auto,
        ),
        Err(TextureError::Unsupported)
    );
}

#[test]
fn raw_rejects_zero_dims_absurd_counts_and_dimension_drift() {
    let zero = TextureSource::Raw {
        format: TextureFormat::Rgba8Unorm,
        width: 0,
        height: 4,
        mip_count: 1,
    };
    assert_eq!(
        TextureDecoder::decode_for_destination(
            zero,
            &[0; 16],
            CompressionSupport::NONE,
            TextureDestination::Auto,
        ),
        Err(TextureError::TooLarge)
    );
    let absurd = TextureSource::Raw {
        format: TextureFormat::Rgba8Unorm,
        width: 1,
        height: 1,
        mip_count: 33,
    };
    assert_eq!(
        TextureDecoder::decode_for_destination(
            absurd,
            &[0; 4],
            CompressionSupport::NONE,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // Two declared levels require both byte ranges back to back.
    let chained = TextureSource::Raw {
        format: TextureFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        mip_count: 2,
    };
    let bytes = [1_u8; 16 + 4];
    let decoded = TextureDecoder::decode_for_destination(
        chained,
        &bytes,
        CompressionSupport::NONE,
        TextureDestination::Auto,
    )
    .unwrap();
    assert_eq!(decoded.mip_count, 2);
    assert_eq!(
        decoded
            .mips
            .iter()
            .map(|mip| (mip.width, mip.height))
            .collect::<Vec<_>>(),
        [(2, 2), (1, 1)]
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            chained,
            &[1_u8; 16],
            CompressionSupport::NONE,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
}
