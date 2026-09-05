//! DDS container integration and contract tests.

use super::*;

fn dds_header(
    width: u32,
    height: u32,
    mip_count: u32,
    pf_flags: u32,
    four_cc: u32,
    caps2: u32,
    depth: u32,
) -> Vec<u8> {
    let mut header = Vec::with_capacity(128);
    header.extend_from_slice(b"DDS ");
    let mut words = [0_u32; 31];
    words[0] = 124;
    words[1] = 0x1 | 0x2 | 0x4 | 0x1000;
    if mip_count > 1 {
        words[1] |= 0x2_0000;
    }
    words[2] = height;
    words[3] = width;
    words[5] = depth;
    words[6] = mip_count;
    words[18] = 32;
    words[19] = pf_flags;
    words[20] = four_cc;
    words[26] = 0x1000;
    words[27] = caps2;
    for word in words {
        header.extend_from_slice(&word.to_le_bytes());
    }
    header
}

fn dxt1_file(width: u32, height: u32, mip_count: u32, payload: &[u8]) -> Vec<u8> {
    let four_cc = u32::from_le_bytes(*b"DXT1");
    let mut file = dds_header(width, height, mip_count, 0x4, four_cc, 0, 0);
    file.extend_from_slice(payload);
    file
}

#[test]
fn dds_dxt1_decodes_and_preserves_format_under_auto() {
    let blocks = [0x11_u8; 8];
    let file = dxt1_file(4, 4, 1, &blocks);
    let decoded = TextureDecoder::decode_for_destination(
        TextureSource::Dds,
        &file,
        CompressionSupport::BC,
        TextureDestination::Auto,
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedTexture {
            width: 4,
            height: 4,
            mip_count: 1,
            format: TextureFormat::Bc1Unorm,
            mips: vec![DecodedMip {
                width: 4,
                height: 4,
                bytes: blocks.to_vec(),
            }],
        }
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &file,
            CompressionSupport::BC,
            TextureDestination::Bc1Unorm,
        )
        .unwrap()
        .format,
        TextureFormat::Bc1Unorm
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &file,
            CompressionSupport::BC,
            TextureDestination::Bc3Unorm,
        ),
        Err(TextureError::Unsupported)
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &file,
            CompressionSupport::NONE,
            TextureDestination::Auto,
        ),
        Err(TextureError::Unsupported)
    );
}

#[test]
fn dds_rejects_malformed_truncated_and_container_violations() {
    let blocks = [0x22_u8; 8];
    let valid = dxt1_file(4, 4, 1, &blocks);
    let mut bad_magic = valid.clone();
    bad_magic[0] = b'X';
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &bad_magic,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &valid[..valid.len() - 1],
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    let mut trailing = valid.clone();
    trailing.push(0);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &trailing,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // Cubemap face bit selects container storage outside the 2D contract.
    let four_cc = u32::from_le_bytes(*b"DXT1");
    let mut cube = dds_header(4, 4, 1, 0x4, four_cc, 0x200 | 0x400, 0);
    cube.extend_from_slice(&blocks);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &cube,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::Unsupported)
    );
    // Depth selects a volume texture outside the 2D contract.
    let mut volume = dds_header(4, 4, 1, 0x4, four_cc, 0, 2);
    volume.extend_from_slice(&blocks);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &volume,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::Unsupported)
    );
    // DXT3 names BC2 explicit-alpha blocks, which have no native storage enum.
    let dxt3 = u32::from_le_bytes(*b"DXT3");
    let mut dxt3_file = dds_header(4, 4, 1, 0x4, dxt3, 0, 0);
    dxt3_file.extend_from_slice(&[0x33_u8; 16]);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &dxt3_file,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::Unsupported)
    );
    // Absurd level counts fail before any allocation attempt.
    let overflow = dxt1_file(4, 4, 33, &blocks);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &overflow,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // Zero dimensions cannot address texels.
    let empty = dxt1_file(0, 4, 1, &blocks);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &empty,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::TooLarge)
    );
}

#[test]
fn dds_dx10_bc7_srgb_decodes() {
    let mut dx10 = dds_header(4, 4, 1, 0x4, u32::from_le_bytes(*b"DX10"), 0, 0);
    for word in [99_u32, 3, 0, 1, 0] {
        dx10.extend_from_slice(&word.to_le_bytes());
    }
    dx10.extend_from_slice(&[0x44_u8; 16]);
    let decoded = TextureDecoder::decode_for_destination(
        TextureSource::Dds,
        &dx10,
        CompressionSupport::BC,
        TextureDestination::Auto,
    )
    .unwrap();
    assert_eq!(decoded.format, TextureFormat::Bc7Srgb);
    // Auto preserves the container; an explicit linear request mismatches.
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &dx10,
            CompressionSupport::BC,
            TextureDestination::Bc7Unorm,
        ),
        Err(TextureError::Unsupported)
    );
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &dx10,
            CompressionSupport::BC,
            TextureDestination::Bc7Srgb,
        )
        .unwrap()
        .format,
        TextureFormat::Bc7Srgb
    );
}

#[test]
fn dds_odd_edges_round_up_to_block_extents() {
    // 5x3 needs 2x1 blocks (16 bytes); the chain continues to 2x1 then 1x1.
    let payload = [0x55_u8; 16 + 8 + 8];
    let file = dxt1_file(5, 3, 3, &payload);
    let decoded = TextureDecoder::decode_for_destination(
        TextureSource::Dds,
        &file,
        CompressionSupport::BC,
        TextureDestination::Auto,
    )
    .unwrap();
    assert_eq!(
        decoded
            .mips
            .iter()
            .map(|mip| (mip.width, mip.height, mip.bytes.len()))
            .collect::<Vec<_>>(),
        [(5, 3, 16), (2, 1, 8), (1, 1, 8)]
    );
}

#[test]
fn dds_rejects_flag_count_pitch_and_alpha_disagreement() {
    let blocks = [0x77_u8; 8];
    // Count field without its flag is malformed, even for a physical chain.
    let mut unflagged = dxt1_file(8, 8, 2, &[0x77_u8; 32 + 8]);
    let flag_offset = 4 + 4;
    let flags = u32::from_le_bytes(unflagged[flag_offset..flag_offset + 4].try_into().unwrap());
    unflagged[flag_offset..flag_offset + 4].copy_from_slice(&(flags & !0x2_0000).to_le_bytes());
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &unflagged,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // A FourCC without its present flag is untrusted.
    let mut unflagged_fourcc = dxt1_file(4, 4, 1, &blocks);
    let pf_offset = 4 + 76;
    let pf = u32::from_le_bytes(
        unflagged_fourcc[pf_offset..pf_offset + 4]
            .try_into()
            .unwrap(),
    );
    unflagged_fourcc[pf_offset..pf_offset + 4].copy_from_slice(&(pf & !0x4).to_le_bytes());
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &unflagged_fourcc,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // Pitch selects unpacked row layouts, which are never ingested.
    let mut pitched = dxt1_file(4, 4, 1, &blocks);
    let flag_offset = 4 + 4;
    let flags = u32::from_le_bytes(pitched[flag_offset..flag_offset + 4].try_into().unwrap()) | 0x8;
    pitched[flag_offset..flag_offset + 4].copy_from_slice(&flags.to_le_bytes());
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &pitched,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // A linear size must name exactly the base level byte count.
    let mut sized = dxt1_file(4, 4, 1, &blocks);
    let flags =
        u32::from_le_bytes(sized[flag_offset..flag_offset + 4].try_into().unwrap()) | 0x8_0000;
    sized[flag_offset..flag_offset + 4].copy_from_slice(&flags.to_le_bytes());
    let pitch_offset = 4 + 16;
    sized[pitch_offset..pitch_offset + 4].copy_from_slice(&7_u32.to_le_bytes());
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &sized,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // Unknown DX10 alpha modes are closed, not defaulted.
    let mut dx10 = dds_header(4, 4, 1, 0x4, u32::from_le_bytes(*b"DX10"), 0, 0);
    for word in [98_u32, 3, 0, 1, 7] {
        dx10.extend_from_slice(&word.to_le_bytes());
    }
    dx10.extend_from_slice(&[0x88_u8; 16]);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &dx10,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
}

#[test]
fn dds_and_raw_reject_unphysical_counts_and_over_budget_chains() {
    // A 4x4 pyramid holds 3 levels; a fourth overruns the physical chain.
    let overlong = dxt1_file(4, 4, 4, &[0x99_u8; 8 + 8 + 8]);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &overlong,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    // A 16384x16384 BC1 base level alone exceeds the 64 MiB budget, with no
    // payload allocated or copied before the rejection.
    let huge = dxt1_file(16384, 16384, 1, &[]);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &huge,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::TooLarge)
    );
    let raw_overlong = TextureSource::Raw {
        format: TextureFormat::Bc1Unorm,
        width: 4,
        height: 4,
        mip_count: 4,
    };
    assert_eq!(
        TextureDecoder::decode_for_destination(
            raw_overlong,
            &[0x99_u8; 8 + 8 + 8],
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
    let raw_huge = TextureSource::Raw {
        format: TextureFormat::Bc1Unorm,
        width: 16384,
        height: 16384,
        mip_count: 1,
    };
    assert_eq!(
        TextureDecoder::decode_for_destination(
            raw_huge,
            &[0],
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::TooLarge)
    );
}

#[test]
fn dds_count_flag_agrees_with_count_field() {
    let blocks = [0xAA_u8; 8];
    // A set flag with a count of one is valid.
    let mut flagged_single = dxt1_file(4, 4, 1, &blocks);
    let flag_offset = 4 + 4;
    let flags = u32::from_le_bytes(
        flagged_single[flag_offset..flag_offset + 4]
            .try_into()
            .unwrap(),
    );
    flagged_single[flag_offset..flag_offset + 4].copy_from_slice(&(flags | 0x2_0000).to_le_bytes());
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &flagged_single,
            CompressionSupport::BC,
            TextureDestination::Auto,
        )
        .unwrap()
        .mip_count,
        1
    );
    // A set flag with a zero count is malformed.
    let mut flagged_zero = dds_header(4, 4, 0, 0x4, u32::from_le_bytes(*b"DXT1"), 0, 0);
    let flags = u32::from_le_bytes(
        flagged_zero[flag_offset..flag_offset + 4]
            .try_into()
            .unwrap(),
    );
    flagged_zero[flag_offset..flag_offset + 4].copy_from_slice(&(flags | 0x2_0000).to_le_bytes());
    flagged_zero.extend_from_slice(&blocks);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &flagged_zero,
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
}

#[test]
fn dds_rejects_later_mip_truncation_and_trailing_payload() {
    // Two declared levels need 8 + 8 bytes; the full chain decodes, while
    // cutting inside the second level fails even though the first is complete.
    let full = dxt1_file(4, 4, 2, &[0xBB_u8; 8 + 8]);
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &full,
            CompressionSupport::BC,
            TextureDestination::Auto,
        )
        .unwrap()
        .mip_count,
        2
    );
    for end in [128 + 8, 128 + 9, 128 + 15] {
        let head = full[..end].to_vec();
        assert_eq!(
            TextureDecoder::decode_for_destination(
                TextureSource::Dds,
                &head,
                CompressionSupport::BC,
                TextureDestination::Auto,
            ),
            Err(TextureError::InvalidData),
            "truncated at payload offset {}",
            end - 128
        );
    }
}

#[test]
fn dds_accepts_tightly_packed_native_pitch_for_dx10_rgba() {
    fn dx10_rgba(pitch: Option<u32>) -> Vec<u8> {
        let mut file = dds_header(4, 4, 1, 0x4, u32::from_le_bytes(*b"DX10"), 0, 0);
        for word in [28_u32, 3, 0, 1, 0] {
            file.extend_from_slice(&word.to_le_bytes());
        }
        if let Some(pitch) = pitch {
            let flag_offset = 4 + 4;
            let flags =
                u32::from_le_bytes(file[flag_offset..flag_offset + 4].try_into().unwrap()) | 0x8;
            file[flag_offset..flag_offset + 4].copy_from_slice(&flags.to_le_bytes());
            let pitch_offset = 4 + 16;
            file[pitch_offset..pitch_offset + 4].copy_from_slice(&pitch.to_le_bytes());
        }
        file.extend_from_slice(&[0xCC_u8; 64]);
        file
    }
    // No pitch flag and the exact native stride both decode.
    for file in [dx10_rgba(None), dx10_rgba(Some(16))] {
        assert_eq!(
            TextureDecoder::decode_for_destination(
                TextureSource::Dds,
                &file,
                CompressionSupport::NONE,
                TextureDestination::Auto,
            )
            .unwrap()
            .format,
            TextureFormat::Rgba8Unorm
        );
    }
    // Any other stride disagrees with the tightly packed layout.
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &dx10_rgba(Some(15)),
            CompressionSupport::NONE,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
}

#[test]
fn dds_alpha_policy_retains_straight_rejects_premultiplied() {
    fn dx10_bc7(alpha: u32) -> Vec<u8> {
        let mut file = dds_header(4, 4, 1, 0x4, u32::from_le_bytes(*b"DX10"), 0, 0);
        for word in [98_u32, 3, 0, 1, alpha] {
            file.extend_from_slice(&word.to_le_bytes());
        }
        file.extend_from_slice(&[0xDD_u8; 16]);
        file
    }
    for alpha in [0_u32, 1, 3] {
        assert_eq!(
            TextureDecoder::decode_for_destination(
                TextureSource::Dds,
                &dx10_bc7(alpha),
                CompressionSupport::BC,
                TextureDestination::Auto,
            )
            .unwrap()
            .format,
            TextureFormat::Bc7Unorm
        );
    }
    for alpha in [2_u32, 4] {
        assert_eq!(
            TextureDecoder::decode_for_destination(
                TextureSource::Dds,
                &dx10_bc7(alpha),
                CompressionSupport::BC,
                TextureDestination::Auto,
            ),
            Err(TextureError::Unsupported)
        );
    }
    // Reserved high bits are malformed, not defaulted.
    assert_eq!(
        TextureDecoder::decode_for_destination(
            TextureSource::Dds,
            &dx10_bc7(8),
            CompressionSupport::BC,
            TextureDestination::Auto,
        ),
        Err(TextureError::InvalidData)
    );
}
