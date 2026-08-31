//! Runtime integration and contract tests.

use ez_gfx_hal::{CompletionToken, QueueKind};
use ez_gfx_runtime::texture::{
    DecodedMip, DecodedTexture, TextureDecoder, TextureError, TextureEvent, TextureRegistry,
    TextureSource, generate_mips,
};
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

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
            mips: vec![DecodedMip {
                width: 1,
                height: 1,
                rgba8: vec![1, 2, 3, 4]
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
            .rgba8,
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
    assert_eq!(rgb.mips[0].rgba8, vec![5, 6, 7, 255]);
}

#[test]
fn generated_mips_box_filter_odd_edges_and_stop_at_one_pixel() {
    let base = DecodedTexture {
        width: 3,
        height: 1,
        mip_count: 1,
        mips: vec![DecodedMip {
            width: 3,
            height: 1,
            rgba8: vec![0, 0, 0, 0, 10, 20, 30, 40, 100, 120, 140, 160],
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
    assert_eq!(generated.mips[1].rgba8, vec![5, 10, 15, 20]);

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
fn malformed_ktx2_is_rejected_without_panics() {
    assert_eq!(
        TextureDecoder::decode(TextureSource::Ktx2, b"bad"),
        Err(TextureError::InvalidData)
    );
}

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
            .all(|mip| mip.rgba8.len() == mip.width as usize * mip.height as usize * 4)
    );
}

#[test]
fn residency_binding_and_events_follow_upload_completion() {
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
    assert_eq!(registry.begin_upload(), Err(TextureError::CapacityExceeded));
}
