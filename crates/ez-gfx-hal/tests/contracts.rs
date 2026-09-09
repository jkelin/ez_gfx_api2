//! HAL allocation, layout, and synchronization contract tests.
use ez_gfx_hal::{
    AllocationBlockPolicy, AllocationBlockPolicyError, AllocationError, AllocationRequest,
    BufferRange, COUNTER_BUFFER_ELEMENT_OFFSET, DEFAULT_ALLOCATION_BLOCK_POLICY, ImageSubresources,
    MemoryClass, QueueKind, ResourceAccess, ResourceState, ShaderStage,
};

#[test]
fn allocation_request_validates_size_alignment_mapping_and_alias_lifetime() {
    assert_eq!(
        AllocationRequest::new(0, 16, MemoryClass::Device, false, None),
        Err(AllocationError::ZeroSize)
    );
    assert_eq!(
        AllocationRequest::new(64, 3, MemoryClass::Device, false, None),
        Err(AllocationError::InvalidAlignment)
    );
    assert_eq!(
        AllocationRequest::new(64, 16, MemoryClass::Device, true, None),
        Err(AllocationError::NotHostVisible)
    );
    assert_eq!(
        AllocationRequest::new(64, 16, MemoryClass::Upload, true, Some(0)),
        Err(AllocationError::InvalidAliasClass)
    );
    assert!(AllocationRequest::new(64, 16, MemoryClass::Upload, true, None).is_ok());
    assert!(AllocationRequest::new(64, 16, MemoryClass::Transient, false, Some(7)).is_ok());
}

#[test]
fn counter_buffer_element_offset_is_portably_aligned() {
    assert_eq!(COUNTER_BUFFER_ELEMENT_OFFSET, 256);
    assert!(COUNTER_BUFFER_ELEMENT_OFFSET.is_power_of_two());
    assert!(COUNTER_BUFFER_ELEMENT_OFFSET >= size_of::<u32>() as u64);
}

#[test]
fn allocation_block_policy_validates_boundaries_and_exposes_bounded_growth() {
    const MIB: u64 = 1024 * 1024;

    assert_eq!(
        DEFAULT_ALLOCATION_BLOCK_POLICY,
        AllocationBlockPolicy::new(16 * MIB, 256 * MIB, 8 * MIB, 64 * MIB).unwrap()
    );
    assert_eq!(
        AllocationBlockPolicy::new(0, 256 * MIB, 8 * MIB, 64 * MIB),
        Err(AllocationBlockPolicyError::ZeroSize)
    );
    assert_eq!(
        AllocationBlockPolicy::new(6 * MIB, 256 * MIB, 8 * MIB, 64 * MIB),
        Err(AllocationBlockPolicyError::InvalidAlignment)
    );
    assert_eq!(
        AllocationBlockPolicy::new(32 * MIB, 16 * MIB, 8 * MIB, 64 * MIB),
        Err(AllocationBlockPolicyError::InvalidRange)
    );
}

#[test]
fn ranges_reject_empty_and_overflowing_boundaries() {
    assert!(BufferRange::new(0, 1).is_ok());
    assert!(BufferRange::new(0, 0).is_err());
    assert!(BufferRange::new(u64::MAX, 2).is_err());

    assert!(ImageSubresources::new(0, 1, 0, 1).is_ok());
    assert!(ImageSubresources::new(u32::MAX, 2, 0, 1).is_err());
    assert!(ImageSubresources::new(0, 0, 0, 1).is_err());
}
#[test]
fn image_mip_chain_requires_halved_extents_and_exact_rgba_rows() {
    use ez_gfx_hal::{ImageMip, validate_rgba8_mips};
    let level0 = [0_u8; 32];
    let level1 = [0_u8; 8];
    let level2 = [0_u8; 4];
    assert!(
        validate_rgba8_mips(&[
            ImageMip {
                width: 4,
                height: 2,
                bytes: &level0
            },
            ImageMip {
                width: 2,
                height: 1,
                bytes: &level1
            },
            ImageMip {
                width: 1,
                height: 1,
                bytes: &level2
            }
        ])
        .is_ok()
    );
    assert!(validate_rgba8_mips(&[]).is_err());
    assert!(
        validate_rgba8_mips(&[
            ImageMip {
                width: 4,
                height: 2,
                bytes: &level0
            },
            ImageMip {
                width: 3,
                height: 1,
                bytes: &level1
            }
        ])
        .is_err()
    );
    assert!(
        validate_rgba8_mips(&[ImageMip {
            width: 1,
            height: 1,
            bytes: &[0; 3]
        }])
        .is_err()
    );
    assert!(
        validate_rgba8_mips(&[
            ImageMip {
                width: 1,
                height: 1,
                bytes: &[0; 4],
            },
            ImageMip {
                width: 1,
                height: 1,
                bytes: &[0; 4],
            },
        ])
        .is_err()
    );
}

#[test]
fn resource_states_validate_queue_stage_access_combinations() {
    assert!(
        ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::Fragment,
            ResourceAccess::SampledRead
        )
        .is_ok()
    );
    assert!(
        ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::Fragment,
            ResourceAccess::TransferRead
        )
        .is_err()
    );
    assert!(
        ResourceState::new(
            QueueKind::Compute,
            ShaderStage::Compute,
            ResourceAccess::ColorAttachmentWrite
        )
        .is_err()
    );
    assert!(
        ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::StorageRead
        )
        .is_err()
    );
    assert!(
        ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::None,
            ResourceAccess::SampledRead
        )
        .is_err()
    );
    assert!(
        ResourceState::new(
            QueueKind::Compute,
            ShaderStage::Vertex,
            ResourceAccess::StorageRead
        )
        .is_err()
    );
    assert!(
        ResourceState::new(
            QueueKind::Compute,
            ShaderStage::None,
            ResourceAccess::IndexRead
        )
        .is_err()
    );
    assert!(
        ResourceState::new(
            QueueKind::Compute,
            ShaderStage::None,
            ResourceAccess::IndirectRead
        )
        .is_ok()
    );
    assert!(
        ResourceState::new(
            QueueKind::Transfer,
            ShaderStage::None,
            ResourceAccess::TransferRead
        )
        .is_ok()
    );
}

#[test]
fn staging_buckets_round_up_with_hard_bounds() {
    use ez_gfx_hal::{StagingPolicy, staging_bucket_size};

    let policy = StagingPolicy::new(64 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64).unwrap();
    assert_eq!(staging_bucket_size(1, policy).unwrap(), 64 * 1024);
    assert_eq!(staging_bucket_size(64 * 1024, policy).unwrap(), 64 * 1024);
    assert_eq!(
        staging_bucket_size(64 * 1024 + 1, policy).unwrap(),
        128 * 1024
    );
    assert_eq!(
        staging_bucket_size(16 * 1024 * 1024, policy).unwrap(),
        16 * 1024 * 1024
    );
    assert!(staging_bucket_size(0, policy).is_err());
    assert!(staging_bucket_size(16 * 1024 * 1024 + 1, policy).is_err());
}

#[test]
fn transfer_batch_policy_flushes_on_count_or_bytes() {
    use ez_gfx_hal::StagingPolicy;

    let policy = StagingPolicy::new(64, 1024, 512, 4).unwrap();
    assert!(!policy.should_flush(3, 511));
    assert!(policy.should_flush(4, 1));
    assert!(policy.should_flush(1, 512));
}

#[test]
fn texture_formats_compute_exact_block_storage() {
    use ez_gfx_hal::TextureFormat;

    for (format, width, height, bytes) in [
        (TextureFormat::Rgba8Unorm, 7, 5, 140),
        (TextureFormat::Bc1Unorm, 7, 5, 32),
        (TextureFormat::Bc1Srgb, 7, 5, 32),
        (TextureFormat::Bc3Unorm, 7, 5, 64),
        (TextureFormat::Bc3Srgb, 7, 5, 64),
        (TextureFormat::Bc7Unorm, 7, 5, 64),
        (TextureFormat::Bc7Srgb, 7, 5, 64),
        (TextureFormat::Astc4x4Unorm, 7, 5, 64),
        (TextureFormat::Astc4x4Srgb, 7, 5, 64),
    ] {
        assert_eq!(format.level_bytes(width, height), Some(bytes));
    }
    assert_eq!(TextureFormat::Bc7Unorm.level_bytes(0, 4), None);
}

#[test]
fn compressed_mips_require_exact_block_payloads_but_not_multiple_extents() {
    use ez_gfx_hal::{ImageMip, TextureFormat, validate_texture_mips};

    let base = [0_u8; 64];
    let mip = [0_u8; 16];
    assert!(
        validate_texture_mips(
            TextureFormat::Bc7Unorm,
            &[
                ImageMip {
                    width: 7,
                    height: 5,
                    bytes: &base,
                },
                ImageMip {
                    width: 3,
                    height: 2,
                    bytes: &mip,
                },
            ],
        )
        .is_ok()
    );
    assert!(
        validate_texture_mips(
            TextureFormat::Bc7Unorm,
            &[ImageMip {
                width: 7,
                height: 5,
                bytes: &[0; 63],
            }],
        )
        .is_err()
    );
}

#[test]
fn texture_regions_enforce_bounds_blocks_and_edge_extents() {
    use ez_gfx_hal::{TextureFormat, TextureRegion, validate_texture_region};

    let aligned = TextureRegion {
        mip_level: 0,
        x: 4,
        y: 4,
        width: 4,
        height: 4,
        bytes: &[0; 16],
    };
    assert!(validate_texture_region(TextureFormat::Bc7Unorm, 10, 9, 1, aligned).is_ok());

    let edge = TextureRegion {
        x: 8,
        y: 8,
        width: 2,
        height: 1,
        ..aligned
    };
    assert!(validate_texture_region(TextureFormat::Bc7Unorm, 10, 9, 1, edge).is_ok());

    for invalid in [
        TextureRegion { x: 2, ..aligned },
        TextureRegion {
            width: 2,
            ..aligned
        },
        TextureRegion {
            x: 8,
            width: 4,
            ..aligned
        },
        TextureRegion {
            bytes: &[0; 15],
            ..aligned
        },
    ] {
        assert!(validate_texture_region(TextureFormat::Bc7Unorm, 10, 9, 1, invalid).is_err());
    }
}
