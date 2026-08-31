use ez_gfx_hal::{
    AllocationBlockPolicy, AllocationBlockPolicyError, AllocationError, AllocationRequest,
    BufferRange, DEFAULT_ALLOCATION_BLOCK_POLICY, ImageSubresources, MemoryClass, QueueKind,
    ResourceAccess, ResourceState, ShaderStage,
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
}
