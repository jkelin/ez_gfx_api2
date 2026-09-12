//! Pipeline transition contracts through a fake backend implementation.

use ez_gfx_core::handle::TextureHandle;
use ez_gfx_hal::{
    AllocationError, CompletionToken, ImageMip, QueueKind, SamplerAddressMode, SamplerFilter,
    TextureFormat, TextureRegion, TextureSamplerDesc,
};
use ez_gfx_texture_manager::{
    MipTransferValues, SharedTransferPool, TextureBackendContext, TextureBackendTexture,
    TextureRegistry, TextureScheduleError, UploadFailure,
    pipeline::{
        PendingUpload, QueuedDecode, SubmitOutcome, TexturePipeline, collect_result, observe_ready,
        pump_fine_uploads, reference_required_prefix, submit_ready_uploads,
    },
    texture::{DecodedMip, DecodedTexture, TextureConfig},
};
use std::collections::HashMap;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

#[derive(Debug)]
struct FakeTexture {
    values: Vec<u64>,
}

impl MipTransferValues for FakeTexture {
    fn mip_transfer_values(&self) -> &[u64] {
        &self.values
    }
}

impl TextureBackendTexture for FakeTexture {
    fn last_transfer_value(&self) -> u64 {
        self.values.iter().copied().max().unwrap_or(0)
    }
}

struct FakeBackend {
    next_value: u64,
    published: Vec<u32>,
}

impl FakeBackend {
    fn new() -> Self {
        Self {
            next_value: 1,
            published: Vec::new(),
        }
    }

    fn token(&mut self) -> CompletionToken {
        let token = CompletionToken::new(QueueKind::TextureTransfer, self.next_value).unwrap();
        self.next_value += 1;
        token
    }
}

impl TextureBackendContext for FakeBackend {
    type Texture = FakeTexture;

    fn create_texture_with_prefix(
        &mut self,
        _format: TextureFormat,
        mips: &[ImageMip<'_>],
        _binding: u32,
        _sampler: ez_gfx_hal::TextureSamplerDesc,
        prefix: u32,
    ) -> Result<(Self::Texture, Vec<CompletionToken>), AllocationError> {
        // Coarse-first tokens over storage-ordered zero values, mirroring
        // the native prefix contract the required-token index relies on.
        let mut tokens = Vec::new();
        for _ in 0..prefix {
            tokens.push(self.token());
        }
        let _ = mips;
        Ok((
            FakeTexture {
                values: vec![0; mips.len()],
            },
            tokens,
        ))
    }

    fn update_texture_region(
        &mut self,
        texture: &mut Self::Texture,
        region: &TextureRegion<'_>,
    ) -> Result<CompletionToken, AllocationError> {
        let token = self.token();
        // Storage order is largest-first; the region level indexes it.
        if let Some(slot) = texture.values.get_mut(region.mip_level as usize) {
            *slot = token.value;
        }
        Ok(token)
    }

    fn reference_texture_prefix(
        &mut self,
        _texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), AllocationError> {
        self.published.push(resident_mips);
        Ok(())
    }

    fn publish_texture_mips(
        &mut self,
        _texture: &mut Self::Texture,
        resident_mips: u32,
    ) -> Result<(), AllocationError> {
        self.published.push(resident_mips);
        Ok(())
    }

    fn texture_descriptors_ready(&self) -> Result<bool, AllocationError> {
        Ok(true)
    }

    fn texture_retirement_ready(
        &self,
        _completion: CompletionToken,
    ) -> Result<bool, AllocationError> {
        Ok(true)
    }

    fn destroy_texture(&mut self, _texture: Self::Texture) -> Result<(), AllocationError> {
        Ok(())
    }

    fn completed_texture_transfer_value(&self) -> Result<u64, AllocationError> {
        Ok(self.next_value)
    }

    fn cancel_texture_transfers(_texture: &Self::Texture) {}
}

fn sampler() -> TextureSamplerDesc {
    TextureSamplerDesc {
        min_filter: SamplerFilter::Nearest,
        mag_filter: SamplerFilter::Nearest,
        max_anisotropy: 1.0,
        address_u: SamplerAddressMode::Clamp,
        address_v: SamplerAddressMode::Clamp,
        address_w: SamplerAddressMode::Clamp,
    }
}

fn config(required: u32) -> TextureConfig {
    TextureConfig {
        source: ez_gfx_texture_manager::texture::TextureSource::Rgba8 {
            width: 4,
            height: 4,
        },
        generate_mips: false,
        required_mips: required,
        width: 4,
        height: 4,
        mip_count: 0,
        destination: ez_gfx_texture_manager::texture::TextureDestination::Rgba8Unorm,
        sampler: sampler(),
    }
}

fn decoded_two_mips() -> DecodedTexture {
    // 4x4 base plus its 2x2 child, tightly packed RGBA8.
    DecodedTexture {
        width: 4,
        height: 4,
        mip_count: 2,
        format: TextureFormat::Rgba8Unorm,
        mips: vec![
            DecodedMip {
                width: 4,
                height: 4,
                bytes: vec![9; 64],
            },
            DecodedMip {
                width: 2,
                height: 2,
                bytes: vec![7; 16],
            },
        ],
    }
}

fn admitted(
    pipe: &mut TexturePipeline,
    registry: &mut TextureRegistry,
    handle: TextureHandle,
    required: u32,
) {
    let id = registry.begin_upload().unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    pipe.admit(
        handle,
        PendingUpload {
            id,
            cancelled: cancelled.clone(),
            config: config(required),
            source_bytes: 4,
            decoded_bytes: None,
            fallback_published: false,
            admitted_at: Instant::now(),
        },
        QueuedDecode {
            handle,
            prepared: ez_gfx_texture_manager::texture::TextureDecoder::prepare(
                ez_gfx_texture_manager::texture::TextureSource::Rgba8 {
                    width: 4,
                    height: 4,
                },
                ez_gfx_core::capability::CompressionSupport::NONE,
                ez_gfx_texture_manager::texture::TextureDestination::Rgba8Unorm,
            )
            .unwrap(),
            bytes: vec![9; 64].into_boxed_slice(),
            generate: false,
            cancelled,
        },
    );
}

fn handle(child_slot: u32) -> TextureHandle {
    // Owner identity is fixed and nonzero; the child slot varies per
    // test so FIFO order stays deterministic.
    use ez_gfx_core::handle::{LocalHandle, PackedHandle};
    let owner = LocalHandle::new(0, 1).unwrap();
    let child = LocalHandle::new(child_slot, 1).unwrap();
    TextureHandle::from_packed(PackedHandle::child(owner, child).unwrap()).unwrap()
}

#[test]
fn submit_tracks_prefix_and_defers_fine_through_the_traits() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let texture = handle(1);
    admitted(&mut pipe, &mut registry, texture, 1);
    collect_result(&mut pipe, texture, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(outcomes[0], SubmitOutcome::Submitted { .. }));
    // One coarse mip submitted; the fine level waits in the phase-two queue.
    assert_eq!(pipe.published()[&texture], 0);
    assert_eq!(pipe.fine()[&texture].len(), 1);
    assert_eq!(pipe.fine()[&texture][0].target, 2);
    // The retained level moved out of the decoded chain, not copied:
    // level zero is the 4x4 base.
    assert_eq!(pipe.fine()[&texture][0].mip.bytes, vec![9; 64]);
    assert!(natives.contains_key(&texture));
    assert!(pipe.ready().contains_key(&texture));
}

#[test]
fn oversized_requirement_fails_without_native_work() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let texture = handle(2);
    admitted(&mut pipe, &mut registry, texture, 4);
    collect_result(&mut pipe, texture, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert_eq!(outcomes.len(), 1);
    let (id, failure, rollback) = match &outcomes[0] {
        SubmitOutcome::Failed {
            id,
            failure,
            rollback,
            ..
        } => (*id, *failure, rollback.is_some()),
        other => panic!("expected failure, got {other:?}"),
    };
    assert_eq!(
        failure,
        UploadFailure::Requirement(TextureScheduleError::InvalidMipCount)
    );
    assert!(!rollback);
    assert!(!natives.contains_key(&texture));
    let _ = id;
}

#[test]
fn fine_pump_advances_coarsest_first_under_budget() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let mut pool = SharedTransferPool::new(1024);
    let texture = handle(3);
    admitted(&mut pipe, &mut registry, texture, 1);
    collect_result(&mut pipe, texture, Ok(decoded_two_mips()));
    let _ = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    // Simulate required observation so phase two unlocks: the prefix
    // publishes, then the required token fires exactly once.
    assert!(
        reference_required_prefix(&mut pipe, &registry, &mut backend, &mut natives, texture)
            .unwrap()
    );
    assert!(observe_ready(&mut pipe, &registry, texture, u64::MAX));
    assert!(!observe_ready(&mut pipe, &registry, texture, u64::MAX));
    let (submitted, failed) = pump_fine_uploads(
        &mut pipe,
        &mut registry,
        &mut backend,
        &mut natives,
        &mut pool,
    );
    assert!(failed.is_empty());
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].target, 2);
    // The submitted level wrote its transfer value into storage order.
    assert_eq!(natives[&texture].values, vec![submitted[0].token.value, 0]);
    assert!(!pipe.fine().contains_key(&texture));
}

#[test]
fn terminal_failure_forgets_every_record_and_retained_payload() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let texture = handle(9);
    admitted(&mut pipe, &mut registry, texture, 1);
    collect_result(&mut pipe, texture, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(outcomes[0], SubmitOutcome::Submitted { .. }));
    // A submitted texture owns residency records plus one retained fine
    // payload; terminal failure must leave none of them behind.
    assert!(pipe.submitted().contains_key(&texture));
    assert!(pipe.fine().contains_key(&texture));
    pipe.forget_transfer(texture);
    pipe.forget_submitted(texture);
    assert!(pipe.submitted().get(&texture).is_none());
    assert!(pipe.published().get(&texture).is_none());
    assert!(pipe.targets().get(&texture).is_none());
    assert!(pipe.last_transfer().get(&texture).is_none());
    assert!(pipe.fine().get(&texture).is_none());
    assert!(pipe.decoded().get(&texture).is_none());
    assert!(pipe.ready().get(&texture).is_none());
    assert!(pipe.work().get(&texture).is_none());
    assert!(pipe.transfer_bytes().get(&texture).is_none());
}

#[test]
fn region_update_rearm_does_not_refire_initial_readiness() {
    use ez_gfx_texture_manager::pipeline::observe_ready;
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let texture = handle(10);
    admitted(&mut pipe, &mut registry, texture, 1);
    collect_result(&mut pipe, texture, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(outcomes[0], SubmitOutcome::Submitted { .. }));
    assert!(
        reference_required_prefix(&mut pipe, &registry, &mut backend, &mut natives, texture)
            .unwrap()
    );
    assert!(observe_ready(&mut pipe, &registry, texture, u64::MAX));
    // A published-view region update re-arms the transfer wait; the initial
    // notification must not fire again.
    pipe.ready_mut().insert(
        texture,
        CompletionToken::new(QueueKind::TextureTransfer, u64::MAX).unwrap(),
    );
    assert!(!observe_ready(&mut pipe, &registry, texture, u64::MAX));
}

#[test]
fn optional_zero_submits_one_coarse_mip_and_stores_zero() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let texture = handle(11);
    admitted(&mut pipe, &mut registry, texture, 0);
    collect_result(&mut pipe, texture, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert_eq!(outcomes.len(), 1);
    let submitted = match &outcomes[0] {
        SubmitOutcome::Submitted { handle } => *handle,
        other => panic!("expected submission, got {other:?}"),
    };
    assert_eq!(submitted, texture);
    // Semantic requirement stays zero while the mechanical single-coarse
    // prefix submits so backends never see prefix zero.
    let id = pipe.submitted()[&texture].id;
    assert_eq!(registry.required_mips(id), Ok(0));
    assert!(pipe.ready().contains_key(&texture));
    assert_eq!(pipe.published()[&texture], 0);
    assert!(natives.contains_key(&texture));
    // One coarse level submitted; the fine level waits in phase two.
    assert_eq!(pipe.fine()[&texture].len(), 1);
}

#[test]
fn required_ready_bypasses_undecoded_optional_head() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let optional = handle(12);
    let required = handle(13);
    admitted(&mut pipe, &mut registry, optional, 0);
    admitted(&mut pipe, &mut registry, required, 1);
    // Only the required tail decoded: it must submit without waiting for
    // the optional head.
    collect_result(&mut pipe, required, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(
        outcomes[0],
        SubmitOutcome::Submitted { handle } if handle == required
    ));
    assert!(pipe.pending().contains_key(&optional));
    assert_eq!(pipe.order().front().copied(), Some(optional));
    // The optional head submits once its own decode arrives, in place.
    collect_result(&mut pipe, optional, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(
        outcomes[0],
        SubmitOutcome::Submitted { handle } if handle == optional
    ));
    assert!(pipe.order().is_empty());
}

#[test]
fn undecoded_required_head_blocks_later_required() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(4, 4).unwrap();
    let mut backend = FakeBackend::new();
    let mut natives = HashMap::new();
    let first = handle(14);
    let second = handle(15);
    admitted(&mut pipe, &mut registry, first, 1);
    admitted(&mut pipe, &mut registry, second, 2);
    // Within one class FIFO holds: the decoded tail waits for its head.
    collect_result(&mut pipe, second, Ok(decoded_two_mips()));
    let outcomes = submit_ready_uploads(&mut pipe, &mut registry, &mut backend, &mut natives, true);
    assert!(outcomes.is_empty());
    assert_eq!(pipe.order().front().copied(), Some(first));
    assert!(pipe.decoded().contains_key(&second));
}

#[test]
fn queued_dispatch_prioritizes_required_without_reordering_class() {
    let mut pipe = TexturePipeline::new();
    let mut registry = TextureRegistry::new(8, 8).unwrap();
    let optional_head = handle(16);
    let required_first = handle(17);
    let required_second = handle(18);
    let optional_tail = handle(19);
    admitted(&mut pipe, &mut registry, optional_head, 0);
    admitted(&mut pipe, &mut registry, required_first, 1);
    admitted(&mut pipe, &mut registry, required_second, 2);
    admitted(&mut pipe, &mut registry, optional_tail, 0);
    // Required jobs dispatch first in admission order; optional jobs keep
    // their own order behind them.
    let order = [
        required_first,
        required_second,
        optional_head,
        optional_tail,
    ];
    for expected in order {
        let job = pipe.pop_queued_prioritized().unwrap();
        assert_eq!(job.handle, expected);
    }
    assert!(pipe.pop_queued_prioritized().is_none());
}
