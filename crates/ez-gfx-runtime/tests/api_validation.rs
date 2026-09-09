//! Runtime integration and contract tests.

use ez_gfx_core::Backend;
use ez_gfx_runtime::{ContextOptions, HeadlessSurfaceOptions, PublicApiError, SurfaceState};

#[test]
fn context_flags_are_validated_independently_of_backend() {
    assert_eq!(
        ContextOptions::new(2, 0),
        Err(PublicApiError::InvalidBoolean)
    );
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        assert!(ContextOptions::new_for_backend(1, 0, backend).is_ok());
    }
}

#[test]
fn headless_surface_options_require_a_nonzero_extent() {
    assert_eq!(
        HeadlessSurfaceOptions::new(64, 64, 0),
        Ok(HeadlessSurfaceOptions {
            width: 64,
            height: 64,
            cache_presented_snapshots: false,
        })
    );
    assert_eq!(
        HeadlessSurfaceOptions::new(0, 1, 0),
        Err(PublicApiError::MixedZeroExtent)
    );
    assert_eq!(
        HeadlessSurfaceOptions::new(0, 0, 0),
        Err(PublicApiError::ZeroInitialExtent)
    );
}

#[test]
fn context_decode_workers_default_to_zero_and_accept_explicit_counts() {
    // Zero preserves the default topology; the builder only records the request.
    let default_options = ContextOptions::new(0, 0).unwrap();
    assert_eq!(default_options.texture_decode_workers, 0);
    let explicit = default_options.with_texture_decode_workers(3);
    assert_eq!(explicit.texture_decode_workers, 3);
    assert_eq!(default_options.texture_decode_workers, 0);
}

#[test]
fn window_surface_state_starts_without_an_extent() {
    let state = SurfaceState::new_window(true);
    assert_eq!(state.extent(), None);
    assert!(state.snapshot_cache());
}

#[test]
fn surface_resize_tracks_minimized_and_pending_state() {
    let mut state = SurfaceState::new(640, 480, false).unwrap();
    assert_eq!(state.resize(0, 0), Err(PublicApiError::NotReady));
    assert_eq!(state.extent(), None);
    assert!(state.resize_pending());
    assert_eq!(state.resize(800, 600), Ok(()));
    assert_eq!(state.extent(), Some((800, 600)));
    assert!(state.take_resize_pending());
    assert!(!state.resize_pending());
}
