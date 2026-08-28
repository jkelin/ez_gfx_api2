use ez_gfx_core::Backend;
use ez_gfx_runtime::{
    ContextOptions, PublicApiError, SurfaceOptions, SurfacePlatform, SurfaceState,
};

#[test]
fn context_flags_and_platform_are_closed_enums() {
    assert_eq!(
        ContextOptions::new(2, 0, 0),
        Err(PublicApiError::InvalidBoolean)
    );
    assert_eq!(
        ContextOptions::new(0, 0, 2),
        Err(PublicApiError::InvalidPlatform)
    );
    assert!(ContextOptions::new(1, 0, 0).is_ok());
}

#[test]
fn context_backend_and_platform_pairs_are_validated() {
    assert!(ContextOptions::new_for_backend(0, 0, 0, Backend::Dx12).is_ok());
    assert!(ContextOptions::new_for_backend(0, 0, 2, Backend::Metal).is_ok());
    assert_eq!(
        ContextOptions::new_for_backend(0, 0, 2, Backend::Vulkan),
        Err(PublicApiError::InvalidPlatform)
    );
    assert_eq!(
        ContextOptions::new_for_backend(0, 0, 1, Backend::Dx12),
        Err(PublicApiError::InvalidPlatform)
    );
}

#[test]
fn surface_contract_rejects_bad_handles_and_mixed_zero_extent() {
    assert_eq!(
        SurfaceOptions::new(0, 1, SurfacePlatform::Win32, 1, 1, 0),
        Err(PublicApiError::MissingNativeHandle)
    );
    assert_eq!(
        SurfaceOptions::new(1, 0, SurfacePlatform::Win32, 1, 1, 0),
        Err(PublicApiError::MissingNativeHandle)
    );
    assert_eq!(
        SurfaceOptions::new(1, 0, SurfacePlatform::Glfw, 1, 1, 0),
        Ok(SurfaceOptions {
            window: 1,
            display: 0,
            platform: SurfacePlatform::Glfw,
            width: 1,
            height: 1,
            cache_presented_snapshots: false
        })
    );
    assert_eq!(
        SurfaceOptions::new(1, 1, SurfacePlatform::Win32, 0, 1, 0),
        Err(PublicApiError::MixedZeroExtent)
    );
    assert_eq!(
        SurfaceOptions::new(1, 1, SurfacePlatform::Win32, 0, 0, 0),
        Err(PublicApiError::ZeroInitialExtent)
    );
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
