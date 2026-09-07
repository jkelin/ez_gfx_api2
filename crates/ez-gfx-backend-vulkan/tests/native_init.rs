//! Native Vulkan initialization smoke test.

#[cfg(not(target_vendor = "apple"))]
#[test]
fn creates_vulkan_13_instance() {
    // Win32 instances need a Win32 loader; every other host probes headless.
    // No surface is created, shown, or activated by this test.
    let platform = if cfg!(windows) {
        ez_gfx_backend_vulkan::SurfacePlatform::Win32
    } else {
        ez_gfx_backend_vulkan::SurfacePlatform::Headless
    };
    let mut context = ez_gfx_backend_vulkan::NativeContext::create(false, false, platform)
        .expect("a Vulkan 1.3 loader is required");

    assert!(matches!(
        context.wait_idle(),
        Err(ez_gfx_hal::HalError::NotReady)
    ));
}
