//! Native Vulkan initialization smoke test.

#[cfg(not(target_vendor = "apple"))]
#[test]
fn creates_vulkan_13_instance() {
    // No surface is created, shown, or activated by this test.
    let mut context = ez_gfx_backend_vulkan::NativeContext::create(false, false)
        .expect("a Vulkan 1.3 loader is required");

    assert!(matches!(
        context.wait_idle(),
        Err(ez_gfx_hal::HalError::NotReady)
    ));
}
