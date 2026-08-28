#[test]
fn creates_vulkan_13_instance() {
    let _context = ez_gfx_backend_vulkan::NativeContext::create(
        false,
        false,
        ez_gfx_backend_vulkan::SurfacePlatform::Win32,
    )
    .expect("a Vulkan 1.3 loader with Win32 surface support is required");
}
