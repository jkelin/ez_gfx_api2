#[cfg(windows)]
#[test]
fn creates_hardware_device() {
    let context = ez_gfx_backend_dx12::native::NativeContext::create_default(false)
        .expect("a D3D12 feature-level 12.1 hardware adapter is required");
    assert!(!windows::core::Interface::as_raw(&context.device).is_null());
}

#[cfg(windows)]
#[test]
fn allocates_maps_flushes_and_retires_upload_memory() {
    use ez_gfx_hal::{AllocationRequest, CompletionToken, MemoryAllocator, MemoryClass, QueueKind};

    let mut context = ez_gfx_backend_dx12::native::NativeContext::create_default(false)
        .expect("a D3D12 feature-level 12.1 hardware adapter is required");
    let request = AllocationRequest::new(4096, 256, MemoryClass::Upload, true, None).unwrap();
    let mut allocation = context.allocate(request).unwrap();
    context.mapped_slice_mut(&mut allocation).unwrap()[..4].copy_from_slice(b"ezgx");
    context.flush(&mut allocation, 0, 4).unwrap();
    context
        .retire(
            allocation,
            CompletionToken::new(QueueKind::Graphics, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(context.reclaim(QueueKind::Graphics, 0).unwrap(), 0);
    assert_eq!(context.reclaim(QueueKind::Graphics, 1).unwrap(), 1);
}

#[cfg(windows)]
#[test]
fn rejects_a_null_hwnd_before_swapchain_creation() {
    assert!(matches!(
        ez_gfx_backend_dx12::native::NativeSurface::new(core::ptr::null_mut()),
        Err(ez_gfx_hal::HalError::InvalidArgument)
    ));
}

#[cfg(windows)]
#[test]
fn uploads_each_texture_mip_under_a_distinct_fence() {
    use ez_gfx_hal::ImageMip;

    let mut context = ez_gfx_backend_dx12::native::NativeContext::create_default(false)
        .expect("a D3D12 feature-level 12.1 hardware adapter is required");
    let level0 = [7_u8; 4 * 4 * 4];
    let level1 = [9_u8; 2 * 2 * 4];
    let (texture, completions) = context
        .create_texture_rgba8(
            &[
                ImageMip {
                    width: 4,
                    height: 4,
                    bytes: &level0,
                },
                ImageMip {
                    width: 2,
                    height: 2,
                    bytes: &level1,
                },
            ],
            0,
        )
        .unwrap();
    assert_eq!(completions.len(), 2);
    assert!(completions[0].value < completions[1].value);
    context.wait_idle().unwrap();
    context.destroy_texture(texture).unwrap();
}
