use super::*;
use static_assertions::assert_not_impl_any;

assert_not_impl_any!(Context: Send, Sync);
assert_not_impl_any!(Surface: Send, Sync);
assert_not_impl_any!(Shader: Send, Sync);
assert_not_impl_any!(Texture: Send, Sync);
assert_not_impl_any!(RenderTarget: Send, Sync);
assert_not_impl_any!(VertexHeap<u32>: Send, Sync);
assert_not_impl_any!(VertexAllocation<u32>: Send, Sync);
assert_not_impl_any!(IndexAllocation: Send, Sync);
assert_not_impl_any!(Buffer<u32>: Send, Sync);
assert_not_impl_any!(CountedBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand>: Send, Sync);
assert_not_impl_any!(Frame: Send, Sync);

#[cfg(windows)]
thread_local! {
    static LATE_CONTEXT: std::cell::RefCell<Option<Context>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(not(target_vendor = "apple"))]
fn headless() -> Result<(Context, Surface)> {
    let options =
        ez_gfx_runtime::ContextOptions::new_for_backend(0, 0, 3, ez_gfx_core::Backend::Vulkan)
            .map_err(|_| Error::InvalidArgument)?;
    let context = create_context(options)?;
    let surface = create_surface(
        &context,
        ez_gfx_runtime::SurfaceOptions::new(
            0,
            0,
            ez_gfx_runtime::SurfacePlatform::Headless,
            1,
            1,
            0,
        )
        .map_err(|_| Error::InvalidArgument)?,
    )?;
    Ok((context, surface))
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn dropping_frame_aborts_and_allows_next_transaction() -> Result<()> {
    let (_context, surface) = headless()?;
    let mut frame = surface.begin_frame()?;
    let structured = frame.acquire_buffer::<u32>(1)?;
    structured.write(&mut frame, 0, &[1])?;
    assert_eq!(surface.begin_frame().err(), Some(Error::NotReady));

    drop(frame);

    let mut next = surface.begin_frame()?;
    assert_eq!(
        structured.write(&mut next, 0, &[2]),
        Err(Error::InvalidContext)
    );
    drop(next);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn configured_swapchain_target_retains_surface_through_completion() -> Result<()> {
    let (_context, surface) = headless()?;
    let surface_lease = Rc::downgrade(&surface.inner);
    let mut frame = surface.begin_frame()?;
    let target = frame.configure_swapchain([1, 1], ez_gfx_runtime::target::Format::Bgra8Srgb)?;

    assert_eq!(target.extent(), Ok((1, 1)));
    assert_eq!(
        target.format(),
        Ok(ez_gfx_runtime::target::Format::Bgra8Srgb)
    );

    assert_eq!(frame.finish(), Err(Error::NotReady));
    let mut next = surface.begin_frame()?;
    let next_target =
        next.configure_swapchain([1, 1], ez_gfx_runtime::target::Format::Bgra8Srgb)?;
    assert!(Rc::ptr_eq(&target.inner, &next_target.inner));
    drop(next);

    drop(surface);
    assert!(surface_lease.upgrade().is_some());
    drop(target);
    assert!(surface_lease.upgrade().is_some());
    drop(next_target);
    assert!(surface_lease.upgrade().is_none());
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn poisoned_frame_finish_returns_exact_record_error_without_submit() -> Result<()> {
    let (_context, surface) = headless()?;
    let mut frame = surface.begin_frame()?;
    assert!(matches!(
        frame.acquire_buffer::<u32>(0),
        Err(Error::InvalidArgument)
    ));

    assert_eq!(frame.finish(), Err(Error::InvalidArgument));
    drop(surface.begin_frame()?);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn frame_retains_surface_after_public_wrapper_drop() -> Result<()> {
    let (_context, surface) = headless()?;

    let lease = Rc::downgrade(&surface.inner);
    let frame = surface.begin_frame()?;

    drop(surface);
    assert!(lease.upgrade().is_some());

    drop(frame);
    assert!(lease.upgrade().is_none());
    Ok(())
}
#[cfg(not(target_vendor = "apple"))]
#[test]
fn frame_retains_vertex_allocation_and_its_heap_in_drop_order() -> Result<()> {
    let (context, surface) = headless()?;
    let heap = context.create_vertex_heap::<f32>("retained.positions")?;
    let allocation = heap.upload(&[0.0])?;
    let heap_lease = Rc::downgrade(&heap.inner);
    let allocation_lease = Rc::downgrade(&allocation.inner);
    let mut frame = surface.begin_frame()?;
    frame.retain_vertex_allocation(&allocation)?;

    drop(allocation);
    drop(heap);
    assert!(allocation_lease.upgrade().is_some());
    assert!(heap_lease.upgrade().is_some());

    drop(frame);
    assert!(allocation_lease.upgrade().is_none());
    assert!(heap_lease.upgrade().is_none());
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn named_render_target_reuses_then_recreates_cached_image() -> Result<()> {
    let (context, _surface) = headless()?;
    let mut first = context.begin_frame()?;
    let first_target = first.configure_render_target(
        "history",
        [2, 2],
        ez_gfx_runtime::target::Format::Rgba8Unorm,
    )?;
    let first_handle = first_target.inner.managed_handle()?;
    drop(first);

    let mut second = context.begin_frame()?;
    let second_target = second.configure_render_target(
        "history",
        [2, 2],
        ez_gfx_runtime::target::Format::Rgba8Unorm,
    )?;
    assert_eq!(second_target.inner.managed_handle()?, first_handle);
    drop(second);

    let mut resized = context.begin_frame()?;
    let resized_target = resized.configure_render_target(
        "history",
        [4, 3],
        ez_gfx_runtime::target::Format::Rgba8Unorm,
    )?;
    assert_ne!(resized_target.inner.managed_handle()?, first_handle);
    assert_eq!(first_target.extent(), Ok((4, 3)));
    drop(resized);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn render_target_readback_is_delivered_only_through_callback() -> Result<()> {
    let (context, _surface) = headless()?;
    let observed = Rc::new(Cell::new(0_usize));
    let callback_observed = Rc::clone(&observed);
    context.register_callback(move |event| {
        if let Event::Readback(bytes) = event {
            callback_observed.set(bytes.len());
        }
    })?;
    let mut frame = context.begin_frame()?;
    let target = frame.configure_render_target(
        "capture",
        [2, 2],
        ez_gfx_runtime::target::Format::Rgba8Unorm,
    )?;
    frame.enqueue_readback(&target.prepare_readback())?;
    frame.finish()?;
    assert_eq!(observed.get(), 16);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn terminal_error_invalidates_frame_transient() -> Result<()> {
    let (_context, surface) = headless()?;
    let mut first = surface.begin_frame()?;
    let structured = first.acquire_buffer::<u32>(1)?;
    assert_eq!(structured.write(&mut first, 0, &[7]), Ok(()));
    assert_eq!(first.finish(), Err(Error::NotReady));

    let mut second = surface.begin_frame()?;
    assert_eq!(
        structured.write(&mut second, 0, &[9]),
        Err(Error::InvalidContext)
    );
    assert_eq!(second.finish(), Err(Error::InvalidContext));
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn failed_atomic_surface_creation_does_not_block_later_surface() -> Result<()> {
    let options =
        ez_gfx_runtime::ContextOptions::new_for_backend(0, 0, 3, ez_gfx_core::Backend::Vulkan)
            .map_err(|_| Error::InvalidArgument)?;
    let context = create_context(options)?;
    let invalid = ez_gfx_runtime::SurfaceOptions {
        window: 0,
        display: 0,
        platform: ez_gfx_runtime::SurfacePlatform::Headless,
        width: 0,
        height: 0,
        cache_presented_snapshots: false,
    };
    assert_eq!(
        create_surface(&context, invalid).err(),
        Some(Error::InvalidArgument)
    );

    let valid = ez_gfx_runtime::SurfaceOptions::new(
        0,
        0,
        ez_gfx_runtime::SurfacePlatform::Headless,
        1,
        1,
        0,
    )
    .map_err(|_| Error::InvalidArgument)?;
    let surface = create_surface(&context, valid)?;
    assert_eq!(surface.extent(), Ok((1, 1)));
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn close_returns_ownership_until_child_leases_are_gone() -> Result<()> {
    let (context, surface) = headless()?;
    let (returned, error) = context.close().expect_err("surface retains context");
    assert_eq!(error, Error::NotReady);
    let context = returned.expect("recoverable close returns ownership");

    drop(surface);
    context.close().map_err(|(_, error)| error)
}

#[cfg(windows)]
#[test]
fn context_drop_after_state_tls_teardown_does_not_panic() {
    std::thread::spawn(|| {
        LATE_CONTEXT.with(|slot| {
            let options = ez_gfx_runtime::ContextOptions::new_for_backend(
                0,
                0,
                3,
                ez_gfx_core::Backend::Vulkan,
            )
            .expect("valid options");
            *slot.borrow_mut() = Some(create_context(options).expect("context"));
        });
    })
    .join()
    .expect("thread-local context drop must not panic");
}
