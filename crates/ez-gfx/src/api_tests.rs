use super::*;
use static_assertions::assert_not_impl_any;

assert_not_impl_any!(Context: Send, Sync);
assert_not_impl_any!(Surface: Send, Sync);
assert_not_impl_any!(Shader: Send, Sync);
assert_not_impl_any!(Texture: Send, Sync);
assert_not_impl_any!(RenderTarget: Send, Sync);
assert_not_impl_any!(VertexHeap: Send, Sync);
assert_not_impl_any!(VertexAllocation: Send, Sync);
assert_not_impl_any!(IndexAllocation: Send, Sync);
assert_not_impl_any!(StructuredBuffer<u32>: Send, Sync);
assert_not_impl_any!(IndirectBuffer: Send, Sync);
assert_not_impl_any!(Frame: Send, Sync);

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
    let (context, surface) = headless()?;
    let mut frame = begin_frame(&context, &surface)?;
    let structured = frame.acquire_structured::<u32>(1)?;
    structured.write(&mut frame, 0, &[1])?;
    assert_eq!(
        begin_frame(&context, &surface).err(),
        Some(Error::InvalidArgument)
    );

    drop(frame);

    let mut next = begin_frame(&context, &surface)?;
    assert_eq!(
        structured.write(&mut next, 0, &[2]),
        Err(Error::InvalidContext)
    );
    drop(next);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn poisoned_frame_finish_returns_exact_record_error_without_submit() -> Result<()> {
    let (context, surface) = headless()?;
    let mut frame = begin_frame(&context, &surface)?;
    assert!(matches!(
        frame.acquire_structured::<u32>(0),
        Err(Error::InvalidArgument)
    ));

    assert_eq!(frame.finish(), Err(Error::InvalidArgument));
    drop(begin_frame(&context, &surface)?);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn frame_retains_surface_after_public_wrapper_drop() -> Result<()> {
    let (context, surface) = headless()?;
    let lease = Rc::downgrade(&surface.inner);
    let frame = begin_frame(&context, &surface)?;

    drop(surface);
    assert!(lease.upgrade().is_some());

    drop(frame);
    assert!(lease.upgrade().is_none());
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn terminal_error_invalidates_frame_transient() -> Result<()> {
    let (context, surface) = headless()?;
    let mut first = begin_frame(&context, &surface)?;
    let structured = first.acquire_structured::<u32>(1)?;
    assert_eq!(structured.write(&mut first, 0, &[7]), Ok(()));
    assert_eq!(first.finish(), Err(Error::NotReady));

    let mut second = begin_frame(&context, &surface)?;
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
