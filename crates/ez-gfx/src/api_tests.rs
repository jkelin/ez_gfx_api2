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
assert_not_impl_any!(CounterBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand>: Send, Sync);
assert_not_impl_any!(Frame: Send, Sync);

#[test]
fn buffer_data_views_scalar_slice_and_vec_without_copying() {
    let scalar = 7_u32;
    let array = [1_u32, 2, 3];
    let slice = &array[1..];
    let values = vec![4_u32, 5];
    let scalar_source = BufferSource::one(&scalar);
    let values_source = &values;

    let scalar_view = buffer_data_slice(&scalar_source);
    let slice_view = buffer_data_slice(&slice);
    let vec_view = buffer_data_slice(&values_source);

    assert_eq!(scalar_view, &[7]);
    assert_eq!(slice_view, &[2, 3]);
    assert_eq!(vec_view, &[4, 5]);
    assert_eq!(scalar_view.as_ptr(), &raw const scalar);
    assert_eq!(slice_view.as_ptr(), slice.as_ptr());
    assert_eq!(vec_view.as_ptr(), values.as_ptr());
}

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
    let context = Context::new(options)?;
    let surface = context.create_surface(
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
    let structured = context.acquire_buffer::<u32>(1)?;
    structured.write(0, &[1])?;
    let frame = surface.begin_frame()?;
    assert_eq!(surface.begin_frame().err(), Some(Error::NotReady));

    drop(frame);

    structured.write(0, &[2])?;
    drop(surface.begin_frame()?);
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
        frame.configure_swapchain([0, 1], ez_gfx_runtime::target::Format::Bgra8Srgb),
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
fn allocation_drop_during_recording_does_not_reuse_its_range() -> Result<()> {
    let (context, surface) = headless()?;
    let heap = context.create_vertex_heap::<f32>("recording.positions")?;
    let allocation = heap.upload(&[0.0])?;
    let first_range = allocation.range()?;
    let frame = surface.begin_frame()?;

    drop(allocation);
    let later = heap.upload(&[1.0])?;

    assert_ne!(later.range()?.0, first_range.0);
    drop(frame);
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
fn repeated_render_target_readbacks_keep_distinct_callback_identities() -> Result<()> {
    let (context, _surface) = headless()?;
    let observed = Rc::new(RefCell::new(Vec::new()));
    let callback_observed = Rc::clone(&observed);
    context.register_callback(move |event| {
        if let Event::Readback { request, bytes, .. } = event {
            callback_observed.borrow_mut().push((request, bytes.len()));
        }
    })?;
    let mut frame = context.begin_frame()?;
    let target = frame.configure_render_target(
        "capture",
        [2, 2],
        ez_gfx_runtime::target::Format::Rgba8Unorm,
    )?;
    let first = target.prepare_readback(&mut frame)?;
    let second = target.prepare_readback(&mut frame)?;
    assert_ne!(first.id(), second.id());
    frame.finish()?;
    assert_eq!(
        observed.borrow().as_slice(),
        &[(first.id(), 16), (second.id(), 16)]
    );
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn buffer_sources_infer_scalar_slice_and_vec_elements() -> Result<()> {
    let (context, _surface) = headless()?;
    let scalar = 7_u32;
    let array = [8_u32, 9];
    let slice = array.as_slice();
    let values = vec![10_u32, 11];

    let scalar_buffer = context.acquire_buffer_from(BufferSource::one(&scalar))?;
    let slice_buffer = context.acquire_buffer_from(slice)?;
    let vec_buffer = context.acquire_buffer_from(&values)?;

    assert_eq!(scalar_buffer.inner.element_count, 1);
    assert_eq!(slice_buffer.inner.element_count, 2);
    assert_eq!(vec_buffer.inner.element_count, 2);
    assert_eq!(
        scalar_buffer.inner.bytes.borrow().as_slice(),
        bytemuck::bytes_of(&scalar)
    );
    assert_eq!(
        slice_buffer.inner.bytes.borrow().as_slice(),
        bytemuck::cast_slice::<u32, u8>(slice)
    );
    assert_eq!(
        vec_buffer.inner.bytes.borrow().as_slice(),
        bytemuck::cast_slice::<u32, u8>(&values)
    );
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn claimed_buffer_is_single_frame_and_same_frame_reuses_materialization() -> Result<()> {
    let (context, surface) = headless()?;
    let structured = context.acquire_buffer_from([7_u32, 8].as_slice())?;
    let draw = ez_gfx_runtime::indirect::DrawIndexedCommand {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
    };
    let counter = context.acquire_counter_buffer_from(BufferSource::one(&draw))?;
    let mut first = surface.begin_frame()?;
    let first_handle = first.materialize_structured(&structured.inner)?;
    let counter_handle = first.materialize_counter(&counter.inner)?;

    assert_eq!(
        first.materialize_structured(&structured.inner)?,
        first_handle
    );
    assert_eq!(first.materialize_counter(&counter.inner)?, counter_handle);
    assert_eq!(structured.write(0, &[9]), Err(Error::NotReady));
    assert_eq!(counter.publish_count(1), Err(Error::NotReady));
    assert_eq!(first.finish(), Err(Error::NotReady));
    assert!(structured.inner.usage.get() == BufferUse::Consumed);
    assert!(counter.inner.usage.get() == BufferUse::Consumed);
    assert_eq!(structured.write(0, &[9]), Err(Error::NotReady));
    assert_eq!(counter.write(0, &[draw]), Err(Error::NotReady));

    let mut later = surface.begin_frame()?;
    assert_eq!(
        later.materialize_structured(&structured.inner),
        Err(Error::NotReady)
    );
    assert_eq!(
        later.materialize_counter(&counter.inner),
        Err(Error::NotReady)
    );
    drop(later);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn failed_atomic_surface_creation_does_not_block_later_surface() -> Result<()> {
    let options =
        ez_gfx_runtime::ContextOptions::new_for_backend(0, 0, 3, ez_gfx_core::Backend::Vulkan)
            .map_err(|_| Error::InvalidArgument)?;
    let context = Context::new(options)?;
    let invalid = ez_gfx_runtime::SurfaceOptions {
        window: 0,
        display: 0,
        platform: ez_gfx_runtime::SurfacePlatform::Headless,
        width: 0,
        height: 0,
        cache_presented_snapshots: false,
    };
    assert_eq!(
        context.create_surface(invalid).err(),
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
    let surface = context.create_surface(valid)?;
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
            *slot.borrow_mut() = Some(Context::new(options).expect("context"));
        });
    })
    .join()
    .expect("thread-local context drop must not panic");
}
