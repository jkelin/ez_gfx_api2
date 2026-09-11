use super::*;
use static_assertions::assert_not_impl_any;

assert_not_impl_any!(Context: Send, Sync);
assert_not_impl_any!(Surface: Send, Sync);
assert_not_impl_any!(ComputeShader: Send, Sync);
assert_not_impl_any!(VertexShader: Send, Sync);
assert_not_impl_any!(FragmentShader: Send, Sync);
assert_not_impl_any!(Texture: Send, Sync);
assert_not_impl_any!(RenderTarget: Send, Sync);
assert_not_impl_any!(VertexHeap<u32>: Send, Sync);
assert_not_impl_any!(VertexAllocation<u32>: Send, Sync);
assert_not_impl_any!(IndexAllocation: Send, Sync);
assert_not_impl_any!(Buffer<u32>: Send, Sync);
assert_not_impl_any!(CounterBuffer<ez_gfx_runtime::indirect::DrawIndexedCommand>: Send, Sync);
assert_not_impl_any!(ValueBuffer<u32>: Send, Sync);
assert_not_impl_any!(Frame: Send, Sync);

#[cfg(not(target_vendor = "apple"))]
struct HostProbe(Rc<Cell<u32>>);

#[cfg(not(target_vendor = "apple"))]
impl HasWindowHandle for HostProbe {
    fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError>
    {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

#[cfg(not(target_vendor = "apple"))]
impl HasDisplayHandle for HostProbe {
    fn display_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError>
    {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

#[cfg(not(target_vendor = "apple"))]
impl Drop for HostProbe {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[cfg(not(target_vendor = "apple"))]
fn attach_host_probe(surface: &mut Surface, drops: &Rc<Cell<u32>>) {
    Rc::get_mut(&mut surface.inner)
        .expect("surface has one public owner")
        .host = Some(Box::new(HostProbe(Rc::clone(drops))));
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn failed_window_surface_creation_retains_host_only_after_abandoned_rollback() {
    for (error, expected_drops) in [(Error::NativeFailure, 1), (Error::TeardownAbandoned, 0)] {
        let drops = Rc::new(Cell::new(0));

        assert_eq!(
            failed_window_surface_creation(HostProbe(Rc::clone(&drops)), error),
            error
        );
        assert_eq!(drops.get(), expected_drops);
    }
}

#[test]
fn owned_host_disposition_follows_typed_teardown_result() {
    struct DropProbe {
        native_destroyed: Rc<Cell<bool>>,
        drops: Rc<Cell<u32>>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            assert!(self.native_destroyed.get());
            self.drops.set(self.drops.get() + 1);
        }
    }

    let drops = Rc::new(Cell::new(0));
    let native_destroyed = Rc::new(Cell::new(false));
    let mut completed_host = Some(DropProbe {
        native_destroyed: Rc::clone(&native_destroyed),
        drops: Rc::clone(&drops),
    });
    teardown_owned_host(&mut completed_host, || {
        native_destroyed.set(true);
        Ok(())
    });
    assert_eq!(drops.get(), 1);

    let mut abandoned_host = Some(DropProbe {
        native_destroyed: Rc::new(Cell::new(true)),
        drops: Rc::clone(&drops),
    });
    teardown_owned_host(&mut abandoned_host, || Err(Error::TeardownAbandoned));
    assert_eq!(drops.get(), 1);

    let mut failed_host = Some(DropProbe {
        native_destroyed: Rc::new(Cell::new(true)),
        drops: Rc::clone(&drops),
    });
    teardown_owned_host(&mut failed_host, || Err(Error::InvalidContext));
    assert_eq!(drops.get(), 1);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn context_destroy_result_controls_later_surface_host_release() -> Result<()> {
    for (outcome, expected_result, expected_drops) in [
        (
            state::CleanupTestOutcome::DrainedFailure(Error::DeviceLost),
            Err(Error::DeviceLost),
            1,
        ),
        (
            state::CleanupTestOutcome::Undrained,
            Err(Error::TeardownAbandoned),
            0,
        ),
    ] {
        let (context, mut surface) = headless()?;
        let drops = Rc::new(Cell::new(0));
        attach_host_probe(&mut surface, &drops);
        state::inject_cleanup_outcome(context.raw(), outcome)?;

        assert_eq!(context.destroy(), expected_result);
        drop(surface);

        assert_eq!(drops.get(), expected_drops);
    }

    let (context, mut surface) = headless()?;
    let drops = Rc::new(Cell::new(0));
    attach_host_probe(&mut surface, &drops);
    let _ = state::cleanup_context_for_thread_exit();

    assert_eq!(context.destroy(), Err(Error::InvalidContext));
    drop(surface);

    assert_eq!(drops.get(), 0);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn implicit_context_drop_result_controls_later_surface_host_release() -> Result<()> {
    for (outcome, expected_drops) in [
        (
            state::CleanupTestOutcome::DrainedFailure(Error::DeviceLost),
            1,
        ),
        (state::CleanupTestOutcome::Undrained, 0),
    ] {
        let (context, mut surface) = headless()?;
        let drops = Rc::new(Cell::new(0));
        attach_host_probe(&mut surface, &drops);
        state::inject_cleanup_outcome(context.raw(), outcome)?;

        drop(context);
        drop(surface);

        assert_eq!(drops.get(), expected_drops);
    }
    Ok(())
}
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

#[cfg(not(target_vendor = "apple"))]
#[test]
fn value_buffer_stores_exactly_one_pod_value() -> Result<()> {
    let (context, _surface) = headless()?;
    let value = [7_u32, 11];

    let buffer = context.acquire_value_buffer(value)?;

    assert_eq!(buffer.inner.element_count, 1);
    assert_eq!(
        buffer.inner.bytes.borrow().as_slice(),
        bytemuck::bytes_of(&value)
    );
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn pooled_buffer_without_initial_data_is_zeroed() -> Result<()> {
    let (context, _surface) = headless()?;
    let buffer = context.acquire_buffer_from([u32::MAX; 2].as_slice())?;
    let original = Rc::as_ptr(&buffer.inner);
    drop(buffer);

    let buffer = context.acquire_buffer::<u32>(2)?;

    assert_eq!(Rc::as_ptr(&buffer.inner), original);
    assert_eq!(buffer.inner.bytes.borrow().as_slice(), &[0; 8]);
    Ok(())
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
        ez_gfx_runtime::ContextOptions::new_for_backend(0, 0, ez_gfx_core::Backend::Vulkan)
            .map_err(|_| Error::InvalidArgument)?;
    let context = Context::new(options)?;
    let surface = context.create_surface_headless(
        ez_gfx_runtime::HeadlessSurfaceOptions::new(1, 1, 0).map_err(|_| Error::InvalidArgument)?,
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
fn headless_surface_rejects_presentation_configuration() -> Result<()> {
    let (_context, surface) = headless()?;

    assert_eq!(surface.presentation_modes(), Ok(PresentationModes::NONE));
    assert_eq!(
        surface.resolve_presentation_mode(PresentationMode::Fifo),
        Err(Error::Unsupported)
    );

    let mut frame = surface.begin_frame()?;
    assert!(matches!(
        frame.configure_swapchain(
            [1, 1],
            ez_gfx_runtime::target::Format::Bgra8Srgb,
            PresentationMode::Fifo,
        ),
        Err(Error::Unsupported)
    ));
    assert_eq!(frame.finish(), Err(Error::Unsupported));
    drop(surface.begin_frame()?);
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn poisoned_frame_finish_returns_exact_record_error_without_submit() -> Result<()> {
    let (context, surface) = headless()?;
    let mut frame = context.begin_frame()?;
    assert!(matches!(
        frame
            .configure_render_target("invalid", [0, 1], ez_gfx_runtime::target::Format::Bgra8Srgb,),
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
fn replacing_unexecuted_binding_leaves_both_buffers_writable_after_abort() -> Result<()> {
    let (context, surface) = headless()?;
    let first = context.acquire_buffer_from([7_u32].as_slice())?;
    let replacement = context.acquire_buffer_from([8_u32].as_slice())?;
    let mut frame = surface.begin_frame()?;

    frame.bind_buffer("value", &first)?;
    frame.bind_buffer("value", &replacement)?;
    first.write(0, &[9])?;
    drop(frame);

    replacement.write(0, &[10])?;
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn failed_atomic_surface_creation_does_not_block_later_surface() -> Result<()> {
    let options =
        ez_gfx_runtime::ContextOptions::new_for_backend(0, 0, ez_gfx_core::Backend::Vulkan)
            .map_err(|_| Error::InvalidArgument)?;
    let context = Context::new(options)?;
    let invalid = ez_gfx_runtime::HeadlessSurfaceOptions {
        width: 0,
        height: 0,
        cache_presented_snapshots: false,
    };
    assert_eq!(
        context.create_surface_headless(invalid).err(),
        Some(Error::InvalidArgument)
    );

    let valid =
        ez_gfx_runtime::HeadlessSurfaceOptions::new(1, 1, 0).map_err(|_| Error::InvalidArgument)?;
    let surface = context.create_surface_headless(valid)?;
    assert_eq!(surface.extent(), Ok((1, 1)));
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn destroy_invalidates_live_child_resources() -> Result<()> {
    let (context, surface) = headless()?;
    context.destroy()?;
    assert_eq!(surface.extent(), Err(Error::InvalidContext));
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn context_drop_invalidates_live_child_resources() -> Result<()> {
    let (context, surface) = headless()?;
    drop(context);
    assert_eq!(surface.extent(), Err(Error::InvalidContext));
    Ok(())
}

#[cfg(windows)]
#[test]
fn context_drop_after_state_tls_teardown_does_not_panic() {
    std::thread::spawn(|| {
        LATE_CONTEXT.with(|slot| {
            let options =
                ez_gfx_runtime::ContextOptions::new_for_backend(0, 0, ez_gfx_core::Backend::Vulkan)
                    .expect("valid options");
            *slot.borrow_mut() = Some(Context::new(options).expect("context"));
        });
    })
    .join()
    .expect("thread-local context drop must not panic");
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn event_dispatch_scratch_bounds_burst_retention() -> Result<()> {
    let (context, _surface) = headless()?;
    // Simulate a burst pinned in scratch; restore must clear and cap capacity
    // so one spike never pins its peak for the context lifetime.
    let mut burst = context.inner.event_scratch.take();
    for _ in 0..(MAX_DISPATCH_SCRATCH_EVENTS + 4000) {
        burst.push(Event::ObservationsDropped(1));
    }
    context.restore_event_scratch(burst);
    let scratch = context.inner.event_scratch.borrow();
    assert!(scratch.is_empty());
    assert!(scratch.capacity() <= MAX_DISPATCH_SCRATCH_EVENTS);
    Ok(())
}
