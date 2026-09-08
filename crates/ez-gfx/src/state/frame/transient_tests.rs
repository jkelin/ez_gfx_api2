use super::*;
use crate::state::{
    Backend, ContextOptions, DrawIndexedCommand, Error, LifecycleError, acquire_indirect,
    acquire_structured, create_context, destroy_context, frame_begin, frame_submit,
    publish_compute_indirect_count, release_indirect, release_structured, write_indirect,
    write_structured,
};

fn test_context() -> Option<ContextHandle> {
    #[cfg(windows)]
    let backend = Backend::Dx12;
    #[cfg(target_vendor = "apple")]
    let backend = Backend::Metal;
    #[cfg(not(any(windows, target_vendor = "apple")))]
    let backend = Backend::Vulkan;

    #[cfg(windows)]
    let platform = 0;
    #[cfg(target_vendor = "apple")]
    let platform = 2;
    #[cfg(not(any(windows, target_vendor = "apple")))]
    // Linux Vulkan tests require the headless surface platform even without presentation.
    let platform = 3;
    let options = ContextOptions::new_for_backend(0, 0, platform, backend).unwrap();
    let context = match create_context(options) {
        Ok(context) => context,
        // Optional hosted runners may expose no usable native device.
        Err(Error::Unsupported) => return None,
        Err(error) => panic!("transient test context creation failed: {error}"),
    };

    #[cfg(not(any(windows, target_vendor = "apple")))]
    {
        // Vulkan initializes its device from a surface; a headless surface keeps this test hidden.
        let options = crate::state::SurfaceOptions::new(
            0,
            0,
            crate::state::SurfacePlatform::Headless,
            1,
            1,
            0,
        )
        .unwrap();
        let surface = match crate::state::create_surface(context, options) {
            Ok(surface) => surface,
            Err(Error::Unsupported) => {
                destroy_context(context).unwrap();
                return None;
            }
            Err(error) => panic!("transient test surface creation failed: {error}"),
        };
        match crate::state::init_device(context, surface) {
            Ok(()) => {}
            Err(Error::Unsupported) => {
                destroy_context(context).unwrap();
                return None;
            }
            Err(error) => panic!("transient test device initialization failed: {error}"),
        }
    }
    Some(context)
}

fn draw() -> DrawIndexedCommand {
    DrawIndexedCommand {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
    }
}

#[test]
fn successful_submission_retires_handles_and_same_frame_reuse_stays_interned() {
    let Some(context) = test_context() else {
        return;
    };
    frame_begin(context).unwrap();
    let structured = acquire_structured::<u32>(context, 4).unwrap();
    let indirect = acquire_indirect(context, 1).unwrap();
    write_structured(context, structured, 0, &[1, 2, 3, 4]).unwrap();
    write_indirect(context, indirect, 0, &[draw()]).unwrap();

    with_context_mut(context, |state| {
        mark_transient_interned(state, structured.packed())?;
        mark_transient_interned(state, indirect.packed())?;
        // The second node in the same frame follows the same validation path.
        mark_transient_interned(state, structured.packed())?;
        mark_transient_interned(state, indirect.packed())
    })
    .unwrap();

    assert_eq!(
        write_structured(context, structured, 0, &[9]),
        Err(Error::NotReady)
    );
    assert_eq!(
        write_indirect(context, indirect, 0, &[draw()]),
        Err(Error::NotReady)
    );
    release_structured(context, structured);
    release_indirect(context, indirect);
    assert_eq!(
        publish_compute_indirect_count(context, indirect, 1),
        Err(Error::NotReady)
    );

    // This is the exact post-finish transition used only after native submission succeeds;
    // hidden renderer smokes cover the preceding backend execution.
    with_context_mut(context, |state| {
        state.frame.abort();
        let completion = ez_gfx_hal::CompletionToken::new(QueueKind::Graphics, 1).unwrap();
        recycle_consumed_transients(state, completion)?;
        // Pending completion keeps both allocations pooled but unavailable.
        assert_eq!(
            state
                .structured_pool
                .get(&4)
                .map(ez_gfx_hal::ReusableStagingPool::len),
            Some(1)
        );
        assert_eq!(state.indirect_pool.len(), 1);
        Ok(())
    })
    .unwrap();
    assert_eq!(
        write_structured(context, structured, 0, &[9]),
        Err(Error::Lifecycle(LifecycleError::StaleHandle))
    );
    assert_eq!(
        write_indirect(context, indirect, 0, &[draw()]),
        Err(Error::Lifecycle(LifecycleError::StaleHandle))
    );

    frame_begin(context).unwrap();
    let fresh_structured = acquire_structured::<u32>(context, 4).unwrap();
    let fresh_indirect = acquire_indirect(context, 1).unwrap();
    assert_ne!(fresh_structured, structured);
    assert_ne!(fresh_indirect, indirect);
    with_context_mut(context, |state| {
        // The future completion token prevents premature native allocation reuse.
        assert_eq!(
            state
                .structured_pool
                .get(&4)
                .map(ez_gfx_hal::ReusableStagingPool::len),
            Some(1)
        );
        assert_eq!(state.indirect_pool.len(), 1);
        Ok(())
    })
    .unwrap();
    assert_eq!(destroy_context(context), Ok(()));
}

#[test]
fn submission_failure_restores_safe_handles_and_quarantines_unsafe_handles() {
    let Some(context) = test_context() else {
        return;
    };
    frame_begin(context).unwrap();
    let safe = acquire_structured::<u32>(context, 1).unwrap();
    with_context_mut(context, |state| {
        mark_transient_interned(state, safe.packed())?;
        state.frame.abort();
        Ok(())
    })
    .unwrap();

    assert_eq!(frame_submit(context), Err(Error::NotReady));
    release_structured(context, safe);
    assert_eq!(
        write_structured(context, safe, 0, &[1]),
        Err(Error::Lifecycle(LifecycleError::StaleHandle))
    );

    frame_begin(context).unwrap();
    let unsafe_handle = acquire_indirect(context, 1).unwrap();
    // Backends expose no safe post-submit failure injector. Exercise the exact
    // failure branch and assert its public stale-handle and quarantine contract.
    with_context_mut(context, |state| {
        mark_transient_interned(state, unsafe_handle.packed())?;
        invalidate_unsafe_transients(state);
        assert!(state.allocations.contains_key(&unsafe_handle.packed()));
        assert!(state.indirect_pool.is_empty());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        write_indirect(context, unsafe_handle, 0, &[draw()]),
        Err(Error::Lifecycle(LifecycleError::StaleHandle))
    );
    let fresh = acquire_indirect(context, 1).unwrap();
    assert_ne!(fresh, unsafe_handle);
    assert_eq!(destroy_context(context), Ok(()));
}
