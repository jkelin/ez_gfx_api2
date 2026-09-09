use crate::Result;

#[cfg(windows)]
use super::Dx12Surface;
#[cfg(target_vendor = "apple")]
use super::MetalSurface;
#[cfg(test)]
use super::SurfaceInsertTestFailure;
use super::{
    Backend, ContextHandle, ContextState, Error, HalError, HeadlessSurfaceOptions, NativeContext,
    NativeSurface, ResourceKind, SurfaceHandle, SurfaceRecord, SurfaceState, SurfaceWindow,
    map_hal, map_lifecycle, map_native_loss, result_status, with_context_mut, with_surface_mut,
};

/// Creates a headless surface with an explicit initial extent.
///
/// # Errors
///
/// Returns an error for an invalid context, unsupported backend, exhausted
/// handles, invalid extent, or native surface failure.
pub fn create_surface_headless(
    context: ContextHandle,
    options: HeadlessSurfaceOptions,
) -> Result<SurfaceHandle> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let native = match &mut context.native {
            NativeContext::Vulkan(native_context) => {
                NativeSurface::Vulkan(native_context.create_headless_surface().map_err(map_hal)?)
            }
            #[cfg(windows)]
            NativeContext::Dx12(_) => return Err(Error::Unsupported),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(_) => return Err(Error::Unsupported),
        };
        let state = SurfaceState::new(
            options.width,
            options.height,
            options.cache_presented_snapshots,
        )
        .map_err(|_| Error::InvalidArgument)?;
        insert_surface(context, native, state, false)
    })
}

/// Creates a surface from a portable native window/display handle pair.
pub(crate) fn create_surface_window(
    context: ContextHandle,
    window: SurfaceWindow,
    cache_presented_snapshots: bool,
) -> Result<SurfaceHandle> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let native = match &mut context.native {
            NativeContext::Vulkan(native) => {
                // SAFETY: the safe facade owns the host; the FFI boundary requires
                // matched live handles through surface destruction.
                NativeSurface::Vulkan(
                    unsafe { native.create_surface(window.display, window.window) }
                        .map_err(map_hal)?,
                )
            }
            #[cfg(windows)]
            NativeContext::Dx12(_) => {
                // SAFETY: inherited from this function's matched live-host contract.
                NativeSurface::Dx12(
                    unsafe { Dx12Surface::new(window.display, window.window) }.map_err(map_hal)?,
                )
            }
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(_) => {
                // SAFETY: inherited from this function's matched live-host contract.
                NativeSurface::Metal(
                    unsafe {
                        MetalSurface::new(window.display, window.window, cache_presented_snapshots)
                    }
                    .map_err(map_hal)?,
                )
            }
        };
        insert_surface(
            context,
            native,
            SurfaceState::new_window(cache_presented_snapshots),
            true,
        )
    })
}

// Any rollback failure leaves native release unproven, regardless of its lower-level error.
fn rollback_surface_insertion(
    context: &mut ContextState,
    native: NativeSurface,
    insertion_error: Error,
) -> Error {
    #[cfg(test)]
    if std::mem::take(&mut context.surface_rollback_test_abandoned) {
        // Simulate the backend retaining a still-live native surface after failed destruction.
        std::mem::forget(native);
        return Error::TeardownAbandoned;
    }

    match destroy_native_surface(&mut context.native, native) {
        Ok(()) => insertion_error,
        Err(_) => Error::TeardownAbandoned,
    }
}

fn insert_surface(
    context: &mut ContextState,
    native: NativeSurface,
    state: SurfaceState,
    is_window: bool,
) -> Result<SurfaceHandle> {
    #[cfg(test)]
    let injected_failure = context.surface_insert_test_failure.take();
    #[cfg(test)]
    if injected_failure == Some(SurfaceInsertTestFailure::IdentityInsertion) {
        return Err(rollback_surface_insertion(
            context,
            native,
            Error::NativeFailure,
        ));
    }

    let handle = match context.identity.insert(ResourceKind::Surface) {
        Ok(handle) => handle,
        Err(error) => {
            let insertion_error = map_lifecycle(error);
            return Err(rollback_surface_insertion(context, native, insertion_error));
        }
    };
    #[cfg(test)]
    let converted = if injected_failure == Some(SurfaceInsertTestFailure::InvalidPackedHandle) {
        None
    } else {
        SurfaceHandle::from_packed(handle).ok()
    };
    #[cfg(not(test))]
    let converted = SurfaceHandle::from_packed(handle).ok();
    let Some(surface) = converted else {
        let _ = context.identity.remove(handle, ResourceKind::Surface);
        return Err(rollback_surface_insertion(
            context,
            native,
            Error::NativeFailure,
        ));
    };
    context.surfaces.insert(
        surface,
        SurfaceRecord {
            native,
            state,
            is_window,
        },
    );
    Ok(surface)
}

/// Copies the current native window extent into the logical surface state.
///
/// # Errors
///
/// Returns an error for stale handles or a failed native window query.
pub fn sync_window_surface_extent(context: ContextHandle, surface: SurfaceHandle) -> Result<()> {
    enum ExtentDisposition {
        Known(u32, u32),
        Minimized,
        HostManaged,
    }

    with_context_mut(context, |context| {
        context
            .identity
            .resolve(surface.packed(), ResourceKind::Surface)
            .map_err(map_lifecycle)?;
        let record = context
            .surfaces
            .get(&surface)
            .ok_or(Error::InvalidContext)?;
        // Headless surfaces retain their explicit extent; only window surfaces query a native host.
        if !record.is_window {
            return Ok(());
        }

        let extent = match (&context.native, &record.native) {
            (NativeContext::Vulkan(native), NativeSurface::Vulkan(surface)) => {
                match native.window_extent(surface).map_err(map_hal)? {
                    ez_gfx_backend_vulkan::NativeWindowExtent::Known(width, height) => {
                        ExtentDisposition::Known(width, height)
                    }
                    ez_gfx_backend_vulkan::NativeWindowExtent::Minimized => {
                        ExtentDisposition::Minimized
                    }
                    ez_gfx_backend_vulkan::NativeWindowExtent::HostManaged => {
                        ExtentDisposition::HostManaged
                    }
                }
            }
            #[cfg(windows)]
            (NativeContext::Dx12(_), NativeSurface::Dx12(surface)) => surface
                .window_extent()
                .map_err(map_hal)?
                .map_or(ExtentDisposition::Minimized, |(width, height)| {
                    ExtentDisposition::Known(width, height)
                }),
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(_), NativeSurface::Metal(surface)) => surface
                .window_extent()
                .map_or(ExtentDisposition::Minimized, |(width, height)| {
                    ExtentDisposition::Known(width, height)
                }),
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => return Err(Error::InvalidArgument),
        };
        match extent {
            ExtentDisposition::Known(width, height) => context
                .surfaces
                .get_mut(&surface)
                .ok_or(Error::InvalidContext)?
                .state
                .resize(width, height)
                .map_err(|error| match error {
                    ez_gfx_runtime::PublicApiError::NotReady => Error::NotReady,
                    _ => Error::InvalidArgument,
                }),
            ExtentDisposition::Minimized => Err(Error::NotReady),
            ExtentDisposition::HostManaged => Ok(()),
        }
    })
}

/// Creates a window surface from opaque native handles at the C boundary.
///
/// # Safety
///
/// `display` and `window` must be a matched pair borrowed from the same live native host and
/// must remain valid until successful surface teardown. This must run on the context creator
/// thread. If teardown returns [`Error::TeardownAbandoned`] or another result that cannot prove
/// native release, the host objects must remain alive for the process lifetime.
///
/// # Errors
///
/// Returns an error when the target has no supported native window ABI or the
/// handles are invalid for the selected backend.
#[cfg(feature = "ffi")]
pub unsafe fn create_surface_window_raw(
    context: ContextHandle,
    display: raw_window_handle::RawDisplayHandle,
    window: raw_window_handle::RawWindowHandle,
    cache_presented_snapshots: bool,
) -> Result<SurfaceHandle> {
    create_surface_window(
        context,
        SurfaceWindow { display, window },
        cache_presented_snapshots,
    )
}

/// Initializes a context device for a surface.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn init_device(context: ContextHandle, surface: SurfaceHandle) -> Result<()> {
    result_status(with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = surface.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Surface)
            .map_err(map_lifecycle)?;
        let record = context
            .surfaces
            .get(&surface)
            .ok_or(Error::InvalidContext)?;
        let first_initialization = context.active_surface.is_none();
        // Explicit selection is enforced at device creation: Vulkan instances
        // are adapter-agnostic, so the stable identity resolves here.
        let selection = context.options.adapter_selection;
        let adapter = match (&mut context.native, &record.native) {
            (NativeContext::Vulkan(native), NativeSurface::Vulkan(surface)) => {
                let presentation_surface = (!surface.is_headless()).then_some(surface);
                match selection {
                    Some(selected) => native.init_device_for_adapter(
                        presentation_surface,
                        selected.stable_id,
                        selected.allow_software,
                    ),
                    None => native.init_device(presentation_surface),
                }
            }
            #[cfg(windows)]
            (NativeContext::Dx12(native), NativeSurface::Dx12(surface)) => {
                native.init_device(surface)
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(native), NativeSurface::Metal(surface)) => {
                native.init_device(surface)
            }
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => Err(HalError::InvalidArgument),
        }
        .map_err(|error| map_native_loss(&context.identity, error))?;
        context.active_surface = Some(surface);
        if first_initialization {
            let backend = match adapter.backend() {
                Backend::Vulkan => "Vulkan",
                Backend::Dx12 => "DirectX 12",
                Backend::Metal => "Metal",
            };
            eprintln!(
                "ez-gfx: initialized GPU `{}` with {backend}",
                adapter.name()
            );
        }
        Ok(())
    }))
}

/// Requests a surface resize.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn resize_surface(
    context: ContextHandle,
    surface: SurfaceHandle,
    width: u32,
    height: u32,
) -> Result<()> {
    result_status(with_surface_mut(context, surface, |record| {
        record
            .state
            .resize(width, height)
            .map_err(|error| match error {
                ez_gfx_runtime::PublicApiError::NotReady => Error::NotReady,
                _ => Error::InvalidArgument,
            })
    }))
}
/// Returns the current surface extent.
///
/// # Errors
///
/// Returns an error when either handle is invalid or the extent is not ready.
pub fn surface_extent(context: ContextHandle, surface: SurfaceHandle) -> Result<(u32, u32)> {
    with_surface_mut(context, surface, |record| {
        record.state.extent().ok_or(Error::NotReady)
    })
}
/// Reports whether a surface resize is pending.
///
/// # Errors
///
/// Returns an error when either handle is invalid or stale.
pub fn surface_resize_pending(context: ContextHandle, surface: SurfaceHandle) -> Result<bool> {
    with_surface_mut(context, surface, |record| Ok(record.state.resize_pending()))
}
/// Enables or disables presented snapshot caching.
///
/// # Errors
///
/// Returns an error when validation, handle ownership, readiness, or a backend operation fails.
pub fn set_snapshot_cache(
    context: ContextHandle,
    surface: SurfaceHandle,
    enabled: bool,
) -> Result<()> {
    result_status(with_surface_mut(context, surface, |record| {
        record.state.set_snapshot_cache(enabled);
        Ok(())
    }))
}
/// Destroys a presentation surface.
///
/// # Errors
///
/// Returns an error for stale, foreign, or wrong-thread handles, or when native teardown must
/// abandon the surface because submitted work cannot be proven complete.
pub fn destroy_surface(context: ContextHandle, surface: SurfaceHandle) -> Result<()> {
    with_context_mut(context, |context| {
        let handle = surface.packed();
        context
            .identity
            .remove(handle, ResourceKind::Surface)
            .map_err(map_lifecycle)?;
        let record = context
            .surfaces
            .remove(&surface)
            .ok_or(Error::InvalidContext)?;
        let result = destroy_native_surface(&mut context.native, record.native);
        if context.active_surface == Some(surface) {
            context.active_surface = None;
        }
        result
    })
}
// Backend destruction remains shared by creation rollback and context teardown.
pub(super) fn destroy_native_surface(
    context: &mut NativeContext,
    surface: NativeSurface,
) -> Result<()> {
    let destroyed = match (context, surface) {
        (NativeContext::Vulkan(context), NativeSurface::Vulkan(surface)) => {
            context.destroy_surface(surface)
        }
        #[cfg(windows)]
        (NativeContext::Dx12(context), NativeSurface::Dx12(surface)) => {
            context.destroy_surface(surface)
        }
        #[cfg(target_vendor = "apple")]
        (NativeContext::Metal(context), NativeSurface::Metal(surface)) => {
            context.destroy_surface(surface)
        }
        #[cfg(any(windows, target_vendor = "apple"))]
        _ => return Err(Error::InvalidArgument),
    };
    destroyed.then_some(()).ok_or(Error::TeardownAbandoned)
}
