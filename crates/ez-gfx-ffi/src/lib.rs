//! C ABI for the ez-gfx runtime.

// `try_handle` must precede every `mod` declaration so the whole tree shares it.
macro_rules! try_handle {
    ($type:ty, $raw:expr) => {
        match <$type>::from_raw($raw) {
            Ok(handle) => handle,
            Err(_) => return EzGfxResult::InvalidContext,
        }
    };
}
macro_rules! try_frame {
    ($context:expr, $frame:expr) => {{
        let context = try_handle!(ez_gfx::raw::ContextHandle, $context);
        match crate::frame::get($frame) {
            Ok(entry) if entry.owner == context => context,
            Ok(_) => return EzGfxResult::InvalidContext,
            Err(status) => return status,
        }
    }};
}
mod adapter;
mod api;
#[doc(hidden)]
pub mod binding_enums;
mod bounded_string;
mod buffer;
mod callback;
mod error;
mod frame;
mod geometry;
mod identity;
mod render_target;
mod texture;

pub use adapter::*;
pub use api::*;
use bounded_string::{
    read_bounded_string, validate_bounded_string, validate_optional_bounded_string,
};
pub use error::*;
pub use geometry::*;
pub use identity::*;
pub use render_target::*;
pub use texture::*;

use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{LazyLock, Mutex},
};

use ez_gfx::raw::{
    ContextHandle, PublicBinding, RenderTargetHandle, ResourceIdentity, ShaderHandle,
    SurfaceHandle, TextureHandle,
};
use ez_gfx::{
    Backend, ContextOptions, DrawIndexedCommand, DynamicPipelineState, HeadlessSurfaceOptions, raw,
};
use raw_window_handle::{
    AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle,
    WaylandDisplayHandle, WaylandWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
    XcbDisplayHandle, XcbWindowHandle, XlibDisplayHandle, XlibWindowHandle,
};
/// Identifies C ABI revision 37 for compatibility checks.
pub const EZ_GFX_ABI_VERSION: u32 = 37;
/// Caps any caller-provided byte range at 16 MiB.
pub const EZ_GFX_MAX_BOUNDARY_BYTES: usize = 16 * 1024 * 1024;
fn nonzero_native_ptr(bits: u64) -> Result<core::ptr::NonNull<core::ffi::c_void>, EzGfxResult> {
    let addr = usize::try_from(bits).map_err(|_| EzGfxResult::InvalidArgument)?;
    core::ptr::NonNull::new(addr as *mut core::ffi::c_void).ok_or(EzGfxResult::InvalidArgument)
}

fn nonzero_win32_value(bits: u64) -> Result<core::num::NonZeroIsize, EzGfxResult> {
    let addr = usize::try_from(bits).map_err(|_| EzGfxResult::InvalidArgument)?;
    core::num::NonZeroIsize::new(isize::from_ne_bytes(addr.to_ne_bytes()))
        .ok_or(EzGfxResult::InvalidArgument)
}

fn nonzero_xid(bits: u64) -> Result<u32, EzGfxResult> {
    let xid = u32::try_from(bits).map_err(|_| EzGfxResult::InvalidArgument)?;
    (xid != 0)
        .then_some(xid)
        .ok_or(EzGfxResult::InvalidArgument)
}

fn native_window_handles(
    system: u8,
    reserved: [u8; 6],
    handle_a: u64,
    handle_b: u64,
) -> Result<(RawDisplayHandle, RawWindowHandle), EzGfxResult> {
    if reserved != [0; 6] {
        return Err(EzGfxResult::InvalidArgument);
    }
    match system {
        value if value == EzGfxNativeWindowSystem::Win32 as u8 => {
            let mut window = Win32WindowHandle::new(nonzero_win32_value(handle_a)?);
            window.hinstance = Some(nonzero_win32_value(handle_b)?);
            Ok((
                RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
                RawWindowHandle::Win32(window),
            ))
        }
        value if value == EzGfxNativeWindowSystem::Xlib as u8 => Ok((
            RawDisplayHandle::Xlib(XlibDisplayHandle::new(
                Some(nonzero_native_ptr(handle_a)?),
                0,
            )),
            RawWindowHandle::Xlib(XlibWindowHandle::new(core::ffi::c_ulong::from(
                nonzero_xid(handle_b)?,
            ))),
        )),
        value if value == EzGfxNativeWindowSystem::Xcb as u8 => Ok((
            RawDisplayHandle::Xcb(XcbDisplayHandle::new(
                Some(nonzero_native_ptr(handle_a)?),
                0,
            )),
            RawWindowHandle::Xcb(XcbWindowHandle::new(
                core::num::NonZeroU32::new(nonzero_xid(handle_b)?)
                    .ok_or(EzGfxResult::InvalidArgument)?,
            )),
        )),
        value if value == EzGfxNativeWindowSystem::Wayland as u8 => Ok((
            RawDisplayHandle::Wayland(WaylandDisplayHandle::new(nonzero_native_ptr(handle_a)?)),
            RawWindowHandle::Wayland(WaylandWindowHandle::new(nonzero_native_ptr(handle_b)?)),
        )),
        value if value == EzGfxNativeWindowSystem::AppKit as u8 => {
            if handle_b != 0 {
                return Err(EzGfxResult::InvalidArgument);
            }
            Ok((
                RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
                RawWindowHandle::AppKit(AppKitWindowHandle::new(nonzero_native_ptr(handle_a)?)),
            ))
        }
        _ => Err(EzGfxResult::InvalidArgument),
    }
}

#[unsafe(no_mangle)]
/// Returns the C ABI revision supported by this library.
pub extern "C" fn ez_gfx_abi_version() -> u32 {
    EZ_GFX_ABI_VERSION
}

#[unsafe(no_mangle)]
/// Creates a graphics context from debug and validation options.
///
/// # Safety
///
/// Any non-null `desc` must address one readable, aligned descriptor, and any non-null `out_context` one writable, aligned handle, for this call.
pub unsafe extern "C" fn ez_gfx_context_create(
    desc: *const EzGfxContextDesc,
    out_context: *mut EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_context.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `desc` is non-null, and the caller keeps readable, properly aligned storage for one `EzGfxContextDesc` alive through this read.
        let desc = unsafe { desc.read() };
        let Ok(options) = ContextOptions::new(desc.enable_debug, desc.enable_validation)
            .map(|options| options.with_texture_decode_workers(desc.texture_decode_workers))
        else {
            return EzGfxResult::InvalidArgument;
        };
        let Ok(options) =
            adapter::apply_adapter_selection(options, desc.adapter_count, desc.adapter)
        else {
            return EzGfxResult::InvalidArgument;
        };
        match raw::create_context(options) {
            Ok(handle) => {
                // SAFETY: `out_context` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxContext` alive through this write.
                unsafe { out_context.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Creates a graphics context for Vulkan, Direct3D 12, or Metal.
///
/// # Safety
///
/// Any non-null `desc` must address one readable, aligned descriptor, and any non-null `out_context` one writable, aligned handle, for this call.
pub unsafe extern "C" fn ez_gfx_context_create_backend(
    desc: *const EzGfxBackendContextDesc,
    out_context: *mut EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_context.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `desc` is non-null, and the caller keeps readable, properly aligned storage for one `EzGfxBackendContextDesc` alive through this read.
        let desc = unsafe { desc.read() };
        let backend = match desc.backend {
            1 => Backend::Vulkan,
            2 => Backend::Dx12,
            3 => Backend::Metal,
            _ => return EzGfxResult::InvalidArgument,
        };
        let Ok(options) =
            ContextOptions::new_for_backend(desc.enable_debug, desc.enable_validation, backend)
                .map(|options| options.with_texture_decode_workers(desc.texture_decode_workers))
        else {
            return EzGfxResult::InvalidArgument;
        };
        let Ok(options) =
            adapter::apply_adapter_selection(options, desc.adapter_count, desc.adapter)
        else {
            return EzGfxResult::InvalidArgument;
        };
        match raw::create_context(options) {
            Ok(handle) => {
                // SAFETY: `out_context` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxContext` alive through this write.
                unsafe { out_context.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Waits until all work submitted through the graphics context is idle, then delivers pending events.
pub extern "C" fn ez_gfx_context_wait_idle(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        if let Err(status) = callback::check_entry(context) {
            return status;
        }
        match raw::wait_idle(context).into_ffi_result() {
            EzGfxResult::Ok => callback::dispatch(context),
            status => status,
        }
    })
}
#[unsafe(no_mangle)]
/// Destroys the graphics context after aborting every live descendant frame. Reentrant entry or
/// a boundary panic reports `TeardownAbandoned` because teardown completion is unproven.
pub extern "C" fn ez_gfx_context_destroy(context: EzGfxContext) -> EzGfxResult {
    catch_context_destroy(|| {
        let context = try_handle!(ContextHandle, context);
        if callback::check_entry(context).is_err() {
            return EzGfxResult::TeardownAbandoned;
        }
        if raw::validate_context_owner(context).is_err() {
            return EzGfxResult::InvalidContext;
        }
        clear_owner_binding_drafts(context);
        while frame::remove_owner_frame(context).is_some() {
            // A failing descendant abort cannot prevent invalidation or context teardown.
            let _ = catch_unwind(AssertUnwindSafe(|| raw::frame_abort(context)));
        }
        buffer::remove_owner(context);
        callback::remove(context);
        raw::destroy_context(context).into_ffi_result()
    })
}
#[unsafe(no_mangle)]
/// Replaces the context event callback and delivers pending events.
///
/// A null callback clears the registration. Delivery runs on the calling
/// owner thread inside this and other dispatching entry points, never on a
/// worker thread. Callbacks must not reenter dispatching operations and must
/// not unwind; a reentrant call fails and a panic unregisters the callback.
///
/// # Safety
///
/// `user_data` must remain valid until the callback is replaced, cleared, or
/// its context is destroyed.
pub unsafe extern "C" fn ez_gfx_context_register_callback(
    context: EzGfxContext,
    callback: EzGfxEventCallback,
    user_data: *mut core::ffi::c_void,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        if let Err(status) = callback::check_entry(context) {
            return status;
        }
        callback::register(context, callback, user_data)
    })
}

/// Loads every stage from a compiler-produced shader artifact; no source compiler is linked into this runtime.
#[unsafe(no_mangle)]
///
/// # Safety
///
/// Non-null `data` must be readable for `data_size` bytes, and non-null
/// `out_shader` must be writable for one aligned handle.
pub unsafe extern "C" fn ez_gfx_shader_load_artifact(
    context: EzGfxContext,
    data: *const u8,
    data_size: usize,
    out_shader: *mut EzGfxShader,
) -> EzGfxResult {
    catch_status(|| {
        if data.is_null()
            || out_shader.is_null()
            || data_size == 0
            || data_size > EZ_GFX_MAX_BOUNDARY_BYTES
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `data_size` is checked in `1..=EZ_GFX_MAX_BOUNDARY_BYTES`; the caller keeps `data` readable for that many `u8` values through shader loading.
        let bytes = unsafe { core::slice::from_raw_parts(data, data_size) };
        let context = try_handle!(ContextHandle, context);
        match raw::load_shader(context, bytes) {
            Ok(shader) => {
                // SAFETY: `out_shader` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxShader` alive through this write.
                unsafe { out_shader.write(shader.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Destroys a shader previously loaded into the context.
pub extern "C" fn ez_gfx_shader_destroy(context: EzGfxContext, shader: EzGfxShader) {
    catch_void(|| {
        if let (Ok(context), Ok(shader)) = (
            ContextHandle::from_raw(context),
            ShaderHandle::from_raw(shader),
        ) {
            raw::destroy_shader(context, shader);
        }
    });
}

#[unsafe(no_mangle)]
/// Begins one surface frame and returns its explicit owner handle.
///
/// # Safety
///
/// `out_frame` must address one writable, aligned handle for this call.
pub unsafe extern "C" fn ez_gfx_frame_begin(
    context: EzGfxContext,
    surface: EzGfxSurface,
    out_frame: *mut EzGfxFrame,
) -> EzGfxResult {
    catch_status(|| {
        if out_frame.is_null() || !out_frame.is_aligned() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        let surface = try_handle!(SurfaceHandle, surface);
        if let Err(status) = callback::check_entry(context) {
            return status;
        }
        match callback::dispatch(context) {
            EzGfxResult::Ok => {}
            status => return status,
        }
        if let Err(status) = raw::begin_render(context, surface) {
            return status.into();
        }
        let serial = match raw::current_frame_serial(context) {
            Ok(serial) => serial,
            Err(status) => {
                let _ = raw::frame_abort(context);
                return status.into();
            }
        };
        let frame = match frame::insert(context, frame::FrameKind::Surface, serial) {
            Ok(frame) => frame,
            Err(status) => {
                let _ = raw::frame_abort(context);
                return status;
            }
        };
        callback::note_frame(frame, context.into_raw(), surface.into_raw(), 0);
        // SAFETY: `out_frame` was validated and remains caller-owned through this write.
        unsafe { out_frame.write(frame) };
        EzGfxResult::Ok
    })
}

#[unsafe(no_mangle)]
/// Acquires a context-owned one-frame counter buffer with a runtime element stride.
///
/// # Safety
///
/// The name and output pointer must cover their documented readable/writable ranges.
pub unsafe extern "C" fn ez_gfx_counter_buffer_acquire(
    context: EzGfxContext,
    element_size: u32,
    element_count: u32,
    debug_name: *const u8,
    debug_name_length: usize,
    out_buffer: *mut EzGfxCounterBuffer,
) -> EzGfxResult {
    catch_status(|| {
        let byte_size = u64::from(element_size) * u64::from(element_count);
        if element_size == 0
            || element_count == 0
            || usize::try_from(byte_size)
                .ok()
                .is_none_or(|size| size > EZ_GFX_MAX_BOUNDARY_BYTES)
            || out_buffer.is_null()
            || !out_buffer.is_aligned()
            || debug_name_length > 255
            || validate_bounded_string(debug_name, debug_name_length).is_err()
        {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match buffer::insert(context, buffer::Kind::Counter, element_size, element_count) {
            Ok(handle) => {
                // SAFETY: the validated output remains writable and aligned for this call.
                unsafe { out_buffer.write(handle) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Writes a contiguous indexed-draw command batch.
///
/// # Safety
///
/// `commands` must cover `command_count` readable commands, or may be null when
/// `command_count` is zero.
pub unsafe extern "C" fn ez_gfx_counter_buffer_write_draws(
    context: EzGfxContext,
    buffer: EzGfxCounterBuffer,
    start_index: u32,
    commands: *const EzGfxDrawIndexedCommand,
    command_count: u32,
) -> EzGfxResult {
    catch_status(|| {
        let byte_size = match usize::try_from(command_count)
            .ok()
            .and_then(|count| count.checked_mul(core::mem::size_of::<EzGfxDrawIndexedCommand>()))
        {
            Some(size) if size <= EZ_GFX_MAX_BOUNDARY_BYTES => size,
            _ => return EzGfxResult::InvalidArgument,
        };
        if command_count != 0 && (commands.is_null() || !commands.is_aligned()) {
            return EzGfxResult::InvalidArgument;
        }
        let bytes = if command_count == 0 {
            &[]
        } else {
            // SAFETY: the checked caller range is readable for this call.
            unsafe { core::slice::from_raw_parts(commands.cast::<u8>(), byte_size) }
        };
        let context = try_handle!(ContextHandle, context);
        let Ok(element_size) = u32::try_from(core::mem::size_of::<DrawIndexedCommand>()) else {
            return EzGfxResult::InvalidArgument;
        };
        buffer::write(
            buffer,
            context,
            buffer::Kind::Counter,
            start_index,
            element_size,
            bytes,
        )
        .map_or_else(|status| status, |()| EzGfxResult::Ok)
    })
}

#[unsafe(no_mangle)]
/// Publishes a CPU-known count for compute-generated indirect commands.
pub extern "C" fn ez_gfx_counter_buffer_publish_count(
    context: EzGfxContext,
    buffer: EzGfxCounterBuffer,
    count: u32,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        buffer::publish(buffer, context, count).map_or_else(|status| status, |()| EzGfxResult::Ok)
    })
}

#[unsafe(no_mangle)]
/// Releases an unconsumed counter buffer.
pub extern "C" fn ez_gfx_counter_buffer_release(context: EzGfxContext, buffer: EzGfxCounterBuffer) {
    catch_void(|| {
        if let Ok(context) = ContextHandle::from_raw(context) {
            let _ = buffer::remove(buffer, context, buffer::Kind::Counter);
        }
    });
}

#[unsafe(no_mangle)]
/// Adds or replaces one named resource in the frame binding set.
///
/// # Safety
///
/// `binding` must address one readable, aligned value. Its name must be a
/// non-empty UTF-8 byte range of at most 255 bytes with no embedded NUL.
pub unsafe extern "C" fn ez_gfx_frame_bind(
    context: EzGfxContext,
    frame: EzGfxFrame,
    binding: *const EzGfxBinding,
) -> EzGfxResult {
    catch_status(|| {
        if binding.is_null() || !binding.is_aligned() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: the validated pointer remains readable for one value through this read.
        let binding = unsafe { binding.read() };
        let (name, resource) = match validate_binding(frame, binding) {
            Ok(binding) => binding,
            Err(status) => return status,
        };
        let _ = try_frame!(context, frame);
        let Ok(mut drafts) = FRAME_BINDINGS.lock() else {
            return EzGfxResult::NativeFailure;
        };
        let draft = drafts.entry(frame).or_default();
        // Replacement preserves the 16-name cap and leaves the prior resource unclaimed.
        if !draft.contains_key(&name) && draft.len() == 16 {
            return EzGfxResult::InvalidArgument;
        }
        draft.insert(name, resource);
        EzGfxResult::Ok
    })
}

#[unsafe(no_mangle)]
/// Executes an indexed graphics operation from the current frame bindings.
///
/// # Safety
///
/// Non-null `dynamic_state` must address one readable, aligned value.
pub unsafe extern "C" fn ez_gfx_frame_execute_graphics(
    context: EzGfxContext,
    frame: EzGfxFrame,
    shader: EzGfxShader,
    buffer: EzGfxCounterBuffer,
    dynamic_state: *const EzGfxDynamicState,
) -> EzGfxResult {
    catch_status(|| {
        if !dynamic_state.is_null() && !dynamic_state.is_aligned() {
            return EzGfxResult::InvalidArgument;
        }
        let state = if dynamic_state.is_null() {
            [0, 0, 0, 0]
        } else {
            // SAFETY: the non-null pointer is aligned and remains readable through this read.
            let value = unsafe { dynamic_state.read() };
            [
                value.cull_mode,
                value.front_face,
                value.primitive_type,
                value.blend_mode,
            ]
        };
        let Ok(state) = DynamicPipelineState::from_abi(state[0], state[1], state[2], state[3])
        else {
            return EzGfxResult::InvalidArgument;
        };
        if let Err(status) = buffer::validate(frame, buffer, buffer::Kind::Counter) {
            return status;
        }
        let context = try_frame!(context, frame);
        let shader = try_handle!(ShaderHandle, shader);
        let bindings = match materialized_bindings(frame) {
            Ok(bindings) => bindings,
            Err(status) => return status,
        };
        let counter = match buffer::materialize(frame, buffer, buffer::Kind::Counter) {
            Ok(ResourceIdentity::Counter(handle)) => handle,
            Ok(_) => return EzGfxResult::InvalidContext,
            Err(status) => return status,
        };
        raw::execute_graphics(context, shader, counter, &bindings, state).into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Executes a compute dispatch from the current frame bindings.
pub extern "C" fn ez_gfx_frame_execute_compute(
    context: EzGfxContext,
    frame: EzGfxFrame,
    shader: EzGfxShader,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
) -> EzGfxResult {
    catch_status(|| {
        if dispatch_x == 0 || dispatch_y == 0 || dispatch_z == 0 {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_frame!(context, frame);
        let shader = try_handle!(ShaderHandle, shader);
        let bindings = match materialized_bindings(frame) {
            Ok(bindings) => bindings,
            Err(status) => return status,
        };
        raw::execute_compute(
            context,
            shader,
            [dispatch_x, dispatch_y, dispatch_z],
            &bindings,
        )
        .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Enqueues a texture readback and returns its stable callback correlator.
///
/// # Safety
///
/// `out_request_id` must address one writable, aligned `u64`.
pub unsafe extern "C" fn ez_gfx_frame_enqueue_texture_readback(
    context: EzGfxContext,
    frame: EzGfxFrame,
    texture: EzGfxTexture,
    out_request_id: *mut u64,
) -> EzGfxResult {
    catch_status(|| {
        if out_request_id.is_null() || !out_request_id.is_aligned() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_frame!(context, frame);
        let texture_handle = try_handle!(TextureHandle, texture);
        let (width, height) = match raw::texture_extent(context, texture_handle) {
            Ok(extent) => extent,
            Err(error) => return error.into(),
        };
        let request_id = match callback::note_readback_source(frame, texture, width, height) {
            Ok(request_id) => request_id,
            Err(status) => return status,
        };
        match raw::frame_enqueue_readback(context, texture_handle) {
            Ok(()) => {
                // SAFETY: The caller keeps validated writable storage alive through this write.
                unsafe { out_request_id.write(request_id) };
                EzGfxResult::Ok
            }
            Err(error) => {
                callback::cancel_readback(frame, request_id);
                error.into()
            }
        }
    })
}

#[unsafe(no_mangle)]
/// Consumes and completes a frame; every result invalidates the handle.
///
/// Queued upload, runtime, and diagnostic events dispatch before ordered
/// requested readbacks. Persistent presentation captures are trailing
/// `Snapshot` events with request ID zero.
pub extern "C" fn ez_gfx_frame_end(context: EzGfxContext, frame: EzGfxFrame) -> EzGfxResult {
    catch_frame_terminal(frame, || {
        let owner = try_frame!(context, frame);
        if let Err(status) = callback::check_entry(owner) {
            let _ = frame::remove(frame, frame::FrameState::Aborted);
            callback::take_frame(frame);
            let _ = raw::frame_abort(owner);
            return status;
        }
        let entry = match frame::remove(frame, frame::FrameState::Ended) {
            Ok(entry) => entry,
            Err(status) => return status,
        };
        let aux = callback::take_frame(frame);
        let submit = match entry.kind {
            frame::FrameKind::Surface => raw::finish_render(entry.owner),
            frame::FrameKind::RenderTarget => raw::frame_submit(entry.owner),
        };
        let readbacks = match submit {
            Err(_) => None,
            Ok(()) => Some(match raw::frame_readbacks(entry.owner) {
                Ok(outputs) => Ok(outputs),
                Err(ez_gfx::Error::NotReady)
                    if aux
                        .as_ref()
                        .is_none_or(|aux| aux.readback_sources.is_empty()) =>
                {
                    Ok(Vec::new())
                }
                Err(ez_gfx::Error::NotReady) => Err(EzGfxResult::NativeFailure),
                Err(error) => Err(EzGfxResult::from(error)),
            }),
        };
        let combined = match submit.map_err(EzGfxResult::from) {
            Err(status) => Err(status),
            Ok(()) => match callback::dispatch(entry.owner) {
                EzGfxResult::Ok => Ok(()),
                status => Err(status),
            },
        };
        let mut outputs = match (combined, readbacks) {
            (Err(status), _) | (Ok(()), Some(Err(status))) => return status,
            (Ok(()), Some(Ok(outputs))) => outputs,
            (Ok(()), None) => return EzGfxResult::Ok,
        };
        let sources = aux
            .as_ref()
            .map_or(&[][..], |aux| aux.readback_sources.as_slice());
        if outputs.len() < sources.len() {
            return EzGfxResult::NativeFailure;
        }
        let (width, height) = readback_extent(&entry, aux.as_ref());
        for (source, bytes) in sources.iter().zip(outputs.drain(..sources.len())) {
            let delivery = callback::ReadbackDelivery {
                kind: EzGfxEventKind::Readback,
                request_id: source.request_id,
                texture: source.texture,
                width: source.width,
                height: source.height,
                bytes,
            };
            let status = callback::dispatch_readback(entry.owner, &delivery);
            if status != EzGfxResult::Ok {
                return status;
            }
        }
        for bytes in outputs {
            let delivery = callback::ReadbackDelivery {
                kind: EzGfxEventKind::Snapshot,
                request_id: 0,
                texture: 0,
                width,
                height,
                bytes,
            };
            let status = callback::dispatch_readback(entry.owner, &delivery);
            if status != EzGfxResult::Ok {
                return status;
            }
        }
        EzGfxResult::Ok
    })
}

/// Resolves readback image metadata from tracked frame auxiliaries.
fn readback_extent(entry: &frame::FrameEntry, aux: Option<&callback::FrameAux>) -> (u32, u32) {
    let Some(aux) = aux else {
        return (0, 0);
    };
    if aux.surface != 0 {
        if let Ok(surface) = SurfaceHandle::from_raw(aux.surface) {
            if let Ok(extent) = raw::surface_extent(entry.owner, surface) {
                return extent;
            }
        }
    }
    if aux.target != 0 {
        if let Ok(target) = RenderTargetHandle::from_raw(aux.target) {
            if let Ok(extent) = raw::render_target_extent(entry.owner, target) {
                return extent;
            }
        }
    }
    (0, 0)
}

#[unsafe(no_mangle)]
/// Consumes a frame without submitting; every result invalidates the handle.
pub extern "C" fn ez_gfx_frame_abort(context: EzGfxContext, frame: EzGfxFrame) -> EzGfxResult {
    catch_frame_terminal(frame, || {
        let owner = try_frame!(context, frame);
        if let Err(status) = callback::check_entry(owner) {
            let _ = frame::remove(frame, frame::FrameState::Aborted);
            callback::take_frame(frame);
            let _ = raw::frame_abort(owner);
            return status;
        }
        let entry = match frame::remove(frame, frame::FrameState::Aborted) {
            Ok(entry) => entry,
            Err(status) => return status,
        };
        callback::take_frame(frame);
        match raw::frame_abort(entry.owner).into_ffi_result() {
            EzGfxResult::Ok => callback::dispatch(entry.owner),
            status => status,
        }
    })
}
#[unsafe(no_mangle)]
/// Creates a native window presentation surface and returns its handle. After device
/// initialization, host-managed or undefined Vulkan extents require one `ez_gfx_surface_resize`
/// with the current host framebuffer extent before the first frame.
///
/// # Safety
///
/// A non-null `desc` must address one readable, aligned descriptor and
/// `out_surface` one writable, aligned handle. The descriptor's native handles must be a matched
/// pair from one live host on the context creator thread. They must remain live until successful
/// surface teardown. If creation or teardown returns `TeardownAbandoned`, retain them for the
/// process lifetime.
pub unsafe extern "C" fn ez_gfx_surface_create_window(
    context: EzGfxContext,
    desc: *const EzGfxWindowSurfaceDesc,
    out_surface: *mut EzGfxSurface,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_surface.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: both pointers were validated and remain live through this call.
        let desc = unsafe { desc.read() };
        let cache = match desc.cache_presented_snapshots {
            0 => false,
            1 => true,
            _ => return EzGfxResult::InvalidArgument,
        };
        let (display, window) =
            match native_window_handles(desc.system, desc.reserved, desc.handle_a, desc.handle_b) {
                Ok(handles) => handles,
                Err(status) => return status,
            };
        let context = try_handle!(ContextHandle, context);
        // SAFETY: descriptor validation constructed a matched handle pair; the FFI contract keeps
        // its host live through successful teardown or process-long after unproven teardown.
        match unsafe { raw::create_surface_window_raw(context, display, window, cache) } {
            Ok(handle) => {
                // SAFETY: `out_surface` is non-null, aligned, and writable.
                unsafe { out_surface.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Creates a headless surface and returns its handle.
///
/// # Safety
///
/// A non-null `desc` must address one readable, aligned descriptor and
/// `out_surface` one writable, aligned handle.
pub unsafe extern "C" fn ez_gfx_surface_create_headless(
    context: EzGfxContext,
    desc: *const EzGfxHeadlessSurfaceDesc,
    out_surface: *mut EzGfxSurface,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_surface.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: both pointers were validated and remain live through this call.
        let desc = unsafe { desc.read() };
        let Ok(options) =
            HeadlessSurfaceOptions::new(desc.width, desc.height, desc.cache_presented_snapshots)
        else {
            return EzGfxResult::InvalidArgument;
        };
        let context = try_handle!(ContextHandle, context);
        match raw::create_surface_headless(context, options) {
            Ok(handle) => {
                // SAFETY: `out_surface` is non-null, aligned, and writable.
                unsafe { out_surface.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Initializes the context device for the specified presentation surface. A successful window
/// initialization with a host-managed or undefined Vulkan extent remains unready until
/// `ez_gfx_surface_resize` publishes the current host framebuffer extent.
pub extern "C" fn ez_gfx_context_init_device(
    context: EzGfxContext,
    surface: EzGfxSurface,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        let surface = try_handle!(SurfaceHandle, surface);
        let initialized = raw::init_device(context, surface)
            .and_then(|()| raw::sync_window_surface_extent(context, surface));
        initialized.into_ffi_result()
    })
}
#[unsafe(no_mangle)]
/// Requests new pixel dimensions and publishes host-managed window extents for a presentation
/// surface.
pub extern "C" fn ez_gfx_surface_resize(
    context: EzGfxContext,
    surface: EzGfxSurface,
    width: u32,
    height: u32,
) -> EzGfxResult {
    catch_status(|| {
        raw::resize_surface(
            try_handle!(ContextHandle, context),
            try_handle!(SurfaceHandle, surface),
            width,
            height,
        )
        .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Queries the current width and height of a presentation surface.
///
/// # Safety
///
/// Each non-null output pointer must address one writable, aligned `u32` for this call.
pub unsafe extern "C" fn ez_gfx_surface_get_extent(
    context: EzGfxContext,
    surface: EzGfxSurface,
    out_width: *mut u32,
    out_height: *mut u32,
) -> EzGfxResult {
    catch_status(|| {
        if out_width.is_null() || out_height.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        let surface = try_handle!(SurfaceHandle, surface);
        match raw::surface_extent(context, surface) {
            Ok((width, height)) => {
                // SAFETY: Both output pointers are non-null; the caller keeps aligned writable storage for one `u32` at each pointer alive through these writes.
                unsafe {
                    out_width.write(width);
                    out_height.write(height);
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Reports whether a presentation-surface resize remains pending.
///
/// # Safety
///
/// A non-null `out_pending` must address one writable, aligned `i32` for this call.
pub unsafe extern "C" fn ez_gfx_surface_resize_pending(
    context: EzGfxContext,
    surface: EzGfxSurface,
    out_pending: *mut i32,
) -> EzGfxResult {
    catch_status(|| {
        if out_pending.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        let surface = try_handle!(SurfaceHandle, surface);
        match raw::surface_resize_pending(context, surface) {
            Ok(pending) => {
                // SAFETY: `out_pending` is non-null, and the caller keeps writable, properly aligned storage for one `i32` alive through this write.
                unsafe {
                    out_pending.write(i32::from(pending));
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Enables or disables caching of presented snapshots for a surface.
pub extern "C" fn ez_gfx_surface_set_snapshot_cache(
    context: EzGfxContext,
    surface: EzGfxSurface,
    enabled: i32,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        let surface = try_handle!(SurfaceHandle, surface);
        match enabled {
            0 => raw::set_snapshot_cache(context, surface, false).into_ffi_result(),
            1 => raw::set_snapshot_cache(context, surface, true).into_ffi_result(),
            _ => EzGfxResult::InvalidArgument,
        }
    })
}

#[unsafe(no_mangle)]
/// Destroys a presentation surface owned by the context.
pub extern "C" fn ez_gfx_surface_destroy(
    context: EzGfxContext,
    surface: EzGfxSurface,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        let surface = try_handle!(SurfaceHandle, surface);
        if let Err(status) = callback::check_entry(context) {
            return status;
        }
        raw::destroy_surface(context, surface).into_ffi_result()
    })
}

trait IntoFfiResult {
    fn into_ffi_result(self) -> EzGfxResult;
}

impl IntoFfiResult for EzGfxResult {
    fn into_ffi_result(self) -> EzGfxResult {
        self
    }
}

impl IntoFfiResult for ez_gfx::Result<()> {
    fn into_ffi_result(self) -> EzGfxResult {
        self.map_or_else(Into::into, |()| EzGfxResult::Ok)
    }
}

fn catch_status<T: IntoFfiResult>(operation: impl FnOnce() -> T) -> EzGfxResult {
    // Reentrant calls would alias the owner-thread runtime while it dispatches.
    if callback::is_invoking() {
        return EzGfxResult::InvalidArgument;
    }
    catch_unwind(AssertUnwindSafe(operation))
        .map(IntoFfiResult::into_ffi_result)
        .unwrap_or(EzGfxResult::NativeFailure)
}

fn catch_context_destroy<T: IntoFfiResult>(operation: impl FnOnce() -> T) -> EzGfxResult {
    // Any panic or reentrant rejection leaves context teardown unproven.
    if callback::is_invoking() {
        return EzGfxResult::TeardownAbandoned;
    }
    catch_unwind(AssertUnwindSafe(operation))
        .map(IntoFfiResult::into_ffi_result)
        .unwrap_or(EzGfxResult::TeardownAbandoned)
}

#[cfg(test)]
mod context_destroy_tests {
    use super::*;

    #[test]
    fn panic_maps_to_teardown_abandoned() {
        assert_eq!(
            catch_context_destroy(|| -> EzGfxResult { panic!("injected teardown panic") }),
            EzGfxResult::TeardownAbandoned
        );
    }
}

fn catch_frame_terminal<T: IntoFfiResult>(
    frame_handle: EzGfxFrame,
    operation: impl FnOnce() -> T,
) -> EzGfxResult {
    // A panic must still retire the opaque handle and unwind the raw transaction.
    let result = if let Ok(result) = catch_unwind(AssertUnwindSafe(operation)) {
        result.into_ffi_result()
    } else {
        if let Ok(entry) = frame::remove(frame_handle, frame::FrameState::Aborted) {
            callback::take_frame(frame_handle);
            let _ = raw::frame_abort(entry.owner);
        }
        EzGfxResult::NativeFailure
    };
    clear_binding_draft(frame_handle);
    buffer::clear_frame(frame_handle);
    result
}

#[derive(Clone, Copy)]
enum ValidatedBindingResource {
    Buffer { handle: u64, kind: buffer::Kind },
    RenderTarget(RenderTargetHandle),
}

type FrameBindingDraft = HashMap<String, ValidatedBindingResource>;
static FRAME_BINDINGS: LazyLock<Mutex<HashMap<EzGfxFrame, FrameBindingDraft>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Validates one foreign binding completely before it can enter a frame draft.
fn validate_binding(
    frame: EzGfxFrame,
    binding: EzGfxBinding,
) -> Result<(String, ValidatedBindingResource), EzGfxResult> {
    if binding.name_length > 255 {
        return Err(EzGfxResult::InvalidArgument);
    }
    let name = read_bounded_string(binding.name, binding.name_length)?;
    let resource = match (
        binding.buffer != 0,
        binding.counter_buffer != 0,
        binding.render_target != 0,
    ) {
        (true, false, false) => ValidatedBindingResource::Buffer {
            handle: binding.buffer,
            kind: buffer::Kind::Buffer,
        },
        (false, true, false) => ValidatedBindingResource::Buffer {
            handle: binding.counter_buffer,
            kind: buffer::Kind::Counter,
        },
        (false, false, true) => {
            let target = RenderTargetHandle::from_raw(binding.render_target)
                .map_err(|_| EzGfxResult::InvalidContext)?;
            ValidatedBindingResource::RenderTarget(target)
        }
        _ => return Err(EzGfxResult::InvalidArgument),
    };
    let frame_entry = frame::get(frame)?;
    match resource {
        ValidatedBindingResource::Buffer { handle, kind } => {
            buffer::validate(frame, handle, kind)?;
        }
        ValidatedBindingResource::RenderTarget(target) => {
            raw::render_target_extent(frame_entry.owner, target).map_err(EzGfxResult::from)?;
        }
    }
    Ok((name, resource))
}

fn materialized_bindings(frame: EzGfxFrame) -> Result<Vec<PublicBinding>, EzGfxResult> {
    let drafts = FRAME_BINDINGS
        .lock()
        .map_err(|_| EzGfxResult::NativeFailure)?;
    let Some(draft) = drafts.get(&frame) else {
        return Ok(Vec::new());
    };
    let mut bindings = Vec::with_capacity(draft.len());
    for (name, binding) in draft {
        let resource = match *binding {
            ValidatedBindingResource::Buffer { handle, kind } => {
                buffer::materialize(frame, handle, kind)?
            }
            ValidatedBindingResource::RenderTarget(target) => {
                ResourceIdentity::RenderTarget(target)
            }
        };
        bindings.push(PublicBinding {
            name: name.clone(),
            resource,
        });
    }
    Ok(bindings)
}

fn clear_binding_draft(frame: EzGfxFrame) {
    if let Ok(mut drafts) = FRAME_BINDINGS.lock() {
        drafts.remove(&frame);
    }
}

fn clear_owner_binding_drafts(owner: ContextHandle) {
    if let Ok(mut drafts) = FRAME_BINDINGS.lock() {
        drafts.retain(|frame, _| frame::get(*frame).is_ok_and(|entry| entry.owner != owner));
    }
}

fn catch_void(operation: impl FnOnce()) {
    if callback::is_invoking() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(operation));
}
