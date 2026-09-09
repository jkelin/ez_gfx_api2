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
    ($raw:expr) => {
        match crate::frame::get($raw) {
            Ok(entry) => entry.owner,
            Err(status) => return status,
        }
    };
}
mod adapter;
mod api;
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

use std::panic::{AssertUnwindSafe, catch_unwind};

use ez_gfx::raw::{
    ContextHandle, PublicBinding, RenderTargetHandle, ResourceIdentity, ShaderHandle,
    SurfaceHandle, TextureHandle,
};
use ez_gfx::{
    Backend, ContextOptions, DrawIndexedCommand, DynamicPipelineState, SurfaceOptions,
    SurfacePlatform, raw,
};
/// Identifies C ABI revision 33 for compatibility checks.
pub const EZ_GFX_ABI_VERSION: u32 = 33;
/// Caps any caller-provided byte range at 16 MiB.
pub const EZ_GFX_MAX_BOUNDARY_BYTES: usize = 16 * 1024 * 1024;

#[unsafe(no_mangle)]
/// Returns the C ABI revision supported by this library.
pub extern "C" fn ez_gfx_abi_version() -> u32 {
    EZ_GFX_ABI_VERSION
}

#[unsafe(no_mangle)]
/// Creates a graphics context from debug, validation, and surface-platform options.
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
        let Ok(options) = ContextOptions::new(
            desc.enable_debug,
            desc.enable_validation,
            desc.surface_platform,
        )
        .map(|options| options.with_texture_decode_workers(desc.texture_decode_workers)) else {
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
        let Ok(options) = ContextOptions::new_for_backend(
            desc.enable_debug,
            desc.enable_validation,
            desc.surface_platform,
            backend,
        )
        .map(|options| options.with_texture_decode_workers(desc.texture_decode_workers)) else {
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
/// Destroys the graphics context after aborting every live descendant frame.
pub extern "C" fn ez_gfx_context_destroy(context: EzGfxContext) {
    catch_void(|| {
        let Ok(context) = ContextHandle::from_raw(context) else {
            return;
        };
        while frame::remove_owner_frame(context).is_some() {
            // A failing descendant abort cannot prevent invalidation or context teardown.
            let _ = catch_unwind(AssertUnwindSafe(|| raw::frame_abort(context)));
        }
        buffer::remove_owner(context);
        callback::remove(context);
        let _ = raw::destroy_context(context);
    });
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
pub unsafe extern "C" fn ez_gfx_callback_register(
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
    data: *const u8,
    data_size: usize,
    out_shader: *mut EzGfxShader,
    context: EzGfxContext,
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
pub extern "C" fn ez_gfx_shader_destroy(shader: EzGfxShader, context: EzGfxContext) {
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
    element_size: u32,
    element_count: u32,
    debug_name: *const u8,
    debug_name_length: usize,
    out_indirect: *mut EzGfxCounterBuffer,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if element_size == 0
            || element_count == 0
            || out_indirect.is_null()
            || !out_indirect.is_aligned()
            || validate_bounded_string(debug_name, debug_name_length).is_err()
        {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match buffer::insert(context, buffer::Kind::Counter, element_size, element_count) {
            Ok(handle) => {
                // SAFETY: the validated output remains writable and aligned for this call.
                unsafe { out_indirect.write(handle) };
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
    indirect: EzGfxCounterBuffer,
    start_index: u32,
    commands: *const EzGfxDrawIndexedCommand,
    command_count: u32,
    context: EzGfxContext,
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
            indirect,
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
    indirect: EzGfxCounterBuffer,
    count: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let context = try_handle!(ContextHandle, context);
        buffer::publish(indirect, context, count).map_or_else(|status| status, |()| EzGfxResult::Ok)
    })
}

#[unsafe(no_mangle)]
/// Releases an unconsumed counter buffer.
pub extern "C" fn ez_gfx_counter_buffer_release(
    indirect: EzGfxCounterBuffer,
    context: EzGfxContext,
) {
    catch_void(|| {
        if let Ok(context) = ContextHandle::from_raw(context) {
            let _ = buffer::remove(indirect, context, buffer::Kind::Counter);
        }
    });
}

#[unsafe(no_mangle)]
/// Records an indexed graphics pipeline operation with bindings, dynamic state, and push constants.
///
/// # Safety
///
/// Non-null `bindings` must be readable for `binding_count` aligned entries; every binding name must be a non-null, non-empty exact UTF-8 byte range without embedded NUL bytes. Non-null `dynamic_state` must address one readable aligned value, and non-null `push_constants` must be readable for `push_constant_size` bytes.
pub unsafe extern "C" fn ez_gfx_render_add_vertex_pipeline(
    shader: EzGfxShader,
    indirect: EzGfxCounterBuffer,
    bindings: *const EzGfxBinding,
    binding_count: u32,
    dynamic_state: *const EzGfxDynamicState,
    push_constants: *const std::ffi::c_void,
    push_constant_size: u32,
    frame: EzGfxFrame,
) -> EzGfxResult {
    catch_status(|| {
        if binding_count > 16
            || binding_count != 0 && (bindings.is_null() || !bindings.is_aligned())
            || !dynamic_state.is_null() && !dynamic_state.is_aligned()
            || push_constant_size > 128
            || !push_constant_size.is_multiple_of(4)
            || push_constant_size != 0 && push_constants.is_null()
        {
            return EzGfxResult::InvalidArgument;
        }
        let state = if dynamic_state.is_null() {
            [0, 0, 0, 0]
        } else {
            // SAFETY: This branch establishes that `dynamic_state` is non-null, and the caller keeps readable, properly aligned storage for one `EzGfxDynamicState` alive through this read.
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
        let push = if push_constant_size == 0 {
            &[][..]
        } else {
            // SAFETY: This branch has non-null `push_constants` and a checked nonzero size of at most 128 bytes; the caller keeps that alignment-1 byte range readable through graphics submission.
            unsafe {
                core::slice::from_raw_parts(
                    push_constants.cast::<u8>(),
                    push_constant_size as usize,
                )
            }
        };
        if let Err(status) = buffer::validate(frame, indirect, buffer::Kind::Counter) {
            return status;
        }
        let bindings = match validate_bindings(frame, bindings, binding_count) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let context = try_frame!(frame);
        let shader = try_handle!(ShaderHandle, shader);
        let bindings = match materialize_bindings(frame, bindings) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let indirect = match buffer::materialize(frame, indirect, buffer::Kind::Counter) {
            Ok(ResourceIdentity::Indirect(handle)) => handle,
            Ok(_) => return EzGfxResult::InvalidContext,
            Err(status) => return status,
        };
        raw::render_add_graphics(context, shader, indirect, &bindings, state, push)
            .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Records a compute dispatch with its shader, bindings, dimensions, and push constants.
///
/// # Safety
///
/// Non-null `bindings` must be readable for `binding_count` aligned entries; every binding name must be a non-null, non-empty exact UTF-8 byte range without embedded NUL bytes. Non-null `push_constants` must be readable for `push_constant_size` bytes.
pub unsafe extern "C" fn ez_gfx_render_add_compute_pipeline(
    shader: EzGfxShader,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
    bindings: *const EzGfxBinding,
    binding_count: u32,
    push_constants: *const std::ffi::c_void,
    push_constant_size: u32,
    frame: EzGfxFrame,
) -> EzGfxResult {
    catch_status(|| {
        if binding_count > 16
            || binding_count != 0 && (bindings.is_null() || !bindings.is_aligned())
            || push_constant_size > 128
            || !push_constant_size.is_multiple_of(4)
            || push_constant_size != 0 && push_constants.is_null()
        {
            return EzGfxResult::InvalidArgument;
        }
        let push = if push_constant_size == 0 {
            &[][..]
        } else {
            // SAFETY: This branch has non-null `push_constants` and a checked nonzero size of at most 128 bytes; the caller keeps that alignment-1 byte range readable through compute submission.
            unsafe {
                core::slice::from_raw_parts(
                    push_constants.cast::<u8>(),
                    push_constant_size as usize,
                )
            }
        };
        let bindings = match validate_bindings(frame, bindings, binding_count) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let context = try_frame!(frame);
        let shader = try_handle!(ShaderHandle, shader);
        let bindings = match materialize_bindings(frame, bindings) {
            Ok(value) => value,
            Err(status) => return status,
        };
        raw::render_add_compute(
            context,
            shader,
            [dispatch_x, dispatch_y, dispatch_z],
            &bindings,
            push,
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
pub unsafe extern "C" fn ez_gfx_graph_enqueue_texture_readback(
    texture: EzGfxTexture,
    frame: EzGfxFrame,
    out_request_id: *mut u64,
) -> EzGfxResult {
    catch_status(|| {
        if out_request_id.is_null() || !out_request_id.is_aligned() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_frame!(frame);
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
pub extern "C" fn ez_gfx_frame_end(frame: EzGfxFrame) -> EzGfxResult {
    catch_frame_terminal(frame, || {
        let owner = match frame::get(frame) {
            Ok(entry) => entry.owner,
            Err(status) => return status,
        };
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
pub extern "C" fn ez_gfx_frame_abort(frame: EzGfxFrame) -> EzGfxResult {
    catch_frame_terminal(frame, || {
        let owner = match frame::get(frame) {
            Ok(entry) => entry.owner,
            Err(status) => return status,
        };
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
/// Creates a Win32, GLFW, or Metal-layer presentation surface and returns its handle.
///
/// # Safety
///
/// A non-null `desc` must address one readable, aligned descriptor and non-null `out_surface` one writable, aligned handle. The descriptor's non-null platform objects must remain valid until the returned surface is destroyed.
pub unsafe extern "C" fn ez_gfx_surface_create(
    desc: *const EzGfxSurfaceDesc,
    out_surface: *mut EzGfxSurface,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_surface.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `desc` is non-null, and the caller keeps readable, properly aligned storage for one `EzGfxSurfaceDesc` alive through this read.
        let desc = unsafe { desc.read() };
        let platform = match desc.platform {
            0 => SurfacePlatform::Win32,
            1 => SurfacePlatform::Glfw,
            2 => SurfacePlatform::MetalLayer,
            3 => SurfacePlatform::Headless,
            _ => return EzGfxResult::InvalidArgument,
        };
        let Ok(options) = SurfaceOptions::new(
            desc.window as usize,
            desc.display as usize,
            platform,
            desc.width,
            desc.height,
            desc.cache_presented_snapshots,
        ) else {
            return EzGfxResult::InvalidArgument;
        };
        let context = try_handle!(ContextHandle, context);
        match raw::create_surface(context, options) {
            Ok(handle) => {
                // SAFETY: `out_surface` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxSurface` alive through this write.
                unsafe { out_surface.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Initializes the context device for the specified presentation surface.
pub extern "C" fn ez_gfx_context_init_device(
    surface: EzGfxSurface,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        raw::init_device(
            try_handle!(ContextHandle, context),
            try_handle!(SurfaceHandle, surface),
        )
        .into_ffi_result()
    })
}
#[unsafe(no_mangle)]
/// Requests new pixel dimensions for a presentation surface.
pub extern "C" fn ez_gfx_surface_resize(
    surface: EzGfxSurface,
    width: u32,
    height: u32,
    context: EzGfxContext,
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
    surface: EzGfxSurface,
    out_width: *mut u32,
    out_height: *mut u32,
    context: EzGfxContext,
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
    surface: EzGfxSurface,
    out_pending: *mut i32,
    context: EzGfxContext,
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
    surface: EzGfxSurface,
    enabled: i32,
    context: EzGfxContext,
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
pub extern "C" fn ez_gfx_surface_destroy(surface: EzGfxSurface, context: EzGfxContext) {
    catch_void(|| {
        if let (Ok(context), Ok(surface)) = (
            ContextHandle::from_raw(context),
            SurfaceHandle::from_raw(surface),
        ) {
            raw::destroy_surface(context, surface);
        }
    });
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
    buffer::clear_frame(frame_handle);
    result
}

enum ValidatedBindingResource {
    Buffer { handle: u64, kind: buffer::Kind },
    RenderTarget(RenderTargetHandle),
}

struct ValidatedBinding {
    name: String,
    resource: ValidatedBindingResource,
}

/// Binding arrays are bounded and validated completely before any one-frame buffer is claimed.
fn validate_bindings(
    frame: EzGfxFrame,
    pointer: *const EzGfxBinding,
    count: u32,
) -> Result<Vec<ValidatedBinding>, EzGfxResult> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if pointer.is_null() || count > 16 {
        return Err(EzGfxResult::InvalidArgument);
    }
    // SAFETY: `count` is checked in `1..=16`; the caller keeps `pointer` readable and aligned for that many `EzGfxBinding` values through binding conversion.
    let raw = unsafe { core::slice::from_raw_parts(pointer, count as usize) };
    let mut validated = Vec::with_capacity(raw.len());
    for binding in raw {
        let name = read_bounded_string(binding.name, binding.name_length)?;
        let resource = match (
            binding.buffer != 0,
            binding.counter_buffer != 0,
            binding.render_target != 0,
        ) {
            (true, false, false) => ValidatedBindingResource::Buffer {
                handle: binding.buffer,
                kind: buffer::Kind::Structured,
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
        validated.push(ValidatedBinding { name, resource });
    }
    // Validate every caller-owned name and resource shape before consulting frame
    // state, so malformed foreign memory fails at the boundary deterministically.
    let frame_entry = frame::get(frame)?;
    for binding in &validated {
        match binding.resource {
            ValidatedBindingResource::Buffer { handle, kind } => {
                buffer::validate(frame, handle, kind)?;
            }
            ValidatedBindingResource::RenderTarget(target) => {
                raw::render_target_extent(frame_entry.owner, target).map_err(EzGfxResult::from)?;
            }
        }
    }

    Ok(validated)
}

fn materialize_bindings(
    frame: EzGfxFrame,
    validated: Vec<ValidatedBinding>,
) -> Result<Vec<PublicBinding>, EzGfxResult> {
    validated
        .into_iter()
        .map(|binding| {
            let resource = match binding.resource {
                ValidatedBindingResource::Buffer { handle, kind } => {
                    buffer::materialize(frame, handle, kind)?
                }
                ValidatedBindingResource::RenderTarget(target) => {
                    ResourceIdentity::RenderTarget(target)
                }
            };
            Ok(PublicBinding {
                name: binding.name,
                resource,
            })
        })
        .collect()
}

fn catch_void(operation: impl FnOnce()) {
    if callback::is_invoking() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(operation));
}
