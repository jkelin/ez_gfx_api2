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
mod adapter;
mod api;
mod bounded_string;
mod error;
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

use ez_gfx::{
    Backend, ContextHandle, ContextOptions, DrawIndexedCommand, DynamicPipelineState,
    IndirectBufferHandle, PublicBinding, RenderTargetHandle, ResourceIdentity, ShaderHandle,
    StructuredBufferHandle, SurfaceHandle, SurfaceOptions, SurfacePlatform, TextureHandle,
    UploadResource, UploadStatus,
};
/// Identifies C ABI revision 30 for compatibility checks.
pub const EZ_GFX_ABI_VERSION: u32 = 30;
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
        match ez_gfx::create_context(options) {
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
        match ez_gfx::create_context(options) {
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
/// Waits until all work submitted through the graphics context is idle.
pub extern "C" fn ez_gfx_context_wait_idle(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| ez_gfx::wait_idle(try_handle!(ContextHandle, context)).into_ffi_result())
}
#[unsafe(no_mangle)]
/// Destroys the graphics context and its owned runtime state.
///
/// Teardown is terminal on the creator thread even if waiting or a native release fails. The
/// stable void ABI cannot report that status; safe Rust callers should use
/// [`ez_gfx::destroy_context`] when cleanup status is required.
pub extern "C" fn ez_gfx_context_destroy(context: EzGfxContext) {
    catch_void(|| {
        if let Ok(context) = ContextHandle::from_raw(context) {
            let _ = ez_gfx::destroy_context(context);
        }
    });
}

#[unsafe(no_mangle)]
/// Polls the context for the next runtime record and reports dropped-record count.
///
/// # Safety
///
/// Every non-null output pointer must address one writable, aligned value for this call.
pub unsafe extern "C" fn ez_gfx_poll_runtime_event(
    out_record: *mut EzGfxRuntimeRecord,
    out_present: *mut u8,
    out_dropped: *mut u64,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_record.is_null() || out_present.is_null() || out_dropped.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match ez_gfx::poll_runtime_event(context) {
            Ok((record, dropped)) => {
                // SAFETY: All three output pointers are non-null; the caller keeps aligned writable storage for one value of each pointed-to type alive through these writes.
                unsafe {
                    out_present.write(u8::from(record.is_some()));
                    out_dropped.write(dropped);
                    if let Some(record) = record {
                        out_record.write(EzGfxRuntimeRecord {
                            correlation_id: record.correlation_id,
                            resource: record.resource,
                            backend: record.backend as u8,
                            phase: record.phase as u8,
                            status: record.status as u8,
                            _padding: [0; 5],
                        });
                    }
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Polls one lossless typed upload transition after progressing owner-thread work.
///
/// # Safety
///
/// Both pointers must address writable, aligned values for this call.
pub unsafe extern "C" fn ez_gfx_poll_upload_event(
    out_event: *mut EzGfxUploadEvent,
    out_present: *mut u8,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_event.is_null() || out_present.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match ez_gfx::poll_upload_event(context) {
            Ok(event) => {
                // SAFETY: pointers were validated and remain caller-owned through the call.
                unsafe {
                    out_present.write(u8::from(event.is_some()));
                    if let Some(event) = event {
                        let (resource, resource_kind) = match event.resource {
                            UploadResource::Texture(handle) => (handle.into_raw(), 1),
                            UploadResource::Vertex(handle) => (handle.into_raw(), 2),
                            UploadResource::Index(handle) => (handle.into_raw(), 3),
                        };
                        let (status, error) = match event.status {
                            UploadStatus::SourceStaged => (1, 0),
                            UploadStatus::DeviceReady => (2, 0),
                            UploadStatus::Failed(error) => (3, error as u8),
                            UploadStatus::Cancelled => (4, 0),
                        };
                        out_event.write(EzGfxUploadEvent {
                            resource,
                            resource_kind,
                            status,
                            error,
                            _padding: [0; 5],
                        });
                    }
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
    })
}

#[unsafe(no_mangle)]
/// Polls the context for the next diagnostic and reports its severity and dropped-record count.
///
/// # Safety
///
/// Every non-null output pointer must address one writable, aligned value for this call.
pub unsafe extern "C" fn ez_gfx_poll_diagnostic(
    out_diagnostic: *mut EzGfxDiagnostic,
    out_present: *mut u8,
    out_dropped: *mut u64,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_diagnostic.is_null() || out_present.is_null() || out_dropped.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match ez_gfx::poll_diagnostic(context) {
            Ok((diagnostic, dropped)) => {
                // SAFETY: All three output pointers are non-null; the caller keeps aligned writable storage for one value of each pointed-to type alive through these writes.
                unsafe {
                    out_present.write(u8::from(diagnostic.is_some()));
                    out_dropped.write(dropped);
                    if let Some((level, record)) = diagnostic {
                        out_diagnostic.write(EzGfxDiagnostic {
                            record: EzGfxRuntimeRecord {
                                correlation_id: record.correlation_id,
                                resource: record.resource,
                                backend: record.backend as u8,
                                phase: record.phase as u8,
                                status: record.status as u8,
                                _padding: [0; 5],
                            },
                            level: level as u8,
                            _padding: [0; 7],
                        });
                    }
                }
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
        }
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
        match ez_gfx::load_shader(context, bytes) {
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
            ez_gfx::destroy_shader(context, shader);
        }
    });
}

#[unsafe(no_mangle)]
/// Begins rendering to the specified surface.
pub extern "C" fn ez_gfx_begin_render(surface: EzGfxSurface, context: EzGfxContext) -> EzGfxResult {
    catch_status(|| {
        ez_gfx::begin_render(
            try_handle!(ContextHandle, context),
            try_handle!(SurfaceHandle, surface),
        )
        .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Begins recording a new frame for the context.
pub extern "C" fn ez_gfx_frame_begin(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| ez_gfx::frame_begin(try_handle!(ContextHandle, context)).into_ffi_result())
}

#[unsafe(no_mangle)]
/// Acquires an indirect draw buffer with the requested command capacity.
///
/// # Safety
///
/// `debug_name` must be non-null and readable for exactly `debug_name_length` bytes; the range must be non-empty UTF-8 without embedded NUL bytes. Non-null `out_indirect` must address one writable, aligned handle for this call.
pub unsafe extern "C" fn ez_gfx_acquire_indirect(
    capacity: u32,
    debug_name: *const u8,
    debug_name_length: usize,
    out_indirect: *mut EzGfxIndirectBuffer,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_indirect.is_null() || validate_bounded_string(debug_name, debug_name_length).is_err()
        {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        match ez_gfx::acquire_indirect(context, capacity) {
            Ok(handle) => {
                // SAFETY: `out_indirect` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxIndirectBuffer` alive through this write.
                unsafe { out_indirect.write(handle.into_raw()) };
                EzGfxResult::Ok
            }
            Err(status) => status.into(),
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
pub unsafe extern "C" fn ez_gfx_indirect_write_draws(
    indirect: EzGfxIndirectBuffer,
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
        if command_count != 0 && commands.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let source = if command_count == 0 {
            &[]
        } else {
            // SAFETY: the caller keeps the validated non-null command range readable.
            unsafe { core::slice::from_raw_parts(commands, command_count as usize) }
        };
        let mut converted = Vec::with_capacity(source.len());
        converted.extend(source.iter().map(|command| DrawIndexedCommand {
            index_count: command.index_count,
            instance_count: command.instance_count,
            first_index: command.first_index,
            vertex_offset: command.vertex_offset,
            first_instance: command.first_instance,
        }));
        debug_assert_eq!(
            converted.len() * core::mem::size_of::<DrawIndexedCommand>(),
            byte_size
        );
        ez_gfx::write_indirect(
            try_handle!(ContextHandle, context),
            try_handle!(IndirectBufferHandle, indirect),
            start_index,
            &converted,
        )
        .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Publishes a CPU-known count for compute-generated indirect commands.
pub extern "C" fn ez_gfx_indirect_publish_compute_count(
    indirect: EzGfxIndirectBuffer,
    count: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        ez_gfx::publish_compute_indirect_count(
            try_handle!(ContextHandle, context),
            try_handle!(IndirectBufferHandle, indirect),
            count,
        )
        .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Releases an indirect draw buffer.
pub extern "C" fn ez_gfx_indirect_release(indirect: EzGfxIndirectBuffer, context: EzGfxContext) {
    catch_void(|| {
        if let (Ok(context), Ok(indirect)) = (
            ContextHandle::from_raw(context),
            IndirectBufferHandle::from_raw(indirect),
        ) {
            ez_gfx::release_indirect(context, indirect);
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
    indirect: EzGfxIndirectBuffer,
    bindings: *const EzGfxBinding,
    binding_count: u32,
    dynamic_state: *const EzGfxDynamicState,
    push_constants: *const std::ffi::c_void,
    push_constant_size: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if binding_count > 16
            || binding_count != 0 && bindings.is_null()
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
        let bindings = match read_bindings(bindings, binding_count) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let context = try_handle!(ContextHandle, context);
        let shader = try_handle!(ShaderHandle, shader);
        let indirect = try_handle!(IndirectBufferHandle, indirect);
        ez_gfx::render_add_graphics(context, shader, indirect, &bindings, state, push)
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
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if binding_count > 16
            || binding_count != 0 && bindings.is_null()
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
        let bindings = match read_bindings(bindings, binding_count) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let context = try_handle!(ContextHandle, context);
        let shader = try_handle!(ShaderHandle, shader);
        ez_gfx::render_add_compute(
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
/// Enqueues a texture readback in the current frame graph.
pub extern "C" fn ez_gfx_graph_enqueue_texture_readback(
    texture: EzGfxTexture,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        ez_gfx::frame_enqueue_readback(
            try_handle!(ContextHandle, context),
            try_handle!(TextureHandle, texture),
        )
        .into_ffi_result()
    })
}

#[unsafe(no_mangle)]
/// Submits the recorded frame to the graphics backend.
pub extern "C" fn ez_gfx_frame_submit(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| ez_gfx::frame_submit(try_handle!(ContextHandle, context)).into_ffi_result())
}

#[unsafe(no_mangle)]
/// Submits the recorded frame and presents its active surface; a failed submission is never followed by presentation.
pub extern "C" fn ez_gfx_finish_render(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| ez_gfx::finish_render(try_handle!(ContextHandle, context)).into_ffi_result())
}

#[unsafe(no_mangle)]
/// Copies the completed frame readback into caller storage and reports the required byte count.
///
/// # Safety
///
/// A non-null `out_size` must address one writable, aligned `usize`. Non-null `data` must be writable for `capacity` bytes.
pub unsafe extern "C" fn ez_gfx_frame_readback(
    data: *mut u8,
    capacity: usize,
    out_size: *mut usize,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_size.is_null()
            || capacity > EZ_GFX_MAX_BOUNDARY_BYTES
            || (capacity != 0 && data.is_null())
        {
            return EzGfxResult::InvalidArgument;
        }
        let context = try_handle!(ContextHandle, context);
        let bytes = match ez_gfx::frame_readback(context) {
            Ok(bytes) => bytes,
            Err(status) => return status.into(),
        };
        // SAFETY: `out_size` is non-null, and the caller keeps writable, properly aligned storage for one `usize` alive through this write.
        unsafe { out_size.write(bytes.len()) };
        if capacity == 0 {
            return EzGfxResult::Ok;
        }
        if capacity < bytes.len() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `capacity >= bytes.len()` and `data` is non-null; runtime-owned `bytes` is live and disjoint from the caller's alignment-1 writable `data` range of `bytes.len()` bytes through the copy.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        EzGfxResult::Ok
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
        match ez_gfx::create_surface(context, options) {
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
        ez_gfx::init_device(
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
        ez_gfx::resize_surface(
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
        match ez_gfx::surface_extent(context, surface) {
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
        match ez_gfx::surface_resize_pending(context, surface) {
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
            0 => ez_gfx::set_snapshot_cache(context, surface, false).into_ffi_result(),
            1 => ez_gfx::set_snapshot_cache(context, surface, true).into_ffi_result(),
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
            ez_gfx::destroy_surface(context, surface);
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
    catch_unwind(AssertUnwindSafe(operation))
        .map(IntoFfiResult::into_ffi_result)
        .unwrap_or(EzGfxResult::NativeFailure)
}

/// Binding arrays are bounded; every item requires one UTF-8 name and exactly one non-null typed handle.
fn read_bindings(
    pointer: *const EzGfxBinding,
    count: u32,
) -> Result<Vec<PublicBinding>, EzGfxResult> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if pointer.is_null() || count > 16 {
        return Err(EzGfxResult::InvalidArgument);
    }
    // SAFETY: `count` is checked in `1..=16`; the caller keeps `pointer` readable and aligned for that many `EzGfxBinding` values through binding conversion.
    let raw = unsafe { core::slice::from_raw_parts(pointer, count as usize) };
    let mut bindings = Vec::with_capacity(raw.len());
    for binding in raw {
        let name = read_bounded_string(binding.name, binding.name_length)?;
        let resource = match (
            binding.structured != 0,
            binding.indirect != 0,
            binding.render_target != 0,
        ) {
            (true, false, false) => ResourceIdentity::Structured(
                StructuredBufferHandle::from_raw(binding.structured)
                    .map_err(|_| EzGfxResult::InvalidContext)?,
            ),
            (false, true, false) => ResourceIdentity::Indirect(
                IndirectBufferHandle::from_raw(binding.indirect)
                    .map_err(|_| EzGfxResult::InvalidContext)?,
            ),
            (false, false, true) => ResourceIdentity::RenderTarget(
                RenderTargetHandle::from_raw(binding.render_target)
                    .map_err(|_| EzGfxResult::InvalidContext)?,
            ),
            _ => return Err(EzGfxResult::InvalidArgument),
        };
        bindings.push(PublicBinding { name, resource });
    }
    Ok(bindings)
}

fn catch_void(operation: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(operation));
}
