//! C ABI for the ez-gfx runtime.

mod api;
mod state;

pub use api::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

use ez_gfx_core::{
    Backend, SemanticId,
    handle::{HandleParts, PackedHandle},
};
use ez_gfx_runtime::{ContextOptions, SurfaceOptions, SurfacePlatform};

/// Identifies C ABI revision 17 for compatibility checks.
pub const EZ_GFX_ABI_VERSION: u32 = 17;
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
        ) else {
            return EzGfxResult::InvalidArgument;
        };
        match state::create_context(options) {
            Ok(handle) => {
                // SAFETY: `out_context` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxContext` alive through this write.
                unsafe { out_context.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
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
        ) else {
            return EzGfxResult::InvalidArgument;
        };
        match state::create_context(options) {
            Ok(handle) => {
                // SAFETY: `out_context` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxContext` alive through this write.
                unsafe { out_context.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Waits until all work submitted through the graphics context is idle.
pub extern "C" fn ez_gfx_context_wait_idle(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::wait_idle(context))
}
#[unsafe(no_mangle)]
/// Destroys the graphics context and its owned runtime state.
pub extern "C" fn ez_gfx_context_destroy(context: EzGfxContext) {
    catch_void(|| state::destroy_context(context));
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
        match state::poll_runtime_event(context) {
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
            Err(status) => status,
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
        match state::poll_diagnostic(context) {
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
            Err(status) => status,
        }
    })
}

/// Loads a compiler-produced shader artifact; no source compiler is linked into this runtime.
#[unsafe(no_mangle)]
/// Loads requested shader stages from a compiler-produced artifact and returns a shader handle.
///
/// # Safety
///
/// Non-null `data` must be readable for `data_size` bytes, non-null `entries` for `entry_count` aligned entries, and non-null `out_shader` writable for one aligned handle. Every non-null entry-point pointer must reference a NUL-terminated string for this call.
pub unsafe extern "C" fn ez_gfx_shader_load_artifact(
    data: *const u8,
    data_size: usize,
    entries: *const EzGfxShaderEntry,
    entry_count: usize,
    out_shader: *mut EzGfxShader,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if data.is_null()
            || entries.is_null()
            || out_shader.is_null()
            || data_size == 0
            || data_size > EZ_GFX_MAX_BOUNDARY_BYTES
            || entry_count == 0
            || entry_count > 16
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `data_size` is checked in `1..=EZ_GFX_MAX_BOUNDARY_BYTES`; the caller keeps `data` readable for that many `u8` values (alignment 1) through shader loading.
        let bytes = unsafe { core::slice::from_raw_parts(data, data_size) };
        // SAFETY: `entry_count` is checked in `1..=16`; the caller keeps `entries` readable and aligned for that many `EzGfxShaderEntry` values through iteration.
        let entries = unsafe { core::slice::from_raw_parts(entries, entry_count) };
        let mut requests = Vec::with_capacity(entry_count);
        for entry in entries {
            let name = match read_c_string(entry.entry) {
                Ok(name) => name,
                Err(status) => return status,
            };
            let stage = match entry.stage {
                1 => ez_gfx_artifact::Stage::Vertex,
                2 => ez_gfx_artifact::Stage::Fragment,
                3 => ez_gfx_artifact::Stage::Compute,
                4 => ez_gfx_artifact::Stage::Geometry,
                5 => ez_gfx_artifact::Stage::TessellationControl,
                6 => ez_gfx_artifact::Stage::TessellationEvaluation,
                _ => return EzGfxResult::InvalidArgument,
            };
            let Ok(request) = ez_gfx_runtime::shader::ShaderRequest::new(name, stage) else {
                return EzGfxResult::InvalidArgument;
            };
            requests.push(request);
        }
        match state::load_shader(context, bytes, &requests) {
            Ok(shader) => {
                // SAFETY: `out_shader` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxShader` alive through this write.
                unsafe { out_shader.write(shader.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Destroys a shader previously loaded into the context.
pub extern "C" fn ez_gfx_shader_destroy(shader: EzGfxShader, context: EzGfxContext) {
    catch_void(|| state::destroy_shader(context, shader));
}

#[unsafe(no_mangle)]
/// Loads texture bytes, configures sampling and mip residency, and returns a texture handle.
///
/// # Safety
///
/// Non-null `data` must be readable for `data_size` bytes, non-null `desc` readable for one aligned descriptor, and non-null `out_texture` writable for one aligned handle. A non-null descriptor label must reference a NUL-terminated string for this call.
pub unsafe extern "C" fn ez_gfx_texture_load(
    data: *const u8,
    data_size: usize,
    desc: *const EzGfxTextureDesc,
    out_texture: *mut EzGfxTexture,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if data.is_null()
            || desc.is_null()
            || out_texture.is_null()
            || data_size == 0
            || data_size > EZ_GFX_MAX_BOUNDARY_BYTES
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `desc` is non-null, and the caller keeps readable, properly aligned storage for one `EzGfxTextureDesc` alive through this read.
        let desc = unsafe { desc.read() };
        if desc.generate_mips > 1
            || desc.destination_format != 0
            || desc.min_filter > 1
            || desc.mag_filter > 1
            || desc.address_mode_u > 1
            || desc.address_mode_v > 1
            || desc.address_mode_w > 1
            || !desc.max_anisotropy.is_finite()
            || !(1.0..=16.0).contains(&desc.max_anisotropy)
            || (!desc.debug_label.is_null() && read_c_string(desc.debug_label).is_err())
        {
            return EzGfxResult::InvalidArgument;
        }
        let source = match desc.source_format {
            0 => ez_gfx_runtime::texture::TextureSource::Rgb8 {
                width: desc.width,
                height: desc.height,
            },
            1 => ez_gfx_runtime::texture::TextureSource::Rgba8 {
                width: desc.width,
                height: desc.height,
            },
            2 => ez_gfx_runtime::texture::TextureSource::Bmp,
            3 => ez_gfx_runtime::texture::TextureSource::Jpeg,
            4 => ez_gfx_runtime::texture::TextureSource::Png,
            5 => ez_gfx_runtime::texture::TextureSource::Tga,
            6 => ez_gfx_runtime::texture::TextureSource::Ktx2,
            _ => return EzGfxResult::InvalidArgument,
        };
        // SAFETY: `data_size` is checked in `1..=EZ_GFX_MAX_BOUNDARY_BYTES`; the caller keeps `data` readable for that many `u8` values (alignment 1) through texture loading.
        let bytes = unsafe { core::slice::from_raw_parts(data, data_size) };
        let filter = |value| match value {
            0 => ez_gfx_hal::SamplerFilter::Nearest,
            _ => ez_gfx_hal::SamplerFilter::Linear,
        };
        let address = |value| match value {
            0 => ez_gfx_hal::SamplerAddressMode::Clamp,
            _ => ez_gfx_hal::SamplerAddressMode::Repeat,
        };
        let config = state::TextureConfig {
            width: desc.width,
            height: desc.height,
            mip_count: desc.mip_count,
            sampler: ez_gfx_hal::TextureSamplerDesc {
                min_filter: filter(desc.min_filter),
                mag_filter: filter(desc.mag_filter),
                max_anisotropy: desc.max_anisotropy,
                address_u: address(desc.address_mode_u),
                address_v: address(desc.address_mode_v),
                address_w: address(desc.address_mode_w),
            },
        };
        match state::load_texture(context, source, bytes, desc.generate_mips != 0, &config) {
            Ok(texture) => {
                // SAFETY: `out_texture` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxTexture` alive through this write.
                unsafe { out_texture.write(texture.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Queries the binding index assigned to a loaded texture.
///
/// # Safety
///
/// A non-null `out_binding` must address one writable, aligned `u32` for this call.
pub unsafe extern "C" fn ez_gfx_texture_get_binding(
    texture: EzGfxTexture,
    out_binding: *mut u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_binding.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        match state::texture_binding(context, texture) {
            Ok(binding) => {
                // SAFETY: `out_binding` is non-null, and the caller keeps writable, properly aligned storage for one `u32` alive through this write.
                unsafe { out_binding.write(binding) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Queries the resident and total mip counts for a loaded texture.
///
/// # Safety
///
/// Each non-null output pointer must address one writable, aligned `u32` for this call.
pub unsafe extern "C" fn ez_gfx_texture_get_residency(
    texture: EzGfxTexture,
    out_resident_mips: *mut u32,
    out_total_mips: *mut u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_resident_mips.is_null() || out_total_mips.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        match state::texture_residency(context, texture) {
            Ok((resident, total)) => {
                // SAFETY: Both output pointers are non-null; the caller keeps aligned writable storage for one `u32` at each pointer alive through these writes.
                unsafe {
                    out_resident_mips.write(resident);
                    out_total_mips.write(total);
                };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Unloads a texture from the context.
pub extern "C" fn ez_gfx_texture_unload(texture: EzGfxTexture, context: EzGfxContext) {
    catch_void(|| state::unload_texture(context, texture));
}

#[unsafe(no_mangle)]
/// Begins rendering to the specified surface.
pub extern "C" fn ez_gfx_begin_render(surface: EzGfxSurface, context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::begin_render(context, surface))
}

#[unsafe(no_mangle)]
/// Begins recording a new frame for the context.
pub extern "C" fn ez_gfx_frame_begin(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::frame_begin(context))
}

#[unsafe(no_mangle)]
/// Acquires an indirect draw buffer with the requested command capacity.
///
/// # Safety
///
/// A non-null `debug_name` must reference a NUL-terminated string, and non-null `out_indirect` one writable, aligned handle, for this call.
pub unsafe extern "C" fn ez_gfx_acquire_indirect(
    capacity: u32,
    debug_name: *const std::ffi::c_char,
    out_indirect: *mut EzGfxIndirectBuffer,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if out_indirect.is_null() || read_c_string(debug_name).is_err() {
            return EzGfxResult::InvalidArgument;
        }
        match state::acquire_indirect(context, capacity) {
            Ok(handle) => {
                // SAFETY: `out_indirect` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxIndirectBuffer` alive through this write.
                unsafe { out_indirect.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Writes an indexed-draw command into an indirect buffer slot.
///
/// # Safety
///
/// A non-null `command` must address one readable, aligned draw command for this call.
pub unsafe extern "C" fn ez_gfx_indirect_write_draw(
    indirect: EzGfxIndirectBuffer,
    index: u32,
    command: *const EzGfxDrawIndexedCommand,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if command.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `command` is non-null, and the caller keeps readable, properly aligned storage for one `EzGfxDrawIndexedCommand` alive through this read.
        let command = unsafe { command.read() };
        state::write_indirect(
            context,
            indirect,
            index,
            ez_gfx_runtime::indirect::DrawIndexedCommand {
                index_count: command.index_count,
                instance_count: command.instance_count,
                first_index: command.first_index,
                vertex_offset: command.vertex_offset,
                first_instance: command.first_instance,
            },
        )
    })
}

#[unsafe(no_mangle)]
/// Sets the number of draw commands consumed from an indirect buffer.
pub extern "C" fn ez_gfx_indirect_set_draw_count(
    indirect: EzGfxIndirectBuffer,
    count: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::set_indirect_count(context, indirect, count))
}

#[unsafe(no_mangle)]
/// Releases an indirect draw buffer.
pub extern "C" fn ez_gfx_indirect_release(indirect: EzGfxIndirectBuffer, context: EzGfxContext) {
    catch_void(|| state::release_indirect(context, indirect));
}

#[unsafe(no_mangle)]
/// Records an indexed graphics pipeline operation with bindings, dynamic state, and push constants.
///
/// # Safety
///
/// Non-null `bindings` must be readable for `binding_count` aligned entries, including valid non-null NUL-terminated binding names. Non-null `dynamic_state` must address one readable aligned value, and non-null `push_constants` must be readable for `push_constant_size` bytes.
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
        let Ok(state) =
            ez_gfx_hal::DynamicPipelineState::from_abi(state[0], state[1], state[2], state[3])
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
        state::render_add_graphics(context, shader, indirect, &bindings, state, push)
    })
}

#[unsafe(no_mangle)]
/// Records a compute dispatch with its shader, bindings, dimensions, and push constants.
///
/// # Safety
///
/// Non-null `bindings` must be readable for `binding_count` aligned entries, including valid non-null NUL-terminated binding names. Non-null `push_constants` must be readable for `push_constant_size` bytes.
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
        state::render_add_compute(
            context,
            shader,
            [dispatch_x, dispatch_y, dispatch_z],
            &bindings,
            push,
        )
    })
}

#[unsafe(no_mangle)]
/// Enqueues a texture readback in the current frame graph.
pub extern "C" fn ez_gfx_graph_enqueue_texture_readback(
    texture: EzGfxTexture,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::frame_enqueue_readback(context, texture))
}

#[unsafe(no_mangle)]
/// Submits the recorded frame to the graphics backend.
pub extern "C" fn ez_gfx_frame_submit(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::frame_submit(context))
}

/// Submits the recorded frame and presents the active surface; partial submission is never hidden.
#[unsafe(no_mangle)]
/// Submits the recorded frame and presents the active surface.
pub extern "C" fn ez_gfx_finish_render(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| {
        let submitted = state::frame_submit(context);
        if submitted != EzGfxResult::Ok {
            return submitted;
        }
        state::present(context)
    })
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
        let bytes = match state::frame_readback(context) {
            Ok(bytes) => bytes,
            Err(status) => return status,
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
        match state::create_surface(context, options) {
            Ok(handle) => {
                // SAFETY: `out_surface` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxSurface` alive through this write.
                unsafe { out_surface.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Initializes the context device for the specified presentation surface.
pub extern "C" fn ez_gfx_context_init_device(
    surface: EzGfxSurface,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::init_device(context, surface))
}
#[unsafe(no_mangle)]
/// Requests new pixel dimensions for a presentation surface.
pub extern "C" fn ez_gfx_surface_resize(
    surface: EzGfxSurface,
    width: u32,
    height: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::resize_surface(context, surface, width, height))
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
        match state::surface_extent(context, surface) {
            Ok((width, height)) => {
                // SAFETY: Both output pointers are non-null; the caller keeps aligned writable storage for one `u32` at each pointer alive through these writes.
                unsafe {
                    out_width.write(width);
                    out_height.write(height);
                }
                EzGfxResult::Ok
            }
            Err(status) => status,
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
        match state::surface_resize_pending(context, surface) {
            Ok(pending) => {
                // SAFETY: `out_pending` is non-null, and the caller keeps writable, properly aligned storage for one `i32` alive through this write.
                unsafe {
                    out_pending.write(i32::from(pending));
                }
                EzGfxResult::Ok
            }
            Err(status) => status,
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
    catch_status(|| match enabled {
        0 => state::set_snapshot_cache(context, surface, false),
        1 => state::set_snapshot_cache(context, surface, true),
        _ => EzGfxResult::InvalidArgument,
    })
}

#[unsafe(no_mangle)]
/// Destroys a presentation surface owned by the context.
pub extern "C" fn ez_gfx_surface_destroy(surface: EzGfxSurface, context: EzGfxContext) {
    catch_void(|| state::destroy_surface(context, surface));
}

#[unsafe(no_mangle)]
/// Creates a named vertex heap with the requested capacity and element stride.
///
/// # Safety
///
/// A non-null `name` must reference a NUL-terminated string for this call.
pub unsafe extern "C" fn ez_gfx_vertex_heap_create(
    name: *const std::ffi::c_char,
    capacity: u64,
    stride: u64,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let name = match read_c_string(name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        state::create_vertex_heap(context, &name, capacity, stride)
    })
}

#[unsafe(no_mangle)]
/// Destroys the named vertex heap.
///
/// # Safety
///
/// A non-null `name` must reference a NUL-terminated string for this call.
pub unsafe extern "C" fn ez_gfx_vertex_heap_destroy(
    name: *const std::ffi::c_char,
    context: EzGfxContext,
) {
    catch_void(|| {
        if let Ok(name) = read_c_string(name) {
            state::destroy_vertex_heap(context, &name);
        }
    });
}

#[unsafe(no_mangle)]
/// Creates the context's index heap with the requested capacity.
///
/// # Safety
///
/// A non-null `debug_name` must reference a NUL-terminated string for this call.
pub unsafe extern "C" fn ez_gfx_index_heap_create(
    capacity: u64,
    debug_name: *const std::ffi::c_char,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if read_c_string(debug_name).is_err() {
            return EzGfxResult::InvalidArgument;
        }
        state::create_index_heap(context, capacity)
    })
}

#[unsafe(no_mangle)]
/// Destroys the context's index heap.
pub extern "C" fn ez_gfx_index_heap_destroy(context: EzGfxContext) {
    catch_void(|| state::destroy_index_heap(context));
}

#[unsafe(no_mangle)]
/// Uploads 32-bit indices and returns their first index in the index heap.
///
/// # Safety
///
/// Non-null `data` must be readable for `count * 4` bytes, and non-null `out_start_index` must address one writable, aligned `u32`, for this call.
pub unsafe extern "C" fn ez_gfx_vertex_upload_indices(
    data: *const std::ffi::c_void,
    count: u32,
    out_start_index: *mut u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let size = match usize::try_from(u64::from(count) * 4) {
            Ok(size) if count != 0 => size,
            _ => return EzGfxResult::InvalidArgument,
        };
        if data.is_null() || out_start_index.is_null() || size > EZ_GFX_MAX_BOUNDARY_BYTES {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `size` is the checked nonzero `count * 4` and does not exceed `EZ_GFX_MAX_BOUNDARY_BYTES`; the caller keeps `data` readable for that many alignment-1 bytes through the upload.
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), size) };
        match state::upload_indices(context, count, bytes) {
            Ok(first) => {
                // SAFETY: `out_start_index` is non-null, and the caller keeps writable, properly aligned storage for one `u32` alive through this write.
                unsafe { out_start_index.write(first) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
/// Uploads elements to a named vertex heap and returns their first element index.
///
/// # Safety
///
/// A non-null `heap_name` must reference a NUL-terminated string, non-null `data` must be readable for `element_count * element_size` bytes, and non-null `out_start_index` must address one writable, aligned `u32`, for this call.
pub unsafe extern "C" fn ez_gfx_vertex_upload(
    heap_name: *const std::ffi::c_char,
    data: *const std::ffi::c_void,
    element_count: u32,
    element_size: u64,
    out_start_index: *mut u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let name = match read_c_string(heap_name) {
            Ok(name) => name,
            Err(status) => return status,
        };
        let size = match u64::from(element_count)
            .checked_mul(element_size)
            .and_then(|size| usize::try_from(size).ok())
        {
            Some(size)
                if element_count != 0 && element_size != 0 && size <= EZ_GFX_MAX_BOUNDARY_BYTES =>
            {
                size
            }
            _ => return EzGfxResult::InvalidArgument,
        };
        if data.is_null() || out_start_index.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `size` is the checked nonzero `element_count * element_size` and does not exceed `EZ_GFX_MAX_BOUNDARY_BYTES`; the caller keeps `data` readable for that many alignment-1 bytes through the upload.
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), size) };
        match state::upload_vertices(context, &name, element_count, element_size, bytes) {
            Ok(first) => {
                // SAFETY: `out_start_index` is non-null, and the caller keeps writable, properly aligned storage for one `u32` alive through this write.
                unsafe { out_start_index.write(first) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

/// Acquires a real mapped upload buffer; products that exceed `u64` or allocation limits fail.
#[unsafe(no_mangle)]
/// Acquires a mapped structured upload buffer sized for the requested elements.
///
/// # Safety
///
/// A non-null `debug_name` must reference a NUL-terminated string, and non-null `out_structured` must address one writable, aligned handle, for this call.
pub unsafe extern "C" fn ez_gfx_structured_acquire(
    element_size: u32,
    element_count: u32,
    debug_name: *const std::ffi::c_char,
    out_structured: *mut EzGfxStructuredBuffer,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if element_size == 0
            || element_count == 0
            || debug_name.is_null()
            || out_structured.is_null()
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: ABI requires a readable NUL-terminated string for the duration of the call.
        let name = unsafe { std::ffi::CStr::from_ptr(debug_name) };
        if name.to_bytes().is_empty()
            || name.to_bytes().len() > EZ_GFX_MAX_BOUNDARY_BYTES
            || name.to_str().is_err()
        {
            return EzGfxResult::InvalidArgument;
        }
        let size = u64::from(element_size) * u64::from(element_count);
        match state::acquire_structured(context, size) {
            Ok(handle) => {
                // SAFETY: `out_structured` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxStructuredBuffer` alive through this write.
                unsafe { out_structured.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

/// Copies the complete caller-provided byte range; zero bytes still require a non-null pointer.
#[unsafe(no_mangle)]
/// Copies caller-provided bytes into a structured upload buffer.
///
/// # Safety
///
/// Non-null `data` must be readable for `data_size` bytes for this call.
pub unsafe extern "C" fn ez_gfx_structured_write(
    structured: EzGfxStructuredBuffer,
    data: *const std::ffi::c_void,
    data_size: u64,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        let Ok(data_size) = usize::try_from(data_size) else {
            return EzGfxResult::InvalidArgument;
        };
        if data.is_null()
            || data_size > EZ_GFX_MAX_BOUNDARY_BYTES
            || data_size > isize::MAX as usize
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `data` is non-null and `data_size` is checked against `isize::MAX` and the byte limit; the caller keeps that alignment-1 range readable through the structured write.
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), data_size) };
        state::write_structured(context, structured, bytes)
    })
}

#[unsafe(no_mangle)]
/// Releases a structured upload buffer.
pub extern "C" fn ez_gfx_structured_release(
    structured: EzGfxStructuredBuffer,
    context: EzGfxContext,
) {
    catch_void(|| state::release_structured(context, structured));
}

/// Decodes a packed handle into its context and optional child slot generations.
///
/// # Safety
///
/// A non-null `out_parts` must address one writable, aligned handle description for this call.
pub unsafe extern "C" fn ez_gfx_handle_inspect(
    handle: u64,
    out_parts: *mut EzGfxHandleParts,
) -> EzGfxResult {
    catch_status(|| {
        if out_parts.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let Ok(parts) = PackedHandle::from_raw(handle).and_then(PackedHandle::parts) else {
            return EzGfxResult::InvalidContext;
        };
        let abi = match parts {
            HandleParts::Context(context) => EzGfxHandleParts {
                context_slot: context.slot(),
                context_generation: context.generation(),
                child_slot: 0,
                child_generation: 0,
                is_context: 1,
                _padding: [0; 3],
            },
            HandleParts::Child { owner, child } => EzGfxHandleParts {
                context_slot: owner.slot(),
                context_generation: owner.generation(),
                child_slot: child.slot(),
                child_generation: child.generation(),
                is_context: 0,
                _padding: [0; 3],
            },
        };
        // SAFETY: `out_parts` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxHandleParts` alive through this write.
        unsafe { out_parts.write(abi) };
        EzGfxResult::Ok
    })
}

/// Computes the fixed-size semantic identifier for a UTF-8 name.
///
/// # Safety
///
/// Non-null `name` must be readable for `length` bytes, and non-null `out_id` writable for 16 bytes, for this call.
pub unsafe extern "C" fn ez_gfx_semantic_id(
    name: *const u8,
    length: usize,
    out_id: *mut u8,
) -> EzGfxResult {
    catch_status(|| {
        if out_id.is_null()
            || name.is_null()
            || length == 0
            || length > EZ_GFX_MAX_BOUNDARY_BYTES
            || length > isize::MAX as usize
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `length` is checked nonzero and within both the byte and `isize` limits; the caller keeps `name` readable for `length` `u8` values (alignment 1) through UTF-8 validation.
        let bytes = unsafe { core::slice::from_raw_parts(name, length) };
        let Ok(name) = core::str::from_utf8(bytes) else {
            return EzGfxResult::InvalidArgument;
        };
        let id = match SemanticId::from_name(name) {
            Ok(id) => id.bytes(),
            Err(_) => return EzGfxResult::InvalidArgument,
        };
        // SAFETY: `out_id` is non-null, and the caller provides an alignment-1 writable range of `id.len()` bytes that is disjoint from the live local `id` storage through the copy.
        unsafe { core::ptr::copy_nonoverlapping(id.as_ptr(), out_id, id.len()) };
        EzGfxResult::Ok
    })
}

fn catch_status(operation: impl FnOnce() -> EzGfxResult) -> EzGfxResult {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(EzGfxResult::NativeFailure)
}
fn read_c_string(pointer: *const std::ffi::c_char) -> Result<String, EzGfxResult> {
    if pointer.is_null() {
        return Err(EzGfxResult::InvalidArgument);
    }
    // SAFETY: every C string boundary requires readable NUL-terminated storage for the call.
    let bytes = unsafe { std::ffi::CStr::from_ptr(pointer) }.to_bytes();
    if bytes.is_empty() || bytes.len() > EZ_GFX_MAX_BOUNDARY_BYTES {
        return Err(EzGfxResult::InvalidArgument);
    }
    core::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| EzGfxResult::InvalidArgument)
}

/// Binding arrays are bounded; every item requires one UTF-8 name and exactly one non-null typed handle.
fn read_bindings(
    pointer: *const EzGfxBinding,
    count: u32,
) -> Result<Vec<ez_gfx_runtime::binding::PublicBinding>, EzGfxResult> {
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
        let name = read_c_string(binding.name)?;
        let resource = match (
            binding.structured != 0,
            binding.indirect != 0,
            binding.render_target != 0,
        ) {
            (true, false, false) => {
                ez_gfx_runtime::binding::ResourceIdentity::Structured(binding.structured)
            }
            (false, true, false) => {
                ez_gfx_runtime::binding::ResourceIdentity::Indirect(binding.indirect)
            }
            (false, false, true) => {
                ez_gfx_runtime::binding::ResourceIdentity::RenderTarget(binding.render_target)
            }
            _ => return Err(EzGfxResult::InvalidArgument),
        };
        bindings.push(ez_gfx_runtime::binding::PublicBinding { name, resource });
    }
    Ok(bindings)
}

fn catch_void(operation: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(operation));
}
