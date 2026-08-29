// C exports keep ordinary C signatures; every pointer/count pair is validated before bounded single-call access.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

mod api;
mod state;

pub use api::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

use ez_gfx_core::{
    Backend, SemanticId,
    handle::{HandleParts, PackedHandle},
};
use ez_gfx_runtime::{ContextOptions, SurfaceOptions, SurfacePlatform};

pub const EZ_GFX_ABI_VERSION: u32 = 17;
pub const EZ_GFX_MAX_BOUNDARY_BYTES: usize = 16 * 1024 * 1024;

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_abi_version() -> u32 {
    EZ_GFX_ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_context_create(
    desc: *const EzGfxContextDesc,
    out_context: *mut EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_context.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: ABI requires a readable aligned descriptor for this call.
        let desc = unsafe { desc.read() };
        let options = match ContextOptions::new(
            desc.enable_debug,
            desc.enable_validation,
            desc.surface_platform,
        ) {
            Ok(value) => value,
            Err(_) => return EzGfxResult::InvalidArgument,
        };
        match state::create_context(options) {
            Ok(handle) => {
                unsafe { out_context.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_context_create_backend(
    desc: *const EzGfxBackendContextDesc,
    out_context: *mut EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_context.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let desc = unsafe { desc.read() };
        let backend = match desc.backend {
            1 => Backend::Vulkan,
            2 => Backend::Dx12,
            3 => Backend::Metal,
            _ => return EzGfxResult::InvalidArgument,
        };
        let options = match ContextOptions::new_for_backend(
            desc.enable_debug,
            desc.enable_validation,
            desc.surface_platform,
            backend,
        ) {
            Ok(value) => value,
            Err(_) => return EzGfxResult::InvalidArgument,
        };
        match state::create_context(options) {
            Ok(handle) => {
                unsafe { out_context.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_context_wait_idle(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::wait_idle(context))
}
#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_context_destroy(context: EzGfxContext) {
    catch_void(|| state::destroy_context(context));
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_poll_runtime_event(
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
pub extern "C" fn ez_gfx_poll_diagnostic(
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
pub extern "C" fn ez_gfx_shader_load_artifact(
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
        let bytes = unsafe { core::slice::from_raw_parts(data, data_size) };
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
            let request = match ez_gfx_runtime::shader::ShaderRequest::new(name, stage) {
                Ok(request) => request,
                Err(_) => return EzGfxResult::InvalidArgument,
            };
            requests.push(request);
        }
        match state::load_shader(context, bytes, &requests) {
            Ok(shader) => {
                unsafe { out_shader.write(shader.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_shader_destroy(shader: EzGfxShader, context: EzGfxContext) {
    catch_void(|| state::destroy_shader(context, shader));
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_texture_load(
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
        match state::load_texture(context, source, bytes, desc.generate_mips != 0, config) {
            Ok(texture) => {
                unsafe { out_texture.write(texture.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_texture_get_binding(
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
                unsafe { out_binding.write(binding) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_texture_get_residency(
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
                unsafe {
                    out_resident_mips.write(resident);
                    out_total_mips.write(total)
                };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_texture_unload(texture: EzGfxTexture, context: EzGfxContext) {
    catch_void(|| state::unload_texture(context, texture));
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_begin_render(surface: EzGfxSurface, context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::begin_render(context, surface))
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_frame_begin(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::frame_begin(context))
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_acquire_indirect(
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
                unsafe { out_indirect.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_indirect_write_draw(
    indirect: EzGfxIndirectBuffer,
    index: u32,
    command: *const EzGfxDrawIndexedCommand,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if command.is_null() {
            return EzGfxResult::InvalidArgument;
        }
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
pub extern "C" fn ez_gfx_indirect_set_draw_count(
    indirect: EzGfxIndirectBuffer,
    count: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::set_indirect_count(context, indirect, count))
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_indirect_release(indirect: EzGfxIndirectBuffer, context: EzGfxContext) {
    catch_void(|| state::release_indirect(context, indirect));
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_render_add_vertex_pipeline(
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
            let value = unsafe { dynamic_state.read() };
            [
                value.cull_mode,
                value.front_face,
                value.primitive_type,
                value.blend_mode,
            ]
        };
        let state = match ez_gfx_hal::DynamicPipelineState::from_abi(
            state[0], state[1], state[2], state[3],
        ) {
            Ok(value) => value,
            Err(_) => return EzGfxResult::InvalidArgument,
        };
        let push = if push_constant_size == 0 {
            &[][..]
        } else {
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
pub extern "C" fn ez_gfx_render_add_compute_pipeline(
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
pub extern "C" fn ez_gfx_graph_enqueue_texture_readback(
    texture: EzGfxTexture,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::frame_enqueue_readback(context, texture))
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_frame_submit(context: EzGfxContext) -> EzGfxResult {
    catch_status(|| state::frame_submit(context))
}

/// Submits the recorded frame and presents the active surface; partial submission is never hidden.
#[unsafe(no_mangle)]
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
pub extern "C" fn ez_gfx_frame_readback(
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
        unsafe { out_size.write(bytes.len()) };
        if capacity == 0 {
            return EzGfxResult::Ok;
        }
        if capacity < bytes.len() {
            return EzGfxResult::InvalidArgument;
        }
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        EzGfxResult::Ok
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_surface_create(
    desc: *const EzGfxSurfaceDesc,
    out_surface: *mut EzGfxSurface,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if desc.is_null() || out_surface.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let desc = unsafe { desc.read() };
        let platform = match desc.platform {
            0 => SurfacePlatform::Win32,
            1 => SurfacePlatform::Glfw,
            2 => SurfacePlatform::MetalLayer,
            _ => return EzGfxResult::InvalidArgument,
        };
        let options = match SurfaceOptions::new(
            desc.window as usize,
            desc.display as usize,
            platform,
            desc.width,
            desc.height,
            desc.cache_presented_snapshots,
        ) {
            Ok(value) => value,
            Err(_) => return EzGfxResult::InvalidArgument,
        };
        match state::create_surface(context, options) {
            Ok(handle) => {
                unsafe { out_surface.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_context_init_device(
    surface: EzGfxSurface,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::init_device(context, surface))
}
#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_surface_resize(
    surface: EzGfxSurface,
    width: u32,
    height: u32,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| state::resize_surface(context, surface, width, height))
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_surface_get_extent(
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
pub extern "C" fn ez_gfx_surface_resize_pending(
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
pub extern "C" fn ez_gfx_surface_destroy(surface: EzGfxSurface, context: EzGfxContext) {
    catch_void(|| state::destroy_surface(context, surface));
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_vertex_heap_create(
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
pub extern "C" fn ez_gfx_vertex_heap_destroy(name: *const std::ffi::c_char, context: EzGfxContext) {
    catch_void(|| {
        if let Ok(name) = read_c_string(name) {
            state::destroy_vertex_heap(context, &name);
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_index_heap_create(
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
pub extern "C" fn ez_gfx_index_heap_destroy(context: EzGfxContext) {
    catch_void(|| state::destroy_index_heap(context));
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_vertex_upload_indices(
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
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), size) };
        match state::upload_indices(context, count, bytes) {
            Ok(first) => {
                unsafe { out_start_index.write(first) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_vertex_upload(
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
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), size) };
        match state::upload_vertices(context, &name, element_count, element_size, bytes) {
            Ok(first) => {
                unsafe { out_start_index.write(first) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

/// Acquires a real mapped upload buffer; products that exceed `u64` or allocation limits fail.
#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_structured_acquire(
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
                unsafe { out_structured.write(handle.get()) };
                EzGfxResult::Ok
            }
            Err(status) => status,
        }
    })
}

/// Copies the complete caller-provided byte range; zero bytes still require a non-null pointer.
#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_structured_write(
    structured: EzGfxStructuredBuffer,
    data: *const std::ffi::c_void,
    data_size: u64,
    context: EzGfxContext,
) -> EzGfxResult {
    catch_status(|| {
        if data.is_null()
            || data_size > EZ_GFX_MAX_BOUNDARY_BYTES as u64
            || data_size > isize::MAX as u64
        {
            return EzGfxResult::InvalidArgument;
        }
        let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), data_size as usize) };
        state::write_structured(context, structured, bytes)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ez_gfx_structured_release(
    structured: EzGfxStructuredBuffer,
    context: EzGfxContext,
) {
    catch_void(|| state::release_structured(context, structured));
}

pub extern "C" fn ez_gfx_handle_inspect(
    handle: u64,
    out_parts: *mut EzGfxHandleParts,
) -> EzGfxResult {
    catch_status(|| {
        if out_parts.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let parts = match PackedHandle::from_raw(handle).and_then(PackedHandle::parts) {
            Ok(parts) => parts,
            Err(_) => return EzGfxResult::InvalidContext,
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
        unsafe { out_parts.write(abi) };
        EzGfxResult::Ok
    })
}

pub extern "C" fn ez_gfx_semantic_id(
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
        let bytes = unsafe { core::slice::from_raw_parts(name, length) };
        let name = match core::str::from_utf8(bytes) {
            Ok(name) => name,
            Err(_) => return EzGfxResult::InvalidArgument,
        };
        let id = match SemanticId::from_name(name) {
            Ok(id) => id.bytes(),
            Err(_) => return EzGfxResult::InvalidArgument,
        };
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
