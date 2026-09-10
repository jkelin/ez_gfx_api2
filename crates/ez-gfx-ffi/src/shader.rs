use crate::{
    EZ_GFX_MAX_BOUNDARY_BYTES, EzGfxComputeShader, EzGfxContext, EzGfxFragmentShader, EzGfxResult,
    EzGfxVertexShader, catch_status, catch_void, read_bounded_string,
};
use ez_gfx::raw::{self, ContextHandle, ShaderHandle};

unsafe fn load_stage_shader(
    context: EzGfxContext,
    data: *const u8,
    data_size: usize,
    entry_point: *const u8,
    entry_point_size: usize,
    stage: ez_gfx::Stage,
    out_shader: *mut u64,
) -> EzGfxResult {
    if data.is_null()
        || out_shader.is_null()
        || !out_shader.is_aligned()
        || data_size == 0
        || data_size > EZ_GFX_MAX_BOUNDARY_BYTES
    {
        return EzGfxResult::InvalidArgument;
    }
    let Ok(entry_point) = read_bounded_string(entry_point, entry_point_size) else {
        return EzGfxResult::InvalidArgument;
    };
    // SAFETY: The artifact range is non-null, alignment-1, and bounded; the caller keeps it
    // readable for this call.
    let bytes = unsafe { core::slice::from_raw_parts(data, data_size) };
    let context = try_handle!(ContextHandle, context);
    match raw::load_shader(context, bytes, stage, &entry_point) {
        Ok(shader) => {
            // SAFETY: The output was checked non-null/aligned and remains writable for this call.
            unsafe { out_shader.write(shader.into_raw()) };
            EzGfxResult::Ok
        }
        Err(status) => status.into(),
    }
}

#[unsafe(no_mangle)]
/// Loads one exact compute entry point from a validated artifact.
///
/// # Safety
///
/// Input ranges must be readable and `out_shader` writable and aligned.
pub unsafe extern "C" fn ez_gfx_compute_shader_load(
    context: EzGfxContext,
    data: *const u8,
    data_size: usize,
    entry_point: *const u8,
    entry_point_size: usize,
    out_shader: *mut EzGfxComputeShader,
) -> EzGfxResult {
    // SAFETY: The caller guarantees the documented input readability and output
    // writability; the stage loader validates ranges before dereferencing.
    catch_status(|| unsafe {
        load_stage_shader(
            context,
            data,
            data_size,
            entry_point,
            entry_point_size,
            ez_gfx::Stage::Compute,
            out_shader,
        )
    })
}

#[unsafe(no_mangle)]
/// Loads one exact vertex entry point from a validated artifact.
///
/// # Safety
///
/// Input ranges must be readable and `out_shader` writable and aligned.
pub unsafe extern "C" fn ez_gfx_vertex_shader_load(
    context: EzGfxContext,
    data: *const u8,
    data_size: usize,
    entry_point: *const u8,
    entry_point_size: usize,
    out_shader: *mut EzGfxVertexShader,
) -> EzGfxResult {
    // SAFETY: The caller guarantees the documented input readability and output
    // writability; the stage loader validates ranges before dereferencing.
    catch_status(|| unsafe {
        load_stage_shader(
            context,
            data,
            data_size,
            entry_point,
            entry_point_size,
            ez_gfx::Stage::Vertex,
            out_shader,
        )
    })
}

#[unsafe(no_mangle)]
/// Loads one exact fragment entry point from a validated artifact.
///
/// # Safety
///
/// Input ranges must be readable and `out_shader` writable and aligned.
pub unsafe extern "C" fn ez_gfx_fragment_shader_load(
    context: EzGfxContext,
    data: *const u8,
    data_size: usize,
    entry_point: *const u8,
    entry_point_size: usize,
    out_shader: *mut EzGfxFragmentShader,
) -> EzGfxResult {
    // SAFETY: The caller guarantees the documented input readability and output
    // writability; the stage loader validates ranges before dereferencing.
    catch_status(|| unsafe {
        load_stage_shader(
            context,
            data,
            data_size,
            entry_point,
            entry_point_size,
            ez_gfx::Stage::Fragment,
            out_shader,
        )
    })
}

fn destroy_shader(context: EzGfxContext, shader: u64) {
    if let (Ok(context), Ok(shader)) = (
        ContextHandle::from_raw(context),
        ShaderHandle::from_raw(shader),
    ) {
        raw::destroy_shader(context, shader);
    }
}

#[unsafe(no_mangle)]
/// Invalidates immediately. An active frame retains its record through its terminal operation.
pub extern "C" fn ez_gfx_compute_shader_destroy(context: EzGfxContext, shader: EzGfxComputeShader) {
    catch_void(|| destroy_shader(context, shader));
}

#[unsafe(no_mangle)]
/// Invalidates immediately. An active frame retains its record through its terminal operation.
pub extern "C" fn ez_gfx_vertex_shader_destroy(context: EzGfxContext, shader: EzGfxVertexShader) {
    catch_void(|| destroy_shader(context, shader));
}

#[unsafe(no_mangle)]
/// Invalidates immediately. An active frame retains its record through its terminal operation.
pub extern "C" fn ez_gfx_fragment_shader_destroy(
    context: EzGfxContext,
    shader: EzGfxFragmentShader,
) {
    catch_void(|| destroy_shader(context, shader));
}
