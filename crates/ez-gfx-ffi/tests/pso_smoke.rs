//! Native Vulkan and DX12 compute-pipeline smoke tests through the C ABI.
#![cfg(not(target_vendor = "apple"))]

#[cfg(windows)]
mod common;
#[cfg(not(any(windows, target_vendor = "apple")))]
#[path = "common/headless.rs"]
mod common;

use common::TestContext;
use ez_gfx_compiler::{CompilerError, Target, compile_shader};
use ez_gfx_ffi::{
    EzGfxBinding, EzGfxRenderTargetDesc, EzGfxResult, ez_gfx_buffer_acquire, ez_gfx_buffer_release,
    ez_gfx_buffer_write, ez_gfx_context_wait_idle, ez_gfx_counted_buffer_acquire,
    ez_gfx_counted_buffer_publish_count, ez_gfx_counted_buffer_release, ez_gfx_frame_end,
    ez_gfx_render_add_compute_pipeline, ez_gfx_render_target_create, ez_gfx_render_target_destroy,
    ez_gfx_render_target_frame_begin, ez_gfx_shader_destroy, ez_gfx_shader_load_artifact,
};

const COMPUTE_SOURCE: &str = r#"[__AttributeUsage(_AttributeTargets.Var)]
struct StructuredBufferAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)]
struct IndirectBufferAttribute { string name; };
struct Draw { uint index_count; uint instance_count; uint first_index; int vertex_offset; uint first_instance; };

[StructuredBuffer("values")]
RWStructuredBuffer<uint> values;
[IndirectBuffer("draw_commands")]
RWStructuredBuffer<Draw> draw_commands;

[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id: SV_DispatchThreadID) {
    values[id.x] += 1;
    Draw draw = { 3, 1, 0, 0, 0 };
    draw_commands[id.x] = draw;
}
"#;

fn begin_offscreen_frame(context: u64) -> (u64, u64) {
    let name = b"compute-target";
    let format = 1_u8;
    let desc = EzGfxRenderTargetDesc {
        name: name.as_ptr(),
        name_length: name.len(),
        usage: 0,
        relative_scale: 1.0,
        samples: 1,
        candidate_formats: &raw const format,
        candidate_count: 1,
        sampleable: 0,
        use_clear: 0,
        clear_color: [0.0; 4],
    };
    let mut target = 0;
    assert_eq!(
        // SAFETY: descriptor, format, and output storage remain live through the call.
        unsafe { ez_gfx_render_target_create(&raw const desc, 1, 1, &raw mut target, context) },
        EzGfxResult::Ok
    );
    let mut frame = 0;
    assert_eq!(
        // SAFETY: frame output storage is live and aligned.
        unsafe { ez_gfx_render_target_frame_begin(context, target, &raw mut frame) },
        EzGfxResult::Ok
    );
    (frame, target)
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_compiles_binds_and_executes_compute_pipeline() {
    run_compute_pipeline(1);
}

#[cfg(windows)]
#[test]
fn dx12_compiles_sm65_pso_and_dispatches_on_hardware() {
    run_compute_pipeline(2);
}

/// Each backend uses a distinct temporary directory so parallel native tests cannot race artifact output.
#[cfg(not(target_vendor = "apple"))]
fn run_compute_pipeline(backend: u8) {
    let root = std::env::temp_dir().join(format!("ez-gfx-pso-{}-{backend}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("compute.slang");
    std::fs::write(&source, COMPUTE_SOURCE).unwrap();

    #[cfg(windows)]
    let targets = &[Target::Spirv, Target::Dxil, Target::Metal];
    #[cfg(not(windows))]
    let targets = &[Target::Spirv];
    let artifact = match compile_shader(&source, targets, cfg!(windows)) {
        Ok(artifact) => artifact,
        Err(CompilerError::NativeUnavailable) => return,
        Err(error) => panic!("shader compilation failed: {error}"),
    };

    let native = TestContext::create(backend);
    let context = native.context;

    let mut shader = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_shader_load_artifact(
                    artifact.as_ptr(),
                    artifact.len(),
                    &raw mut shader,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );

    let (frame, target) = begin_offscreen_frame(context);

    let binding_name = b"values";
    let mut buffer = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_buffer_acquire(
                    4,
                    1,
                    binding_name.as_ptr(),
                    binding_name.len(),
                    &raw mut buffer,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let value = 41_u32.to_ne_bytes();
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_buffer_write(buffer, 0, value.as_ptr().cast(), 1, 4, context) }
        },
        EzGfxResult::Ok
    );
    let indirect_name = b"draw_commands";
    let mut indirect = 0;
    assert_eq!(
        {
            // SAFETY: The name and output ranges remain valid for the call.
            unsafe {
                ez_gfx_counted_buffer_acquire(
                    u32::try_from(core::mem::size_of::<ez_gfx_ffi::EzGfxDrawIndexedCommand>())
                        .expect("draw command size fits u32"),
                    1,
                    indirect_name.as_ptr(),
                    indirect_name.len(),
                    &raw mut indirect,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_counted_buffer_publish_count(indirect, 1, context),
        EzGfxResult::Ok
    );
    let bindings = [
        EzGfxBinding {
            name: binding_name.as_ptr(),
            name_length: binding_name.len(),
            buffer,
            counted_buffer: 0,
            render_target: 0,
        },
        EzGfxBinding {
            name: indirect_name.as_ptr(),
            name_length: indirect_name.len(),
            buffer: 0,
            counted_buffer: indirect,
            render_target: 0,
        },
    ];

    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_compute_pipeline(
                    shader,
                    1,
                    1,
                    1,
                    bindings.as_ptr(),
                    2,
                    core::ptr::null(),
                    0,
                    frame,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: The source range remains valid; rejection occurs before any copy.
            unsafe { ez_gfx_buffer_write(buffer, 0, value.as_ptr().cast(), 1, 4, context) }
        },
        EzGfxResult::NotReady
    );
    assert_eq!(
        ez_gfx_counted_buffer_publish_count(indirect, 1, context),
        EzGfxResult::NotReady
    );
    assert_eq!(ez_gfx_frame_end(frame), EzGfxResult::Ok);
    ez_gfx_render_target_destroy(target, context);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: Context-owned CPU storage is writable again after frame completion.
            unsafe { ez_gfx_buffer_write(buffer, 0, value.as_ptr().cast(), 1, 4, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_counted_buffer_publish_count(indirect, 1, context),
        EzGfxResult::Ok
    );
    ez_gfx_buffer_release(buffer, context);
    ez_gfx_counted_buffer_release(indirect, context);

    ez_gfx_shader_destroy(shader, context);
    drop(native);
    let _ = std::fs::remove_dir_all(root);
}
