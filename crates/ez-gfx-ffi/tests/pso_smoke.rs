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
    EzGfxBinding, EzGfxResult, ez_gfx_acquire_indirect, ez_gfx_context_wait_idle,
    ez_gfx_frame_begin, ez_gfx_frame_submit, ez_gfx_indirect_publish_compute_count,
    ez_gfx_render_add_compute_pipeline, ez_gfx_shader_destroy, ez_gfx_shader_load_artifact,
    ez_gfx_structured_acquire, ez_gfx_structured_write,
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

    assert_eq!(ez_gfx_frame_begin(context), EzGfxResult::Ok);

    let binding_name = b"values";
    let mut structured = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_structured_acquire(
                    4,
                    1,
                    binding_name.as_ptr(),
                    binding_name.len(),
                    &raw mut structured,
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
            unsafe { ez_gfx_structured_write(structured, 0, value.as_ptr().cast(), 1, 4, context) }
        },
        EzGfxResult::Ok
    );
    let indirect_name = b"draw_commands";
    let mut indirect = 0;
    assert_eq!(
        {
            // SAFETY: The name and output ranges remain valid for the call.
            unsafe {
                ez_gfx_acquire_indirect(
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
        ez_gfx_indirect_publish_compute_count(indirect, 1, context),
        EzGfxResult::Ok
    );
    let bindings = [
        EzGfxBinding {
            name: binding_name.as_ptr(),
            name_length: binding_name.len(),
            structured,
            indirect: 0,
            render_target: 0,
        },
        EzGfxBinding {
            name: indirect_name.as_ptr(),
            name_length: indirect_name.len(),
            structured: 0,
            indirect,
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
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: The source range remains valid; rejection occurs before any copy.
            unsafe { ez_gfx_structured_write(structured, 0, value.as_ptr().cast(), 1, 4, context) }
        },
        EzGfxResult::NotReady
    );
    assert_eq!(
        ez_gfx_indirect_publish_compute_count(indirect, 1, context),
        EzGfxResult::NotReady
    );
    assert_eq!(ez_gfx_frame_submit(context), EzGfxResult::Ok);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: The source range remains valid; the stale handle is validated first.
            unsafe { ez_gfx_structured_write(structured, 0, value.as_ptr().cast(), 1, 4, context) }
        },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        ez_gfx_indirect_publish_compute_count(indirect, 1, context),
        EzGfxResult::InvalidContext
    );

    ez_gfx_shader_destroy(shader, context);
    drop(native);
    let _ = std::fs::remove_dir_all(root);
}
