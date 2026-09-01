//! Native Vulkan and DX12 compute-pipeline smoke tests through the C ABI.
#![cfg(windows)]

use std::ffi::CString;

use ez_gfx_compiler::{CompilerError, Target, compile_shader};
use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxBinding, EzGfxResult, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_wait_idle, ez_gfx_frame_begin, ez_gfx_frame_submit,
    ez_gfx_render_add_compute_pipeline, ez_gfx_shader_destroy, ez_gfx_shader_load_artifact,
    ez_gfx_structured_acquire, ez_gfx_structured_release, ez_gfx_structured_write,
};
#[cfg(windows)]
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
fn run_compute_pipeline(backend: u8) {
    let root = std::env::temp_dir().join(format!("ez-gfx-pso-{}-{backend}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("compute.slang");
    std::fs::write(
        &source,
        r#"[__AttributeUsage(_AttributeTargets.Var)]
struct StructuredBufferAttribute { string name; };

[StructuredBuffer("values")]
RWStructuredBuffer<uint> values;

[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id: SV_DispatchThreadID) { values[id.x] += 1; }
"#,
    )
    .unwrap();

    let artifact =
        match compile_shader(&source, &[Target::Spirv, Target::Dxil, Target::Metal], true) {
            Ok(artifact) => artifact,
            Err(CompilerError::NativeUnavailable) => return,
            Err(error) => panic!("shader compilation failed: {error}"),
        };

    let desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 0,
        backend,
    };
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );

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

    let binding_name = CString::new("values").unwrap();
    let mut structured = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_structured_acquire(4, 1, binding_name.as_ptr(), &raw mut structured, context)
            }
        },
        EzGfxResult::Ok
    );
    let value = 41_u32.to_ne_bytes();
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_structured_write(
                    structured,
                    value.as_ptr().cast(),
                    value.len() as u64,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let binding = EzGfxBinding {
        name: binding_name.as_ptr(),
        structured,
        indirect: 0,
        render_target: 0,
    };

    assert_eq!(ez_gfx_frame_begin(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_compute_pipeline(
                    shader,
                    1,
                    1,
                    1,
                    &raw const binding,
                    1,
                    core::ptr::null(),
                    0,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_submit(context), EzGfxResult::Ok);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);

    ez_gfx_structured_release(structured, context);
    ez_gfx_shader_destroy(shader, context);
    ez_gfx_context_destroy(context);
    let _ = std::fs::remove_dir_all(root);
}
