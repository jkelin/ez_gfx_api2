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
    ez_gfx_buffer_write, ez_gfx_context_wait_idle, ez_gfx_counter_buffer_acquire,
    ez_gfx_counter_buffer_publish_count, ez_gfx_counter_buffer_release, ez_gfx_frame_bind,
    ez_gfx_frame_end, ez_gfx_frame_execute_compute, ez_gfx_render_target_create,
    ez_gfx_render_target_destroy, ez_gfx_render_target_frame_begin, ez_gfx_shader_destroy,
    ez_gfx_shader_load_artifact, ez_gfx_value_buffer_acquire,
};

const COMPUTE_SOURCE: &str = r#"import "ez_gfx_api";

struct Draw { uint index_count; uint instance_count; uint first_index; int vertex_offset; uint first_instance; };

[Buffer("values")]
RWStructuredBuffer<uint> values;
[CounterBuffer("draw_commands")]
CounterBuffer<Draw> draw_commands;

[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id: SV_DispatchThreadID) {
    values[id.x] += 1;
    Draw draw = { 3, 1, 0, 0, 0 };
    draw_commands.set_count(1);
    draw_commands.set(id.x, draw);
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
        unsafe { ez_gfx_render_target_create(context, &raw const desc, 1, 1, &raw mut target) },
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
fn reject_invalid_bindings_without_claiming(
    context: u64,
    frame: u64,
    buffer: u64,
    value: &[u8; 4],
    binding_name: &[u8],
) -> u64 {
    let invalid_name = b"invalid";
    let invalid_binding = EzGfxBinding {
        name: invalid_name.as_ptr(),
        name_length: invalid_name.len(),
        buffer: 0,
        counter_buffer: 0,
        render_target: 0,
    };
    assert_eq!(
        // SAFETY: The binding record and name remain readable through validation.
        unsafe { ez_gfx_frame_bind(context, frame, &raw const invalid_binding) },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Failed validation must leave the valid buffer writable.
        unsafe { ez_gfx_buffer_write(context, buffer, 0, value.as_ptr().cast(), 1, 4) },
        EzGfxResult::Ok
    );
    assert_eq!(
        // SAFETY: Null dynamic state selects the default; the zero counter is rejected.
        unsafe {
            ez_gfx_ffi::ez_gfx_frame_execute_graphics(context, frame, 0, 0, core::ptr::null())
        },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        // SAFETY: Invalid counter validation occurs before any buffer can be claimed.
        unsafe { ez_gfx_buffer_write(context, buffer, 0, value.as_ptr().cast(), 1, 4) },
        EzGfxResult::Ok
    );

    ez_gfx_buffer_release(context, buffer);
    let mut replacement = 0;
    assert_eq!(
        // SAFETY: The name and output storage remain valid through reacquisition.
        unsafe {
            ez_gfx_buffer_acquire(
                context,
                4,
                1,
                binding_name.as_ptr(),
                binding_name.len(),
                &raw mut replacement,
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        // SAFETY: The reacquired buffer accepts the same live source value.
        unsafe { ez_gfx_buffer_write(context, replacement, 0, value.as_ptr().cast(), 1, 4) },
        EzGfxResult::Ok
    );
    replacement
}

/// Each backend uses a distinct temporary directory so parallel native tests cannot race artifact output.
#[cfg(not(target_vendor = "apple"))]
#[allow(
    clippy::too_many_lines,
    reason = "one hardware scenario keeps compile, bind, dispatch, and lifetime assertions ordered"
)]
fn run_compute_pipeline(backend: u8) {
    let root = std::env::temp_dir().join(format!("ez-gfx-pso-{}-{backend}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("compute.slang");
    std::fs::write(&source, COMPUTE_SOURCE).unwrap();
    std::fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ez_gfx_api.slang"),
        root.join("ez_gfx_api.slang"),
    )
    .unwrap();

    #[cfg(windows)]
    let targets = &[Target::Spirv, Target::Dxil, Target::Metal];
    #[cfg(not(windows))]
    let targets = &[Target::Spirv];
    let artifact = match compile_shader(&source, targets, cfg!(windows)) {
        Ok(artifact) => artifact,
        Err(CompilerError::NativeUnavailable) => return,
        Err(error) => panic!("shader compilation failed: {error}"),
    };

    let native = TestContext::create_with_validation(backend, backend == 1);
    let context = native.context;

    let mut shader = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_shader_load_artifact(
                    context,
                    artifact.as_ptr(),
                    artifact.len(),
                    &raw mut shader,
                )
            }
        },
        EzGfxResult::Ok
    );

    let (frame, target) = begin_offscreen_frame(context);

    let binding_name = b"values";
    let value = 41_u32.to_ne_bytes();
    let mut buffer = 0;
    assert_eq!(
        // SAFETY: Value, name, and output ranges remain live through the call.
        unsafe {
            ez_gfx_value_buffer_acquire(
                context,
                value.as_ptr().cast(),
                u32::try_from(value.len()).unwrap(),
                binding_name.as_ptr(),
                binding_name.len(),
                &raw mut buffer,
            )
        },
        EzGfxResult::Ok
    );
    let indirect_name = b"draw_commands";
    let mut indirect = 0;
    assert_eq!(
        {
            // SAFETY: The name and output ranges remain valid for the call.
            unsafe {
                ez_gfx_counter_buffer_acquire(
                    context,
                    u32::try_from(core::mem::size_of::<ez_gfx_ffi::EzGfxDrawIndexedCommand>())
                        .expect("draw command size fits u32"),
                    1,
                    indirect_name.as_ptr(),
                    indirect_name.len(),
                    &raw mut indirect,
                )
            }
        },
        EzGfxResult::Ok
    );
    buffer = reject_invalid_bindings_without_claiming(context, frame, buffer, &value, binding_name);
    let values_binding = EzGfxBinding {
        name: binding_name.as_ptr(),
        name_length: binding_name.len(),
        buffer,
        counter_buffer: 0,
        render_target: 0,
    };
    assert_eq!(
        // SAFETY: The binding and exact name range remain readable through the call.
        unsafe { ez_gfx_frame_bind(context, frame, &raw const values_binding) },
        EzGfxResult::Ok
    );
    let mut replacement = 0;
    assert_eq!(
        // SAFETY: Value, name, and output ranges remain live through the call.
        unsafe {
            ez_gfx_value_buffer_acquire(
                context,
                value.as_ptr().cast(),
                u32::try_from(value.len()).unwrap(),
                binding_name.as_ptr(),
                binding_name.len(),
                &raw mut replacement,
            )
        },
        EzGfxResult::Ok
    );
    let replacement_binding = EzGfxBinding {
        buffer: replacement,
        ..values_binding
    };
    assert_eq!(
        // SAFETY: The replacement binding and name remain readable through the call.
        unsafe { ez_gfx_frame_bind(context, frame, &raw const replacement_binding) },
        EzGfxResult::Ok
    );
    assert_eq!(
        // SAFETY: A never-executed replaced buffer remains writable and unclaimed.
        unsafe { ez_gfx_buffer_write(context, buffer, 0, value.as_ptr().cast(), 1, 4) },
        EzGfxResult::Ok
    );
    ez_gfx_buffer_release(context, buffer);
    buffer = replacement;
    let indirect_binding = EzGfxBinding {
        name: indirect_name.as_ptr(),
        name_length: indirect_name.len(),
        buffer: 0,
        counter_buffer: indirect,
        render_target: 0,
    };
    assert_eq!(
        // SAFETY: The binding and exact name range remain readable through the call.
        unsafe { ez_gfx_frame_bind(context, frame, &raw const indirect_binding) },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_frame_execute_compute(context, frame, shader, 1, 1, 1),
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_frame_execute_compute(context, frame, shader, 1, 1, 1),
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: The source range remains valid; rejection occurs before any copy.
            unsafe { ez_gfx_buffer_write(context, buffer, 0, value.as_ptr().cast(), 1, 4) }
        },
        EzGfxResult::NotReady
    );
    assert_eq!(
        ez_gfx_counter_buffer_publish_count(context, indirect, 1),
        EzGfxResult::NotReady
    );
    assert_eq!(ez_gfx_frame_end(context, frame), EzGfxResult::Ok);
    ez_gfx_render_target_destroy(context, target);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: The source range remains valid; the consumed handle is rejected before copy.
            unsafe { ez_gfx_buffer_write(context, buffer, 0, value.as_ptr().cast(), 1, 4) }
        },
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        ez_gfx_counter_buffer_publish_count(context, indirect, 1),
        EzGfxResult::InvalidContext
    );
    ez_gfx_buffer_release(context, buffer);
    ez_gfx_counter_buffer_release(context, indirect);

    ez_gfx_shader_destroy(context, shader);
    drop(native);
    let _ = std::fs::remove_dir_all(root);
}
