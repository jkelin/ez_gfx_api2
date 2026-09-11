//! Observable indexed-indirect count and zero-tail rendering contracts.
#![cfg(not(target_vendor = "apple"))]

#[cfg(windows)]
mod common;
#[cfg(not(windows))]
#[path = "common/headless.rs"]
mod common;

use std::sync::LazyLock;

use common::TestContext;
use ez_gfx_compiler::{EasyGraphicsCompiler, Target};
use ez_gfx_ffi::{
    EzGfxDrawIndexedCommand, EzGfxDynamicState, EzGfxEvent, EzGfxEventKind,
    EzGfxReadbackSourceKind, EzGfxResult, ez_gfx_context_register_callback,
    ez_gfx_counter_buffer_acquire, ez_gfx_counter_buffer_publish_count,
    ez_gfx_counter_buffer_write_draws, ez_gfx_fragment_shader_destroy, ez_gfx_fragment_shader_load,
    ez_gfx_frame_begin, ez_gfx_frame_end, ez_gfx_frame_execute_graphics,
    ez_gfx_index_allocation_create, ez_gfx_index_allocation_get_range,
    ez_gfx_index_allocation_remove, ez_gfx_surface_set_snapshot_cache,
    ez_gfx_vertex_shader_destroy, ez_gfx_vertex_shader_load,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

#[derive(Default)]
struct Captures(Vec<Vec<u8>>);

unsafe extern "C" fn collect_snapshot(event: *const EzGfxEvent, user_data: *mut core::ffi::c_void) {
    // SAFETY: registration retains live test-owned pointers for every callback invocation.
    let (event, captures) = unsafe { (&*event, &mut *user_data.cast::<Captures>()) };
    if event.kind == EzGfxEventKind::Snapshot {
        assert_eq!(event.readback_source_kind, EzGfxReadbackSourceKind::None);
        assert_eq!(event.readback_source, 0);
        // SAFETY: snapshot bytes remain readable for this callback and are copied before return.
        captures.0.push(unsafe {
            core::slice::from_raw_parts(event.readback_bytes, event.readback_byte_count).to_vec()
        });
    }
}

#[test]
fn vulkan_zeroed_counter_tail_matches_single_counted_draw_pixels() {
    exercise_zero_tail(1);
}

#[cfg(windows)]
#[test]
fn dx12_zeroed_counter_tail_matches_single_counted_draw_pixels() {
    exercise_zero_tail(2);
}

#[cfg(windows)]
#[test]
fn dx12_gpu_count_limits_nonzero_commands_and_uses_aligned_command_offset() {
    let mut fixture = Fixture::new(2);
    let (status, baseline) = fixture.render(1, &[command(0)], None);
    assert_eq!(status, EzGfxResult::Ok);
    let (status, counted) = fixture.render(2, &[command(0), command(1)], Some(1));
    assert_eq!(status, EzGfxResult::Ok);
    let baseline = baseline.unwrap();
    let counted = counted.unwrap();

    assert_eq!(counted, baseline);
    assert_red(&counted);
}

fn exercise_zero_tail(backend: u8) {
    let mut fixture = Fixture::new(backend);
    let (status, baseline) = fixture.render(1, &[command(0)], None);
    if status == EzGfxResult::Unsupported && !cfg!(windows) {
        eprintln!("Vulkan headless presentation is unsupported by the selected ICD");
        return;
    }
    assert_eq!(status, EzGfxResult::Ok);
    let baseline = baseline.unwrap();
    let (status, padded) = fixture.render(4, &[command(0)], None);
    assert_eq!(status, EzGfxResult::Ok);
    let padded = padded.unwrap();

    assert_eq!(padded, baseline);
    assert_red(&padded);
}

fn command(first_instance: u32) -> EzGfxDrawIndexedCommand {
    EzGfxDrawIndexedCommand {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance,
    }
}

fn assert_red(bytes: &[u8]) {
    assert_eq!(bytes.len(), WIDTH as usize * HEIGHT as usize * 4);
    let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) * 4) as usize;
    let pixel = &bytes[center..center + 4];
    assert!(pixel[0] > 200 && pixel[1] < 40 && pixel[2] < 40 && pixel[3] == 255);
}

struct Fixture {
    native: TestContext,
    vertex_shader: u64,
    fragment_shader: u64,
    index_allocation: u64,
    captures: Box<Captures>,
}

impl Fixture {
    fn new(backend: u8) -> Self {
        let native = TestContext::create_with_validation(backend, backend == 1);
        let artifact = artifact();
        let mut vertex_shader = 0;
        let mut fragment_shader = 0;
        for (entry, output, load) in [
            (
                b"vertexmain".as_slice(),
                &raw mut vertex_shader,
                ez_gfx_vertex_shader_load as unsafe extern "C" fn(_, _, _, _, _, _) -> _,
            ),
            (
                b"fragmentmain".as_slice(),
                &raw mut fragment_shader,
                ez_gfx_fragment_shader_load as unsafe extern "C" fn(_, _, _, _, _, _) -> _,
            ),
        ] {
            assert_eq!(
                // SAFETY: artifact, entry, and output storage remain live and aligned.
                unsafe {
                    load(
                        native.context,
                        artifact.as_ptr(),
                        artifact.len(),
                        entry.as_ptr(),
                        entry.len(),
                        output,
                    )
                },
                EzGfxResult::Ok
            );
        }
        let indices = [0_u32, 1, 2];
        let mut index_allocation = 0;
        assert_eq!(
            // SAFETY: index and output storage remain live and aligned through the call.
            unsafe {
                ez_gfx_index_allocation_create(
                    native.context,
                    indices.as_ptr().cast(),
                    u32::try_from(indices.len()).unwrap(),
                    &raw mut index_allocation,
                )
            },
            EzGfxResult::Ok
        );
        let mut captures = Box::<Captures>::default();
        assert_eq!(
            // SAFETY: captures outlives registration and is cleared in Drop.
            unsafe {
                ez_gfx_context_register_callback(
                    native.context,
                    Some(collect_snapshot),
                    (&raw mut *captures).cast(),
                )
            },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_surface_set_snapshot_cache(native.context, native.surface, 1),
            EzGfxResult::Ok
        );
        Self {
            native,
            vertex_shader,
            fragment_shader,
            index_allocation,
            captures,
        }
    }

    fn render(
        &mut self,
        capacity: u32,
        commands: &[EzGfxDrawIndexedCommand],
        published_count: Option<u32>,
    ) -> (EzGfxResult, Option<Vec<u8>>) {
        let mut first_index = 0;
        let mut index_count = 0;
        assert_eq!(
            // SAFETY: both outputs are writable aligned test-owned storage.
            unsafe {
                ez_gfx_index_allocation_get_range(
                    self.native.context,
                    self.index_allocation,
                    &raw mut first_index,
                    &raw mut index_count,
                )
            },
            EzGfxResult::Ok
        );
        assert_eq!((first_index, index_count), (0, 3));

        let mut frame = 0;
        assert_eq!(
            // SAFETY: frame output storage is writable and aligned.
            unsafe {
                ez_gfx_frame_begin(self.native.context, self.native.surface, 0, &raw mut frame)
            },
            EzGfxResult::Ok
        );
        let label = b"counter-pixels";
        let mut counter = 0;
        assert_eq!(
            // SAFETY: label and output storage remain live through the call.
            unsafe {
                ez_gfx_counter_buffer_acquire(
                    self.native.context,
                    u32::try_from(size_of::<EzGfxDrawIndexedCommand>()).unwrap(),
                    capacity,
                    label.as_ptr(),
                    label.len(),
                    &raw mut counter,
                )
            },
            EzGfxResult::Ok
        );
        let commands: Vec<_> = commands
            .iter()
            .map(|command| EzGfxDrawIndexedCommand {
                first_index,
                ..*command
            })
            .collect();
        assert_eq!(
            // SAFETY: the command slice remains readable for the declared count.
            unsafe {
                ez_gfx_counter_buffer_write_draws(
                    self.native.context,
                    counter,
                    0,
                    commands.as_ptr(),
                    u32::try_from(commands.len()).unwrap(),
                )
            },
            EzGfxResult::Ok
        );
        if let Some(count) = published_count {
            assert_eq!(
                ez_gfx_counter_buffer_publish_count(self.native.context, counter, count),
                EzGfxResult::Ok
            );
        }
        let state = EzGfxDynamicState {
            cull_mode: 0,
            front_face: 0,
            primitive_type: 0,
            blend_mode: 0,
        };
        assert_eq!(
            // SAFETY: state remains readable through the call.
            unsafe {
                ez_gfx_frame_execute_graphics(
                    self.native.context,
                    frame,
                    self.vertex_shader,
                    self.fragment_shader,
                    counter,
                    &raw const state,
                )
            },
            EzGfxResult::Ok
        );
        let status = ez_gfx_frame_end(self.native.context, frame);
        let pixels =
            (status == EzGfxResult::Ok).then(|| self.captures.0.pop().expect("rendered snapshot"));
        (status, pixels)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: clearing a live registration retains no user-data pointer.
        let _ = unsafe {
            ez_gfx_context_register_callback(self.native.context, None, core::ptr::null_mut())
        };
        ez_gfx_index_allocation_remove(self.native.context, self.index_allocation);
        ez_gfx_vertex_shader_destroy(self.native.context, self.vertex_shader);
        ez_gfx_fragment_shader_destroy(self.native.context, self.fragment_shader);
    }
}

fn artifact() -> &'static [u8] {
    static ARTIFACT: LazyLock<Vec<u8>> = LazyLock::new(|| {
        // The process-specific directory is removed after compilation; LazyLock compiles once per test binary.
        let root =
            std::env::temp_dir().join(format!("ez-gfx-counter-pixels-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("ez_gfx_api.slang"),
            include_str!("../../../ez_gfx_api.slang"),
        )
        .unwrap();
        let source = root.join("counter_pixels.slang");
        std::fs::write(
            &source,
            r#"import ez_gfx_api;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertexmain(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VertexOutput output;
    output.position = float4(positions[vertex_id], 0.5, 1.0);
    output.color = instance_id == 0 ? float4(1.0, 0.0, 0.0, 1.0) : float4(0.0, 1.0, 0.0, 1.0);
    return output;
}

struct FragmentOutput {
    [ColorTarget("swapchain", "write")]
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragmentmain(VertexOutput input) {
    FragmentOutput output;
    output.color = input.color;
    return output;
}
"#,
        )
        .unwrap();
        #[cfg(windows)]
        let targets = &[Target::Spirv, Target::Dxil];
        #[cfg(not(windows))]
        let targets = &[Target::Spirv];
        let artifact = EasyGraphicsCompiler::compile_shader(&source, targets, false)
            .expect("compile indirect pixel shader");
        let _ = std::fs::remove_dir_all(root);
        artifact.save_shader()
    });
    &ARTIFACT
}
