//! Native mesh-shader pixel contracts through the exported C ABI.
//!
//! Mesh-only draws dispatch a nontrivial `[4, 2, 1]` grid onto a 64x64
//! single-sample target with no culling, a black clear, and opaque output.
//! Task+mesh draws carry a real payload fed by a task input buffer that a
//! compute stage writes, changing both the triangle and its color with no CPU
//! wait between the compute write and the mesh read.

#[cfg(windows)]
mod common;
#[cfg(all(target_vendor = "apple", not(test)))]
#[path = "common/metal.rs"]
mod common;
#[cfg(not(any(windows, target_vendor = "apple")))]
#[path = "common/headless.rs"]
mod common;

#[cfg(all(test, target_vendor = "apple"))]
use std::process::Command;
#[cfg(test)]
use std::sync::LazyLock;
#[cfg(not(test))]
use std::sync::OnceLock;

#[cfg(any(not(test), not(target_vendor = "apple")))]
use common::TestContext;
#[cfg(test)]
use ez_gfx_compiler::{EasyGraphicsCompiler, Target};
#[cfg(any(not(test), not(target_vendor = "apple")))]
use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxBinding, EzGfxDrawIndexedCommand, EzGfxEvent, EzGfxEventKind,
    EzGfxMeshShaders, EzGfxMeshState, EzGfxRenderTargetDesc, EzGfxResult, EzGfxShaderCapabilities,
    ez_gfx_buffer_acquire, ez_gfx_compute_shader_destroy, ez_gfx_compute_shader_load,
    ez_gfx_context_create_backend, ez_gfx_context_destroy, ez_gfx_context_register_callback,
    ez_gfx_context_shader_capabilities, ez_gfx_counter_buffer_acquire,
    ez_gfx_counter_buffer_write_draws, ez_gfx_fragment_shader_destroy, ez_gfx_fragment_shader_load,
    ez_gfx_frame_abort, ez_gfx_frame_bind, ez_gfx_frame_end,
    ez_gfx_frame_enqueue_render_target_readback, ez_gfx_frame_execute_compute,
    ez_gfx_frame_execute_graphics, ez_gfx_frame_execute_mesh, ez_gfx_index_allocation_create,
    ez_gfx_index_allocation_remove, ez_gfx_mesh_shader_destroy, ez_gfx_mesh_shader_load,
    ez_gfx_render_target_create, ez_gfx_render_target_destroy, ez_gfx_render_target_frame_begin,
    ez_gfx_render_target_get_clear, ez_gfx_task_shader_destroy, ez_gfx_task_shader_load,
    ez_gfx_vertex_shader_destroy, ez_gfx_vertex_shader_load,
};

#[cfg(any(not(test), not(target_vendor = "apple")))]
const EXTENT: u32 = 64;
#[cfg(any(not(test), not(target_vendor = "apple")))]
static OPAQUE_STATE: EzGfxMeshState = EzGfxMeshState {
    cull_mode: 0,
    front_face: 0,
    blend_mode: 0,
    reserved: 0,
};

#[cfg(any(not(test), not(target_vendor = "apple")))]
/// An aborted target frame retires its handle exactly once: a repeat abort is
/// rejected instead of double retiring native frame state. Submit-side
/// retirement is proven inline by every successful render, which ends its
/// frame twice and requires `InvalidContext` the second time.
fn assert_abort_retires_once(context: u64, target: u64) {
    let mut frame = 0;
    assert_eq!(
        // SAFETY: frame output storage is writable and aligned.
        unsafe { ez_gfx_render_target_frame_begin(context, target, &raw mut frame) },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_abort(context, frame), EzGfxResult::Ok);
    assert_eq!(
        ez_gfx_frame_abort(context, frame),
        EzGfxResult::InvalidContext,
        "aborted frames retire exactly once"
    );
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_mesh_pixels() {
    exercise_mesh(1);
}

#[cfg(windows)]
#[test]
fn dx12_mesh_pixels() {
    exercise_mesh(2);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn vulkan_task_mesh_pixels() {
    exercise_task_mesh(1);
}

#[cfg(windows)]
#[test]
fn dx12_task_mesh_pixels() {
    exercise_task_mesh(2);
}

#[cfg(target_vendor = "apple")]
#[test]
fn metal_mesh_pixels() {
    run_metal_helper("mesh", mesh_artifact());
}

#[cfg(target_vendor = "apple")]
#[test]
fn metal_task_mesh_pixels() {
    run_metal_helper("task-mesh", task_artifact());
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn metal_mesh_pixels() {
    exercise_mesh(3);
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn metal_task_mesh_pixels() {
    exercise_task_mesh(3);
}

#[cfg(test)]
fn compile_fixture(name: &str, source: &str) -> Vec<u8> {
    // The process-specific directory is removed after compilation; each
    // LazyLock compiles its fixture once per test binary.
    let root = std::env::temp_dir().join(format!(
        "ez-gfx-shader-stages-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("ez_gfx_api.slang"),
        include_str!("../../../ez_gfx_api.slang"),
    )
    .unwrap();
    let path = root.join(format!("{name}.slang"));
    std::fs::write(&path, source).unwrap();
    #[cfg(windows)]
    let targets = &[Target::Spirv, Target::Dxil, Target::Metal];
    #[cfg(not(any(windows, target_vendor = "apple")))]
    let targets = &[Target::Spirv];
    #[cfg(target_vendor = "apple")]
    let targets = &[Target::Metal];
    let artifact = EasyGraphicsCompiler::compile_shader(&path, targets, cfg!(windows))
        .unwrap_or_else(|error| panic!("compile {name} mesh fixture: {error:?}"));
    let _ = std::fs::remove_dir_all(root);
    artifact.save_shader()
}

#[cfg(test)]
static MESH_ARTIFACT: LazyLock<Vec<u8>> =
    LazyLock::new(|| compile_fixture("mesh", include_str!("shader_stages/mesh.slang")));
#[cfg(test)]
static TASK_ARTIFACT: LazyLock<Vec<u8>> =
    LazyLock::new(|| compile_fixture("task_mesh", include_str!("shader_stages/task_mesh.slang")));

#[cfg(not(test))]
static MESH_ARTIFACT: OnceLock<Vec<u8>> = OnceLock::new();
#[cfg(not(test))]
static TASK_ARTIFACT: OnceLock<Vec<u8>> = OnceLock::new();

fn mesh_artifact() -> &'static [u8] {
    #[cfg(test)]
    {
        &MESH_ARTIFACT
    }
    #[cfg(not(test))]
    {
        MESH_ARTIFACT.get().expect("mesh artifact must be set")
    }
}

fn task_artifact() -> &'static [u8] {
    #[cfg(test)]
    {
        &TASK_ARTIFACT
    }
    #[cfg(not(test))]
    {
        TASK_ARTIFACT.get().expect("task-mesh artifact must be set")
    }
}

#[cfg(all(test, target_vendor = "apple"))]
fn run_metal_helper(scenario: &str, artifact: &[u8]) {
    // Each test gets an isolated file so a failed child cannot contaminate its sibling.
    let root = std::env::temp_dir().join(format!(
        "ez-gfx-metal-shader-stages-{scenario}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let artifact_path = root.join("shader.ezgfxshader");
    std::fs::write(&artifact_path, artifact).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_metal_shader_stages"))
        .arg(scenario)
        .arg(&artifact_path)
        .status()
        .unwrap();
    let _ = std::fs::remove_dir_all(root);
    assert!(
        status.success(),
        "metal_shader_stages helper failed: {status}"
    );
}

#[cfg(all(not(test), target_vendor = "apple"))]
pub fn run_metal_scenario(scenario: &str, artifact: Vec<u8>) {
    // A helper process executes one scenario, so artifact initialization must happen once.
    match scenario {
        "mesh" => {
            assert!(
                MESH_ARTIFACT.set(artifact).is_ok(),
                "mesh artifact initialized twice"
            );
            exercise_mesh(3);
        }
        "task-mesh" => {
            assert!(
                TASK_ARTIFACT.set(artifact).is_ok(),
                "task-mesh artifact initialized twice"
            );
            exercise_task_mesh(3);
        }
        _ => panic!("unknown Metal shader-stage scenario: {scenario}"),
    }
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
/// Loads one exact stage entry point, asserting success for capable devices.
macro_rules! load_stage {
    ($load:ident, $context:expr, $artifact:expr, $entry:expr) => {{
        let mut shader = 0;
        assert_eq!(
            // SAFETY: artifact and entry ranges plus output storage are live through the call.
            unsafe {
                $load(
                    $context,
                    $artifact.as_ptr(),
                    $artifact.len(),
                    $entry.as_ptr(),
                    $entry.len(),
                    &mut shader,
                )
            },
            EzGfxResult::Ok,
            "stage load failed",
        );
        shader
    }};
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
#[derive(Default)]
struct Captures(Vec<(u64, Vec<u8>)>);

#[cfg(any(not(test), not(target_vendor = "apple")))]
unsafe extern "C" fn collect_readback(event: *const EzGfxEvent, user_data: *mut core::ffi::c_void) {
    // SAFETY: registration retains live test-owned pointers for every callback invocation.
    let (event, captures) = unsafe { (&*event, &mut *user_data.cast::<Captures>()) };
    if event.kind != EzGfxEventKind::Readback {
        return;
    }
    // SAFETY: readback bytes remain readable for this callback and are copied before return.
    let bytes = unsafe {
        core::slice::from_raw_parts(event.readback_bytes, event.readback_byte_count).to_vec()
    };
    captures.0.push((event.readback_request_id, bytes));
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
fn take_readback(captures: &mut Captures, request: u64) -> Vec<u8> {
    let position = captures
        .0
        .iter()
        .position(|(id, _)| *id == request)
        .expect("target readback delivered for its request");
    captures.0.remove(position).1
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
fn create_black_target(context: u64) -> u64 {
    let name = b"mesh-black-target";
    let format = 1_u8;
    let desc = EzGfxRenderTargetDesc {
        name: name.as_ptr(),
        name_length: name.len(),
        usage: 0,
        relative_scale: 1.0,
        samples: 1,
        candidate_formats: &raw const format,
        candidate_count: 1,
        sampleable: 1,
        use_clear: 1,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let mut target = 0;
    assert_eq!(
        // SAFETY: descriptor, format, and output storage remain live through the call.
        unsafe { ez_gfx_render_target_create(context, &raw const desc, 64, 64, &raw mut target) },
        EzGfxResult::Ok
    );
    let mut use_clear = 0;
    let mut clear = [0.0_f32; 4];
    assert_eq!(
        // SAFETY: output storage is live and aligned through the call.
        unsafe {
            ez_gfx_render_target_get_clear(context, target, &raw mut use_clear, &raw mut clear[0])
        },
        EzGfxResult::Ok
    );
    assert_eq!((use_clear, clear), (1, [0.0, 0.0, 0.0, 1.0]));
    target
}

#[cfg(all(target_vendor = "apple", not(test)))]
fn create_context(backend: u8) -> TestContext {
    TestContext::create(backend)
}

#[cfg(not(target_vendor = "apple"))]
fn create_context(backend: u8) -> TestContext {
    TestContext::create_with_validation(backend, backend == 1)
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
fn query_capabilities(context: u64) -> EzGfxShaderCapabilities {
    let mut capabilities = EzGfxShaderCapabilities { task: 0, mesh: 0 };
    assert_eq!(
        // SAFETY: output storage is live and aligned through the call.
        unsafe { ez_gfx_context_shader_capabilities(context, &raw mut capabilities) },
        EzGfxResult::Ok,
    );
    capabilities
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
/// A failed capability query on a device-less context must report a typed
/// error and leave caller-owned output untouched.
fn assert_capability_output_unchanged_on_failure(backend: u8) {
    let desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        backend,
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
    let mut context = 0;
    // SAFETY: descriptor and output storage are live and aligned through the call.
    if unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
        != EzGfxResult::Ok
    {
        return;
    }
    let mut capabilities = EzGfxShaderCapabilities {
        task: 0xff,
        mesh: 0xff,
    };
    // SAFETY: output storage is live and aligned through the call.
    let status = unsafe { ez_gfx_context_shader_capabilities(context, &raw mut capabilities) };
    if status == EzGfxResult::Ok {
        // A backend that initializes without a surface reports coherent stages.
        assert!(capabilities.task <= 1 && capabilities.mesh <= 1);
        assert!(capabilities.task == 0 || capabilities.mesh == 1);
    } else {
        assert_eq!(
            (capabilities.task, capabilities.mesh),
            (0xff, 0xff),
            "failed query must not write output"
        );
    }
    assert_eq!(ez_gfx_context_destroy(context), EzGfxResult::Ok);
}

#[cfg(not(target_vendor = "apple"))]
fn assert_metal_unavailable() {
    let desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        backend: 3,
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
    let mut context = 0;
    // SAFETY: descriptor and output storage are live and aligned through the call.
    let created = unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) };
    assert_ne!(
        created,
        EzGfxResult::Ok,
        "Metal context creation unexpectedly succeeded on a non-Apple host"
    );
    assert_eq!(
        created,
        EzGfxResult::Unsupported,
        "Metal without a device must fail context creation with a typed rejection"
    );
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
fn pixel(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
    let start = ((y * EXTENT + x) * 4) as usize;
    assert_eq!(bytes.len(), (EXTENT * EXTENT * 4) as usize);
    [
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ]
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
fn exercise_mesh(backend: u8) {
    #[cfg(not(target_vendor = "apple"))]
    if backend == 3 {
        assert_metal_unavailable();
        return;
    }
    let native = create_context(backend);
    if query_capabilities(native.context).mesh == 0 {
        // A device without mesh support rejects exact stage loads with a
        // typed capability error instead of drawing nothing.
        for entry in [b"meshmain".as_slice(), b"taskmain".as_slice()] {
            let mut shader: u64 = 0;
            let status = if entry == b"meshmain" {
                // SAFETY: artifact, entry, and output storage are live through the call.
                unsafe {
                    ez_gfx_mesh_shader_load(
                        native.context,
                        mesh_artifact().as_ptr(),
                        mesh_artifact().len(),
                        entry.as_ptr(),
                        entry.len(),
                        &raw mut shader,
                    )
                }
            } else {
                // SAFETY: artifact, entry, and output storage are live through the call.
                unsafe {
                    ez_gfx_task_shader_load(
                        native.context,
                        mesh_artifact().as_ptr(),
                        mesh_artifact().len(),
                        entry.as_ptr(),
                        entry.len(),
                        &raw mut shader,
                    )
                }
            };
            assert_eq!(
                status,
                EzGfxResult::Unsupported,
                "incapable device must reject mesh stages",
            );
        }
        panic!(
            "backend {backend} lacks mesh-shader support on this device; the named mesh pixel path cannot silently pass"
        );
    }
    let mut fixture = MeshFixture::new(native);
    let target = create_black_target(fixture.native.context);
    fixture.target = target;
    // A null state selects the opaque defaults: no culling, no blending.
    let (status, pixels) = fixture.render_mesh(0, [4, 2, 1], core::ptr::null());
    assert_eq!(status, EzGfxResult::Ok);
    let pixels = pixels.expect("mesh readback");
    assert_mesh_frame(&pixels);
    // The backend is proven by the render above, so abort retirement runs
    // after real work rather than as a cold precheck.
    assert_abort_retires_once(fixture.native.context, fixture.target);
    // Explicit opaque state reproduces the identical frame.
    let (status, repeat) = fixture.render_mesh(0, [4, 2, 1], &raw const OPAQUE_STATE);
    assert_eq!(status, EzGfxResult::Ok);
    assert_eq!(repeat.expect("mesh readback"), pixels);
    // Degenerate and overflowing grids fail closed and abort their frame
    // without producing a readback.
    for groups in [[0, 1, 1], [1, 0, 1], [u32::MAX, 1, 1]] {
        let (status, pixels) = fixture.render_mesh_rejected(0, groups, &raw const OPAQUE_STATE);
        assert_eq!(status, EzGfxResult::InvalidArgument, "groups {groups:?}");
        assert!(pixels.is_none(), "failed dispatch must not present");
    }
    // Traditional, mesh, and traditional nodes share one framebuffer pass in
    // record order: blue background, red mesh center, green corner.
    let (status, pixels) = fixture.render_interleaved();
    assert_eq!(status, EzGfxResult::Ok);
    assert_interleaved(&pixels.expect("interleaved readback"));
    // Current groups ride the graph template: distinct valid grids both draw.
    for groups in [[1, 1, 1], [2, 1, 1]] {
        let (status, pixels) = fixture.render_mesh(0, groups, &raw const OPAQUE_STATE);
        assert_eq!(status, EzGfxResult::Ok, "groups {groups:?}");
        assert_mesh_frame(&pixels.expect("mesh readback"));
    }
    // Handles owned by a foreign context are rejected in every mesh slot.
    // A second device proves cross-owner rejection through the C boundary.
    let foreign = create_context(backend);
    let foreign_task = (query_capabilities(foreign.context).task != 0).then(|| {
        load_stage!(
            ez_gfx_task_shader_load,
            foreign.context,
            task_artifact(),
            b"taskmain"
        )
    });
    let foreign_mesh = load_stage!(
        ez_gfx_mesh_shader_load,
        foreign.context,
        mesh_artifact(),
        b"meshmain"
    );
    let foreign_fragment = load_stage!(
        ez_gfx_fragment_shader_load,
        foreign.context,
        mesh_artifact(),
        b"fragmentmain_target"
    );
    let mut foreign_cases = vec![
        (
            "mesh",
            EzGfxMeshShaders {
                task_shader: 0,
                mesh_shader: foreign_mesh,
                fragment_shader: fixture.target_fragment,
            },
        ),
        (
            "fragment",
            EzGfxMeshShaders {
                task_shader: 0,
                mesh_shader: fixture.mesh,
                fragment_shader: foreign_fragment,
            },
        ),
    ];
    if let Some(foreign_task) = foreign_task {
        foreign_cases.push((
            "task",
            EzGfxMeshShaders {
                task_shader: foreign_task,
                mesh_shader: fixture.mesh,
                fragment_shader: fixture.target_fragment,
            },
        ));
    }
    for (what, shaders) in foreign_cases {
        let (status, _) = fixture.render_custom(
            shaders,
            [1, 1, 1],
            &raw const OPAQUE_STATE,
            Some(EzGfxResult::InvalidContext),
        );
        assert_eq!(status, EzGfxResult::InvalidContext, "foreign {what} handle");
    }
    drop(foreign);
    fixture.assert_handle_validation();
    // Destroying every shader owner after recording must still submit the
    // retained frame correctly.
    let (status, pixels) = fixture.render_after_owner_destroy();
    assert_eq!(status, EzGfxResult::Ok);
    assert_mesh_frame(&pixels.expect("retained readback"));
    assert_capability_output_unchanged_on_failure(backend);
}
#[cfg(any(not(test), not(target_vendor = "apple")))]
fn assert_mesh_frame(bytes: &[u8]) {
    // Pure red is exact in either color space. Corners come from the target's
    // exact black clear; the center proves opaque mesh output.
    assert_eq!(pixel(bytes, 32, 32), [255, 0, 0, 255], "mesh center");
    for (x, y) in [(2, 2), (61, 2), (2, 61), (61, 61)] {
        assert_eq!(
            pixel(bytes, x, y),
            [0, 0, 0, 255],
            "black corner ({x}, {y})"
        );
    }
}
#[cfg(any(not(test), not(target_vendor = "apple")))]
fn assert_interleaved(bytes: &[u8]) {
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    assert_eq!(
        pixel(bytes, 32, 32),
        RED,
        "mesh covers traditional background"
    );
    // The corner triangle lands in exactly one Y-flip corner; the other keeps
    // the traditional background, proving record order within one pass.
    let top = pixel(bytes, 6, 10);
    let bottom = pixel(bytes, 6, 53);
    assert!(
        (top == GREEN && bottom == BLUE) || (top == BLUE && bottom == GREEN),
        "corner triangle must overwrite exactly one flip corner: top={top:?} bottom={bottom:?}",
    );
    assert_eq!(
        pixel(bytes, 61, 32),
        BLUE,
        "background survives both triangles"
    );
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
struct MeshFixture {
    native: TestContext,
    captures: Box<Captures>,
    mesh: u64,
    target_fragment: u64,
    vertex: u64,
    indices: u64,
    target: u64,
}
#[cfg(any(not(test), not(target_vendor = "apple")))]
impl MeshFixture {
    fn new(native: TestContext) -> Self {
        let mut captures = Box::<Captures>::default();
        assert_eq!(
            // SAFETY: captures outlives registration and is cleared in Drop.
            unsafe {
                ez_gfx_context_register_callback(
                    native.context,
                    Some(collect_readback),
                    (&raw mut *captures).cast(),
                )
            },
            EzGfxResult::Ok
        );
        let mesh = load_stage!(
            ez_gfx_mesh_shader_load,
            native.context,
            mesh_artifact(),
            b"meshmain"
        );
        let target_fragment = load_stage!(
            ez_gfx_fragment_shader_load,
            native.context,
            mesh_artifact(),
            b"fragmentmain_target"
        );
        let vertex = load_stage!(
            ez_gfx_vertex_shader_load,
            native.context,
            mesh_artifact(),
            b"vertexmain"
        );
        Self {
            native,
            captures,
            mesh,
            target_fragment,
            vertex,
            indices: 0,
            target: 0,
        }
    }

    fn render_mesh(
        &mut self,
        task: u64,
        groups: [u32; 3],
        state: *const EzGfxMeshState,
    ) -> (EzGfxResult, Option<Vec<u8>>) {
        let shaders = EzGfxMeshShaders {
            task_shader: task,
            mesh_shader: self.mesh,
            fragment_shader: self.target_fragment,
        };
        self.render_custom(shaders, groups, state, None)
    }
    fn render_mesh_rejected(
        &mut self,
        task: u64,
        groups: [u32; 3],
        state: *const EzGfxMeshState,
    ) -> (EzGfxResult, Option<Vec<u8>>) {
        let shaders = EzGfxMeshShaders {
            task_shader: task,
            mesh_shader: self.mesh,
            fragment_shader: self.target_fragment,
        };
        self.render_custom(shaders, groups, state, Some(EzGfxResult::InvalidArgument))
    }

    fn render_custom(
        &mut self,
        shaders: EzGfxMeshShaders,
        groups: [u32; 3],
        state: *const EzGfxMeshState,
        expected_failure: Option<EzGfxResult>,
    ) -> (EzGfxResult, Option<Vec<u8>>) {
        let mut frame = 0;
        assert_ne!(self.target, 0, "offscreen target must be initialized");
        assert_eq!(
            // SAFETY: frame output storage is writable and aligned.
            unsafe {
                ez_gfx_render_target_frame_begin(self.native.context, self.target, &raw mut frame)
            },
            EzGfxResult::Ok
        );
        // SAFETY: the stage descriptor and optional state stay readable through the call.
        let status = unsafe {
            ez_gfx_frame_execute_mesh(
                self.native.context,
                frame,
                &raw const shaders,
                groups[0],
                groups[1],
                groups[2],
                state,
            )
        };
        if let Some(expected) = expected_failure {
            assert_ne!(expected, EzGfxResult::Ok, "failure helper cannot accept Ok");
            assert_eq!(status, expected, "rejected execute status");
            let readbacks = self.captures.0.len();
            assert_eq!(
                ez_gfx_frame_end(self.native.context, frame),
                EzGfxResult::NotReady,
                "rejected execute must leave no submittable frame"
            );
            assert_eq!(
                self.captures.0.len(),
                readbacks,
                "failed frame must not read back"
            );
            return (status, None);
        }
        assert_eq!(status, EzGfxResult::Ok, "successful execute status");
        let mut request = 0;
        assert_eq!(
            // SAFETY: request output storage is writable and aligned.
            unsafe {
                ez_gfx_frame_enqueue_render_target_readback(
                    self.native.context,
                    frame,
                    self.target,
                    &raw mut request,
                )
            },
            EzGfxResult::Ok
        );
        let status = ez_gfx_frame_end(self.native.context, frame);
        let pixels =
            (status == EzGfxResult::Ok).then(|| take_readback(&mut self.captures, request));
        assert_eq!(
            ez_gfx_frame_end(self.native.context, frame),
            EzGfxResult::InvalidContext,
            "submitted frames retire exactly once"
        );
        (status, pixels)
    }

    fn acquire_commands(&mut self, command: EzGfxDrawIndexedCommand, label: &[u8]) -> u64 {
        let mut counter = 0;
        assert_eq!(
            // SAFETY: label and output storage remain live through the call.
            unsafe {
                ez_gfx_counter_buffer_acquire(
                    self.native.context,
                    u32::try_from(size_of::<EzGfxDrawIndexedCommand>()).unwrap(),
                    1,
                    label.as_ptr(),
                    label.len(),
                    &raw mut counter,
                )
            },
            EzGfxResult::Ok
        );
        assert_eq!(
            // SAFETY: the command remains readable for the declared count.
            unsafe {
                ez_gfx_counter_buffer_write_draws(
                    self.native.context,
                    counter,
                    0,
                    &raw const command,
                    1,
                )
            },
            EzGfxResult::Ok
        );
        counter
    }

    fn render_interleaved(&mut self) -> (EzGfxResult, Option<Vec<u8>>) {
        assert_eq!(
            self.indices, 0,
            "interleaving index storage is allocated once"
        );
        let indices = [0_u32, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            // SAFETY: index and output storage remain live through the call.
            unsafe {
                ez_gfx_index_allocation_create(
                    self.native.context,
                    indices.as_ptr().cast(),
                    u32::try_from(indices.len()).unwrap(),
                    &raw mut self.indices,
                )
            },
            EzGfxResult::Ok
        );
        let mut frame = 0;
        assert_ne!(self.target, 0, "offscreen target must be initialized");
        assert_eq!(
            // SAFETY: frame output storage is writable and aligned.
            unsafe {
                ez_gfx_render_target_frame_begin(self.native.context, self.target, &raw mut frame)
            },
            EzGfxResult::Ok
        );
        // Fullscreen blue background through the traditional path.
        let background = self.acquire_commands(
            EzGfxDrawIndexedCommand {
                index_count: 3,
                instance_count: 1,
                first_index: 0,
                vertex_offset: 0,
                first_instance: 0,
            },
            b"mesh-trad-background",
        );
        assert_eq!(
            // SAFETY: a null state selects the opaque pipeline defaults.
            unsafe {
                ez_gfx_frame_execute_graphics(
                    self.native.context,
                    frame,
                    self.vertex,
                    self.target_fragment,
                    background,
                    core::ptr::null(),
                )
            },
            EzGfxResult::Ok
        );
        // Red mesh center on top of the background.
        assert_eq!(
            // SAFETY: the stage descriptor and state stay readable through the call.
            unsafe {
                ez_gfx_frame_execute_mesh(
                    self.native.context,
                    frame,
                    &EzGfxMeshShaders {
                        task_shader: 0,
                        mesh_shader: self.mesh,
                        fragment_shader: self.target_fragment,
                    },
                    4,
                    2,
                    1,
                    &raw const OPAQUE_STATE,
                )
            },
            EzGfxResult::Ok
        );
        // Green corner through the traditional path, proving the mesh node
        // shares the framebuffer pass instead of clearing it.
        let corner = self.acquire_commands(
            EzGfxDrawIndexedCommand {
                index_count: 3,
                instance_count: 1,
                first_index: 3,
                vertex_offset: 0,
                first_instance: 1,
            },
            b"mesh-trad-corner",
        );
        assert_eq!(
            // SAFETY: a null state selects the opaque pipeline defaults.
            unsafe {
                ez_gfx_frame_execute_graphics(
                    self.native.context,
                    frame,
                    self.vertex,
                    self.target_fragment,
                    corner,
                    core::ptr::null(),
                )
            },
            EzGfxResult::Ok
        );
        let mut request = 0;
        assert_eq!(
            // SAFETY: request output storage is writable and aligned.
            unsafe {
                ez_gfx_frame_enqueue_render_target_readback(
                    self.native.context,
                    frame,
                    self.target,
                    &raw mut request,
                )
            },
            EzGfxResult::Ok
        );
        let status = ez_gfx_frame_end(self.native.context, frame);
        let pixels =
            (status == EzGfxResult::Ok).then(|| take_readback(&mut self.captures, request));
        (status, pixels)
    }

    fn render_after_owner_destroy(&mut self) -> (EzGfxResult, Option<Vec<u8>>) {
        // Fresh owners prove the recorded frame retains its own references:
        // destroying every shader after recording must still submit correctly.
        let mesh = load_stage!(
            ez_gfx_mesh_shader_load,
            self.native.context,
            mesh_artifact(),
            b"meshmain"
        );
        let fragment = load_stage!(
            ez_gfx_fragment_shader_load,
            self.native.context,
            mesh_artifact(),
            b"fragmentmain_target"
        );
        let mut frame = 0;
        assert_ne!(self.target, 0, "offscreen target must be initialized");
        assert_eq!(
            // SAFETY: frame output storage is writable and aligned.
            unsafe {
                ez_gfx_render_target_frame_begin(self.native.context, self.target, &raw mut frame)
            },
            EzGfxResult::Ok
        );
        let shaders = EzGfxMeshShaders {
            task_shader: 0,
            mesh_shader: mesh,
            fragment_shader: fragment,
        };
        // SAFETY: the stage descriptor and state stay readable through the call.
        let status = unsafe {
            ez_gfx_frame_execute_mesh(
                self.native.context,
                frame,
                &raw const shaders,
                4,
                2,
                1,
                &raw const OPAQUE_STATE,
            )
        };
        ez_gfx_mesh_shader_destroy(self.native.context, mesh);
        ez_gfx_fragment_shader_destroy(self.native.context, fragment);
        assert_eq!(status, EzGfxResult::Ok);
        let mut request = 0;
        assert_eq!(
            // SAFETY: request output storage is writable and aligned.
            unsafe {
                ez_gfx_frame_enqueue_render_target_readback(
                    self.native.context,
                    frame,
                    self.target,
                    &raw mut request,
                )
            },
            EzGfxResult::Ok
        );
        let status = ez_gfx_frame_end(self.native.context, frame);
        let pixels =
            (status == EzGfxResult::Ok).then(|| take_readback(&mut self.captures, request));
        (status, pixels)
    }

    fn assert_handle_validation(&mut self) {
        // A zero mesh handle never reaches the device.
        let (status, _) = self.render_custom(
            EzGfxMeshShaders {
                task_shader: 0,
                mesh_shader: 0,
                fragment_shader: self.target_fragment,
            },
            [1, 1, 1],
            &raw const OPAQUE_STATE,
            Some(EzGfxResult::InvalidContext),
        );
        assert_eq!(status, EzGfxResult::InvalidContext);
        // A live vertex handle is the wrong stage for the mesh slot.
        let (status, _) = self.render_custom(
            EzGfxMeshShaders {
                task_shader: 0,
                mesh_shader: self.vertex,
                fragment_shader: self.target_fragment,
            },
            [1, 1, 1],
            &raw const OPAQUE_STATE,
            Some(EzGfxResult::InvalidContext),
        );
        assert_eq!(status, EzGfxResult::InvalidContext);
        // A zero fragment handle is rejected just like a zero mesh handle.
        let (status, _) = self.render_custom(
            EzGfxMeshShaders {
                task_shader: 0,
                mesh_shader: self.mesh,
                fragment_shader: 0,
            },
            [1, 1, 1],
            &raw const OPAQUE_STATE,
            Some(EzGfxResult::InvalidContext),
        );
        assert_eq!(status, EzGfxResult::InvalidContext);
        // Destroyed shaders stay rejected even when their slot is recycled.
        ez_gfx_mesh_shader_destroy(self.native.context, self.mesh);
        let (status, _) = self.render_custom(
            EzGfxMeshShaders {
                task_shader: 0,
                mesh_shader: self.mesh,
                fragment_shader: self.target_fragment,
            },
            [1, 1, 1],
            &raw const OPAQUE_STATE,
            Some(EzGfxResult::InvalidContext),
        );
        assert_eq!(status, EzGfxResult::InvalidContext);
    }
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
impl Drop for MeshFixture {
    fn drop(&mut self) {
        // SAFETY: clearing a live registration retains no user-data pointer.
        let _ = unsafe {
            ez_gfx_context_register_callback(self.native.context, None, core::ptr::null_mut())
        };
        if self.indices != 0 {
            ez_gfx_index_allocation_remove(self.native.context, self.indices);
        }
        ez_gfx_mesh_shader_destroy(self.native.context, self.mesh);
        ez_gfx_fragment_shader_destroy(self.native.context, self.target_fragment);
        ez_gfx_vertex_shader_destroy(self.native.context, self.vertex);
        if self.target != 0 {
            ez_gfx_render_target_destroy(self.native.context, self.target);
        }
    }
}
#[cfg(any(not(test), not(target_vendor = "apple")))]
fn exercise_task_mesh(backend: u8) {
    #[cfg(not(target_vendor = "apple"))]
    if backend == 3 {
        assert_metal_unavailable();
        return;
    }
    let native = create_context(backend);
    let capabilities = query_capabilities(native.context);
    if capabilities.mesh == 0 || capabilities.task == 0 {
        // Without both stages the task entry point is rejected as
        // unsupported; a mesh-only device cannot run this contract.
        let mut shader = 0;
        assert_eq!(
            // SAFETY: artifact, entry, and output storage are live through the call.
            unsafe {
                ez_gfx_task_shader_load(
                    native.context,
                    task_artifact().as_ptr(),
                    task_artifact().len(),
                    b"taskmain".as_ptr(),
                    b"taskmain".len(),
                    &raw mut shader,
                )
            },
            EzGfxResult::Unsupported,
            "incapable device must reject task stages",
        );
        panic!(
            "backend {backend} lacks task/mesh-shader support on this device; the named task/mesh pixel path cannot silently pass"
        );
    }
    let mut fixture = TaskFixture::new(native);
    let target = create_black_target(fixture.native.context);
    fixture.target = target;
    // The compute write feeds the task input buffer on the GPU: no CPU wait
    // separates the dispatch from the draw, so correct pixels prove the
    // dependency. Seed zero selects the small red triangle.
    let (status, pixels) = fixture.render_task(b"compute_red");
    assert_eq!(status, EzGfxResult::Ok);
    assert_task_frame(&pixels.expect("task readback"), [255, 0, 0, 255], None);
    // The backend is proven by the render above, so abort retirement runs
    // after real work rather than as a cold precheck.
    assert_abort_retires_once(fixture.native.context, fixture.target);
    // Seed one selects the large green triangle: both triangle and color
    // change through the real payload.
    let (status, pixels) = fixture.render_task(b"compute_green");
    assert_eq!(status, EzGfxResult::Ok);
    assert_task_frame(
        &pixels.expect("task readback"),
        [0, 255, 0, 255],
        Some([0, 255, 0, 255]),
    );
    // Changing only the task shader must change pixels rather than reuse a
    // stale pipeline: same buffer and mesh, fixed blue output.
    let (task_fixed, mesh_fixed) = (fixture.task_fixed, fixture.mesh_fixed);
    let (status, pixels) =
        fixture.render_task_stages(b"compute_red", task_fixed, fixture.mesh, [1, 1, 1]);
    assert_eq!(status, EzGfxResult::Ok);
    assert_task_frame(&pixels.expect("task readback"), [0, 0, 255, 255], None);
    // Changing only the mesh shader must change pixels rather than reuse a
    // stale pipeline: same buffer and task, fixed yellow output.
    let (status, pixels) =
        fixture.render_task_stages(b"compute_red", fixture.task, mesh_fixed, [1, 1, 1]);
    assert_eq!(status, EzGfxResult::Ok);
    assert_task_frame(&pixels.expect("task readback"), [255, 255, 0, 255], None);
    // Destroying every shader owner after recording must still submit the
    // retained frame correctly.
    let (status, pixels) = fixture.render_task_after_owner_destroy();
    assert_eq!(status, EzGfxResult::Ok);
    assert_task_frame(&pixels.expect("retained readback"), [255, 0, 0, 255], None);
    // A zero task grid fails before payload setup or backend delegation.
    let (status, pixels) = fixture.render_task_groups([0, 1, 1]);
    assert_eq!(status, EzGfxResult::InvalidArgument);
    assert!(pixels.is_none(), "failed dispatch must not present");
    assert_capability_output_unchanged_on_failure(backend);
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
fn assert_task_frame(bytes: &[u8], center: [u8; 4], probe: Option<[u8; 4]>) {
    assert_eq!(pixel(bytes, 32, 32), center, "task center");
    // Pixel (40, 32) sits inside the large triangle but outside the small
    // one regardless of the backend Y convention; the black background
    // makes outside pixels exactly black.
    match probe {
        Some(expected) => assert_eq!(pixel(bytes, 40, 32), expected, "task triangle extent"),
        None => assert_eq!(
            pixel(bytes, 40, 32),
            [0, 0, 0, 255],
            "small triangle must not reach probe"
        ),
    }
    for (x, y) in [(2, 2), (61, 2), (2, 61), (61, 61)] {
        assert_eq!(
            pixel(bytes, x, y),
            [0, 0, 0, 255],
            "black corner ({x}, {y})"
        );
    }
}

include!("shader_stages/task_fixture.rs");
