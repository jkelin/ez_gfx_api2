//! Native Vulkan and DX12 compute-pipeline smoke tests through the C ABI.
#![cfg(windows)]

use std::{ffi::CString, sync::Once};

use ez_gfx_compiler::{CompilerError, Target, compile_shader};
use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxBinding, EzGfxResult, EzGfxSurfaceDesc,
    ez_gfx_context_create_backend, ez_gfx_context_destroy, ez_gfx_context_init_device,
    ez_gfx_context_wait_idle, ez_gfx_frame_begin, ez_gfx_frame_submit,
    ez_gfx_render_add_compute_pipeline, ez_gfx_shader_destroy, ez_gfx_shader_load_artifact,
    ez_gfx_structured_acquire, ez_gfx_structured_release, ez_gfx_structured_write,
    ez_gfx_surface_create, ez_gfx_surface_destroy,
};
use windows::{
    Win32::{
        Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM},
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DefWindowProcW, DestroyWindow, IsWindowVisible, RegisterClassW,
            WINDOW_EX_STYLE, WNDCLASSW, WS_OVERLAPPED,
        },
    },
    core::w,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
static REGISTER_WINDOW_CLASS: Once = Once::new();

struct TestWindow {
    handle: HWND,
    instance: HINSTANCE,
}

impl TestWindow {
    fn create_hidden() -> Self {
        // SAFETY: the null module name requests the current test executable's loaded module.
        let module = unsafe { GetModuleHandleW(None) }.expect("get test module");
        let instance = HINSTANCE(module.0);
        REGISTER_WINDOW_CLASS.call_once(|| {
            let class = WNDCLASSW {
                lpfnWndProc: Some(test_window_proc),
                hInstance: instance,
                lpszClassName: w!("EzGfxPsoSmokeWindow"),
                ..Default::default()
            };
            // SAFETY: `class` points only to static class-name and function storage; registration copies the descriptor before returning.
            assert_ne!(unsafe { RegisterClassW(&raw const class) }, 0);
        });
        // The absence of WS_VISIBLE makes the native test surface hidden from creation.
        // SAFETY: the registered class, current module, and static strings remain live; this call has no parent, menu, or creation payload.
        let handle = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("EzGfxPsoSmokeWindow"),
                w!("ez-gfx pso smoke"),
                WS_OVERLAPPED,
                0,
                0,
                i32::try_from(WIDTH).expect("test width fits Win32"),
                i32::try_from(HEIGHT).expect("test height fits Win32"),
                None,
                None,
                Some(instance),
                None,
            )
        }
        .expect("create hidden test window");
        // SAFETY: `handle` is the successfully created live window owned by this fixture.
        assert!(!unsafe { IsWindowVisible(handle) }.as_bool());
        Self { handle, instance }
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        // SAFETY: `handle` is the live window uniquely owned by this fixture and is destroyed once.
        unsafe { DestroyWindow(self.handle) }.expect("destroy test window");
    }
}

unsafe extern "system" fn test_window_proc(
    window: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: unhandled messages are forwarded with the exact arguments supplied by the system.
    unsafe { DefWindowProcW(window, message, wparam, lparam) }
}
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
    let window = TestWindow::create_hidden();
    let mut context = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );

    let surface_desc = EzGfxSurfaceDesc {
        window: window.handle.0,
        display: window.instance.0,
        platform: 0,
        width: WIDTH,
        height: HEIGHT,
        cache_presented_snapshots: 0,
    };
    let mut surface = 0;
    assert_eq!(
        {
            // SAFETY: the descriptor and output storage live through the call, and `window` retains both borrowed Win32 handles until surface destruction.
            unsafe { ez_gfx_surface_create(&raw const surface_desc, &raw mut surface, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_context_init_device(surface, context),
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
    ez_gfx_surface_destroy(surface, context);
    ez_gfx_context_destroy(context);
    let _ = std::fs::remove_dir_all(root);
}
