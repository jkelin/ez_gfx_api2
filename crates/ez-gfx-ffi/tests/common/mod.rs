use std::sync::Once;

use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxContext, EzGfxResult, EzGfxSurface, EzGfxWindowSurfaceDesc,
    ez_gfx_context_create_backend, ez_gfx_context_destroy, ez_gfx_context_init_device,
    ez_gfx_surface_create_window, ez_gfx_surface_destroy,
};
use windows::{
    Win32::{
        Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM},
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DefWindowProcW, DestroyWindow, GWL_STYLE, GetClientRect,
            GetForegroundWindow, GetWindowLongW, IsWindowVisible, RegisterClassW, WINDOW_EX_STYLE,
            WNDCLASSW, WS_POPUP, WS_VISIBLE,
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
                lpszClassName: w!("EzGfxNativeTestWindow"),
                ..Default::default()
            };
            // SAFETY: `class` points only to static class-name and function storage; registration copies the descriptor before returning.
            assert_ne!(unsafe { RegisterClassW(&raw const class) }, 0);
        });

        // A borderless popup preserves the requested client extent; WS_OVERLAPPED adds non-client
        // decorations even when hidden. Neither visibility nor activation is requested.
        // SAFETY: the registered class, current module, and static strings remain live; this call has no parent, menu, or creation payload.
        let handle = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("EzGfxNativeTestWindow"),
                w!("ez-gfx native test"),
                WS_POPUP,
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
        let mut client = RECT::default();
        // SAFETY: the live fixture window and writable RECT belong to this thread.
        unsafe {
            assert!(!IsWindowVisible(handle).as_bool());
            assert_eq!(
                GetWindowLongW(handle, GWL_STYLE).cast_unsigned() & WS_VISIBLE.0,
                0
            );
            assert_ne!(GetForegroundWindow(), handle);
            GetClientRect(handle, &raw mut client).expect("read hidden client extent");
        }
        assert_eq!(client.right - client.left, i32::try_from(WIDTH).unwrap());
        assert_eq!(client.bottom - client.top, i32::try_from(HEIGHT).unwrap());
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

pub struct TestContext {
    pub context: EzGfxContext,
    pub surface: EzGfxSurface,
    // Ownership keeps HWND/HINSTANCE alive through TestContext::drop.
    _window: TestWindow,
}

impl TestContext {
    // Validation is opt-in so existing fixture callers retain their original device requirements.
    pub fn create_with_validation(backend: u8, validation: bool) -> Self {
        let window = TestWindow::create_hidden();
        let desc = EzGfxBackendContextDesc {
            enable_debug: u8::from(validation),
            enable_validation: u8::from(validation),
            backend,
            texture_decode_workers: 0,
            adapter_count: 0,
            adapter: core::ptr::null(),
        };
        let mut context = 0;
        assert_eq!(
            {
                // SAFETY: descriptor and output storage are live and correctly aligned through the FFI call.
                unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut context) }
            },
            EzGfxResult::Ok
        );

        let surface_desc = EzGfxWindowSurfaceDesc {
            window: window.handle.0,
            display: window.instance.0,
            cache_presented_snapshots: 0,
        };
        let mut surface = 0;
        assert_eq!(
            {
                // SAFETY: descriptor and output storage live through the call; `window` retains native handles until surface destruction.
                unsafe {
                    ez_gfx_surface_create_window(context, &raw const surface_desc, &raw mut surface)
                }
            },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_context_init_device(context, surface),
            EzGfxResult::Ok
        );

        Self {
            context,
            surface,
            // Ownership keeps HWND/HINSTANCE alive through TestContext::drop.
            _window: window,
        }
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        ez_gfx_surface_destroy(self.context, self.surface);
        ez_gfx_context_destroy(self.context);
    }
}
