use ez_gfx_ffi::EzGfxSurfaceDesc;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

pub struct HostSurface {
    pub desc: EzGfxSurfaceDesc,
    #[cfg(target_vendor = "apple")]
    metal_layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
}

impl HostSurface {
    pub fn attach(window: &Window, width: u32, height: u32) -> Result<Self, String> {
        let raw = window
            .window_handle()
            .map_err(|error| format!("get native window handle: {error}"))?
            .as_raw();

        #[cfg(windows)]
        if let RawWindowHandle::Win32(handle) = raw {
            let display = handle
                .hinstance
                .ok_or_else(|| "Win32 handle omitted its module instance".to_owned())?;
            return Ok(Self {
                desc: EzGfxSurfaceDesc {
                    window: handle.hwnd.get() as *mut _,
                    display: display.get() as *mut _,
                    platform: 0,
                    width,
                    height,
                    cache_presented_snapshots: 1,
                },
            });
        }
        #[cfg(target_vendor = "apple")]
        if let RawWindowHandle::AppKit(handle) = raw {
            use objc2::{msg_send, rc::Retained};
            use objc2_quartz_core::CAMetalLayer;

            let layer = CAMetalLayer::new();
            layer.setDrawableSize(objc2_core_foundation::CGSize {
                width: f64::from(width),
                height: f64::from(height),
            });
            let view = unsafe { &*(handle.ns_view.as_ptr() as *const objc2::runtime::AnyObject) };
            unsafe {
                let _: () = msg_send![view, setWantsLayer: true];
                let _: () = msg_send![view, setLayer: Retained::as_ptr(&layer)];
            }
            return Ok(Self {
                desc: EzGfxSurfaceDesc {
                    window: Retained::as_ptr(&layer) as *mut _,
                    display: core::ptr::null_mut(),
                    platform: 2,
                    width,
                    height,
                    cache_presented_snapshots: 1,
                },
                metal_layer: layer,
            });
        }

        Err("unsupported native window handle for example host".to_owned())
    }

    pub fn resize(&self, width: u32, height: u32) {
        #[cfg(not(target_vendor = "apple"))]
        let _ = (width, height);
        #[cfg(target_vendor = "apple")]
        self.metal_layer
            .setDrawableSize(objc2_core_foundation::CGSize {
                width: f64::from(width),
                height: f64::from(height),
            });
    }
}
