use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePlatform {
    Win32,
    MetalLayer,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeSurface {
    pub window: usize,
    pub display: usize,
    pub platform: NativePlatform,
}

pub struct HostSurface {
    descriptor: NativeSurface,
    #[cfg(target_vendor = "apple")]
    metal_layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
}

impl HostSurface {
    pub fn attach(window: &Window, width: u32, height: u32) -> Result<Self, String> {
        #[cfg(not(target_vendor = "apple"))]
        let _ = (width, height);
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
                descriptor: NativeSurface {
                    window: handle.hwnd.get() as usize,
                    display: display.get() as usize,
                    platform: NativePlatform::Win32,
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
            // SAFETY: winit owns a live NSView for the duration of this host.
            let view = unsafe { &*(handle.ns_view.as_ptr() as *const objc2::runtime::AnyObject) };
            // SAFETY: both messages are valid for NSView and the retained layer outlives attachment.
            unsafe {
                let _: () = msg_send![view, setWantsLayer: true];
                let _: () = msg_send![view, setLayer: Retained::as_ptr(&layer)];
            }
            return Ok(Self {
                descriptor: NativeSurface {
                    window: Retained::as_ptr(&layer) as usize,
                    display: 0,
                    platform: NativePlatform::MetalLayer,
                },
                metal_layer: layer,
            });
        }
        Err("unsupported native window handle for example host".to_owned())
    }

    pub const fn descriptor(&self) -> NativeSurface {
        self.descriptor
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
