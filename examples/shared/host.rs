use anyhow::Context as _;
use ez_gfx::{Backend, SurfacePlatform};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePlatform {
    Win32,
    MetalLayer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendConfig {
    pub backend: Backend,
    pub name: &'static str,
    pub platform: SurfacePlatform,
}

/// Resolves the host-compatible backend and native surface platform.
///
/// An omitted backend defaults to Vulkan off Apple and Metal on Apple.
/// Explicit backends unsupported by this host fail rather than silently falling
/// back; this also keeps the environment boundary deterministic for examples.
pub fn backend_config(native: NativePlatform) -> anyhow::Result<BackendConfig> {
    let backend = parse_backend(std::env::var("EZ_GFX_BACKEND").ok().as_deref())?;
    Ok(BackendConfig {
        backend,
        name: backend_name_for(backend),
        platform: surface_platform(native),
    })
}

const fn surface_platform(native: NativePlatform) -> SurfacePlatform {
    match native {
        NativePlatform::Win32 => SurfacePlatform::Win32,
        NativePlatform::MetalLayer => SurfacePlatform::MetalLayer,
    }
}

fn parse_backend(value: Option<&str>) -> anyhow::Result<Backend> {
    match value {
        #[cfg(target_vendor = "apple")]
        None | Some("metal") => Ok(Backend::Metal),
        #[cfg(not(target_vendor = "apple"))]
        None | Some("vulkan") => Ok(Backend::Vulkan),
        #[cfg(windows)]
        Some("dx12") => Ok(Backend::Dx12),
        Some(value) => Err(anyhow::anyhow!("unsupported EZ_GFX_BACKEND `{value}`")),
    }
}

const fn backend_name_for(backend: Backend) -> &'static str {
    match backend {
        Backend::Vulkan => "Vulkan",
        Backend::Dx12 => "DX12",
        Backend::Metal => "Metal",
    }
}

/// Maps a backend to its clip-space convention.
pub const fn clip_y(backend: Backend) -> crate::shared::math::ClipY {
    match backend {
        Backend::Vulkan => crate::shared::math::ClipY::Vulkan,
        Backend::Dx12 => crate::shared::math::ClipY::Dx12,
        Backend::Metal => crate::shared::math::ClipY::Metal,
    }
}

/// Returns the selected backend's stable display name for process reports.
pub fn backend_name() -> anyhow::Result<&'static str> {
    let native = if cfg!(target_vendor = "apple") {
        NativePlatform::MetalLayer
    } else {
        NativePlatform::Win32
    };
    Ok(backend_config(native)?.name)
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
    pub fn attach(window: &Window, width: u32, height: u32) -> anyhow::Result<Self> {
        #[cfg(not(target_vendor = "apple"))]
        let _ = (width, height);
        let raw = window
            .window_handle()
            .context("get native window handle")?
            .as_raw();
        #[cfg(windows)]
        if let RawWindowHandle::Win32(handle) = raw {
            let display = handle
                .hinstance
                .ok_or_else(|| anyhow::anyhow!("Win32 handle omitted its module instance"))?;
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
        Err(anyhow::anyhow!(
            "unsupported native window handle for example host"
        ))
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_defaults_to_host_backend() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(parse_backend(None).unwrap(), Backend::Metal);
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(parse_backend(None).unwrap(), Backend::Vulkan);
    }

    #[test]
    fn explicit_supported_backends_report_stable_names() {
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(parse_backend(Some("vulkan")).unwrap(), Backend::Vulkan);
        #[cfg(windows)]
        assert_eq!(parse_backend(Some("dx12")).unwrap(), Backend::Dx12);
        #[cfg(target_vendor = "apple")]
        assert_eq!(parse_backend(Some("metal")).unwrap(), Backend::Metal);
    }

    #[test]
    fn unsupported_backend_is_rejected() {
        assert!(parse_backend(Some("unsupported")).is_err());
        #[cfg(not(target_vendor = "apple"))]
        assert!(parse_backend(Some("metal")).is_err());
        #[cfg(not(windows))]
        assert!(parse_backend(Some("dx12")).is_err());
    }

    #[test]
    fn native_platform_maps_to_surface_platform() {
        assert_eq!(
            surface_platform(NativePlatform::Win32),
            SurfacePlatform::Win32
        );
        assert_eq!(
            surface_platform(NativePlatform::MetalLayer),
            SurfacePlatform::MetalLayer
        );
    }
}
