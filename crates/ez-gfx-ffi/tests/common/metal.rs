use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxResult, EzGfxWindowSurfaceDesc, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_init_device, ez_gfx_surface_create_window,
    ez_gfx_surface_destroy,
};
use objc2::rc::Retained;
use objc2_core_foundation::CGSize;
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::CAMetalLayer;

pub struct TestContext {
    pub context: u64,
    pub surface: u64,
    // Never attached to an NSView or NSWindow; retained through native surface destruction.
    layer: Retained<CAMetalLayer>,
}

impl TestContext {
    pub fn create(backend: u8) -> Self {
        // Fail closed on a mismatched fixture instead of creating another backend's surface.
        assert_eq!(backend, 3, "windowless fixture requires Metal");
        let layer = CAMetalLayer::new();
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
        layer.setDrawableSize(CGSize {
            width: 64.0,
            height: 64.0,
        });
        let mut native = Self {
            context: 0,
            surface: 0,
            layer,
        };
        let desc = EzGfxBackendContextDesc {
            enable_debug: 0,
            enable_validation: 0,
            backend,
            texture_decode_workers: 0,
            adapter_count: 0,
            adapter: core::ptr::null(),
        };
        assert_eq!(
            // SAFETY: The descriptor and output storage remain live and aligned through the call.
            unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut native.context) },
            EzGfxResult::Ok
        );
        let surface_desc = EzGfxWindowSurfaceDesc {
            window: Retained::as_ptr(&native.layer).cast_mut().cast(),
            display: core::ptr::null_mut(),
            cache_presented_snapshots: 0,
        };
        assert_eq!(
            // SAFETY: The retained unattached layer outlives the surface; all storage is live.
            unsafe {
                ez_gfx_surface_create_window(
                    native.context,
                    &raw const surface_desc,
                    &raw mut native.surface,
                )
            },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_context_init_device(native.context, native.surface),
            EzGfxResult::Ok
        );
        native
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        // Partial construction also releases the context, without destroying absent handles.
        if self.surface != 0 {
            ez_gfx_surface_destroy(self.context, self.surface);
        }
        if self.context != 0 {
            ez_gfx_context_destroy(self.context);
        }
    }
}
