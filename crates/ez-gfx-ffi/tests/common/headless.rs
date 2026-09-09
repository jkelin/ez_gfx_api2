use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxContext, EzGfxHeadlessSurfaceDesc, EzGfxResult, EzGfxSurface,
    ez_gfx_context_create_backend, ez_gfx_context_destroy, ez_gfx_context_init_device,
    ez_gfx_surface_create_headless, ez_gfx_surface_destroy,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

pub struct TestContext {
    pub context: EzGfxContext,
    pub surface: EzGfxSurface,
}

impl TestContext {
    // Validation is opt-in so existing fixture callers retain their original device requirements.
    pub fn create_with_validation(backend: u8, validation: bool) -> Self {
        // No native window exists, is shown, or is activated.
        let mut native = Self {
            context: 0,
            surface: 0,
        };
        let desc = EzGfxBackendContextDesc {
            enable_debug: u8::from(validation),
            enable_validation: u8::from(validation),
            backend,
            texture_decode_workers: 0,
            adapter_count: 0,
            adapter: core::ptr::null(),
        };
        assert_eq!(
            {
                // SAFETY: descriptor and output storage are live and correctly aligned through the FFI call.
                unsafe { ez_gfx_context_create_backend(&raw const desc, &raw mut native.context) }
            },
            EzGfxResult::Ok
        );

        let surface_desc = EzGfxHeadlessSurfaceDesc {
            width: WIDTH,
            height: HEIGHT,
            cache_presented_snapshots: 0,
        };
        assert_eq!(
            {
                // SAFETY: descriptor and output storage live through the call; no native handles are borrowed.
                unsafe {
                    ez_gfx_surface_create_headless(
                        native.context,
                        &raw const surface_desc,
                        &raw mut native.surface,
                    )
                }
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
        let surface = if self.surface == 0 {
            EzGfxResult::Ok
        } else {
            ez_gfx_surface_destroy(self.context, self.surface)
        };
        let context = if self.context == 0 {
            EzGfxResult::Ok
        } else {
            ez_gfx_context_destroy(self.context)
        };
        assert!(
            (surface == EzGfxResult::Ok && context == EzGfxResult::Ok) || std::thread::panicking(),
            "headless fixture teardown failed: surface={surface:?} context={context:?}"
        );
    }
}
