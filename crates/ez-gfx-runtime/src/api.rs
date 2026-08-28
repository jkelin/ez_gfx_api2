use ez_gfx_core::Backend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfacePlatform {
    Win32,
    Glfw,
    MetalLayer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextOptions {
    pub enable_debug: bool,
    pub enable_validation: bool,
    pub surface_platform: SurfacePlatform,
    pub backend: Backend,
}

impl ContextOptions {
    /// ABI booleans and backend/platform discriminants are validated before native setup.
    pub fn new(
        enable_debug: u8,
        enable_validation: u8,
        surface_platform: u8,
    ) -> Result<Self, PublicApiError> {
        Self::new_for_backend(
            enable_debug,
            enable_validation,
            surface_platform,
            Backend::Vulkan,
        )
    }

    /// Backend/platform pairs are closed: DX12 requires Win32, Metal requires a layer, and Vulkan
    /// accepts host-window platforms but never a Metal layer.
    pub fn new_for_backend(
        enable_debug: u8,
        enable_validation: u8,
        surface_platform: u8,
        backend: Backend,
    ) -> Result<Self, PublicApiError> {
        let surface_platform = parse_platform(surface_platform)?;
        if !matches!(
            (backend, surface_platform),
            (
                Backend::Vulkan,
                SurfacePlatform::Win32 | SurfacePlatform::Glfw
            ) | (Backend::Dx12, SurfacePlatform::Win32)
                | (Backend::Metal, SurfacePlatform::MetalLayer)
        ) {
            return Err(PublicApiError::InvalidPlatform);
        }
        Ok(Self {
            enable_debug: parse_bool(enable_debug)?,
            enable_validation: parse_bool(enable_validation)?,
            surface_platform,
            backend,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceOptions {
    pub window: usize,
    pub display: usize,
    pub platform: SurfacePlatform,
    pub width: u32,
    pub height: u32,
    pub cache_presented_snapshots: bool,
}

impl SurfaceOptions {
    /// GLFW may omit display; Win32 requires both borrowed handles; Metal accepts a borrowed CAMetalLayer.
    pub fn new(
        window: usize,
        display: usize,
        platform: SurfacePlatform,
        width: u32,
        height: u32,
        cache: u8,
    ) -> Result<Self, PublicApiError> {
        if window == 0 || (platform == SurfacePlatform::Win32 && display == 0) {
            return Err(PublicApiError::MissingNativeHandle);
        }
        validate_extent(width, height)?;
        if width == 0 {
            return Err(PublicApiError::ZeroInitialExtent);
        }
        Ok(Self {
            window,
            display,
            platform,
            width,
            height,
            cache_presented_snapshots: parse_bool(cache)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceState {
    width: u32,
    height: u32,
    resize_pending: bool,
    snapshot_cache: bool,
}

impl SurfaceState {
    /// Construction requires a drawable extent; minimized state is entered only through resize.
    pub fn new(width: u32, height: u32, snapshot_cache: bool) -> Result<Self, PublicApiError> {
        if width == 0 || height == 0 {
            return Err(PublicApiError::ZeroInitialExtent);
        }
        Ok(Self {
            width,
            height,
            resize_pending: false,
            snapshot_cache,
        })
    }

    /// A 0x0 extent is a valid minimized transition reported as NotReady; mixed-zero is invalid.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), PublicApiError> {
        validate_extent(width, height)?;
        self.width = width;
        self.height = height;
        self.resize_pending = true;
        if width == 0 {
            Err(PublicApiError::NotReady)
        } else {
            Ok(())
        }
    }

    pub const fn extent(&self) -> Option<(u32, u32)> {
        if self.width == 0 {
            None
        } else {
            Some((self.width, self.height))
        }
    }
    pub const fn resize_pending(&self) -> bool {
        self.resize_pending
    }
    pub fn take_resize_pending(&mut self) -> bool {
        core::mem::replace(&mut self.resize_pending, false)
    }
    pub const fn snapshot_cache(&self) -> bool {
        self.snapshot_cache
    }
    pub fn set_snapshot_cache(&mut self, enabled: bool) {
        self.snapshot_cache = enabled;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublicApiError {
    InvalidBoolean,
    InvalidPlatform,
    MissingNativeHandle,
    MixedZeroExtent,
    ZeroInitialExtent,
    NotReady,
}

fn parse_bool(value: u8) -> Result<bool, PublicApiError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(PublicApiError::InvalidBoolean),
    }
}

fn parse_platform(value: u8) -> Result<SurfacePlatform, PublicApiError> {
    match value {
        0 => Ok(SurfacePlatform::Win32),
        1 => Ok(SurfacePlatform::Glfw),
        2 => Ok(SurfacePlatform::MetalLayer),
        _ => Err(PublicApiError::InvalidPlatform),
    }
}

fn validate_extent(width: u32, height: u32) -> Result<(), PublicApiError> {
    if (width == 0) != (height == 0) {
        Err(PublicApiError::MixedZeroExtent)
    } else {
        Ok(())
    }
}
