use ez_gfx_core::Backend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Native window integration modes accepted by the runtime.
pub enum SurfacePlatform {
    /// Uses Win32 window and instance handles.
    Win32,
    /// Uses a GLFW-created native window.
    Glfw,
    /// Uses a Core Animation Metal layer.
    MetalLayer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Settings used to initialize a graphics context.
pub struct ContextOptions {
    /// Enables graphics API debug facilities.
    pub enable_debug: bool,
    /// Enables backend validation checks.
    pub enable_validation: bool,
    /// Selects the native surface integration mode.
    pub surface_platform: SurfacePlatform,
    /// Selects the graphics backend.
    pub backend: Backend,
    /// Decode worker threads for async texture uploads. Zero selects the default
    /// topology (`available_parallelism - 1`, at least one thread).
    pub texture_decode_workers: u32,
}

impl ContextOptions {
    /// ABI booleans and backend/platform discriminants are validated before native setup.
    ///
    /// # Errors
    ///
    /// Returns `InvalidBoolean` when either boolean byte is not 0 or 1, or `InvalidPlatform` when the platform code is invalid or incompatible with Vulkan.
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
    ///
    /// # Errors
    ///
    /// Returns `InvalidBoolean` when either boolean byte is not 0 or 1, or `InvalidPlatform` when the platform code is invalid or incompatible with the selected backend.
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
            texture_decode_workers: 0,
        })
    }

    /// Overrides the async texture decode worker count. Zero (the default)
    /// keeps the default topology; a nonzero value requests exactly that many
    /// worker threads. The count is validated when the context is created.
    #[must_use]
    pub const fn with_texture_decode_workers(mut self, workers: u32) -> Self {
        self.texture_decode_workers = workers;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Native handles, dimensions, and presentation settings for a surface.
pub struct SurfaceOptions {
    /// Borrowed native window or Metal layer handle.
    pub window: usize,
    /// Borrowed native display or Win32 instance handle.
    pub display: usize,
    /// Identifies how the native handles are interpreted.
    pub platform: SurfacePlatform,
    /// Initial drawable width in pixels.
    pub width: u32,
    /// Initial drawable height in pixels.
    pub height: u32,
    /// Enables caching of presented snapshots.
    pub cache_presented_snapshots: bool,
}

impl SurfaceOptions {
    /// GLFW may omit display; Win32 requires both borrowed handles; Metal accepts a borrowed `CAMetalLayer`.
    ///
    /// # Errors
    ///
    /// Returns `MissingNativeHandle` for a null required handle, `MixedZeroExtent` or `ZeroInitialExtent` for an invalid extent, or `InvalidBoolean` when `cache` is not 0 or 1.
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
/// Tracks a surface's drawable extent and pending presentation changes.
pub struct SurfaceState {
    /// Current drawable width in pixels, or zero while minimized.
    width: u32,
    /// Current drawable height in pixels, or zero while minimized.
    height: u32,
    /// Indicates whether an unconsumed resize has occurred.
    resize_pending: bool,
    /// Indicates whether presented snapshots are cached.
    snapshot_cache: bool,
}

impl SurfaceState {
    /// Construction requires a drawable extent; minimized state is entered only through resize.
    ///
    /// # Errors
    ///
    /// Returns `ZeroInitialExtent` when either dimension is zero.
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

    /// A 0x0 extent is a valid minimized transition reported as `NotReady`; mixed-zero is invalid.
    ///
    /// # Errors
    ///
    /// Returns `MixedZeroExtent` when exactly one dimension is zero, or `NotReady` when both dimensions are zero.
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

    /// Returns the drawable extent, or `None` while minimized.
    pub const fn extent(&self) -> Option<(u32, u32)> {
        if self.width == 0 {
            None
        } else {
            Some((self.width, self.height))
        }
    }
    /// Reports whether a resize is awaiting consumption.
    pub const fn resize_pending(&self) -> bool {
        self.resize_pending
    }
    /// Consumes and returns the pending-resize flag.
    pub fn take_resize_pending(&mut self) -> bool {
        core::mem::replace(&mut self.resize_pending, false)
    }
    /// Reports whether presented snapshots are cached.
    pub const fn snapshot_cache(&self) -> bool {
        self.snapshot_cache
    }
    /// Enables or disables caching of presented snapshots.
    pub fn set_snapshot_cache(&mut self, enabled: bool) {
        self.snapshot_cache = enabled;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Failures exposed by runtime option and surface-state validation.
pub enum PublicApiError {
    /// A boolean byte was neither zero nor one.
    InvalidBoolean,
    /// A platform code or backend/platform pairing is unsupported.
    InvalidPlatform,
    /// A required native window, display, or layer handle was null.
    MissingNativeHandle,
    /// Exactly one extent dimension was zero.
    MixedZeroExtent,
    /// Surface creation requested a zero-sized drawable.
    ZeroInitialExtent,
    /// The surface is temporarily unavailable, such as while minimized.
    NotReady,
}

/// Decodes an ABI boolean encoded as zero or one.
///
/// # Errors
///
/// Returns `InvalidBoolean` when `value` is neither 0 nor 1.
fn parse_bool(value: u8) -> Result<bool, PublicApiError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(PublicApiError::InvalidBoolean),
    }
}

/// Decodes an ABI platform discriminant.
///
/// # Errors
///
/// Returns `InvalidPlatform` when `value` is not a recognized platform discriminant.
fn parse_platform(value: u8) -> Result<SurfacePlatform, PublicApiError> {
    match value {
        0 => Ok(SurfacePlatform::Win32),
        1 => Ok(SurfacePlatform::Glfw),
        2 => Ok(SurfacePlatform::MetalLayer),
        _ => Err(PublicApiError::InvalidPlatform),
    }
}

/// Accepts extents whose dimensions are either both zero or both nonzero.
///
/// # Errors
///
/// Returns `MixedZeroExtent` when exactly one dimension is zero.
fn validate_extent(width: u32, height: u32) -> Result<(), PublicApiError> {
    if (width == 0) == (height == 0) {
        Ok(())
    } else {
        Err(PublicApiError::MixedZeroExtent)
    }
}
