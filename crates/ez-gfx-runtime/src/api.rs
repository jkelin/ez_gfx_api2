use ez_gfx_core::Backend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Explicit adapter request carried by [`ContextOptions`].
///
/// `None` (the default) keeps legacy first-fit device selection. `Some`
/// bypasses ranking but never bypasses admission: the stable identity must
/// still exist in the enumeration and satisfy the semantic profile.
pub struct AdapterSelection {
    /// Stable 128-bit adapter identity from enumeration.
    pub stable_id: [u8; 16],
    /// Whether a software-class adapter is acceptable.
    pub allow_software: bool,
}

impl AdapterSelection {
    /// Builds an explicit adapter request. An all-zero identity never matches.
    #[must_use]
    pub const fn new(stable_id: [u8; 16], allow_software: bool) -> Self {
        Self {
            stable_id,
            allow_software,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Settings used to initialize a graphics context.
pub struct ContextOptions {
    /// Enables graphics API debug facilities.
    pub enable_debug: bool,
    /// Enables backend validation checks.
    pub enable_validation: bool,
    /// Selects the graphics backend.
    pub backend: Backend,
    /// Decode worker threads for async texture uploads. Zero selects the default
    /// topology (`available_parallelism - 1`, at least one thread).
    pub texture_decode_workers: u32,
    /// Explicit adapter request. `None` keeps legacy first-fit selection.
    pub adapter_selection: Option<AdapterSelection>,
}

impl ContextOptions {
    /// Validates ABI booleans for a Vulkan context.
    ///
    /// # Errors
    ///
    /// Returns [`PublicApiError::InvalidBoolean`] when either byte is not zero or one.
    pub fn new(enable_debug: u8, enable_validation: u8) -> Result<Self, PublicApiError> {
        Self::new_for_backend(enable_debug, enable_validation, Backend::Vulkan)
    }

    /// Validates ABI booleans for a context using `backend`.
    ///
    /// # Errors
    ///
    /// Returns [`PublicApiError::InvalidBoolean`] when either byte is not zero or one.
    pub fn new_for_backend(
        enable_debug: u8,
        enable_validation: u8,
        backend: Backend,
    ) -> Result<Self, PublicApiError> {
        Ok(Self {
            enable_debug: parse_bool(enable_debug)?,
            enable_validation: parse_bool(enable_validation)?,
            backend,
            texture_decode_workers: 0,
            adapter_selection: None,
        })
    }
    /// Requests an explicit adapter by stable identity. The request bypasses
    /// default ranking but never bypasses admission: unknown identities fail
    /// `InvalidArgument`, disallowed software fails `InvalidArgument`, and
    /// inadmissible adapters fail `Unsupported`. `None` (the default) keeps
    /// legacy first-fit selection.
    #[must_use]
    pub const fn with_adapter(mut self, stable_id: [u8; 16], allow_software: bool) -> Self {
        self.adapter_selection = Some(AdapterSelection::new(stable_id, allow_software));
        self
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
/// Presentation settings for a headless surface.
pub struct HeadlessSurfaceOptions {
    /// Initial drawable width in pixels.
    pub width: u32,
    /// Initial drawable height in pixels.
    pub height: u32,
    /// Enables caching of rendered snapshots.
    pub cache_presented_snapshots: bool,
}

impl HeadlessSurfaceOptions {
    /// Validates a headless surface descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`PublicApiError::MixedZeroExtent`] or
    /// [`PublicApiError::ZeroInitialExtent`] for an invalid extent, or
    /// [`PublicApiError::InvalidBoolean`] when `cache` is not zero or one.
    pub fn new(width: u32, height: u32, cache: u8) -> Result<Self, PublicApiError> {
        validate_extent(width, height)?;
        if width == 0 {
            return Err(PublicApiError::ZeroInitialExtent);
        }
        Ok(Self {
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
    /// Creates state for a window whose drawable extent has not been queried yet.
    #[must_use]
    pub const fn new_window(snapshot_cache: bool) -> Self {
        Self {
            width: 0,
            height: 0,
            resize_pending: false,
            snapshot_cache,
        }
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
    /// A native window handle was unavailable or unsupported.
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
