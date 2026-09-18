/// Formats used by a cached managed render target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderTargetDescriptor {
    /// Color attachment and readback format.
    pub color_format: ez_gfx_runtime::target::Format,
    /// Optional depth attachment format.
    pub depth_format: Option<ez_gfx_runtime::target::Format>,
    /// Color applied when the attachment begins with a clear load operation.
    pub clear_color: [f32; 4],
    /// Preferred MSAA count; allocation automatically degrades to the highest
    /// hardware-supported count not exceeding this value.
    pub maximum_samples: u8,
}

impl RenderTargetDescriptor {
    /// Creates a color-only target descriptor.
    #[must_use]
    pub const fn color(color_format: ez_gfx_runtime::target::Format) -> Self {
        Self {
            color_format,
            depth_format: None,
            clear_color: [0.1, 0.1, 0.1, 1.0],
            maximum_samples: 1,
        }
    }

    /// Creates a color target with a depth companion.
    #[must_use]
    pub const fn color_depth(color_format: ez_gfx_runtime::target::Format) -> Self {
        Self {
            color_format,
            depth_format: Some(ez_gfx_runtime::target::Format::Depth32Float),
            clear_color: [0.1, 0.1, 0.1, 1.0],
            maximum_samples: 1,
        }
    }
    /// Overrides the clear color used by the attachment pass.
    #[must_use]
    pub const fn with_clear_color(mut self, clear_color: [f32; 4]) -> Self {
        self.clear_color = clear_color;
        self
    }
    /// Requests up to `maximum_samples` samples per pixel.
    ///
    /// Values other than `1`, `2`, `4`, or `8` are rejected when the target
    /// is configured.
    #[must_use]
    pub const fn with_maximum_samples(mut self, maximum_samples: u8) -> Self {
        self.maximum_samples = maximum_samples;
        self
    }
}


impl From<ez_gfx_runtime::target::Format> for RenderTargetDescriptor {
    fn from(color_format: ez_gfx_runtime::target::Format) -> Self {
        Self::color(color_format)
    }
}

enum RenderTargetBacking {
    Managed(String),
    Surface {
        format: ez_gfx_runtime::target::Format,
        extent: (u32, u32),
    },
}

struct RenderTargetInner {
    context: Rc<ContextInner>,
    backing: RenderTargetBacking,
}
impl RenderTargetInner {
    fn managed_handle(&self) -> Result<RawRenderTargetHandle> {
        let RenderTargetBacking::Managed(name) = &self.backing else {
            return Err(Error::Unsupported);
        };
        self.context
            .render_targets
            .borrow()
            .get(name)
            .map(|target| target.handle)
            .ok_or(Error::InvalidContext)
    }
}

/// Persistent context-owned managed render target.
pub struct RenderTargetHandle {
    inner: Rc<RenderTargetInner>,
}

impl RenderTargetHandle {
    /// Returns the resolved target format.
    ///
    /// # Errors
    /// Returns [`Error`] when the target was released.
    pub fn format(&self) -> Result<ez_gfx_runtime::target::Format> {
        self.inner
            .context
            .render_targets
            .borrow()
            .get(self.name()?)
            .map(|target| target.format)
            .ok_or(Error::InvalidContext)
    }

    /// Returns the target extent.
    ///
    /// # Errors
    /// Returns [`Error`] when the target was released.
    pub fn extent(&self) -> Result<(u32, u32)> {
        self.inner
            .context
            .render_targets
            .borrow()
            .get(self.name()?)
            .map(|target| target.extent)
            .ok_or(Error::InvalidContext)
    }

    /// Returns the stable bindless sampled-image slot.
    ///
    /// # Errors
    /// Returns [`Error`] when the target was released.
    pub fn binding(&self) -> Result<u32> {
        state::render_target_binding(self.inner.context.handle, self.inner.managed_handle()?)
    }

    /// Transitions this target for bindless sampling after rendering or copying.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is foreign, released, or unused by `frame`.
    pub fn prepare_sampling(&self, frame: &mut Frame) -> Result<()> {
        frame.prepare_persistent_target_sampling(self)
    }

    fn name(&self) -> Result<&str> {
        let RenderTargetBacking::Managed(name) = &self.inner.backing else {
            return Err(Error::Unsupported);
        };
        Ok(name)
    }
}

/// Pixel rectangle used by render-target blits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderTargetRect {
    /// Horizontal origin in pixels.
    pub x: u32,
    /// Vertical origin in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl RenderTargetRect {
    /// Creates a rectangle.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Pixel combination used by a render-target blit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Blending {
    /// Replaces destination pixels exactly.
    Copy,
}

/// Existing color contents policy for an attached persistent target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderTargetLoad {
    /// Clears the target before the first graphics pass.
    Clear,
    /// Preserves the target before the first graphics pass.
    Preserve,
}

/// Owning logical render target.
pub struct RenderTarget {
    inner: Rc<RenderTargetInner>,
    surface_lease: Option<Rc<SurfaceInner>>,
}

impl RenderTarget {
    /// Returns the resolved target format.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn format(&self) -> Result<ez_gfx_runtime::target::Format> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(name) => self
                .inner
                .context
                .render_targets
                .borrow()
                .get(name)
                .map(|target| target.format)
                .ok_or(Error::InvalidContext),
            RenderTargetBacking::Surface { format, .. } => Ok(*format),
        }
    }
    /// Returns the optional managed depth format.
    ///
    /// Surface targets expose `Depth32Float`, matching the depth attachment
    /// created on demand by every backend.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn depth_format(&self) -> Result<Option<ez_gfx_runtime::target::Format>> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(name) => self
                .inner
                .context
                .render_targets
                .borrow()
                .get(name)
                .map(|target| target.depth_format)
                .ok_or(Error::InvalidContext),
            RenderTargetBacking::Surface { .. } => {
                Ok(Some(ez_gfx_runtime::target::Format::Depth32Float))
            }
        }
    }

    /// Returns the negotiated sample count.
    ///
    /// Managed targets may report fewer samples than requested when hardware
    /// support is lower. Surface targets are currently single-sampled.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn samples(&self) -> Result<u8> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(_) => state::render_target_samples(
                self.inner.context.handle,
                self.inner.managed_handle()?,
            ),
            RenderTargetBacking::Surface { .. } => Ok(1),
        }
    }


    /// Returns the target extent.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn extent(&self) -> Result<(u32, u32)> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(name) => self
                .inner
                .context
                .render_targets
                .borrow()
                .get(name)
                .map(|target| target.extent)
                .ok_or(Error::InvalidContext),
            RenderTargetBacking::Surface { extent, .. } => Ok(*extent),
        }
    }
    /// Returns the stable bindless sampled-image slot for a managed target.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the target is a surface or is stale.
    pub fn binding(&self) -> Result<u32> {
        state::render_target_binding(self.inner.context.handle, self.inner.managed_handle()?)
    }

    /// Transitions this managed target for bindless sampling after its render pass.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the target is foreign, stale, or not attached to `frame`.
    pub fn prepare_sampling(&self, frame: &mut Frame) -> Result<()> {
        frame.prepare_target_sampling(self)
    }

    /// Returns the target clear value.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale.
    pub fn clear(&self) -> Result<ez_gfx_runtime::target::ClearValue> {
        let context = Context {
            inner: Rc::clone(&self.inner.context),
            owner: false,
        };
        context.check_entry()?;
        match &self.inner.backing {
            RenderTargetBacking::Managed(_) => state::render_target_clear(
                self.inner.context.handle,
                self.inner.managed_handle()?,
            ),
            RenderTargetBacking::Surface { .. } => Ok(ez_gfx_runtime::target::ClearValue::Color([
                0.1, 0.1, 0.1, 1.0,
            ])),
        }
    }

    /// Creates and schedules a unique readback request on `frame`.
    ///
    /// Managed targets must use [`ez_gfx_runtime::target::Format::Rgba8Unorm`].
    ///
    /// # Errors
    /// Returns [`Error`] when the target is stale, foreign, already released, or not RGBA8-readable.
    pub fn prepare_readback(&self, frame: &mut Frame) -> Result<Readback> {
        frame.prepare_target_readback(self)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ReadbackState {
    Queued,
    Complete,
    Aborted,
}

struct ReadbackInner {
    _context: Rc<ContextInner>,
    _target: Rc<RenderTargetInner>,
    id: ReadbackId,
    width: u32,
    height: u32,
    state: Cell<ReadbackState>,
}

/// Owning identity and target lease for one submitted readback.
pub struct Readback {
    inner: Rc<ReadbackInner>,
}

impl Readback {
    /// Returns the opaque request identity delivered with its callback.
    pub fn id(&self) -> ReadbackId {
        self.inner.id
    }
}

impl Context {
    /// Probes one render-target format without allocating it.
    ///
    /// # Errors
    /// Returns [`Error`] when the format or sample count is unsupported.
    pub fn probe_render_target_format(
        &self,
        format: ez_gfx_runtime::target::Format,
        samples: u8,
    ) -> Result<()> {
        self.check_entry()?;
        self.complete(state::probe_render_target_format(
            self.raw(),
            format,
            samples,
        ))
    }
}

impl Context {
    /// Acquires one persistent named managed render target.
    ///
    /// Names are unique for the context until [`Self::release_render_target`] is called.
    ///
    /// # Errors
    /// Returns [`Error`] for an invalid descriptor, duplicate name, or allocation failure.
    pub fn acquire_render_target(
        &self,
        name: impl Into<String>,
        size: [u32; 2],
        descriptor: impl Into<RenderTargetDescriptor>,
    ) -> Result<RenderTargetHandle> {
        self.check_entry()?;
        let name = name.into();
        let descriptor = descriptor.into();
        let [width, height] = size;
        if name.is_empty()
            || self.inner.render_targets.borrow().contains_key(&name)
            || width == 0
            || height == 0
            || !matches!(descriptor.maximum_samples, 1 | 2 | 4 | 8)
        {
            return Err(Error::InvalidArgument);
        }
        let declaration = ez_gfx_runtime::target::TargetDeclaration::new(
            name.clone(),
            ez_gfx_runtime::target::TargetUsage::Color,
            1.0,
            descriptor.maximum_samples,
            vec![descriptor.color_format],
            ez_gfx_runtime::target::ClearValue::Color(descriptor.clear_color),
            true,
        )
        .map_err(|_| Error::InvalidArgument)?;
        let raw = self.complete(state::create_render_target(
            self.raw(),
            &declaration,
            descriptor.depth_format,
            width,
            height,
        ))?;
        self.inner.render_targets.borrow_mut().insert(
            name.clone(),
            CachedRenderTarget {
                handle: raw,
                format: descriptor.color_format,
                extent: (width, height),
                depth_format: descriptor.depth_format,
            },
        );
        Ok(RenderTargetHandle {
            inner: Rc::new(RenderTargetInner {
                context: Rc::clone(&self.inner),
                backing: RenderTargetBacking::Managed(name),
            }),
        })
    }

    /// Releases one persistent managed render target.
    ///
    /// # Errors
    /// Returns [`Error`] when the target is foreign, stale, or referenced by a recording frame.
    pub fn release_render_target(&self, target: RenderTargetHandle) -> Result<()> {
        self.check_entry()?;
        if !Rc::ptr_eq(&self.inner, &target.inner.context) {
            return Err(Error::InvalidContext);
        }
        let name = target.name()?.to_owned();
        state::validate_render_target_release(self.raw(), target.inner.managed_handle()?)?;
        let Some(cached) = self.inner.render_targets.borrow_mut().remove(&name) else {
            return Err(Error::InvalidContext);
        };
        state::destroy_render_target(self.raw(), cached.handle);
        drop(target);
        Ok(())
    }
}
