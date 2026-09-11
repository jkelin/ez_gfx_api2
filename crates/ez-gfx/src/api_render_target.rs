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
    fn managed_handle(&self) -> Result<RenderTargetHandle> {
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
            RenderTargetBacking::Managed(_) | RenderTargetBacking::Surface { .. } => {
                Ok(ez_gfx_runtime::target::ClearValue::Color([
                    0.1, 0.1, 0.1, 1.0,
                ]))
            }
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
