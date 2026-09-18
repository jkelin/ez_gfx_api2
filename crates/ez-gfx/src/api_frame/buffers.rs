#[derive(Clone, Copy, Eq, PartialEq)]
enum BufferUse {
    Available,
    Claimed,
    Consumed,
}

mod buffer_data {
    pub trait Sealed {}

    impl<T: bytemuck::Pod> Sealed for super::BufferSource<'_, T> {}
    impl<T: bytemuck::Pod> Sealed for &[T] {}
    impl<T: bytemuck::Pod> Sealed for &Vec<T> {}
}

/// Explicit source for one POD buffer element.
///
/// Slices and vectors can be passed directly to context acquisition helpers.
/// Arrays use `.as_slice()` so they cannot be mistaken for one array-valued element.
pub struct BufferSource<'a, T: bytemuck::Pod> {
    value: &'a T,
}

impl<'a, T: bytemuck::Pod> BufferSource<'a, T> {
    /// Borrows one element without allocating an intermediate collection.
    pub const fn one(value: &'a T) -> Self {
        Self { value }
    }
}

/// POD input accepted by context buffer acquisition helpers.
pub trait BufferData: buffer_data::Sealed {
    /// Element stored by the acquired buffer.
    type Element: bytemuck::Pod;

    #[doc(hidden)]
    fn as_buffer_slice(&self) -> &[Self::Element];
}

impl<T: bytemuck::Pod> BufferData for BufferSource<'_, T> {
    type Element = T;

    fn as_buffer_slice(&self) -> &[T] {
        core::slice::from_ref(self.value)
    }
}

impl<T: bytemuck::Pod> BufferData for &[T] {
    type Element = T;

    fn as_buffer_slice(&self) -> &[T] {
        self
    }
}

impl<T: bytemuck::Pod> BufferData for &Vec<T> {
    type Element = T;

    fn as_buffer_slice(&self) -> &[T] {
        self
    }
}

fn buffer_data_slice<D: BufferData + ?Sized>(data: &D) -> &[D::Element] {
    data.as_buffer_slice()
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum TransientState {
    Live,
    Consumed,
}

enum TransientHandle {
    Buffer(BufferHandle),
    Counter(CounterBufferHandle),
}

struct TransientInner {
    context: Rc<ContextInner>,
    buffer: Rc<BufferInner>,
    handle: TransientHandle,
    state: Cell<TransientState>,
}

impl Drop for TransientInner {
    fn drop(&mut self) {
        // The frame owns the sole native lease after successful materialization.
        if self.state.get() != TransientState::Live {
            return;
        }
        // Successful completion consumes raw transients. Only live, abandoned
        // transactions reach this release path after frame rollback.
        match self.handle {
            TransientHandle::Buffer(handle) => {
                state::release_buffer(self.context.handle, handle);
            }
            TransientHandle::Counter(handle) => {
                state::release_counter(self.context.handle, handle);
            }
        }
    }
}

/// Caps reusable facade storage without rejecting valid one-frame acquisitions.
const MAX_FACADE_BUFFER_POOL_ENTRIES: usize = 64;
const MAX_FACADE_BUFFER_POOL_BYTES: usize = 32 * 1024 * 1024;
const MAX_FACADE_BUFFER_BYTES: usize = 16 * 1024 * 1024;

fn next_buffer_capacity(current: usize, required: usize) -> usize {
    if required <= current {
        return current;
    }
    let required_power = match required.checked_next_power_of_two() {
        Some(value) => value,
        None => usize::MAX,
    };
    let doubled = current.saturating_mul(2);
    if doubled > required_power {
        doubled
    } else {
        required_power
    }
}


const fn facade_pool_can_retain(
    entry_count: usize,
    retained_bytes: usize,
    candidate_bytes: usize,
) -> bool {
    entry_count < MAX_FACADE_BUFFER_POOL_ENTRIES
        && retained_bytes.saturating_add(candidate_bytes) <= MAX_FACADE_BUFFER_POOL_BYTES
}

struct BufferInner {
    context: Weak<ContextInner>,
    element_size: u32,
    element_count: Cell<u32>,
    bytes: RefCell<Vec<u8>>,
    published_count: Cell<u32>,
    usage: Cell<BufferUse>,
}

impl BufferInner {
    fn write<T: bytemuck::Pod>(&self, start_index: usize, values: &[T]) -> Result<()> {
        // Claimed and consumed one-frame values cannot be rewritten.
        if self.usage.get() != BufferUse::Available {
            return Err(Error::NotReady);
        }
        if core::mem::size_of::<T>() != self.element_size as usize {
            return Err(Error::InvalidArgument);
        }
        let end = start_index
            .checked_add(values.len())
            .filter(|end| *end <= self.element_count.get() as usize)
            .ok_or(Error::InvalidArgument)?;
        let start_byte = start_index
            .checked_mul(self.element_size as usize)
            .ok_or(Error::InvalidArgument)?;
        let end_byte = end
            .checked_mul(self.element_size as usize)
            .ok_or(Error::InvalidArgument)?;
        self.bytes.borrow_mut()[start_byte..end_byte].copy_from_slice(bytemuck::cast_slice(values));
        Ok(())
    }
}

/// Context-owned typed buffer uploaded when a frame first binds it.
pub struct Buffer<T: bytemuck::Pod> {
    context: Rc<ContextInner>,
    inner: Rc<BufferInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    /// Replaces a typed element range.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is dispatching a callback or the range is invalid.
    pub fn write(&self, start_index: usize, values: &[T]) -> Result<()> {
        let context = Context { inner: Rc::clone(&self.context), owner: false, };
        context.check_entry()?;
        self.inner.write(start_index, values)
    }
}

/// Context-acquired one-frame buffer with a shader-writable count.
pub struct CounterBuffer<T: bytemuck::Pod> {
    context: Rc<ContextInner>,
    inner: Rc<BufferInner>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> CounterBuffer<T> {
    /// Replaces a typed element range and advances the initial visible count.
    ///
    /// GPU producers can replace that count with `set_count` or `add_count`.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is dispatching a callback or the range is invalid.
    pub fn write(&self, start_index: usize, values: &[T]) -> Result<()> {
        let context = Context { inner: Rc::clone(&self.context), owner: false, };
        context.check_entry()?;
        self.inner.write(start_index, values)?;
        let end = start_index
            .checked_add(values.len())
            .and_then(|end| u32::try_from(end).ok())
            .ok_or(Error::InvalidArgument)?;
        self.inner.published_count.set(self.inner.published_count.get().max(end));
        Ok(())
    }

    /// Appends one element and returns its buffer index.
    ///
    /// Capacity grows geometrically before the buffer is claimed by a frame.
    ///
    /// # Errors
    /// Returns [`Error`] when the context is dispatching a callback, the buffer
    /// is already claimed, or the enlarged buffer would exceed facade limits.
    pub fn add(&self, value: T) -> Result<usize> {
        let context = Context {
            inner: Rc::clone(&self.context),
            owner: false,
        };
        context.check_entry()?;
        if self.inner.usage.get() != BufferUse::Available {
            return Err(Error::NotReady);
        }

        let index = self.inner.published_count.get() as usize;
        let required = index.checked_add(1).ok_or(Error::InvalidArgument)?;
        let capacity = self.inner.element_count.get() as usize;
        if required > capacity {
            let grown = next_buffer_capacity(capacity, required);
            let byte_count = grown
                .checked_mul(self.inner.element_size as usize)
                .filter(|bytes| *bytes <= MAX_FACADE_BUFFER_BYTES)
                .ok_or(Error::InvalidArgument)?;
            let grown = u32::try_from(grown).map_err(|_| Error::InvalidArgument)?;
            self.inner.bytes.borrow_mut().resize(byte_count, 0);
            self.inner.element_count.set(grown);
        }

        self.inner.write(index, core::slice::from_ref(&value))?;
        self.inner
            .published_count
            .set(u32::try_from(required).map_err(|_| Error::InvalidArgument)?);
        Ok(index)
    }

    /// Returns the published element count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.published_count.get() as usize
    }

    /// Returns whether no elements are published.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.published_count.get() == 0
    }

    /// Returns the allocated element capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.element_count.get() as usize
    }
}

/// Context-acquired one-frame buffer containing exactly one POD value.
pub struct ValueBuffer<T: bytemuck::Pod> {
    /// Keeps the creating context alive exactly as long as this public resource wrapper.
    context: Rc<ContextInner>,
    inner: Rc<BufferInner>,
    marker: PhantomData<T>,
}

impl Context {
    fn allocate_buffer<T: bytemuck::Pod>(
        &self,
        element_count: usize,
        initial: Option<&[T]>,
        published_count: u32,
    ) -> Result<Rc<BufferInner>> {
        // Empty and zero-sized buffers cannot produce valid native bindings.
        self.check_entry()?;
        let element_size =
            u32::try_from(core::mem::size_of::<T>()).map_err(|_| Error::InvalidArgument)?;
        let element_count = u32::try_from(element_count).map_err(|_| Error::InvalidArgument)?;
        let byte_count = (element_size as usize)
            .checked_mul(element_count as usize)
            .filter(|size| {
                element_size != 0 && element_count != 0 && *size <= MAX_FACADE_BUFFER_BYTES
            })
            .ok_or(Error::InvalidArgument)?;
        if initial.is_some_and(|values| values.len() != element_count as usize) {
            return Err(Error::InvalidArgument);
        }
        let mut pool = self.inner.facade_buffers.borrow_mut();
        if let Some(inner) = pool.iter().find(|inner| {
            Rc::strong_count(inner) == 1
                && inner.element_size == element_size
                && inner.element_count.get() == element_count
        }) {
            let mut bytes = inner.bytes.borrow_mut();
            bytes.resize(byte_count, 0);
            if let Some(values) = initial {
                bytes.copy_from_slice(bytemuck::cast_slice(values));
            } else {
                bytes.fill(0);
            }
            inner.published_count.set(published_count);
            inner.element_count.set(element_count);
            inner.usage.set(BufferUse::Available);
            return Ok(Rc::clone(inner));
        }
        let mut bytes = vec![0; byte_count];
        if let Some(values) = initial {
            bytes.copy_from_slice(bytemuck::cast_slice(values));
        }
        let inner = Rc::new(BufferInner {
            context: Rc::downgrade(&self.inner),
            element_size,
            element_count: Cell::new(element_count),
            bytes: RefCell::new(bytes),
            published_count: Cell::new(published_count),
            usage: Cell::new(BufferUse::Available),
        });
        // Oversized or shape-heavy workloads remain valid but bypass retention once bounded.
        let retained_bytes = pool.iter().fold(0_usize, |total, buffer| {
            total.saturating_add(buffer.bytes.borrow().capacity())
        });
        if facade_pool_can_retain(pool.len(), retained_bytes, byte_count) {
            pool.push(Rc::clone(&inner));
        }
        Ok(inner)
    }

    /// Acquires a one-frame typed buffer by element count.
    ///
    /// # Errors
    /// Returns [`Error`] when the count, type size, or context is invalid.
    pub fn acquire_buffer<T: bytemuck::Pod>(&self, element_count: usize) -> Result<Buffer<T>> {
        Ok(Buffer {
            context: Rc::clone(&self.inner),
            inner: self.allocate_buffer::<T>(element_count, None, 0)?,
            marker: PhantomData,
        })
    }

    /// Acquires a correctly-sized one-frame typed buffer initialized from POD data.
    ///
    /// # Errors
    /// Returns [`Error`] when the input is empty, oversized, zero-sized, or the context is invalid.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the source type carries element inference and prevents scalar/array ambiguity"
    )]
    pub fn acquire_buffer_from<D: BufferData>(&self, data: D) -> Result<Buffer<D::Element>> {
        let values = buffer_data_slice(&data);
        Ok(Buffer {
            context: Rc::clone(&self.inner),
            inner: self.allocate_buffer(values.len(), Some(values), 0)?,
            marker: PhantomData,
        })
    }

    /// Acquires a one-frame buffer containing exactly one POD value.
    ///
    /// # Errors
    pub fn acquire_value_buffer<T: bytemuck::Pod>(&self, value: T) -> Result<ValueBuffer<T>> {
        Ok(ValueBuffer {
            context: Rc::clone(&self.inner),
            inner: self.allocate_buffer(1, Some(core::slice::from_ref(&value)), 0)?,
            marker: PhantomData,
        })
    }

    /// Acquires a one-frame typed counter buffer by element count.
    ///
    /// # Errors
    /// Returns [`Error`] when the count, type size, or context is invalid.
    pub fn acquire_counter_buffer<T: bytemuck::Pod>(
        &self,
        element_count: usize,
    ) -> Result<CounterBuffer<T>> {
        Ok(CounterBuffer {
            context: Rc::clone(&self.inner),
            inner: self.allocate_buffer::<T>(element_count, None, 0)?,
            marker: PhantomData,
        })
    }

    /// Acquires a correctly-sized one-frame counter buffer initialized from POD data.
    ///
    /// The initialized element count becomes the GPU-visible initial count.
    ///
    /// # Errors
    /// Returns [`Error`] when the input is empty, oversized, zero-sized, or the context is invalid.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the source type carries element inference and prevents scalar/array ambiguity"
    )]
    pub fn acquire_counter_buffer_from<D: BufferData>(
        &self,
        data: D,
    ) -> Result<CounterBuffer<D::Element>> {
        let values = buffer_data_slice(&data);
        let published_count = u32::try_from(values.len()).map_err(|_| Error::InvalidArgument)?;
        Ok(CounterBuffer {
            context: Rc::clone(&self.inner),
            inner: self.allocate_buffer(values.len(), Some(values), published_count)?,
            marker: PhantomData,
        })
    }
}
