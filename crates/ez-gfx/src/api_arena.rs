use core::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;

/// Generational identity of one arena element.
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct GpuArenaHandle<T> {
    slot: u32,
    generation: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> Clone for GpuArenaHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GpuArenaHandle<T> {}

impl<T> GpuArenaHandle<T> {
    const fn new(slot: u32, generation: u32) -> Self {
        Self {
            slot,
            generation,
            marker: PhantomData,
        }
    }

    /// Returns the stable slot index.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// Returns the slot generation.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Debug)]
struct ArenaSlot {
    generation: u32,
    occupied: bool,
}

#[derive(Debug)]
struct ArenaStorage<T> {
    slots: Vec<ArenaSlot>,
    values: Vec<T>,
    free: Vec<u32>,
    dirty: Vec<u64>,
    dirty_ranges: Vec<Range<usize>>,
    live: usize,
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> ArenaStorage<T> {
    fn new() -> Result<Self> {
        if core::mem::size_of::<T>() == 0 {
            return Err(Error::InvalidArgument);
        }
        Ok(Self {
            slots: vec![ArenaSlot {
                generation: 1,
                occupied: true,
            }],
            values: vec![T::zeroed()],
            free: Vec::new(),
            dirty: vec![1],
            dirty_ranges: Vec::new(),
            live: 0,
        })
    }

    fn validate(&self, handle: GpuArenaHandle<T>) -> Result<usize> {
        let index = handle.slot as usize;
        self.slots
            .get(index)
            .filter(|slot| slot.occupied && slot.generation == handle.generation)
            .map(|_| index)
            .ok_or(Error::InvalidArgument)
    }

    fn mark_dirty(&mut self, index: usize) {
        let word = index / 64;
        if self.dirty.len() <= word {
            self.dirty.resize(word + 1, 0);
        }
        self.dirty[word] |= 1_u64 << (index % 64);
    }

    fn reserve(&mut self) -> Result<GpuArenaHandle<T>> {
        let required = self
            .slots
            .len()
            .checked_add(usize::from(self.free.is_empty()))
            .ok_or(Error::InvalidArgument)?;
        required
            .checked_mul(core::mem::size_of::<T>())
            .filter(|bytes| *bytes <= MAX_FACADE_BUFFER_BYTES)
            .ok_or(Error::InvalidArgument)?;
        if required > u32::MAX as usize {
            return Err(Error::InvalidArgument);
        }

        let index = if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(!slot.occupied);
            slot.occupied = true;
            self.values[index as usize] = T::zeroed();
            index
        } else {
            let index = u32::try_from(self.slots.len()).map_err(|_| Error::InvalidArgument)?;
            self.slots.push(ArenaSlot {
                generation: 1,
                occupied: true,
            });
            self.values.push(T::zeroed());
            index
        };
        self.live += 1;
        self.mark_dirty(index as usize);
        Ok(GpuArenaHandle::new(
            index,
            self.slots[index as usize].generation,
        ))
    }

    fn reserve_many(&mut self, count: usize) -> Result<Vec<GpuArenaHandle<T>>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let required = self
            .slots
            .len()
            .checked_add(count.saturating_sub(self.free.len()))
            .ok_or(Error::InvalidArgument)?;
        let bytes = required
            .checked_mul(core::mem::size_of::<T>())
            .filter(|bytes| *bytes <= MAX_FACADE_BUFFER_BYTES)
            .ok_or(Error::InvalidArgument)?;
        let _ = bytes;
        if required > u32::MAX as usize {
            return Err(Error::InvalidArgument);
        }

        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            let index = if let Some(index) = self.free.pop() {
                let slot = &mut self.slots[index as usize];
                debug_assert!(!slot.occupied);
                slot.occupied = true;
                self.values[index as usize] = T::zeroed();
                index
            } else {
                let index =
                    u32::try_from(self.slots.len()).map_err(|_| Error::InvalidArgument)?;
                self.slots.push(ArenaSlot {
                    generation: 1,
                    occupied: true,
                });
                self.values.push(T::zeroed());
                index
            };
            self.live += 1;
            self.mark_dirty(index as usize);
            handles.push(GpuArenaHandle::new(
                index,
                self.slots[index as usize].generation,
            ));
        }
        Ok(handles)
    }

    fn set(&mut self, handle: GpuArenaHandle<T>, value: T) -> Result<()> {
        let index = self.validate(handle)?;
        if bytemuck::bytes_of(&self.values[index]) == bytemuck::bytes_of(&value) {
            return Ok(());
        }
        self.values[index] = value;
        self.mark_dirty(index);
        Ok(())
    }

    fn set_many(&mut self, values: &[(GpuArenaHandle<T>, T)]) -> Result<()> {
        for (handle, _) in values {
            self.validate(*handle)?;
        }
        for (handle, value) in values {
            let index = self.validate(*handle)?;
            if bytemuck::bytes_of(&self.values[index]) != bytemuck::bytes_of(value) {
                self.values[index] = *value;
                self.mark_dirty(index);
            }
        }
        Ok(())
    }

    fn get(&self, handle: GpuArenaHandle<T>) -> Result<T> {
        self.validate(handle).map(|index| self.values[index])
    }

    fn get_many(&self, handles: &[GpuArenaHandle<T>]) -> Result<Vec<T>> {
        handles.iter().map(|handle| self.get(*handle)).collect()
    }

    fn remove(&mut self, handle: GpuArenaHandle<T>) -> Result<T> {
        let index = self.validate(handle)?;
        let slot = &mut self.slots[index];
        if slot.generation == u32::MAX {
            return Err(Error::InvalidArgument);
        }
        let value = self.values[index];
        self.values[index] = T::zeroed();
        slot.occupied = false;
        slot.generation += 1;
        self.free.push(handle.slot);
        self.live -= 1;
        self.mark_dirty(index);
        Ok(value)
    }
    fn remove_many(&mut self, handles: &[GpuArenaHandle<T>]) -> Result<Vec<T>> {
        for (index, handle) in handles.iter().enumerate() {
            self.validate(*handle)?;
            if handles[..index]
                .iter()
                .any(|prior| prior.slot == handle.slot)
            {
                return Err(Error::InvalidArgument);
            }
        }
        handles.iter().map(|handle| self.remove(*handle)).collect()
    }

    fn prepare_dirty_ranges(&mut self) {
        self.dirty_ranges.clear();
        if self.dirty.iter().all(|word| *word == 0) {
            return;
        }
        let mut start = None;
        for index in 0..self.slots.len() {
            let marked = self
                .dirty
                .get(index / 64)
                .is_some_and(|word| word & (1_u64 << (index % 64)) != 0);
            match (start, marked) {
                (None, true) => start = Some(index),
                (Some(first), false) => {
                    self.dirty_ranges.push(first..index);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(first) = start {
            self.dirty_ranges.push(first..self.slots.len());
        }
    }

    fn clear_dirty(&mut self) {
        self.dirty.fill(0);
    }
}

#[derive(Debug)]
struct ArenaInner<T> {
    owner: ez_gfx_core::handle::ContextHandle,
    handle: ez_gfx_core::handle::BufferHandle,
    storage: Mutex<ArenaStorage<T>>,
}

impl<T> Drop for ArenaInner<T> {
    fn drop(&mut self) {
        state::release_gpu_arena_buffer(self.owner, self.handle);
    }
}

impl<T> ArenaInner<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    fn synchronize<R>(
        &self,
        operation: impl FnOnce(&[T], &[Range<usize>]) -> Result<R>,
    ) -> Result<R> {
        let mut storage = self.storage.lock();
        storage.prepare_dirty_ranges();
        let result = operation(&storage.values, &storage.dirty_ranges);
        if result.is_ok() {
            storage.clear_dirty();
        }
        result
    }
}

/// Thread-safe typed arena whose contents become shader-readable when bound.
#[derive(Debug)]
pub struct GpuArena<T> {
    inner: Arc<ArenaInner<T>>,
}

impl<T> Clone for GpuArena<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> GpuArena<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Send + Sync,
{
    /// Reserves one zero-initialized element.
    ///
    /// Slot zero is permanently reserved as the shader-side "no data" sentinel.
    ///
    /// # Errors
    /// Returns [`Error::NativeFailure`] when arena identity is exhausted.
    pub fn reserve(&self) -> Result<GpuArenaHandle<T>> {
        self.inner.storage.lock().reserve()
    }

    /// Reserves `count` zero-initialized elements.
    ///
    /// Slot zero is permanently reserved as the shader-side "no data" sentinel.
    ///
    /// # Errors
    /// Returns an error when the count overflows or identities are exhausted.
    pub fn reserve_many(&self, count: usize) -> Result<Vec<GpuArenaHandle<T>>> {
        self.inner.storage.lock().reserve_many(count)
    }

    /// Replaces one live element.
    ///
    /// # Errors
    /// Returns [`Error::InvalidArgument`] for a stale or foreign handle.
    pub fn set(&self, handle: GpuArenaHandle<T>, value: T) -> Result<()> {
        self.inner.storage.lock().set(handle, value)
    }

    /// Atomically validates and replaces several live elements.
    ///
    /// # Errors
    /// Returns [`Error::InvalidArgument`] if any handle is stale, foreign, or duplicated.
    pub fn set_many(&self, values: &[(GpuArenaHandle<T>, T)]) -> Result<()> {
        self.inner.storage.lock().set_many(values)
    }

    /// Reads one live element.
    ///
    /// # Errors
    /// Returns [`Error::InvalidArgument`] for a stale or foreign handle.
    pub fn get(&self, handle: GpuArenaHandle<T>) -> Result<T> {
        self.inner.storage.lock().get(handle)
    }

    /// Reads several live elements after validating every handle.
    ///
    /// # Errors
    /// Returns [`Error::InvalidArgument`] if any handle is stale or foreign.
    pub fn get_many(&self, handles: &[GpuArenaHandle<T>]) -> Result<Vec<T>> {
        self.inner.storage.lock().get_many(handles)
    }

    /// Removes one live element and returns its previous value.
    ///
    /// # Errors
    /// Returns [`Error::InvalidArgument`] for a stale or foreign handle.
    pub fn remove(&self, handle: GpuArenaHandle<T>) -> Result<T> {
        self.inner.storage.lock().remove(handle)
    }

    /// Atomically validates and removes several distinct live elements.
    ///
    /// # Errors
    /// Returns [`Error::InvalidArgument`] if any handle is stale, foreign, or duplicated.
    pub fn remove_many(&self, handles: &[GpuArenaHandle<T>]) -> Result<Vec<T>> {
        self.inner.storage.lock().remove_many(handles)
    }

    /// Returns the number of live elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.storage.lock().live
    }

    /// Returns whether the arena contains no live elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.storage.lock().live == 0
    }
}

impl Context {
    /// Creates a thread-safe typed arena owned by this context.
    ///
    /// # Errors
    /// Returns [`Error`] for zero-sized element types, exhausted identities, or
    /// callback reentrancy.
    pub fn create_gpu_arena<T>(&self) -> Result<GpuArena<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Send + Sync,
    {
        self.check_entry()?;
        let element_size =
            u32::try_from(core::mem::size_of::<T>()).map_err(|_| Error::InvalidArgument)?;
        let handle = state::create_gpu_arena_buffer(self.raw(), element_size)?;
        Ok(GpuArena {
            inner: Arc::new(ArenaInner {
                owner: self.raw(),
                handle,
                storage: Mutex::new(ArenaStorage::new()?),
            }),
        })
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn slot_zero_is_reserved_for_missing_data() {
        let mut storage = ArenaStorage::<u32>::new().unwrap();
        assert_eq!(storage.live, 0);

        let first = storage.reserve().unwrap();
        assert_eq!(first.slot(), 1);
        assert_eq!(storage.values, vec![0, 0]);
    }
    #[test]
    fn stale_handles_are_rejected_after_slot_reuse() {
        let mut storage = ArenaStorage::<u32>::new().unwrap();
        let first = storage.reserve().unwrap();
        storage.set(first, 17).unwrap();
        storage.remove(first).unwrap();
        let replacement = storage.reserve().unwrap();

        assert_eq!(first.slot(), replacement.slot());
        assert_ne!(first.generation(), replacement.generation());
        assert_eq!(storage.get(first), Err(Error::InvalidArgument));
        assert_eq!(storage.get(replacement), Ok(0));
    }

    #[test]
    fn batch_mutations_validate_before_changing_storage() {
        let mut storage = ArenaStorage::<u32>::new().unwrap();
        let handles = storage.reserve_many(2).unwrap();
        let stale = GpuArenaHandle::new(handles[0].slot(), handles[0].generation() + 1);

        assert_eq!(
            storage.set_many(&[(handles[0], 7), (stale, 9)]),
            Err(Error::InvalidArgument)
        );
        assert_eq!(storage.get_many(&handles).unwrap(), vec![0, 0]);
    }

    #[test]
    fn dirty_regions_merge_adjacent_slots() {
        let mut storage = ArenaStorage::<u32>::new().unwrap();
        let handles = storage.reserve_many(5).unwrap();
        storage.clear_dirty();
        storage.set(handles[1], 1).unwrap();
        storage.set(handles[2], 2).unwrap();
        storage.set(handles[4], 4).unwrap();

        storage.prepare_dirty_ranges();
        assert_eq!(storage.dirty_ranges, vec![2..4, 5..6]);
        assert_eq!(storage.values, vec![0, 0, 1, 2, 0, 4]);
    }

    #[test]
    fn unchanged_values_do_not_dirty_the_arena() {
        let mut storage = ArenaStorage::<u32>::new().unwrap();
        let handle = storage.reserve().unwrap();
        storage.clear_dirty();

        storage.set(handle, 0).unwrap();
        storage.prepare_dirty_ranges();

        assert!(storage.dirty_ranges.is_empty());
    }
}
