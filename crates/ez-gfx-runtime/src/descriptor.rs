use ez_gfx_hal::{CompletionToken, QueueKind};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Identifies a bindless descriptor slot at a specific generation.
pub struct DescriptorHandle {
    /// Index of the descriptor slot.
    slot: u32,
    /// Generation required for the handle to remain valid.
    generation: u32,
}
impl DescriptorHandle {
    /// Returns the descriptor slot index.
    pub const fn slot(self) -> u32 {
        self.slot
    }
    /// Returns the descriptor slot generation.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
    retired: bool,
}

#[derive(Debug)]
/// Stores bindless descriptors behind generation-checked handles.
pub struct BindlessRegistry<T> {
    /// Descriptor storage indexed by handles.
    slots: Vec<Slot<T>>,
    /// Indices available for reuse.
    free: Vec<u32>,
    /// Maximum number of descriptor slots.
    capacity: u32,
}

impl<T> BindlessRegistry<T> {
    /// Creates an empty registry with the specified slot capacity.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::InvalidCapacity` if `capacity` is zero.
    pub fn new(capacity: u32) -> Result<Self, DescriptorError> {
        if capacity == 0 {
            return Err(DescriptorError::InvalidCapacity);
        }
        Ok(Self {
            slots: Vec::new(),
            free: Vec::new(),
            capacity,
        })
    }

    /// Reuse advances generation; exhausted generations retire slots permanently.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::CapacityExhausted` if no reusable slot is available and the registry is at capacity or its slot count cannot be represented as `u32`.
    pub fn insert(&mut self, value: T) -> Result<DescriptorHandle, DescriptorError> {
        while let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            if slot.retired || slot.value.is_some() {
                continue;
            }
            slot.value = Some(value);
            return Ok(DescriptorHandle {
                slot: index,
                generation: slot.generation,
            });
        }
        if self.slots.len() >= self.capacity as usize {
            return Err(DescriptorError::CapacityExhausted);
        }
        let index =
            u32::try_from(self.slots.len()).map_err(|_| DescriptorError::CapacityExhausted)?;
        self.slots.push(Slot {
            generation: 1,
            value: Some(value),
            retired: false,
        });
        Ok(DescriptorHandle {
            slot: index,
            generation: 1,
        })
    }

    /// Returns the descriptor referenced by a live handle.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::StaleHandle` if the handle references a missing, retired, removed, or different-generation slot.
    pub fn get(&self, handle: DescriptorHandle) -> Result<&T, DescriptorError> {
        let slot = self
            .slots
            .get(handle.slot as usize)
            .ok_or(DescriptorError::StaleHandle)?;
        if slot.retired || slot.generation != handle.generation {
            return Err(DescriptorError::StaleHandle);
        }
        slot.value.as_ref().ok_or(DescriptorError::StaleHandle)
    }

    /// Removes and returns the descriptor referenced by a live handle.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::StaleHandle` if the handle references a missing, retired, removed, or different-generation slot.
    pub fn remove(&mut self, handle: DescriptorHandle) -> Result<T, DescriptorError> {
        let slot = self
            .slots
            .get_mut(handle.slot as usize)
            .ok_or(DescriptorError::StaleHandle)?;
        if slot.retired || slot.generation != handle.generation {
            return Err(DescriptorError::StaleHandle);
        }
        let value = slot.value.take().ok_or(DescriptorError::StaleHandle)?;
        match slot.generation.checked_add(1) {
            Some(next) => {
                slot.generation = next;
                self.free.push(handle.slot);
            }
            None => slot.retired = true,
        }
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes a contiguous range in a frame descriptor arena.
pub struct DescriptorRange {
    /// Index of the first descriptor in the range.
    pub start: u32,
    /// Number of descriptors in the range.
    pub count: u32,
}

#[derive(Debug)]
/// Allocates linear descriptor ranges for one frame of GPU work.
pub struct FrameDescriptorArena {
    /// Maximum number of descriptors available per frame.
    capacity: u32,
    /// Index where the next allocation begins.
    cursor: u32,
    /// GPU completion token that gates arena reuse.
    retirement: Option<CompletionToken>,
}

impl FrameDescriptorArena {
    /// Creates an empty frame arena with the specified descriptor capacity.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::InvalidCapacity` if `capacity` is zero.
    pub fn new(capacity: u32) -> Result<Self, DescriptorError> {
        if capacity == 0 {
            return Err(DescriptorError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            cursor: 0,
            retirement: None,
        })
    }

    /// Zero allocations and allocations after retirement are rejected; ranges are linear until reset.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::InvalidCount` if `count` is zero, `DescriptorError::GpuWorkPending` if the arena is retired, or `DescriptorError::CapacityExhausted` if the range overflows or exceeds capacity.
    pub fn allocate(&mut self, count: u32) -> Result<DescriptorRange, DescriptorError> {
        if count == 0 {
            return Err(DescriptorError::InvalidCount);
        }
        if self.retirement.is_some() {
            return Err(DescriptorError::GpuWorkPending);
        }
        let end = self
            .cursor
            .checked_add(count)
            .ok_or(DescriptorError::CapacityExhausted)?;
        if end > self.capacity {
            return Err(DescriptorError::CapacityExhausted);
        }
        let range = DescriptorRange {
            start: self.cursor,
            count,
        };
        self.cursor = end;
        Ok(range)
    }

    /// Retirement is single-assignment until reset, preventing completion-token replacement races.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::AlreadyRetired` if a retirement token is already recorded.
    pub fn retire(&mut self, completion: CompletionToken) -> Result<(), DescriptorError> {
        if self.retirement.is_some() {
            return Err(DescriptorError::AlreadyRetired);
        }
        self.retirement = Some(completion);
        Ok(())
    }

    /// Queue must match and its completed value must reach the recorded token before reuse.
    ///
    /// # Errors
    ///
    /// Returns `DescriptorError::WrongQueue` if `queue` differs from the retirement token's queue, or `DescriptorError::GpuWorkPending` if `completed` has not reached the token's value.
    pub fn reset(&mut self, queue: QueueKind, completed: u64) -> Result<(), DescriptorError> {
        if let Some(retirement) = self.retirement {
            if retirement.queue != queue {
                return Err(DescriptorError::WrongQueue);
            }
            if completed < retirement.value {
                return Err(DescriptorError::GpuWorkPending);
            }
        }
        self.cursor = 0;
        self.retirement = None;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Reports descriptor allocation, handle, and retirement failures.
pub enum DescriptorError {
    /// The requested capacity is zero.
    InvalidCapacity,
    /// The requested descriptor count is zero.
    InvalidCount,
    /// The registry or arena lacks space for the request.
    CapacityExhausted,
    /// The handle references a missing, removed, retired, or reused slot.
    StaleHandle,
    /// Recorded GPU work has not completed.
    GpuWorkPending,
    /// The completion query uses a different queue than the retirement token.
    WrongQueue,
    /// The arena already has a retirement token.
    AlreadyRetired,
}
