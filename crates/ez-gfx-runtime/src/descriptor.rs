use ez_gfx_hal::{CompletionToken, QueueKind};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DescriptorHandle {
    slot: u32,
    generation: u32,
}
impl DescriptorHandle {
    pub const fn slot(self) -> u32 {
        self.slot
    }
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
pub struct BindlessRegistry<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    capacity: u32,
}

impl<T> BindlessRegistry<T> {
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
        let index = self.slots.len() as u32;
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
pub struct DescriptorRange {
    pub start: u32,
    pub count: u32,
}

#[derive(Debug)]
pub struct FrameDescriptorArena {
    capacity: u32,
    cursor: u32,
    retirement: Option<CompletionToken>,
}

impl FrameDescriptorArena {
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
    pub fn retire(&mut self, completion: CompletionToken) -> Result<(), DescriptorError> {
        if self.retirement.is_some() {
            return Err(DescriptorError::AlreadyRetired);
        }
        self.retirement = Some(completion);
        Ok(())
    }

    /// Queue must match and its completed value must reach the recorded token before reuse.
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
pub enum DescriptorError {
    InvalidCapacity,
    InvalidCount,
    CapacityExhausted,
    StaleHandle,
    GpuWorkPending,
    WrongQueue,
    AlreadyRetired,
}
