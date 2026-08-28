use core::fmt;

const CONTEXT_SLOT_BITS: u32 = 20;
const CONTEXT_GENERATION_BITS: u32 = 20;
const CHILD_SLOT_BITS: u32 = 12;
const CHILD_GENERATION_BITS: u32 = 12;
const CONTEXT_SLOT_MASK: u64 = (1 << CONTEXT_SLOT_BITS) - 1;
const CONTEXT_GENERATION_MASK: u64 = (1 << CONTEXT_GENERATION_BITS) - 1;
const CHILD_SLOT_MASK: u64 = (1 << CHILD_SLOT_BITS) - 1;
const CHILD_GENERATION_MASK: u64 = (1 << CHILD_GENERATION_BITS) - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandleError {
    Null,
    ZeroGeneration,
    SlotOutOfRange,
    GenerationOutOfRange,
    Malformed,
    Stale,
    CapacityExhausted,
    GenerationExhausted,
}

impl fmt::Display for HandleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HandleError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LocalHandle {
    slot: u32,
    generation: u32,
}

impl LocalHandle {
    /// Generation zero is reserved so zero-filled boundary values never become valid handles.
    pub fn new(slot: u32, generation: u32) -> Result<Self, HandleError> {
        if generation == 0 {
            return Err(HandleError::ZeroGeneration);
        }

        Ok(Self { slot, generation })
    }

    pub const fn slot(self) -> u32 {
        self.slot
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandleParts {
    Context(LocalHandle),
    Child {
        owner: LocalHandle,
        child: LocalHandle,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct PackedHandle(u64);

impl PackedHandle {
    /// Context slot values are stored as slot+1; the largest bit-pattern is therefore not a slot.
    pub fn context(local: LocalHandle) -> Result<Self, HandleError> {
        validate_local(local, CONTEXT_SLOT_MASK, CONTEXT_GENERATION_MASK)?;
        let raw = u64::from(local.slot + 1) | (u64::from(local.generation) << CONTEXT_SLOT_BITS);
        Ok(Self(raw))
    }

    /// Child identity is scoped by its owner; resource kind remains an arena-side validation.
    pub fn child(owner: LocalHandle, child: LocalHandle) -> Result<Self, HandleError> {
        let context = Self::context(owner)?;
        validate_local(child, CHILD_SLOT_MASK, CHILD_GENERATION_MASK)?;
        let raw = context.0
            | (u64::from(child.slot + 1) << (CONTEXT_SLOT_BITS + CONTEXT_GENERATION_BITS))
            | (u64::from(child.generation)
                << (CONTEXT_SLOT_BITS + CONTEXT_GENERATION_BITS + CHILD_SLOT_BITS));
        Ok(Self(raw))
    }

    /// Rejects zero and internally inconsistent child slot/generation pairs before use.
    pub fn from_raw(raw: u64) -> Result<Self, HandleError> {
        let handle = Self(raw);
        handle.parts()?;
        Ok(handle)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// A zero child slot denotes a context only when its child generation is also zero.
    pub fn parts(self) -> Result<HandleParts, HandleError> {
        if self.0 == 0 {
            return Err(HandleError::Null);
        }

        let context_slot = self.0 & CONTEXT_SLOT_MASK;
        let context_generation = (self.0 >> CONTEXT_SLOT_BITS) & CONTEXT_GENERATION_MASK;
        let child_slot =
            (self.0 >> (CONTEXT_SLOT_BITS + CONTEXT_GENERATION_BITS)) & CHILD_SLOT_MASK;
        let child_generation = (self.0
            >> (CONTEXT_SLOT_BITS + CONTEXT_GENERATION_BITS + CHILD_SLOT_BITS))
            & CHILD_GENERATION_MASK;

        if context_slot == 0 || context_generation == 0 {
            return Err(HandleError::ZeroGeneration);
        }

        let owner = LocalHandle::new((context_slot - 1) as u32, context_generation as u32)?;
        match (child_slot, child_generation) {
            (0, 0) => Ok(HandleParts::Context(owner)),
            (0, _) | (_, 0) => Err(HandleError::Malformed),
            _ => Ok(HandleParts::Child {
                owner,
                child: LocalHandle::new((child_slot - 1) as u32, child_generation as u32)?,
            }),
        }
    }
}

fn validate_local(
    handle: LocalHandle,
    slot_mask: u64,
    generation_mask: u64,
) -> Result<(), HandleError> {
    if u64::from(handle.slot) + 1 > slot_mask {
        return Err(HandleError::SlotOutOfRange);
    }
    if u64::from(handle.generation) > generation_mask {
        return Err(HandleError::GenerationOutOfRange);
    }

    Ok(())
}

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
    retired: bool,
}

#[derive(Debug)]
pub struct GenerationalArena<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    live: usize,
}

impl<T> Default for GenerationalArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GenerationalArena<T> {
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// Retired exhausted slots are skipped; growth fails before a slot index exceeds `u32`.
    pub fn insert(&mut self, value: T) -> Result<LocalHandle, HandleError> {
        while let Some(slot_index) = self.free.pop() {
            let slot = &mut self.slots[slot_index as usize];
            if slot.retired || slot.value.is_some() {
                continue;
            }
            slot.value = Some(value);
            self.live += 1;
            return LocalHandle::new(slot_index, slot.generation);
        }

        let slot_index =
            u32::try_from(self.slots.len()).map_err(|_| HandleError::CapacityExhausted)?;
        self.slots.push(Slot {
            generation: 1,
            value: Some(value),
            retired: false,
        });
        self.live += 1;
        LocalHandle::new(slot_index, 1)
    }

    /// A stale, vacant, or out-of-range handle is indistinguishable to callers.
    pub fn get(&self, handle: LocalHandle) -> Result<&T, HandleError> {
        let slot = self
            .slots
            .get(handle.slot as usize)
            .ok_or(HandleError::Stale)?;
        if slot.retired || slot.generation != handle.generation {
            return Err(HandleError::Stale);
        }
        slot.value.as_ref().ok_or(HandleError::Stale)
    }

    pub fn get_mut(&mut self, handle: LocalHandle) -> Result<&mut T, HandleError> {
        let slot = self
            .slots
            .get_mut(handle.slot as usize)
            .ok_or(HandleError::Stale)?;
        if slot.retired || slot.generation != handle.generation {
            return Err(HandleError::Stale);
        }
        slot.value.as_mut().ok_or(HandleError::Stale)
    }

    /// Generation exhaustion permanently retires the slot rather than wrapping into stale handles.
    pub fn remove(&mut self, handle: LocalHandle) -> Result<T, HandleError> {
        let slot = self
            .slots
            .get_mut(handle.slot as usize)
            .ok_or(HandleError::Stale)?;
        if slot.retired || slot.generation != handle.generation {
            return Err(HandleError::Stale);
        }
        let value = slot.value.take().ok_or(HandleError::Stale)?;
        self.live -= 1;
        advance_slot(slot, handle.slot, &mut self.free);
        Ok(value)
    }

    /// Live slots advance generation; already-free slots retain it and remain reusable.
    pub fn clear(&mut self) -> Result<(), HandleError> {
        self.free.clear();
        self.live = 0;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.value.take().is_some() {
                advance_slot(slot, index as u32, &mut self.free);
            } else if !slot.retired {
                self.free.push(index as u32);
            }
        }
        Ok(())
    }

    pub const fn len(&self) -> usize {
        self.live
    }

    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }
}

fn advance_slot<T>(slot: &mut Slot<T>, index: u32, free: &mut Vec<u32>) {
    match slot.generation.checked_add(1) {
        Some(generation) => {
            slot.generation = generation;
            free.push(index);
        }
        None => slot.retired = true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_generation_retires_slot() {
        let mut arena = GenerationalArena {
            slots: vec![Slot {
                generation: u32::MAX,
                value: Some(1),
                retired: false,
            }],
            free: Vec::new(),
            live: 1,
        };
        let exhausted = LocalHandle::new(0, u32::MAX).unwrap();
        assert_eq!(arena.remove(exhausted).unwrap(), 1);
        assert_ne!(arena.insert(2).unwrap().slot(), exhausted.slot());
    }
}
