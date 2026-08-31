use core::fmt;

const CONTEXT_SLOT_BITS: u32 = 20;
const CONTEXT_GENERATION_BITS: u32 = 20;
const CHILD_SLOT_BITS: u32 = 12;
const CHILD_GENERATION_BITS: u32 = 12;
const CONTEXT_SLOT_MASK: u64 = (1 << CONTEXT_SLOT_BITS) - 1;
const CONTEXT_GENERATION_MASK: u64 = (1 << CONTEXT_GENERATION_BITS) - 1;
const CHILD_SLOT_MASK: u64 = (1 << CHILD_SLOT_BITS) - 1;
const CHILD_GENERATION_MASK: u64 = (1 << CHILD_GENERATION_BITS) - 1;

/// Errors produced while constructing, decoding, or using a generational handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandleError {
    /// The raw handle is zero.
    Null,
    /// A generation value is zero.
    ZeroGeneration,
    /// A slot exceeds the encoding or arena capacity.
    SlotOutOfRange,
    /// A generation exceeds the encoding range.
    GenerationOutOfRange,
    /// Slot and generation fields form an invalid combination.
    Malformed,
    /// The handle generation no longer matches the arena slot.
    Stale,
    /// The arena cannot allocate another slot.
    CapacityExhausted,
    /// A slot generation cannot be incremented.
    GenerationExhausted,
}

impl fmt::Display for HandleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HandleError {}

/// A slot index and generation identifying one arena value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LocalHandle {
    slot: u32,
    generation: u32,
}

impl LocalHandle {
    /// Generation zero is reserved so zero-filled boundary values never become valid handles.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::ZeroGeneration`] when `generation` is zero.
    pub fn new(slot: u32, generation: u32) -> Result<Self, HandleError> {
        if generation == 0 {
            return Err(HandleError::ZeroGeneration);
        }

        Ok(Self { slot, generation })
    }

    /// Returns the slot index.
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// Returns the generation.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Decoded context or child identity represented by a packed handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandleParts {
    /// A context handle.
    Context(LocalHandle),
    /// A child handle and its owning context.
    Child {
        /// Context owning the child.
        owner: LocalHandle,
        /// Child resource identity.
        child: LocalHandle,
    },
}

/// Compact wire representation of a context or child handle.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct PackedHandle(u64);

impl PackedHandle {
    /// Context slot values are stored as slot+1; the largest bit-pattern is therefore not a slot.
    ///
    /// # Errors
    ///
    /// Returns a range error when slot or generation exceeds the context encoding.
    pub fn context(local: LocalHandle) -> Result<Self, HandleError> {
        validate_local(local, CONTEXT_SLOT_MASK, CONTEXT_GENERATION_MASK)?;
        let raw = u64::from(local.slot + 1) | (u64::from(local.generation) << CONTEXT_SLOT_BITS);
        Ok(Self(raw))
    }

    /// Child identity is scoped by its owner; resource kind remains an arena-side validation.
    ///
    /// # Errors
    ///
    /// Returns a range error when either handle exceeds its packed encoding.
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
    ///
    /// # Errors
    ///
    /// Returns the decoding error reported by [`Self::parts`].
    pub fn from_raw(raw: u64) -> Result<Self, HandleError> {
        let handle = Self(raw);
        handle.parts()?;
        Ok(handle)
    }

    /// Returns the encoded handle value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// A zero child slot denotes a context only when its child generation is also zero.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] when the raw value is zero, has zero generations,
    /// or contains an inconsistent child pair.
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

        let owner = LocalHandle::new(
            u32::try_from(context_slot - 1).map_err(|_| HandleError::SlotOutOfRange)?,
            u32::try_from(context_generation).map_err(|_| HandleError::GenerationOutOfRange)?,
        )?;
        match (child_slot, child_generation) {
            (0, 0) => Ok(HandleParts::Context(owner)),
            (0, _) | (_, 0) => Err(HandleError::Malformed),
            _ => Ok(HandleParts::Child {
                owner,
                child: LocalHandle::new(
                    u32::try_from(child_slot - 1).map_err(|_| HandleError::SlotOutOfRange)?,
                    u32::try_from(child_generation)
                        .map_err(|_| HandleError::GenerationOutOfRange)?,
                )?,
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

/// Reusable storage keyed by generation-checked local handles.
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
    /// Creates an empty arena.
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// Retired exhausted slots are skipped; growth fails before a slot index exceeds `u32`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::CapacityExhausted`] when no new slot index fits
    /// in the handle representation.
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
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::Stale`] for an out-of-range, retired, vacant, or
    /// generation-mismatched handle.
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

    /// Returns a mutable reference when `handle` is current.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::Stale`] when `handle` does not identify a live slot.
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
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::Stale`] when `handle` does not identify a live slot.
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
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::SlotOutOfRange`] only if the arena cannot represent
    /// an existing platform index as a handle.
    pub fn clear(&mut self) -> Result<(), HandleError> {
        self.free.clear();
        self.live = 0;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.value.take().is_some() {
                advance_slot(
                    slot,
                    u32::try_from(index).map_err(|_| HandleError::SlotOutOfRange)?,
                    &mut self.free,
                );
            } else if !slot.retired {
                self.free
                    .push(u32::try_from(index).map_err(|_| HandleError::SlotOutOfRange)?);
            }
        }
        Ok(())
    }

    /// Returns the number of live values.
    pub const fn len(&self) -> usize {
        self.live
    }

    /// Returns whether the arena contains no live values.
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
