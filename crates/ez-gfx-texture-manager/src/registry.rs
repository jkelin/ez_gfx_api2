//! Generational texture registry: slots, residency, bindings, and events.

use super::TextureError;
use ez_gfx_hal::{CompletionToken, QueueKind};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Generational handle identifying a texture registry slot.
pub struct TextureId {
    /// Index of the registry slot and reserved descriptor binding.
    slot: u32,
    /// Revision used to reject stale texture handles.
    generation: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TextureState {
    Allocated,
    Uploading {
        binding: u32,
        resident_mips: u32,
        pending: VecDeque<(CompletionToken, u32)>,
    },
    Resident {
        binding: u32,
        resident_mips: u32,
    },
}

#[derive(Clone, Debug)]
struct Slot {
    generation: u32,
    state: Option<TextureState>,
}

/// Tracks texture allocation, upload completion, residency, and descriptor bindings.
pub struct TextureRegistry {
    /// Maximum number of texture slots.
    capacity: u32,
    /// Maximum number of descriptor bindings addressable by texture slots.
    binding_capacity: u32,
    /// Generational storage for texture residency states.
    slots: Vec<Slot>,
    /// Reusable indices of vacant texture slots.
    free: Vec<u32>,
    /// Coarse-prefix mip counts gating initial readiness per texture.
    required: HashMap<TextureId, u32>,
}
impl TextureRegistry {
    /// Creates an empty registry with nonzero slot and binding limits.
    ///
    /// # Errors
    ///
    /// Returns an error if either capacity is zero.
    pub fn new(capacity: u32, binding_capacity: u32) -> Result<Self, TextureError> {
        if capacity == 0 || binding_capacity == 0 {
            return Err(TextureError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            binding_capacity,
            slots: Vec::new(),
            free: Vec::new(),
            required: HashMap::new(),
        })
    }

    /// Allocates or reuses a texture slot for a pending upload.
    ///
    /// # Errors
    ///
    /// Returns an error if no texture slot remains available.
    ///
    /// # Panics
    ///
    /// The registry capacity bounds every texture slot conversion.
    pub fn begin_upload(&mut self) -> Result<TextureId, TextureError> {
        if let Some(slot) = self.free.pop() {
            let entry = &mut self.slots[slot as usize];
            entry.state = Some(TextureState::Allocated);
            return Ok(TextureId {
                slot,
                generation: entry.generation,
            });
        }
        if self.slots.len() >= self.capacity as usize {
            return Err(TextureError::CapacityExceeded);
        }
        let slot = u32::try_from(self.slots.len()).map_err(|_| TextureError::CapacityExceeded)?;
        self.slots.push(Slot {
            generation: 1,
            state: Some(TextureState::Allocated),
        });
        Ok(TextureId {
            slot,
            generation: 1,
        })
    }

    /// Associates an allocated texture with its initial upload completion token.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid, its binding exceeds the binding capacity, or the texture is not allocated.
    pub fn mark_submitted(
        &mut self,
        texture: TextureId,
        completion: CompletionToken,
    ) -> Result<(), TextureError> {
        let binding = texture.slot;
        if binding >= self.binding_capacity {
            return Err(TextureError::CapacityExceeded);
        }
        let state = self.state_mut(texture)?;
        if *state != TextureState::Allocated {
            return Err(TextureError::InvalidState);
        }
        *state = TextureState::Uploading {
            binding,
            resident_mips: 0,
            pending: VecDeque::from([(completion, 1)]),
        };
        Ok(())
    }

    /// Applies completed uploads for one queue and advances residency.
    ///
    /// # Errors
    ///
    /// This implementation advances in-memory state only and does not fail;
    /// the error case is reserved for fallible polling.
    pub fn poll(&mut self, queue: QueueKind, completed: u64) -> Result<usize, TextureError> {
        let mut changed = 0;
        for entry in &mut self.slots {
            let Some(TextureState::Uploading {
                binding,
                resident_mips,
                pending,
            }) = entry.state.as_mut()
            else {
                continue;
            };
            while let Some((token, target_mips)) = pending.front().copied() {
                if token.queue != queue || token.value > completed {
                    break;
                }
                pending.pop_front();
                *resident_mips = target_mips;
                changed += 1;
            }
            if pending.is_empty() {
                entry.state = Some(TextureState::Resident {
                    binding: *binding,
                    resident_mips: *resident_mips,
                });
            }
        }
        Ok(changed)
    }

    /// Returns the descriptor binding once at least one mip is resident.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or the texture has no resident mip levels.
    pub fn binding_index(&self, texture: TextureId) -> Result<u32, TextureError> {
        match self.state(texture)? {
            TextureState::Uploading {
                binding,
                resident_mips,
                ..
            } if *resident_mips > 0 => Ok(*binding),
            TextureState::Resident { binding, .. } => Ok(*binding),
            _ => Err(TextureError::NotReady),
        }
    }

    /// Returns the number of mip levels currently resident.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or the texture has no resident mip levels.
    pub fn resident_mips(&self, texture: TextureId) -> Result<u32, TextureError> {
        match self.state(texture)? {
            TextureState::Uploading { resident_mips, .. } if *resident_mips > 0 => {
                Ok(*resident_mips)
            }
            TextureState::Resident { resident_mips, .. } => Ok(*resident_mips),
            _ => Err(TextureError::NotReady),
        }
    }

    /// Records the coarse-prefix mip count gating initial readiness.
    ///
    /// Zero records optional residency: frame submission never waits for the
    /// texture, and its binding samples fallback until real residency
    /// publishes. Positive values gate exactly like before.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is stale. Requirements exceeding the
    /// decoded chain fail at submission, not here, because the chain length
    /// is unknown until decode completes.
    pub fn set_required_mips(
        &mut self,
        texture: TextureId,
        required: u32,
    ) -> Result<(), TextureError> {
        // Stale handles must not plant requirements on recycled slots.
        self.state(texture)?;
        self.required.insert(texture, required);
        Ok(())
    }

    /// Returns the coarse-prefix mip count gating initial readiness.
    ///
    /// Textures submitted without a recorded requirement default to the
    /// single coarse mip; a stored zero means optional residency and is
    /// returned verbatim.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle does not identify a current texture.
    pub fn required_mips(&self, texture: TextureId) -> Result<u32, TextureError> {
        self.state(texture)?;
        Ok(self.required.get(&texture).copied().unwrap_or(1))
    }

    /// Higher mip counts queue behind prior transfers; duplicate or decreasing targets are rejected.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid, the requested mip count is zero or not increasing, or the texture is in an incompatible state.
    pub fn mark_mips_submitted(
        &mut self,
        texture: TextureId,
        resident_mips: u32,
        completion: CompletionToken,
    ) -> Result<(), TextureError> {
        if resident_mips == 0 {
            return Err(TextureError::InvalidData);
        }
        let state = self.state_mut(texture)?;
        match state {
            TextureState::Uploading {
                resident_mips: current,
                pending,
                ..
            } => {
                let highest = pending.back().map_or(*current, |(_, target)| *target);
                if resident_mips <= highest {
                    return Err(TextureError::InvalidState);
                }
                pending.push_back((completion, resident_mips));
                Ok(())
            }
            TextureState::Resident {
                binding,
                resident_mips: current,
            } if resident_mips > *current => {
                *state = TextureState::Uploading {
                    binding: *binding,
                    resident_mips: *current,
                    pending: VecDeque::from([(completion, resident_mips)]),
                };
                Ok(())
            }
            _ => Err(TextureError::InvalidState),
        }
    }

    /// Rolls back an unexposed allocation/upload without notifications.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot does not exist, the handle or state is invalid, or the generation counter is exhausted.
    pub fn cancel_upload(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation
            || !matches!(
                entry.state,
                Some(TextureState::Allocated | TextureState::Uploading { .. })
            )
        {
            return Err(TextureError::InvalidState);
        }
        // Exhaustion fails before clearing state so callers never receive an error after destruction.
        let next_generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        entry.state = None;
        entry.generation = next_generation;
        self.required.remove(&texture);
        self.free.push(texture.slot);
        Ok(())
    }

    /// Invalidates a submitted texture while withholding its descriptor slot from reuse.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is stale or its generation cannot advance.
    pub fn retire(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation || entry.state.is_none() {
            return Err(TextureError::NotFound);
        }
        let next_generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        entry.state = None;
        entry.generation = next_generation;
        self.required.remove(&texture);
        Ok(())
    }

    /// Releases a retired descriptor slot after native transfer and frame dependencies complete.
    ///
    /// # Errors
    ///
    /// Returns an error unless `texture` is the immediately preceding generation of a withheld slot.
    pub fn release_retired(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.state.is_some()
            || entry.generation != texture.generation.saturating_add(1)
            || self.free.contains(&texture.slot)
        {
            return Err(TextureError::InvalidState);
        }
        self.free.push(texture.slot);
        Ok(())
    }

    /// Releases a texture slot, invalidates its handle, and emits an unload notification.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle does not identify a current texture or the generation counter is exhausted.
    pub fn unload(&mut self, texture: TextureId) -> Result<(), TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation || entry.state.is_none() {
            return Err(TextureError::NotFound);
        }
        // Exhaustion fails before clearing state or omitting the corresponding unload event.
        let next_generation = entry
            .generation
            .checked_add(1)
            .ok_or(TextureError::GenerationExhausted)?;
        entry.state = None;
        entry.generation = next_generation;
        self.required.remove(&texture);
        self.free.push(texture.slot);
        Ok(())
    }

    /// Invalidates every texture slot.
    ///
    /// # Errors
    ///
    /// Returns an error without mutation if an occupied slot cannot advance its generation.
    pub fn clear(&mut self) -> Result<(), TextureError> {
        if self
            .slots
            .iter()
            .any(|entry| entry.state.is_some() && entry.generation == u32::MAX)
        {
            return Err(TextureError::GenerationExhausted);
        }

        self.free.clear();
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            if entry.state.take().is_some() {
                entry.generation += 1;
            }
            if entry.generation != u32::MAX {
                self.free
                    .push(u32::try_from(slot).map_err(|_| TextureError::CapacityExceeded)?);
            }
        }
        self.required.clear();
        Ok(())
    }

    /// Returns the descriptor binding reserved by an existing texture handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or its reserved binding exceeds the binding capacity.
    pub fn reserved_binding(&self, texture: TextureId) -> Result<u32, TextureError> {
        let _ = self.state(texture)?;
        if texture.slot >= self.binding_capacity {
            return Err(TextureError::CapacityExceeded);
        }
        Ok(texture.slot)
    }

    /// Resolves a current texture handle to its residency state.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle does not identify a current texture.
    fn state(&self, texture: TextureId) -> Result<&TextureState, TextureError> {
        let entry = self
            .slots
            .get(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation {
            return Err(TextureError::NotFound);
        }
        entry.state.as_ref().ok_or(TextureError::NotFound)
    }
    /// Resolves a current texture handle to mutable residency state.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle does not identify a current texture.
    fn state_mut(&mut self, texture: TextureId) -> Result<&mut TextureState, TextureError> {
        let entry = self
            .slots
            .get_mut(texture.slot as usize)
            .ok_or(TextureError::NotFound)?;
        if entry.generation != texture.generation {
            return Err(TextureError::NotFound);
        }
        entry.state.as_mut().ok_or(TextureError::NotFound)
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn generation_exhaustion_does_not_partially_cancel_or_unload() {
        for cancel in [true, false] {
            let mut registry = TextureRegistry::new(1, 1).unwrap();
            registry.slots.push(Slot {
                generation: u32::MAX,
                state: Some(TextureState::Allocated),
            });
            let texture = TextureId {
                slot: 0,
                generation: u32::MAX,
            };

            let result = if cancel {
                registry.cancel_upload(texture)
            } else {
                registry.unload(texture)
            };

            assert_eq!(result, Err(TextureError::GenerationExhausted));
            assert!(registry.state(texture).is_ok());
        }
    }

    #[test]
    fn required_mips_defaults_and_clears_with_lifecycle() {
        let mut registry = TextureRegistry::new(2, 2).unwrap();
        let first = registry.begin_upload().unwrap();
        // Untracked textures gate on the single coarse mip.
        assert_eq!(registry.required_mips(first), Ok(1));
        // Zero records optional residency instead of failing.
        registry.set_required_mips(first, 0).unwrap();
        assert_eq!(registry.required_mips(first), Ok(0));
        registry.set_required_mips(first, 2).unwrap();
        assert_eq!(registry.required_mips(first), Ok(2));
        // Stale handles cannot plant requirements on recycled slots.
        let stale = TextureId {
            slot: first.slot,
            generation: first.generation.wrapping_add(1),
        };
        assert_eq!(
            registry.set_required_mips(stale, 1),
            Err(TextureError::NotFound)
        );
        registry.cancel_upload(first).unwrap();
        assert_eq!(registry.required_mips(first), Err(TextureError::NotFound));
        // A recycled slot starts untracked again.
        let second = registry.begin_upload().unwrap();
        assert_eq!(registry.required_mips(second), Ok(1));
    }
}
