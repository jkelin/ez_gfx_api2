#![deny(unsafe_op_in_unsafe_fn)]

//! Metal backend for ez-gfx native resource creation and frame execution.
//!
//! The native implementation is compiled only for Apple targets.

use ez_gfx_core::{Backend, capability::MAX_BINDLESS_SAMPLED_TEXTURES};

/// Identifies this adapter as the Metal backend.
pub const BACKEND: Backend = Backend::Metal;
/// Reports whether the compilation target is an Apple platform that supports Metal object creation.
pub const SUPPORTED_ON_TARGET: bool = cfg!(target_vendor = "apple");
/// Maximum number of textures admitted by the tier-two argument-buffer heap.
pub const TEXTURE_DESCRIPTOR_CAPACITY: u32 = MAX_BINDLESS_SAMPLED_TEXTURES;

#[cfg(any(test, target_vendor = "apple"))]
mod frame_slots {
    pub(super) const FRAMES_IN_FLIGHT: usize = 3;

    #[derive(Debug, Default)]
    pub(super) struct FrameSlotTracker {
        cursor: usize,
        occupied: [bool; FRAMES_IN_FLIGHT],
    }

    impl FrameSlotTracker {
        /// A wrapped slot must be completed before reuse; an unused slot needs no wait.
        pub(super) fn acquire(&mut self) -> (usize, bool) {
            let slot = self.cursor;
            self.cursor = (self.cursor + 1) % FRAMES_IN_FLIGHT;
            (slot, self.occupied[slot])
        }

        pub(super) fn mark_submitted(&mut self, slot: usize) {
            self.occupied[slot] = true;
        }

        pub(super) fn mark_completed(&mut self, slot: usize) {
            self.occupied[slot] = false;
        }

        pub(super) fn in_flight_mask(&self) -> u8 {
            self.occupied
                .iter()
                .enumerate()
                .fold(0, |mask, (slot, occupied)| {
                    mask | (u8::from(*occupied) << slot)
                })
        }
    }

    pub(super) fn complete_deferred_slot(mask: &mut u8, slot: usize) -> bool {
        *mask &= !(1 << slot);
        *mask == 0
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn waits_only_after_three_outstanding_submissions() {
            let mut tracker = FrameSlotTracker::default();

            for expected in 0..FRAMES_IN_FLIGHT {
                assert_eq!(tracker.acquire(), (expected, false));
                tracker.mark_submitted(expected);
            }

            assert_eq!(tracker.acquire(), (0, true));
            tracker.mark_completed(0);
            assert_eq!(tracker.in_flight_mask(), 0b110);
        }

        #[test]
        fn deferred_resource_completes_after_every_referencing_slot() {
            let mut mask = 0b101;

            assert!(!complete_deferred_slot(&mut mask, 0));
            assert_eq!(mask, 0b100);
            assert!(complete_deferred_slot(&mut mask, 2));
        }
    }
}

#[cfg(target_vendor = "apple")]
/// Apple-only Metal implementation.
pub mod native;
