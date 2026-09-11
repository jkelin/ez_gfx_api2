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

#[cfg(any(test, target_vendor = "apple"))]
const fn mesh_pipeline_capabilities_supported(
    capabilities: ez_gfx_core::capability::ShaderCapabilities,
    has_task: bool,
) -> bool {
    capabilities.mesh && (!has_task || capabilities.task)
}
#[cfg(any(test, target_vendor = "apple"))]
const METAL3_MAX_MESH_THREADGROUPS_PER_GRID: u32 = 1_024;
#[cfg(any(test, target_vendor = "apple"))]
const APPLE9_MAX_MESH_THREADGROUPS_PER_GRID: u32 = 1_048_575;

#[cfg(any(test, target_vendor = "apple"))]
const fn metal_mesh_group_limit(supports_apple9: bool) -> u32 {
    // Apple7 is the documented floor for every Metal 3 device; Apple9 raises the limit.
    if supports_apple9 {
        APPLE9_MAX_MESH_THREADGROUPS_PER_GRID
    } else {
        METAL3_MAX_MESH_THREADGROUPS_PER_GRID
    }
}

#[cfg(any(test, target_vendor = "apple"))]
const fn metal_mesh_dispatch_limits(
    max_groups: u32,
    max_mesh_threads: u32,
    max_task_threads: u32,
) -> ez_gfx_hal::MeshDispatchLimits {
    // A nonzero 3D grid cannot exceed the documented total without an axis exceeding it.
    ez_gfx_hal::MeshDispatchLimits {
        max_groups: [max_groups; 3],
        max_total_groups: max_groups as u64,
        max_mesh_threads,
        max_task_threads,
    }
}

#[cfg(any(test, target_vendor = "apple"))]
const METAL_LIBRARY_ERROR_UNSUPPORTED: isize = 1;
#[cfg(any(test, target_vendor = "apple"))]
const METAL_LIBRARY_ERROR_COMPILE_FAILURE: isize = 3;

#[cfg(any(test, target_vendor = "apple"))]
fn contains_ascii_case_insensitive_identifier(value: &str, identifier: &str) -> bool {
    let bytes = value.as_bytes();
    let identifier = identifier.as_bytes();

    // Empty identifiers cannot name a Metal property and would make `windows` panic.
    if identifier.is_empty() {
        return false;
    }

    bytes
        .windows(identifier.len())
        .enumerate()
        .any(|(start, candidate)| {
            let end = start + identifier.len();
            // Reject longer diagnostic tokens that merely contain the property name.
            let has_start_boundary =
                start == 0 || !bytes[start - 1].is_ascii_alphanumeric() && bytes[start - 1] != b'_';
            let has_end_boundary =
                end == bytes.len() || !bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_';

            has_start_boundary && has_end_boundary && candidate.eq_ignore_ascii_case(identifier)
        })
}

#[cfg(any(test, target_vendor = "apple"))]
fn is_mesh_thread_limit_pipeline_error(
    is_library_error: bool,
    code: isize,
    description: &str,
) -> bool {
    if !is_library_error
        || !matches!(
            code,
            METAL_LIBRARY_ERROR_UNSUPPORTED | METAL_LIBRARY_ERROR_COMPILE_FAILURE
        )
    {
        return false;
    }

    // Metal exposes no pipeline-specific preflight query. NSError identifies
    // the rejected limit only through these stable descriptor property names.
    contains_ascii_case_insensitive_identifier(description, "maxTotalThreadsPerMeshThreadgroup")
        || contains_ascii_case_insensitive_identifier(
            description,
            "maxTotalThreadsPerObjectThreadgroup",
        )
}

#[cfg(test)]
mod mesh_pipeline_tests {
    use super::{
        is_mesh_thread_limit_pipeline_error, mesh_pipeline_capabilities_supported,
        metal_mesh_dispatch_limits, metal_mesh_group_limit,
    };
    use ez_gfx_core::capability::ShaderCapabilities;
    use ez_gfx_hal::{MeshDispatchError, validate_mesh_dispatch};

    #[test]
    fn capability_gate_rejects_before_native_mesh_allocation() {
        assert!(!mesh_pipeline_capabilities_supported(
            ShaderCapabilities::default(),
            false,
        ));
        assert!(!mesh_pipeline_capabilities_supported(
            ShaderCapabilities {
                task: false,
                mesh: true,
            },
            true,
        ));
        assert!(mesh_pipeline_capabilities_supported(
            ShaderCapabilities {
                task: false,
                mesh: true,
            },
            false,
        ));
    }
    #[test]
    fn mesh_grid_limits_follow_documented_family_floors() {
        assert_eq!(metal_mesh_group_limit(false), 1_024);
        assert_eq!(metal_mesh_group_limit(true), 1_048_575);
    }

    #[test]
    fn mesh_grid_limits_accept_boundaries_and_reject_excess() {
        let limits = metal_mesh_dispatch_limits(metal_mesh_group_limit(false), 64, 32);

        for groups in [[1_024, 1, 1], [32, 32, 1]] {
            assert_eq!(
                validate_mesh_dispatch(groups, [64, 1, 1], Some([32, 1, 1]), limits),
                Ok(())
            );
        }
        for groups in [[1_025, 1, 1], [1_024, 2, 1], [u32::MAX, 1, 1]] {
            assert_eq!(
                validate_mesh_dispatch(groups, [64, 1, 1], Some([32, 1, 1]), limits),
                Err(MeshDispatchError::InvalidGroups)
            );
        }
        assert_eq!(
            validate_mesh_dispatch([1, 1, 1], [65, 1, 1], Some([32, 1, 1]), limits),
            Err(MeshDispatchError::UnsupportedWorkgroup)
        );
    }

    #[test]
    fn mesh_thread_limit_pipeline_error_requires_library_limit_diagnostic() {
        for (code, description) in [
            (
                3,
                "Fehler: MAXTOTALTHREADSPERMESHTHREADGROUP (1024) ist nicht verfügbar",
            ),
            (
                3,
                "エラー: MaxTotalThreadsPerObjectThreadgroup の値は 256 です",
            ),
            (1, "[maxTotalThreadsPerMeshThreadgroup]"),
        ] {
            assert!(is_mesh_thread_limit_pipeline_error(true, code, description,));
        }

        for (is_library_error, code, description) in [
            (
                false,
                3,
                "maxTotalThreadsPerMeshThreadgroup exceeds the maximum",
            ),
            (
                true,
                2,
                "maxTotalThreadsPerMeshThreadgroup exceeds the maximum",
            ),
            (true, 3, "fragment function failed to compile"),
            (true, 3, "maxTotalThreadsPerThreadgroup is unsupported"),
            (
                true,
                3,
                "notmaxTotalThreadsPerMeshThreadgroupSuffix is unsupported",
            ),
        ] {
            assert!(!is_mesh_thread_limit_pipeline_error(
                is_library_error,
                code,
                description,
            ));
        }
    }
}

#[cfg(target_vendor = "apple")]
/// Apple-only Metal implementation.
pub mod native;
