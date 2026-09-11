use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[test]
fn frame_waits_collapse_to_one_maximum_per_transfer_queue() {
    let actions = [
        NativeFrameAction::Wait(CompletionToken {
            queue: QueueKind::Transfer,
            value: 1,
        }),
        NativeFrameAction::Wait(CompletionToken {
            queue: QueueKind::TextureTransfer,
            value: 2,
        }),
        NativeFrameAction::Wait(CompletionToken {
            queue: QueueKind::Transfer,
            value: 3,
        }),
    ];
    let plan = validate_frame_plan(&actions, (1, 1), false, false).unwrap();
    assert_eq!(plan.external_waits.len(), 2);
    // First-seen queue order is preserved while each queue keeps its maximum.
    assert_eq!(plan.external_waits[0].queue, QueueKind::Transfer);
    assert_eq!(plan.external_waits[0].value, 3);
    assert_eq!(plan.external_waits[1].queue, QueueKind::TextureTransfer);
    assert_eq!(plan.external_waits[1].value, 2);
}

#[test]
fn graphics_wait_fails_validation() {
    let actions = [NativeFrameAction::Wait(CompletionToken {
        queue: QueueKind::Graphics,
        value: 1,
    })];
    assert!(matches!(
        validate_frame_plan(&actions, (1, 1), false, false),
        Err(HalError::InvalidArgument)
    ));
}

static ALLOCATION_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// SAFETY: every operation delegates unchanged pointers and layouts to
// `System`; the atomics only observe sizes and never affect ownership.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the unchanged request is delegated to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if ALLOCATION_ENABLED.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from the system allocator above.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn spill_cardinality_plan_validation_performs_no_allocations() {
    // Sixty-five actions exceed every former inline action-scratch threshold.
    let actions: [NativeFrameAction<'_>; 65] = std::array::from_fn(|index| {
        NativeFrameAction::Wait(CompletionToken {
            queue: if index % 2 == 0 {
                QueueKind::Transfer
            } else {
                QueueKind::TextureTransfer
            },
            value: index as u64 + 1,
        })
    });
    for _ in 0..4 {
        validate_frame_plan(&actions, (1, 1), false, false).unwrap();
    }
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATION_BYTES.store(0, Ordering::Relaxed);
    ALLOCATION_ENABLED.store(true, Ordering::Relaxed);
    for _ in 0..500 {
        validate_frame_plan(&actions, (1, 1), false, false).unwrap();
    }
    ALLOCATION_ENABLED.store(false, Ordering::Relaxed);
    assert_eq!(ALLOCATION_CALLS.load(Ordering::Relaxed), 0);
    assert_eq!(ALLOCATION_BYTES.load(Ordering::Relaxed), 0);
}
