//! Development-only, headless allocation regression workload for warmed frame compilation.

use ez_gfx_core::handle::{LocalHandle, PackedHandle, TextureHandle};
use ez_gfx_hal::{QueueKind, ResourceAccess, ResourceState, ShaderStage};
use ez_gfx_runtime::{
    frame::{ExecutableNode, FrameRecorder, FrameWorkspaceStats},
    graph::{
        Access, Format, ImageRange, LoadOp, NodeDesc, PassInfo, ResourceDesc, ResourceLifetime,
        StoreOp,
    },
    indirect::DrawIndexedCommand,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

const WARM_FRAME_SLOTS: usize = 4;
const MEASURED_FRAMES: usize = 500;

struct Counter;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

fn record_peak(live: usize) {
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            peak,
            live,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

// SAFETY: every allocation operation delegates unchanged pointers and layouts to `System`,
// while the atomics observe sizes without affecting allocator ownership.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the unchanged request is delegated to the process allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            record_peak(live);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: the pointer and layout came from the system allocator above.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the pointer and old layout came from the system allocator.
        let replacement = unsafe { System.realloc(pointer, old, new_size) };
        if !replacement.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            let live = if new_size >= old.size() {
                LIVE_BYTES.fetch_add(new_size - old.size(), Ordering::Relaxed) + new_size
                    - old.size()
            } else {
                LIVE_BYTES.fetch_sub(old.size() - new_size, Ordering::Relaxed)
                    - (old.size() - new_size)
            };
            record_peak(live);
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: Counter = Counter;

#[derive(Clone, Copy)]
struct AllocationStats {
    calls: usize,
    bytes: usize,
    peak_live_bytes: usize,
    live_delta_bytes: i128,
}

fn texture() -> TextureHandle {
    TextureHandle::from_packed(
        PackedHandle::child(
            LocalHandle::new(1, 1).expect("valid owner"),
            LocalHandle::new(1, 1).expect("valid resource"),
        )
        .expect("valid packed handle"),
    )
    .expect("valid texture handle")
}

fn graphics_state(access: ResourceAccess) -> ResourceState {
    ResourceState::new(QueueKind::Graphics, ShaderStage::Fragment, access).expect("valid state")
}

fn payload() -> ExecutableNode {
    ExecutableNode::TextureReadback { texture: texture() }
}

fn record_triangle(frame: &mut FrameRecorder) {
    frame.begin().expect("begin triangle");
    frame
        .write_counter(
            0,
            DrawIndexedCommand {
                index_count: 3,
                instance_count: 1,
                ..Default::default()
            },
        )
        .expect("triangle counter");
    let color = frame
        .add_resource(
            ResourceDesc::image(
                640,
                480,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::External,
            )
            .expect("triangle image"),
        )
        .expect("triangle resource");
    frame
        .record_node(
            NodeDesc::new("triangle", QueueKind::Graphics)
                .access(Access::image(
                    color,
                    ImageRange::all(1, 1).expect("triangle range"),
                    graphics_state(ResourceAccess::ColorAttachmentWrite),
                ))
                .pass(
                    PassInfo::single_color(
                        color,
                        [0, 0, 640, 480],
                        1,
                        LoadOp::Clear,
                        StoreOp::Store,
                    )
                    .expect("triangle pass"),
                ),
            payload(),
        )
        .expect("triangle node");
}

fn record_multipass(frame: &mut FrameRecorder) {
    frame.begin().expect("begin multipass");
    let first = frame
        .add_resource(
            ResourceDesc::image(
                640,
                480,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::Transient,
            )
            .expect("first image"),
        )
        .expect("first resource");
    let output = frame
        .add_resource(
            ResourceDesc::image(
                640,
                480,
                1,
                1,
                Format::Rgba8Unorm,
                1,
                ResourceLifetime::External,
            )
            .expect("output image"),
        )
        .expect("output resource");
    let produce = frame
        .record_node(
            NodeDesc::new("produce", QueueKind::Graphics)
                .access(Access::image(
                    first,
                    ImageRange::all(1, 1).expect("produce range"),
                    graphics_state(ResourceAccess::ColorAttachmentWrite),
                ))
                .pass(
                    PassInfo::single_color(
                        first,
                        [0, 0, 640, 480],
                        1,
                        LoadOp::Clear,
                        StoreOp::Store,
                    )
                    .expect("produce pass"),
                ),
            payload(),
        )
        .expect("produce node");
    let consume = frame
        .record_node(
            NodeDesc::new("consume", QueueKind::Graphics)
                .access(Access::image(
                    first,
                    ImageRange::all(1, 1).expect("consume range"),
                    graphics_state(ResourceAccess::SampledRead),
                ))
                .depends_on(produce),
            payload(),
        )
        .expect("consume node");
    frame
        .record_node(
            NodeDesc::new("compose", QueueKind::Graphics)
                .access(Access::image(
                    output,
                    ImageRange::all(1, 1).expect("compose range"),
                    graphics_state(ResourceAccess::ColorAttachmentWrite),
                ))
                .depends_on(consume)
                .pass(
                    PassInfo::single_color(
                        output,
                        [0, 0, 640, 480],
                        1,
                        LoadOp::Clear,
                        StoreOp::Store,
                    )
                    .expect("compose pass"),
                ),
            payload(),
        )
        .expect("compose node");
}

fn run_frame(frame: &mut FrameRecorder, record: fn(&mut FrameRecorder)) {
    record(frame);
    let submission = frame.submit().expect("submit frame");
    frame.finish(submission).expect("recycle frame");
}

fn allocation_stats_since(baseline_live: usize) -> AllocationStats {
    let final_live = LIVE_BYTES.load(Ordering::SeqCst);
    AllocationStats {
        calls: ALLOCATIONS.load(Ordering::SeqCst),
        bytes: ALLOCATED_BYTES.load(Ordering::SeqCst),
        peak_live_bytes: PEAK_LIVE_BYTES
            .load(Ordering::SeqCst)
            .saturating_sub(baseline_live),
        live_delta_bytes: i128::try_from(final_live).expect("usize fits i128")
            - i128::try_from(baseline_live).expect("usize fits i128"),
    }
}

fn begin_measurement() -> usize {
    let baseline_live = LIVE_BYTES.load(Ordering::SeqCst);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    PEAK_LIVE_BYTES.store(baseline_live, Ordering::SeqCst);
    baseline_live
}

fn measure(record: fn(&mut FrameRecorder)) -> (AllocationStats, FrameWorkspaceStats) {
    let mut frame = FrameRecorder::new(1).expect("recorder");
    for _ in 0..WARM_FRAME_SLOTS {
        run_frame(&mut frame, record);
    }

    let baseline_live = begin_measurement();
    for _ in 0..MEASURED_FRAMES {
        run_frame(&mut frame, record);
    }
    (
        allocation_stats_since(baseline_live),
        frame.workspace_stats(),
    )
}

fn measure_alternating() -> (AllocationStats, FrameWorkspaceStats) {
    let mut frame = FrameRecorder::new(1).expect("recorder");
    for _ in 0..WARM_FRAME_SLOTS {
        run_frame(&mut frame, record_triangle);
        run_frame(&mut frame, record_multipass);
    }

    let baseline_live = begin_measurement();
    for index in 0..MEASURED_FRAMES {
        let record = if index % 2 == 0 {
            record_triangle
        } else {
            record_multipass
        };
        run_frame(&mut frame, record);
    }
    (
        allocation_stats_since(baseline_live),
        frame.workspace_stats(),
    )
}

fn report(
    label: &str,
    allocations: AllocationStats,
    workspace: FrameWorkspaceStats,
    expected_templates: u64,
    warmed_frames: u64,
) {
    let cache = workspace.graph_cache;
    println!(
        "{label}: frames={MEASURED_FRAMES} allocations={} bytes={} peak-live={} live-delta={} retained={} workspace-high-water={} cache-hits={} cache-misses={} cache-compiles={} cache-evictions={} cache-entries={}/{} cache-retained={}/{} cache-high-water={} cache-generation={} cache-schema={}",
        allocations.calls,
        allocations.bytes,
        allocations.peak_live_bytes,
        allocations.live_delta_bytes,
        workspace.retained_bytes,
        workspace.high_water_bytes,
        cache.hits,
        cache.misses,
        cache.compiles,
        cache.evictions,
        cache.entries,
        cache.entry_limit,
        cache.retained_bytes,
        cache.byte_limit,
        cache.high_water_bytes,
        cache.generation,
        cache.schema,
    );
    assert_eq!(allocations.calls, 0, "warmed unchanged frames allocated");
    assert_eq!(
        allocations.bytes, 0,
        "warmed unchanged frames allocated bytes"
    );
    assert_eq!(
        allocations.peak_live_bytes, 0,
        "warmed unchanged frames raised live heap"
    );
    assert_eq!(
        allocations.live_delta_bytes, 0,
        "warmed unchanged frames leaked live heap"
    );
    assert!(workspace.retained_bytes <= workspace.byte_limit);
    assert_eq!(cache.misses, expected_templates);
    assert_eq!(cache.compiles, expected_templates);
    assert_eq!(
        cache.hits,
        warmed_frames + MEASURED_FRAMES as u64 - expected_templates,
    );
    assert_eq!(cache.evictions, 0);
    assert_eq!(cache.entries as u64, expected_templates);
    assert!(cache.entries <= cache.entry_limit);
    assert!(cache.retained_bytes <= cache.byte_limit);
    assert!(cache.retained_bytes <= cache.high_water_bytes);
}

fn main() {
    println!(
        "configuration: warm-slots={WARM_FRAME_SLOTS} measured-frames={MEASURED_FRAMES} node-desc-bytes={} pass-info-bytes={}",
        core::mem::size_of::<NodeDesc>(),
        core::mem::size_of::<PassInfo>(),
    );
    let (triangle_allocations, triangle_workspace) = measure(record_triangle);
    let (multipass_allocations, multipass_workspace) = measure(record_multipass);
    let (alternating_allocations, alternating_workspace) = measure_alternating();
    report(
        "triangle",
        triangle_allocations,
        triangle_workspace,
        1,
        WARM_FRAME_SLOTS as u64,
    );
    report(
        "multipass",
        multipass_allocations,
        multipass_workspace,
        1,
        WARM_FRAME_SLOTS as u64,
    );
    report(
        "alternating",
        alternating_allocations,
        alternating_workspace,
        2,
        (WARM_FRAME_SLOTS * 2) as u64,
    );
}
