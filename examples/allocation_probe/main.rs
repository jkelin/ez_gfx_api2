//! Development-only native allocation regression workload.
#[path = "../shared/mod.rs"]
mod shared;

use ez_gfx::*;
use glam::Mat4;
use shared::{mesh::*, *};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const WARM_FRAMES: usize = 4;
const MEASURED_FRAMES: usize = 500;

struct Counter;
static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

fn record_peak(live: usize) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
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

// SAFETY: every operation delegates unchanged pointers and layouts to `System`; atomics only
// observe allocation sizes and never alter allocator ownership.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the unchanged request is delegated to the process allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let enabled = ENABLED.load(Ordering::Relaxed);
            if enabled {
                CALLS.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
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
            if ENABLED.load(Ordering::Relaxed) {
                CALLS.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(new_size, Ordering::Relaxed);
            }
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

#[derive(Clone, Copy, Default)]
struct AllocationStats {
    calls: usize,
    bytes: usize,
    peak_live: usize,
    live_delta: i128,
}

impl AllocationStats {
    fn add(&mut self, frame: Self) {
        self.calls = self.calls.saturating_add(frame.calls);
        self.bytes = self.bytes.saturating_add(frame.bytes);
        self.peak_live = self.peak_live.max(frame.peak_live);
        self.live_delta = self.live_delta.saturating_add(frame.live_delta);
    }

    fn assert_zero(self, workload: &str) -> anyhow::Result<()> {
        if self.calls == 0 && self.bytes == 0 && self.peak_live == 0 && self.live_delta == 0 {
            Ok(())
        } else {
            anyhow::bail!(
                "{workload} allocated after warm-up: calls={} bytes={} peak-live={} live-delta={}",
                self.calls,
                self.bytes,
                self.peak_live,
                self.live_delta
            )
        }
    }
}
/// Fixed-workload residual ceilings over 500 measured frames.
struct ResidualBaseline {
    calls: usize,
    bytes: usize,
}

/// Returns the residual ceiling for one fixed backend workload.
///
/// Residual traffic lives only in execute/finish phases and is classified as
/// excluded shader metadata/pipeline-key ownership, owned by the reserved
/// shader redesign. Unknown backends have no measured baseline, so probing
/// them fails instead of silently passing.
fn residual_baseline(backend: &str, workload: &str) -> anyhow::Result<ResidualBaseline> {
    match (backend, workload) {
        ("Vulkan", "triangle") => Ok(ResidualBaseline {
            calls: 7_300,
            bytes: 600_000,
        }),
        ("Vulkan", "multipass") => Ok(ResidualBaseline {
            calls: 32_000,
            bytes: 5_000_000,
        }),
        ("DX12", "triangle") => Ok(ResidualBaseline {
            calls: 7_300,
            bytes: 585_000,
        }),
        ("DX12", "multipass") => Ok(ResidualBaseline {
            calls: 34_500,
            bytes: 5_200_000,
        }),
        _ => anyhow::bail!("no allocation baseline for backend={backend} workload={workload}"),
    }
}

/// Fails any whole-window increase above the fixed-workload residual ceiling.
///
/// Peak-live and live-delta stay informational: they include allocator and
/// event-loop noise, while calls and requested bytes are deterministic.
fn check_residual_within_baseline(
    backend: &str,
    workload: &str,
    stats: AllocationStats,
) -> anyhow::Result<()> {
    let baseline = residual_baseline(backend, workload)?;
    if stats.calls > baseline.calls || stats.bytes > baseline.bytes {
        anyhow::bail!(
            "{workload} on {backend} exceeded its residual baseline: observed calls={} bytes={} vs baseline calls={} bytes={}; only the reserved shader redesign may rebaseline",
            stats.calls,
            stats.bytes,
            baseline.calls,
            baseline.bytes
        )
    } else {
        println!(
            "{workload}-residual: within baseline calls={} bytes={} (baseline calls={} bytes={})",
            stats.calls, stats.bytes, baseline.calls, baseline.bytes
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ExcludedBaseline {
    calls: usize,
    bytes: usize,
}

/// Returns the single-frame ceiling for an excluded execute/finish phase.
///
/// Excluded phases carry reserved shader metadata/pipeline-key ownership, so
/// they are bounded rather than zero. Triangle execute showed one 8-call/271-byte
/// sample against a usual 7/167 on Vulkan, and DX12 observed 541 B post-ABI40
/// rebase against a usual 125 B, so the DX12 triangle-execute ceiling is 768 B
/// while the other ceilings carry headroom; multipass finish varies run to run
/// (Vulkan 12-13 calls and 3.9-7.4 KiB, DX12 17-20 calls and 4.4-7.6 KiB), and
/// the whole-window baseline below still catches any systematic per-frame
/// regression. `None` means the phase is in-scope and must allocate nothing.
fn excluded_baseline(backend: Backend, workload: &str, phase: &str) -> Option<ExcludedBaseline> {
    let (calls, bytes) = match (backend, workload, phase) {
        (Backend::Vulkan, "triangle", "execute") => (10, 512),
        (Backend::Vulkan, "triangle", "finish") => (8, 1_024),
        (Backend::Vulkan, "multipass", "compute") => (20, 4_096),
        (Backend::Vulkan, "multipass", "graphics") => (32, 2_048),
        (Backend::Vulkan, "multipass", "finish") => (20, 8_192),
        (Backend::Dx12, "triangle", "execute") => (10, 768),
        (Backend::Dx12, "triangle", "finish") => (8, 1_024),
        (Backend::Dx12, "multipass", "compute") => (20, 4_096),
        (Backend::Dx12, "multipass", "graphics") => (32, 2_048),
        (Backend::Dx12, "multipass", "finish") => (28, 8_192),
        _ => return None,
    };
    Some(ExcludedBaseline { calls, bytes })
}

fn validate_phase(
    backend: Backend,
    workload: &str,
    phase: &str,
    stats: AllocationStats,
    strict_all: bool,
) -> anyhow::Result<()> {
    report_phase(workload, phase, stats);
    if strict_all {
        return stats.assert_zero(&format!("{workload}-{phase}"));
    }
    if let Some(baseline) = excluded_baseline(backend, workload, phase) {
        println!(
            "excluded-{workload}-{phase}: shader-metadata/pipeline-key baseline calls<={} bytes<={}",
            baseline.calls, baseline.bytes
        );
        if stats.calls <= baseline.calls && stats.bytes <= baseline.bytes {
            return Ok(());
        }
        anyhow::bail!(
            "{workload}-{phase} exceeded excluded shader-metadata/pipeline-key baseline: calls={}/{} bytes={}/{}; only the reserved shader redesign may rebaseline",
            stats.calls,
            baseline.calls,
            stats.bytes,
            baseline.bytes
        );
    }
    stats.assert_zero(&format!("{workload}-{phase}"))
}

struct MeasurementGuard;

impl Drop for MeasurementGuard {
    fn drop(&mut self) {
        ENABLED.store(false, Ordering::SeqCst);
    }
}

fn measure_value<T>(
    run: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<(T, AllocationStats)> {
    let baseline = LIVE_BYTES.load(Ordering::SeqCst);
    CALLS.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    PEAK_LIVE_BYTES.store(baseline, Ordering::SeqCst);
    ENABLED.store(true, Ordering::SeqCst);
    let guard = MeasurementGuard;
    let result = run();
    drop(guard);
    let final_live = LIVE_BYTES.load(Ordering::SeqCst);
    let stats = AllocationStats {
        calls: CALLS.load(Ordering::SeqCst),
        bytes: BYTES.load(Ordering::SeqCst),
        peak_live: PEAK_LIVE_BYTES
            .load(Ordering::SeqCst)
            .saturating_sub(baseline),
        live_delta: i128::try_from(final_live).expect("usize fits i128")
            - i128::try_from(baseline).expect("usize fits i128"),
    };
    Ok((result?, stats))
}

fn measure_frame(run: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<AllocationStats> {
    measure_value(run).map(|(_, stats)| stats)
}

fn report_phase(workload: &str, phase: &str, stats: AllocationStats) {
    println!(
        "trace-{workload}-{phase}: calls={} bytes={} peak-live={} live-delta={}",
        stats.calls, stats.bytes, stats.peak_live, stats.live_delta
    );
}

struct Triangle {
    first_index: u32,
    _indices: IndexAllocation,
    _positions: VertexAllocation<[f32; 4]>,
    vertex: VertexShader,
    fragment: FragmentShader,
}

impl Triangle {
    fn new(context: &Context, positions_heap: &VertexHeap<[f32; 4]>) -> anyhow::Result<Self> {
        let artifact = ez_gfx_compiler::EasyGraphicsCompiler::compile_shader(
            std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/01_triangle/01_triangle.slang"
            )),
            &[
                ez_gfx_compiler::Target::Spirv,
                ez_gfx_compiler::Target::Dxil,
                ez_gfx_compiler::Target::Metal,
            ],
            !cfg!(target_vendor = "apple"),
        )?;
        let vertical = if cfg!(target_vendor = "apple") {
            -1.0
        } else {
            1.0
        };
        let positions = [
            [-0.5_f32, -0.5 * vertical, 0.0, 1.0],
            [0.5, -0.5 * vertical, 0.0, 1.0],
            [0.0, 0.5 * vertical, 0.0, 1.0],
        ];
        let indices = context.upload_indices(&[0, 1, 2])?;
        let first_index = indices.range()?.0;
        let positions = positions_heap.upload(&positions)?;
        Ok(Self {
            first_index,
            _indices: indices,
            _positions: positions,
            vertex: artifact.load_vertex_shader(context, "vertexmain")?,
            fragment: artifact.load_fragment_shader(context, "fragmentmain")?,
        })
    }

    fn render(
        &self,
        context: &Context,
        surface: &Surface,
        example: &mut Example,
        size: [u32; 2],
    ) -> anyhow::Result<()> {
        let mut frame = surface.begin_frame()?;
        let target = frame.configure_swapchain(
            size,
            Format::Bgra8Srgb,
            ez_gfx::PresentationMode::Immediate,
        )?;
        let commands = [DrawIndexedCommand {
            index_count: 3,
            instance_count: 1,
            first_index: self.first_index,
            vertex_offset: 0,
            first_instance: 0,
        }];
        let indirect = context.acquire_counter_buffer_from(commands.as_slice())?;
        frame.execute_graphics(
            &self.vertex,
            &self.fragment,
            &indirect,
            DynamicPipelineState::from_abi(0, 0, 0, 0).expect("static render state"),
        )?;
        example.handle_allocation_frame(frame, target)?;
        Ok(())
    }

    fn trace(
        &self,
        context: &Context,
        surface: &Surface,
        example: &mut Example,
        size: [u32; 2],
    ) -> anyhow::Result<()> {
        let strict_all = example.strict_all();
        let (mut frame, begin) = measure_value(|| Ok::<_, anyhow::Error>(surface.begin_frame()?))?;
        let (target, configure) = measure_value(|| {
            Ok::<_, anyhow::Error>(frame.configure_swapchain(
                size,
                Format::Bgra8Srgb,
                ez_gfx::PresentationMode::Immediate,
            )?)
        })?;
        let commands = [DrawIndexedCommand {
            index_count: 3,
            instance_count: 1,
            first_index: self.first_index,
            vertex_offset: 0,
            first_instance: 0,
        }];
        let (indirect, acquire) = measure_value(|| {
            Ok::<_, anyhow::Error>(context.acquire_counter_buffer_from(commands.as_slice())?)
        })?;
        let (_, execute) = measure_value(|| {
            frame.execute_graphics(
                &self.vertex,
                &self.fragment,
                &indirect,
                DynamicPipelineState::from_abi(0, 0, 0, 0).expect("static render state"),
            )?;
            Ok::<_, anyhow::Error>(())
        })?;
        let (_, finish) = measure_value(|| {
            example.handle_allocation_frame(frame, target)?;
            Ok::<_, anyhow::Error>(())
        })?;
        for (phase, stats) in [
            ("begin", begin),
            ("configure", configure),
            ("acquire", acquire),
            ("execute", execute),
            ("finish", finish),
        ] {
            validate_phase(example.backend(), "triangle", phase, stats, strict_all)?;
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneParams {
    mvp: Mat4,
    primitive_count: u32,
    padding: [u32; 3],
}

struct Multipass {
    records: Vec<BasicPrimitive>,
    params: SceneParams,
    primitive_count: u32,
    _indices: IndexAllocation,
    _positions: VertexAllocation<[f32; 4]>,
    _normals: VertexAllocation<[f32; 4]>,
    compute: ComputeShader,
    vertex: VertexShader,
    fragment: FragmentShader,
}

impl Multipass {
    fn new(context: &Context, positions_heap: &VertexHeap<[f32; 4]>) -> anyhow::Result<Self> {
        let artifact = ez_gfx_compiler::EasyGraphicsCompiler::compile_shader(
            std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/03_compute_structured/03_compute_structured.slang"
            )),
            &[
                ez_gfx_compiler::Target::Spirv,
                ez_gfx_compiler::Target::Dxil,
                ez_gfx_compiler::Target::Metal,
            ],
            !cfg!(target_vendor = "apple"),
        )?;
        let mesh = load_geometry_glb(include_bytes!("../shared/assets/sponza.glb"))?;
        let primitive_count = u32::try_from(mesh.primitives.len())?;
        let indices = context.upload_indices(&mesh.indices)?;
        let first_index = indices.range()?.0;
        let records = basic_primitives(&mesh, first_index)?;
        let positions = positions_heap.upload(&mesh.positions)?;
        let normals_heap = context.create_vertex_heap("normals")?;
        let normals = normals_heap.upload(&mesh.normals)?;
        Ok(Self {
            records,
            params: SceneParams {
                mvp: Mat4::IDENTITY,
                primitive_count,
                padding: [0; 3],
            },
            primitive_count,
            _indices: indices,
            _positions: positions,
            _normals: normals,
            compute: artifact.load_compute_shader(context, "computemain")?,
            vertex: artifact.load_vertex_shader(context, "vertexmain")?,
            fragment: artifact.load_fragment_shader(context, "fragmentmain")?,
        })
    }

    fn render(
        &self,
        context: &Context,
        surface: &Surface,
        example: &mut Example,
        size: [u32; 2],
    ) -> anyhow::Result<()> {
        let primitives = context.acquire_buffer_from(&self.records)?;
        let indirect = context
            .acquire_counter_buffer::<DrawIndexedCommand>(usize::try_from(self.primitive_count)?)?;
        let params = context.acquire_value_buffer(self.params)?;
        let mut frame = surface.begin_frame()?;
        let target = frame.configure_swapchain(
            size,
            Format::Bgra8Srgb,
            ez_gfx::PresentationMode::Immediate,
        )?;
        frame.bind_buffer("params", &params)?;
        frame.bind_buffer("primitives", &primitives)?;
        frame.bind_buffer("draw_commands", &indirect)?;
        frame.execute_compute(&self.compute, [self.primitive_count, 1, 1])?;
        frame.execute_graphics(
            &self.vertex,
            &self.fragment,
            &indirect,
            DynamicPipelineState::from_abi(0, 0, 0, 0).expect("static render state"),
        )?;
        example.handle_allocation_frame(frame, target)?;
        Ok(())
    }

    fn trace(
        &self,
        context: &Context,
        surface: &Surface,
        example: &mut Example,
        size: [u32; 2],
    ) -> anyhow::Result<()> {
        let strict_all = example.strict_all();
        let ((primitives, indirect, params), acquire) = measure_value(|| {
            Ok::<_, anyhow::Error>((
                context.acquire_buffer_from(&self.records)?,
                context.acquire_counter_buffer::<DrawIndexedCommand>(usize::try_from(
                    self.primitive_count,
                )?)?,
                context.acquire_value_buffer(self.params)?,
            ))
        })?;
        let (mut frame, begin) = measure_value(|| Ok::<_, anyhow::Error>(surface.begin_frame()?))?;
        let (target, configure) = measure_value(|| {
            Ok::<_, anyhow::Error>(frame.configure_swapchain(
                size,
                Format::Bgra8Srgb,
                ez_gfx::PresentationMode::Immediate,
            )?)
        })?;
        let (_, bind) = measure_value(|| {
            frame.bind_buffer("params", &params)?;
            frame.bind_buffer("primitives", &primitives)?;
            frame.bind_buffer("draw_commands", &indirect)?;
            Ok::<_, anyhow::Error>(())
        })?;
        let (_, compute) = measure_value(|| {
            frame.execute_compute(&self.compute, [self.primitive_count, 1, 1])?;
            Ok::<_, anyhow::Error>(())
        })?;
        let (_, graphics) = measure_value(|| {
            frame.execute_graphics(
                &self.vertex,
                &self.fragment,
                &indirect,
                DynamicPipelineState::from_abi(0, 0, 0, 0).expect("static render state"),
            )?;
            Ok::<_, anyhow::Error>(())
        })?;
        let (_, finish) = measure_value(|| {
            example.handle_allocation_frame(frame, target)?;
            Ok::<_, anyhow::Error>(())
        })?;
        for (phase, stats) in [
            ("acquire", acquire),
            ("begin", begin),
            ("configure", configure),
            ("bind", bind),
            ("compute", compute),
            ("graphics", graphics),
            ("finish", finish),
        ] {
            validate_phase(example.backend(), "multipass", phase, stats, strict_all)?;
        }
        Ok(())
    }
}

fn next_size(
    example: &mut Example,
    context: &Context,
    surface: &Surface,
) -> anyhow::Result<[u32; 2]> {
    example
        .wait_for_next_frame(context, surface)?
        .map(|frame| frame.size)
        .ok_or_else(|| anyhow::anyhow!("native host stopped before allocation workload completed"))
}

fn run_workload(
    name: &str,
    example: &mut Example,
    context: &Context,
    surface: &Surface,
    mut render: impl FnMut(&mut Example, [u32; 2]) -> anyhow::Result<()>,
) -> anyhow::Result<AllocationStats> {
    for _ in 0..WARM_FRAMES {
        let size = next_size(example, context, surface)?;
        render(example, size)?;
    }
    let mut total = AllocationStats::default();
    for _ in 0..MEASURED_FRAMES {
        let size = next_size(example, context, surface)?;
        total.add(measure_frame(|| render(example, size))?);
    }
    println!(
        "{name}: frames={MEASURED_FRAMES} calls={} bytes={} peak-live={} live-delta={}",
        total.calls, total.bytes, total.peak_live, total.live_delta
    );
    Ok(total)
}

fn main() -> anyhow::Result<()> {
    let mut example = Example::new("allocation_probe", WIDTH, HEIGHT, "ez_gfx allocation probe")?;
    if !example.is_hidden() {
        anyhow::bail!("allocation probe must run with --hidden");
    }
    let backend = backend_config(example.backend());
    let context = Context::new(ContextOptions {
        enable_debug: example.debug_enabled(),
        enable_validation: example.validation_enabled(),
        backend: backend.backend,
        texture_decode_workers: 0,
        adapter_selection: None,
    })?;
    let surface = context.create_surface_window(example.native_surface()?, false)?;
    let positions_heap = context.create_vertex_heap("positions")?;
    let triangle = Triangle::new(&context, &positions_heap)?;
    let multipass = Multipass::new(&context, &positions_heap)?;

    let triangle_stats = run_workload(
        "triangle",
        &mut example,
        &context,
        &surface,
        |example, size| triangle.render(&context, &surface, example, size),
    )?;
    let trace_size = next_size(&mut example, &context, &surface)?;
    triangle.trace(&context, &surface, &mut example, trace_size)?;
    let multipass_stats = run_workload(
        "multipass",
        &mut example,
        &context,
        &surface,
        |example, size| multipass.render(&context, &surface, example, size),
    )?;
    let trace_size = next_size(&mut example, &context, &surface)?;
    multipass.trace(&context, &surface, &mut example, trace_size)?;

    let memory = context.memory_telemetry()?;
    let resources = context.resource_diagnostics()?;
    println!(
        "telemetry: backend={} allocator-live={} allocator-blocks={} allocator-waste={} frame-slots={} staging-buckets={} staging-bytes={} staging-high-water={} counter-scratch={} decode-workers={} pipeline-entries={} readback-bytes={}",
        example.backend_name(),
        memory.backend.allocator.map_or(0, |value| value.live_bytes),
        memory
            .backend
            .allocator
            .map_or(0, |value| value.block_bytes),
        memory
            .backend
            .allocator
            .map_or(0, |value| value.waste_bytes()),
        memory.backend.frame_slots,
        memory.staging_buckets,
        memory.staging_bytes,
        memory.staging_high_water,
        memory.counter_scratch_bytes,
        memory.decode_workers,
        resources.pipeline_entries,
        resources.readback_bytes,
    );
    // Whole-window totals stay visible in both modes: the per-workload lines
    // above plus the phase traces classify every call. Scoped mode hard-fails
    // in-scope regressions inside the traces and fails residual increases
    // above the fixed baselines here; strict-all instead requires whole-window
    // zero and reports the excluded shader residual as the failure.
    if example.strict_all() {
        triangle_stats.assert_zero("triangle")?;
        multipass_stats.assert_zero("multipass")?;
    } else {
        check_residual_within_baseline(example.backend_name(), "triangle", triangle_stats)?;
        check_residual_within_baseline(example.backend_name(), "multipass", multipass_stats)?;
    }
    Ok(())
}
