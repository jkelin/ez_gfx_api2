//! Generic automation support shared by rendering examples and their tests.
#![allow(
    dead_code,
    reason = "Standalone examples and smoke tests use different subsets; host and asset values cross fixed OS and file-format widths."
)]
pub mod data;
pub mod host;
pub mod input;
pub mod lifecycle;
pub mod math;
pub mod mesh;
pub mod observability;

#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use data::{byte_len, bytes_of, slice_bytes};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use host::{
    BackendConfig, HostSurface, NativePlatform, NativeSurface, backend_config, backend_name, clip_y,
};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use input::{FrameInput, SceneInput, SceneKey, dispatch_window_input};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use lifecycle::{LifecycleCallbacks, LifecycleConfig, run};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use observability::{ObservationCounts, drain_bounded};

use anyhow::Context as _;
use std::{ffi::OsString, path::Path, process::Command, time::Instant};
/// Captured terminal frame and neutral observability totals.
#[derive(Debug)]
pub struct PresentedFrame {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub rgba8: Vec<u8>,
    pub runtime_events: u32,
    pub diagnostics: u32,
    pub dropped_observations: u64,
}

/// Frame ranges used by benchmark automation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkConfig {
    /// Frames rendered before timing starts.
    pub warmup_frames: u32,
    /// Frames included in the timing interval.
    pub measured_frames: u32,
}

/// Timing result for a completed benchmark run.
#[derive(Debug, Clone, Copy)]
pub struct BenchmarkReport {
    /// Frames rendered before timing started.
    pub warmup_frames: u32,
    /// Frames included in the timing interval.
    pub measured_frames: u32,
    /// Total measured duration in nanoseconds.
    pub elapsed_ns: u128,
}
/// Benchmark clock shared by every example callback.
pub struct BenchmarkRunner {
    config: Option<BenchmarkConfig>,
    started: Option<Instant>,
    report: Option<BenchmarkReport>,
}

impl BenchmarkRunner {
    pub const fn new(config: Option<BenchmarkConfig>) -> Self {
        Self {
            config,
            started: None,
            report: None,
        }
    }

    pub fn begin_frame(&mut self, frame_index: u32) {
        if self
            .config
            .is_some_and(|config| frame_index == config.warmup_frames)
        {
            self.started = Some(Instant::now());
        }
    }

    pub fn end_frame(&mut self, rendered_frames: u32) {
        if let Some(config) = self.config
            && rendered_frames == config.warmup_frames + config.measured_frames
        {
            self.report = Some(BenchmarkReport {
                warmup_frames: config.warmup_frames,
                measured_frames: config.measured_frames,
                elapsed_ns: self
                    .started
                    .expect("benchmark timer starts before measured frames")
                    .elapsed()
                    .as_nanos(),
            });
        }
    }

    pub const fn report(&self) -> Option<BenchmarkReport> {
        self.report
    }
}

/// Captured frame plus optional benchmark timing.
pub struct ProgramReport {
    pub frame: PresentedFrame,
    pub benchmark: Option<BenchmarkReport>,
}

/// Runs one configured example and emits stable snapshot/benchmark reports.
pub fn run_program(
    identity: &str,
    backend: &str,
    run: impl FnOnce(Option<u32>, Option<BenchmarkConfig>) -> anyhow::Result<Option<ProgramReport>>,
) {
    let requested_limit = max_frames_from_env().unwrap_or_else(|error| exit_config(error));
    let benchmark = benchmark_from_env().unwrap_or_else(|error| exit_config(error));
    let frame_limit = benchmark.map_or(requested_limit, |config| {
        Some(benchmark_frame_limit(config).unwrap_or_else(|error| exit_config(error)))
    });
    let report = run(frame_limit, benchmark).unwrap_or_else(|error| {
        eprintln!("example failed: {error}");
        std::process::exit(1)
    });
    let Some(report) = report else { return };
    let frame = report.frame;
    publish_snapshot(
        std::env::var_os("EZ_GFX_EXAMPLE_SNAPSHOT"),
        std::env::var("EZ_GFX_UPDATE_SNAPSHOTS").ok().as_deref() == Some("1"),
        frame.width,
        frame.height,
        frame.frames,
        &frame.rgba8,
        frame.runtime_events,
        frame.diagnostics,
        frame.dropped_observations,
    );
    if let Some(benchmark) = report.benchmark {
        let frame_time_ns = benchmark.elapsed_ns as f64 / f64::from(benchmark.measured_frames);
        let fps = 1_000_000_000.0 / frame_time_ns;
        println!(
            "{{\"benchmark\":\"{identity}\",\"backend\":\"{backend}\",\"warmup_frames\":{},\"measured_frames\":{},\"elapsed_ns\":{},\"frame_time_ns\":{frame_time_ns:.3},\"fps\":{fps:.3}}}",
            benchmark.warmup_frames, benchmark.measured_frames, benchmark.elapsed_ns,
        );
    }
}

fn exit_config(error: anyhow::Error) -> ! {
    eprintln!("{error}");
    std::process::exit(2)
}

/// Reads an optional environment flag that accepts only `0` or `1`.
pub fn env_flag(name: &str) -> anyhow::Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse_env_flag(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_env_flag(name, None),
        Err(error) => Err(error).with_context(|| name.to_owned()),
    }
}

fn parse_env_flag(name: &str, value: Option<&str>) -> anyhow::Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err(anyhow::anyhow!("{name} must be 0 or 1")),
    }
}

/// Reads the optional positive frame cap; malformed and zero values are errors.
pub fn max_frames_from_env() -> anyhow::Result<Option<u32>> {
    match std::env::var("EZ_GFX_EXAMPLE_MAX_FRAMES") {
        Ok(value) => {
            let frames = value
                .parse::<u32>()
                .context("EZ_GFX_EXAMPLE_MAX_FRAMES must be a positive integer")?;
            if frames == 0 {
                anyhow::bail!("EZ_GFX_EXAMPLE_MAX_FRAMES must be positive");
            }
            Ok(Some(frames))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).context("EZ_GFX_EXAMPLE_MAX_FRAMES"),
    }
}

/// Parses benchmark settings while preserving the historical defaults and opt-in flag.
pub fn benchmark_from_env() -> anyhow::Result<Option<BenchmarkConfig>> {
    if std::env::var("EZ_GFX_EXAMPLE_BENCHMARK").ok().as_deref() != Some("1") {
        return Ok(None);
    }
    Ok(Some(BenchmarkConfig {
        warmup_frames: positive_env("EZ_GFX_EXAMPLE_BENCHMARK_WARMUP", 120)?,
        measured_frames: positive_env("EZ_GFX_EXAMPLE_BENCHMARK_FRAMES", 600)?,
    }))
}

/// Includes one uncaptured terminal frame so the snapshot cache is populated outside timing.
pub fn benchmark_frame_limit(config: BenchmarkConfig) -> anyhow::Result<u32> {
    config
        .warmup_frames
        .checked_add(config.measured_frames)
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("benchmark frame counts exceed u32 limit"))
}

/// Applies a captured RGBA frame to the optional snapshot path and report stream.
pub fn publish_snapshot(
    path: Option<OsString>,
    update: bool,
    width: u32,
    height: u32,
    frames: u32,
    rgba8: &[u8],
    runtime_events: u32,
    diagnostics: u32,
    dropped_observations: u64,
) {
    if let Some(path) = path {
        if update {
            image::save_buffer_with_format(
                &path,
                rgba8,
                width,
                height,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            )
            .unwrap_or_else(|error| panic!("update snapshot: {error}"));
        } else {
            let expected = image::open(&path)
                .unwrap_or_else(|error| panic!("open snapshot: {error}"))
                .into_rgba8();
            assert_eq!(expected.dimensions(), (width, height));
            assert_eq!(expected.as_raw(), rgba8);
        }
    }
    if std::env::var_os("EZ_GFX_EXAMPLE_REPORT").is_some() {
        println!(
            "ez-gfx-snapshot {width} {height} {frames} {} {runtime_events} {diagnostics} {dropped_observations}",
            blake3::hash(rgba8)
        );
    }
}

/// Builds the smoke-test command with the stable automation environment.
pub fn snapshot_command(binary: &str, path: &Path, backend: &str) -> Command {
    let mut command = Command::new(binary);
    command
        .env("EZ_GFX_BACKEND", backend)
        .env("EZ_GFX_EXAMPLE_MAX_FRAMES", "1")
        .env("EZ_GFX_EXAMPLE_REPORT", "1")
        .env("EZ_GFX_EXAMPLE_SNAPSHOT", path)
        .env("VK_LOADER_LAYERS_DISABLE", "~implicit~");
    command
}

fn positive_env(name: &str, default: u32) -> anyhow::Result<u32> {
    // Empty means default; zero and malformed input must never silently measure no frames.
    let value = std::env::var(name).unwrap_or_default();
    if value.is_empty() {
        return Ok(default);
    }
    let parsed = value
        .parse::<u32>()
        .with_context(|| format!("{name} must be a positive integer"))?;
    (parsed > 0)
        .then_some(parsed)
        .ok_or_else(|| anyhow::anyhow!("{name} must be positive"))
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkConfig, benchmark_frame_limit, parse_env_flag};
    #[test]
    fn benchmark_limit_includes_capture_frame_and_checks_overflow() {
        assert_eq!(
            benchmark_frame_limit(BenchmarkConfig {
                warmup_frames: 2,
                measured_frames: 3
            })
            .unwrap(),
            6
        );
        assert!(
            benchmark_frame_limit(BenchmarkConfig {
                warmup_frames: u32::MAX,
                measured_frames: 1
            })
            .is_err()
        );
    }

    #[test]
    fn env_flags_accept_only_absent_zero_or_one() {
        for (value, expected) in [(None, false), (Some("0"), false), (Some("1"), true)] {
            assert_eq!(parse_env_flag("FLAG", value).unwrap(), expected);
        }
        assert_eq!(
            parse_env_flag("FLAG", Some("true"))
                .unwrap_err()
                .to_string(),
            "FLAG must be 0 or 1"
        );
    }
}
