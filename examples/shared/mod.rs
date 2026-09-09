//! Generic automation support shared by rendering examples and their tests.
#![allow(
    dead_code,
    reason = "Standalone examples and smoke tests use different subsets; host and asset values cross fixed OS and file-format widths."
)]
pub mod data;
mod error;
mod example;
pub mod host;
pub mod input;
pub mod math;
pub mod mesh;
pub mod observability;

#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use data::{byte_len, bytes_of, slice_bytes};
pub use error::{Error, Result};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use example::{Example, WindowFrame};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use host::{BackendConfig, HostSurface, NativePlatform, NativeSurface, backend_config, clip_y};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use input::{FrameInput, SceneInput, SceneKey, dispatch_window_input};
#[allow(
    unused_imports,
    reason = "Standalone examples use different shared interfaces."
)]
pub use observability::{ObservationCounts, drain_bounded};

use clap::Parser;
use ez_gfx::Backend;
use std::{ffi::OsString, num::NonZeroU32, path::Path, process::Command, time::Instant};
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

#[derive(Debug, Parser)]
#[command(disable_help_subcommand = true)]
struct Cli {
    #[arg(long)]
    backend: Option<String>,
    #[arg(long)]
    max_frames: Option<NonZeroU32>,
    #[arg(long)]
    hidden: bool,
    #[arg(long)]
    report: bool,
    #[arg(long)]
    snapshot: Option<OsString>,
    #[arg(long)]
    update_snapshots: bool,
    #[arg(long)]
    debug: bool,
    #[arg(long)]
    validation: bool,
    #[arg(long)]
    benchmark: bool,
    #[arg(long)]
    benchmark_warmup: Option<NonZeroU32>,
    #[arg(long)]
    benchmark_frames: Option<NonZeroU32>,
}

#[derive(Debug)]
pub(crate) struct ProgramOptions {
    pub(crate) backend: Backend,
    pub(crate) frame_limit: Option<u32>,
    pub(crate) benchmark: Option<BenchmarkConfig>,
    pub(crate) visible: bool,
    pub(crate) report: bool,
    pub(crate) snapshot: Option<OsString>,
    pub(crate) update_snapshots: bool,
    pub(crate) debug: bool,
    pub(crate) validation: bool,
}

pub(crate) fn exit_config(error: impl std::fmt::Display) -> ! {
    eprintln!("{error}");
    std::process::exit(2)
}
pub(crate) fn program_options() -> Result<ProgramOptions> {
    let cli = Cli::try_parse().map_err(Error::from)?;
    program_options_from(cli, |name| std::env::var_os(name))
}

fn program_options_from(
    cli: Cli,
    env: impl Fn(&str) -> Option<OsString>,
) -> Result<ProgramOptions> {
    fn reject_conflict(
        present: bool,
        env_name: &str,
        env: &impl Fn(&str) -> Option<OsString>,
    ) -> Result<()> {
        if present && env(env_name).is_some() {
            return Err(Error::message(format!(
                "CLI option conflicts with {env_name}"
            )));
        }
        Ok(())
    }
    fn env_text(name: &str, env: &impl Fn(&str) -> Option<OsString>) -> Result<Option<String>> {
        env(name)
            .map(|value| {
                value
                    .into_string()
                    .map_err(|_| Error::message(format!("{name} must be valid Unicode")))
            })
            .transpose()
    }

    reject_conflict(cli.backend.is_some(), "EZ_GFX_BACKEND", &env)?;
    reject_conflict(cli.max_frames.is_some(), "EZ_GFX_EXAMPLE_MAX_FRAMES", &env)?;
    reject_conflict(cli.hidden, "EZ_GFX_EXAMPLE_HIDDEN", &env)?;
    reject_conflict(cli.report, "EZ_GFX_EXAMPLE_REPORT", &env)?;
    reject_conflict(cli.snapshot.is_some(), "EZ_GFX_EXAMPLE_SNAPSHOT", &env)?;
    reject_conflict(cli.update_snapshots, "EZ_GFX_UPDATE_SNAPSHOTS", &env)?;
    reject_conflict(cli.debug, "EZ_GFX_EXAMPLE_DEBUG", &env)?;
    reject_conflict(cli.validation, "EZ_GFX_EXAMPLE_VALIDATION", &env)?;
    reject_conflict(cli.benchmark, "EZ_GFX_EXAMPLE_BENCHMARK", &env)?;
    reject_conflict(
        cli.benchmark_warmup.is_some(),
        "EZ_GFX_EXAMPLE_BENCHMARK_WARMUP",
        &env,
    )?;
    reject_conflict(
        cli.benchmark_frames.is_some(),
        "EZ_GFX_EXAMPLE_BENCHMARK_FRAMES",
        &env,
    )?;

    let backend_text = cli.backend.or(env_text("EZ_GFX_BACKEND", &env)?);
    let backend = host::parse_backend(backend_text.as_deref())?;
    let requested_limit = match cli.max_frames {
        Some(value) => Some(value.get()),
        None => env_text("EZ_GFX_EXAMPLE_MAX_FRAMES", &env)?
            .map(|value| value.parse::<u32>())
            .transpose()?
            .map(|value| {
                NonZeroU32::new(value)
                    .ok_or_else(|| Error::message("EZ_GFX_EXAMPLE_MAX_FRAMES must be positive"))
                    .map(NonZeroU32::get)
            })
            .transpose()?,
    };
    let benchmark_enabled =
        cli.benchmark || env_text("EZ_GFX_EXAMPLE_BENCHMARK", &env)?.as_deref() == Some("1");
    let benchmark = if benchmark_enabled {
        let warmup_frames = cli.benchmark_warmup.map_or_else(
            || {
                positive_env_value(
                    "EZ_GFX_EXAMPLE_BENCHMARK_WARMUP",
                    env_text("EZ_GFX_EXAMPLE_BENCHMARK_WARMUP", &env)?.as_deref(),
                    120,
                )
            },
            |value| Ok(value.get()),
        )?;
        let measured_frames = cli.benchmark_frames.map_or_else(
            || {
                positive_env_value(
                    "EZ_GFX_EXAMPLE_BENCHMARK_FRAMES",
                    env_text("EZ_GFX_EXAMPLE_BENCHMARK_FRAMES", &env)?.as_deref(),
                    600,
                )
            },
            |value| Ok(value.get()),
        )?;
        Some(BenchmarkConfig {
            warmup_frames,
            measured_frames,
        })
    } else {
        None
    };
    let frame_limit = benchmark.map_or(Ok(requested_limit), |config| {
        benchmark_frame_limit(config).map(Some)
    })?;

    let env_flag_value = |name| -> Result<bool> {
        let value = env_text(name, &env)?;
        parse_env_flag(name, value.as_deref())
    };
    Ok(ProgramOptions {
        backend,
        frame_limit,
        benchmark,
        visible: !(cli.hidden || env_flag_value("EZ_GFX_EXAMPLE_HIDDEN")?),
        report: cli.report || env("EZ_GFX_EXAMPLE_REPORT").is_some(),
        snapshot: cli.snapshot.or_else(|| env("EZ_GFX_EXAMPLE_SNAPSHOT")),
        update_snapshots: cli.update_snapshots
            || env_text("EZ_GFX_UPDATE_SNAPSHOTS", &env)?.as_deref() == Some("1"),
        debug: cli.debug || env_flag_value("EZ_GFX_EXAMPLE_DEBUG")?,
        validation: cli.validation || env_flag_value("EZ_GFX_EXAMPLE_VALIDATION")?,
    })
}

fn positive_env_value(name: &str, value: Option<&str>, default: u32) -> Result<u32> {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return Ok(default);
    };
    let parsed = value.parse::<u32>()?;
    NonZeroU32::new(parsed)
        .map(NonZeroU32::get)
        .ok_or_else(|| Error::message(format!("{name} must be positive")))
}

fn parse_env_flag(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err(Error::message(format!("{name} must be 0 or 1"))),
    }
}

/// Includes one uncaptured terminal frame so the snapshot cache is populated outside timing.
pub fn benchmark_frame_limit(config: BenchmarkConfig) -> Result<u32> {
    config
        .warmup_frames
        .checked_add(config.measured_frames)
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| Error::message(format!("benchmark frame counts exceed u32 limit")))
}

#[derive(Debug, PartialEq, Eq)]
struct ByteDifference {
    index: usize,
    expected: Option<u8>,
    actual: Option<u8>,
}

#[derive(Debug, PartialEq, Eq)]
struct SnapshotMismatch {
    expected_len: usize,
    actual_len: usize,
    expected_hash: blake3::Hash,
    actual_hash: blake3::Hash,
    first_difference: ByteDifference,
}

fn snapshot_mismatch(expected: &[u8], actual: &[u8]) -> Option<SnapshotMismatch> {
    let content_difference = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual);
    if content_difference.is_none() && expected.len() == actual.len() {
        return None;
    }

    // Prefix-only length mismatches differ at the first byte present on only one side.
    let index = content_difference.unwrap_or_else(|| expected.len().min(actual.len()));
    Some(SnapshotMismatch {
        expected_len: expected.len(),
        actual_len: actual.len(),
        expected_hash: blake3::hash(expected),
        actual_hash: blake3::hash(actual),
        first_difference: ByteDifference {
            index,
            expected: expected.get(index).copied(),
            actual: actual.get(index).copied(),
        },
    })
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
    report: bool,
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
            assert!(
                expected.dimensions() == (width, height),
                "snapshot dimensions differ: expected_path={} expected={}x{} actual_path=<captured frame> actual={}x{}",
                Path::new(&path).display(),
                expected.width(),
                expected.height(),
                width,
                height
            );
            if let Some(mismatch) = snapshot_mismatch(expected.as_raw(), rgba8) {
                panic!(
                    "snapshot pixels differ: expected_path={} expected_dimensions={}x{} expected_bytes={} expected_blake3={} actual_path=<captured frame> actual_dimensions={}x{} actual_bytes={} actual_blake3={} first_difference_index={} expected_byte={:?} actual_byte={:?}",
                    Path::new(&path).display(),
                    expected.width(),
                    expected.height(),
                    mismatch.expected_len,
                    mismatch.expected_hash,
                    width,
                    height,
                    mismatch.actual_len,
                    mismatch.actual_hash,
                    mismatch.first_difference.index,
                    mismatch.first_difference.expected,
                    mismatch.first_difference.actual
                );
            }
        }
    }
    if report {
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
        .env("EZ_GFX_EXAMPLE_HIDDEN", "1")
        .env("EZ_GFX_EXAMPLE_REPORT", "1")
        .env("EZ_GFX_EXAMPLE_SNAPSHOT", path)
        .env("VK_LOADER_LAYERS_DISABLE", "~implicit~");
    command
}

#[cfg(test)]
mod tests {
    use super::{
        BenchmarkConfig, Cli, benchmark_frame_limit, parse_env_flag, program_options_from,
        snapshot_mismatch,
    };
    use clap::Parser;
    use std::{collections::HashMap, ffi::OsString};
    #[test]
    fn snapshot_comparison_accepts_equal_bytes() {
        assert!(snapshot_mismatch(b"same", b"same").is_none());
    }

    #[test]
    fn snapshot_comparison_reports_first_content_mismatch() {
        let mismatch = snapshot_mismatch(&[1, 2, 3], &[1, 9, 3]).unwrap();

        assert_eq!(mismatch.expected_len, 3);
        assert_eq!(mismatch.actual_len, 3);
        assert_eq!(mismatch.expected_hash, blake3::hash(&[1, 2, 3]));
        assert_eq!(mismatch.actual_hash, blake3::hash(&[1, 9, 3]));
        assert_eq!(mismatch.first_difference.index, 1);
        assert_eq!(mismatch.first_difference.expected, Some(2));
        assert_eq!(mismatch.first_difference.actual, Some(9));
    }

    #[test]
    fn snapshot_comparison_reports_length_mismatch_at_prefix_end() {
        let mismatch = snapshot_mismatch(&[1, 2], &[1]).unwrap();

        assert_eq!(mismatch.expected_len, 2);
        assert_eq!(mismatch.actual_len, 1);
        assert_eq!(mismatch.first_difference.index, 1);
        assert_eq!(mismatch.first_difference.expected, Some(2));
        assert_eq!(mismatch.first_difference.actual, None);
    }

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
    #[test]
    fn cli_parser_builds_automation_and_benchmark_config() {
        let backend = if cfg!(target_vendor = "apple") {
            "metal"
        } else {
            "vulkan"
        };
        let cli = Cli::try_parse_from([
            "example",
            "--backend",
            backend,
            "--max-frames",
            "9",
            "--hidden",
            "--benchmark",
            "--benchmark-warmup",
            "2",
            "--benchmark-frames",
            "4",
        ])
        .unwrap();
        let options = program_options_from(cli, |_| None).unwrap();

        assert!(!options.visible);
        assert_eq!(options.frame_limit, Some(7));
        assert_eq!(
            options.benchmark,
            Some(BenchmarkConfig {
                warmup_frames: 2,
                measured_frames: 4,
            })
        );
    }

    #[test]
    fn smoke_environment_defaults_are_preserved() {
        let backend = if cfg!(target_vendor = "apple") {
            "metal"
        } else {
            "vulkan"
        };
        let env = HashMap::from([
            ("EZ_GFX_BACKEND", OsString::from(backend)),
            ("EZ_GFX_EXAMPLE_MAX_FRAMES", OsString::from("1")),
            ("EZ_GFX_EXAMPLE_HIDDEN", OsString::from("1")),
            ("EZ_GFX_EXAMPLE_REPORT", OsString::from("1")),
            ("EZ_GFX_EXAMPLE_SNAPSHOT", OsString::from("capture.png")),
        ]);
        let cli = Cli::try_parse_from(["example"]).unwrap();
        let options = program_options_from(cli, |name| env.get(name).cloned()).unwrap();

        assert_eq!(options.frame_limit, Some(1));
        assert!(!options.visible);
        assert!(options.report);
        assert_eq!(options.snapshot, Some(OsString::from("capture.png")));
    }

    #[test]
    fn cli_and_environment_conflicts_are_rejected() {
        let cli = Cli::try_parse_from(["example", "--hidden"]).unwrap();
        let error = program_options_from(cli, |name| {
            (name == "EZ_GFX_EXAMPLE_HIDDEN").then(|| OsString::from("0"))
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "CLI option conflicts with EZ_GFX_EXAMPLE_HIDDEN"
        );
    }
}
