//! Smoke tests for the migrated examples.
#[path = "../shared/mod.rs"]
mod shared;

use std::{
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

const BINARIES: [(&str, &str, &str, u32); 6] = [
    (
        "triangle",
        "01_triangle.png",
        env!("CARGO_BIN_EXE_01_triangle"),
        1,
    ),
    (
        "textured cube",
        "02_textured_cube.png",
        env!("CARGO_BIN_EXE_02_textured_cube"),
        1,
    ),
    (
        "compute structured",
        "03_compute_structured.png",
        env!("CARGO_BIN_EXE_03_compute_structured"),
        1,
    ),
    ("imgui", "04_imgui.png", env!("CARGO_BIN_EXE_04_imgui"), 2),
    (
        "helmet",
        "05_helmet.png",
        env!("CARGO_BIN_EXE_05_helmet"),
        1,
    ),
    (
        "sponza ktx2",
        "06_sponza_ktx2.png",
        env!("CARGO_BIN_EXE_06_sponza_ktx2"),
        1,
    ),
];

#[cfg(target_vendor = "apple")]
const TARGET_BACKENDS: &[&str] = &["metal"];
#[cfg(windows)]
const TARGET_BACKENDS: &[&str] = &["vulkan", "dx12"];
#[cfg(all(not(windows), not(target_vendor = "apple")))]
const TARGET_BACKENDS: &[&str] = &["vulkan"];
static CAPTURE_ID: AtomicU64 = AtomicU64::new(0);

const fn uses_reference_snapshot(backend: &str) -> bool {
    // Unknown and secondary backends must never overwrite Vulkan's immutable reference.
    matches!(backend.as_bytes(), b"vulkan")
}

#[test]
fn snapshot_reference_policy_is_vulkan_only() {
    for (backend, expected) in [("vulkan", true), ("dx12", false), ("metal", false)] {
        assert_eq!(uses_reference_snapshot(backend), expected, "{backend}");
    }
}

#[test]
fn snapshot_report_requires_exact_field_count() {
    for (report, valid) in [
        ("0 1 2 3 4 5 6", false),
        ("0 1 2 3 4 5 6 7", true),
        ("0 1 2 3 4 5 6 7 8", false),
    ] {
        assert_eq!(report_fields("test", report).is_ok(), valid);
    }
}

fn report_fields<'a>(label: &str, report: &'a str) -> anyhow::Result<[&'a str; 8]> {
    report
        .split_whitespace()
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| anyhow::anyhow!("{label}: malformed report {report:?}"))
}

fn snapshot(binary: &str, file: &str, backend: &str) -> anyhow::Result<(String, image::RgbaImage)> {
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("snapshots")
        .join(file);
    // Only Vulkan owns immutable references; every other backend captures independently.
    let temporary = !uses_reference_snapshot(backend);
    let path = if temporary {
        std::env::temp_dir().join(format!(
            "ez-gfx-smoke-{}-{}-{backend}-{file}",
            std::process::id(),
            CAPTURE_ID.fetch_add(1, Ordering::Relaxed)
        ))
    } else {
        reference
    };
    let mut command = shared::snapshot_command(binary, &path, backend);
    if temporary {
        command.env("EZ_GFX_UPDATE_SNAPSHOTS", "1");
    }
    #[cfg(target_vendor = "apple")]
    command.env("MTL_DEBUG_LAYER", "1");
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{binary}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = String::from_utf8(output.stdout)?;
    let fields = report_fields(binary, &report)?;
    assert_eq!(
        (fields[1], fields[2], fields[3]),
        ("640", "480", "1"),
        "{binary}"
    );
    assert!(
        fields[5].parse::<u32>().unwrap() > 0,
        "{binary} emitted no runtime events"
    );
    assert_eq!(fields[7], "0", "{binary} dropped observations");
    let image = image::open(&path)?.into_rgba8();
    assert_eq!(
        image.dimensions(),
        (640, 480),
        "{binary} snapshot dimensions"
    );
    assert_eq!(
        fields[4],
        blake3::hash(image.as_raw()).to_string(),
        "{binary} report/image hash mismatch"
    );
    if temporary {
        let _ = std::fs::remove_file(path);
    }
    Ok((report, image))
}

#[derive(Debug)]
struct HiddenReport {
    frames: u32,
    hash: String,
    runtime_events: u32,
}

fn hidden_report(
    binary: &str,
    backend: &str,
    frames: u32,
    deadline: Instant,
) -> anyhow::Result<HiddenReport> {
    // Removing both snapshot variables keeps this path asynchronous even when the parent process
    // is running golden-update automation.
    let mut child = shared::snapshot_command(binary, std::path::Path::new("unused"), backend)
        .env_remove("EZ_GFX_EXAMPLE_SNAPSHOT")
        .env_remove("EZ_GFX_UPDATE_SNAPSHOTS")
        .env("EZ_GFX_EXAMPLE_MAX_FRAMES", frames.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    while child.try_wait()?.is_none() {
        if Instant::now() >= deadline {
            child.kill()?;
            let output = child.wait_with_output()?;
            anyhow::bail!(
                "{backend} Sponza streaming timed out at {frames} frames: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output()?;
    anyhow::ensure!(
        output.status.success(),
        "{backend} Sponza streaming failed at {frames} frames: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = String::from_utf8(output.stdout)?;
    let fields = report_fields(backend, &report)?;
    anyhow::ensure!(
        fields[0] == "ez-gfx-snapshot",
        "{backend}: malformed report {report:?}"
    );
    anyhow::ensure!(
        (fields[1].parse::<u32>()?, fields[2].parse::<u32>()?) == (640, 480),
        "{backend}: unexpected dimensions in {report:?}"
    );
    anyhow::ensure!(
        fields[3].parse::<u32>()? == frames,
        "{backend}: unexpected frame count in {report:?}"
    );
    let runtime_events = fields[5].parse::<u32>()?;
    anyhow::ensure!(
        runtime_events > 0,
        "{backend}: no runtime events in {report:?}"
    );
    anyhow::ensure!(
        fields[7].parse::<u64>()? == 0,
        "{backend}: dropped observations in {report:?}"
    );
    Ok(HiddenReport {
        frames,
        hash: fields[4].to_owned(),
        runtime_events,
    })
}

fn assert_semantics(name: &str, report: &str, image: &image::RgbaImage, min_events: u32) {
    let fields = report_fields(name, report).expect("snapshot already validated its report");
    assert!(
        fields[5].parse::<u32>().unwrap() >= min_events,
        "{name} missing expected observable structure"
    );
    let pixels = image.pixels().map(|pixel| pixel.0);
    let non_background = pixels
        .clone()
        .filter(|pixel| pixel[..3] != [0, 0, 0])
        .count();
    assert!(
        non_background > 100,
        "{name} rendered a blank/near-blank image"
    );
    let colors = pixels
        .map(|pixel| (pixel[0], pixel[1], pixel[2]))
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(colors > 1, "{name} rendered a single corrupt color");
    if name == "triangle" {
        assert!(
            image
                .pixels()
                .any(|p| p[0] > 200 && p[1] < 100 && p[2] < 100),
            "triangle lacks red geometry"
        );
    }
    if name == "imgui" {
        assert!(
            fields[5].parse::<u32>().unwrap() >= 2,
            "ImGui emitted no UI texture/draw structure"
        );
    }
}

macro_rules! independent_scene_test {
    ($test:ident, $index:expr) => {
        #[test]
        fn $test() -> anyhow::Result<()> {
            let (name, file, binary, min_events) = BINARIES[$index];
            for backend in TARGET_BACKENDS {
                let (report, image) = snapshot(binary, file, backend)?;
                assert_semantics(name, &report, &image, min_events);
            }
            Ok(())
        }
    };
}

independent_scene_test!(triangle_smoke, 0);
independent_scene_test!(textured_cube_smoke, 1);
independent_scene_test!(compute_structured_smoke, 2);
independent_scene_test!(imgui_smoke, 3);
independent_scene_test!(helmet_smoke, 4);
independent_scene_test!(sponza_ktx2_smoke, 5);

#[test]
fn sponza_ktx2_streams_to_a_stable_terminal_hash() -> anyhow::Result<()> {
    const FRAME_LIMITS: [u32; 6] = [1, 4, 16, 64, 256, 512];
    let (_, file, binary, _) = BINARIES[5];
    let deadline = Instant::now() + Duration::from_secs(55);
    for backend in TARGET_BACKENDS {
        let (_, waited_image) = snapshot(binary, file, backend)?;
        let target_hash = blake3::hash(waited_image.as_raw()).to_string();
        let mut converged = false;
        let mut observed = Vec::with_capacity(FRAME_LIMITS.len());

        for frames in FRAME_LIMITS {
            let current = hidden_report(binary, backend, frames, deadline)?;
            converged = current.hash == target_hash;
            observed.push((current.frames, current.hash, current.runtime_events));
            if converged {
                break;
            }
        }

        anyhow::ensure!(
            converged,
            "{backend} Sponza streaming did not reach target {target_hash} before the 55s/512-frame bound: {observed:?}"
        );
    }
    Ok(())
}

#[test]
fn triangle_snapshot_is_deterministic_across_processes() -> anyhow::Result<()> {
    let (_, file, binary, _) = BINARIES[0];
    for backend in TARGET_BACKENDS {
        assert_eq!(
            snapshot(binary, file, backend)?.0,
            snapshot(binary, file, backend)?.0,
            "{backend}"
        );
    }
    Ok(())
}

#[test]
fn hidden_windows_complete_multiple_frames_without_redraw_events() -> anyhow::Result<()> {
    for backend in TARGET_BACKENDS {
        // No snapshot file is needed; the terminal capture still reports its completed frame count.
        let mut child =
            shared::snapshot_command(BINARIES[0].2, std::path::Path::new("unused"), backend)
                .env_remove("EZ_GFX_EXAMPLE_SNAPSHOT")
                .env("EZ_GFX_EXAMPLE_MAX_FRAMES", "3")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while child.try_wait()?.is_none() {
            if std::time::Instant::now() >= deadline {
                child.kill()?;
                let output = child.wait_with_output()?;
                anyhow::bail!(
                    "{backend} hidden frames stalled: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let output = child.wait_with_output()?;
        anyhow::ensure!(
            output.status.success(),
            "{backend}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report = String::from_utf8(output.stdout)?;
        assert_eq!(
            report.split_whitespace().nth(3),
            Some("3"),
            "{backend}: {report}"
        );
    }
    Ok(())
}

#[test]
fn triangle_ten_frame_timings_reject_system_timer_pacing() -> anyhow::Result<()> {
    const FRAMES: usize = 10;
    const TIMER_CAP_MIN_NS: u128 = 12_000_000;
    const TIMER_CAP_MAX_NS: u128 = 20_000_000;

    for backend in TARGET_BACKENDS {
        let mut child = std::process::Command::new(BINARIES[0].2)
            .args([
                "--hidden",
                "--backend",
                backend,
                "--max-frames",
                "10",
                "--frame-timings",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while child.try_wait()?.is_none() {
            if std::time::Instant::now() >= deadline {
                child.kill()?;
                let output = child.wait_with_output()?;
                anyhow::bail!(
                    "{backend} ten-frame timing run stalled: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let output = child.wait_with_output()?;
        anyhow::ensure!(
            output.status.success(),
            "{backend}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout)?;
        let timings = stdout
            .lines()
            .filter(|line| line.starts_with("ez-gfx-frame-timing "))
            .map(|line| {
                let fields = line.split_whitespace().collect::<Vec<_>>();
                anyhow::ensure!(fields.len() == 7, "{backend}: {line}");
                anyhow::ensure!(fields[1] == "01_triangle", "{backend}: {line}");
                Ok((
                    fields[3].parse::<usize>()?,
                    fields[4].parse::<u128>()?,
                    fields[5].parse::<u128>()?,
                    fields[6].parse::<u128>()?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        assert_eq!(timings.len(), FRAMES, "{backend}: {stdout}");
        for (index, &(frame, host_wait_ns, record_ns, submit_present_ns)) in
            timings.iter().enumerate()
        {
            assert_eq!(frame, index + 1, "{backend}");
            assert!(host_wait_ns > 0 && record_ns > 0 && submit_present_ns > 0);
        }

        // Ignore one startup frame and the terminal readback. A 15.625 ms wait bug clusters
        // nearly every steady frame in this band; allowing two outliers keeps loaded CI stable.
        let timer_capped_frames = timings[1..FRAMES - 1]
            .iter()
            .filter(|timing| (TIMER_CAP_MIN_NS..=TIMER_CAP_MAX_NS).contains(&timing.3))
            .count();
        assert!(
            timer_capped_frames <= 2,
            "{backend}: {timer_capped_frames} steady frames hit the system-timer pacing band: {timings:?}"
        );
    }
    Ok(())
}

#[cfg(windows)]
#[test]
fn threaded_triangle_matches_benchmark_timing_and_terminal_report_contract() -> anyhow::Result<()> {
    for backend in TARGET_BACKENDS {
        let expected_backend =
            shared::backend_config(shared::host::parse_backend(Some(backend))?).name;
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_01_triangle_second_thread"))
            .args([
                "--hidden",
                "--backend",
                backend,
                "--report",
                "--benchmark",
                "--benchmark-warmup",
                "2",
                "--benchmark-frames",
                "4",
                "--frame-timings",
            ])
            .output()?;
        anyhow::ensure!(
            output.status.success(),
            "{backend}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout)?;
        let lines = stdout.lines().collect::<Vec<_>>();
        let snapshot = lines
            .iter()
            .find(|line| line.starts_with("ez-gfx-snapshot "))
            .ok_or_else(|| anyhow::anyhow!("{backend}: missing terminal report: {stdout}"))?;
        let snapshot_fields = snapshot.split_whitespace().collect::<Vec<_>>();
        assert_eq!(snapshot_fields.len(), 8, "{backend}: {snapshot}");
        assert_eq!(snapshot_fields[3], "7", "{backend}: {snapshot}");
        assert_ne!(
            snapshot_fields[4],
            blake3::hash(&[]).to_string(),
            "{backend}: {snapshot}"
        );
        let benchmark_line = lines
            .iter()
            .find(|line| line.starts_with("{\"benchmark\":"))
            .ok_or_else(|| anyhow::anyhow!("{backend}: missing benchmark report: {stdout}"))?;
        let benchmark: serde_json::Value = serde_json::from_str(benchmark_line)?;
        assert_eq!(benchmark["benchmark"], "01_triangle_second_thread");
        assert_eq!(benchmark["backend"], expected_backend);
        assert_eq!(benchmark["warmup_frames"], 2);
        assert_eq!(benchmark["measured_frames"], 4);
        assert!(
            benchmark["elapsed_ns"]
                .as_u64()
                .is_some_and(|value| value > 0),
            "{backend}: {benchmark_line}"
        );
        for field in ["frame_time_ns", "fps"] {
            assert!(
                benchmark[field].as_f64().is_some_and(|value| value > 0.0),
                "{backend} {field}: {benchmark_line}"
            );
        }
        let timings = lines
            .iter()
            .filter(|line| line.starts_with("ez-gfx-frame-timing "))
            .collect::<Vec<_>>();
        assert_eq!(timings.len(), 7, "{backend}: {stdout}");
        assert!(timings.iter().all(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            fields.len() == 7 && fields[1] == "01_triangle_second_thread" && fields[4] == "0"
        }));
    }
    Ok(())
}

#[cfg(windows)]
#[test]
fn dx12_hidden_window_survives_large_swapchain_resize() -> anyhow::Result<()> {
    let mut child = shared::snapshot_command(BINARIES[0].2, std::path::Path::new("unused"), "dx12")
        .env_remove("EZ_GFX_EXAMPLE_SNAPSHOT")
        .env("EZ_GFX_EXAMPLE_MAX_FRAMES", "3")
        .env("EZ_GFX_EXAMPLE_RESIZE_AFTER_FIRST_FRAME", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while child.try_wait()?.is_none() {
        if std::time::Instant::now() >= deadline {
            child.kill()?;
            let output = child.wait_with_output()?;
            anyhow::bail!(
                "DX12 hidden resize stalled: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child.wait_with_output()?;
    anyhow::ensure!(
        output.status.success(),
        "DX12 hidden resize failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout)?.split_whitespace().nth(3),
        Some("3")
    );
    Ok(())
}
