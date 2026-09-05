//! Smoke tests for the migrated examples.
#[path = "../shared/mod.rs"]
mod shared;
use anyhow::Context as _;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
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

fn snapshot(binary: &str, file: &str, backend: &str) -> anyhow::Result<(String, image::RgbaImage)> {
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("snapshots")
        .join(file);
    // Secondary backends use isolated captures because rasterization permits backend-specific edge pixels.
    let temporary = backend != TARGET_BACKENDS[0];
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
    let output = command
        .output()
        .with_context(|| format!("launch {binary}"))?;
    assert!(
        output.status.success(),
        "{binary}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = String::from_utf8(output.stdout).context("report must be UTF-8")?;
    let fields = report.split_whitespace().collect::<Vec<_>>();
    assert_eq!(fields.len(), 8, "{binary}: {report:?}");
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
    let image = image::open(&path)
        .with_context(|| format!("open {backend} snapshot"))?
        .into_rgba8();
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

fn assert_semantics(name: &str, report: &str, image: &image::RgbaImage, min_events: u32) {
    let fields = report.split_whitespace().collect::<Vec<_>>();
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
