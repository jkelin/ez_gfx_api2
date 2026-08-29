use std::{path::PathBuf, process::Command};

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
fn target_backend() -> &'static str {
    "metal"
}

#[cfg(not(target_vendor = "apple"))]
fn target_backend() -> &'static str {
    "vulkan"
}

fn snapshot(binary: &str, file: &str) -> (String, image::RgbaImage) {
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("snapshots")
        .join(file);
    let mut command = Command::new(binary);
    command
        .env("EZ_GFX_BACKEND", target_backend())
        .env("EZ_GFX_EXAMPLE_MAX_FRAMES", "1")
        .env("EZ_GFX_EXAMPLE_REPORT", "1")
        .env("EZ_GFX_EXAMPLE_SNAPSHOT", &reference)
        .env("VK_LOADER_LAYERS_DISABLE", "~implicit~");
    #[cfg(target_vendor = "apple")]
    command.env("MTL_DEBUG_LAYER", "1");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("launch {binary}: {error}"));
    assert!(
        output.status.success(),
        "{binary}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = String::from_utf8(output.stdout).expect("report must be UTF-8");
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
    let image = image::open(reference)
        .expect("snapshot exists")
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
    (report, image)
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
        fn $test() {
            let (name, file, binary, min_events) = BINARIES[$index];
            let (report, image) = snapshot(binary, file);
            assert_semantics(name, &report, &image, min_events);
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
fn triangle_snapshot_is_deterministic_across_processes() {
    let (_, _, binary, _) = BINARIES[0];
    assert_eq!(
        snapshot(binary, BINARIES[0].1).0,
        snapshot(binary, BINARIES[0].1).0
    );
}
