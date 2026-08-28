use std::{path::PathBuf, process::Command};

const BINARIES: [(&str, &str, &str); 6] = [
    (
        "Triangle",
        "01_triangle.png",
        env!("CARGO_BIN_EXE_01_triangle"),
    ),
    (
        "TexturedCube",
        "02_textured_cube.png",
        env!("CARGO_BIN_EXE_02_textured_cube"),
    ),
    (
        "ComputeStructured",
        "03_compute_structured.png",
        env!("CARGO_BIN_EXE_03_compute_structured"),
    ),
    ("ImGui", "04_imgui.png", env!("CARGO_BIN_EXE_04_imgui")),
    ("Helmet", "05_helmet.png", env!("CARGO_BIN_EXE_05_helmet")),
    (
        "SponzaKtx2",
        "06_sponza_ktx2.png",
        env!("CARGO_BIN_EXE_06_sponza_ktx2"),
    ),
];

// Each process owns one OS event loop; winit intentionally forbids recreating it in a test process.
fn snapshot(binary: &str, file: &str) -> String {
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("snapshots")
        .join(file);
    let output = Command::new(binary)
        .env("EZ_GFX_EXAMPLE_MAX_FRAMES", "1")
        .env("EZ_GFX_EXAMPLE_REPORT", "1")
        .env("EZ_GFX_EXAMPLE_SNAPSHOT", reference)
        .env("VK_LOADER_LAYERS_DISABLE", "~implicit~")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = String::from_utf8(output.stdout).unwrap();
    assert!(
        report.starts_with("ez-gfx-snapshot 640 480 1 "),
        "{report:?}"
    );
    report
}

#[test]
fn every_migrated_binary_matches_its_rendered_pixel_snapshot() {
    for (name, file, binary) in BINARIES {
        let report = snapshot(binary, file);
        let fields = report.split_whitespace().collect::<Vec<_>>();
        assert_eq!(fields.len(), 8, "{name}");
        assert!(
            fields[5].parse::<u32>().unwrap() > 0,
            "{name} emitted no runtime events"
        );
        assert_eq!(fields[7], "0", "{name} dropped observations");
    }
}

#[test]
fn triangle_snapshot_is_deterministic_across_processes() {
    assert_eq!(
        snapshot(BINARIES[0].2, BINARIES[0].1),
        snapshot(BINARIES[0].2, BINARIES[0].1)
    );
}
