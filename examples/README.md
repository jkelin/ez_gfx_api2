# Rust examples

Each directory is a standalone program. Its `main.rs` imports only local `host` and `scenes` modules. Those modules own platform setup, application lifecycle, concrete rendering, and required helpers; no Rust implementation is shared between examples.

| Binary | Local scene | Owned inputs |
| --- | --- | --- |
| `01_triangle` | `01_triangle/scenes/triangle.rs` | shader config, shader, artifact |
| `02_textured_cube` | `02_textured_cube/scenes/textured_cube.rs` | shader config, shader, artifact, PNG |
| `03_compute_structured` | `03_compute_structured/scenes/model.rs` | shader config, shader, artifact |
| `04_imgui` | `04_imgui/scenes/imgui_scene.rs` | shader config, shader, artifact |
| `05_helmet` | `05_helmet/scenes/model.rs` | shader config, shader, artifact, GLB |
| `06_sponza_ktx2` | `06_sponza_ktx2/scenes/model.rs` | shader config, shader, artifact |

The original Sponza GLB is the only shared input: both examples 03 and 06 consume `shared/assets/sponza.glb`. Root `Cargo.toml` declares only the six binaries and smoke test. `tests/smoke.rs` launches every binary independently and compares `snapshots/`.

The original `tests/snapshots/example3.expected.png` covers the companion structured-buffer test, not the interactive example 03. The Rust snapshot follows the standalone program.
