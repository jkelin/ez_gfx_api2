# Rust examples

The six numbered directories are standalone programs that own their `ez-gfx` callbacks, scenes, shader sources, compiled artifacts, and assets. Shared `host` and `lifecycle` support supplies window integration and event-loop orchestration. Start with the per-example guides:

- [01 Triangle](01_triangle/README.md)
- [02 Textured Cube](02_textured_cube/README.md)
- [03 Compute Structured](03_compute_structured/README.md)
- [04 Dear ImGui](04_imgui/README.md)
- [05 Helmet](05_helmet/README.md)
- [06 Sponza KTX2](06_sponza_ktx2/README.md)

| Binary | Complete renderer | Owned inputs |
| --- | --- | --- |
| `01_triangle` | `01_triangle/main.rs` | shader config, shader, artifact |
| `02_textured_cube` | `02_textured_cube/main.rs` | shader config, shader, artifact, PNG |
| `03_compute_structured` | `03_compute_structured/main.rs` | shader config, shader, artifact |
| `04_imgui` | `04_imgui/main.rs` | shader config, shader, artifact |
| `05_helmet` | `05_helmet/main.rs` | shader config, shader, artifact, GLB |
| `06_sponza_ktx2` | `06_sponza_ktx2/main.rs` | shader config, shader, artifact |

The original Sponza GLB is the only shared asset: examples 03 and 06 consume `shared/assets/sponza.glb`. Shared support is split into [`shared/data.rs`](shared/data.rs) (neutral Pod byte views), [`shared/host.rs`](shared/host.rs) (native window handles and Metal layer), [`shared/input.rs`](shared/input.rs) (winit input translation), [`shared/lifecycle.rs`](shared/lifecycle.rs) (event-loop and callback orchestration), [`shared/observability.rs`](shared/observability.rs) (bounded runtime/diagnostic draining), [`shared/math.rs`](shared/math.rs) (matrices and orbit camera), and [`shared/mesh.rs`](shared/mesh.rs) (GLB decoding, normalization, and neutral primitive records). `shared/mod.rs` re-exports these helpers and owns environment-flag parsing, benchmark timing, frame-limit override, snapshot reporting, and benchmark JSON output.

`shared/` contains no `ez-gfx` calls. Each example's `main.rs` is its complete renderer: it directly loads shaders/textures, acquires and writes buffers, records compute/graphics work, and releases safe ez-gfx resources. No shared or local graphics-wrapper layer obscures that flow.

Benchmark mode is available on every binary. Stable JSON identities are `01_triangle`, `02_textured_cube`, `03_compute_structured`, `04_imgui`, `05_helmet`, and `06_sponza_ktx2`; benchmark frame limits always override the ordinary max-frame setting with warmup + measured + one terminal capture frame.

The smoke test in [`tests/smoke.rs`](tests/smoke.rs) covers six independent scene tests plus one triangle determinism test: seven process/snapshot tests (the determinism test launches twice). Shared contributes thirteen helper tests: benchmark (1), environment flags (1), byte views (1), input (1), lifecycle (1), observability (2), math (3), and mesh (3), for 20 checks total. The current layout has six binaries under `examples/`, one shared asset under `shared/assets/`, six references under `snapshots/`, and one integration smoke test under `tests/`.
