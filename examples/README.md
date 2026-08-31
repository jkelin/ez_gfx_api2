# Rust examples

The six numbered directories are standalone Rust programs that own their `ez-gfx` callbacks, scenes, shader manifests/sources, and assets. Cargo's build script calls `ez-gfx-compiler` for all six manifests and writes `.ezgfxshader` files under `OUT_DIR`; generated artifacts are never tracked. Non-Apple builds embed SPIR-V, DXIL, and portable MSL coverage without invoking `xcrun`; Apple builds embed metallib coverage. Every shader imports the root [`ez_gfx_api.slang`](../ez_gfx_api.slang) module and remains free of backend-specific binding syntax.

- [01 Triangle](01_triangle/README.md)
- [02 Textured Cube](02_textured_cube/README.md)
- [03 Compute Structured](03_compute_structured/README.md)
- [04 Dear ImGui](04_imgui/README.md)
- [05 Helmet](05_helmet/README.md)
- [06 Sponza KTX2](06_sponza_ktx2/README.md)

The Win32 [`C structured-buffer cube`](c/structured_cube/README.md) is a separate ABI v18 flow. CMake compiles its manifest, links `ez-gfx-ffi`, and copies the generated artifact beside the executable. CI builds it on Windows and executes Vulkan with SwiftShader; the hosted DX12 row is compile-only.

| Binary | Complete renderer | Owned inputs |
| --- | --- | --- |
| `01_triangle` | `01_triangle/main.rs` | shader manifest, shader source |
| `02_textured_cube` | `02_textured_cube/main.rs` | shader manifest, shader source, PNG |
| `03_compute_structured` | `03_compute_structured/main.rs` | shader manifest, shader source |
| `04_imgui` | `04_imgui/main.rs` | shader manifest, shader source |
| `05_helmet` | `05_helmet/main.rs` | shader manifest, shader source, GLB |
| `06_sponza_ktx2` | `06_sponza_ktx2/main.rs` | shader manifest, shader source |

The original Sponza GLB is the only shared Rust asset: examples 03 and 06 consume `shared/assets/sponza.glb`. Shared support is split into [`shared/data.rs`](shared/data.rs) (neutral Pod byte views), [`shared/host.rs`](shared/host.rs) (native window handles and Metal layer), [`shared/input.rs`](shared/input.rs) (winit input translation), [`shared/lifecycle.rs`](shared/lifecycle.rs) (event-loop and callback orchestration), [`shared/observability.rs`](shared/observability.rs) (bounded runtime/diagnostic draining), [`shared/math.rs`](shared/math.rs) (`glam`-backed projection/camera adapters), and [`shared/mesh.rs`](shared/mesh.rs) (`gltf` decoding, `glam` transforms, normalization, and neutral primitive records). `shared/mod.rs` re-exports these helpers and owns environment parsing, benchmark timing, snapshot readback, and external PNG comparison.

`shared/` contains no `ez-gfx` calls. Each example's `main.rs` is its complete renderer: it directly loads shaders/textures, acquires and writes buffers, records compute/graphics work, and releases safe ez-gfx resources. No shared or local graphics-wrapper layer obscures that flow.

Benchmark mode is available on every binary. Stable JSON identities are `01_triangle`, `02_textured_cube`, `03_compute_structured`, `04_imgui`, `05_helmet`, and `06_sponza_ktx2`; benchmark frame limits always override the ordinary max-frame setting with warmup + measured + one terminal capture frame.

[`tests/smoke.rs`](tests/smoke.rs) exercises all six independent scene/snapshot paths and triangle determinism. [`tests/shader_artifacts.rs`](tests/shader_artifacts.rs) verifies that build-generated artifacts cover every stage and target without caller entry-point selection. Shared helper tests cover benchmark, environment, byte-view, input, lifecycle, observability, math, and mesh contracts.
