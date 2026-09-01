# Rust examples

The six numbered directories are standalone development programs that own their `ez-gfx` callbacks, scenes, Slang shader sources, and assets. Each binary calls `ez-gfx-compiler` once during resource initialization with its source path, SPIR-V/DXIL/Metal targets, and development mode, then passes validated artifact bytes to `load_shader`; no `build.rs` or generated `.ezgfxshader` is used. On Apple the compiler emits metallib; elsewhere it emits portable MSL alongside SPIR-V and DXIL. These non-distributed binaries intentionally carry Slang/DXC compiler tooling. `ez-gfx`, runtime/FFI crates, and packaged runtime distributions remain compiler-free. Every shader imports the root [`ez_gfx_api.slang`](../ez_gfx_api.slang) module and remains free of backend-specific syntax.

- [01 Triangle](01_triangle/README.md)
- [02 Textured Cube](02_textured_cube/README.md)
- [03 Compute Structured](03_compute_structured/README.md)
- [04 Dear ImGui](04_imgui/README.md)
- [05 Helmet](05_helmet/README.md)
- [06 Sponza KTX2](06_sponza_ktx2/README.md)

The Win32 [`C textured cube`](c/textured_cube/README.md) is a separate ABI v18 flow. CMake compiles its Slang source with SPIR-V, DXIL, and Metal targets in development mode, links `ez-gfx-ffi`, and copies the generated artifact beside the executable. CI builds it on Windows and executes Vulkan with SwiftShader; the hosted DX12 row is compile-only.

| Binary | Complete renderer | Owned inputs |
| --- | --- | --- |
| `01_triangle` | `01_triangle/main.rs` | Slang source, target list |
| `02_textured_cube` | `02_textured_cube/main.rs` | Slang source, target list, PNG |
| `03_compute_structured` | `03_compute_structured/main.rs` | Slang source, target list |
| `04_imgui` | `04_imgui/main.rs` | Slang source, target list |
| `05_helmet` | `05_helmet/main.rs` | Slang source, target list, GLB |
| `06_sponza_ktx2` | `06_sponza_ktx2/main.rs` | Slang source, target list |

The original Sponza GLB is the only shared Rust asset: examples 03 and 06 consume `shared/assets/sponza.glb`. Shared support is split into [`shared/data.rs`](shared/data.rs) (neutral Pod byte views), [`shared/host.rs`](shared/host.rs) (native window handles and Metal layer), [`shared/input.rs`](shared/input.rs) (winit input translation), [`shared/lifecycle.rs`](shared/lifecycle.rs) (event-loop and callback orchestration), [`shared/observability.rs`](shared/observability.rs) (bounded runtime/diagnostic draining), [`shared/math.rs`](shared/math.rs) (`glam`-backed projection/camera adapters), and [`shared/mesh.rs`](shared/mesh.rs) (`gltf` decoding, `glam` transforms, normalization, and neutral primitive records). `shared/mod.rs` re-exports these helpers and owns environment parsing, benchmark timing, snapshot readback, and external PNG comparison.

`shared/` owns neutral data, host/backend selection, input, lifecycle, observability, math, and mesh adapters. Each example's `main.rs` remains its complete renderer: it directly loads shaders/textures, acquires and writes buffers, and records compute/graphics work. Once a context exists, `destroy_context` performs the complete safe ez-gfx teardown; examples do not manually release child resources. No shared rendering-wrapper layer obscures that flow.

Benchmark mode is available on every binary. Stable JSON identities are `01_triangle`, `02_textured_cube`, `03_compute_structured`, `04_imgui`, `05_helmet`, and `06_sponza_ktx2`; benchmark frame limits always override the ordinary max-frame setting with warmup + measured + one terminal capture frame.

[`tests/smoke.rs`](tests/smoke.rs) exercises all six independent startup-compilation, scene, and immutable-snapshot paths plus triangle determinism. [`tests/shader_artifacts.rs`](tests/shader_artifacts.rs) compiles all six Slang sources with every target in development mode and verifies every stage, target, and reflected physical binding without caller entry-point selection. Shared helper tests cover benchmark, environment, byte-view, input, lifecycle, observability, math, and mesh contracts.
