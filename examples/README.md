# Rust examples

The six numbered directories are procedural development renderers hosted by the shared `Example` lifecycle. Each main calls `run_program(identity, width, height, title, run_example)`; the wrapper derives `clap::Parser` once, resolves CLI/environment conflicts, builds `ExampleConfig`, then calls `run(config, setup)`. The host owns winit inversion, native window/context/surface construction, resize, input queuing, pacing, benchmark, capture, and reporting. `setup(&Context, &Surface, Backend)` returns a per-frame closure that begins, records, and returns an owning `Frame`; `Example::handle_frame(Frame)` consumes it with `Frame::finish`.

- [01 Triangle](01_triangle/README.md)
- [02 Textured Cube](02_textured_cube/README.md)
- [03 Compute Structured](03_compute_structured/README.md)
- [04 Dear ImGui](04_imgui/README.md)
- [05 Helmet](05_helmet/README.md)
- [06 Sponza KTX2](06_sponza_ktx2/README.md)

The Rust examples enable `ktx2` and `basis`; the Sponza example therefore retains universal decoding. Library/FFI default builds omit those decoders. Enable both for universal KTX2, `ktx2` for native blocks, or `basis` for standalone Basis; see [texture admission](../docs/textures.md#admission-and-memory).

The Win32 [`C textured cube`](c/textured_cube/README.md) is a separate ABI 31 flow. C uses an opaque generational `EzGfxFrame` and must explicitly call `ez_gfx_frame_end` or `ez_gfx_frame_abort`; Rust examples never use those raw completion functions.

| Binary | Complete renderer | Owned inputs |
| --- | --- | --- |
| `01_triangle` | `01_triangle/main.rs` | Slang source, target list |
| `02_textured_cube` | `02_textured_cube/main.rs` | Slang source, target list, PNG |
| `03_compute_structured` | `03_compute_structured/main.rs` | Slang source, target list |
| `04_imgui` | `04_imgui/main.rs` | Slang source, target list |
| `05_helmet` | `05_helmet/main.rs` | Slang source, target list, GLB |
| `06_sponza_ktx2` | `06_sponza_ktx2/main.rs` | Slang source, target list |

The original Sponza GLB is the only shared Rust asset: examples 03 and 06 consume `shared/assets/sponza.glb`. Shared support owns neutral data, native window attachment, input translation, observability, math, and mesh decoding. `shared/example.rs` is the single host for `ApplicationHandler`, native window, `Context`, `Surface`, resize, queued input, frame pacing, benchmark, capture, and reporting; `shared/error.rs` provides host errors.

Persistent resources are owning wrappers whose leases retain their context until `Drop`. The context owns the singleton index heap. Structured and indirect buffers are acquired from `&mut Frame` and become stale on every terminal frame path, including implicit abort by `Drop`; no renderer manually destroys or releases safe resources.

Benchmark mode is available on every binary. Stable JSON identities are `01_triangle`, `02_textured_cube`, `03_compute_structured`, `04_imgui`, `05_helmet`, and `06_sponza_ktx2`; benchmark frame limits always override the ordinary max-frame setting with warmup + measured + one terminal capture frame.

[`tests/smoke.rs`](tests/smoke.rs) exercises all six independent startup-compilation, scene, and immutable-snapshot paths plus triangle determinism. [`tests/shader_artifacts.rs`](tests/shader_artifacts.rs) compiles all six Slang sources with every target in development mode and verifies every stage, target, and reflected physical binding without caller entry-point selection. Shared helper tests cover benchmark, environment, byte-view, input, lifecycle, observability, math, and mesh contracts.
