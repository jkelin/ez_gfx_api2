# Rust examples

The six numbered directories are linear procedural renderers hosted by one shared `Example`. Each main constructs `Example`, creates persistent resources top-to-bottom, then loops over `wait_for_next_frame`. The host hides clap configuration, winit inversion, native context/surface setup, resize, queued input, pacing, callbacks, benchmark, capture, and reporting. User code explicitly begins and configures each swapchain frame, records it, then calls `Example::handle_frame(frame, swapchain_target)`.

- [01 Triangle](01_triangle/README.md)
- [02 Textured Cube](02_textured_cube/README.md)
- [03 Compute Structured](03_compute_structured/README.md)
- [04 Dear ImGui](04_imgui/README.md)
- [05 Helmet](05_helmet/README.md)
- [06 Sponza KTX2](06_sponza_ktx2/README.md)

The Rust examples enable `ktx2` and `basis`; the Sponza example therefore retains universal decoding. Library/FFI default builds omit those decoders. Enable both for universal KTX2, `ktx2` for native blocks, or `basis` for standalone Basis; see [texture admission](../docs/textures.md#admission-and-memory).

The Win32 [`C textured cube`](c/textured_cube/README.md) is a separate ABI 33 flow. C uses an opaque generational `EzGfxFrame` and must explicitly call `ez_gfx_frame_end` or `ez_gfx_frame_abort`; Rust examples never use those raw completion functions.

| Binary | Complete renderer | Owned inputs |
| --- | --- | --- |
| `01_triangle` | `01_triangle/main.rs` | Slang source, target list |
| `02_textured_cube` | `02_textured_cube/main.rs` | Slang source, target list, PNG |
| `03_compute_structured` | `03_compute_structured/main.rs` | Slang source, target list |
| `04_imgui` | `04_imgui/main.rs` | Slang source, target list |
| `05_helmet` | `05_helmet/main.rs` | Slang source, target list, GLB |
| `06_sponza_ktx2` | `06_sponza_ktx2/main.rs` | Slang source, target list |

The original Sponza GLB is the only shared Rust asset: examples 03 and 06 consume `shared/assets/sponza.glb`. Shared support owns neutral data, native window attachment, input translation, observability, math, and mesh decoding. `shared/example.rs` is the single host for `ApplicationHandler`, native window, `Context`, `Surface`, resize, queued input, frame pacing, benchmark, capture, and reporting; `shared/error.rs` provides host errors.

Persistent resources are owning wrappers whose leases retain their context until `Drop`. The context owns the lazy singleton index heap. `Buffer<T>` and `CounterBuffer<T>` are acquired from `Context` for one frame, populated before first use, and claimed when bound. Same-frame reuse is valid; terminal completion invalidates them. Each loop explicitly begins and configures a swapchain frame before passing the frame and target to `Example::handle_frame`; no renderer manually destroys or releases safe resources.

Benchmark mode is available on every binary. Stable JSON identities are `01_triangle`, `02_textured_cube`, `03_compute_structured`, `04_imgui`, `05_helmet`, and `06_sponza_ktx2`; benchmark frame limits always override the ordinary max-frame setting with warmup + measured + one terminal capture frame.

[`tests/smoke.rs`](tests/smoke.rs) exercises all six startup-compilation, scene, and immutable-snapshot paths plus triangle determinism. [`tests/shader_artifacts.rs`](tests/shader_artifacts.rs) compiles all six Slang sources for every target and verifies stage, target, and reflected binding contracts. Shared helper tests cover benchmark, environment, byte views, input, host behavior, observability, math, and mesh contracts.
