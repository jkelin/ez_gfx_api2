# Rust examples

The six numbered directories are linear procedural renderers hosted by one shared `Example`. Each main creates a platform-free `Context`, then passes an owned clone of the winit host to `context.create_surface_window`; the surface retains it and queries its native drawable extent. The host handles winit inversion, resize, input, pacing, callbacks, benchmark, capture, and reporting.

- [01 Triangle](01_triangle/README.md)
- [02 Textured Cube](02_textured_cube/README.md)
- [03 Compute Structured](03_compute_structured/README.md)
- [04 Dear ImGui](04_imgui/README.md)
- [05 Helmet](05_helmet/README.md)
- [06 Sponza KTX2](06_sponza_ktx2/README.md)

The Rust examples enable `ktx2` and `basis`; the Sponza example therefore retains universal decoding. Library/FFI default builds omit those decoders. Enable both for universal KTX2, `ktx2` for native blocks, or `basis` for standalone Basis; see [texture admission](../docs/textures.md#admission-and-memory).

The portable GLFW [`C textured cube`](02_textured_cube_c/README.md) is a separate Vulkan flow using tagged native window handles and immediate presentation with deterministic fallback. C owns its opaque generational frames and explicitly ends or aborts them.

| Binary | Complete renderer | Owned inputs |
| --- | --- | --- |
| `01_triangle` | `01_triangle/main.rs` | Slang source, target list |
| `02_textured_cube` | `02_textured_cube/main.rs` | Slang source, target list, PNG |
| `03_compute_structured` | `03_compute_structured/main.rs` | Slang source, target list |
| `04_imgui` | `04_imgui/main.rs` | Slang source, target list |
| `05_helmet` | `05_helmet/main.rs` | Slang source, target list, GLB |
| `06_sponza_ktx2` | `06_sponza_ktx2/main.rs` | Slang source, target list |

The original Sponza GLB is the only shared Rust asset: examples 03 and 06 consume `shared/assets/sponza.glb`. Shared support owns neutral data, native window attachment, input translation, observability, math, and mesh decoding. `shared/example.rs` is the single host for `ApplicationHandler`, native window, resize, queued input, frame pacing, benchmark, capture, and reporting; each main owns its `Context` and `Surface`. Presented-snapshot caching remains off during interactive and benchmark frames; only the terminal automation frame requests readback.

The context owns every graphics resource and destroys remaining resources when its owner is destroyed or dropped. Texture wrapper drop does not unload stable bindless heap entries. `Buffer<T>` and `CounterBuffer<T>` are acquired for one frame, populated before first use, and claimed when bound; same-frame reuse is valid and terminal completion invalidates them. Each loop explicitly begins and configures a swapchain frame before passing the frame and target to `Example::handle_frame`. Ordinary reverse declaration order handles example teardown without artificial scopes or explicit surface/context calls.

Benchmark mode is available on every binary. Stable JSON identities are `01_triangle`, `02_textured_cube`, `03_compute_structured`, `04_imgui`, `05_helmet`, and `06_sponza_ktx2`; benchmark frame limits always override the ordinary max-frame setting with warmup + measured + one terminal capture frame. Title FPS samples completed presentations, independent of redraw-event delivery.

`--frame-timings` requires a finite `--max-frames` or benchmark run. It emits one `ez-gfx-frame-timing` record per presented frame: example identity, backend, frame number, host-wait nanoseconds, recording nanoseconds, and submit/present nanoseconds.

[`tests/smoke.rs`](tests/smoke.rs) exercises all six startup-compilation, scene, and immutable-snapshot paths plus triangle determinism. [`tests/shader_artifacts.rs`](tests/shader_artifacts.rs) compiles all six Slang sources for every target and verifies stage, target, and reflected binding contracts. Shared helper tests cover benchmark, environment, byte views, input, host behavior, observability, math, and mesh contracts.
