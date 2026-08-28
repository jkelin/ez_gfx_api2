# Rust examples

The six binaries preserve the original example sequence and exercise the corresponding public API path:

| Binary | Original source | Exercised path |
| --- | --- | --- |
| `01_triangle` | `examples/1_triangle/main.odin` | vertex/index uploads and indexed-indirect command recording |
| `02_textured_cube` | `examples/2_textured_cube/main.odin` | cube geometry plus RGBA texture upload and bindless index resolution |
| `03_compute_structured` | `examples/3_compute_structured_buffer/main.odin`, `tests/example3.odin` | precompiled universal artifact loading, structured bindings, compute PSO creation, dispatch, and indexed draw |
| `04_imgui` | `examples/4_imgui/main.odin` | dynamic UI vertex/index stream and font-atlas texture lifecycle |
| `05_helmet` | `examples/5_helmet_cgltf/main.odin`, `tests/example5.odin` | multi-primitive geometry upload and indirect command lifecycle |
| `06_sponza_ktx2` | `examples/6_sponza_ktx2/main.odin`, `tests/example6.odin` | larger geometry stream plus BasisLZ KTX2 decode, upload, and binding |

`tests/smoke.rs` is the Rust successor to the original snapshot harness. It runs each binary in a separate compiler-free process, presents through a real native window and swapchain, polls runtime events and diagnostics, reads back RGBA8 pixels, and compares them with `snapshots/*.png`. All six load the checked-in multi-target `assets/scene.ezgfx`; compiler tests exercise Slang separately.

Run one with `mise run example-3`. Rebuild `assets/scene.ezgfx` from `scene.slang` with `mise run examples-artifacts`; this is the separate compiler path and requires Slang. Run the compiler-free rendered snapshot gate with `mise run examples-smoke`. To intentionally replace references, run that gate with `EZ_GFX_UPDATE_SNAPSHOTS=1`; ordinary runs fail on missing, malformed, resized, or pixel-different images. Set `EZ_GFX_EXAMPLE_MAX_FRAMES` only when automating one binary.
