# ez-gfx

`ez-gfx` is the safe Rust graphics API. Rust applications depend on this crate, never `ez-gfx-ffi`; the FFI crate exists only for foreign-language C ABI clients.

## Lifecycle

Keep the owning context handle with every surface and resource, and follow this order:

1. Create a context, then a compatible surface, then initialize the device for that surface.
2. Load shaders and textures and acquire geometry, structured, and indirect resources.
3. Begin a presented frame with `begin_render`, record compute or graphics work, and call `finish_render` to submit and present. A failed submission is not presented.
4. Before teardown, call `wait_idle`; release or destroy resources, destroy the surface, then destroy the context.

Release and destroy functions must be paired with successful acquisitions even though context destruction ultimately owns native cleanup.

## API and handles

Import public items from the crate root. Types such as `PublicBinding`, `ResourceIdentity`, `DrawIndexedCommand`, `ShaderRequest`, and `TextureSource` have no nested compatibility paths.

Contexts and resources are opaque generational `u64` handles. Resource handles encode their owning context and generational slot; resource kind is validated by the owning context's registry. Validated operations reject zero, malformed, stale, wrong-context, and kind-mismatched handles. Handles do not provide RAII: callers remain responsible for the matching release or destroy operation.

A context is bound to its creator thread. Context and resource operations, including teardown, must run on that thread; wrong-thread work is rejected, and a wrong-thread void destruction cannot complete cleanup. Contexts currently share one process-global mutex, so independent contexts are serialized.

Operations that produce a handle or value, including acquisitions and indexed uploads, return `Result<T, EzGfxResult>`. Commands with no returned value, such as recording, writes, synchronization, and lifecycle transitions, return `EzGfxResult` directly. Release and destroy operations return no status, so validate lifecycle ordering before cleanup.

## Shader artifacts

`load_shader` consumes caller-provided compiled artifact bytes. The host owns artifact selection, authenticity, filesystem access, and event polling; `ez-gfx` does not compile shaders or choose files at runtime.

## Complete flows

| Example | Demonstrates |
| --- | --- |
| [01 Triangle](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/01_triangle/README.md) | Minimal indexed-indirect graphics |
| [02 Textured Cube](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/02_textured_cube/README.md) | Texture binding, camera, and push constants |
| [03 Compute Structured](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/03_compute_structured/README.md) | Compute-generated indirect graphics |
| [04 Dear ImGui](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/04_imgui/README.md) | Dynamic UI buffers and per-command clipping |
| [05 Helmet](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/05_helmet/README.md) | GLB geometry and depth-tested rendering |
| [06 Sponza KTX2](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/06_sponza_ktx2/README.md) | KTX2 materials and compute-to-graphics flow |
