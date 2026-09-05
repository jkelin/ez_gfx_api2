# ez-gfx

`ez-gfx` is the safe Rust graphics API. Rust applications depend on this crate, never `ez-gfx-ffi`; the FFI crate exists only for foreign-language C ABI clients.

## Lifecycle

Keep the owning context handle with every surface and resource, and follow this order:

1. Create a context, then a compatible surface, then initialize the device for that surface.
2. Load shaders and textures and acquire geometry, structured, and indirect resources.
3. Begin a presented frame with `begin_render`, record compute or graphics work, and call `finish_render` to submit and present. A failed submission is not presented.
4. Call `destroy_context` on the creator thread. It is mandatory on Windows for any reclamation before process exit. If device initialization completed, it waits idle before destroying every context-owned resource in dependency-safe order; a context destroyed before device initialization has no GPU work to wait for. Explicit resource destruction remains available for early reclamation.

`destroy_context` is terminal once cleanup begins. It returns the first initialized-device wait or release failure after attempting every remaining release; the context and all child handles are stale even when teardown reports an error. Invalid, stale, repeated, and wrong-thread destroys return `EzGfxResult::InvalidContext` without consuming a live context.

## API and handles

Import public items from the crate root. Types such as `PublicBinding`, `ResourceIdentity`, `DrawIndexedCommand`, and `TextureSource` have no nested compatibility paths. `load_shader(context, bytes)` selects the artifact-owned entry point for each available stage.

Contexts and resources use distinct transparent Rust handle types, so a surface, shader, buffer, or texture cannot be passed to an operation for another resource kind. Each type preserves the packed `u64` wire representation used by the C ABI and exposes explicit `from_raw`/`into_raw` conversion at that boundary. Resource handles encode their owning context and generational slot; validated operations reject zero, malformed, stale, wrong-context, and kind-mismatched handles. Handles do not provide RAII. Call an individual release or destroy operation for early reclamation, or rely on `destroy_context` for terminal cascading cleanup.

A globally synchronized generational arena allocates unique context handles. Context state stays in creator-thread-local storage, and every context/resource operation remains creator-thread-affine. Independent creator threads do not share a state mutex. Normal creator-thread TLS teardown invalidates remaining handles; `ExitProcess` may terminate other threads without running their TLS destructors. Windows TLS teardown cannot safely run or join DX12/COM cleanup under loader lock, so it deliberately retains the entire remaining context until process termination; use explicit `destroy_context` for any earlier reclamation or observable cleanup. Non-Windows normal TLS teardown retains synchronous best-effort cleanup.

Operations that produce a handle or value, including acquisitions and indexed uploads, return `Result<T, EzGfxResult>`. Commands with no returned value, including synchronization and `destroy_context`, return `EzGfxResult` directly. Individual resource release functions remain status-free.

`load_texture` copies the caller payload, reserves a texture handle, and returns after bounded Rayon work is queued. `poll_texture_load` reports `NotReady` through decode and transfer completion; `cancel_texture_load` can invalidate a pending or native-admitted request before its first sample-ready completion. `set_texture_residency` controls the contiguous coarse mip range without exposing incomplete transfers. `unload_texture` invalidates immediately and defers native reclamation behind transfer/graphics completion. `wait_idle` drains CPU work and transfer queues. See the [texture guide](../../docs/textures.md) for the complete lifecycle, render integration, backpressure, and backend queue contract.

## Shader artifacts

`load_shader` consumes caller-provided `.ezgfxshader` bytes and selects each stage's artifact-owned entry point. Applications compile artifacts offline or during their build; `ez-gfx` does not choose files, compile source, link Slang, or provide JIT/fallback behavior. The host owns artifact authenticity, storage, and event polling.

## Complete flows

| Example | Demonstrates |
| --- | --- |
| [01 Triangle](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/01_triangle/README.md) | Minimal indexed-indirect graphics |
| [02 Textured Cube](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/02_textured_cube/README.md) | Texture binding, camera, and push constants |
| [03 Compute Structured](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/03_compute_structured/README.md) | Compute-generated indirect graphics |
| [04 Dear ImGui](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/04_imgui/README.md) | Dynamic UI buffers and per-command clipping |
| [05 Helmet](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/05_helmet/README.md) | GLB geometry and depth-tested rendering |
| [06 Sponza KTX2](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/06_sponza_ktx2/README.md) | KTX2 materials and compute-to-graphics flow |
| [C textured cube](../../examples/c/textured_cube/README.md) | ABI v23 compute-written indexed-indirect cube on Win32 |
