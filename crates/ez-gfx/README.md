# ez-gfx

`ez-gfx` is the safe Rust graphics API. Rust applications depend on this crate, never `ez-gfx-ffi`; the FFI crate exists only for foreign-language C ABI clients.

## Lifecycle

Keep the owning context handle with every surface and resource, and follow this order:

1. Create a context, then a compatible surface, then initialize the device for that surface.
2. Load persistent shaders, textures, and typed geometry heaps/allocations.
3. Begin a frame, acquire fresh structured and indirect buffers, write or publish their active ranges, record compute/graphics work, and submit. Successful recording transfers transient ownership to the frame; successful submission invalidates their handles and recycles native storage only after GPU completion.
4. On recording failure, release unconsumed transient handles or destroy the context. Submission rollback restores handles only after no native work started or an idle drain succeeds; uncertain native failures quarantine storage until terminal context cleanup.

`destroy_context` is terminal once cleanup begins. It returns the first initialized-device wait or release failure after attempting every remaining release; the context and all child handles are stale even when teardown reports an error. Invalid, stale, repeated, and wrong-thread destroys return `Err(Error::InvalidContext)` without consuming a live context.

## API and handles

Import public items from the crate root. Types such as `PublicBinding`, `ResourceIdentity`, `DrawIndexedCommand`, and `TextureSource` have no nested compatibility paths. `load_shader(context, bytes)` selects the artifact-owned entry point for each available stage.

Contexts and resources use distinct transparent Rust handle types, including `VertexHeapHandle`. Each preserves the packed `u64` C representation and exposes explicit boundary conversion. Resource handles encode owner and generation; validated operations reject zero, malformed, stale, foreign, and kind-mismatched handles. Reflection resolves heap semantic names internally; public uploads and destruction use the typed heap handle, never a name.

A globally synchronized generational arena allocates unique context handles. Context state stays in creator-thread-local storage, and every context/resource operation remains creator-thread-affine. Independent creator threads do not share a state mutex. Normal creator-thread TLS teardown invalidates remaining handles; `ExitProcess` may terminate other threads without running their TLS destructors. Windows TLS teardown cannot safely run or join DX12/COM cleanup under loader lock, so it deliberately retains the entire remaining context until process termination; use explicit `destroy_context` for any earlier reclamation or observable cleanup. Non-Windows normal TLS teardown retains synchronous best-effort cleanup.

Fallible operations return the crate-root `Result<T, Error>`. `Error` is the single `thiserror` facade and preserves lifecycle/capability sources where they cross the public boundary; C status codes exist only in `ez-gfx-ffi`. Vertex uploads accept `&[T]` where `T: Pod`, and index uploads accept `&[u32]`; both infer count and byte layout, return typed allocation handles, and must be removed before heap destruction.
`acquire_structured<T: Pod>(context, element_count)` infers stride and capacity; `write_structured` validates that stride and writes an indexed element range. `write_indirect` writes a command slice and advances the active count to the maximum written end. GPU-generated commands use the narrow `publish_compute_indirect_count` path because current backends require a CPU-known draw count.

`load_texture` copies caller bytes and schedules unbounded CPU admission subject to real allocation failure. `poll_upload_event` is the lossless authoritative progress path for texture, vertex, and index uploads and must be drained once per frame. `cancel_texture_load` invalidates work before initial readiness; residency and unload retain their completion-safe behavior. See the [texture](../../docs/textures.md) and [geometry](../../docs/geometry.md) guides.

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
| [C textured cube](../../examples/c/textured_cube/README.md) | ABI 30 typed-heap, transient compute-to-graphics indexed-indirect cube |
