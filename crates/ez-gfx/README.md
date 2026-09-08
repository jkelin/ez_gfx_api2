# ez-gfx

`ez-gfx` is the safe Rust graphics API. Rust applications depend on this crate, never `ez-gfx-ffi`; the FFI crate exists only for foreign-language C ABI clients.

## Lifecycle

`create_context` returns the creator-thread-affine `Context`. `create_surface(&context, options)` atomically creates the native surface, initializes the device, and applies the initial extent; failure rolls back the unpublished surface. Persistent resources retain the context and their parent resource leases. Their `Drop` implementations delegate release to the internal raw seam, so safe Rust exposes no destroy, release, remove, or free functions.

`begin_frame(&context, &surface)` returns one owning recording transaction. Recording and frame-local acquisition require `&mut Frame`. `Frame::finish(self)` consumes the transaction and preserves exact submission or presentation errors. Dropping an unfinished frame aborts and rolls back because `Drop` cannot return errors.

Frame-local `StructuredBuffer<T>` and `IndirectBuffer` values share transaction state. Successful completion or abort invalidates them. Native storage is recycled only after GPU completion; indeterminate native failures quarantine it until terminal context cleanup. A `VertexAllocation` retains its `VertexHeap`, and a recording frame retains every referenced persistent allocation through completion.

## API and ownership

Import public items from the crate root. `Context`, `Surface`, `Shader`, `Texture`, `RenderTarget`, `VertexHeap`, `VertexAllocation`, `IndexAllocation`, `StructuredBuffer<T>`, and `IndirectBuffer` are non-`Copy`, non-`Send`, and non-`Sync` owners. Raw handles and explicit lifecycle functions are doc-hidden and reserved for `ez-gfx-ffi`.

The context-owned index heap remains a singleton behind `upload_indices`. Vertex uploads accept `&[T]` where `T: Pod`; index uploads accept `&[u32]`. `Frame::acquire_structured<T: Pod>` infers stride and capacity. Transient writes validate element ranges, and indirect writes publish the maximum written command count. GPU-generated commands use `IndirectBuffer::publish_compute_count`.

`load_shader` accepts owned validated artifact bytes. `load_texture` copies caller bytes before returning and schedules CPU work. Context polling methods expose the lossless upload/runtime queues; applications drain them once per frame on the creator thread. See the [texture](../../docs/textures.md) and [geometry](../../docs/geometry.md) guides.

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
| [C textured cube](../../examples/c/textured_cube/README.md) | ABI 31 typed-heap, transient compute-to-graphics indexed-indirect cube |
