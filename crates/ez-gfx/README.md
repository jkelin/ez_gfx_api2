# ez-gfx

`ez-gfx` is the safe Rust graphics API. Rust applications depend on this crate, never `ez-gfx-ffi`; the FFI crate exists only for foreign-language C ABI clients.

## Lifecycle

`Context::new` returns the creator-thread-affine owner. `Context::create_surface` atomically creates the native surface, initializes the device, and applies the initial extent; failure rolls back the unpublished surface. Persistent resources retain context and parent leases. Their `Drop` implementations delegate release to the internal raw seam, so safe Rust exposes no destroy, release, remove, or free functions.

`Surface::begin_frame()` returns one target-less recording transaction. `Frame::configure_swapchain(size, format)` attaches the surface and yields its logical render target. `Context::begin_frame()` plus `Frame::configure_render_target(name, size, format)` selects a cached named target; extent or format changes recreate its native image. Recording requires `&mut Frame`. `Frame::finish(self)` consumes the transaction and preserves exact errors. Dropping an unfinished frame aborts because `Drop` cannot return errors.

`Buffer<T>` and `CounterBuffer<T>` are context-acquired one-frame CPU-backed values. They are populated before `begin_frame`; the first frame that binds one claims it, repeated compute/graphics uses in that frame share one native materialization, and every later write or frame use fails with `NotReady`. Finishing or aborting consumes the wrapper while native storage returns to completion-gated backend pools; indeterminate native failures quarantine storage until context cleanup. Geometry allocation wrappers retain their heap or context, and drops during recording defer range reuse through that frame without public retain calls.

## API and ownership

Import public items from the crate root. `Context`, `Surface`, `Shader`, `Texture`, `RenderTarget`, `VertexHeap<T>`, `VertexAllocation<T>`, `IndexAllocation`, `Buffer<T>`, `CounterBuffer<T>`, and `Frame` are non-`Copy`, non-`Send`, and non-`Sync`. Raw handles and explicit lifecycle functions are doc-hidden for `ez-gfx-ffi`.

The context-owned index heap is lazy behind `Context::upload_indices`. `Context::create_vertex_heap<T>(name)` derives stride and auto-grows storage; `VertexHeap::upload` derives checked count and byte size from `T: Pod`. `Context::acquire_buffer<T: Pod>` and `Context::acquire_counter_buffer<T: Pod>` allocate typed capacity. `Context::acquire_buffer_from` and `Context::acquire_counter_buffer_from` accept `BufferSource::one(&value)`, a slice, or a borrowed `Vec<T>` without an intermediate collection; arrays must use `.as_slice()` so they cannot be inferred as one array-valued element. Counter initialization publishes its input length. Writes validate element ranges; counter buffers publish their visible count explicitly.

`Context::load_shader` accepts validated artifact bytes. `Context::load_texture` copies caller bytes before returning and schedules CPU work. `Context::register_callback` is the sole safe event channel for upload, runtime, diagnostic, dropped-record, and callback-scoped readback events. `RenderTarget::prepare_readback(&mut frame)` creates and attaches an opaque owner-and-generation request; the callback receives its identity, dimensions, and scoped bytes.

## Shader artifacts

`Context::load_shader` consumes caller-provided `.ezgfxshader` bytes and selects each stage's artifact-owned entry point. Applications compile artifacts offline or during their build; `ez-gfx` does not choose files, compile source, link Slang, or provide JIT/fallback behavior. The host owns artifact authenticity and storage; the registered callback receives asynchronous outcomes.

## Complete flows

| Example | Demonstrates |
| --- | --- |
| [01 Triangle](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/01_triangle/README.md) | Minimal indexed-indirect graphics |
| [02 Textured Cube](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/02_textured_cube/README.md) | Texture binding, camera, and push constants |
| [03 Compute Structured](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/03_compute_structured/README.md) | Compute-generated indirect graphics |
| [04 Dear ImGui](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/04_imgui/README.md) | Dynamic UI buffers and per-command clipping |
| [05 Helmet](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/05_helmet/README.md) | GLB geometry and depth-tested rendering |
| [06 Sponza KTX2](https://github.com/jkelin/ez_gfx_api2/blob/main/examples/06_sponza_ktx2/README.md) | KTX2 materials and compute-to-graphics flow |
| [C textured cube](../../examples/c/textured_cube/README.md) | ABI 33 typed-heap, one-frame compute-to-graphics indexed-indirect cube |
