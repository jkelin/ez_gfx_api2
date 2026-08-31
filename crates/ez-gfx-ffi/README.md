# ez-gfx-ffi

`ez-gfx-ffi` is the C ABI boundary for the `ez-gfx` runtime. C and other foreign-language clients must include [`include/ez_gfx_api.h`](../../include/ez_gfx_api.h); the declarations and numeric values in that header are canonical. Rust clients should depend on `ez-gfx`, not `ez-gfx-ffi`.

## Compatibility and ownership

Before any other call, read `ez_gfx_abi_version()` and require `EZ_GFX_ABI_VERSION` (ABI v17). Do not call the ABI when the version does not match.

`ez_gfx_handle_inspect` decodes a packed handle into its context/child slot and generation fields; it does not validate that the handle is live in a context. `ez_gfx_semantic_id` accepts a non-empty UTF-8 byte range and writes its fixed 16-byte identifier.

The context and all context/resource operations, including teardown, are creator-thread-affine in the delegated runtime. A call from another thread is rejected; a void destruction call cannot report that failure, so perform destruction on the creator thread. The host owns the native window/display/connection or `CAMetalLayer` pointers supplied in `EzGfxSurfaceDesc` and must keep them valid until the returned surface is destroyed. The runtime owns created surfaces, shaders, textures, buffers, heaps, and their native graphics objects; the host owns the opaque handle values and must release them through this ABI.

## Call order

1. Create a context with `ez_gfx_context_create` or `ez_gfx_context_create_backend`.
2. Create a presentation surface with `ez_gfx_surface_create`, then initialize its device with `ez_gfx_context_init_device`.
3. Create/load resources as needed: `ez_gfx_shader_load_artifact`, `ez_gfx_texture_load`, `ez_gfx_vertex_heap_create`, `ez_gfx_index_heap_create`, `ez_gfx_acquire_indirect`, and `ez_gfx_structured_acquire`.
4. For each presented frame, call `ez_gfx_begin_render(surface, context)`; it begins the frame and clears prior frame state. For a non-presented or headless frame, call `ez_gfx_frame_begin(context)` instead. Then record work with `ez_gfx_indirect_write_draw`, `ez_gfx_indirect_set_draw_count`, `ez_gfx_render_add_vertex_pipeline`, `ez_gfx_render_add_compute_pipeline`, and, when needed, `ez_gfx_graph_enqueue_texture_readback`.
5. Complete an `ez_gfx_begin_render` frame with `ez_gfx_finish_render`, which submits and presents it. `ez_gfx_frame_submit` submits either flow without presentation, as implementation permits. A readback is available only after `ez_gfx_graph_enqueue_texture_readback` was recorded and the submitted work completed; then query and copy it with `ez_gfx_frame_readback`.
6. Poll `ez_gfx_poll_runtime_event` and `ez_gfx_poll_diagnostic` from host-owned output storage. Inspect `out_present`; when it is zero there is no record, and always account for `out_dropped`.
7. On the creator thread, release resources with `ez_gfx_shader_destroy`, `ez_gfx_texture_unload`, `ez_gfx_indirect_release`, and `ez_gfx_structured_release`; destroy named heaps with `ez_gfx_vertex_heap_destroy` and `ez_gfx_index_heap_destroy`; destroy the surface with `ez_gfx_surface_destroy`; then destroy the context with `ez_gfx_context_destroy`. Use `ez_gfx_context_wait_idle` before teardown when the host needs an idle context.

`ez_gfx_surface_resize`, `ez_gfx_surface_get_extent`, `ez_gfx_surface_resize_pending`, and `ez_gfx_surface_set_snapshot_cache` are surface operations between creation and destruction. A minimized surface can produce `EzGfxResult_NotReady`. Texture binding and residency queries use `ez_gfx_texture_get_binding` and `ez_gfx_texture_get_residency` and may report not-ready state through the normal result contract.

## Boundary rules

Every pointer must be non-null where its declaration or operation requires it, correctly aligned for its pointed-to type, and readable or writable for the complete duration of that call. Caller-owned descriptors, arrays, output values, artifacts, upload data, push constants, and readback storage are borrowed; the host must not free or mutate them until the call returns. The ABI does not retain those pointers.

All `const char *` values are UTF-8, NUL-terminated, non-empty strings for the call. Optional strings may be null only where the header marks them optional (for example, the texture debug label). Shader-entry arrays and binding arrays must remain valid for the call; binding arrays contain at most 16 entries and each entry names exactly one nonzero typed resource handle. Push-constant data is borrowed for the call, at most 128 bytes, and its size must be a multiple of four.

Pointer-plus-length byte ranges must describe the complete readable or writable range. Caller-provided byte ranges and bounded strings are capped at 16 MiB; oversized, overflowing, zero-sized, or otherwise invalid ranges are rejected according to the individual declaration. In particular, shader artifacts and texture data must be non-empty and at most 16 MiB; vertex/index uploads must be non-empty and within the cap; structured writes require a non-null data pointer even for zero bytes; and a zero-capacity frame-readback call is the size query and may omit its data pointer.

Destruction and release functions return no status. They catch panics and cannot tell the caller that a nonzero handle was invalid, stale, wrong-context, wrong-kind, or used from the wrong thread. Treat lifecycle order, ownership, handle provenance, and creator-thread affinity as the host's responsibility. All status-returning exports contain Rust panics at the FFI boundary and convert an unwinding operation to `EzGfxResult_NativeFailure`; void exports contain the panic but provide no failure result.

`ez_gfx_texture_load`, `ez_gfx_texture_get_binding`, and `ez_gfx_texture_get_residency` return `EzGfxResult`; `ez_gfx_texture_unload` is void. Texture loading can additionally surface the ordinary native, argument, context, and readiness outcomes defined there.

Shader loading consumes a compiler-produced artifact; this runtime does not compile shader source. The host owns and supplies that artifact memory for the call. Native window-system objects remain host-owned, while native graphics resources created from successful calls remain runtime-owned and are released by the matching ABI operation. Runtime progress and diagnostics are bounded queues: the host must poll both functions and use the reported dropped count to detect lost records rather than assuming that every event was delivered.
