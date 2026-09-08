# ez-gfx-ffi

`ez-gfx-ffi` is the C ABI boundary for the `ez-gfx` runtime. C and other foreign-language clients must include [`include/ez_gfx_api.h`](../../include/ez_gfx_api.h); the declarations and numeric values in that header are canonical. Rust clients should depend on `ez-gfx`, not `ez-gfx-ffi`.

The complete [C textured cube](../../examples/c/textured_cube/README.md) exercises ABI 31 typed heap/allocation handles, transient compute-to-graphics buffers, presentation, stable error printing, and snapshot readback on Win32.

## Compatibility and ownership

Before any other call, require `ez_gfx_abi_version() == EZ_GFX_ABI_VERSION` (ABI 31). Version 31 adds explicit generation-, kind-, owner-, state-, and frame-serial-validated `EzGfxFrame` handles and makes recording and transient operations frame-scoped. Version 30 added typed vertex heaps, batched indirect writes, and frame-transient structured/indirect buffers; version 29 made `EzGfxResult` C-ABI-only and added `ez_gfx_print_error`; version 28 added typed geometry allocations and upload events.

`ez_gfx_handle_inspect` decodes a packed handle into its context/child slot and generation fields; it does not validate that the handle is live in a context. `ez_gfx_semantic_id` accepts an exact 1-to-255-byte canonical semantic name and writes its fixed 16-byte identifier. Semantic names are ASCII dot-separated identifiers: every non-empty segment starts with an ASCII letter and continues with ASCII letters, digits, or underscores. Empty segments, non-ASCII bytes, embedded NUL, and terminators included in the supplied length are invalid.

The context and all context/resource operations, including teardown, are creator-thread-affine in the delegated runtime. Frame handles are also thread-local: every recording or transient operation requires its live frame, and `ez_gfx_frame_end` or `ez_gfx_frame_abort` consumes it on every result. Context destruction aborts its live descendant frames before teardown. The host owns native surface pointers until surface destruction; the runtime owns created graphics resources and native objects.

## Call order

`ez_gfx_shader_load_artifact(data, size, out_shader, context)` loads every stage present for the context backend/profile. Entry-point names remain artifact metadata and are not caller inputs.

1. Create a context with `ez_gfx_context_create` or `ez_gfx_context_create_backend`.
2. Create a presentation surface with `ez_gfx_surface_create`, then initialize its device with `ez_gfx_context_init_device`.
3. Create/load persistent resources. Heap creation returns `EzGfxVertexHeap`; vertex upload and heap destruction require it. Geometry allocation handles retain their heap owner internally.
4. Begin a surface frame with `ez_gfx_frame_begin(context, surface, out_frame)` or an offscreen frame with `ez_gfx_render_target_frame_begin(context, target, out_frame)`. Pass that frame to every recording and transient-buffer operation.
5. Consume the frame with `ez_gfx_frame_end`; it submits and presents surface frames. On early exit, consume it with `ez_gfx_frame_abort`. Terminal calls invalidate the frame even when submission, presentation, or abort reports an error.
6. Remove live geometry allocations before destroying typed heaps. Destroying the context aborts any remaining descendant frame, then tears down on the creator thread.

Texture loading copies caller bytes and schedules unbounded CPU work subject to real allocation failure. `ez_gfx_poll_upload_event` reports source ownership transfer, device readiness, cancellation, and terminal failure. Custom decoder callbacks may execute concurrently; successful output remains valid until ez-gfx copies it and calls the paired release callback.

## Boundary rules

Every pointer must be non-null where its declaration or operation requires it, correctly aligned for its pointed-to type, and readable or writable for the complete duration of that call. Caller-owned descriptors, arrays, output values, artifacts, upload data, push constants, and readback storage are borrowed; the host must not free or mutate them until the call returns. Except for registered decoder callbacks and `user_data`, the ABI does not retain those pointers.

Every `const char *` input has an adjacent `size_t` byte length. The pointer denotes exactly that many UTF-8 bytes, excluding and not requiring a terminator; embedded NUL is invalid. Required strings accept only a non-null pointer and a length in `1..=16 MiB`. Optional strings accept only null plus zero, or a non-null pointer and a length in that same nonzero range. This contract also applies independently to every string field nested in shader, texture, and binding structures. Binding arrays contain at most 16 entries and each entry names exactly one nonzero typed resource handle. Push-constant data is borrowed for the call, at most 128 bytes, and its size must be a multiple of four.

Pointer-plus-count ranges must describe the complete range and stay within 16 MiB. Structured acquire validates nonzero stride/count; structured write validates start/count/stride and permits null data only for zero elements. Indirect batch write likewise permits null commands only for zero count and rejects checked end overflow. Vertex/index uploads and artifacts remain nonempty and capped.

Void destruction exports contain panics but cannot report stale, foreign, wrong-kind, or wrong-thread handles. Status-returning exports validate their complete boundary, convert the safe facade `Error` explicitly, and map contained panics to `EzGfxResult_NativeFailure`. `ez_gfx_print_error` uses caller-owned storage: null+zero queries the required NUL-inclusive size, sufficient storage receives stable UTF-8 plus NUL, and insufficient storage is rejected without modifying the buffer.

The upload-event queue is lossless and unbounded; drain it until empty once per frame. Runtime progress and diagnostic queues remain bounded and report dropped counts. `QueueFull` is retained for genuine counter/channel failure, not routine texture or transfer admission.
