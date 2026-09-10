# ez-gfx-ffi

`ez-gfx-ffi` is the C ABI boundary for the `ez-gfx` runtime. C and other foreign-language clients include [`bindings/c/include/ez_gfx_api.h`](../../bindings/c/include/ez_gfx_api.h). Rust declarations and documentation are authoritative; `tools/bindgen` generates the XML contract and C header. Rust clients should depend on `ez-gfx`, not `ez-gfx-ffi`.

The complete [C textured cube](../../examples/02_textured_cube_c/README.md) exercises ABI 40 portable GLFW window handles, presentation-mode selection, typed heap/allocation handles, one-frame compute-to-graphics buffers, creator-thread callbacks, resource diagnostics, stable error printing, and snapshot readback.

## Compatibility and ownership

Before any other call, require `ez_gfx_abi_version() == EZ_GFX_ABI_VERSION` (ABI 40). Window and headless creation use separate descriptors. `EzGfxWindowSurfaceDesc` carries a validated native-window-system tag and handles but no extent; native code queries the drawable size. `EzGfxHeadlessSurfaceDesc` alone carries an explicit extent. Operations use canonical `ez_gfx_{object}_{operation}` names and context-first signatures.

`ez_gfx_handle_inspect` decodes a packed handle into its context/child slot and generation fields; it does not validate that the handle is live in a context. `ez_gfx_semantic_id` accepts an exact 1-to-255-byte canonical semantic name and writes its fixed 16-byte identifier. Semantic names are ASCII dot-separated identifiers: every non-empty segment starts with an ASCII letter and continues with ASCII letters, digits, or underscores. Empty segments, non-ASCII bytes, embedded NUL, and terminators included in the supplied length are invalid.

Context and resource operations, including status-returning teardown, are creator-thread-affine. Buffer handles are writable until first claimed by a frame; terminal frame paths invalidate claimed handles. Window hosts must outlive successful surface teardown. `TeardownAbandoned` means native work could not be proven complete, so borrowed host handles must remain alive process-long.

## Call order

`ez_gfx_compute_shader_load`, `ez_gfx_vertex_shader_load`, and `ez_gfx_fragment_shader_load` validate artifact bytes and select one exact UTF-8 entry-point name into a context-owned stage handle. `ez_gfx_frame_execute_compute` accepts only the compute handle; `ez_gfx_frame_execute_graphics` accepts the vertex and fragment handles separately.

Frame execution retains each referenced shader record through `frame_end` or `frame_abort`; destroying the caller's stage handle invalidates it immediately but defers backend cleanup until that terminal operation.

1. Create a context with `ez_gfx_context_create` or `ez_gfx_context_create_backend`.
2. Create a presentation surface with `ez_gfx_surface_create_window`, or a windowless surface with `ez_gfx_surface_create_headless`, then initialize its device with `ez_gfx_context_init_device`.
3. Create/load persistent resources. Vertex heaps auto-grow; vertex upload and heap destruction require the typed heap. The context lazily owns its index heap. Geometry allocation handles retain their heap owner internally. Acquire and populate one-frame `EzGfxBuffer` and `EzGfxCounterBuffer` values through their owning context.
4. After successful device initialization, query surface support with `ez_gfx_surface_get_presentation_modes`, then begin a surface frame with `ez_gfx_frame_begin(context, surface, presentation_mode, out_frame)`. A pre-initialization query returns `EzGfxResult_NotReady`; valid unsupported modes use the documented deterministic fallback. Begin offscreen work with `ez_gfx_render_target_frame_begin(context, target, out_frame)`. Pass that frame to recording operations; the first binding claims each buffer.
5. Consume the frame with `ez_gfx_frame_end`; it submits and presents surface frames. On early exit, consume it with `ez_gfx_frame_abort`. Terminal calls invalidate the frame and claimed buffers even when submission, presentation, or abort reports an error.
6. Release unconsumed C buffers and persistent C resources explicitly. Remove live geometry allocations before destroying typed heaps. Destroying the context aborts any remaining descendant frame, then tears down on the creator thread.

Texture loading copies caller bytes and schedules unbounded CPU work subject to real allocation failure. `ez_gfx_context_register_callback` delivers source ownership transfer, device readiness, cancellation, terminal failure, runtime diagnostics, dropped-count reports, and borrowed readback bytes at creator-thread graphics safe points. Each requested readback returns a process-unique correlator and reports its texture handle and exact extent; callback bytes remain valid only for that invocation. Custom decoder callbacks receive one exact borrowed source range and return owned decoded mip descriptions; malformed outputs are rejected and every accepted source is released exactly once.

## Boundary rules

Every pointer must be non-null where its declaration or operation requires it, correctly aligned for its pointed-to type, and readable or writable for the complete duration of that call. Caller-owned descriptors, arrays, output values, artifacts, upload data, push constants, and readback storage are borrowed; the host must not free or mutate them until the call returns. Except for registered decoder callbacks and `user_data`, the ABI does not retain those pointers.

Every `const char *` input has an adjacent `size_t` byte length. The pointer denotes exactly that many UTF-8 bytes, excluding and not requiring a terminator; embedded NUL is invalid. Required strings accept only a non-null pointer and a length in `1..=16 MiB`. Optional strings accept only null plus zero, or a non-null pointer and a length in that same nonzero range. This contract also applies independently to every string field nested in shader, texture, and binding structures. Binding arrays contain at most 16 entries and each entry names exactly one nonzero typed resource handle. Push-constant data is borrowed for the call, at most 128 bytes, and its size must be a multiple of four.

Pointer-plus-count ranges must describe the complete range and stay within 16 MiB. Buffer acquire validates nonzero stride/count; buffer write validates start/count/stride and permits null data only for zero elements. Counter-buffer batch write likewise permits null commands only for zero count and rejects checked end overflow. Vertex/index uploads and artifacts remain nonempty and capped.

Void destruction exports contain panics but cannot report stale, foreign, wrong-kind, or wrong-thread handles. Status-returning exports validate their complete boundary, convert the safe facade `Error` explicitly, and map contained panics to `EzGfxResult_NativeFailure`. `ez_gfx_error_print` uses caller-owned storage: null+zero queries the required NUL-inclusive size, sufficient storage receives stable UTF-8 plus NUL, and insufficient storage is rejected without modifying the buffer.

The upload-event queue is lossless and unbounded; creator-thread graphics safe points deliver it through the registered callback. Runtime progress and diagnostic queues remain bounded and report dropped counts through that callback. `QueueFull` is retained for genuine counter/channel failure, not routine texture or transfer admission.
