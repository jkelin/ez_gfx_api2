# P-002: Public API and C ABI bindings

## Decision

The safe Rust interface is the authoritative ownership seam:

- One owning `Context` controls the graphics lifetime. Resource wrappers carry non-owning context access and generation-checked identities; destroying or dropping the context invalidates all descendants and tears down native state.
- `Surface`, shader, render-target, geometry-heap, and geometry-allocation `Drop` paths may release their individual identity early. Texture wrapper drop is intentionally inert: the context-owned bindless heap retains texture identity and storage until context teardown.
- `Surface::begin_frame()` and `Context::begin_frame()` return target-less owners; `Frame::configure_swapchain` or `Frame::configure_render_target` attaches one logical target. Named target images are cached and recreated implicitly for extent/format changes. Recording requires `&mut Frame`; `Frame::finish(self)` preserves the exact failed phase.
- Dropping an unfinished `Frame` aborts it. Context-acquired `Buffer<T>`, `CounterBuffer<T>`, and single-value `ValueBuffer<T>` values are writable until their first frame claim, reusable within that frame, then invalid after every terminal path. `Frame::bind_buffer` adds or replaces one named frame binding; `execute_compute` and `execute_graphics` read the current set without removing entries. `RenderTarget::prepare_readback` creates an opaque request whose result is callback-scoped.
- Window surface creation accepts `HasWindowHandle`, dispatches from `RawWindowHandle`, and queries the initial drawable extent. Headless creation is a separate explicit-extent path. Any failure destroys the unpublished raw surface.

The stable C interface remains a narrow validated adapter in `ez-gfx-ffi`, not a second implementation. ABI 37 uses fixed-width layouts and opaque generational `u64` handles, removes platform fields, and separates window from headless surface creation. `ez_gfx_frame_begin` and `ez_gfx_render_target_frame_begin` return owner-validated `EzGfxFrame` handles; C completes them with `ez_gfx_frame_end` or `ez_gfx_frame_abort`. Every result invalidates the frame and its claimed buffers. ABI 37 C parity covers `ez_gfx_value_buffer_acquire`, `ez_gfx_frame_bind` persistent frame-local bindings with same-name replacement, and `ez_gfx_frame_execute_compute`/`graphics` reuse, with no push-constant arguments or legacy add exports. Resource destruction remains explicit in C because the ABI cannot rely on Rust destructors.

Every stable string carries an adjacent explicit byte length. Required strings are non-null and nonzero; optional strings are null+zero or non-null+nonzero. All ranges are capped, valid UTF-8 where required, and contain no embedded NUL. The FFI validates pointer/count pairs, alignment, multiplication and address ranges, enum/layout values, handle kind/owner/generation, out-pointers, and asynchronous ownership before delegation. Typed Rust errors map to fixed C statuses, and no panic unwinds across the ABI.

## Interface rationale

The raw explicit lifecycle exists only at the foreign-language seam where deterministic destructors are unavailable. The clean cutover intentionally provides no safe compatibility aliases for the former handle-plus-free interface.

`Rc` records that the facade is context-affine rather than thread-safe. The owning `Context` may invalidate every descendant regardless of outstanding wrappers; those wrappers retain memory safety but no independent native lifetime.

## Risks and validation

Risks are accidental reference cycles, stale wrappers after context destruction, stale/cross-context C handles, leaked aborted frames, loss of the original submission/presentation error, enum/layout drift, invalid foreign memory, and panic mode. Validate context-owner destruction with live descendants, exact completion errors, implicit frame abort, one-frame buffer claim/reuse/invalidation, atomic surface rollback, stale/foreign/generation rejection, ABI 37 generated-contract freshness, layout probes, invalid calls, panic containment, and Windows C-example build/link.

Bulk slices need not copy, but call, validation, reference-count, handle-check, allocation, and submission costs remain unmeasured. Measure them separately with the compiler, CPU, workload, and backend recorded.
