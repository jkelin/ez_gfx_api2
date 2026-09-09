# P-002: Public API and C ABI bindings

## Decision

The safe Rust interface is the authoritative ownership seam:

- `Context` owns `Rc<ContextInner>`. Every owning resource wrapper retains a cloned context lease, so dropping the apparent parent before a child cannot invalidate the child.
- `Surface`, shader, texture, render-target, geometry-heap, and geometry-allocation wrappers own generation-checked resource leases. `Drop` performs the one release; the safe interface exposes no public destroy, release, or free functions.
- `Surface::begin_frame()` and `Context::begin_frame()` return target-less owners; `Frame::configure_swapchain` or `Frame::configure_render_target` attaches one logical target. Named target images are cached and recreated implicitly for extent/format changes. Recording requires `&mut Frame`; `Frame::finish(self)` preserves the exact failed phase.
- Dropping an unfinished `Frame` aborts it. Context-acquired `Buffer<T>` and `CounterBuffer<T>` values are writable until their first frame claim, reusable within that frame, then invalid after every terminal path. `RenderTarget::prepare_readback` creates an opaque request whose result is callback-scoped.
- Surface creation is atomic. If native creation, device initialization, or initial resize fails, construction destroys the unpublished raw surface and returns the original error without retaining a safe wrapper.

The stable C interface remains a narrow validated adapter in `ez-gfx-ffi`, not a second implementation. ABI 33 uses fixed-width layouts and opaque generational `u64` handles. `ez_gfx_frame_begin` and `ez_gfx_render_target_frame_begin` return owner-validated `EzGfxFrame` handles; C completes them with `ez_gfx_frame_end` or `ez_gfx_frame_abort`. Every result invalidates the frame and its claimed buffers. Resource destruction remains explicit in C because the language has no Rust `Drop`.

Every stable string carries an adjacent explicit byte length. Required strings are non-null and nonzero; optional strings are null+zero or non-null+nonzero. All ranges are capped, valid UTF-8 where required, and contain no embedded NUL. The FFI validates pointer/count pairs, alignment, multiplication and address ranges, enum/layout values, handle kind/owner/generation, out-pointers, and asynchronous ownership before delegation. Typed Rust errors map to fixed C statuses, and no panic unwinds across the ABI.

## Interface rationale

Owning wrappers keep the safe interface small: callers cannot separately coordinate context lifetime, resource release, frame abort, and transient invalidation. The raw explicit lifecycle exists only at the foreign-language seam where deterministic destructors are unavailable. The clean cutover intentionally provides no safe compatibility aliases for the former handle-plus-free interface.

`Rc` records that the safe facade is context-affine rather than thread-safe. Deferred native retirement remains internal: dropping a wrapper invalidates public identity immediately while completion tracking decides when backing storage can be reused.

## Risks and validation

Risks are accidental reference cycles, early or duplicate raw-handle release, stale/cross-context C handles, leaked aborted frames, loss of the original submission/presentation error, enum/layout drift, invalid foreign memory, and panic mode. Validate drop-order permutations, context retention, exact completion errors, implicit frame abort, one-frame buffer claim/reuse/invalidation, atomic surface rollback, stale/foreign/generation rejection, ABI 33 header/XML/export parity, layout probes, invalid calls, and panic containment.

Bulk slices need not copy, but call, validation, reference-count, handle-check, allocation, and submission costs remain unmeasured. Measure them separately with the compiler, CPU, workload, and backend recorded.
