# P-011: Vertex and index geometry heaps

## Problem

Define safe ownership for named vertex heaps, their allocations, the singleton context index heap, index allocations, and one-frame buffer/counter/value buffers. Preserve generation/owner validation at the C seam.

## Constraints

- The safe Rust interface uses owning wrappers and `Drop`, never public destroy, remove, release, or free functions.
- A vertex allocation retains its parent heap; every owning wrapper retains `Rc<ContextInner>`.
- The index heap is a singleton owned by `Context`; `IndexAllocation` retains the context.
- `Buffer`, `CounterBuffer`, and `ValueBuffer` are context-acquired one-frame values: first use claims them, same-frame reuse is valid, and terminal frame paths invalidate them.
- ABI 37 retains explicit opaque-handle lifecycle functions and rejects stale, foreign, wrong-kind, and duplicate handles.

## Retired exploration

The original candidate analysis proposed a caller-writable mapped staging lease and compared free-list, copied-staging, and buddy allocation variants. That lease is not part of the implemented public interface, so its zero-copy and performance hypotheses are not current behavior. The existing slice upload path copies caller bytes into runtime-owned mapped staging. Future lease work remains recorded only in `TODO.md`.

The selected ordered range free list and generation-indexed identity remain. Allocation drops during recording are conservatively held through that frame; completed frame tokens gate later reuse.

## Selected solution

Named auto-growing vertex heaps and one lazy context-owned index heap use ordered range free lists and generation-checked allocation identities. Allocation drop may retire its range before context teardown; a drop during recording waits for that frame's terminal completion, so no public per-frame retain method is needed. The context owner can invalidate and destroy the entire hierarchy regardless of live wrappers. ABI 37 retains explicit opaque-handle release and rejects stale, foreign, wrong-kind, and duplicate handles.

Frame readiness conservatively covers imported named vertex heaps and the singleton index heap. `Buffer<T>`, `CounterBuffer<T>`, and single-value `ValueBuffer<T>` are materialized once for their claiming frame, support repeated compute/graphics use there, and are invalid afterward. `[Buffer]` and `[CounterBuffer]` shader names reflect as `buffer` and `counter_buffer` binding kinds; counters expose `set_count`/`add_count`/`set`/`get` over count at byte 0 plus elements at shared HAL offset 256 with bytes 4..255 zeroed (252 padding bytes); Vulkan/DX12 read the count at byte 0 and commands at offset 256 while Metal encodes capacity over zeroed tails. Native storage returns to completion-gated backend pools.

Slice uploads copy caller bytes into runtime-owned mapped staging and emit `SourceStaged`. A direct caller-writable mapped staging lease is not implemented and remains tracked in `TODO.md`; no zero-copy claim is made.

Geometry range and physical-allocation retirement use transfer and graphics completion tokens; recording-time drops are stamped when that frame terminates.

### Validation

Cover singleton index-heap admission, range coalescing, parent-before-child drop, recording-time drop deferral, stale/foreign/duplicate C release, generation identity, upload failure, one-frame buffer/counter/value claim, draft replacement and consumption, same-frame reuse, and no native backing reuse before completion.
