# P-011: Vertex and index geometry heaps

## Problem

Define safe ownership for named vertex heaps, their allocations, the singleton context index heap, index allocations, and persistent context-owned structured/counted buffers. Preserve generation/owner validation at the C seam.

## Constraints

- The safe Rust interface uses owning wrappers and `Drop`, never public destroy, remove, release, or free functions.
- A vertex allocation retains its parent heap; every owning wrapper retains `Rc<ContextInner>`.
- The index heap is a singleton owned by `Context`; `IndexAllocation` retains the context.
- Structured and counted buffers are context-owned persistent CPU storage imported into frames on demand; mutation resumes after the importing frame terminates.
- ABI 32 retains explicit opaque-handle lifecycle functions and rejects stale, foreign, wrong-kind, and duplicate handles.

## Retired exploration

The original candidate analysis proposed a caller-writable mapped staging lease and compared free-list, copied-staging, and buddy allocation variants. That lease is not part of the implemented public interface, so its zero-copy and performance hypotheses are not current behavior. The existing slice upload path copies caller bytes into runtime-owned mapped staging. Future lease work remains recorded only in `TODO.md`.

The selected ordered range free list and generation-indexed identity remain. Fine-grained completion-token retirement is not yet implemented; dropped allocation ranges currently wait for native idle before reuse.

## Selected solution

Named auto-growing vertex heaps and one lazy context-owned index heap use ordered range free lists and generation-checked allocation identities. Safe heap and allocation wrappers retain `Rc<ContextInner>` plus their parent resource leases. Dropping an allocation retires its range; dropping a heap retires it after child leases and recorded uses. The safe interface exposes no remove, destroy, release, or free operations. ABI 32 retains explicit opaque-handle release and rejects stale, foreign, wrong-kind, and duplicate handles.

Frame readiness conservatively covers imported named vertex heaps and the singleton index heap. Applications schedule visible use from lossless `DeviceReady` events. `Buffer<T>` and `CountedBuffer<T>` belong to `Context`; a frame snapshots and retains native materializations while recording, and the same wrapper is reusable after terminal completion or abort.

Slice uploads copy caller bytes into runtime-owned mapped staging and emit `SourceStaged`. A direct caller-writable mapped staging lease is not implemented and remains tracked in `TODO.md`; no zero-copy claim is made.

Range retirement currently waits for native idle. Fine-grained completion-token retirement remains tracked in `TODO.md`.

### Validation

Cover singleton index-heap admission, range coalescing, parent-before-child drop, stale/foreign/duplicate C release, generation identity, upload failure, persistent-buffer mutation exclusion and reuse across frame completion/abort, and no native backing reuse before completion.
