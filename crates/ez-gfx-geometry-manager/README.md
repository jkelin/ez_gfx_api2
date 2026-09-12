# ez-gfx-geometry-manager

Backend-neutral geometry policy: named vertex heaps, the singleton index
heap, and per-allocation upload readiness. This crate never touches native
handles and never depends on a backend, the runtime, or `ez-gfx`.

## Architecture

`GeometryManager` tracks heap ranges (ordered free lists), live allocations
by generational handle, and readiness tokens. Staging reuse lives in the
shared HAL `ReusableStagingPool`; transfer admission shares the context's
one `ez-gfx-texture-manager::SharedTransferPool` so a single hardware
transfer queue sees one admission domain. `ez-gfx` owns lifetimes, native
allocation, and upload events; backends own device memory and copies.

## Interfaces

Create heaps (`create_vertex_heap` / index heap), allocate ranges, poll
readiness with transfer completion values, release ranges. All payload and
token types are HAL/core types. Errors are fail-fast typed values, never
silent defaults.

## Synchronization

Allocations report readiness only after their transfer token retires.
Dropping an allocation during recording invalidates its public identity
immediately but defers range reuse through that frame's terminal
completion. No path blocks a caller on transfer completion.

## Priorities and batching

Geometry bytes are background class in the shared pool: admitted only when
they fit, and they never block required texture work. Geometry uploads
record staging bytes until the transfer timeline retires them.

## Shared transfer and cache behavior

See `ez-gfx-texture-manager`: one pool, ledger separation, and
`ReclaimableStaging` eviction of the largest completed bucket across every
pool under the aggregate ceiling.

## Examples

```rust
manager.create_vertex_heap("positions", capacity, stride)?;
let upload = manager.reserve_vertices("positions", count, handle)?;
manager.mark_ready(handle, completion)?;
```

## Testing

Unit tests cover range allocation/reuse, readiness gating, and generation
validation. Native behavior is verified through the backend matrix.

## Backend adapter responsibilities

Backends own device heap storage and transfer submission, report
`CompletionToken` progress, and reuse staging through HAL pools. No
backend type crosses this crate's interface.
