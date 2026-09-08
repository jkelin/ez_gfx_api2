# P-011: Vertex and index geometry heaps

## Problem

Decide the internal architecture of the bindless vertex manager (named vertex heaps) and global index heap, incorporating free-list tracking with double-free safety diagnostics and adding a zero-copy direct staging-write upload API for procedurally generated geometry.

## Prompt context

Source evidence: `TODO.md` ("Add a direct staging-write upload API for generated vertex data. The current async API copies caller memory into a staging buffer... callers that procedurally generate vertices could avoid one CPU-side copy by writing directly into a mapped staging allocation", "Add stronger Vertex manager allocation ownership checks if async upload failures and user frees start interacting with range reuse").

## Constraints and acceptance criteria

- Manage named bindless vertex heaps and unified index heap with stride and capacity validation.
- Implement robust free-list allocation tracking that detects double-frees, stale handles, and invalid range frees.
- Expose a direct staging-write API returning a mapped slice/pointer for zero-copy CPU vertex generation.
- Explicit non-goals: CPU vertex layout reformatting or automatic mesh optimization inside the API.

## Dependencies

- Incoming dependency: `P-011` depends on `P-004` (Allocator) for GPU buffer allocations.
- Outgoing dependency: `P-012` depends on `P-011` for geometry staging and transfer batches.

## Unresolved questions

- How should dynamic heap expansion/growth be handled when an allocation exceeds initial capacity?
- How to track generation/version tokens per allocation handle to guarantee O(1) double-free detection?

## Candidate solutions

### S-P-011-generation-freelist-mapped-ring: Generation-indexed free-list allocator with mapped staging lease

#### Approach and integration

Maintain each named vertex heap as a GPU-resident storage buffer with an internal generation-indexed slot map and an ordered range free-list. For procedural mesh generation, expose `acquire_staging_vertex_lease(heap, element_count)` which reserves a mapped slice in a host-visible ring buffer and returns a lease guard. The lease is committed or cancelled explicitly, exactly once; dropping an uncommitted lease cancels it and must never submit partially initialized data, so the GPU transfer command covers only committed bytes without any intermediate CPU-to-CPU buffer copy. Handle frees validate the generation token against the slot table, guaranteeing deterministic detection of double-frees or stale handle releases.

#### Performance evidence

- **CPU Overhead:** Eliminates an intermediate CPU memory allocation and buffer copy (`memcpy`) for procedural meshes by writing directly into mapped staging memory (`[INFERENCE]` from eliminating redundant buffer copies). Exact latency impact depends on host memory bandwidth and vertex count.
- **Handle Validation:** O(1) slot array lookup compared to linear search over live allocations (`[INFERENCE]` from indexed array access characteristics).

#### Tradeoffs and failure modes

- **Tradeoffs:** Requires managing ring-buffer head/tail pointers and tracking GPU timeline semaphores for staging buffer reclamation.
- **Failure Modes:** If a caller retains a staging write lease across frame boundaries without committing, staging ring buffer space may stall subsequent allocations.

#### Sources

- [Vulkan Dynamic State and Staging Guidance](https://docs.vulkan.org/guide/latest/vertex_input_data_processing.html) — host-visible staging buffer design.
- `F:/Projects/oss/ez_gfx_api/src/vertex_manager.odin` — original Odin vertex manager implementation.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — direct staging writes and allocation ownership safety.

### S-P-011-offset-freelist-copy-staging: Incumbent raw offset free-list with caller-buffer copying

#### Approach and integration

Retain the original pattern: caller provides a pre-filled CPU slice (`&[u8]`). The engine allocates a dedicated or pooled staging buffer, executes `memcpy` from user memory into staging memory, and enqueues a transfer copy. Allocation handles store raw byte offsets and lengths without generation indices.

#### Performance evidence

- **CPU Overhead:** Incurs two CPU memory copies (caller buffer generation + `memcpy` into staging buffer) before GPU transfer.
- **Diagnostics:** Free operations trust caller handles; double-freeing corrupts the internal free-list without immediate panic (`[OBSERVED]` in source `src/vertex_manager.odin`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Simple calling convention requiring no lifetime lease objects.
- **Failure Modes:** Silent memory corruption on double-frees or out-of-order async failure recoveries; high memory bandwidth overhead for dynamic procedural meshes.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/vertex_manager.odin` — incumbent implementation.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — free-list diagnostic issues.
- [Vulkan Memory Allocation Guide](https://gpuopen.com/learn/vulkan-memory-management/) — allocation and suballocation considerations.

### S-P-011-buddy-allocator-persistent-staging: Binary buddy allocator with persistent mapped host chunks

#### Approach and integration

Implement a binary buddy suballocator (power-of-two blocks) for GPU heap space, backed by persistent per-thread host-visible memory pools. Generates 64-bit packed handles combining block index, size class, and generation counter.

#### Performance evidence

- **Memory Efficiency:** Fast allocation/deallocation O(log N), but introduces internal fragmentation on irregular mesh sizes (`[INFERENCE]` based on standard buddy allocator properties).

#### Tradeoffs and failure modes

- **Tradeoffs:** Fast coalesce logic, but significant internal fragmentation waste for non-power-of-two vertex buffers.
- **Failure Modes:** Premature heap exhaustion due to fragmentation on diverse mesh element counts.

#### Sources

- [GPU Open Memory Allocation Best Practices](https://gpuopen.com/learn/vulkan-device-memory/) — suballocation and fragmentation analysis.
- [Vulkan Memory Allocation Guide](https://gpuopen.com/learn/vulkan-memory-management/) — buddy and suballocation tradeoffs.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| --------- | ------------------------------- | -------------- | ---------------- | ----- |
| Generation-indexed free-list with mapped staging lease | Eliminates procedural CPU copy, O(1) double-free detection, zero internal fragmentation | High | Direct source evidence & memory architecture analysis | Ring buffer lease stalling if held across frames |
| Incumbent raw offset free-list with caller copying | Redundant CPU copy, unsafe on double-free | Low-Moderate | Direct source code inspection | Free-list corruption under concurrent async frees |
| Binary buddy allocator with persistent host chunks | O(log N) alloc/free, internal fragmentation on irregular sizes | Moderate | Memory architecture literature | VRAM waste on irregular mesh vertex counts |

## Selected solution

### Selection

`S-P-011-generation-freelist-mapped-ring`: Generation-indexed free-list allocator with mapped staging lease.

### Selection rationale

The generation-checked range allocator portion is implemented. Named vertex heaps and the global index heap return typed owner-validated allocation handles, coalesce removed ranges, reject stale/double/wrong-heap frees, and bind automatically from `[VertexHeap(\"name\")]` reflection on Vulkan, Direct3D 12, and Metal. The index heap remains a native index binding.

Removal currently waits for native idle before returning a range to the free list. This is safe but more conservative than token-retired reuse.
Frame readiness is tracked per heap maximum: that conservative gate remains library safety, while applications decide per-allocation visibility from lossless `DeviceReady` upload events. The forthcoming shared `FrameBeginConfig` keeps this aggregate geometry wait by default and lets event-driven applications set `wait_for_geometry_uploads = false`; its independent `TextureMipWait::None`, `TextureMipWait::Coarsest`, and `TextureMipWait::ThroughLevel(level)` variants govern texture waits. No per-allocation graph-wait optimization is tracked.

The mapped staging lease is not implemented. Slice uploads copy caller bytes directly into runtime-owned mapped staging and emit `SourceStaged`; plans and docs must not claim zero-copy procedural generation.

### Remaining work

- expose a lifetime-safe direct mapped staging lease for Rust and C;
- retire removed ranges against fine-grained graphics completion rather than full context idle;

### Validation

Allocator tests cover coalescing, stale/double/wrong-heap frees, generation identity, and unbounded staging-slot growth. Examples exercise typed heap handles, reflected semantic lookup, queried indirect offsets, and per-frame transient buffers.

Evidence on 2026-09-08: workspace nextest 555/555 plus doc tests, strict workspace all-target/all-feature Clippy, formatting, source-line limits, ABI 30 export/layout parity, C11/C++17 header checks, the MSVC C build, hidden examples 32/32, focused ABI 29/29, error 2/2, and Vulkan/DX12 PSO 2/2 passed. The 60-second per-case backend matrices passed on local Windows (HAL/Vulkan/DX12/safe facade 94/94), Linux Vulkan (73/73), and macOS Metal (51/51).
