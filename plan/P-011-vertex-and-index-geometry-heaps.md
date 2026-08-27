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
- Outgoing dependency: `P-012` and `P-013` depend on `P-011` for staging buffers and upload dependency tracking.

## Unresolved questions

- How should dynamic heap expansion/growth be handled when an allocation exceeds initial capacity?
- How to track generation/version tokens per allocation handle to guarantee O(1) double-free detection?

## Candidate solutions

### S-P-011-generation-freelist-mapped-ring: Generation-indexed free-list allocator with mapped staging lease

#### Approach and integration

Maintain each named vertex heap as a GPU-resident storage buffer with an internal generation-indexed slot map and an ordered range free-list. For procedural mesh generation, expose `acquire_staging_vertex_lease(heap, element_count)` which reserves a mapped slice in a host-visible ring buffer and returns a lease guard. Dropping or submitting the lease commits the GPU transfer command without any intermediate CPU-to-CPU buffer copy. Handle frees validate the generation token against the slot table, guaranteeing deterministic detection of double-frees or stale handle releases.

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

`S-P-011-generation-freelist-mapped-ring` directly satisfies all explicit constraints and both inherited TODO items:
1. It provides a zero-copy direct staging write lease API (`acquire_staging_vertex_lease`) for procedural mesh generation, avoiding redundant CPU-to-CPU intermediate buffer copies.
2. It pairs the free-list range allocator with a generation-tagged slot map, guaranteeing O(1) double-free and stale handle detection to eliminate the memory corruption risks documented in the original project.
3. Unlike buddy allocation, it eliminates internal power-of-two fragmentation on arbitrary mesh vertex counts.

### Rejected alternatives

- **`S-P-011-offset-freelist-copy-staging`**: Rejected because raw offset tracking fails to detect double-frees or stale range frees during asynchronous failure recovery, and requires redundant CPU-side buffer allocations for procedural geometry generation.
- **`S-P-011-buddy-allocator-persistent-staging`**: Rejected due to internal fragmentation waste on non-power-of-two vertex counts, which causes premature GPU memory exhaustion on complex geometry assets.

### Evidence summary

Eliminates redundant CPU memory copy steps for dynamic mesh uploads and guarantees deterministic O(1) slot validation against slot map generation counters (`[INFERENCE]` from indexed array characteristics and memory copy elimination).

### Key assumptions

- Named vertex heaps are bound via bindless storage buffer descriptors or vertex buffer bindings in shader reflection.
- Callers commit staging write leases within the frame they are acquired or return them on cancelation.

### Risks and mitigations

- **Risk:** Staging ring-buffer space exhaustion if procedural leases are held across multiple frames.
- **Mitigation:** Enforce frame-scoped lifetime bounds on lease guards and implement automatic lease retirement on frame boundary submission.

### Validation actions

1. Unit test free-list coalescing, double-free detection, and handle generation rollover.
2. Integration test procedural geometry generation using `acquire_staging_vertex_lease` in Example 1 (Triangle) and Example 2 (Textured Cube).
