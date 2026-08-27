# P-004: GPU memory allocation and suballocation

## Problem

Decide how to integrate a Rust-native GPU memory allocation approach to replace the C-based Vulkan Memory Allocator (VMA), evaluating the user's requested `memory-allocator` crate, `gpu-allocator`, or HAL-native suballocators across Vulkan, DX12, and Metal.

## Prompt context

Full user prompt: "use the memory-allocator rust crate instead of vma."
Source evidence: Original Odin code (`src/defs.odin`, `src/buffer.odin`, `src/render_target.odin`) linked `vendor/odin-vma` directly for device buffers, staging memory, and render targets.

## Constraints and acceptance criteria

- Eliminate C/C++ VMA dependencies in favor of a pure/idiomatic Rust memory allocation approach.
- Support dedicated and suballocated GPU memory blocks for buffers, render targets, textures, and mapped staging buffers.
- Provide GPU memory allocation compatible with Vulkan, DX12, and Metal.
- Explicit non-goals: writing a custom low-level GPU kernel page-table driver from scratch.

## Dependencies

- Incoming dependency: `P-004` depends on `P-003` for backend memory type discovery and device handles.
- Outgoing dependency: `P-009`, `P-011`, `P-012`, `P-014` depend on `P-004` for buffer and image suballocations.

## Unresolved questions

- Does the Vulkano `memory-allocator` crate support DX12/Metal or is `gpu-allocator` required for cross-backend portability?
- How should memory aliasing for transient render targets be exposed across the chosen allocation layer?

## Candidate solutions

### S-P-004-literal-memory-allocator

#### Architecture, integration, and applicability

Treat the requested crate identity literally and block integration until an exact package/repository is identified. Registry/docs queries found no published crate named exactly `memory-allocator`. Vulkano instead exposes `vulkano::memory::allocator` as a module inside the `vulkano` crate; its allocators operate on Vulkan `DeviceMemory`, so it cannot directly allocate D3D12 heaps or Metal heaps.

#### Evidence, tradeoffs, and failure modes

Vulkano documents buddy, bump, and free-list suballocators and Vulkan-specific memory requirements, fragmentation, and buffer-image granularity. No official benchmark was found. The exact requested package remains ambiguous; silently substituting another package would violate the prompt. Vulkano's module is disqualified for a uniform Vulkan/DX12/Metal allocator. The unresolved identity is a hard input to later selection, not permission to rename another crate.

#### Sources

- [Vulkano memory allocator module](https://docs.rs/vulkano/latest/vulkano/memory/allocator/)
- [Vulkano `MemoryAllocator` trait](https://docs.rs/vulkano/latest/vulkano/memory/allocator/trait.MemoryAllocator.html)
- Exact-name checks: `https://crates.io/crates/memory-allocator` and `https://docs.rs/memory-allocator` returned no package during this exploration.

### S-P-004-gpu-allocator-cross-backend

#### Architecture, integration, and applicability

Use `gpu-allocator` behind the P-003 allocation interface. Version 0.28 documents Vulkan, D3D12, and Metal modules: Vulkan binds device memory/offsets, D3D12 creates heaps and placed resources from allocation information, and Metal creates resources in `MTLHeap` allocations. Preserve allocation class, alignment, mapping, dedicated/managed choice, and retirement tokens so P-009 can request aliasable transient storage.

#### Evidence, tradeoffs, and failure modes

The crate documents free-list and dedicated block allocators but publishes no workload benchmark for allocation latency, fragmentation, VRAM overhead, or contention; all are unknown until measured with resource-size/lifetime traces on each backend. It now supports Metal, so a separate Metal allocator is not inherently required. Failures include wrong linear/nonlinear classification, mapping/coherency errors, heap-type mismatch, premature free, alignment/granularity violations, and aliasing unsupported by the wrapper path. This is a deliberate name deviation and is disqualified if the user's exact crate name is literal.

#### Sources

- [`gpu-allocator` 0.28 documentation and three-backend examples](https://docs.rs/gpu-allocator/latest/gpu_allocator/)
- [`gpu-allocator` source](https://github.com/Traverse-Research/gpu-allocator)
- [Metal heaps](https://developer.apple.com/documentation/metal/mtlheap)
- Original `src/buffer.odin`, `src/render_target.odin`, and TODO aliasing/staging entries.

### S-P-004-native-per-backend-allocators

#### Architecture, integration, and applicability

Implement the allocation policy inside each raw backend: Vulkan memory blocks/suballocators, D3D12 heaps/placed resources, and Metal heaps. A shared request/result vocabulary carries size, alignment, memory location, resource class, mapping, and alias lifetime, while algorithms may differ per API.

#### Evidence, tradeoffs, and failure modes

This exposes every aliasing primitive and avoids third-party constraints, but recreates difficult fragmentation, budget, residency, and synchronization logic three times. Native specifications define requirements, not allocator performance; latency, memory waste, and scaling remain unknown. Incorrect buffer-image granularity, D3D12 heap flags/alignment, Metal storage/cache modes, or residency behavior can corrupt data or exhaust memory. Maintenance burden can disqualify it.

#### Sources

- [Vulkan memory allocation](https://docs.vulkan.org/spec/latest/chapters/memory.html)
- [D3D12 placed resources](https://learn.microsoft.com/en-us/windows/win32/direct3d12/placed-resources)
- [Metal resource heaps](https://developer.apple.com/documentation/metal/resource_fundamentals/implementing_a_memory_pool_using_heaps)

## Performance comparison

No allocator candidate has comparable latency, fragmentation, VRAM, or contention measurements for the expected traces. The exact-name ambiguity is a blocking prompt-interpretation issue.

| Rank | Candidate | Hard constraints | Backend coverage | Allocation/runtime evidence | Reliability and implementation cost | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (conditional) | `gpu-allocator` 0.28 | Passes pure-Rust replacement and capabilities; may fail literal crate-name wording | Vulkan, DX12, Metal documented | Algorithms documented; workload performance unknown | One maintained abstraction; alias path still must be proven | Blocked on identity confirmation |
| — | Literal `memory-allocator` / Vulkano module | Exact package not verified; Vulkano interpretation fails three-backend requirement | Vulkan only if Vulkano intended | No applicable benchmark | Cannot satisfy unified backend need | Hard failure as currently identified |
| 2 | Native per-backend allocators | Passes backend and Rust constraints, but conflicts with preference for an allocator crate | Vulkan, DX12, Metal | Unknown | Highest fragmentation/residency/correctness burden | Survives only as fallback |

## Selected solution

**Provisional selection, blocked: `S-P-004-gpu-allocator-cross-backend`.**

`gpu-allocator` 0.28 is the only researched Rust crate with documented Vulkan, D3D12, and Metal allocator modules and the needed managed/dedicated allocation shape. However, it is not named `memory-allocator`. Do not add the dependency until the exact user wording is validated. If the user meant a specific package, that package must be evaluated by URL and version before this decision becomes final.

**Rejected:** Vulkano's `memory::allocator` module is Vulkan-only and therefore fails a hard requirement. Native allocators survive only if the requested crate cannot satisfy all backends or transient aliasing; their much larger correctness and maintenance surface ranks them lower.

**Blocking assumption:** “memory-allocator rust crate” describes the desired Rust GPU-allocation role and may refer to `gpu-allocator`, rather than an exact published package name. If false, the provisional selection is invalid.

**Risks:** `gpu-allocator` aliasing support may not expose every P-009 lifetime reuse operation; performance and fragmentation are unknown; version/platform support can change.

**Validation:** obtain the exact package URL/name from the user or prompt provenance; verify its current crate metadata and Vulkan/DX12/Metal modules. Then replay representative buffer/image/staging/transient traces, recording allocation latency distributions, peak committed/used bytes, fragmentation, mapped coherency, contention, and alias correctness on all three backends.
