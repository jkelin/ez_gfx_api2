# P-007: Pipeline caching and descriptor management

## Problem

Decide how to decouple graphics/compute pipeline caching from descriptor set/pool lifetimes, persist precompiled pipeline caches across application runs, and implement a unified bindless descriptor model across Vulkan, DX12, and Metal.

## Prompt context

Source evidence: `TODO.md` ("Decouple graphics pipeline caching from descriptor set/pool ownership. `Ez_Gfx_Pipeline_Record` still owns both `VkPipeline` and descriptor resources... per-frame descriptor lifetimes are still tied to cached pipeline records", "Add a shader cache using precompiled shader modules... persist Slang/SPIR-V/reflection artifacts across runs").

## Constraints and acceptance criteria

- Separate pipeline state object (PSO) caching from descriptor allocation and per-frame lifetime management.
- Provide a persistent pipeline cache on disk keyed by shader bytecode and attachment state.
- Support bindless descriptor heaps (textures, buffers) uniformly across backends.
- Explicit non-goals: forcing classic per-draw individual descriptor sets for static resources.

## Dependencies

- Incoming dependency: `P-007` depends on `P-003` (HAL) and `P-006` (Reflection).
- Outgoing dependency: `P-008` and `P-016` depend on `P-007` for pipeline binding and descriptor updates during execution.

## Unresolved questions

- How to handle dynamic render target format changes without leaking descriptor pool resources?
- What disk serialization format should be used for persistent pipeline caches?

## Candidate solutions

### S-P-007-global-bindless-table-independent-pso-cache

#### Architecture, integration, and applicability

Separate immutable/reflection-derived pipeline layouts and backend PSOs from a device-level bindless registry. Stable resource indices map to a Vulkan update-after-bind descriptor set/descriptor buffer, one D3D12 shader-visible resource heap plus sampler heap, and Metal argument-buffer arrays. Cache keys include shader/interface hash, attachment formats, fixed state, backend, driver/device identity, and cache schema; backend-native cache blobs are validated and invalidated independently of descriptor storage.

#### Evidence, tradeoffs, and failure modes

Official APIs expose large indexed descriptor tables, but limits and update semantics differ. D3D12 warns that switching shader-visible heaps may stall; Vulkan update-after-bind has separate limits and synchronization rules; Metal argument-buffer tiers differ. No repository measurement exists for descriptor updates, cache-hit startup, memory, or contention. Stable slots need generations and GPU-retirement tracking. Failures include updating in-use descriptors, slot reuse before completion, exceeding tier/heap limits, incompatible pipeline-cache blobs, and mixing sampler/resource namespaces. Hardware lacking required indexing capacity disqualifies this form.

#### Sources

- [Vulkan descriptor indexing](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_descriptor_indexing.html)
- [Vulkan descriptor sets and pool synchronization](https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html)
- [D3D12 descriptor heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview)
- [Metal argument buffers](https://developer.apple.com/documentation/metal/buffers/about_argument_buffers)
- Original `src/pipeline.odin` and `TODO.md` line 3.

### S-P-007-global-table-plus-frame-local-arenas

#### Architecture, integration, and applicability

Keep stable textures/buffers in a device-level bindless table, but allocate transient/dynamic tables from one linear arena per in-flight `FrameContext`. Reset an arena only after its fence/timeline completes. PSO records own neither pool nor sets. Vulkan uses a persistent update-after-bind set plus frame pools; D3D12 suballocates one long-lived shader-visible heap; Metal writes per-frame argument-buffer records.

#### Evidence, tradeoffs, and failure modes

Vulkan pool reset recycles all sets only after submitted uses finish. D3D12 explicitly presents bulk per-frame allocation/suballocation patterns and limits bound heap counts. This reduces fine-grained free operations by design; no measured latency or memory result exists. Capacity is an estimate from peak descriptors per frame times frames in flight and platform alignment. Pool exhaustion, early reset, D3D12 heap switches, Metal tier limits, and persistent/transient index confusion are failures.

#### Sources

- [Vulkan descriptor-pool reset/lifetime rules](https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html)
- [D3D12 shader-visible heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shader-visible-descriptor-heaps)
- [Apple GPU-encoded argument-buffer sample](https://developer.apple.com/documentation/metal/encoding-argument-buffers-on-the-gpu)

### S-P-007-incumbent-pipeline-owned-descriptors

#### Architecture, integration, and applicability

Retain descriptor layouts, pools, per-frame sets, and PSO in each cached pipeline record. This matches the existing Odin ownership model and is the simplest baseline.

#### Evidence, tradeoffs, and failure modes

The original source demonstrates functionality, not performance. Eviction and render-target/vertex rebinding couple unrelated lifetimes, duplicate pools, and can invalidate in-flight descriptors. It is disqualified by the explicit TODO to decouple these lifetimes.

#### Sources

- Original `src/defs.odin`, `src/pipeline.odin`, and `TODO.md` line 3.

## Performance comparison

No candidate has comparable descriptor-update, cache-startup, memory, or contention measurements. The incumbent fails an explicit TODO; surviving candidates are ranked by complete lifetime separation.

| Rank | Candidate | Hard constraints | Runtime/scaling | Reliability/operations | Implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Global table plus frame-local arenas and independent PSO cache | Passes bindless and descriptor/PSO separation | Stable indices plus linear transient allocation; capacity/performance unknown | Fence-gated reset; platform tier/limit risks | More lifetime classes, but explicit ownership | API lifetime patterns sourced |
| 2 | Global table with independent PSO cache only | Passes static bindless separation; transient policy incomplete | Stable-slot scaling unknown | Simpler global ownership; dynamic descriptors still need policy | Lower initial cost | API capabilities sourced |
| — | Pipeline-owned incumbent | Hard failure: descriptor lifetime remains coupled to cached PSO | Existing functionality only | Eviction/in-flight hazards | Lowest migration cost | Disqualified by TODO |

## Selected solution

**Selected: `S-P-007-global-table-plus-frame-local-arenas`.**

Use a device-level bindless registry for stable resource indices and one transient linear arena per owning `Frame`. `Frame::finish` and abort invalidate its transient descriptors immediately; native arena reuse waits for GPU completion or remains quarantined after an indeterminate failure. PSO/cache records own layouts and backend pipeline objects, never descriptor pools/sets.

**Rejected:** the incumbent violates the required decoupling. A global-only table is incomplete for transient/dynamic descriptor lifetimes and becomes preferable only if all descriptors can be proven stable and persistent.

**Assumptions and risks:** required indexing tiers and capacities exist on supported hardware; slot generations and deferred reuse prevent stale handles. Pool exhaustion, early reset, heap switching, cache incompatibility, and differing sampler/resource rules remain risks. Performance is unknown.

**Validation:** stress maximum live and frame-transient descriptors; test finish/abort invalidation and delayed-completion reuse; corrupt/invalidate cache blobs; measure update latency, allocations, memory, cache startup, and heap switches on each backend.
