# P-008: Render graph scheduling and hazard tracking

## Problem

Decide the automatic render graph DAG compilation, topological sorting, and fine-grained resource hazard tracking model. Replace blanket barriers with precise node-to-node state transitions for attachments, structured buffers, indirect draw buffers, and managed storage images.

## Prompt context

Source evidence: `TODO.md` ("Render-graph structured-buffer and indirect barriers use blanket source stage/access masks... Replace with tracked node-to-node hazard metadata", "Support arbitrary sampled and writable access to managed render targets... lacks a fully general hazard model", "Expand explicit storage-image render target semantics beyond the first managed RWTexture2D path").

## Constraints and acceptance criteria

- Build a frame-local Directed Acyclic Graph (DAG) derived from shader-declared render target dependencies.
- Implement fine-grained resource tracking for buffers and images, inserting minimal necessary pipeline barriers and image layout transitions.
- Support arbitrary read/write, storage-image access, and sampled access across graph nodes.
- Explicit non-goals: forcing manual barrier placement or manual render pass management on the user.

## Dependencies

- Incoming dependency: `P-008` depends on `P-003` (HAL), `P-006` (Reflection), and `P-007` (Pipelines).
- Outgoing dependency: `P-009` depends on `P-008` for pass coalescing and target aliasing.
- Outgoing dependency: `P-016` depends on `P-008` for executing recorded draw passes.

## Unresolved questions

- How should cross-frame resource state persistence (e.g. `LoadTarget` history) be represented in the barrier generator?
- What data structure provides the lowest CPU overhead for frame DAG compilation (<0.1ms)?

## Candidate solutions

### S-P-008-precise-subresource-state-compiler

#### Architecture, integration, and applicability

Compile shader-declared accesses into a DAG, topologically order nodes, and track each buffer range and image subresource as `(queue, stage, access, layout/state, last writers/readers)`. Emit backend-neutral transitions, then lower them to Vulkan Synchronization2 barriers, D3D12 enhanced barriers/resource states, or Metal encoder boundaries, fences/events, and hazard modes. Persist final state for history resources across frames; transient state begins undefined. Batch compatible transitions at node boundaries.

#### Evidence, tradeoffs, and failure modes

The APIs provide precise state vocabularies, but no evidence establishes the proposed `<0.1 ms` compile target. Complexity is approximately proportional to graph edges plus tracked interval/subresource operations; actual latency and allocations must be measured by node/resource/subresource counts on a specified CPU. Finer barriers can permit more overlap, but GPU benefit is workload-dependent inference. Failures include missing WAR/WAW/RAW edges, wrong queue ownership, merging incompatible ranges, stale cross-frame state, and backend semantic mismatch; these are data corruption risks.

#### Sources

- [Vulkan Synchronization2](https://docs.vulkan.org/guide/latest/extensions/VK_KHR_synchronization2.html)
- [D3D12 enhanced barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/enhanced-barriers)
- [Metal resource synchronization](https://developer.apple.com/documentation/metal/resource_synchronization)
- [Granite render graph implementation](https://github.com/Themaister/Granite/tree/master/vulkan)
- Original `src/render_graph.odin` and `TODO.md` lines 9-13 and 64.

### S-P-008-resource-level-conservative-tracker

#### Architecture, integration, and applicability

Track one state per whole resource and insert a conservative transition before every node that changes access class. Keep declaration-derived DAG ordering but do not split mip/layer/range intervals. This materially improves correctness metadata over blanket global masks while remaining close to the incumbent.

#### Evidence, tradeoffs, and failure modes

Whole-resource tracking reduces state entries and branch work, but may serialize unrelated mip levels, buffer ranges, or stages. Official synchronization semantics establish correctness requirements, not measured stall cost. CPU compile time, GPU bubbles, memory, and scaling remain unknown. It fails workloads requiring simultaneous independent subresource use and may not satisfy the TODO's “fine-grained node-to-node” intent.

#### Sources

- [Vulkan synchronization examples](https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples)
- [D3D12 resource barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
- Original blanket-mask implementation in `src/render_graph.odin`.

### S-P-008-incumbent-blanket-node-barriers

#### Architecture, integration, and applicability

Retain broad source stages/access masks before each node. This is the lowest implementation-risk baseline.

#### Evidence, tradeoffs, and failure modes

It has incumbent functional evidence but no GPU timing. Broad barriers are correct only if masks/layouts cover every access; they can unnecessarily order independent work. It is disqualified by the TODO requiring tracked hazards and general sampled/writable target semantics.

#### Sources

- Original `src/render_graph.odin` and `TODO.md` line 64.

## Performance comparison

No candidate has measured graph-compile or GPU-stall data for comparable graphs. Fine-grained correctness is a hard TODO requirement; the `<0.1 ms` target is unsubstantiated and remains a validation goal, not evidence.

| Rank | Candidate | Hard constraints | CPU/memory scaling | GPU concurrency | Reliability/implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Precise subresource state compiler | Passes fine-grained buffers/images and general access | Expected edge plus interval/subresource work; constants unknown | Can avoid unrelated transitions; benefit unknown | Highest correctness burden | API semantics sourced |
| 2 | Resource-level conservative tracker | Partial failure for independent ranges/subresources | Fewer state entries, unmeasured | May serialize unrelated work | Simpler, still tracked | Correctness semantics sourced; TODO fit partial |
| — | Blanket incumbent | Hard failure: no tracked node-to-node hazards | Existing functional evidence only | Broad ordering, cost unmeasured | Lowest cost | Disqualified by TODO |

## Selected solution

**Selected: `S-P-008-precise-subresource-state-compiler`.**

Compile declarations into a DAG and track buffer ranges and image subresources with queue, stage, access, layout/state, and last reader/writer metadata. Lower neutral transitions through P-003 to Vulkan Synchronization2, D3D12 enhanced barriers, and Metal encoder/fence/event operations. Persist final states only for history resources.

**Rejected:** blanket barriers fail the explicit TODO. Whole-resource tracking cannot satisfy arbitrary independent subresource access and is retained only as a conservative bring-up/debug mode, not the selected architecture.

**Assumptions and risks:** declarations/reflection fully describe accesses; backend lowering can preserve the neutral model. Missing edges or ownership transfers can corrupt data. Compile time, allocations, and GPU benefit remain unknown.

**Validation:** property-test RAW/WAR/WAW and queue-transfer graphs; compare generated transitions with validation layers/debug tooling; snapshot deterministic schedules/barriers; run fork/join, history, storage, sampled, indirect, and subresource cases; benchmark by node/edge/resource/subresource count on a named CPU and capture GPU timestamps.
