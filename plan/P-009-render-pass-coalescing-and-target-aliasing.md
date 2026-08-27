# P-009: Render pass coalescing and target aliasing

## Problem

Decide how to coalesce compatible render graph nodes into unified hardware render passes and implement GPU memory aliasing for transient render targets whose lifetimes do not overlap within a frame.

## Prompt context

Source evidence: `TODO.md` ("Coalesce compatible render graph nodes into larger passes when no resource dependency requires a barrier between them. `ez_gfx_render_graph_execute` still records/submits each node separately...", "Add render target aliasing based on render graph node dependencies. The manager still creates one image allocation per acquired target... does not reuse memory for targets whose lifetimes do not overlap").

## Constraints and acceptance criteria

- Coalesce adjacent compatible render graph nodes sharing identical attachments into single dynamic rendering passes when no intermediate barrier is needed.
- Compute non-overlapping target lifetime intervals and alias backing GPU memory to reduce total VRAM consumption.
- Explicit non-goals: aliasing persistent history targets across frame boundaries.

## Dependencies

- Incoming dependency: `P-009` depends on `P-004` (Allocator) and `P-008` (Render Graph DAG).
- Outgoing dependency: `P-010` depends on `P-009` for physical target allocation resolution.

## Unresolved questions

- How to safely handle image layout transitions when multiple aliased textures share the same physical memory?
- What heuristic should balance memory savings versus pass-merging complexity?

## Candidate solutions

### S-P-009-integrated-greedy-merge-and-interval-aliasing

#### Architecture, integration, and applicability

After topological scheduling, greedily merge consecutive nodes only when attachment identities/formats/sample counts/render areas/load-store semantics agree and no required transition or dependency crosses the boundary. Compute each transient target's first/last scheduled use, group compatible memory classes, and best-fit intervals into reusable heap ranges. Emit explicit alias boundaries before a range is rebound to a different resource.

#### Evidence, tradeoffs, and failure modes

Vulkan permits aliasing only with compatible creation/allocation requirements and alias flags; D3D12 placed-resource aliasing needs aliasing barriers; Metal heaps support aliasable resources with explicit synchronization. The original Sponza/Helmet examples have no VRAM/pass-timing baseline, so savings, CPU compile cost, and GPU benefit are unknown. Greedy scheduling is predictable but can miss a globally smaller peak. Failures include lifetime off-by-one, asynchronous queue overlap ignored by interval time, load/store mismatch, attachment feedback, alignment/class mismatch, and missing alias barriers.

#### Sources

- [Vulkan memory aliasing rules](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-memory-aliasing)
- [D3D12 aliasing barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12#aliasing-barrier)
- [Metal heap aliasing](https://developer.apple.com/documentation/metal/mtlresource/makealiasable())
- [Filament frame graph](https://github.com/google/filament/tree/main/libs/fg)
- Original `src/render_target.odin`, `src/render_graph.odin`, and TODO lines 41-43.

### S-P-009-separated-clustering-and-global-placement

#### Architecture, integration, and applicability

Treat pass clustering and memory placement as separate optimization passes. First evaluate legal merge groups and scheduling orders; then run interval coloring/best-fit-decreasing over the chosen schedule, optionally testing several deterministic topological orders for peak memory. Backend placement consumes abstract compatibility classes and returns explicit alias events.

#### Evidence, tradeoffs, and failure modes

Separating passes permits independent validation and can explore memory/parallelism tradeoffs that a single greedy walk fixes early. Multiple schedules increase compile work, and minimizing memory can reduce queue overlap or merge opportunities. No contextual benchmark establishes better VRAM or frame time; algorithmic improvement is a hypothesis. Unbounded search is disqualified for frame-time compilation, so candidate count and time need hard limits. Backend compatibility and alias-barrier failures remain.

#### Sources

- [Diligent Engine render graph transient-resource discussion/source](https://github.com/DiligentGraphics/DiligentCore)
- [Direct3D 12 placed resources](https://learn.microsoft.com/en-us/windows/win32/direct3d12/placed-resources)
- [Vulkan memory requirements](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-memory-requirements)

### S-P-009-incumbent-unaliased-independent-passes

#### Architecture, integration, and applicability

Allocate every acquired target separately and begin/end dynamic rendering for each node, as the original code does.

#### Evidence, tradeoffs, and failure modes

This has functional evidence from existing examples but no measured memory or pass overhead. It avoids alias lifetime corruption and complex merge predicates, but cannot deliver the TODO's coalescing or transient-memory reuse and is therefore a baseline, not a complete requirement fit.

#### Sources

- Original `src/render_target.odin`, `src/render_graph.odin`, and TODO lines 41-43.

## Performance comparison

No candidate has comparable VRAM, graph-compile, pass-count, or GPU-time measurements. Both surviving optimizers can meet the hard requirements; bounded implementation cost breaks the tie.

| Rank | Candidate | Hard constraints | CPU/startup | Memory/GPU | Reliability/implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Integrated greedy merge plus interval aliasing | Passes coalescing and lifetime reuse | One deterministic pass; latency unknown | May reduce peak memory/pass boundaries; amounts unknown | Local optimum, simpler to bound/validate | Native alias rules sourced |
| 2 | Separated clustering/global placement | Passes | Multiple schedules increase unknown compile cost | May improve peak placement but can reduce parallelism; unmeasured | More policy/search complexity | Algorithmic hypothesis; no workload evidence |
| — | Incumbent independent/unaliased | Hard failure: TODOs remain | Existing functional evidence only | No reuse/coalescing | Lowest correctness risk | Disqualified by requirements |

## Selected solution

**Selected: `S-P-009-integrated-greedy-merge-and-interval-aliasing`.**

Use the selected topological schedule, greedily merge adjacent compatible render nodes, compute first/last use intervals for transient targets, place compatible intervals into reusable heap ranges, and emit backend-specific alias boundaries. Never alias persistent history targets.

**Rejected:** the incumbent fails both TODOs. Multi-schedule/global placement is rejected because its extra frame-compile and parallelism tradeoffs have no evidence; it becomes viable if representative traces show materially lower peak memory within a hard compile-time budget.

**Assumptions and risks:** scheduled interval order reflects all queue overlap; compatibility classes capture alignment, usage, sample count, and backend rules. Lifetime off-by-one, missed alias barriers, load/store incompatibility, and asynchronous overlap are critical risks. Benefits are unmeasured.

**Validation:** derive intervals from deterministic graph fixtures; poison/reuse aliased memory under API validation; test queue overlap and history exclusions; compare rendered snapshots; record peak committed/used bytes, alias count, merged pass count, graph CPU time, and GPU timestamps on Sponza/Helmet plus synthetic worst cases.
