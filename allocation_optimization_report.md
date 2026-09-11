# Allocation optimization report

## Summary

The continuous activity is allocation churn, not demonstrated heap growth.

Two independent costs exist:

1. Presented-frame snapshot caching dominates the default examples: about 2.47 MB and 106 Rust allocations per 640×480 frame on DX12.
2. With snapshot caching disabled, the fresh triangle probes perform about 11 Rust allocations and 1.1 KiB per frame on DX12 and Vulkan.

The main causes are fresh owned collections at every layer:

- frame recording discards reusable capacity;
- graph compilation constructs temporary vectors, trees, and range fragments;
- immutable shader metadata is cloned into frame payloads and pipeline keys;
- execution plans are repeatedly lowered into backend-owned vectors;
- readback bytes are allocated and then cloned;
- transient uploads and event dispatch construct short-lived vectors.

The first objective should be zero Rust global-allocator calls after warm-up for an unchanged frame outside the separately planned shader and readback systems.

Do not begin with a global allocator replacement or a general arena. Typed reusable scratch storage, bounded inline collections, and targeted graph/backend caches directly remove measured work with less lifetime and destruction risk.

The reported roughly 250 MB fullscreen footprint is real at the process level, but it is not a 250 MB Rust container footprint. At 1920×1080, Vulkan reached 251.2 MB private commit at the terminal capture and DX12 reached 233.4 MB. DHAT still found only about 4.92 MiB of live Rust heap. The difference is primarily native driver/runtime state, GPU allocator block commitments and host-visible mappings, swapchain/depth/readback resources, worker threads, and the development shader compiler image.

## Scope and method

The representative workload was `examples/01_triangle` in release mode, hidden/headless, with a fixed frame limit. DHAT profiling began after context, shader, surface, index, and vertex-heap initialization. Temporary instrumentation and generated profiles were removed after analysis.

Steady-state values use the slope between independent 10-frame and 50-frame captures:

`(50-frame total - 10-frame total) / 40`

A separate DX12 5-frame/20-frame comparison measured the default snapshot-cache path.

Static analysis covered:

- `ez-gfx` frame API and state;
- `ez-gfx-runtime` graph, frame, render, binding, and shader code;
- Vulkan, DX12, and Metal frame lowering;
- Vulkan and DX12 transfer workers;
- workspace Clippy plus allocation-adjacent lints.

### Limits

- DHAT measures Rust global-allocator traffic. It does not fully observe driver, OS, COM, Vulkan implementation, or GPU memory allocation.
- DX12 and Vulkan were measured locally; Metal is covered by the executed remote macOS matrix (3 packages, 69 tests at HEAD 5917787), not just static inspection.
- The triangle is a small graph. Larger graphs amplify graph compiler and lowering costs.
- Snapshot cost scales with pixel count.
- Independent-process end-state differences include allocator and event-loop noise.

### Planned-system exclusions

Shader compilation/loading and readback are scheduled for separate redesigns. Their measured allocation costs remain as baseline evidence, but this report does not prescribe their replacement interfaces, caches, ownership, or implementation. Reprofile both after those redesigns land.

## Measurements

| Backend and mode | Rust allocations/frame | Rust bytes/frame | Traffic at 60 FPS |
|---|---:|---:|---:|
| DX12, snapshot cache disabled | 11.1 | 1,076 | 665 allocations/s; 0.062 MiB/s |
| Vulkan, snapshot cache disabled | 11.1 | 1,111 | 668 allocations/s; 0.064 MiB/s |
| DX12, snapshot cache enabled | about 106 | about 2,470,750 | about 141 MiB/s |

Fresh hidden probes (500 frames, triangle workload): DX12 reports 5,539 calls and 537,846 bytes; Vulkan reports 5,559 calls and 554,622 bytes. Safe-facade phases (begin/configure/acquire/bind) and independent shader-free frame-plan/wait validation assert zero; the residual is excluded shader metadata and pipeline-key storage under fixed ceilings. The snapshot-enabled row predates the scratch-retention work and was not re-measured here.

Live Rust heap usage stayed near 4 KiB for the triangle workload (DX12 peak-live increase 3,925 bytes with a -698-byte ending delta; Vulkan 4,181 bytes with a 364-byte ending delta). The captures do not show linear retained growth.

Fresh on-demand telemetry for the same runs: DX12 allocator live 11,665,408 bytes across 25,165,824 bytes of blocks, staging 9 buckets and 217,880 current bytes, counter scratch 2,316 bytes; Vulkan allocator live 11,022,448 bytes across 25,165,824 bytes of blocks, staging 6 buckets and 207,244 current bytes, counter scratch 2,316 bytes. The later repeated Vulkan run measured 217,880 current staging bytes. Both backends retain 3 frame slots, 3 pipelines, and zero readback bytes. Staging retention is bounded by per-pool ceilings under a 64 MiB context-wide aggregate cap enforced by largest matching-queue completed-first eviction; current bytes sit three orders of magnitude below it. The old aggregate high-water updated only when telemetry was queried and could miss an earlier burst, so those numeric high-water values are withdrawn. The corrected implementation records the aggregate at every safe staging-pool mutation boundary and requires fresh backend measurements before publishing replacement values.

### Snapshot scaling

One 640×480 RGBA8 image contains 1,228,800 bytes. The measured cache-enabled slope is approximately two full image buffers plus base frame traffic:

`2 × 640 × 480 × 4 = 2,457,600 bytes/frame`

At 1920×1080, two buffers are about 15.8 MiB per frame before base frame churn.

Every presentation example enables presented-snapshot caching. The hidden `allocation_probe` explicitly disables it so snapshot copies stay outside its frame-allocation boundary. This automation behavior should not be mistaken for an allocation-free production configuration.

## Resident-memory analysis

### Method and metric definitions

The fullscreen analysis ran the release `01_triangle` executable at 1920×1080 for 600 frames with an invisible window. Process memory was sampled every 10–20 ms through Windows process counters. Additional temporary pause points separated context/surface creation, shader compilation, persistent resource creation, and rendering. Comparisons also disabled continuous presented-frame snapshot caching and varied allocator block minima and texture-worker count. All temporary source changes and generated profiles were removed.

Windows reports several different quantities:

- **working set/RSS** is the resident subset of process pages. It includes shared executable and driver pages and is not an ownership total;
- **private commit** is memory committed specifically to the process. It includes native heaps and mapped host-visible GPU allocations, but not all dedicated VRAM;
- **DHAT live bytes** cover Rust global-allocator allocations only;
- **GPU resource bytes** cover device allocations and swapchain images, which neither DHAT nor ordinary process counters measure completely.

Consequently, Task Manager's roughly 250 MB value must not be interpreted as 250 MB of Rust heap or as a leak.

### Fullscreen process measurements

| Backend and mode | Stable working set | Stable private commit | Observed peak working set | Observed peak private commit |
|---|---:|---:|---:|---:|
| DX12, continuous snapshot disabled | about 116 MiB | about 144 MiB | about 178 MiB | about 221 MiB |
| Vulkan, continuous snapshot disabled | about 147 MiB | about 200 MiB | about 184 MiB | about 240 MiB |
| DX12, continuous snapshot enabled | about 147–155 MiB | about 180–188 MiB | about 179 MiB | about 221 MiB |
| Vulkan, continuous snapshot enabled | about 170 MiB | about 223–232 MiB | about 186 MiB | about 240 MiB |

The frame-limited workload always performs one terminal readback for its report. That terminal operation explains why snapshot-disabled peak values approach snapshot-enabled peaks. The stable snapshot-disabled row is the useful production-like comparison.

Both snapshot-disabled backends reached a stable plateau during unchanged rendering. DX12 private commit remained at 151,228,416 bytes across the sampled steady interval. Vulkan remained near 200 MiB before the terminal report. These runs support bounded retention, not continuing growth.

### Lifecycle attribution

The pause-point run produced these approximate private-commit plateaus:

| Completed lifecycle stage | DX12 | Vulkan | Increment explained |
|---|---:|---:|---|
| Context and native surface | 114.1 MiB | 79.4 MiB | Window system, backend device/queues, driver state, three frame slots, descriptor storage, and texture worker pool |
| Development shader compilation | 117.6 MiB | 83.6 MiB | About 3.5–4.2 MiB private commit; loading compiler code added about 14 MiB to the working set |
| Index, vertex heap, and shader resources | 142.7 MiB | 108.8 MiB | About 25.1 MiB on both backends |
| Early fullscreen rendering, snapshots disabled | 144.5 MiB | 166.2 MiB | Swapchain, depth, pipeline/command recording, and driver lazy initialization |
| Warm fullscreen rendering, snapshots disabled | 144.2 MiB | about 200 MiB | Stable backend/driver high-water state |

The approximately 25 MiB persistent-resource jump matches the allocator policy closely. The first touched device-local memory type receives a 16 MiB general block and the first touched host-visible type receives an 8 MiB block. Small logical allocations therefore commit full allocator blocks:

- the first vertex/index allocation is only 64 KiB but opens device-local backing storage;
- the one-draw counter upload is tiny but opens host-visible backing storage;
- `gpu-allocator` retains the last empty general block for each touched memory type;
- DX12 may maintain separate heap categories, so each newly touched category can establish its own minimum.

`gpu-allocator` already has geometric growth. New shared blocks double from the configured minimum up to 256 MiB for device memory and 64 MiB for host memory; allocations larger than the current block size receive a personal block. It also releases empty personal blocks and all but the last empty general block. The issue is therefore the first-block floor per memory type, not absence of growth or trimming.

An experimental 4 MiB device/4 MiB host minimum reduced the post-resource increment from 25.1 MiB to 10.1 MiB on DX12 and 11.8 MiB on Vulkan. Stable private commit fell by about 16 MiB on DX12 and roughly 14–16 MiB on Vulkan. This is a credible small-workload optimization, but not yet a safe default change: it can increase native allocation count and turn medium resources into personal blocks. Accept it only after the full backend matrix and representative texture, geometry, multi-pass, resize, and streaming workloads show acceptable allocation latency and fragmentation. Metal uses the same policy and must be measured before changing the shared contract.

### Resolution-scaled GPU storage

A 1920×1080 four-byte image is 8,294,400 bytes, or 7.91 MiB. Before driver metadata and alignment:

- three swapchain color images require at least 23.73 MiB;
- one D32 depth image requires about 7.91 MiB;
- one readback buffer requires about 7.91 MiB when requested;
- one retained CPU snapshot requires about 7.91 MiB;
- the current readback path creates another image-sized owned byte vector during delivery.

The first two are expected renderer resources. Readback and snapshot storage belong to the separately planned readback redesign. Fullscreen resolution does not materially enlarge graph vectors, handle maps, or other ordinary Rust control structures.

### Preallocation audit

The suspected oversized Rust containers are not the primary resident-memory source:

| Structure | Current behavior | Footprint assessment |
|---|---|---|
| `IndexedIndirectBuffer` | Allocates 1,024 commands at context creation | 20 KiB; fixed and reusable |
| Observability queues | Reserve 1,024 events and 256 diagnostics | Tens of KiB; bounded |
| Texture registry | Capacity is 1,024, but slot/free/event vectors start empty | Limit only; no 1,024-slot allocation |
| Pipeline cache | Maximum is 1,024 entries, but the map starts empty | Limit only; no eager cache allocation |
| Resource maps and retirement lists | Start empty and grow on demand | No large initial reserve |
| Staging pools | Start empty; retain size-classed GPU allocations after use | High-water retention, not startup preallocation |
| Geometry heaps | Logical initial capacity is 64 KiB | Triggers a much larger native allocator block |
| Vulkan frame descriptors | Three pools each admit 1,024 sets and 4,096 storage-buffer descriptors | Native-driver allocation; fixed and potentially oversized for small graphs |
| Texture decode pool | Zero means `available_parallelism - 1` Rayon threads | 31 workers on this 32-thread host, even when no texture is decoded |

Changing the texture worker count from the default 31 to one reduced process private commit by only about 2.0 MiB and working set by about 1.2–1.3 MiB in the paused context. It also removed 30 threads. Lazy pool creation is still worthwhile for startup time, virtual address space, kernel objects, and scheduling noise, but it does not explain most of the 250 MB observation.

The Vulkan descriptor pools deserve measurement rather than an arbitrary smaller constant. They may be lazily materialized by the driver, and a fixed reduction can make valid multi-pass frames fail with `OUT_OF_POOL_MEMORY`. Preflight the actual set and descriptor requirement, then retain completion-gated per-slot pools that grow to a bounded high-water mark. Keep the same public capacity contract across backends.

### Native modules are contextual, not allocation evidence

The process maps NVIDIA's DX12/Vulkan user-mode drivers and the development-only Slang compiler. Shared image mappings and per-module working-set reports are not additive ownership measurements, so they are excluded from allocation attribution and optimization estimates. DHAT slopes and requested allocator reports are the authoritative evidence for Rust churn and allocator-block waste.

Profile runtime distributions with precompiled artifacts separately from development examples. This report intentionally makes no shader loading/compilation redesign recommendation.

### Resident-memory recommendations

1. **Expose requested allocator telemetry.** Use `gpu_allocator::Allocator::generate_report()` only on demand to report live allocation bytes, block capacity bytes, block count, and capacity minus live bytes. Add swapchain image count/extent/format, depth bytes, staging-pool retained bytes, worker count, and backend frame-slot counts. Report generation allocates and must never run automatically per frame.
2. **Make texture decoding lazy.** Store worker policy at context creation and construct the pool on the first asynchronous texture request. After workload benchmarks, cap the automatic topology if 31 decoder threads do not improve throughput.
3. **Evaluate 4 MiB allocator minima across all backends.** Preserve current maxima and geometric growth. Measure native allocation count, frame-time spikes, fragmentation, and retained capacity before adopting the smaller floor.
4. **Bound retained staging bytes.** `ReusableStagingPool` trims after 256 pool epochs but has no aggregate byte ceiling and cannot trim while idle. Add retained-byte/high-water counters, a byte budget, and an explicit memory-pressure or idle trim operation.
5. **Make Vulkan frame descriptor capacity demand-driven.** Preflight each frame's descriptor requirements, grow only a retired frame slot, retain its high-water pool, and enforce a documented ceiling.
6. **Separate benchmark categories.** Record Rust heap, process working set/private commit, allocator block capacity/live bytes, and estimated GPU resource bytes independently. No single “memory” number is actionable.

These changes complement the allocation-churn plan below. Arenas and small-vector libraries target call frequency and locality; they do not remove driver mappings, swapchain storage, thread stacks, or allocator block floors.

## Allocation inventory

### Measured steady-state groups

| Source | Approximate allocations/frame | Approximate bytes/frame | Cause |
|---|---:|---:|---|
| Graph compilation | 0 | 0 | Warmed template cache serves every frame; fresh probes report 503 cache hits against 1 miss per 500 frames |
| Execution-plan lowering | 0 | 0 | Retained frame workspace and scratch reuse across frames |
| Safe/backend lowering | 0 in-scope | 0 in-scope | Safe begin/configure/acquire/bind phases and shader-free frame-plan/wait validation allocate nothing; descriptor preparation is not isolated by the current tests, and shader metadata plus pipeline keys remain under phase ceilings |
| Frame recording | 0 | 0 | Retained recorder storage with high-water caching |
| One-frame counter buffer | 0 | 0 | Retained serialization scratch plus pooled native buckets |
| Transfer worker | about 3 | 136–225 | Live-job and texture-transition collections; occasional channel block growth |
| Event dispatch | 0 | 0 | Reused dispatch scratch with bounded retention |
| Safe `Frame` facade | 0 | 0 | Recycled facade buffers and conversion scratch |
| Example/event loop | 1 | 24 | Host bookkeeping |

The landed rows above read zero on the fresh probes: the CPU probe reports 0 allocations and 0 bytes for 500 frames each of triangle, multipass, and alternating workloads, and the native probe asserts zero for the safe begin/configure/acquire/bind phases. Independent shader-free tests cover frame-plan/wait validation only; they do not isolate descriptor preparation. Dispatch, facade, and counter reuse hold no per-frame allocation by construction (bounded retained scratch and pooled buckets), confirmed by whole-window residual accounting. The transfer-worker and example rows are carried over unmeasured. Remaining whole-window traffic is native residual: triangle execute holds 8 calls and about 270 bytes and finish holds 6 calls and 646 bytes per frame, both inside the excluded shader-metadata and pipeline-key ceilings.

### Snapshot readback

Relevant paths:

- `crates/ez-gfx-backend-vulkan/src/frame.rs`
- `crates/ez-gfx-backend-dx12/src/native/frame.rs`
- `crates/ez-gfx-backend-metal/src/native/frame.rs`
- `crates/ez-gfx/src/state/frame/{vulkan,dx12,metal}.rs`
- `crates/ez-gfx/src/state/frame/mod.rs`
- `crates/ez-gfx/src/api_frame.rs`

The backend creates an owned pixel vector. State stores the returned `Vec<Vec<u8>>`. `frame_readbacks()` then clones the complete collection before `Frame::finish` dispatches callbacks. Surface snapshot storage is another persistent copy, although its capacity is generally reusable after the first frame.

This explains the two image-sized allocations in the cache-enabled measurement.

### Frame recorder capacity loss

`crates/ez-gfx-runtime/src/frame.rs` prevents effective reuse:

- `begin()` replaces the graph with `FrameGraph::new()`;
- `submit()` moves `nodes` out with `mem::take`;
- `submit()` copies indirect commands with `to_vec()`;
- `finish()` replaces the graph again.

The submission is consumed synchronously, but its interface forces ownership transfer. The next frame must regrow most vectors.

### Graph compiler

`crates/ez-gfx-runtime/src/graph.rs` creates fresh collections for every compile:

- two stable topological sorts;
- cloned explicit edges plus inferred hazards;
- a `BTreeMap<NodeId, usize>` for positions;
- initialized-range and tracked-state trees;
- wait, transition, pass, and alias vectors;
- per-resource use vectors;
- nested adjacency vectors.

`NodeId` and `ResourceId` are dense zero-based indexes. Tree maps add node allocations and pointer chasing where indexed vectors are sufficient.

`crates/ez-gfx-runtime/src/graph/validation.rs::subtract` returns a `Vec<ResourceRange>` even though subtracting one buffer or image range has a small fixed maximum. Callers repeatedly `collect` new uncovered-range vectors.

### Shader and binding metadata

`crates/ez-gfx-runtime/src/shader.rs::bindings` clones `ReflectedBindings` on every lookup. Graphics paths merge cloned stage layouts during recording.

Backend pipeline preparation then creates:

- a fresh native layout vector;
- a second sorted layout-key vector;
- cloned entry-point strings;
- an owned `PipelineKey` containing those strings and vectors.

These values describe shader-load-time state. They should not be reconstructed per frame.

These shader-path findings should inform the separate shader redesign; they are not implementation recommendations in this report.

### Backend lowering

All backends lower the same frame through several owned representations:

1. `CompiledGraph`
2. `FrameExecutionPlan`
3. backend binding vectors
4. backend native actions
5. backend-local validation/resource structures

Vulkan additionally allocates short wait vectors, a descriptor-set result vector, external waits, and a vector from `get_swapchain_images` each frame.

DX12 creates an action-index-sized `Vec<Option<NativeAllocation>>` for indirect copies. When the indirect argument buffer is also shader-bound, DX12 performs a logical device-memory allocation and copy each frame to avoid self-aliasing. The HAL allocator may satisfy this from retained blocks, but the operation and bookkeeping remain.

Metal uses the same owned action/binding architecture with equivalent scratch storage; the remote macOS matrix executes its backend suite rather than only compiling it.

### Transient upload and worker paths

`counter_payload()` serializes commands into a new byte vector. The first frame use then sends transfer work.

Vulkan and DX12 transfer batches collect live jobs into a new vector and then build a second texture-transition vector. The source job slice can be iterated directly, or worker-owned scratch can be retained across batches.

### Event dispatch and safe frame facade

`Context::dispatch_events` creates a new `Vec<Event>` whenever a callback is registered. The examples register a callback, so this appears every frame.

`Surface::begin_frame` creates fresh storage for retained leases. `Frame` also owns fresh transient, readback, and binding collections. Typical cardinalities are small, but inline collections must be selected from measured distributions rather than assumed.

## Small collection and string libraries

`arrayvec`, `smallvec`, and `compact_str` are now explicit minimal-feature workspace dependencies in the crates that own their storage. `ArrayVec` covers mathematically bounded range fragments, attachments, and wait sets; its four-fragment subtraction test and two-queue wait tests reproduce those bounds. `SmallVec` covers common-small graph/facade records; the CPU allocation probe covers the accepted set and the facade unit test separately warms cardinality above four, verifies backing reuse, and verifies retained owners are dropped. `CompactString` is limited to private graph/binding names; built-in names fit its documented inline representation without changing public fields.

The dependencies landed together, so the zero-allocation CPU probe is aggregate evidence rather than an isolated before/after attribution for each crate. They remain accepted because each has a distinct constrained role above; no arena or allocator fallback was added. Any future use outside those roles requires its own allocation or size evidence.

### Recommended library matrix

| Library/type | Storage | Overflow | Best project use | Verdict |
|---|---|---|---|---|
| `arrayvec::ArrayVec<T, N>` | Inline fixed array | Returns/panics according to API | Mathematically bounded range fragments and backend wait sets | Strong recommendation |
| `arrayvec::ArrayString<N>` | Inline fixed bytes | Fixed capacity | Protocol fields with a hard, deliberately small bound | Use only with an existing contract bound |
| `smallvec::SmallVec<[T; N]>` | Inline, then heap | Transparent spill | Common-small node accesses, dependencies, attachments, retained leases | Useful after cardinality measurement |
| `compact_str::CompactString` | 24 inline bytes on 64-bit, then heap | Transparent spill | Short mutable/owned diagnostic and semantic names | Preferred general small-string candidate |
| `smol_str::SmolStr` | 23 inline bytes or static/heap representation | Transparent spill | Immutable shared names cloned frequently | Good where O(1) clone matters |
| `smartstring::LazyCompact` | 23 inline bytes, then heap | Transparent spill | Mutable `String`-like values and B-tree keys | Viable, but weaker fit than `CompactString` here |
| `smallstr::SmallString<[u8; N]>` | Configurable `SmallVec` bytes | Transparent spill | Cases needing a selected inline capacity | Avoid unless 24 bytes is measurably insufficient |
| `bumpalo::Bump` | Arena chunks | Allocates another chunk | Phase-local heterogeneous compiler data | Defer |

References:

- [ArrayVec](https://docs.rs/arrayvec/latest/arrayvec/)
- [SmallVec](https://docs.rs/smallvec/latest/smallvec/)
- [CompactString](https://docs.rs/compact_str/latest/compact_str/)
- [SmolStr](https://docs.rs/smol_str/latest/smol_str/struct.SmolStr.html)
- [SmartString](https://docs.rs/smartstring/latest/smartstring/)
- [SmallString](https://docs.rs/smallstr/latest/smallstr/struct.SmallString.html)
- [bumpalo](https://docs.rs/bumpalo/latest/bumpalo/)

### Exact collection candidates

#### `ArrayVec`

Use where capacity follows from the algorithm:

- `subtract`: `ArrayVec<ResourceRange, 4>`;
- Vulkan wait semaphores, values, and stages: bounded by surface plus known transfer queues;
- other one-to-four-item backend API argument lists with enforced validation.

This is preferable to `SmallVec` for strict allocation-free behavior because it cannot silently spill.

#### `SmallVec`

Candidates requiring measured inline capacities:

- `NodeDesc.accesses`, commonly one to four;
- `NodeDesc.dependencies`, commonly zero to a few;
- `PassInfo.colors`, currently constrained to one by measured backend execution paths, though the shared contract may grow;
- `CompiledPass.nodes`;
- `Frame.retained`, `Frame.transients`, and `Frame.readbacks`;
- transfer texture-transition lists.

Do not use `SmallVec` for graph-size vectors such as topological order, actions, pipeline-key slots, or descriptor-set results. Those should retain heap capacity in reusable scratch. Large inline capacities would inflate every object, increase copies, and consume stack space without guaranteeing a win.

### Small-string candidates

#### `CompactString`: preferred default experiment

`CompactString` has the same 24-byte object size as `String` on 64-bit targets and stores up to 24 bytes inline. Longer values spill to the heap. It supports borrowed `str` lookup in maps and optional Serde/rkyv integration.

The documented `compact_str` rkyv feature currently targets a different rkyv major version than this workspace. Do not enable it without compatibility verification. Artifact-serialized fields should remain format-stable rather than changing type for an internal allocation optimization.

Good candidates:

- private `NodeDesc.name`; current names such as `graphics`, `compute`, `present`, `texture-readback`, and `render-target-readback` fit inline;
- private target and graph diagnostic names when profiling confirms churn.

Most measured built-in node names are under 24 bytes, so this can remove one allocation per constructed node without increasing the struct size relative to `String`.

#### `SmolStr`: preferred for immutable shared names

`SmolStr` is 24 bytes, stores up to 23 bytes inline, can reference a static string without allocation, and clones in O(1). It is immutable.

It is attractive when the same shader semantic or entry name must be copied into several long-lived records. However, an identifier/interner or compact numeric identity is better when equality and hashing occur every frame.

#### `SmartString`

`SmartString::LazyCompact` is `String`-sized and stores up to 23 bytes inline. It offers a close mutable `String` API. The lazy mode avoids repeatedly reallocating if a mutated value crosses the inline threshold.

The project rarely mutates hot-path names after construction, so its mutability advantage is not compelling. Prefer `CompactString` unless benchmarks show otherwise.

#### `SmallString`

`SmallString<[u8; N]>` permits a custom inline capacity. That flexibility increases each containing struct by the selected capacity and can make `NodeDesc` or binding arrays materially larger. Use it only after measuring real name-length distributions and cache effects.

#### `ArrayString`

Reflection semantic names are validated up to 255 bytes. An `ArrayString<255>` would bloat every binding requirement and graph copy. A smaller `ArrayString` would introduce a new public rejection rule. It is therefore inappropriate for general semantic names.

It may fit a genuinely fixed internal token, but a numeric enum is usually better.

### Where small strings are not the right optimization

#### Pipeline keys and reflected bindings

Pipeline-key strings and reflected binding names belong to the planned shader redesign. Do not independently introduce a compact-string or interning layer that the redesign would immediately replace.

#### Public API compatibility

Changing a public `String` field to a compact-string type is a Rust interface change. Prefer private compact storage where the owning module controls construction. Any deliberate public cutover must update every caller and relevant FFI/binding evidence atomically.

#### Errors

Owned names in error variants occur on failure paths. Optimize successful frame execution first. Do not complicate diagnostics to save rare allocations.

## Arena and cache placement

An arena-centric design makes sense at one narrow seam: CPU-only frame compilation and lowering. It should not become the repository-wide ownership model.

### Recommended arena seam

A deep `FrameWorkspace` module should own all temporary CPU memory from graph compilation through native command encoding. Callers should provide frame inputs and receive a borrowed execution view; they should not select allocators or manage individual arena objects.

The workspace has two lifetime domains:

1. **CPU frame phase:** graph validation, scheduling, execution-plan construction, binding projection, and native action construction. Memory can be reset immediately after synchronous command encoding finishes.
2. **GPU frame slot:** metadata or resources referenced after submission. Memory can be reset only after the slot fence/completion value retires.

A context-owned bump arena can serve the first domain after the submission interface becomes borrowed. A per-frame-slot arena can serve the second domain only for plain data whose lifetime is exactly that slot. Never reset it before completion.

Keep owning native objects, `Arc`/`Rc`, COM wrappers, Metal objects, and other values requiring `Drop` out of bulk-reset arena storage unless they are explicitly drained and dropped first. Store non-owning handles, indexes, ranges, barriers, and compact action records in the arena; retain owning resources in the existing typed registries and completion-gated pools.

### Typed scratch before a bump arena

Most measured allocations are growable arrays with stable roles. Reusable `Vec` fields are preferable because they:

- run element destructors normally;
- expose capacity and fallible growth directly;
- avoid self-referential arena lifetimes;
- preserve type-local invariants;
- retain only each collection's high-water storage;
- integrate with current synchronous code incrementally.

Start with typed scratch. Add `bumpalo` inside `FrameWorkspace` only if nested variable-sized compiler structures remain costly after dense indexed vectors and retained capacities are implemented. An arena should be an implementation detail, not a parameter propagated through graph, HAL, or backend interfaces.

### Poor arena candidates

- Readback memory: excluded because the readback system is being redesigned separately.
- GPU allocations: use existing backend allocators and completion tokens.
- Persistent resources and handles: use existing generational registries.
- Pipeline objects: use the bounded pipeline cache.
- Transfer jobs: use worker-owned typed scratch; jobs contain owned resources and cancellation state.
- Shader artifact decoding: excluded because shader compilation/loading is being redesigned separately.

### Recommended caches

#### Graph template cache

Repeated frames usually rebuild the same topology. Caching a compiled template can remove scheduling work as well as allocations. Separate stable structure from dynamic inputs:

- stable: resource shapes/lifetimes, node queues, accesses, dependencies, pass structure, hazard edges, topological order, pass coalescing, alias slots;
- dynamic: external completion tokens, current history states, surface extent-dependent descriptions, and concrete transient handles.

Prefer an explicit reusable graph/template identity over hashing the complete rebuilt graph every frame. A content-addressed fallback cache must use a bounded structural digest, collision-safe equality, and deterministic eviction. Recompile when shape, access, queue, pass, extent-dependent resource description, or backend-relevant layout changes.

Do not reuse cached transitions blindly: their `before` state can depend on persistent history and external state. Patch or recompute dynamic waits and history-derived transitions against the cached schedule.

Cache values should contain stable node order, adjacency offsets, hazard edges, coalesced pass membership, alias assignments, and pre-sized offsets for dynamic action emission.

Prefer a key containing explicit graph/template identity and generation. Without explicit identity, use a structural digest followed by collision-safe equality. Include the backend contract/schema version when cached values contain backend-relevant assumptions.

Bound entry count and retained bytes. Use deterministic LRU or clock eviction. Never evict an entry referenced by active CPU work. Expose hit, miss, compile, eviction, entry-count, retained-byte, and high-water counters.

This cache has the highest semantic complexity. First make recompilation allocation-free with typed scratch; then measure whether avoiding compilation itself earns the cache.

#### Frame workspace high-water cache

This is reusable storage rather than a keyed semantic cache. Retain capacities for graph order, adjacency, hazards, transitions, passes, aliases, execution actions, and binding/action projections. Clear lengths after encoding without shrinking.

Apply explicit byte ceilings so one pathological frame cannot permanently pin unbounded memory. Use size classes only where element sizes differ materially; otherwise one retained vector per role is simpler.

#### Backend frame-slot cache

Retain per-slot CPU vectors for descriptor-set results, action auxiliaries, indirect-copy metadata, and fixed submit lists. Cache Vulkan swapchain images at creation/recreation. Slot storage becomes reusable only after its fence retires.

Key entries by frame-slot identity, not transient resource handle. Invalidate surface-dependent entries on swapchain recreation and backend-wide entries on device loss.

#### Transient upload payload cache

GPU staging allocations are already pooled. Add worker/context-owned CPU byte scratch for counter serialization and batch transition lists. Reuse by sufficient capacity rather than exact requested size. Keep cancellation and completion ownership in existing typed job records.

#### Event dispatch scratch

Retain one context-owned event vector. Temporarily take it during dispatch, invoke callbacks without a context-state borrow, clear it, and return it. Cap retained capacity because a burst can otherwise pin the maximum event batch indefinitely.

#### Existing caches to deepen

- Keep the bounded native pipeline cache; coordinate any key redesign with the separate shader work.
- Keep `ReusableStagingPool` for buffer, counter, and upload allocations.
- Reuse existing generational registries rather than adding a second resource cache.
- Add retained-byte and high-water observability before introducing another eviction policy.

#### Deliberately excluded caches

This report makes no recommendation for shader compilation/loading caches or readback caches because both systems are being redesigned separately. Their future caches must still follow the safety rules below.

### Cache safety rules

Every cache must be:

- bounded by count and/or bytes;
- owner- and generation-aware;
- backend-, adapter-, format-, sample-count-, and schema-aware where applicable;
- invalidated on surface recreation, shader destruction, device loss, or incompatible state changes;
- completion-gated when entries reference in-flight work;
- observable through hit, miss, retained-byte, eviction, and high-water counters.

Avoid caching tiny values whose validation or lookup costs more than reconstruction. Never cache an error fallback as valid runtime state.

## Clippy and static analysis

The workspace already enables `all` and `pedantic`. Additional analysis included:

- `clippy::perf`;
- `clippy::nursery`;
- `redundant_clone`;
- `implicit_clone`;
- `iter_cloned_collect`;
- `needless_collect`;
- `collection_is_never_read`;
- `or_fun_call`;
- `format_collect`;
- `large_stack_arrays`;
- `large_types_passed_by_value`;
- `large_enum_variant`.

Only two additional allocation-adjacent warnings appeared: redundant initialization-time clones in DX12 device setup and Vulkan worker-device setup. They do not explain frame churn.

Clippy cannot infer that valid owned return values should reuse storage across frames. It also cannot infer mathematical capacity bounds for `subtract`, or that shader metadata is immutable across submissions.

Recommendation:

- enable `redundant_clone` explicitly;
- retain focused allocation lints in profiling workflows;
- do not enable all of `nursery`, which produced substantial unrelated noise;
- enforce the allocation-free contract with measurement, not lint policy.

## External API impact

Most recommendations can preserve the safe `ez-gfx` API and stable C ABI because they replace private storage and ownership mechanics.

| Recommendation | Consumer impact |
|---|---|
| `FrameScratch`, retained graph capacity, dense indexed scratch | None; private execution machinery |
| Graph template cache | None when kept behind existing frame operations |
| `ArrayVec` for `subtract` and fixed backend waits | None; private return/storage types |
| Heap-free internal `PipelineKey` | Coordinate with the separate shader redesign |
| Backend frame-slot scratch and cached swapchain images | None |
| Transfer-worker and event scratch reuse | None |
| Compact private frame collections | None while public signatures remain unchanged |
| `NodeDesc::new` changed from `Into<String>` to a compact-name input | Technically breaking for custom `Into<String>` input types |
| `PassInfo::new` changed from `Vec<ResourceId>` to a generic or inline collection | Potentially breaking for function-item typing and explicit type assumptions |
| `FrameRecorder::submit()` changed from owned output to borrowed execution | Breaking for direct `ez-gfx-runtime` consumers |

Do not preserve old and new variants as compatibility aliases or parallel paths. For a breaking optimization:

1. select the new authoritative interface;
2. migrate every workspace caller;
3. update tests and relevant documentation;
4. remove the old method, constructor, type, and re-export;
5. if the C ABI is affected, update Rust exports, ABI version, binding metadata, generated bindings, layout probes, and ABI tests atomically.

The preferred design avoids unnecessary public changes:

- keep scratch and cache types private to the runtime and backend layers;
- keep graph template lookup behind existing frame operations;
- preserve public graph result semantics while replacing private storage.

If borrowed frame submission is selected, treat it as a deliberate Rust interface cutover rather than retaining an allocating compatibility method. This repository is pre-1.0, but the migration still needs to be explicit and complete.

No current recommendation inherently requires a C ABI change. Internal reuse can still change non-contractual behavior: memory stays reserved at the observed high-water mark, and allocation failures may occur when scratch grows rather than during every frame. Bound retained capacity while preserving validation, ordering, typed errors, and callback semantics.

## Recommended design

### Module interfaces

The target data flow should be:

1. `FrameRecorder` records into context-owned reusable storage.
2. `GraphCompiler` looks up or builds a stable graph template and emits dynamic state into `FrameScratch`.
3. `ExecutionPlanner` fills reusable action storage in that scratch.
4. Backend lowering borrows the compiled data and fills completion-gated backend frame-slot scratch.
5. The backend submits borrowed actions, then CPU-only scratch is cleared without reducing capacity.

### `FrameScratch` responsibilities

Use typed fields rather than one byte arena:

- graph order and indegree arrays;
- adjacency and hazard storage;
- initialized and tracked resource ranges;
- waits, transitions, passes, and aliases;
- execution actions;
- payload and binding projections;
- backend-native action auxiliaries;
- pipeline-key slots;
- backend frame-slot auxiliaries.

Scratch must grow fallibly and retain its high-water mark. Invalid or unexpectedly large external inputs must fail before large allocation, consistent with existing fail-fast boundaries.

## Prioritized optimization plan

### Priority 1: retain frame and graph scratch

Expected win: most of the measured graph/compiler allocations.

- Stop replacing `FrameGraph` with a new value.
- Avoid moving recorder vectors out permanently.
- Compile into reusable storage.
- Replace dense-ID trees with indexed vectors or generation-stamped arrays.
- Retain nested capacities carefully; clear elements without shrinking.
- Add byte ceilings and high-water counters.

### Priority 2: apply bounded inline collections

Expected win: range-fragment, node-name, node-access, pass, and fixed backend-list allocations.

- Use `ArrayVec<ResourceRange, 4>` for subtraction.
- Use stack arrays or `ArrayVec` for fixed Vulkan waits.
- Experiment with `CompactString` for private node names.
- Measure `SmallVec` capacities for accesses, dependencies, pass colors, and frame leases.
- Remove `PassInfo::new`'s cloned color vector by validating duplicates without allocating.

### Priority 3: introduce the graph template cache

Expected win: repeated scheduling, hazard, pass-coalescing, and alias work.

- Cache only stable graph structure.
- Recompute or patch dynamic waits, history-derived transitions, and concrete resources.
- Prefer explicit template identity; use structural digests only as a bounded fallback.
- Enforce collision-safe equality, deterministic eviction, byte/count ceilings, and generation-aware invalidation.
- Measure hit rate and saved compile time after scratch makes misses allocation-free.

### Priority 4: reuse backend frame-slot lowering storage

Expected win: backend action, descriptor-set result, indirect-copy metadata, and validation vectors.

- Attach CPU scratch to Vulkan/DX12/Metal frame slots.
- Cache Vulkan swapchain images at swapchain creation.
- Reuse public descriptor-set result storage.
- Replace action-index-sized optional arrays with compact active-entry lists when random indexing is unnecessary.
- Ensure scratch is not reused before the corresponding completion token/fence.

### Priority 5: remove transient upload collections

Expected win: counter payload and worker batch allocations.

- Serialize commands directly into retained CPU scratch or pooled mapped staging storage.
- Iterate filtered transfer jobs without collecting.
- Reuse texture-transition scratch in each worker.
- Replace the unbounded channel only if measurement proves its occasional block growth violates the strict contract.

### Priority 6: compact API-local common-small storage

Expected win: several small allocations per frame.

- Inline retained leases, transient references, and graph requests after measuring their cardinalities.
- Reuse event dispatch storage.
- Avoid a hash map where frame binding counts are small and the code already scans linearly.
- Check struct-size and copy-cost regressions before accepting each `SmallVec` capacity.

### Priority 7: evaluate a CPU frame arena

Expected win: residual irregular temporary allocations after typed scratch conversion.

- Keep the arena private inside `FrameWorkspace`.
- Store only phase-local plain data or values explicitly dropped before reset.
- Reset CPU memory after encoding; reset slot memory only after GPU completion.
- Reject the arena if it complicates ownership without a measured improvement over typed scratch.

**Decision: rejected for the measured frame path.** The release allocation probe ran 500 measured frames after four warm-up slots. Triangle and representative multi-pass workloads each recorded zero Rust allocator calls, zero requested bytes, zero measured-window peak-live increase, and zero ending live-byte delta. The alternating two-template workload also reached zero calls and bytes. Typed graph, plan, and recorder scratch therefore leaves no residual allocation for an arena to remove. Adding one would increase reset, destructor, and ownership risk without a measured benefit.

This rejection is scoped to CPU frame recording, graph compilation, execution-plan lowering, and submission recycling. The probe does not execute a native backend, shader compilation, readback delivery, process-memory sampling, or GPU allocation. Residual work in those systems must be measured in its own category rather than used to justify a CPU frame arena.

### Priority 8: evaluate the global allocator

`mimalloc` or jemalloc may reduce residual allocator latency, but neither satisfies the contract. They add packaging and cross-platform complexity while masking avoidable ownership churn. Benchmark them only after the fixed workload reaches zero post-warm-up calls or all remaining calls are justified.

**Decision: rejected.** Replacing the production global allocator cannot improve the CPU recorder's established unchanged-frame result of zero calls, and the native probe now demonstrates that substantial avoidable ownership churn remains. A replacement would add packaging and cross-platform behavior while obscuring regressions that the development-only counting allocators make deterministic. Keep the system allocator in production and keep counting allocators isolated to the probes.

## Verification contract

Use two complementary development-only workloads rather than instrumenting production runtime crates:

1. A CPU-only recorder probe isolates frame recording, graph compilation, execution-plan lowering, and submission recycling.
2. `examples/allocation_probe/main.rs` is the hidden native workload. It initializes a real context, window surface, shader pipelines, index/vertex heaps, and persistent geometry; warms four frames (covering the three retained backend slots); then runs 500 fixed-extent triangle frames and 500 representative compute-plus-graphics frames. Shader compilation happens before measurement in the development-only examples package, so runtime distributions remain compiler-free.

Both probes use the same calls/requested-bytes/peak-live/live-delta boundary. The CPU probe fails if unchanged frames allocate after warm-up. The native probe defaults to the scoped contract: in-scope phases (begin, swapchain configure, transient/counter acquisition, frame binding) hard-assert zero, while execute/finish residual traffic is classified as excluded shader metadata/pipeline-key ownership and must stay within the fixed per-backend whole-window baselines below; any increase above a baseline fails. `--strict-all` instead requires whole-window zero and fails reporting exactly that residual. The native probe queries `MemoryTelemetryReport` and resource diagnostics only after measurement, keeping allocator report generation outside the counted window. Presented readback remains disabled. Context-level graph-cache statistics are not exposed, so cache statistics remain authoritative in the CPU probe.

Use DHAT for attribution. Use a small development-only counting allocator or allocation-counter library for deterministic regression thresholds. Do not ship either in runtime distributions.

### Development evidence and metric boundaries

The release CPU probe's observed results are:

| Workload | Measured frames | Rust calls | Rust requested bytes | Peak-live increase | Ending live delta | Retained frame capacity | Frame high-water |
|---|---:|---:|---:|---:|---:|---:|---:|
| Triangle | 500 | 0 | 0 | 0 | 0 | 6,297 B | 6,569 B |
| Multi-pass | 500 | 0 | 0 | 0 | 0 | 8,460 B | 9,268 B |
| Alternating templates | 500 | 0 | 0 | 0 | 0 | 8,781 B | 9,589 B |

The native probe defaults to scoped success and keeps whole-window totals visible (fresh hidden runs below; ceilings fail any increase):

| Backend | Workload | Measured frames | Rust calls | Rust requested bytes | Residual ceiling (calls / bytes) | Scoped result |
|---|---|---:|---:|---:|---:|---|
| Vulkan (Windows, RTX 3080) | Triangle | 500 | 5,559 | 554,622 B | 7,300 / 600,000 B | pass |
| Vulkan (Windows, RTX 3080) | Compute + graphics | 500 | 11,199 | 2,333,504 B | 32,000 / 5,000,000 B | pass |
| DX12 (Windows, RTX 3080) | Triangle | 500 | 5,539 | 537,846 B | 7,300 / 585,000 B | pass |
| DX12 (Windows, RTX 3080) | Compute + graphics | 500 | 13,683 | 2,563,534 B | 34,500 / 5,200,000 B | pass |

In-scope safe phases hard-assert zero after warm-up on both backends: begin, swapchain configure, transient/counter acquisition, and frame binding each record zero calls, bytes, peak-live increase, and live delta. Independent shader-free unit tests (`spill_cardinality_plan_validation_performs_no_allocations` in the Vulkan and DX12 backends) assert frame-plan and wait validation at 65 actions. They do not execute pipeline actions, descriptor accounting, descriptor allocation, or descriptor lowering. Residual traffic is confined to execute/finish phases and classified as excluded shader metadata/pipeline-key ownership: triangle execute observes 4 calls and 125 B; triangle finish observes 7 calls and 742–750 B; compute observes 3 calls and 2,540–2,580 B; graphics observes 8 calls and 465 B; multipass finish observes 12 calls and 4,601 B on Vulkan and 16 calls and 1,595 B on DX12.

Post-ABI40 rebase, DX12 triangle execute observes 125 B usually with one 541 B sample (6 calls, within the 10-call ceiling), so its excluded byte ceiling is 768 B: the smallest fixed margin above the observed maximum that keeps the phase bounded while the unchanged whole-window residual ceiling still catches systematic regression. No whole-window ceiling and no in-scope zero assertion changed.

Post-workload telemetry remained separately categorized:

| Backend | GPU live / block / waste | Frame slots | Staging current / high-water | Counter scratch | Pipelines | Readback |
|---|---:|---:|---:|---:|---:|---:|
| Vulkan | 11,032,352 / 25,165,824 / 14,133,472 B | 3 | 207,244 B / 207,244 B | 2,316 B | 3 | 0 B |
| DX12 | 11,665,408 / 25,165,824 / 13,500,416 B | 3 | 217,880 B / 217,880 B | 2,316 B | 3 | 0 B |

High-water now reports the true aggregate peak: the telemetry query stores the maximum summed current total across pools instead of summing disjoint per-pool peaks, and both runs held retention flat at the observed totals.

Final backend evidence at HEAD 5917787: local Windows runs give ez-gfx-hal 20/20, ez-gfx-backend-vulkan 46/46, ez-gfx-backend-dx12 24/24, and ez-gfx lib 76/76 tests green, with hidden native probes holding in-scope zero phases and whole-window residuals inside the excluded ceilings above. Remote Linux Vulkan passes 4 packages and 152 tests; remote macOS Metal passes 3 packages (`ez-gfx-hal`, `ez-gfx-backend-metal`, `ez-gfx`) and 69 tests from a clean shallow clone, bypassing the stalled rsync sync. Metal execution is therefore claimed through the matrix, not compile-only.

These columns must remain separate:
| Category | Authoritative fields | Collection boundary |
|---|---|---|
| Rust allocator traffic | calls, requested bytes, measured-window peak-live increase, ending live delta | Development counting allocator or DHAT. The probe resets counters only after warm-up. |
| Retained Rust capacity | current retained bytes, byte ceiling, high-water bytes, graph-cache entries/bytes | Explicit capacity accounting from the owning workspace/cache. This is not the allocator's process-wide live heap. |
| Process memory | working set/RSS, private commit, and their peaks | External OS process counters around a hidden native workload. These include native and mapped memory and are not Rust allocation totals. |
| Requested GPU allocator state | live allocation bytes, block-capacity bytes, block count, allocation count, and `block capacity - live` waste | Explicit on-demand `MemoryTelemetryReport`; allocator report generation may allocate and must stay outside per-frame measurement. |
| Estimated GPU resources | swapchain image count/extent/format and estimated swapchain/depth bytes; backend frame slots | Explicit on-demand backend telemetry. These are format-policy estimates, not measured VRAM, allocator capacity, or process memory. |
| Retained support pools | staging retained/high-water bytes, bounded counter-scratch bytes, and decode worker count | Explicit on-demand safe telemetry, reported independently from GPU allocator and process metrics. |

No aggregate “memory” total may add these categories together. Native backend matrix runs must pair the unchanged-frame allocator assertion with separately sampled process counters and one post-workload telemetry query.

Backend verification must follow the repository matrix:

- local Windows DX12;
- local Windows Vulkan;
- remote Linux Vulkan;
- remote macOS Metal;
- focused runtime/graph tests;
- source-line checks, Clippy, then formatting at handoff.

Also verify that runtime dependency trees remain compiler-free and that any accepted inline-storage crate is a direct, minimal-feature dependency.

## Expected end state

For a fixed graph, extent, and resource capacity after warm-up:

- zero Rust global-allocator calls per unchanged frame outside the classified shader metadata/pipeline-key residual, which stays within the baselined ceilings above (`--strict-all` records that residual as the remaining gap);
- no per-frame graph tree-node allocation;
- no backend wait or swapchain-image vector allocation;
- no transient upload serialization vector;
- bounded graph-template, workspace, backend-slot, and event-scratch retention;
- stable live Rust heap and bounded GPU/driver resource pools;
- identical behavior across Vulkan, DX12, and Metal.
