# Components

> **Do not read the individual problem files.** This document and `SOLUTIONS.md` are the canonical, self-contained implementation plan.

## System context

### Outcome

Migrate the original Odin/Vulkan `ez_gfx_api` to Rust/Cargo while roughly preserving its recognizable API and render-graph model; support Vulkan, DirectX 12, Metal, universal Slang source compilation, Basis Universal/KTX2 and compressed textures, a runtime that does not bundle Slang, Rust GPU allocation instead of VMA, the inherited TODO work, asynchronous uploads, and unit/integration/snapshot testing.

### Constraints, non-goals, and assumptions

- Runtime packages must not depend on or bundle the Slang compiler.
- Vulkan, DX12, and Metal are required; Vulkan-only abstractions are incomplete.
- Explicit shader target attributes are authoritative for target intent.
- Rust uses the clean context-owned interface; C/C# use the explicit ABI 37 lifecycle through the dedicated FFI seam.
- External inputs and binary artifacts require validation; no panic crosses FFI.
- OpenGL, DX11, software rasterizers, a custom shader DSL, and a custom window system are out of scope.
- `gpu-allocator` 0.28 is the selected cross-backend Rust allocator.
- Supported devices meet a declared capability floor or receive explicit unsupported errors.
- Compiler/build environments may contain Slang, DXC/signing, and Apple tools; runtime deployments do not.
- Measurements are workload- and environment-specific; unknown values remain unknown.

## Selected-solution summary

### P-001: Cargo workspace and delivery boundaries â€” Strict workspace boundary

A virtual workspace separates core types, runtime/artifact loading, offline in-process Slang compiler bindings, FFI, backend dependencies, and optional decoders. Runtime dependency audits prevent compiler leakage; feature resolution follows the selected MSRV.

### P-002: Public API and C ABI bindings â€” Owning Rust facade and raw FFI

One owning `Context` controls native lifetime and invalidates every descendant on destruction or drop. Resource wrappers retain memory-safe access but not an independent native context lifetime; texture wrapper drop intentionally leaves its stable bindless heap entry resident until context teardown. `Surface::begin_frame` and `Context::begin_frame` return target-less owning `Frame` values; configure methods attach logical swapchain or cached named targets. Recording borrows frames mutably, `Frame::finish(self)` preserves exact errors, and `Drop` aborts. ABI 37 alone exposes explicit lifecycle calls and opaque generational `u64` handles, including `EzGfxFrame`.

### P-003: Multi-backend hardware abstraction â€” Custom static raw HAL

A narrow backend-neutral contract is implemented directly over Vulkan, DX12, and Metal bindings. Capability discovery and state lowering remain backend-local; concrete dispatch avoids hot-path trait-object dependence.

### P-004: GPU memory allocation â€” `gpu-allocator`

A HAL allocation interface carries size, alignment, memory class, mapping, retirement, and alias lifetime. `gpu-allocator` 0.28 supplies the Vulkan, DX12, and Metal implementations.

### P-005: Universal Slang compilation â€” Native multi-target Slang

Offline Slang compilation through `shader-slang`/slang-rs emits native SPIR-V, DXIL, and Metal products plus canonical metadata. Target attributes are captured before optimization and validated per target.

### P-006: Precompiled shader container and reflection â€” Framed, validated `rkyv`

A bounded, versioned `.ezgfxshader` file uses a fixed little-endian frame around one bytechecked `rkyv` payload. It stores stage-grouped target products, reflection, compiler provenance, and an execution digest. Runtime validates and loads it without JIT/compiler fallback.

### P-007: Pipeline caching and descriptors â€” Global table plus frame-owned arenas

Stable generation-checked bindless indices are independent of PSO ownership. Transient descriptors belong to the owning `Frame` and are invalidated on completion or abort; native reuse remains GPU-completion-safe.

### P-008: Render-graph hazards â€” Precise subresource state compiler

A frame DAG tracks ranges/subresources, queues, stages, access, layouts, readers/writers, and history. It lowers precise transitions to each backend and rejects invalid/unreachable dependencies.

### P-009: Pass coalescing and transient aliasing â€” Integrated greedy compiler

Compatible adjacent nodes merge; first/last-use intervals assign compatible transient memory ranges. Persistent history is excluded and alias boundaries are emitted per backend.

### P-010: Target declarations and formats â€” Shader authority with runtime probing

Canonical target metadata preserves kind, usage, scale, sampleability, load/store, abstract format candidates, and clears. Device capability queries resolve physical formats and report unsupported intent.

### P-011: Vertex and index geometry heaps â€” Owning generation-checked leases

Named vertex heaps and the singleton context-owned index heap use range free lists plus generation-checked allocation leases. Safe wrappers retain parent/context ownership and release through `Drop`; C retains explicit handle release. One-frame `Buffer`, `CounterBuffer`, and single-value `ValueBuffer` bind by `[Buffer]`/`[CounterBuffer]` shader name in a persistent frame-local set; execute calls read the current set, and same-name replacement leaves a never-executed prior resource unclaimed. Native counters store the count at byte 0 with the element region at shared HAL offset 256 (bytes 4..255 zeroed, 252 bytes padding for Vulkan `minStorageBufferOffsetAlignment`); Vulkan/DX12 read the count at byte 0 and commands at offset 256, and Metal encodes capacity over zeroed tails (emulation: no indirect-count opcode).

### P-012: Transfer staging and batching â€” Timeline-recycled bucket pools with adaptive batches

Size-classed staging pools recycle after completion; a transfer owner batches copies and flushes on explicit readiness/frame/threshold events. Independent ordered timeline domains avoid manager serialization.

### P-013: Explicit frame ownership

Presented and managed-target begin functions return an owning `Frame`. Recording requires `&mut Frame`; `Frame::finish(self)` consumes it, returns recording/submit/present errors unchanged, and `Drop` aborts. Named buffer/counter/value entries persist in the frame binding set until replaced or terminal completion; `execute_compute`/`execute_graphics` materialize and read without removing them. Context-acquired buffers are claimed by their first execution, reusable only within that frame, and invalid after every terminal path; native backing is recycled only after completion.

### P-014: Basis Universal and compressed textures â€” Feature-gated official transcoder wrapper

An optional `basis-universal` wrapper handles universal `.basis`/KTX2 input and backend-supported BC/ASTC output; direct compressed payloads bypass transcoding.

### P-015: Texture streaming and partial updates â€” Progressive mip streamer

Coarse mips become sample-ready first; stable descriptors publish after handoff. Validated subregion updates and phase/byte telemetry support dynamic atlases and streaming.

### P-016: Multi-draw indirect and dynamic state â€” Scissor-batched MDI

Standard indirect records pair with validated viewport/scissor side tables. Consecutive equal-state ranges batch native dynamic state updates while retaining compute-filled MDI.

### P-017: Owning surface and guarded presentation

An owning `Surface` carries memory-safe access to its context while the host retains the native window; context destruction still invalidates it. Window construction dispatches from `RawWindowHandle`, queries the initial native extent, and rolls back partial native/init state. Headless construction uses an explicit extent. Zero extent returns `NotReady`; consuming frame completion presents; presentation targets reject shader reads and use transfer readback.

### P-018: Async workers â€” Scoped Rayon compute pool and bounded transfer channel

A library-owned CPU pool decodes/transcodes; a bounded channel feeds one transfer owner. Cancellation, shutdown, backpressure, and owned completion-event production are explicit.

### P-019: Tests and golden snapshots â€” Real backend offscreen goldens

Backend-specific offscreen/readback fixtures provide PNG goldens and tolerances; deterministic IR/unit tests complement them. Missing required adapters are unavailable, never passing.

### P-020: Migration cutover â€” Clean ownership cutover

The final cutover uses the shared `Example` host for winit inversion, native window hosting, resize, input, automation, and consuming frame dispatch. Each main creates platform-free `ContextOptions` and calls `Context::create_surface_window` with the host's `HasWindowHandle`; initial extent comes from the native window. Resources need no artificial scopes or ordered manual teardown: `Context::destroy` and owner drop invalidate and destroy all context-owned resources, including surfaces and retained texture-heap entries. Rust exposes no compatibility aliases or per-frame texture retention; ABI 37 preserves explicit C lifecycle and separate window/headless surface constructors.

### P-021: Cross-backend shader execution semantics â€” Target-native layouts with canonical semantic ABI

One root Slang module defines stable semantic resource declarations, while `.ezgfxshader` retains target-native products and reflection. Each stage owns exactly one internal entry point; callers load without naming it. Runtime never assumes identical physical slots or aggregate layouts. DXIL variants target Shader Model 6.5 and use explicit descriptor tables/root descriptors; no 6.6-only direct heap indexing is part of the semantic ABI.

### P-022: Backend, device, and capability admission â€” Single modern semantic floor

Rust/FFI expose explicit adapter enumeration/selection and a deterministic default. One semantic capability floor maps to native features and limits, including Shader Model 6.5 as the lowest model required by implemented DXIL semantics; unsupported devices fail before manager creation. Cache and snapshot identity includes backend, stable device/driver identity, and profile schema.

### P-023: Device loss and runtime recovery â€” Terminal lost runtime

The first fatal device result atomically poisons the runtime. New work, outstanding tokens, leases, transfers, and callbacks fail exactly once; no path waits for lost GPU progress. Handles remain invalid, diagnostics remain bounded, and the host creates a fresh runtime.

### P-024: Persistent pipeline cache lifecycle â€” Host-owned validated blobs

Runtime imports/exports bounded, validated, backend-specific cache envelopes. Hosts own filesystem paths, atomic commits, locks, quotas, eviction, and cross-process policy. Bad or absent data falls back uncached; cache I/O never enters the render hot path.

### P-025: Metal shader artifact production â€” Offline metallib variants

Offline tooling converts Slang-generated MSL through Apple tools into `.metallib` products stored in `.ezgfxshader`; non-Apple builds retain portable MSL for coverage. Runtime selects a compatible product without source compilation, and Metal binary archives remain separate derived PSO caches.

### P-026: Runtime threading and event delivery â€” Host-polled bounded event queue

Each context owns a bounded queue of owned events. Hosts poll/drain on their chosen thread; frame recording remains context-affine and other concurrency is explicit. Overflow is observable and non-blocking; CPU shutdown barriers prevent callback/payload use after destruction.

### P-027: Artifact integrity and provenance â€” Host-owned authenticity

Runtime owns bounded parsing, compatibility checks, complete execution-content digests, and versioned provenance reporting. The host/package boundary owns signing and authenticity; the runtime never labels an unkeyed digest as authentication.

### P-028: Local diagnostics and profiling â€” Bounded structured event stream

All components emit typed, correlated local diagnostic/profiling events through P-026's queue. Events declare severity, category, IDs, sequence/domain, clocks, units, availability, and payload. Overflow emits a loss marker/count; no remote upload or hidden persistence exists.

### P-029: Cross-platform native build and distribution â€” Centrally built separate signed artifacts

Target-native release CI builds pinned sources and publishes separate runtime/FFI, compiler/tool, and optional Basis packages with deterministic manifests, notices, provenance, signing/notarization, and binary-import audits proving compiler-free runtime delivery.

## Connection map

- Workspace boundary -> release engineering -> every package: Cargo feature/dependency contracts feed target-native CI, which emits separate audited runtime/FFI, compiler, and optional decoder products.
- Public API/FFI -> adapter admission, runtime, cache, events, and resources: validates external calls/handles/bytes, enumerates/selects devices, imports/exports cache blobs, exposes lost state, and drains owned events.
- Shader compiler -> semantic ABI -> artifact store -> runtime pipeline/cache: one source produces target-native layouts/blobs and offline metallib variants; runtime consumes semantic IDs without compiler linkage.
- Artifact trust -> host/package boundary: runtime owns bounded parsing, compatibility, digests, and provenance; hosts own authenticity, signing, cache storage, and deployment policy.
- Backend HAL -> capability admission, allocator, descriptors, graph, transfers, surfaces, diagnostics, and snapshots: owns native lowering, queues, feature probes, device errors, and bounded diagnostic inputs.
- Allocator -> geometry, textures, staging, transient graph targets: owns memory classes, mapping, retirement, alias compatibility, and terminal lost-device invalidation.
- Pipeline/cache -> host cache blob boundary, graph, and draw submission: maps target-native reflection to PSOs/descriptors; imports/exports validated native cache envelopes without filesystem ownership.
- Graph compiler -> HAL command recording and diagnostics: converts semantic declarations/resources into order, barriers, waits, merges, clears, alias boundaries, and correlated schedule events.
- Worker orchestration -> texture/geometry ingestion -> event queue and frame policy: bounded jobs and payloads cross into transfer ownership; completion/error events flow to hosts, while submitted aggregate prefixes feed frame readiness.
- Runtime event/diagnostic boundary -> host: all components publish bounded owned events; the host chooses polling thread/cadence, while overflow/loss/shutdown are explicit.
- Surface/presentation -> HAL and validation: borrowed host window handles or explicit headless extents enter; context-owned lifetime, atomic construction rollback, and acquire/present/readback/loss results leave.
- Validation/cutover -> release artifacts and every component: executes contracts, records adapter/driver/profile/provenance evidence, compares goldens, audits binary imports/signatures, and gates publication.

## End-to-end flows

### Offline shader to runtime draw

The compiler receives a backend-agnostic Slang source importing the root shared module, assigns canonical semantic resource IDs, emits target-native SPIR-V and Shader Model 6.5 DXIL products, and invokes Apple tools for metallib on Apple hosts. It writes a bounded `.ezgfxshader` whose framed `rkyv` payload contains one entry point per stage, target products, reflection, provenance, and a BLAKE3 digest. Runtime bytechecks and semantically validates the payload before choosing backend/stage products without Slang or source compilation; the host/package boundary applies authenticity policy.

### Asynchronous texture load

The API validates encoded bytes and submits a bounded job. Workers decode/transcode and choose a format admitted by the selected device floor; texture ownership creates mip resources and sends validated payloads to pooled staging. Transfers submit higher/coarser mips first; descriptor publication follows sample-ready completion, and the frame policy may wait through a configured minimum mip using an aggregate prefix. Typed progress/completion/error diagnostics enter the bounded context event stream and are observed only when the host polls. Cancellation, overflow, shutdown, and device loss complete deterministically.

### Windowed frame, resize, and screenshot

The host supplies borrowed native handles and observed extent. `Surface` construction either returns an owning wrapper or destroys the unpublished raw surface after native/init/resize failure. `Surface::begin_frame()` returns a frame that `configure_swapchain(size, format)` attaches inside the winit callback. Graph validation enforces write-only presentation use; `Frame::finish(self)` submits then presents with exact errors, while `Drop` aborts. Readback bytes are callback-scoped.

### Migration validation and cutover

Unit/property fixtures test handles, artifacts, allocators, graph states, queues, semantic reflection, lost-device transitions, cache envelopes, and diagnostic overflow. Backend fixtures execute offscreen scenes under the single admitted profile and compare recorded adapter/driver goldens; ABI/C# smoke tests exercise polling and cache-blob ownership. Target-native CI builds separate signed/notarized runtime/FFI, compiler, and optional Basis packages; dependency-tree and binary-import audits prove compiler-free runtime delivery. A requirement-to-gate matrix covers six examples and all TODOs. Missing required adapters, metadata, signatures, imports, or snapshots block publication.

### Device loss and host recreation

A fatal acquire/submit/transfer/present result atomically poisons `ContextInner`. Outstanding owners invalidate GPU state without waiting for lost progress. The host drains diagnostics, drops lost resource wrappers, creates a fresh admitted context/surface, and reloads resources. No frame, handle, timeline, or native cache object crosses the loss boundary.

### Host event, diagnostics, and cache loop

Components publish owned typed events with correlation, sequence/domain, clock/unit, and backend context into one bounded per-context queue. Hosts poll/drain on their chosen thread; overflow yields a loss marker/count and never blocks rendering. Hosts may import/export validated opaque pipeline-cache envelopes and persist them under their own atomicity, quota, locking, and authenticity policies. Runtime keeps no event or cache filesystem.

## Component: Workspace and delivery boundary

### Responsibility and boundary

Own Cargo package topology, resolver/features, target-specific dependency edges, and source-build contracts. It enforces compiler/runtime/optional-decoder separation but does not own release signing or graphics execution.

### Problems and selected solutions

Realizes P-001 and workspace portions of P-029. Runtime cannot depend on Slang; optional Basis/C++ linkage is explicit; target/package/feature combinations are reproducible and inspectable.

### Interfaces and connections

Provides package dependency/feature contracts, compiler-binding boundary, pinned lockfile/profile inputs, runtime-only dependency audits, and target matrix to release engineering and validation.

### Data and persistence

Owns Cargo manifests, lockfile, profiles, generated build metadata, and dependency reports. It owns no mutable runtime GPU state or signing keys.

### Technology

Rust/Cargo package boundaries encapsulate native backends, compiler libraries, FFI, and optional decoders. Target-native build scripts are permitted only behind their owning package.

### Lifecycle and performance

Release CI consumes pinned source contracts; expert source builds remain possible. Measure clean/incremental time, dependency count, feature leakage, and binary inputs per target.

### Failures and trust

Feature unification/leakage, unsupported target graphs, or optional decoder leakage fail before release assembly. Runtime-only dependency-tree checks are mandatory.

## Component: Release engineering and distribution

### Responsibility and boundary

Own target-native CI, archive/installer assembly, dependency manifests, architecture/minimum-OS labels, notices, symbols, signing/notarization, provenance publication, and clean-install proof. It does not define runtime semantics.

### Problems and selected solutions

Realizes P-029 and deployment portions of P-001/P-014/P-025/P-027. Canonical products are separate runtime/FFI, compiler/tool, and optional Basis-enabled signed artifacts.

### Interfaces and connections

Consumes workspace contracts and shader products; emits packages, manifests, signatures/notarization records, provenance, and dependency/import audits to hosts and cutover gates.

### Data and persistence

Owns archives/installers, manifests, symbols, notices, attestations, signatures, and CI records. Credentials never enter runtime packages.

### Technology

Target-appropriate Rust/native SDKs/linkers, Apple Metal/signing/notarization tools, Windows signing, deterministic layouts, and binary-import inspection.

### Lifecycle and performance

Pinned sources build on declared hosts, are audited/signed, then installed in clean target environments. Measure build/cache/sign/install/startup time, package size, dependencies, retention, and matrix cost.

### Failures and trust

Wrong architecture, SDK drift, missing redistributables/notices, signature/notarization failure, stale provenance, or compiler leakage blocks publication.

## Component: Public Rust API and FFI boundary

### Responsibility and boundary

Own safe Rust resource/context APIs and C exports, layouts, handles, adapter selection, cache-blob import/export, event polling, ownership, concurrency declarations, and boundary validation.

### Problems and selected solutions

Realizes P-002, public portions of P-022/P-023/P-024/P-026/P-028, and P-020 ABI gates. Backend-native handles and shader physical layouts remain private.

### Interfaces and connections

RAII resources, typed results, opaque adapter/handle IDs, `extern C` pointer/count structs, `poll_events`/`drain_events`, cache-envelope bytes, and bindings connect hosts/C# to runtime. Frame recording is context-affine; other `Send`/`Sync` behavior is explicit.

### Data and persistence

Owns ABI reports, handle owner/generation metadata, and payload ownership rules. The runtime lifecycle component owns bounded event state; hosts own durable caches and artifact authenticity.

### Technology

Rust `repr(C)`/FFI contracts and existing `bindings.xml`/C# consumers; unsafe marshalling and panic containment stay here.

### Lifecycle and performance

Creation selects one admitted adapter; hosts poll/release events. Lost contexts reject calls until destruction. Measure call/validation/selection/polling/cache-copy/handle/upload/submission costs.

### Failures and trust

Invalid input, stale/lost owners, malformed cache blobs, overflow, finalizer races, and panics become typed errors. No unwind crosses ABI and shutdown prevents callback-after-free.


## Component: Backend capability and native HAL

### Responsibility and boundary

Own Vulkan, DX12, and Metal enumeration, devices, queues, synchronization, native states, bindless/indirect/presentation primitives, capability inputs, loss detection, and native diagnostics.

### Problems and selected solutions

Realizes P-003/P-022/P-023 and backend portions of P-004/P-007/P-008/P-009/P-010/P-013/P-016/P-017/P-021/P-028. One semantic floor maps to backend checks; physical shader/state lowering remains backend-local.

### Interfaces and connections

Neutral device/resource/queue/barrier/presentation contracts; stable adapter records and feature/limit matrix; target-native shader layouts; typed fatal/nonfatal errors; bounded diagnostics; allocator/graph lowering.

### Data and persistence

Owns native resources/state/timelines, adapter/driver identity, and live native cache objects. It owns neither durable cache bytes nor recovery across a lost device.

### Technology

Rust `ash`, D3D12, and `objc2-metal` bindings. Capability spikes prove the full floor; concrete dispatch has no semantic-tier hot path.

### Lifecycle and performance

Startup ranks/probes adapters and enables required features. The first fatal result poisons the context; HAL submits or waits no further. Measure probe/device creation, features, recording, diagnostics, loss notification, and binary size.

### Failures and trust

Missing floor features, unstable identity, state/layout errors, driver/device loss, and lifetime violations fail explicitly. Unsafe handles stay adapter-local; diagnostics are bounded and conformance-tested.


## Component: Shader compiler and reflection frontend

### Responsibility and boundary

Own offline Slang compilation, canonical semantic IDs, target-native layouts/products, attribute authority, provenance, diagnostics, and Apple metallib production. It is never a runtime dependency.

### Problems and selected solutions

Realizes P-005/P-021/P-025 and compiler portions of P-006/P-010/P-027. One source emits SPIR-V, DXIL, and metadata-qualified metallib variants with no runtime source fallback.

### Interfaces and connections

Source/entries/options -> canonical semantic graph plus target-indexed binding/packing/entry/specialization layouts, blobs, tool/source provenance, hashes, and diagnostics. Release engineering consumes its products.

### Data and persistence

Reads source/includes/modules and writes artifact sections/products/provenance. Apple tools produce IR/metallib variants; compiler state never ships at runtime.

### Technology

Rust bindings around in-process Slang, DXC where required, and Apple `metal`/link tools. Target-native physical layouts preserve stable logical semantic IDs.

### Lifecycle and performance

Runs offline on declared hosts. Measure compile/link time/RSS, layout/blob/metallib bytes, variant coverage, instruction quality, and provenance/hash cost.

### Failures and trust

Semantic-ID collision, cross-target type/access mismatch, specialization/packing error, missing tool/variant/provenance, or compilation failure prevents publication. Tests compare metadata, code, and goldens.

## Component: Shader artifact and pipeline/descriptor cache

### Responsibility and boundary

Own bounded `.ezgfxshader` framing, bytecheck/schema/digest/provenance validation, semantic-ID to target-layout resolution, PSO keys/native cache objects, stable bindless registry, frame descriptor arenas, and cache-envelope import/export. It owns no compiler execution, filesystem, authenticity keys, or graph scheduling.

### Problems and selected solutions

Realizes P-006/P-007/P-021/P-024/P-027 and runtime portions of P-025: versioned bundles, target-native binding/packing/specialization layouts, independent descriptor lifetimes, host-owned persistent cache bytes, and host-owned authenticity.

### Interfaces and connections

Validated artifact loader; canonical semantic graph -> selected target layout/PSO; descriptor allocation/update; bounded cache-envelope import/export; provenance/verification status. Compiler output connects to HAL/graph/textures/draws without exposing physical bindings publicly.

### Data and persistence

`.ezgfxshader` files and host cache blobs are externally durable, bounded inputs. Runtime owns validated artifact data, live PSOs/native cache objects, stable tables, and frame arenas. Hosts own authentication and cache storage/locking/quota/atomicity; no runtime cache filesystem exists.

### Technology

Rust bounded parser/digester and backend cache adapters. Cache envelopes key schema, semantic interface, backend, admitted profile, stable adapter/driver identity, length, checksum, and opaque payload. Metallib metadata includes platform/SDK/minimum OS/Metal language/compiler/entry/interface fields.

### Lifecycle and performance

Artifact validation/digest precedes target selection and native calls. Cache import failure falls back uncached; explicit export occurs only from a healthy device. On `Lost`, native cache state is discarded and never exported. Arenas reset only after valid GPU completion. Measure parse/hash, semantic lookup, metallib selection/load, cache copies/hits, PSO creation, descriptor updates, RSS, and heap switches.

### Failures and trust

Reject bounds/overlap/schema/digest/provenance incompatibility, semantic-ID/type/access/specialization drift, missing Metal variant, mismatched profile/device cache, malformed envelopes, and lost-device export. Hashes identify content but never assert authenticity; no JIT/source fallback exists.


## Component: GPU memory and resource lifetime

### Responsibility and boundary

Own Rust allocator integration, dedicated/suballocated resources, mapping/coherency, transient alias compatibility, retirement, and terminal invalidation. It hides the unresolved package behind a stable interface.

### Problems and selected solutions

Realizes P-004/P-023 and allocation portions of P-009/P-011/P-012/P-014. `gpu-allocator` 0.28 provides backend allocation behind the HAL contract.

### Interfaces and connections

Allocate/free/map with size, alignment, class, usage, alias class, and completion token; invalidate all allocations on terminal loss. Consumers never access allocator-native objects.

### Data and persistence

Owns GPU heaps, metadata, mapped blocks, retirement queues, and alias ranges. State is runtime-local and never survives device loss.

### Technology

`gpu-allocator` adapters support Vulkan, DX12, Metal, and unified memory.

### Lifecycle and performance

Pools start with a healthy device and retire by completion. On `Lost`, outstanding tokens fail exactly once, no allocation is reported complete/reused, and CPU metadata tears down without GPU waits. Measure allocation/fragmentation/coherency/contention plus loss fan-out/teardown.

### Failures and trust

Unsupported classes, alignment/coherency errors, double ownership, premature reuse, or post-loss access fail closed.


## Component: Render graph and hazard compiler

### Responsibility and boundary

Own frame DAG construction, semantic resource declarations, subresource hazards, waits, history, merges, alias boundaries, clears, schedule diagnostics, and loss-aware compile rejection. It does not own native commands.

### Problems and selected solutions

Realizes P-008/P-009/P-010/P-021/P-022/P-023/P-028 and graph portions of P-016/P-017: precise state, target-native semantic reflection, one admitted profile, greedy optimization, and correlated diagnostics.

### Interfaces and connections

Semantic node/declaration registration -> schedule with order, transitions, generic waits, physical targets, clears, merges, aliases, and diagnostic correlation IDs. Consumes selected target reflection, API resources, descriptors, and context health; emits HAL work/events.

### Data and persistence

Frame DAG/state/interval tables are local; history final state persists only while the same context/device remains healthy. Loss discards frame/history state.

### Technology

Backend-neutral IR lowers to Synchronization2, D3D12 enhanced barriers, and Metal operations after the capability floor is admitted.

### Lifecycle and performance

Compile once per frame only in `Running`; a loss transition aborts schedules and fails waits exactly once. Measure graph CPU/allocations, barriers, merges, alias bytes, diagnostics overhead, and GPU timestamps.

### Failures and trust

Reject cycles, missing/mismatched semantic declarations, invalid access, unreachable or lost tokens, illegal merges/aliases, swapchain reads, and compilation after loss. Property-test hazards and correlate failures through the bounded event stream.


## Component: Geometry and transfer ingestion

### Responsibility and boundary

Own named geometry heaps, generation handles, mapped leases, pooled staging, transfer batches, completion tokens, upload readiness, and loss fan-out.

### Problems and selected solutions

Realizes P-011/P-012/P-023/P-028: generation free-list/leases, timeline pools/batches, terminal cancellation, and correlated transfer events.

### Interfaces and connections

Allocation/free, lease commit/cancel, upload/flush, submitted aggregate prefix, and owned diagnostic/completion events connect API/workers to allocator, HAL, frame policy, and hosts.

### Data and persistence

GPU heaps, staging pools, bounded transfer queues, and token tables are runtime-owned. None survive context loss.

### Technology

Rust lifetime/state contracts hide HAL copies and timeline domains.

### Lifecycle and performance

Leases and blocks retire only by healthy completion. Device loss atomically fails queued/in-flight tokens and leases exactly once without reuse or GPU wait. Measure copy/submit/staging/stall/event cost and loss fan-out.

### Failures and trust

Reject invalid ranges/generations/lease states, stale frees, reuse before completion, and post-loss operations. Test rollback, rollover, overflow, and injected-loss completion cardinality.


## Component: Texture decode, compression, and streaming

### Responsibility and boundary

Own optional image/Basis/KTX2 decode, admitted-format selection, mip readiness, region updates, sample-ready descriptors, upload telemetry, and loss/cancellation outcomes.

### Problems and selected solutions

Realizes P-014/P-015/P-022/P-023/P-028 and texture portions of P-018/P-012: feature-gated transcoding, progressive streaming, one capability floor with per-format probes, deferred descriptors, and correlated events.

### Interfaces and connections

Decode/transcode -> validated payload; admitted capability -> format; region update -> validated layout; readiness/progress/error events connect workers, transfers, descriptors, frame policy, and hosts.

### Data and persistence

Host owns durable assets. Runtime owns bounded decoded payloads, GPU mips, sample-ready state, and phase/byte counters only until completion/cancellation/loss.

### Technology

Optional `basis-universal`, KTX2, and image adapters; feature-off runtime/release audits exclude native decoder linkage.

### Lifecycle and performance

Publish coarse then fine mips only while healthy. Loss cancels jobs, discards payloads/descriptors, and emits one terminal event. Measure decode/transcode/staging/queue/handoff/event latency, peak memory, descriptor delay, and quality.

### Failures and trust

Reject malformed containers, unsupported admitted formats, block/row/mip errors, early descriptors, unload/cancel/loss races, and duplicate terminal events. Fuzz all external dimensions/regions.


## Component: Surface, swapchain, and presentation readback

### Responsibility and boundary

Own safe surface/context leases, atomic surface construction rollback, extent/DPI/minimize, presentation, readback, and presentation-side loss reporting. It never owns host windows/event loops.

### Problems and selected solutions

Realizes P-017/P-023/P-026/P-028 and presentation portions of P-019/P-002: guarded presentation, terminal loss propagation, host-polled events, and correlated diagnostics.

### Interfaces and connections

Owning `Surface`, `begin_frame`, `Frame::finish`, borrowed host handles, resize state, and readback connect host to HAL/graph/FFI/events/snapshots.

### Data and persistence

Swapchain images/recreation/readback are runtime-local; host window lifetime is external. No surface state survives terminal device loss.

### Technology

`raw-window-handle` 0.6 adapters with native backend surface APIs inside HAL.

### Lifecycle and performance

Zero extent returns `NotReady`; resize retires safely. Begin returns acquisition errors, `Frame::finish` preserves submit/present errors, and `Frame::drop` aborts. Measure acquire/present/recreate/event/loss latency and minimized work.

### Failures and trust

Validate atomic construction rollback, host-handle lifetime, out-of-date/suboptimal paths, swapchain-read rejection, wrapper drop order, and exactly-once loss propagation.


## Component: Async task and transfer orchestration

### Responsibility and boundary

Own bounded decode/transcode scheduling, transfer ownership, backpressure, cancellation, CPU shutdown barriers, and production of owned events. It does not invoke host callbacks, decode formats, or own GPU resources.

### Problems and selected solutions

Realizes P-018/P-023/P-026/P-028 and orchestration portions of P-012/P-014/P-015: scoped Rayon work, bounded transfer channel, terminal fan-out, and host-polled event delivery.

### Interfaces and connections

Submit/cancel, bounded jobs/bytes, transfer handoff, owned completion/error/diagnostic events, and shutdown barrier connect API/events to texture/geometry.

### Data and persistence

Owns bounded job/payload queues until drain/cancel/loss. Events transfer ownership to the context queue; no durable task store exists.

### Technology

Rayon and bounded channels; workers never call application code or hold internal locks while publishing.

### Lifecycle and performance

Starts with context; shutdown closes producers then drains/cancels before event storage destruction. Device loss cancels queued/in-flight handoffs and produces exactly one terminal outcome. Measure scaling, queue bytes, throughput, RSS, event latency/overflow, and render stalls.

### Failures and trust

Full queues, decoder panic/error, stopped host polling, cancellation, reentrancy, oversubscription, shutdown, and loss remain bounded and observable; test callback-after-free prevention and exact terminal cardinality.

## Component: Runtime lifecycle, event, diagnostics, and profiling

### Responsibility and boundary

Own the context state machine, atomic `Running -> Lost -> Destroyed` transition, bounded per-context event queue, sequence/correlation assignment, typed local diagnostic/profiling schema, overflow accounting, and exactly-once terminal completion coordination. It never calls host code, persists telemetry, or recreates a device.

### Problems and selected solutions

Realizes P-023/P-026/P-028 and lifecycle/event portions of every async/GPU component. Device loss is terminal; hosts create a new context. Delivery is host-polled and local.

### Interfaces and connections

Components publish owned typed events with severity/category/correlation/domain/clock/unit/backend context; hosts poll/drain/release them. `poison(cause)` elects one transition owner, rejects new work, fans terminal state to surfaces/graph/transfers/workers/cache/allocator/handles, and returns only after CPU-side shutdown ordering is established.

### Data and persistence

Owns bounded event payloads, monotonic sequence numbers, dropped-event counters/loss markers, context state, and completion registry. No remote or durable telemetry exists; host persistence is outside this component.

### Technology

Atomic state transition plus bounded Rust queue/ring and owned FFI-safe event payloads. Backend clocks remain separate domains with declared units/availability; unspecified workloads are never combined.

### Lifecycle and performance

Producers start after context admission and close before queue destruction. The first fatal cause becomes canonical; every pending operation reaches one terminal result before shutdown completes, stale handles fail, and no GPU wait occurs. Measure publish/poll/wakeup/contention/overflow, disabled/enabled diagnostics, loss detection-to-notification, fan-out, retained bytes, and teardown.

### Failures and trust

Queue saturation emits a non-blocking loss marker/count; typed API errors never disappear into logs. Reject duplicate completion, callback-after-free, clock conflation, unbounded labels/payloads, post-destruction publication, and fatal-cause overwrite.

## Component: Validation, snapshot, and cutover gates

### Responsibility and boundary

Own unit/property tests, semantic/reflection/IR snapshots, backend goldens, capability-floor adapter/profile manifests, artifact/cache/loss/event fixtures, ABI smoke tests, six examples, release-install audits, and requirement-to-gate reports. It never weakens production semantics.

### Problems and selected solutions

Realizes P-019/P-020 and validation obligations of P-021 through P-029: real backend goldens plus deterministic snapshots, Vulkan-first staged parity, and complete selected-contract gates.

### Interfaces and connections

Fixtures exercise all components. Manifests bind stable adapter/driver identity, admitted semantic profile, shader/cache schema, tool provenance, backend, and tolerance policy; cutover consumes pass/fail/unavailable evidence, clean-package audits, and TODO dispositions.

### Data and persistence

Versioned semantic/IR snapshots, generated target code, PNG goldens, adapter/profile/tool metadata, loss/event traces, cache fixtures, package manifests, logs, and reviewed updates are durable test artifacts. Missing required profiles are not passes.

### Technology

Cargo tests, property/fuzz tooling, backend offscreen fixtures, deterministic snapshots, clean target installs, binary-import/signature audits, and exact or documented fixture-specific image tolerances.

### Lifecycle and performance

CPU schema/state tests run independently; GPU matrix admits each adapter through the same public floor before cache/golden use. Release jobs install each product cleanly. Measure runtime, flake/readback cost, matrix/build/install scaling, diagnostics perturbation, and loss/cache/event paths with driver/compiler/tool versions.

### Failures and trust

Reject stale goldens, malformed fixtures, silent adapter/profile absence, cache identity drift, missing Metal variants, compiler leakage, duplicate/missing terminal outcomes, unverifiable provenance, unsigned release products, and TODO/example gaps.


## Cross-cutting concerns

### Artifact and schema governance

Compiler/runtime/reflection/ABI/cache/golden/event/release schemas have explicit versions, bounded parsing, compatibility rules, owners, and invalidation keys. Canonical semantic IDs connect target layouts without exposing physical bindings. Digests identify complete execution sections; host/package policy owns authenticity.

### Capability and backend neutrality

One semantic floor covers bindless/indexing, indirect drawing, synchronization, aliasing, dynamic rendering, presentation, and at least one required compressed path. Public enumeration/admission produces stable profile/device identity consumed unchanged by formats, shader/metallib selection, cache keys, diagnostics, and test manifests. Optional raw limits never create semantic tiers.

### Ownership and synchronization

Resources flow API -> manager -> allocator/HAL; submitted prefixes flow from transfers to frame policy, while completion drives descriptors and events. Timelines are domain-specific. One lifecycle owner performs terminal loss fan-out and exactly-once completion. Hosts own windows/event loops, event polling, artifact authenticity, and durable cache bytes; runtime owns bounded live queues and cache validation only.

### Validation and release policy

All external inputs fail closed. Required adapter absence blocks cutover; explicitly optional exploratory profiles may skip. ABI compatibility, semantic/capability floor, Metal matrix, tolerance, authenticity, signing/notarization, six examples, and every TODO disposition are release gates.


## Coverage audit

### Pass 1

Pre-recursive pass pending. This document maps P-001 through P-020 and their selected solutions to cohesive components and cross-cutting concerns. A recursive audit must next inspect these component boundaries, interfaces, data flows, storage/cache/queue ownership, failure paths, deployment boundaries, and cross-component interactions for newly exposed consequential decisions; no fixed point is claimed.

### Pass 2 â€” recursive Step 1 after Step 4

The component-boundary, interface, end-to-end data-flow, mutable/durable storage, cache, bounded-queue, failure/trust, deployment, hot-path, and cross-component interaction auditâ€”reconciled with the independent coverage auditâ€”found nine consequential decisions without selected solutions:

- P-021: cross-backend shader execution semantics, including the canonical coordinate, layout, binding, and specialization ABI required for genuinely universal Slang shaders.
- P-022: backend/adapter selection and the exact capability-admission/tier policy consumed by HAL clients, caches, and validation profiles.
- P-023: device-loss state transitions, teardown, pending-work completion, handle invalidation, diagnostics, and any recovery boundary.
- P-024: persistent pipeline-cache storage ownership, atomicity, concurrency, bounds, invalidation, and read-only/sandbox behavior.
- P-025: the exact Metal shader product, Apple toolchain boundary, variant metadata, and runtime selection/deployment contract.
- P-026: public thread-affinity/concurrency rules and bounded asynchronous event/callback delivery across Rust, FFI, workers, and host event loops.
- P-027: artifact integrity, compiler/toolchain provenance, and the boundary between structural validation, content identity, and authenticity.
- P-028: a bounded local diagnostic/profiling schema that preserves typed failures and causal correlation across components without remote telemetry.
- P-029: cross-platform native build/linkage/distribution policy for backend, compiler, Basis, runtime-only, and C ABI deliverables.

Reconciliation rejected four duplicate partitions. The proposed backend-neutral resource-state/synchronization contract is already selected by P-003/P-008's neutral HAL transition contract and precise state compiler; implementation vocabulary belongs there. Capability tiers merge into P-022 rather than forming a second admission problem. Handle identity is selected by P-002, while its uncovered thread-safety/callback-ownership portion is P-026. Coordinated artifact/cache/golden lifecycle is already partitioned by P-006 artifact compatibility, P-019 reviewed golden/profile policy, P-024 cache persistence, and the aggregate schema-governance rules; another coordinator would duplicate those owners.

Each retained problem is decision-sized, crosses existing component ownership, and is not an implementation task or validation-only concern. No solution is selected in P-021 through P-029. Because this reconciled pass added problems, the plan has not reached a fixed point; Step 2 should investigate only P-021 through P-029 before the next recursive coverage pass.

### Step 4 integration

P-021 through P-029 are now integrated into the canonical component plan and flows. Ownership is explicit for semantic shader IDs versus target layouts/specialization, capability admission and its cache/test identity, terminal `Lost` fan-out and exactly-once outcomes, host-owned cache persistence/authenticity, Metal build/selection metadata, host-polled events, bounded local diagnostics, and native release products. P-004 is resolved by the selected allocator. A later recursive coverage pass is still required before claiming a fixed point.

### Coverage Pass 3 â€” recursive Step 1 after second Step 4

The component-boundary, interface, end-to-end flow, mutable/durable storage, cache, bounded-queue, failure/trust, deployment, performance, and cross-component interaction audit reached a fixed point. Every prompt requirement and consequential cross-cutting choice is owned by P-001 through P-029 with a selected solution.

No P-030 problem was created. Remaining specificsâ€”exact semantic-profile limits, source-convention constants, queue capacities, Apple/target matrix entries, cache envelope fields, diagnostic payload bounds, benchmark thresholds, and test fixturesâ€”are implementation or validation refinements inside P-021 through P-029, not new decision boundaries.
