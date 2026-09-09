# Selected solutions

> **Do not read the individual problem files.** This document is the canonical, self-contained decision record.

## System context

### Outcome

Migrate the original Odin/Vulkan `ez_gfx_api` into a Rust/Cargo implementation that roughly preserves its API and render-graph concepts; supports Vulkan, DirectX 12, and Metal; compiles one Slang source for all three; ships a runtime without Slang; supports Basis Universal/KTX2; replaces VMA with a Rust allocator; prepares the original TODO work; and supports behavioral and snapshot testing.

### Constraints and non-goals

- Shipping runtime packages must not depend on or bundle the Slang compiler.
- Vulkan, DX12, and Metal are required; Vulkan-only designs are incomplete.
- Explicit shader target attributes remain the source of truth for render-target intent.
- Rust exposes only the ownership-based interface. C/C# use the explicit ABI 34 lifecycle through `ez-gfx-ffi`.
- External inputs and binary artifacts require boundary validation.
- Measurements from different or unspecified workloads are not averaged. Missing performance data remains unknown.
- OpenGL, DX11, software rasterizers, a custom shader DSL, and a custom window system are out of scope.

### Assumptions

- ABI 34 is the non-Rust compatibility seam; it does not dictate safe Rust ownership or retain safe compatibility aliases.
- Supported hardware exposes sufficient modern bindless/indexing features. Devices below the declared capability floor receive an explicit unsupported error.
- Slang, DXC/signing tools, and Apple Metal tools are available in compiler/build environments, not runtime deployments.
- No repository-specific performance baseline exists for P-001 through P-020; every selected design carries explicit measurement actions.

## P-001: Cargo workspace and delivery boundaries — Strict workspace boundary

### Problem and required outcome

Separate compiler, runtime, optional decoders, backends, and FFI so runtime applications cannot accidentally ship Slang. This boundary feeds the API, HAL, compiler, and artifact decisions in P-002, P-003, P-005, and P-006.

### Decision

Use a virtual Cargo workspace with `ez-gfx-core` for API/graph/reflection types, `ez-gfx-runtime` for execution and precompiled-artifact loading, `ez-gfx-compiler` with in-process `shader-slang`/slang-rs bindings, and `ez-gfx-ffi` for `cdylib`/`staticlib`. Keep backend and decoder native dependencies outside core. The runtime graph must not depend on the compiler package. Use resolver 2 or 3 according to the selected MSRV.

### Performance and tradeoffs

Runtime throughput is not expected to differ from a feature-gated monolith, but that is unmeasured. Separate packages add manifests and schema seams while making runtime dependencies and native linkage inspectable. Clean/incremental build time and binary size are unknown.

### Rejected alternatives

- Single package with features: Cargo unions features across ordinary dependency edges, so another consumer can reintroduce Slang. Reconsider only if compiler exclusion becomes a documented configuration rather than an enforced boundary.

### Evidence

- [Cargo feature unification](https://doc.rust-lang.org/stable/cargo/reference/features.html#feature-unification) shows features are additive and unioned.
- [Cargo resolver v2](https://doc.rust-lang.org/stable/cargo/reference/resolver.html#feature-resolver-version-2) limits only specific unification contexts.
- [Slang in-process compilation API](https://shader-slang.org/docs/compilation-api/) and [shader-slang Rust bindings](https://crates.io/crates/shader-slang)
- Original `TODO.md` requires precompiled shaders and optional KTX2 linkage.

### Assumptions, risks, and validation

- Risk—compiler dependency leaks into runtime: fail a runtime-only `cargo tree` check if Slang/compiler libraries appear.
- Risk—package seams slow builds or drift: record clean/incremental times and validate shared artifact schemas.
- Validate runtime, compiler, decoder-disabled, and each backend package on its supported target.

## P-002: Public API and C ABI bindings — Owning Rust facade and raw FFI

### Problem and required outcome

Provide a safe ownership-based Rust interface while keeping a validated C seam. P-002 supplies lifecycle and identity semantics to graph, geometry, texture, surface, and drawing modules.

### Decision

`Context` owns `Rc<ContextInner>`. Every owning resource wrapper retains context and resource leases; `Drop` invalidates public identity and arranges deferred native retirement. The safe interface exposes no public destroy, release, or free functions.

`Surface::begin_frame()` and `Context::begin_frame()` return target-less owners. `Frame::configure_swapchain` attaches presentation; `Frame::configure_render_target` caches a named image and implicitly recreates it for size or format changes. `Frame::finish(self)` returns exact errors, while `Drop` aborts because it cannot report failure. `RenderTarget::prepare_readback` yields an opaque request delivered only through callback-scoped bytes.

`ez-gfx-ffi` remains a validated adapter. ABI 34 uses opaque generation- and owner-checked `u64` handles, including `EzGfxFrame`. Surface and managed-target begin functions return explicit frame owners; `ez_gfx_frame_end` and `ez_gfx_frame_abort` invalidate on every result, while context destruction aborts descendants. C buffers are one-frame values: their first valid recording claims them, same-frame reuse is allowed, and terminal paths invalidate claimed handles while completion gates native recycling. Validate pointer/count pairs, arithmetic, UTF-8, layouts, handles, out-pointers, and asynchronous ownership; contain every panic.

Surface construction is transactional: a native creation, device-initialization, or initial-resize failure destroys the unpublished raw surface and returns the original error without retaining a safe wrapper.

### Performance and tradeoffs

The safe interface gains lifetime locality at the cost of `Rc` increments and internal deferred retirement. Boundary validation and reference-count costs are unmeasured. Maintaining Rust ownership and C explicit lifecycles is intentional; compatibility aliases inside the safe facade are not.

### Risks and validation

Validate wrapper drop order, absence of reference cycles, exact submit/present errors, implicit abort, one-frame buffer claim and terminal invalidation, atomic surface rollback, stale/foreign/generation rejection, ABI 34 generated-contract freshness, layout probes, invalid calls, and panic containment.

## P-003: Multi-backend hardware abstraction — Custom static raw HAL

### Problem and required outcome

Support Vulkan, DX12, and Metal while exposing the queue, synchronization, bindless, indirect-draw, resource-aliasing, render, and presentation controls required by later allocator, pipeline, graph, and swapchain work. Depends on P-001.

### Decision

Define a narrow internal backend contract and implement it directly over `ash`, Windows D3D12 bindings, and `objc2-metal`. Keep capability discovery and barrier/state lowering backend-local. Compile a concrete backend so hot recording does not depend on trait-object dispatch.

### Performance and tradeoffs

Static dispatch removes virtual calls by construction, but its practical value and code-size cost are unknown. Direct bindings maximize feature reach and make backend differences explicit, at the cost of three unsafe implementations and the highest maintenance burden.

### Rejected alternatives

- `wgpu-hal`: backend coverage is attractive, but required bindless, indirect, aliasing, and synchronization escape hatches are not yet proven through its evolving unsafe interface. Reconsider after a capability spike proves every path without a private fork.
- Vulkan incumbent: hard failure because DX12 and Metal are required.

### Evidence

- [Vulkan dynamic rendering](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_dynamic_rendering.html), [D3D12 descriptor heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview), and [Metal argument buffers](https://developer.apple.com/documentation/metal/buffers/about_argument_buffers) expose the required native concepts.
- [`wgpu-hal`](https://docs.rs/wgpu-hal/latest/wgpu_hal/) documents native backend coverage but an unsafe low-level contract.
- Original `src/ctx.odin`, `src/render.odin`, and `src/swapchain.odin` establish the Vulkan baseline.

### Assumptions, risks, and validation

- Assumption—the project can sustain three unsafe backends.
- Risks are false semantic equivalence, state translation errors, capability-tier gaps, queue/fence lifetime bugs, and divergence.
- Before implementation, prove aliasing, bindless indexing, indirect drawing, timeline equivalents, dynamic rendering, and presentation in a capability spike for every backend; then benchmark identical recording workloads and binary sizes.

## P-004: GPU memory allocation — `gpu-allocator`

### Problem and required outcome

Replace VMA with a Rust allocator supporting dedicated/suballocated buffers, images, render targets, mapped staging, and transient aliasing on Vulkan, DX12, and Metal. Depends on P-003 and supplies allocation behavior to P-009, P-011, P-012, and P-014.

### Decision

Use `gpu-allocator` 0.28 behind the HAL allocation interface. Its published package provides Vulkan, D3D12, and Metal implementations and managed/dedicated allocations. Carry size, alignment, memory location, class, mapping, and retirement/alias lifetime through the interface.

### Performance and tradeoffs

No comparable allocation-latency, fragmentation, peak-memory, or contention benchmark exists. One cross-backend crate is substantially less implementation risk than three custom allocators, but transient alias control remains unproven.

### Rejected alternatives

- Vulkano `memory::allocator`: Vulkan-only, so it fails the three-backend requirement; it is also a module, not a verified crate named `memory-allocator`.
- Native per-backend allocators: survive as fallback if no requested crate meets requirements, but have the highest fragmentation, residency, synchronization, and maintenance risk.

### Evidence

- [`gpu-allocator` 0.28](https://docs.rs/gpu-allocator/latest/gpu_allocator/) documents Vulkan, D3D12, and Metal examples.
- [Vulkano allocator module](https://docs.rs/vulkano/latest/vulkano/memory/allocator/) is Vulkan-specific.
- Exact-name registry/docs checks found no published `memory-allocator` package during exploration.

### Assumptions, risks, and validation

- Verify future upgrades against license, maintenance, all three backend modules, and alias behavior.
- Replay representative buffer/image/staging/transient traces and record allocation latency distributions, committed/used bytes, fragmentation, coherency, contention, and alias correctness.

## P-005: Universal Slang compilation — Native multi-target Slang

### Problem and required outcome

Compile one Slang source and entry-point set to SPIR-V, DXIL, and Metal output with common bindless conventions and authoritative target/reflection metadata. Depends on P-001 and feeds P-006, P-007, P-008, and P-010.

### Decision

The offline compiler creates native `SLANG_SPIRV`, `SLANG_DXIL`, and `SLANG_METAL`/`SLANG_METAL_LIB` targets. DXIL variants target Shader Model 6.5: implemented binding uses explicit descriptor tables and root descriptors, so no Shader Model 6.6 direct-heap-indexing semantic is required. Extract target declarations and canonical interface metadata before optimization can erase intent; emit target blobs and metadata to P-006. Validate target-specific binding layouts instead of introducing another semantic compiler path.

### Performance and tradeoffs

Compile time, peak RSS, blob size, and shader-runtime performance are unknown. Native targets retain one frontend/reflection authority but still require DXC/signing support and Apple/Xcode tools where appropriate, plus target-specific legalization checks.

### Rejected alternatives

- SPIR-V pivot through SPIRV-Cross/Naga: those tools emit HLSL/MSL source rather than DXIL, add remapping, and risk reflection drift. Retain only as fallback for a native target defect.
- Runtime Vulkan compilation: hard failure because it bundles Slang and lacks DX12/Metal.

### Evidence

- [Slang targets](https://shader-slang.org/slang/user-guide/targets), [compilation API](https://shader-slang.org/docs/compilation-api/), and [reflection](https://shader-slang.org/slang/user-guide/reflection) document native target generation.
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) documents HLSL/MSL output, not direct DXIL.
- Original `src/shader.odin` proves the Vulkan/Slang attribute baseline.

### Assumptions, risks, and validation

- Assumption—the shared Slang subset covers current shaders and compiler environments provide native tools.
- Risks include stripped attributes, binding differences, matrix/layout changes, compiler-version drift, and unavailable signing/Xcode tools.
- Compile every shader/entry point for all targets in CI, compare canonical metadata, create pipelines/render snapshots, and record compiler version/options, time, RSS, and blob size.

## P-006: Precompiled shader container and reflection — Framed, validated `rkyv`

### Problem and required outcome

Ship multi-backend shader code and complete reflection without Slang at runtime, while keeping explicit target declarations authoritative. Depends on P-005; feeds P-007 and P-008.

### Decision

Use one `.ezgfxshader` file with a fixed 56-byte little-endian frame containing magic, format version, reserved flags, payload length, and BLAKE3 digest. Store stage-grouped target products, canonical reflection, compiler provenance, and one internal entry point per stage in a bounded `rkyv` payload. Verify exact framing and digest, copy into aligned storage, run `bytecheck`, deserialize, and validate semantic/target invariants before allocation or backend calls. Never invoke Slang at runtime.

### Performance and tradeoffs

Cold/warm load time, RSS, copy behavior, and artifact size remain workload-dependent. The bounded aligned copy permits safe validation of arbitrarily aligned caller bytes before deserialization. A single artifact gives atomic deployment and hashing; explicit framing and `rkyv` validation replace the former custom section parser.

### Rejected alternatives

- Manifest plus sidecars: partial deployment, path confinement, and synchronization risks are higher.
- Custom section encoding or generated schema tooling: both add owned parsing/evolution machinery already covered by framed `rkyv`; reconsider only if compatibility requirements exceed the explicit format-version cutover.

### Evidence

- [DXIL container](https://github.com/microsoft/DirectXShaderCompiler/blob/main/include/dxc/DxilContainer/DxilContainer.h) and [KTX2](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html) provide mature typed-section/index patterns.
- [Vulkan shader-module validation](https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateInfo.html) requires strict SPIR-V size/content validation.
- Original `TODO.md` requires precompiled shaders and runtime/compiler separation.

### Assumptions, risks, and validation

- Risks include truncation, integer overflow, duplicate/overlapping sections, stale schemas, reflection/blob mismatch, and treating hashes as signatures.
- Fuzz malformed containers, round-trip compiler output, reject incompatible versions, and load every target with Slang absent.
- Measure cold/warm load, opens, allocations, RSS, and artifact size on named OS/storage.

## P-007: Pipeline caching and descriptors — Global table plus frame-local arenas

### Problem and required outcome

Decouple PSO cache ownership from descriptor lifetime, persist valid backend pipeline caches, and provide bindless resources across Vulkan, DX12, and Metal. Depends on P-003/P-006 and feeds P-008/P-016.

### Decision

Use a device-level bindless registry with stable, generation-checked indices plus a linear descriptor arena per in-flight frame, reset only after GPU completion. PSO records own layouts and pipeline objects, never pools/sets. Cache keys include shader/interface, attachment/fixed state, backend, device/driver, and schema identity.

### Performance and tradeoffs

Descriptor-update latency, memory, contention, capacity, and cache startup are unknown. Linear transient allocation avoids fine-grained frees by design; stable tables reduce rebinding, but hardware tiers and capacities differ. Two lifetime classes add explicit complexity.

### Rejected alternatives

- Global table only: incomplete transient/dynamic descriptor policy. Reconsider if every descriptor can be proven persistent.
- Pipeline-owned descriptors: hard failure because it preserves the explicit TODO's lifetime coupling.

### Evidence

- [Vulkan descriptor indexing and lifetime rules](https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html) define update/pool/reset synchronization.
- [D3D12 heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview) support bulk allocation and warn about heap switching.
- [Metal argument buffers](https://developer.apple.com/documentation/metal/buffers/about_argument_buffers) provide indexed resource tables.
- Original `src/pipeline.odin` demonstrates the coupled incumbent.

### Assumptions, risks, and validation

- Assumption—supported devices meet declared indexing/tier limits.
- Risks include stale slot reuse, early arena reset, pool exhaustion, heap switches, cache incompatibility, and sampler/resource namespace differences.
- Snapshot queried limits; stress maximum live/per-frame descriptors and delayed completion; corrupt caches; measure update latency, allocations, memory, startup hits/misses, and heap switches.

## P-008: Render-graph hazards — Precise subresource state compiler

### Problem and required outcome

Compile shader-declared dependencies into a DAG with precise buffer/image hazards and minimal required backend transitions, including sampled, storage, indirect, attachment, queue, and history cases. Depends on P-003/P-006/P-007 and feeds P-009/P-016.

### Decision

Track buffer ranges and image subresources with queue, stage, access, layout/state, and last readers/writers. Lower neutral transitions to Vulkan Synchronization2, D3D12 enhanced barriers, and Metal encoder/fence/event operations. Persist final states for history resources; transient resources begin undefined.

### Performance and tradeoffs

The `<0.1 ms` graph target is unverified. Compile complexity scales with edges and tracked intervals/subresources; constants and allocations are unknown. Fine tracking can preserve concurrency but has the highest correctness burden and an unmeasured GPU benefit.

### Rejected alternatives

- Whole-resource conservative tracker: cannot satisfy independent subresource/range access; retain only as debug/bring-up mode.
- Blanket barriers: hard failure against the tracked-hazard TODO.

### Evidence

- [Vulkan Synchronization2](https://docs.vulkan.org/guide/latest/extensions/VK_KHR_synchronization2.html) and [D3D12 enhanced barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/enhanced-barriers) expose precise state models.
- [Metal resource synchronization](https://developer.apple.com/documentation/metal/resource_synchronization) defines Metal ownership/synchronization mechanisms.
- Original `TODO.md` identifies blanket barriers and incomplete sampled/writable hazards.

### Assumptions, risks, and validation

- Assumption—declarations/reflection completely describe accesses.
- Missing RAW/WAR/WAW edges, ownership transfers, or cross-frame states can corrupt output.
- Property-test hazard graphs, compare against API validation, snapshot schedules/barriers, run fork/join/history/storage/sampled/indirect/subresource cases, and benchmark by graph dimensions on named CPU/GPU.

## P-009: Pass coalescing and transient aliasing — Integrated greedy compiler

### Problem and required outcome

Merge compatible adjacent graph nodes and reuse backing memory for non-overlapping transient targets without aliasing persistent history. Depends on P-004/P-008 and supplies physical target resolution to P-010.

### Decision

Use the selected topological order, greedily merge nodes whose attachments, formats, samples, areas, load/store behavior, and transition needs are compatible. Compute first/last-use intervals, best-fit compatible transient resources into reusable heap ranges, and emit backend-specific alias boundaries.

### Performance and tradeoffs

Peak VRAM reduction, graph CPU time, pass-count effect, and GPU time are unknown. One deterministic greedy pass is bounded and testable but can miss a globally better placement. Correctness depends on queue-aware lifetimes and backend compatibility classes.

### Rejected alternatives

- Multi-schedule/global placement: added compile/search and parallelism tradeoffs lack evidence. Reconsider if representative traces show materially lower memory within a fixed budget.
- Unaliased independent passes: hard failure because both TODOs remain.

### Evidence

- [Vulkan aliasing](https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-memory-aliasing), [D3D12 aliasing barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12#aliasing-barrier), and [Metal aliasability](https://developer.apple.com/documentation/metal/mtlresource/makealiasable()) define backend rules.
- [Filament frame graph](https://github.com/google/filament/tree/main/libs/fg) provides mature frame-graph implementation evidence.
- Original TODO lines 41-43 require merging and aliasing.

### Assumptions, risks, and validation

- Assumption—intervals include asynchronous queue overlap and compatibility includes every native requirement.
- Risks include off-by-one lifetimes, missing alias barriers, invalid load/store merging, and history reuse.
- Test deterministic intervals, poison reused memory, exercise queue overlap/history exclusion, compare image snapshots, and measure committed/used bytes, aliases, merged passes, graph time, and GPU timestamps.

## P-010: Target declarations and formats — Shader authority with runtime probing

### Problem and required outcome

Keep shader attributes authoritative for target intent, probe supported cross-backend depth/stencil formats, prefer lower precision where declared sufficient, model rich usage, and provide target clear values. Depends on P-003/P-006 and informs P-008/P-009.

### Decision

Compile attributes into canonical kind, usage, scale, sampleability, load/store, abstract format candidates, and default clear values. Resolve physical formats through backend capability queries at device creation. Include the resolution in pipeline/target keys and report unsupported intent explicitly. Host clear overrides require an explicit declaration capability and cannot alter format or usage intent.

### Performance and tradeoffs

Probe time, clear cost, and frame impact are unknown. D16 has fewer raw bytes per texel than D32, but actual bandwidth/performance depends on compression, tiling, and hardware. Centralized probing improves portability while increasing mapping and diagnostic complexity.

### Rejected alternatives

- General host instance policy: unrestricted overrides weaken shader authority. Reconsider only as an explicitly declared dynamic-clear capability.
- Hardcoded mapping: hard failure because it can choose unsupported formats, silently alter precision/aspects, and lacks rich clears.

### Evidence

- [Vulkan format queries](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceFormatProperties2.html), [D3D12 format support](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-feature-levels#format-support), and [Metal pixel formats](https://developer.apple.com/documentation/metal/mtlpixelformat) provide capability mechanisms.
- [Vulkan rendering pipeline formats](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineRenderingCreateInfo.html) require selected attachment formats in pipeline state.
- Original TODO lines 13, 19, and 37-39 define authority, probing, metadata, and clear requirements.

### Assumptions, risks, and validation

- Assumption—attributes can express unambiguous precision/aspect candidate classes.
- Risks include D24 portability, aspect loss, unsupported usage/sample combinations, reflection stripping, and pipeline-key mismatch.
- Build a backend capability matrix; test candidate ordering and diagnostics; snapshot metadata; measure probe time, allocated bytes, clear GPU time, and equivalent D16/D24/D32 workloads.

## P-011: Vertex and index geometry heaps — Owning generation-checked leases

### Problem and required outcome

Manage named bindless vertex heaps and one singleton context-owned index heap with validated stride/capacity and deterministic stale, foreign, and duplicate raw-handle rejection.

### Decision

GPU ranges use ordered free lists and generation-indexed identities. Safe heap and allocation wrappers retain their context and parent-resource leases; dropping an allocation retires its range, and dropping a heap retires it after child leases and recorded uses. The safe interface has no remove, destroy, release, or free operation. ABI 34 retains explicit opaque-handle release for C.

`Buffer<T>` and `CounterBuffer<T>` belong to `Context` for one frame. Their first valid binding claims them; repeated compute/graphics use in that frame shares one native materialization, and every later write or frame use fails. Native reuse remains completion-gated or quarantined after an indeterminate failure.

The slice upload path copies caller bytes into runtime-owned mapped staging. A direct caller-writable staging lease is not part of the implemented public interface and is not claimed here.

### Performance and tradeoffs

Indexed validation is constant-time by data-structure design. Range free lists can fragment externally, and reference-count/resource leases defer parent reclamation while children remain live. No end-to-end latency or memory-bandwidth benefit is claimed.

### Risks and validation

Test singleton index-heap admission, allocation/drop/coalescing, parent-before-child drop, generation rollover, failed-upload rollback, stale/foreign C handles, transient invalidation after submit and abort, and no native reuse before completion.

## P-012: Transfer staging and batching — Timeline-recycled bucket pools with adaptive batches

### Problem and required outcome

Recycle host-visible staging memory, batch vertex/index/texture copies, and remove shared timeline serialization without synchronously waiting per upload. Depends on P-003/P-004/P-011; supplies staging and submitted transfer prefixes to P-013/P-015.

### Decision

Use power-of-two staging buckets with dedicated fallback for oversize requests. Each block records its submission timeline and is reusable only after completion. A transfer engine coalesces copy requests into command buffers and flushes at explicit readiness requests, frame boundaries, or measured byte/count thresholds. Keep ordered timeline domains per transfer submission stream rather than making vertex and texture managers serialize each other. The transfer engine owns pools, batches, trimming, and completion publication.

### Performance and tradeoffs

Pooling removes repeated allocation/destruction by construction; batching reduces submit count. Driver cost, DMA throughput, optimal bucket sizes, thresholds, memory high-water mark, and readiness latency are unmeasured. Buckets can retain excess memory; delayed flushes can postpone first use.

### Rejected alternatives

- One fixed mapped ring: fast bump allocation but imposes a rigid capacity and requires fallback/stall behavior for large assets.
- Dedicated staging and submit per upload: preserves incumbent allocation/submit churn and fails batching TODOs.

### Evidence

- [NVIDIA Vulkan dos and don’ts](https://developer.nvidia.com/blog/vulkan-dos-donts/) recommends batching submissions and avoiding unnecessary allocations.
- [Vulkan transfer-queue guidance](https://docs.vulkan.org/guide/latest/transfer_queue.html) describes transfer command/queue operation.
- Original `src/vertex_manager.odin`, `src/texture_manager.odin`, and `TODO.md` identify pool, batching, and timeline issues.

### Assumptions, risks, and validation

- Assumption—P-003 supplies comparable completion values and queue dependencies on each backend.
- Risks include high-water memory retention, starvation behind batch thresholds, oversize churn, non-coherent flush errors, and completion-domain mixups.
- Stress repeated mixed-size uploads; verify no reuse before completion; compare allocations, submits, staging bytes/capacity, queue latency, readiness latency, and CPU recording time across threshold policies; test idle trimming.

## P-013: Explicit frame ownership

### Decision

Presented and managed-target begin functions return owning `Frame` values; all recording requires `&mut Frame`.

`Frame::finish(self)` aborts and returns any prior recording error unchanged, otherwise submits and presents surface frames only after successful submission. It preserves the exact error from each phase. Dropping an unfinished frame aborts. Claimed `Buffer<T>` and `CounterBuffer<T>` values support same-frame reuse, then become invalid on every terminal path; native backing remains completion-gated or quarantined internally.

ABI 34 exposes opaque generational `EzGfxFrame` handles from surface or managed-target begin. C terminates them with `ez_gfx_frame_end` or `ez_gfx_frame_abort`; every result invalidates the frame and its claimed buffer handles, and context destruction aborts descendants.

### Risks and validation

Validate exact recording/submit/present errors, implicit abort, transient invalidation after every terminal path, stale/foreign/double-completed C frames, and Rust/C/header/XML/export/layout parity.

## P-014: Basis Universal and compressed textures — Feature-gated official transcoder wrapper

### Problem and required outcome

Load `.basis` and KTX2 UASTC/ETC1S, choose supported BC/ASTC outputs, ingest already compressed blocks directly, and make decoder linkage optional. Depends on P-001/P-003/P-004; supplies formats and upload payloads to P-015.

### Decision

Gate the `basis-universal` wrapper and native C++ transcoder behind a `basis` Cargo feature in the compiler/decoder package, not core runtime. Query backend format/usage support before selecting BC7, BC3/BC1, ASTC, or validated RGBA fallback. Direct BCn/ASTC KTX2 payloads bypass transcoding when their format and block geometry are supported. Decoder ownership ends at a validated staging payload and metadata.

### Performance and tradeoffs

BC1 nominally stores 4 bits/texel; BC3/BC7/ASTC 4×4 store 8 versus 32 for RGBA8. These format ratios are exact, but realized VRAM, bandwidth, quality, disk size, and transcoding throughput are workload/device dependent and unmeasured. Enabling the wrapper adds a C++ build toolchain and native code.

### Rejected alternatives

- Pure-Rust KTX2 parser with precompressed assets only: cannot satisfy universal Basis transcoding without an external plugin.
- Uncompressed expansion only: hard failure against compressed/Basis support.

### Evidence

- [Basis Universal](https://github.com/BinomialLLC/basis_universal) documents ETC1S/UASTC targets and transcoder behavior.
- [`basis-universal` Rust bindings](https://docs.rs/basis-universal) expose the official transcoder wrapper.
- [KTX specification](https://github.khronos.org/KTX-Specification/) defines container and block payload semantics.
- Original `TODO.md` requires compressed formats and optional KTX2 linkage.

### Assumptions, risks, and validation

- Assumption—a C++ compiler is acceptable only when the feature is enabled.
- Risks include unsupported target choice, alpha/quality mismatch, malformed container sizes, block-edge padding, and accidental decoder linkage in minimal runtime builds.
- Test Basis/KTX2/direct BC/ASTC on all backends; fuzz dimensions/levels/offsets; verify feature-off dependency trees; measure transcode MB/s, peak RSS, staging bytes, artifact size, and sampled output quality on named assets/CPUs.

## P-015: Texture streaming and partial updates — Progressive mip streamer

### Problem and required outcome

Make coarse mips sample-ready before fine mips, update texture subregions for dynamic atlases, publish descriptors only after graphics-ready transitions, and expose upload telemetry. Depends on P-007/P-012/P-014; feeds UI rendering in P-016.

### Decision

Allocate the full image/mip set, upload an initial declared mip range, and publish the bindless descriptor only after queue handoff establishes sample-ready state. Later batches update finer subresources without changing the stable descriptor slot. Expose a validated region-update API carrying mip, offset, extent, row layout, and bytes; enforce compressed-block boundaries. Record per-job phase timestamps and byte counts, aggregating telemetry without placing contended atomics in per-byte loops.

### Performance and tradeoffs

For the documented example dimensions, a 64×64 RGBA8 update is 16 KiB versus 16 MiB for a 2048×2048 full image: exactly 1024× fewer payload bytes. Those dimensions are illustrative, not a general frame-time benchmark. Streaming latency, descriptor publication delay, transition cost, and telemetry contention are unknown.

### Rejected alternatives

- Virtual-texture atlas/page system: exceeds scope and requires page tables/shader coordinate policy.
- Full synchronous reload: fails partial update, progressive readiness, and deferred descriptor requirements.

### Evidence

- [Vulkan image subresources](https://docs.vulkan.org/guide/latest/image_subresources.html) defines mip/layer transition granularity.
- [Vulkan resource rules](https://docs.vulkan.org/spec/latest/chapters/resources.html) define compressed copy constraints.
- Original `src/texture_manager.odin`, `src/imgui.odin`, and `TODO.md` identify streaming, atlas, readiness, and profiling work.

### Assumptions, risks, and validation

- Assumption—callers or decoders provide dirty rectangles and valid mip payloads.
- Risks include sampling unavailable mips, publishing descriptors early, compressed-block misalignment, row-pitch errors, and telemetry perturbation.
- Test progressive visibility, cancellation/unload races, region boundaries, compressed edges, and delayed handoffs; compare atlas snapshots; record transferred/staged bytes, decode/queue/handoff/callback latency, frame time, and counter overhead.

## P-016: Multi-draw indirect and dynamic state — Scissor-batched MDI

### Problem and required outcome

Acquire/populate/execute indexed indirect buffers while adding portable per-draw viewport/scissor controls and replacing ImGui fragment `discard` clipping. Depends on P-003/P-007/P-008 and P-015 for UI textures; supplies scenes to P-019.

### Decision

Use the standard indexed indirect command layout plus a side table of validated viewport/scissor state associated with command ranges. During recording, group consecutive commands with equal dynamic state, set native viewport/scissor once per group, and issue the largest supported indirect range. Compute-generated indirect buffers use the same draw layout; CPU side tables remain explicit where native indirect commands cannot encode dynamic state.

### Performance and tradeoffs

Hardware scissors prevent fragment work outside the rectangle by rasterization semantics, but actual invocation/frame savings are unmeasured. Distinct rectangles split one monolithic MDI call into batches and add recording commands. The balance depends on UI clip distribution and backend indirect limits.

### Rejected alternatives

- Device-generated commands with inline scissor: hard portability failure across required Vulkan devices and Metal.
- Shader-discard clipping: retains unnecessary fragment work and fails the hardware-scissor TODO.

### Evidence

- [Vulkan dynamic state](https://docs.vulkan.org/guide/latest/dynamic_state.html) documents viewport/scissor commands.
- [D3D12 `ExecuteIndirect`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-executeindirect) defines indirect execution constraints.
- Original `src/indirect_buffer.odin`, `src/imgui.odin`, and `TODO.md` establish the incumbent flow and requirement.

### Assumptions, risks, and validation

- Assumption—UI clip rectangles can be associated with contiguous draw ranges.
- Risks include invalid negative/overflowed rectangles, wrong framebuffer scaling, state leakage, unsupported command counts, and excess batching.
- Clamp/validate rectangles; snapshot scaled/offscreen/UI output; test CPU- and compute-filled indirect buffers; measure batches, state changes, recording time, indirect count, fragment invocations, and GPU pass time against discard clipping.

## P-017: Surface lifecycle and swapchain presentation — Owning surface with guarded presentation

### Problem and required outcome

Interoperate with host-owned native windows, keep safe context/surface lifetimes valid in either drop order, roll back partial construction, handle resize/minimize, and preserve exact presentation errors.

### Decision

An owning `Surface` retains `Rc<ContextInner>` and its resource lease while the host retains the native window/display. Construction is atomic: native creation, device initialization, and initial resize either succeed together or destroy the unpublished raw surface and return the original error without retaining a safe wrapper.

`Surface::begin_frame()` returns a target-less frame; `configure_swapchain(size, format)` acquires presentation after resize handling. `Frame::finish(self)` preserves exact errors; dropping aborts. Named target recreation and presentation acquisition remain internal. Readback is requested opaquely and delivered only during the unified callback.

### Performance and tradeoffs

Skipping zero-extent work is deterministic. Recreation cost and presentation latency remain unmeasured. Resource leases prevent safe dangling context/surface relationships but cannot own the host's native window.

### Risks and validation

Inject every construction failure point; verify complete rollback, original errors, and drop-order safety. Exercise resize/minimize/restore/DPI and exact submit/present errors. Verify presentation-target shader-read rejection and screenshot readback.

## P-018: Async workers — Scoped Rayon compute pool and bounded transfer channel

### Problem and required outcome

Decode/transcode assets off the render thread, move completed payloads to one transfer submission owner, configure CPU concurrency, and route completion safely across Rust/C ABI without a heavyweight async runtime. Depends on P-001; schedules work for P-012/P-014/P-015.

### Decision

Create a library-owned Rayon pool for CPU-bound decode/transcode and a dedicated transfer worker that receives validated payloads over a bounded `crossbeam-channel`. Capacity is defined in bytes/jobs and applies backpressure before decoded memory grows without bound. Cancellation and shutdown are explicit; application callbacks are queued to a documented dispatch context rather than invoked while internal locks are held.

### Performance and tradeoffs

Work stealing is appropriate for heterogeneous CPU tasks, but scaling, queue contention, binary size, memory, and latency are unmeasured. A dedicated transfer owner serializes native queue submission intentionally while CPU decode remains parallel. The prior estimates of fixed binary size or 50–200 ms decode stalls are unsupported and are not used.

### Rejected alternatives

- Fixed OS-worker queues: less adaptive for heterogeneous tasks; reconsider if Rayon overhead measures worse.
- Tokio/full async runtime: violates the lightweight-runtime non-goal without an I/O requirement that justifies it.
- Synchronous decode: hard failure because it blocks callers.

### Evidence

- [Rayon](https://docs.rs/rayon/latest/rayon/) documents custom work-stealing thread pools.
- [Crossbeam channel](https://docs.rs/crossbeam-channel/latest/crossbeam_channel/) documents bounded channels and blocking/select behavior.
- Original `src/texture_manager.odin`, `src/vertex_manager.odin`, and `src/defs.odin` provide incumbent worker/queue evidence.

### Assumptions, risks, and validation

- Assumption—decoders are safe for independent concurrent instances and thread creation is allowed.
- Risks include backpressure deadlock, shutdown races, callback reentrancy, panic propagation, priority inversion, and oversubscription.
- Test cancellation/shutdown/full-channel/decoder-panic behavior; verify callback thread guarantees through Rust and C; benchmark asset mixes across worker counts, recording utilization, throughput, queue depth/bytes, peak RSS, callback latency, and render-thread stalls.

## P-019: Tests and golden snapshots — Real backend offscreen goldens

### Problem and required outcome

Provide allocator/parser/graph unit tests and deterministic automated visual integration tests for simple rendering, storage images, scaled targets, history, and fork/join graphs across required backends. Depends on P-003/P-008/P-016; gates P-020.

### Decision

Build backend-specific offscreen render/readback fixtures and store versioned PNG goldens per backend/profile only where output differences require them. Use exact comparison for integer/deterministic fixtures and an explicitly justified per-channel/region tolerance for floating-point fixtures. Every unavailable adapter reports a skipped/unavailable capability, never a pass. Complement image tests with deterministic unit tests for graph IR, barriers, reflection, allocators, and container parsing; image goldens remain the end-to-end gate.

### Performance and tradeoffs

Adapter startup, shader/pipeline creation, readback, matrix runtime, storage, and cross-driver determinism are unknown. Real GPU tests catch backend execution defects that IR snapshots cannot, but require hardware/software adapters and disciplined tolerance review.

### Rejected alternatives

- IR snapshots alone: insufficient for rasterization, format, synchronization, and shader-codegen defects; retained as complementary unit coverage.
- Manual window screenshots: non-deterministic and cannot satisfy automated CI.

### Evidence

- [Vulkan headless surface](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_headless_surface.html) and [dynamic rendering info](https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderingInfo.html) support headless/offscreen fixtures.
- [`insta`](https://docs.rs/insta/latest/insta/) supports deterministic Rust data snapshots for complementary IR tests.
- Original `tests/snapshot.odin`, snapshot assets, and `TODO.md` establish existing coverage and missing graph shapes.

### Assumptions, risks, and validation

- Assumption—CI provides declared hardware or software adapters for each required profile.
- Risks include false failures from driver/color-space/rounding changes, over-broad tolerances, stale goldens, and silent missing coverage.
- Calibrate tolerances from repeated runs without hiding single-pixel defects; record adapter/driver/profile; run examples plus targeted fork/join/scaled/history/storage fixtures; measure runtime and flake rate; require reviewed golden updates.

## P-020: Migration cutover — Vulkan-first vertical slice with staged parity gates

### Problem and required outcome

Sequence all P-001 through P-019 decisions into bounded milestones, obtain executable evidence early, and define final cutover without a big-bang port or untracked TODO deferral. This terminal coordinator depends on every prior problem.

### Decision

Start with one end-to-end backend-neutral vertical slice, then require Vulkan, DX12, and Metal conformance, compressed assets/streaming/UI, and remaining selected work. Final cutover uses the shared `Example` host for winit inversion, owning context/surface, resize, input, automation, and consuming frame dispatch. It requires all six examples, ABI 34 gates, backend-required snapshots, every inherited TODO disposition, and no obsolete safe handle/free or multi-begin compatibility path.

### Performance and tradeoffs

Time-to-first-snapshot, defect discovery, and total migration duration are unknown. Vulkan-first gives earlier end-to-end evidence than a full platform matrix but risks encoding Vulkan assumptions. Backend-neutral contracts and an immediate cross-backend capability gate mitigate rather than eliminate that risk.

### Rejected alternatives

- Cross-backend horizontal foundation first: delays executable feature evidence until every toolchain/adapter is ready; retained as the second parity gate.
- CPU-only headless core first: useful preflight tests but cannot validate shader binaries, transitions, or pixels; retained as supporting tests.

### Evidence

- [Cargo workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html) support package-scoped milestone validation.
- [Cargo test](https://doc.rust-lang.org/cargo/commands/cargo-test.html) supports targeted package/test gates.
- Original six examples, snapshot harness, `README.md`, `CONTEXT.md`, and `TODO.md` define observable migration scope.

### Assumptions, risks, and validation

- Assumption—a Vulkan environment is available immediately and slice contracts do not expose Vulkan-specific public semantics.
- Risks include deferred backend incompatibility, “temporary” shims becoming permanent, untracked TODOs, and declaring cutover before C#/asset parity.
- Record time-to-first-snapshot and defects by milestone; run cross-backend contract fixtures before expanding features; maintain a requirement-to-gate matrix; require all examples, ABI smoke tests, selected backend snapshots, and resolved TODOs before final cutover.

## P-021: Cross-backend shader execution semantics — Target-native layouts with canonical semantic ABI

### Problem and required outcome

One Slang source must render equivalently on Vulkan, DX12, and Metal without Slang at runtime. The contract must define coordinates, winding, matrix/constant layout, logical bindings, specialization, and texture/sampler semantics while roughly preserving API concepts. Depends on P-003/P-005/P-006/P-007/P-010.

### Decision

Use the root `ez_gfx_api.slang` module and stable, collision-safe semantic resource IDs, but retain target-native products and reflection in `.ezgfxshader`. Every application shader imports the shared module and contains no Vulkan namespace/location/register syntax. Each stage has one artifact-owned entry point, so runtime callers select only the artifact and backend/profile. DXIL uses Shader Model 6.5 with explicit descriptor tables/root descriptors rather than Shader Model 6.6 direct heap indexing.

### Performance and tradeoffs

Target-native layouts avoid an owned Vulkan-to-DX12/Metal physical remapper, but add reflection bytes, semantic lookup, and per-target CPU packing. No comparable benchmark establishes lookup, artifact-size, pipeline, or GPU-time impact; measure them on identical scenes and named adapters.

### Rejected alternatives

- Vulkan-compatible logical/physical ABI with generated adapters: unrequested physical compatibility, unproven wrapper/remap cost, and high double-transform/layout risk. Reconsider only if incumbent consumers require persisted raw Vulkan bindings or byte layouts.

### Evidence

- [Slang reflection](https://shader-slang.org/slang/user-guide/reflection) separates declarations from target-specific layouts.
- [Slang targets](https://shader-slang.org/slang/user-guide/targets) documents distinct D3D12, Vulkan, and Metal parameter models.
- [Slang Metal behavior](https://shader-slang.org/slang/user-guide/metal-target-specific) documents entry, binding, matrix, and specialization legalization.

### Assumptions, risks, and validation

- Assumption—rough compatibility preserves concepts, not Vulkan physical layouts.
- Risks include reflection drift, semantic-ID collision, packing mismatch, coordinate double transforms, and artifact growth.
- Validate deterministic IDs and cross-target type/access sets; snapshot reflection/code; test packing, matrices, specialization, samplers, and coordinates; run backend goldens; measure lookup, artifact, compile/pipeline, instruction, and GPU costs.

## P-022: Backend, device, and capability admission — Single modern semantic floor

### Problem and required outcome

Rust/FFI callers need deterministic backend/adapter selection and an exact admission policy covering bindless resources, indirect draws, synchronization, aliasing, rendering, compression, presentation, and cache/snapshot identity. Unsupported devices must fail explicitly. Depends on P-002/P-003 and supplies P-007/P-008/P-009/P-014/P-016/P-017/P-019/P-020.

### Decision

Expose opaque stable adapter records, explicit selection, and a documented deterministic default. Admit one mandatory semantic profile mapped to native features and minimum capacities; the shader-model floor is 6.5 because current DXIL semantics use explicit descriptor tables/root descriptors and no 6.6-only feature. Probe format/usage separately. Reject missing requirements before manager creation with detailed diagnostics. Cache/golden keys use backend, stable device identity, driver, and profile schema; software adapters require explicit opt-in. Optional raw capabilities remain queryable but do not create semantic tiers.

### Performance and tradeoffs

One floor avoids hot-path/profile branching and multiplies neither cache nor conformance paths, but can exclude older hardware. Enumeration/probe latency, admitted-device coverage, and workload impact are unknown. Enable required features only; Vulkan notes some advertised features can cost performance when enabled.

### Rejected alternatives

- Baseline plus optional tiers: no measured coverage need justifies multiple HAL/cache/test paths; any semantics-changing tier is invalid. Reconsider only if spikes show meaningful exclusion and a lower path preserves pixels, hazards, handles, and artifacts.

### Evidence

- [Vulkan devices/properties](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html) defines enumeration, IDs, limits, and cache UUID.
- [Vulkan features](https://docs.vulkan.org/spec/latest/chapters/features.html) defines explicit query/enablement and possible feature cost.
- [D3D12 feature queries](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-checkfeaturesupport) and [Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) expose native tiers and limits.

### Assumptions, risks, and validation

- Assumption—modern-hardware exclusion is acceptable for the required semantics.
- Risks include an excessive floor, unstable identity, poor default ranking, and costly unused features.
- Build and test the semantic-to-native matrix on named devices; verify every diagnostic/default; record coverage, startup, limits, enabled features, cache keys, binary size, recording time, and GPU time before freezing the floor.

## P-023: Device loss and runtime recovery — Terminal lost runtime

### Problem and required outcome

Device loss/removal must transition queues, resources, pending jobs, callbacks, handles, caches, and FFI callers without deadlock, stale reuse, unwinding, or stranded work. Depends on P-002/P-003/P-004/P-007/P-012/P-017/P-018 and gates P-020.

### Decision

The first fatal result atomically transitions the runtime to `Lost`. Reject new work; fail every unsignaled token, lease, queued transfer, and callback exactly once; retain bounded portable/native diagnostics; never wait for lost GPU progress. Existing handles remain permanently invalid. The host explicitly creates a fresh runtime and reloads resources.

### Performance and tradeoffs

The terminal model has minimal retained state but visible interruption. State-check, notification, and teardown costs are unknown. DRED reports 2–5% automatic-breadcrumb loss on typical AAA D3D12 engines, so enhanced diagnostics are optional and measured.

### Rejected alternatives

- Managed recreation: requires retained bytes/reload callbacks, replay semantics, and epoch replacement not requested. Reconsider only with an explicit continuity requirement and durable resource-description API.

### Evidence

- [Vulkan lost device](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-lost-device) treats the logical device as lost.
- [D3D12 removal reason and DRED](https://microsoft.github.io/DirectX-Specs/d3d/DeviceRemovedExtendedData.html) provides terminal diagnostics and contextual overhead.
- [WebGPU device loss](https://www.w3.org/TR/webgpu/#device-lost) provides a mature permanent-old-device model.

### Assumptions, risks, and validation

- Assumption—applications can reconstruct state after notification.
- Risks include double completion, callback races, shutdown deadlock, missing diagnostics, and mistaken GPU-memory completion.
- Inject loss at acquire/submit/present/transfer/idle; assert bounded completion and no GPU waits; fuzz concurrency/destruction; verify stale Rust/FFI handles; measure detection, fan-out, diagnostics, retained bytes, and teardown.

## P-024: Persistent pipeline cache lifecycle — Host-owned validated blobs

### Problem and required outcome

Vulkan, DX12, and Metal cache products need bounded, validated persistence with corruption fallback, read-only/sandbox/custom-store support, and no rendering-semantic dependence on filesystem policy. Depends on P-001/P-006/P-007/P-022/P-023.

### Decision

Runtime imports and exports bounded opaque cache envelopes; hosts persist them. The envelope contains schema, backend, adapter/driver, artifact-interface identity, length, checksum, and native payload. Validate before native import; stale, corrupt, incompatible, or absent data falls back uncached. Explicit export returns a fresh blob. Hosts own paths, atomic commits, locks, quotas, eviction, and cross-process behavior; cache I/O never runs on the render hot path.

### Performance and tradeoffs

This keeps core storage-free and portable but adds host/FFI persistence work and possible blob copies. Native mechanisms are incomparable, and hit rate, startup benefit, copy cost, and cache size are unknown.

### Rejected alternatives

- Runtime filesystem cache: adds three-platform permissions, durability, locking, and eviction without evidence. Reconsider only if host duplication and measured cache value justify an optional helper outside core.

### Evidence

- [Vulkan pipeline caches](https://docs.vulkan.org/guide/latest/pipeline_cache.html), [D3D12 pipeline libraries](https://learn.microsoft.com/en-us/windows/win32/direct3d12/pipeline-state-object-cache), and [Metal binary archives](https://developer.apple.com/documentation/metal/mtlbinaryarchive) expose incompatible persistent native products.
- [Vulkan cache headers](https://docs.vulkan.org/spec/latest/chapters/pipelines.html#pipelines-cache-header) define compatibility identity fields.

### Assumptions, risks, and validation

- Assumption—hosts can store opaque bytes; misses never affect correctness.
- Risks include mismatched payloads, unbounded buffers, host races, fatal cache failures, and treating checksums as authenticity.
- Fuzz envelopes/import failures; test disabled, corrupt, stale, read-only, and concurrent host behavior; verify uncached equivalence; measure cold/warm creation, import/export, copies, bytes, hit rate, and host storage cost.

## P-025: Metal shader artifact production — Offline metallib variants

### Problem and required outcome

Choose a Metal artifact that preserves universal Slang source and the compiler/runtime split across the Apple deployment matrix. Runtime must select compatible products deterministically without Slang. Depends on P-001/P-005/P-006/P-007/P-020/P-029.

### Decision

Offline tooling emits MSL with Slang and, on Apple hosts, compiles it to `.metallib`. `.ezgfxshader` stores the Metal product with the same stage entry identity and provenance as SPIR-V/DXIL. Runtime selects and loads a compatible product or fails before pipeline creation. Non-Apple compiler builds emit MSL for cross-target validation, not a runtime-loadable hosted Metal claim.

### Performance and tradeoffs

This gives the strongest precompiled/runtime boundary and avoids runtime source compilation, but requires supported Apple tools and a variant matrix. The documented archive-size example concerns GPU binary archives, not comparable metallib performance. Project build time, bytes, compatibility, load, pipeline, and first-frame costs are unknown.

### Rejected alternatives

- Runtime MSL: hard failure under the inherited precompiled-shader requirement; cold compiler cost/failure remains.
- Metallib plus MSL fallback: same hard failure and can hide broken packaging. Reconsider either only if offline coverage is impossible and runtime Metal compilation is explicitly permitted while Slang remains excluded.

### Evidence

- [Apple precompiled libraries](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files.md) defines MSL-to-IR-to-metallib and runtime loading.
- [Slang Metal behavior](https://shader-slang.org/slang/user-guide/metal-target-specific) documents Metal source legalization.
- [Apple archive manipulation](https://developer.apple.com/documentation/metal/manipulating-metal-binary-archives) distinguishes shader libraries from GPU-specific binary archives.

### Assumptions, risks, and validation

- Assumption—release infrastructure can run supported Metal tools for every target.
- Risks include SDK/OS incompatibility, missing variants, confused archive/library roles, stripped diagnostics, and build-host constraints.
- Compile/load every entry across the declared Apple matrix with Slang absent at runtime; reject metadata mismatches; compare reflection/goldens; measure offline time/RSS, bytes, load/pipeline, and first frame; audit runtime linkage.

## P-026: Runtime threading and event delivery — Host-polled bounded event queue

### Problem and required outcome

Define context/resource affinity and async completion, error, cancellation, and shutdown delivery across Rust/C/C# without Tokio or runtime ownership of host event loops. Payload lifetime, ordering, overflow, reentrancy, and finalization must be deterministic. Depends on P-002/P-012/P-015/P-017/P-018.

### Decision

Each context owns a bounded event queue. Workers enqueue owned events; Rust hosts poll owned values that release through `Drop`, while C/C# releases returned payloads explicitly. Frame recording remains context-affine through `&mut Frame`. Overflow is observable and non-blocking. Dropping the last context/resource owner closes production, completes or cancels pending events, and waits on a CPU shutdown barrier so no payload or callback outlives its owner.

### Performance and tradeoffs

Memory is bounded; latency follows host pump cadence. Throughput, contention, queue bytes, wakeups, overflow frequency, and render-thread effect are unknown. Polling avoids a runtime callback thread and gives FFI deterministic ownership, but a host that stops polling delays delivery.

### Rejected alternatives

- Host-supplied dispatcher: adds registration/finalizer lifetime complexity. Reconsider if GUI integrations require direct UI-thread dispatch.
- Runtime callback thread: adds a thread, wakeups, reentrancy, and UI-affinity hazards. Reconsider only if polling cannot meet a measured latency requirement.

### Evidence

- [Rust bounded `sync_channel`](https://doc.rust-lang.org/std/sync/mpsc/fn.sync_channel.html) provides ordered bounded transport semantics.
- [Rust FFI guidance](https://doc.rust-lang.org/nomicon/ffi.html) documents ownership, callback, and unwind boundaries.
- [Rust `Send` and `Sync`](https://doc.rust-lang.org/nomicon/send-and-sync.html) defines explicit thread-safety contracts.

### Assumptions, risks, and validation

- Assumption—hosts pump at a documented cadence and release owned payloads.
- Risks include starvation, overflow, callback-after-free, reentrancy, ordering ambiguity, and shutdown races.
- Test ordering, overflow, cancellation, stopped polling, reentrancy, finalization, and destruction through Rust/C#; measure latency, bytes, wakeups, contention, and render-thread cost.

## P-027: Artifact integrity and provenance — Host-owned authenticity

### Problem and required outcome

`.ezgfxshader` inputs need distinct structural-integrity, compatibility, provenance, and authenticity contracts. Runtime validates framed lengths, BLAKE3 digest, bytechecked archive structure, and semantic coverage before native calls; the unkeyed digest does not prove authenticity. No remote trust service is requested.

### Decision

Runtime owns bounded parsing, schema/target/interface compatibility, full execution-content digests, and versioned compiler/tool/source provenance reporting. The host/package boundary owns artifact authenticity and signing policy. Runtime reports provenance and verification state explicitly but performs no signature/key-store management by default. Every execution-relevant section and canonical metadata participates in the digest.

### Performance and tradeoffs

This keeps runtime offline and avoids key lifecycle dependencies, but a compromised host can replace a structurally valid artifact. Parse/hash latency and metadata-size overhead are unknown and scale with bundle size; measure representative artifacts.

### Rejected alternatives

- Runtime signature verification: adds keys, rotation, revocation, secure storage, and crypto dependencies. Reconsider when runtime must distrust its host.
- Signed deployment manifest: couples runtime acceptance to installer/package lifecycle. Reconsider when coordinated shader/native/cache-set authenticity is required.

### Evidence

- [Vulkan shader-module validation](https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateInfo.html) establishes strict native-input validation context.
- [SLSA provenance](https://slsa.dev/spec/v1.0/provenance) distinguishes build provenance fields from artifact identity.
- [in-toto attestations](https://github.com/in-toto/attestation) provide a package-level signed metadata alternative.

### Assumptions, risks, and validation

- Assumption—the host/package deployment boundary authenticates artifacts when needed.
- Risks include host compromise, incomplete digest coverage, ambiguous verification labels, stale provenance, and compatibility bugs.
- Fuzz bounds/overlap/truncation; prove digest coverage; round-trip compiler/runtime versions; test absent/unknown/incompatible provenance; measure parse/hash time and bytes.

## P-028: Local diagnostics and profiling — Bounded structured event stream

### Problem and required outcome

Errors, warnings, graph/backend diagnostics, async failures, and profiling observations need one local causal schema across compiler artifacts, HAL, graph, workers, transfers, textures, presentation, Rust, and FFI. Retention must be bounded with no remote upload or hidden persistence. Depends on P-002/P-003/P-008/P-015/P-018/P-019/P-026 and feeds P-020.

### Decision

Components emit typed diagnostic events into a bounded per-context stream delivered through P-026's host-polled queue. Each event carries severity/category, operation/resource/frame/node/job/submission correlation IDs, component/backend details, sequence/domain, timestamp with clock/domain/unit/availability, and typed payload. Overflow emits a synthetic loss event and dropped count; it never blocks rendering indefinitely. Pull snapshots are derived summaries, not the authoritative history.

### Performance and tradeoffs

This best preserves transient causality within a bound, but adds event copies, synchronization, timestamp cost, memory, and overflow policy. Disabled/enabled overhead, contention, event rate, memory, loss rate, and readout latency are unknown.

### Rejected alternatives

- Pull snapshots alone: lose transient events and ordering. Reconsider for minimal latest-state-only deployments.
- Capture sessions alone: miss failures outside active captures. Retain as optional detail mode if continuous-stream overhead proves material.

### Evidence

- [Rust bounded `sync_channel`](https://doc.rust-lang.org/std/sync/mpsc/fn.sync_channel.html) defines bounded ordered transport.
- [Vulkan debug utils](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_debug_utils.html) supplies native diagnostic context.
- [Vulkan query semantics](https://docs.vulkan.org/spec/latest/chapters/queries.html) and [D3D12 timing](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing) define GPU timing domains.

### Assumptions, risks, and validation

- Assumption—a bounded per-context stream provides sufficient local retention.
- Risks include lost causality across domains, overflow, misleading clocks/units, contention, and profiling perturbation.
- Exercise every severity/component path; force overflow; verify correlation/order/loss reporting and continued rendering; compare diagnostics-disabled/enabled cost, timestamp/readout latency, queue memory, and contention.

## P-029: Cross-platform native build and distribution — Centrally built separate signed artifacts

### Problem and required outcome

Rust/backend bindings, Slang/DXC/Apple tools, optional Basis, and C ABI outputs need deterministic per-platform build/linkage/distribution with a compiler-free runtime proof. Targets require explicit hosts, dependencies, features, notices, signing/notarization, and package layout. Depends on P-001/P-002/P-003/P-005/P-014/P-025/P-027 and feeds P-019/P-020.

### Decision

Pinned source/lockfile builds run on target-appropriate release CI, which publishes separate versioned runtime/FFI, compiler/tool, and optional Basis-enabled archives/installers. Each artifact declares target triple, architecture, minimum OS, Cargo package/features, static/dynamic/platform dependencies, symbols policy, licenses/notices, and provenance. Runtime packages undergo dependency-tree and binary-import audits proving absence of Slang/compiler and feature-off C++ decoder linkage. CI owns signing/notarization and canonical package manifests; consumers install prebuilt products.

### Performance and tradeoffs

Prebuilt delivery removes consumer toolchain burden and makes ABI assets deterministic, but expands CI, signing, retention, and target-matrix operations. Build/install/startup time, package size, dependency count, cache rate, signing latency, and matrix cost remain unknown per target.

### Rejected alternatives

- Source-built consumer delivery: requires every consumer to own native toolchains. Retain as the release-build method and an expert option; reconsider as primary only if prebuilt distribution is unsustainable.
- Platform package managers as primary: host dependency/version variability weakens canonical delivery. Reconsider as secondary ecosystem packaging after archives stabilize.

### Evidence

- [Cargo build scripts](https://doc.rust-lang.org/cargo/reference/build-scripts.html), [profiles](https://doc.rust-lang.org/cargo/reference/profiles.html), and [Rust platform support](https://doc.rust-lang.org/rustc/platform-support.html) define native integration and target matrices.
- [Microsoft Artifact Signing](https://learn.microsoft.com/en-us/azure/artifact-signing/overview) and [Apple notarization](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution) establish platform distribution controls.

### Assumptions, risks, and validation

- Assumption—release CI can access every native SDK, linker, credential, and Apple tool required by the declared matrix.
- Risks include architecture mismatch, SDK drift, missing redistributables/notices, key rotation, stale packages, and compiler feature leakage.
- Build every package/feature target; audit trees/imports; install in clean environments; run ABI/backend/artifact/snapshot gates; verify signatures/notarization/manifests; measure build, install, startup, size, dependencies, and matrix cost.