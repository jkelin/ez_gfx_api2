# P-022: Backend, device, and capability admission policy

## Problem

Select how applications enumerate and choose a graphics backend and adapter, and define the exact capability floor that admits a device. Capability discovery exists in the HAL plan, but no selected policy determines required versus optional features, fallback behavior, multi-adapter identity, or deterministic backend choice.

## Prompt context

The Rust migration must support Vulkan, DX12, and Metal, remain backend-neutral, and report unsupported devices explicitly. Backend-specific snapshots and compressed-format selection depend on stable device/profile identity.

## Constraints and acceptance criteria

- The public Rust and FFI APIs must expose deterministic enumeration, selection, and diagnostics.
- The required floor must cover every feature assumed by graph, bindless descriptors, indirect draws, synchronization, aliasing, compressed textures, and presentation.
- Missing required capabilities must return an explicit unsupported result; no backend may silently emulate semantics that change the API contract.
- Optional capabilities and format support must be queryable without fragmenting the common API.
- Adapter, driver, backend, and capability-profile identity must be stable inputs to caches and snapshot evidence.

## Dependencies

- P-002 exposes selection and diagnostics.
- P-003 supplies backend enumeration and raw capability discovery.
- P-007, P-008, P-009, P-014, P-016, P-017, and P-019 consume the admitted profile.
- P-020 uses admission policy in cross-backend release gates.

## Unresolved questions

- Is backend choice compile-time, runtime, host-specified, preference-ordered, or some combination?
- What exact limits and features define the minimum profile on each backend?
- Are lower tiers rejected, or may explicitly selected optional paths preserve the same semantics?
- How are software adapters, integrated/discrete preferences, headless adapters, and multiple matching devices ranked?
- Which stable identifiers participate in persistent cache and golden-profile keys?

## Candidate solutions

### S-P-022-single-modern-floor: One mandatory semantic profile with explicit adapter selection

#### Approach and integration

Expose enumeration records through Rust/FFI, require callers to select a stable opaque adapter ID or request a documented default, and admit only devices satisfying one cross-backend profile. The profile maps semantic requirements—not API names—to per-backend checks: indexed/bindless resources and limits, indirect indexed draws/count support, queue synchronization, resource heaps/aliasing, required barriers, dynamic rendering equivalents, presentation, timestamp/query support, and at least one required compressed-format path. Format/usage support remains per-format probing rather than part of a coarse GPU name check.

The default rank is deterministic within one enumeration: requested backend first, hardware over software unless explicitly allowed, then host power preference, dedicated/discrete preference only as a tie-breaker, and stable ID. Diagnostics report every failed requirement. Candidate stable inputs include Vulkan device/driver properties plus `pipelineCacheUUID`, DXGI adapter LUID plus driver identity, and Metal `registryID`/GPU-family data; display names never key caches.

#### Constraint applicability

One profile guarantees the same graph, descriptor, alias, and draw semantics everywhere and keeps hot paths free of profile branches. It may set a high floor: Apple's 2026 table places argument-buffer tier 2 at Apple6, while older families have smaller indexed-resource limits. A required native feature may exclude otherwise capable devices; any software fallback must be an explicitly selected test profile, never a silent production substitute.

#### Performance evidence

Feature/property queries occur during enumeration/device creation; no primary source supplies a portable latency figure, and project startup cost is unknown. Vulkan notes that enabling some features such as robust buffer access may have runtime cost, so the implementation must enable required features rather than every advertised feature. Measure enumeration/probe/device-creation time, admitted adapter coverage, descriptor limits, binary size, and identical recording/draw workloads on named devices.

#### Tradeoffs and failure modes

This is easiest to reason about and test, and cache/golden identity stays singular. It excludes lower-tier hardware and may force the common floor upward because of bindless/alias requirements. Ranking by device type alone can choose an unusable or power-hungry adapter; unstable IDs can poison caches. A capability must be rejected before device-dependent managers initialize.

#### Sources

- [Vulkan physical-device enumeration and properties](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html) defines enumeration, device type, driver/device IDs, limits, and `pipelineCacheUUID`.
- [Vulkan feature queries](https://docs.vulkan.org/spec/latest/chapters/features.html) requires unsupported features to remain disabled and documents possible runtime cost for some enabled features.
- [D3D12 `CheckFeatureSupport`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-checkfeaturesupport) is the driver feature/tier query boundary.
- [DXGI adapter enumeration by GPU preference](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_6/nf-dxgi1_6-idxgifactory6-enumadapterbygpupreference) provides explicit preference ordering.
- [Apple Metal feature-set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) maps features and limits to GPU families and calls out properties that still require runtime checks.

### S-P-022-baseline-plus-tiers: Stable baseline with explicit optional capability profiles

#### Approach and integration

Define a small baseline that preserves every public semantic, then attach named, monotonic capability profiles for performance or capacity: for example, higher bindless limits, enhanced/native barrier paths, dedicated transfer concurrency, broader compression, or larger MDI counts. Enumeration returns the baseline result, supported profiles, raw limits, backend/driver identity, and reasons. The host may require a profile or accept the best supported one. Pipeline/cache/golden keys include the selected profile; graph and HAL choose implementation paths at device creation, not per draw.

Vulkan Profiles provide a mature precedent for grouping features, extensions, properties, formats, and queue families into named profiles. D3D12 exposes feature levels plus independent option tiers, and Metal exposes programming-model/GPU-family capabilities plus runtime properties. The project would own a semantic profile mapping rather than pretending these native tiers align automatically.

#### Constraint applicability

This widens hardware coverage only when baseline paths preserve identical public behavior. A tier may change capacity, scheduling, or speed, but cannot change pixels, hazard guarantees, handle semantics, or artifact meaning. If no viable baseline implementation exists for bindless descriptors or aliasing, that feature belongs in the floor and the candidate collapses toward the single-profile option.

#### Performance evidence

No source quantifies the total cost of this project's profile branching. Moving selection to device creation makes per-draw branch cost avoidable by concrete dispatch, but increases compiled code, pipeline/cache variants, and CI matrix size. Measure adapter coverage, startup probes, code/binary size, cache entries, backend-path counts, CPU recording, GPU time, and validation matrix duration per named profile.

#### Tradeoffs and failure modes

Profiles expose hardware differences honestly and can keep optional fast paths, but multiply ownership across HAL, graph, caches, tests, and documentation. Non-monotonic native features make simplistic “tier numbers” misleading. Silent downgrade, profile-dependent rendering, stale cache reuse, and combinatorial feature flags are disqualifiers; profiles must be few, named, and whole-device immutable.

#### Sources

- [Vulkan Profiles](https://github.khronos.org/Vulkan-Site/guide/latest/vulkan_profiles.html) describes named collections of features, extensions, properties, formats, and queue-family requirements.
- [Vulkan Roadmap profiles](https://docs.vulkan.org/spec/latest/appendices/roadmap.html) provides Khronos-defined capability baselines rather than GPU-name inference.
- [D3D12 hardware feature levels](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-feature-levels) distinguishes base feature levels from separately queried support.
- [D3D12 capability querying](https://learn.microsoft.com/en-us/windows/win32/direct3d12/capability-querying) documents option/tier queries.
- [Apple Metal feature-set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) shows that features and numerical limits vary independently across GPU families.

## Performance comparison

| Rank | Candidate | Hard-constraint result | Normalized evidence | Reliability/operational cost |
| ---- | --------- | ---------------------- | ------------------- | ---------------------------- |
| 1 | Single modern semantic floor | Passes if the floor includes every graph/bindless/alias/MDI/presentation requirement | Native query mechanisms and limits are sourced; startup cost and admitted-device coverage are unknown | One implementation/cache/test profile; may exclude older hardware |
| 2 | Baseline plus optional tiers | Passes only when every baseline path preserves identical public semantics | Profile mechanisms are mature; code-size, matrix, and hot-path benefits are unknown | More hardware reach, but multiple HAL/cache/test paths |

Neither candidate has comparable project measurements. A strict floor ranks first because the prompt requires complete Vulkan/DX12/Metal behavior and TODO implementation, not broad legacy-device reach. Optional raw limits and formats remain queryable without creating semantic tiers.

## Selected solution

**Select S-P-022-single-modern-floor: one mandatory semantic profile with explicit adapter selection.**

Rust/FFI enumerate opaque stable adapter records, accept explicit backend/adapter selection, and provide a deterministic documented default. Admission maps semantic requirements to native checks: bindless/indexed resources and minimum capacities, indirect indexed draws, queue synchronization, resource heaps/aliasing, barrier/rendering equivalents, presentation, timestamps, and required compressed-format capability. Vulkan admission explicitly requires and enables `drawIndirectCount`, `multiDrawIndirect`, and `vertexPipelineStoresAndAtomics`. Unsupported requirements fail before manager creation with per-requirement diagnostics. Cache/golden identity uses backend, stable adapter/device identity, driver, and profile schema—not display names. Software adapters require explicit opt-in.
Optional `ShaderCapabilities{task, mesh}` rides `AdapterCapabilities` under profile schema version 2; task without mesh is `InvalidCapabilities`, V1 admission ignores optional stages, and enumeration plus `ez_gfx_context_shader_capabilities` report the same bits.

Reject profile tiers because no measured coverage need justifies multiplying backend, cache, shader, and snapshot paths, and any tier that changes semantics is a hard failure. Reconsider only if capability spikes show meaningful required-device exclusion and a lower path preserves pixels, hazards, handles, and artifact meaning with bounded implementation/test cost.

Evidence: Vulkan, D3D12, DXGI, and Metal expose explicit enumeration, stable identity inputs, feature/tier queries, and numerical limits. Apple's table demonstrates meaningful family differences; Vulkan warns some enabled features can cost runtime performance. Enumeration/probe latency, device coverage, and workload performance remain unknown.

Assumptions: the project may require modern hardware and can publish the exact minimum capacities after backend spikes. Risks are an excessive floor, unstable identity, nondeterministic default ranking, and accidentally enabling costly unused features. Validate the semantic-to-native matrix on named adapters; test every rejection diagnostic and deterministic selection; record admitted/excluded devices, probe/device-creation time, enabled features, limits, binary size, cache keys, CPU recording, and GPU time; require Vulkan/DX12/Metal conformance before freezing the floor.
