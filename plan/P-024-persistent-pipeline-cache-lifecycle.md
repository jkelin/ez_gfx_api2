# P-024: Persistent pipeline cache lifecycle

## Problem

Define how backend pipeline caches are located, bounded, synchronized, invalidated, and recovered. P-007 selects keys, but not persistence ownership, atomicity, concurrency, eviction, or read-only behavior.

## Prompt context

The runtime creates Vulkan, DX12, and Metal pipelines from precompiled shader artifacts without Slang. Cache identity includes backend, device/driver, shader/interface, state, and schema.

## Constraints and acceptance criteria

- Bad cache bytes must never prevent uncached pipeline creation or reach native APIs without validation.
- Crash and concurrent-process behavior must be defined.
- Growth must be bounded and host-configurable.
- Disabled, read-only, sandboxed, and custom-storage deployments must work.
- Storage policy must not leak into rendering semantics.

## Dependencies

- P-001, P-006, P-007, P-022, and P-023.

## Unresolved questions

- Does the host store opaque blobs, does runtime own files, or both?
- What commit/locking/eviction policy is portable?
- When are dirty caches serialized, and may backend cache products merge?

## Candidate solutions

### S-P-024-host-owned-blobs: Import/export validated opaque caches

#### Approach and integration

Runtime accepts optional cache bytes at device creation and exports fresh bytes explicitly or at orderly shutdown. The host owns paths, atomic writes, locks, quotas, and eviction. A small envelope carries backend, schema, artifact-interface hash, adapter/driver identity, length, and checksum before the native payload. Incompatibility/corruption discards the blob and creates pipelines uncached.

Map the envelope to `VkPipelineCache` data and merge APIs, D3D12 pipeline-library serialization or cached PSO blobs, and Metal binary-archive URLs/data through backend adapters. The Rust API uses byte slices/results; FFI uses bounded pointer/count pairs.

#### Constraint applicability

Naturally supports sandboxes, custom stores, and read-only hosts, and avoids filesystem policy in core. The host must serialize calls and bound bytes; cache export is explicit so loss/shutdown cannot silently block.

#### Performance evidence

Vulkan states persistent caches can eliminate costly portions of pipeline creation; Apple positions binary archives to reduce runtime pipeline compilation/stutter. Neither supplies portable gains for this workload. Host marshalling adds at least a blob copy unless ownership/mapping APIs avoid it. Measure cold/warm pipeline creation, import/export latency, bytes copied, cache size/hit rate, and host lock/write cost per named backend/adapter.

#### Tradeoffs and failure modes

Deep and portable, but burdens every host and C# binding with storage. Cross-process policy is outside the library. Importing native bytes with a mismatched driver/device, trusting checksum as authenticity, or making cache failure fatal are disqualifiers.

#### Sources

- [Vulkan pipeline cache guide](https://docs.vulkan.org/guide/latest/pipeline_cache.html) documents persistence between runs and costly pipeline creation.
- [Vulkan pipeline-cache data header](https://docs.vulkan.org/spec/latest/chapters/pipelines.html#pipelines-cache-header) defines vendor/device/UUID compatibility fields.
- [D3D12 pipeline libraries](https://learn.microsoft.com/en-us/windows/win32/direct3d12/pipeline-state-object-cache) documents library load/store/serialization.
- [Metal binary archives](https://developer.apple.com/documentation/metal/mtlbinaryarchive) provides Metal's serializable pipeline cache object.

### S-P-024-runtime-filesystem-cache: Bounded per-profile atomic store

#### Approach and integration

Runtime owns an optional cache root supplied by the host. Namespace by application/schema/backend/adapter/driver and store backend blobs behind the same validated envelope. Write a temporary file, flush as configured, atomically replace the committed file, and use a process lock or immutable generation files plus manifest to prevent clobbering. Enforce byte/entry/age bounds at startup or maintenance; lock failure, read-only storage, corruption, or quota failure falls back uncached.

Backend cache collection happens at defined checkpoints, never on the render hot path. P-023 loss policy decides whether dirty data from a lost device is discarded.

#### Constraint applicability

Convenient for common desktop deployments and centralizes correctness. It must remain optional because platform rename/locking/durability semantics differ and some hosts deny filesystem access.

#### Performance evidence

No evidence quantifies this project's cache hit benefit or filesystem cost. Atomic replacement can add open/write/flush/rename work proportional to blob size; eviction scans scale with entries unless indexed. Measure cold/warm startup, pipeline creation, lock wait, serialization/write amplification, cache bytes, hit rate, and cleanup time on named OS/filesystem/storage.

#### Tradeoffs and failure modes

Hosts get turnkey caching, but runtime inherits path security, permissions, crash consistency, stale temp files, multi-process coordination, and eviction policy. Holding a global lock during pipeline creation, unbounded growth, or assuming rename implies durable storage are disqualifiers.

#### Sources

- [Vulkan pipeline-cache sample](https://docs.vulkan.org/samples/latest/samples/performance/pipeline_cache/README.html) provides a mature persistent-cache implementation pattern.
- [Windows `ReplaceFile`](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew) defines atomic-style replacement semantics on Windows.
- [POSIX `rename`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/rename.html) defines replacement/atomic visibility within its constraints.
- [Apple binary archive manipulation](https://developer.apple.com/documentation/metal/manipulating-metal-binary-archives) shows archives may contain many GPU slices and need targeted size management.

## Performance comparison

| Rank | Candidate | Hard-constraint result | Normalized evidence | Operational/implementation cost |
| ---- | --------- | ---------------------- | ------------------- | ------------------------------- |
| 1 | Host-owned validated blobs | Passes disabled, read-only, sandbox, custom-store, bounded, and uncached-fallback requirements | Native persistence mechanisms are sourced; copy, hit-rate, and startup gains are unknown | Small runtime boundary; storage/locking burden belongs to host |
| 2 | Runtime filesystem cache | Passes only if storage stays optional and every OS has correct atomicity/locking/eviction | OS replacement primitives are sourced; durability, contention, and workload benefit are unknown | Highest path, permission, crash, multi-process, and eviction complexity |

No cross-backend benchmark makes cache benefit comparable, and native products differ. Host-owned blobs rank first because storage policy is not rendering semantics and the required FFI already provides explicit boundary validation.

## Selected solution

**Select S-P-024-host-owned-blobs: runtime imports/exports validated opaque cache envelopes; hosts persist them.**

At device creation the host may supply bounded bytes containing schema, backend, adapter/driver, artifact-interface identity, length, checksum, and native payload. Runtime validates the envelope and backend compatibility before native import; any failure falls back uncached. Explicit export returns a bounded new blob. The host owns path choice, locking, atomic commit, quota, eviction, read-only policy, and cross-process coordination. Cache I/O never occurs on the render hot path.

Reject runtime-owned filesystem persistence because it adds three-platform storage policy, permissions, durability, locking, and eviction to the graphics runtime without prompt evidence. Reconsider only if host integrations repeatedly duplicate the same policy and measured cache value justifies a narrow optional helper outside core.

Evidence: Vulkan pipeline caches, D3D12 pipeline libraries/cached PSOs, and Metal binary archives all support persistence but use incompatible payloads. Vulkan identifies device compatibility; Apple demonstrates GPU-slice size variation. No source supplies portable hit-rate or startup improvements for this workload, and blob copy/serialization costs remain unknown.

Assumptions: hosts can persist opaque bytes when desired; cache misses are correctness-neutral. Risks are mismatched native bytes, unbounded FFI buffers, host races, cache failure becoming fatal, and treating checksums as authenticity. Fuzz envelopes and native import failure; test disabled/empty/corrupt/stale/read-only/concurrent host scenarios; verify uncached equivalence; measure cold/warm creation, import/export latency, copies, bytes, hit rate, and host storage costs per backend/adapter.
