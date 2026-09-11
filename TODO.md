# TODO

## P0

- Lower compiled transient alias assignments into Vulkan, DX12, and Metal resource placement, including alias barriers and overlap-safe retirement (P-004, P-008, P-009).
- Complete managed render-target history and cross-backend storage-image evidence. Cached resize, sampled binding, graph attachment, readback, and ownership-based lifetime are implemented; C ABI 40 retains explicit opaque-handle release (P-008, P-009, P-010, P-015).

## P1

- Publish an API parity matrix, add screenshot-save and expanded graph-authoring interfaces, and cover them with ABI tests. Cached render-target configuration and callback-scoped readback are implemented; C value-buffer acquisition, frame bind draft, and frame execute parity are implemented at ABI 40 (P-002, P-010, P-019).
- Add the selected caller-writable mapped staging lease for procedural vertex/index writes (P-011). The existing slice path copies into mapped staging. The lease must retain its context/resource ownership, commit or cancel exactly once, cancel safely on `Drop`, and fail after loss. Acceptance is procedural-upload pixel parity, commit/cancel/failure coverage, and no cross-frame lease stalls.
- Expose validated per-draw/per-pipeline viewport and scissor state and batch consecutive equal-state MDI ranges; current backends set only full-render-area state (P-016).
- Complete Metal execution evidence for deterministic adapter enumeration, selection, admitted limits/formats, and rejection diagnostics through safe Rust and ABI 40 (P-003, P-022).
- Add Linux X11/Wayland Vulkan surfaces, DPI-aware recreation coverage, and native presentation tests. Safe surfaces remain owning wrappers with atomic construction rollback; ABI 40 validates borrowed native handles without a caller-supplied platform discriminator (P-017, P-022).
- Add bounded, validated host-owned pipeline-cache import/export envelopes with backend/device/driver/schema compatibility; current caches are process-local only (P-007, P-024).
- Finish terminal device-loss behavior for staging leases and waits. Pending texture decode/transfer uploads emit terminal loss events and transfer workers retain sticky loss; loader-lock teardown remains abandon-only (P-023, P-026).
- Add correlation IDs, sequence/domain, clocks, units, payloads, and cleanup outcomes to the lossless typed upload-event queue. Texture/vertex/index ownership, readiness, cancellation, and failure transitions are already lossless; bounded runtime diagnostics remain separate and report dropped counts (P-023, P-026, P-028).
- Design GPU-side transition/acquire handoff for Vulkan, DX12, and Metal so submission never waits on transfer progress (P-012, P-015). Metal command submission is asynchronous, but buffer-transfer waits still flush the worker and call `waitUntilCompleted` on the caller thread. Preserve barrier ordering and error/loss propagation. Acceptance is nonblocking submit behavior under texture/vertex streaming load with pixel parity on all three backends.
- Reserved shader redesign: rebaseline the native allocation probe's excluded shader metadata/pipeline-key residual (`examples/allocation_probe` single-frame phase ceilings and whole-window baselines in `allocation_optimization_report.md`) after the redesign lands; safe begin/configure/acquire/bind phases and independent shader-free frame-plan/wait validation already assert zero on Vulkan and DX12, while descriptor preparation remains inside execute and Metal execution is covered by its backend matrix run (3 packages, 69 tests at HEAD 5917787).
- Optional: raise the synchronized texture heap capacity above 1024 (`docs/textures.md` §Texture heap capacity, P-007). The cap is one contract across core admission, compiler/HAL reflection, runtime handles, Vulkan/DX12 descriptor counts, Metal argument-buffer layout, and the Slang static array length, so the change must land atomically in every layer plus device-capability admission. Acceptance is updated capacity/admission tests and pixel coverage on all three backends.

## P2

- Add deterministic offscreen pixel goldens for fork/join graphs, scaled targets, storage images, history, aliasing, and all supported backends; run Vulkan, DX12, and Metal native suites on guaranteed-capability GPU runners (P-019, P-020, P-022).
- Benchmark graph compilation, pass coalescing, alias savings, event latency, async Sponza loading, GPU frame time, and package size before claiming broader selected-plan performance gains. Scoped texture staging/batching and 2048² atlas measurements are recorded in `docs/textures.md`, not full-scene performance evidence (P-007, P-008, P-009, P-012, P-018, P-028).
- Resolve the P-004 allocator-selection mismatch: `gpu-allocator` covers Vulkan/DX12, while Metal uses a backend-native allocator; either supply equivalent selected evidence and amend the decision or adopt a maintained cross-backend implementation (P-004).

## Tooling

- Add target-native release jobs that publish separate runtime/FFI, compiler/tool, and optional Basis artifacts with deterministic manifests, license/provenance records, binary-import audits, signing, and Apple notarization (P-001, P-025, P-027, P-029).
- Configure Miri and integrate it with nextest (P-019, P-029).
- Linux remote multi-ICD enumeration is unstable in-process (not a driver fix yet): with 9 ICD manifests the full ez-gfx lib suite deterministically fails `explicit_selection_creates_context_for_enumerated_adapter` (report enumeration sees the NVIDIA adapter, the immediately following create enumeration lacks it, AdapterNotFound maps to InvalidArgument), while forced-NVIDIA and forced-Lavapipe full suites are each 32/32 green and deviceUUIDs are stable across processes. Interim: optional `REMOTE_TEST_LINUX_VK_DRIVER_FILES` pins `VK_ICD_FILENAMES` for remote-test-linux. Revisit with loader/ICD isolation diagnostics before claiming multi-ICD determinism.

- Fix remote-test rsync to exclude `.git` exactly instead of `.git/`: the current rule copies Windows worktree `.git` files onto macOS/Linux mirrors, breaking remote git, and macOS sync intermittently stalls before Cargo (current Metal verification used a clean clone workaround at HEAD 5917787). Validate stable macOS sync after the change.

## Evidence

- Allocator block minima stay at the shared default (16 MiB device / 256 MiB max, 8 MiB host / 64 MiB max).
- The 4 MiB / 4 MiB experiment saved ~16 MiB private commit on DX12 and ~14–16 MiB on Vulkan for the small triangle workload, but that is single-workload evidence only:
  - DX12 maintains separate heap categories that each establish their own floor.
  - Metal shares the policy unmeasured.
  - Smaller floors risk more native allocations, personal blocks, and fragmentation under texture/geometry/multi-pass/resize/streaming load.
- Adopt the smaller floor only after the full backend matrix plus those workloads show acceptable latency and fragmentation on all three backends.
- On-demand allocator telemetry (`Context::memory_telemetry`, HAL `BackendMemoryTelemetry`) exists to gather that evidence without per-frame cost.
