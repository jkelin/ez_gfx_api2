# TODO

## P0

- Lower compiled transient alias assignments into Vulkan, DX12, and Metal resource placement, including alias barriers and overlap-safe retirement (P-004, P-008, P-009).
- Add managed render-target creation, format-capability probing, per-target clears, sampled/storage bindings, resize/history, graph attachment, and lifecycle APIs across Rust/FFI/backends (P-008, P-009, P-010, P-015).

## P1

- Finish native texture ingestion and streaming: optional `.basis`, KTX2 UASTC/ETC1S transcoding to supported BC/ASTC formats, direct compressed uploads, progressive mip admission/eviction, validated region updates, deferred descriptor publication, and upload telemetry (P-014, P-015).
- Add safe Rust RAII resource owners while retaining typed raw handles for FFI, then publish an API parity matrix and add missing target, texture-update, decoder-callback, screenshot-save, and graph-authoring interfaces with ABI tests (P-002, P-010, P-015, P-019).
- Add the selected generation-checked mapped staging lease interface for zero-copy procedural vertex/index writes; current public uploads still copy caller slices (P-011).
- Expose validated per-draw/per-pipeline viewport and scissor state and batch consecutive equal-state MDI ranges; current backends set only full-render-area state (P-016).
- Expose deterministic adapter enumeration and explicit stable adapter selection through Rust and FFI, including admitted limits/formats and rejection diagnostics (P-003, P-022).
- Add `raw-window-handle` Rust adapters, Linux X11/Wayland Vulkan surfaces, DPI-aware recreation coverage, and native presentation tests (P-017, P-022).
- Add bounded, validated host-owned pipeline-cache import/export envelopes with backend/device/driver/schema compatibility; current caches are process-local only (P-007, P-024).
- Finish terminal device-loss behavior for queued CPU jobs, transfers, staging leases, pending handles, and waits; support explicit cross-thread context destruction without cleanup under Windows loader lock (P-023, P-026).
- Expand runtime events to preserve one correlation across admission/decode/transfer/bind, and add severity, category, sequence/domain, clocks, units, payloads, overflow markers, and cleanup/resource/device-loss outcomes (P-023, P-026, P-028).

## P2

- Add deterministic offscreen pixel goldens for fork/join graphs, scaled targets, storage images, history, aliasing, and all supported backends; run Vulkan, DX12, and Metal native suites on guaranteed-capability GPU runners (P-019, P-020, P-022).
- Benchmark graph compilation, pass coalescing, alias savings, staging reuse/batch sizes, event latency, async Sponza loading, GPU frame time, and package size before claiming selected-plan performance gains (P-007, P-008, P-009, P-012, P-018, P-028).
- Resolve the P-004 allocator-selection mismatch: `gpu-allocator` covers Vulkan/DX12, while Metal uses a backend-native allocator; either supply equivalent selected evidence and amend the decision or adopt a maintained cross-backend implementation (P-004).

## Tooling

- Add target-native release jobs that publish separate runtime/FFI, compiler/tool, and optional Basis artifacts with deterministic manifests, license/provenance records, binary-import audits, signing, and Apple notarization (P-001, P-025, P-027, P-029).
- Configure Miri and integrate it with nextest (P-019, P-029).
