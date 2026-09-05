# TODO

## P0

- Lower compiled transient alias assignments into Vulkan, DX12, and Metal resource placement, including alias barriers and overlap-safe retirement (P-004, P-008, P-009).
- Add managed render-target creation, format-capability probing, per-target clears, sampled/storage bindings, resize/history, graph attachment, and lifecycle APIs across Rust/FFI/backends (P-008, P-009, P-010, P-015).
- Investigate/fix source-inferred readiness binding before descriptor publication (crates/ez-gfx/src/state/texture.rs:437-463,529-543; not runtime reproduced) (P-015).

## P1

- Add safe Rust RAII resource owners while retaining typed raw handles for FFI, publish an API parity matrix, and add missing render-target, screenshot-save, and graph-authoring interfaces with ABI tests (P-002, P-010, P-019).
- Add the selected generation-checked mapped staging lease interface for zero-copy procedural vertex/index writes; current public uploads still copy caller slices (P-011).
- Expose validated per-draw/per-pipeline viewport and scissor state and batch consecutive equal-state MDI ranges; current backends set only full-render-area state (P-016).
- Expose deterministic adapter enumeration and explicit stable adapter selection through Rust and FFI, including admitted limits/formats and rejection diagnostics (P-003, P-022).
- Add `raw-window-handle` Rust adapters, Linux X11/Wayland Vulkan surfaces, DPI-aware recreation coverage, and native presentation tests (P-017, P-022).
- Add bounded, validated host-owned pipeline-cache import/export envelopes with backend/device/driver/schema compatibility; current caches are process-local only (P-007, P-024).
- Finish terminal device-loss behavior for queued CPU jobs, transfers, staging leases, pending handles, and waits; support explicit cross-thread context destruction without cleanup under Windows loader lock (P-023, P-026).
- Expand runtime events to preserve one correlation across admission/decode/transfer/bind, and add severity, category, sequence/domain, clocks, units, payloads, overflow markers, and cleanup/resource/device-loss outcomes (P-023, P-026, P-028).
- Prove sustained coarse-to-fine rendering, deterministic unsignaled-fence unload/reuse, and native Metal compressed pixels; benchmark dynamic-atlas transfer/frame time before declaring P-015 evidence complete. Retained hidden BC1/BC3/BC7/RGBA8 pixel/order/lifetime regressions pass on RTX 3080 Vulkan/DX12 with clean Vulkan validation (P-014, P-015, P-023).
- Unify Basis/KTX2 concrete target selection: explicit BC1/BC3 fail the strict transcoded-format match, ETC1S R/Rg yields BC4/BC5 rejected even under Auto, and KTX2 Auto preserves container sRGB while standalone Basis Auto stays linear (P-014).
- Make KTX2/native Basis decoder linkage optional (P-014).
- Add DDS and direct raw compressed-mip ingestion (P-014).
- Fix Metal independent BC/ASTC capability probing; current exclusive BC-else-ASTC never reports both (P-003, P-014).
- Reconcile P-012 cross-texture native batching with real per-mip completion (P-012, P-015).
- Reconcile original loaded-callback, replaceable built-in decoders, and GPU-vs-CPU mip differences as explicit deltas, not mandatory bugs.
- Record Zstd/Zlib/container/format limits as evaluated support gaps, not mandatory full-extension support (P-014).
- Retained evidence gap: no ASTC, sRGB, or partial-edge-block pixel coverage; existing BC1/BC3/BC7/RGBA8 RTX 3080 regressions stand (P-014, P-015).

## P2

- Add deterministic offscreen pixel goldens for fork/join graphs, scaled targets, storage images, history, aliasing, and all supported backends; run Vulkan, DX12, and Metal native suites on guaranteed-capability GPU runners (P-019, P-020, P-022).
- Benchmark graph compilation, pass coalescing, alias savings, staging reuse/batch sizes, event latency, async Sponza loading, GPU frame time, and package size before claiming selected-plan performance gains (P-007, P-008, P-009, P-012, P-018, P-028).
- Resolve the P-004 allocator-selection mismatch: `gpu-allocator` covers Vulkan/DX12, while Metal uses a backend-native allocator; either supply equivalent selected evidence and amend the decision or adopt a maintained cross-backend implementation (P-004).

## Tooling

- Add target-native release jobs that publish separate runtime/FFI, compiler/tool, and optional Basis artifacts with deterministic manifests, license/provenance records, binary-import audits, signing, and Apple notarization (P-001, P-025, P-027, P-029).
- Configure Miri and integrate it with nextest (P-019, P-029).
