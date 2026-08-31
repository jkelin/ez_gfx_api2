# Handoff

## Completed baseline

- Cargo workspace with compiler-free runtime, safe Rust contracts, C ABI v17, opaque generational handles, validated semantic/artifact/capability data, and panic containment.
- Vulkan, DX12, and Metal backend crates use a backend-neutral HAL and `gpu-allocator` 0.28.
- Offline Slang compilation emits SPIR-V 1.5, Shader Model 6.5 DXIL, and Metal products in bounded `.ezgfx` artifacts.
- Vulkan and DX12 resource upload, compute, indexed-indirect draw, presentation, and opt-in readback paths are implemented. DX12 hardware compute and graphics proofs passed on the local RTX 3080.
- KTX2/Basis parsing/decoding, progressive-residency data structures, bounded CPU worker/event primitives, six migrated examples, external PNG comparison, runtime/compiler package isolation, and canonical plan updates are present. `ez-gfx` texture loading currently expands to bounded RGBA8 and does not connect the worker/event path; the FFI delegates to it.
- Final post-format strict verification passed: `cargo clippy --workspace --all-targets --all-features -- -D warnings` and xtask source-line validation.

## Verification status

- `examples/shared` owns neutral Pod byte views, host attachment, winit lifecycle/input orchestration, bounded observability draining, environment-flag parsing, math/mesh preparation (including neutral basic primitive records), and benchmark/snapshot automation. Each example's sole Rust source, `main.rs`, owns its ez-gfx callbacks, direct resource/frame calls, renderer, shaders, and artifacts; every renderer calls safe `ez-gfx` APIs, and no example library or graphics-wrapper target remains.
- `include/ez_gfx_api.h` now declares all 43 production extern-C exports; strict GCC C11 and C++17 standalone header validation pass; FFI ABI suite passes 13/13.
- All six one-frame Metal Validation runs pass with zero diagnostics and zero dropped observations.
- `cargo check -p ez-gfx-examples --bins --tests` passes; the latest examples smoke passes 20/20.
- Vulkan benchmark smoke passes for all six binaries with one warmup frame, one measured frame, and one uncaptured terminal snapshot frame; each emits its stable numbered benchmark identity and zero dropped observations.
- Immutable example snapshots matched; they were not regenerated.
- `cargo fmt --all -- --check` and final strict workspace Clippy/source-line validation pass.
- On the local Windows RTX 3080, one-frame Vulkan and DX12 example runs both pass.
## Remaining verification

```text
cargo run -p xtask -- package x86_64-pc-windows-msvc 0.1.0 target/package-smoke
  blocked: the x86_64-pc-windows-msvc Rust target is not installed
```


## Constraints

- Runtime packages must not contain or depend on Slang, DXC, compiler crates, or source/JIT fallback.
- Artifact loading selects an exact backend, stage, entry, and semantic profile; malformed or mismatched data fails closed.
- The semantic floor is `ez-gfx-v1`; implemented DXIL requires Shader Model 6.5, not 6.6 direct heap indexing.
- Native handles and physical shader layouts remain backend-local. Public bindings use stable semantic IDs.
- FFI validates null/count/size/UTF-8 boundaries and catches panics. Non-null pointer validity remains the C caller's obligation.
- Snapshot-disabled presentation must not allocate or copy readback memory.

## Architecture backlog

### P0 — graph execution is discarded

- **Status:** Partial — compiled execution and backend synchronization are connected; alias lowering remains.
- **Evidence:** `crates/ez-gfx-runtime/src/frame.rs` retains compiled graphs and typed payloads; `render.rs::execute_compiled_graph` produces one immutable `FrameExecutionPlan`. `NativeFrameAdapter` validates waits and native resources, then lowers ordered transitions, pass boundaries, compute, graphics, readback, and present into one command buffer/list on Vulkan, DX12, and Metal. `crates/ez-gfx-runtime/tests/render.rs` proves reordered payload lookup, barriers, waits, coalescing, and atomic backend failure.
- **Impact:** Graph hazards, pass order, and submission atomicity reach native execution. Transient alias plans still cannot reduce memory.
- **Acceptance:** Lower compiled alias assignments into backend resource placement without changing executor ordering.

### P0 — multi-pass rendering clears prior work

- **Status:** Implemented for active-surface passes; managed targets remain separate backlog.
- **Evidence:** Vulkan, DX12, and Metal frame encoders retain one acquired drawable/back buffer across every compiled pass, honor load/store metadata, and present only after the full plan. `crates/ez-gfx-runtime/tests/render.rs::compatible_graphics_nodes_execute_inside_one_pass` proves pass coalescing. `crates/ez-gfx-ffi/tests/metal_present.rs` submits two distinct colored draws and verifies both survive in the captured frame.
- **Impact:** Repeated and ordered surface passes preserve prior work according to load/store declarations. Managed-target composition remains unavailable because target lifecycle is missing.
- **Acceptance:** Covered for surfaces; managed attachment proofs belong to the managed-render-target item.

### P0 — managed render targets are unreachable

- **Status:** Missing.
- **Evidence:** `crates/ez-gfx-runtime/src/binding.rs:7-19` and `crates/ez-gfx/src/state/frame/mod.rs` recognize render-target bindings, but neither the safe API nor FFI has a create/describe/release path; `ez-gfx::frame_submit` always renders the active surface.
- **Impact:** Shader-declared targets, relative scales, storage/sample transitions, and target history cannot be authored.
- **Acceptance:** Public target lifecycle validates dimensions/format/ownership, target handles bind successfully, resize/history work, and graph passes attach them.

### P1 — texture loading is synchronous RGBA8

- **Status:** Partial.
- **Evidence:** `crates/ez-gfx/src/state/texture.rs::load_texture` decodes and optionally generates every mip before upload; `crates/ez-gfx/src/state/native.rs` calls `create_texture_rgba8` for all backends. `crates/ez-gfx-runtime/src/texture.rs:17-22,143-177` expands supported KTX2 paths to RGBA8.
- **Impact:** Caller stalls and memory/bandwidth increase; native BC/ASTC upload, partial updates, eviction, and streaming control are unavailable.
- **Acceptance:** Context validation precedes work; asynchronous mip/region uploads expose cancellation/completion; native block formats remain compressed through upload.


### P1 — viewport and scissor are not part of the API

- **Status:** Missing.
- **Evidence:** `crates/ez-gfx-hal/src/lib.rs:353-359` exposes only cull/front-face/topology/blend; Vulkan and DX12 emit full-extent rectangles at `crates/ez-gfx-backend-vulkan/src/frame.rs` and `crates/ez-gfx-backend-dx12/src/native/frame.rs`.
- **Impact:** Subpasses, clipping, and target-relative viewports cannot be represented.
- **Acceptance:** Validated viewport/scissor state crosses the ABI and is emitted consistently by Metal, Vulkan, and DX12 with boundary tests.

### P1 — asset workers are disconnected from the public API

- **Status:** Partial.
- **Evidence:** `crates/ez-gfx-assets/src/lib.rs:224-276,290-414` implements bounded queues, Rayon workers, cancellation, permits, and events, but `crates/ez-gfx/Cargo.toml` does not depend on it and `ez-gfx` loading remains synchronous (`crates/ez-gfx/src/state/texture.rs::load_texture`).
- **Impact:** Worker safety exists in isolation; applications receive no asynchronous decode/upload lifecycle.
- **Acceptance:** `ez-gfx` submits bounded jobs, publishes decode/transcode/upload outcomes, supports cancellation, and ties GPU completion tokens to resource state; FFI exposes that lifecycle without reimplementing it.

### P1 — threading and global context mutex constrain ownership

- **Status:** Partial.
- **Evidence:** `crates/ez-gfx-runtime/src/lifecycle.rs:27-130` enforces creator-thread, health, owner, generation, and resource-kind checks; `crates/ez-gfx/src/state/mod.rs` stores all contexts behind one global mutex. Wrong-thread destruction is rejected while preserving the context; global serialization remains.
- **Impact:** Normal stale-handle use fails closed, but global serialization limits concurrency and cross-thread destruction can hide failures.
- **Acceptance:** Document ownership rules, validate destruction affinity or make destruction explicitly thread-safe, surface cleanup failures, and test concurrent independent contexts.

### P1 — platform surface and validation gaps

- **Status:** Partial.
- **Evidence:** `ez-gfx-hal::FrameExecutionBackend` is the common execution boundary; `render.rs::execute_compiled_graph` submits one immutable plan through it, and `ez-gfx` adapters lower that plan for Vulkan, DX12, and Metal. Context creation still accepts Vulkan only with Win32, DX12 only on Windows/Win32, and Metal only on Apple/MetalLayer (`crates/ez-gfx/src/state/context.rs::create_context`); `examples/shared/host.rs:31-68` handles only Win32/AppKit.
- **Impact:** Linux Vulkan surfaces are unavailable, and Windows native execution remains unverified on the current host.
- **Acceptance:** Add Linux surface support and native CI for macOS Metal, Windows Vulkan/DX12, and Linux Vulkan.

### P1 — diagnostics are weak around cleanup and asynchronous work

- **Status:** Partial.
- **Evidence:** `crates/ez-gfx/src/state/frame/mod.rs::frame_submit` records submit failures, but `crates/ez-gfx/src/state/context.rs::destroy_context` and resource-release paths discard release/destructor errors; worker events are not exposed through the public APIs.
- **Impact:** Resource leaks, upload failures, and partial submissions can be silent or lack correlation.
- **Acceptance:** Every failure has a correlated diagnostic/event, cleanup errors are observable, and queue overflow/cancellation/device-loss semantics are tested.

### P1 — FFI parity gaps

- **Status:** Partial.
- **Evidence:** `crates/ez-gfx-ffi/src/lib.rs:228-355,429-550` covers basic textures, pipelines, frames, and readback but has no texture update, target lifecycle, decoder callback, screenshot-save, graph authoring, or native compressed-upload API.
- **Impact:** Original integrations needing streaming, targets, callbacks, screenshots, or explicit graph control cannot migrate without ABI additions.
- **Acceptance:** Publish a parity matrix; add validated APIs and ABI tests for each required operation, including null/count/size/ownership/error boundaries.

### P2 — package and test coverage gaps

- **Status:** Partial.
- **Evidence:** `crates/ez-gfx-runtime/tests/{frame,frame_graph,render}.rs` cover retained graph execution, ordering, waits, transitions, pass coalescing, payload mapping, and failure boundaries. `crates/ez-gfx-ffi/tests/metal_present.rs` covers native separated-pass color/depth preservation and surface-free readback. `examples/tests/smoke.rs` covers all six binaries and snapshots. Target lifecycle, async FFI uploads, viewport variation, and non-macOS native execution remain uncovered.
- **Impact:** Core graph and active-surface multi-pass contracts have native Metal coverage; the remaining subsystems and platform matrix can still regress.
- **Acceptance:** Add target/upload/viewport regression suites, artifact freshness checks, and native Vulkan/DX12 CI/package smoke.

## Ordered implementation sequence

1. [Complete] Backend-neutral submission, transition, wait, pass, target, and completion-token interfaces.
2. [Complete] Compiled-graph execution retained and consumed by `FrameRecorder`.
3. Add the managed render-target manager, attachment views, resize/history, and multi-pass load/store behavior.
4. Connect bounded workers to asynchronous uploads; add texture updates, mip streaming, and native compressed-format paths.
5. Complete FFI APIs and enforce/document threading, destruction, ownership, and error propagation.
6. [Current] Metal, Vulkan, and DX12 route frame plans through the common executor; platform conformance remains.
7. Expand diagnostics, regression tests, artifact freshness checks, packaging checks, and performance benchmarks.


## New todos
- format slang shaders so that they look well with spacing and stuff
- inside the examples, call the shader compiler rather than precompiling the shaders
- .ezgfx should be renamed to .ezgfxshader and it should have at most one entrypoint per stage so that you wont need to include entrypoint name when calling load_shader
- update shaders so that there is one global ez_gfx_api.slang inside root which will contain the shared slang code. when importing the library the users will also import this shared slang code
- the shader should not use anything from the vk:: namespace or vk_location or whatever, the shaders are backend agnostic
- use newtype for handle u64 instead of passing raw u64 around
- write AGENTS.md with policies for agents on what to ensure inside the codebase (like keep all backends in sync, keep the ffi in sync with the bindings and so on)
- migrate the code to anyhow errors and use best practices for error handling
- use library for the linear algebra (mats, vecs, quaternions) instead of rolling your own inside the examples
- in fact analyse if there is anything in the whole project that would be better served by a library rather handrolling
- use rkyv for the serialization/deserialization of the shader format rather than custom bincoder
- use clap library for the xtask and shader compiler
