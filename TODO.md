# Handoff

## Completed baseline

- Cargo workspace with compiler-free runtime packages, safe typed Rust handles, C ABI v18 opaque `u64` handles, validated semantic/artifact/capability data, and panic containment.
- Vulkan, DX12, and Metal backend crates implement a backend-neutral HAL and use `gpu-allocator` 0.28.
- The compiler emits SPIR-V 1.5, Shader Model 6.5 DXIL, and Metal products in format-v3 `.ezgfxshader` containers. Archived count/string/metadata/provenance/variant ceilings are checked before owned decode. Runtime load validates required backend/stage reflection once before native calls, and metallib selection enforces platform, architecture, OS, SDK, language, and library compatibility. Compiler toolchain identity is recorded and bounded but is not a runtime admission criterion.
- The root `ez_gfx_api.slang` module supplies backend-agnostic shared declarations. Each non-distributed Rust example passes its adjacent Slang source to the compiler with SPIR-V, DXIL, and Metal targets in development mode, then loads validated artifact bytes in memory at process startup. These development binaries intentionally depend on compiler tooling; `ez-gfx`, runtime/FFI crates, and packaged runtime distributions remain compiler-free. The C textured cube invokes the same source-path compiler CLI from CMake. Generated shader artifacts are not tracked.
- Safe Rust resources use distinct context, surface, shader, texture, indirect-buffer, structured-buffer, and render-target handle types. The FFI performs explicit checked conversion while preserving opaque ABI values.
- Tooling uses `clap` derives and `anyhow`; library seams retain typed errors. Example math uses `glam`. Repository dependency review replaced applicable archive, traversal, hashing, temporary-file, serialization, and CLI helpers with maintained crates.

## Verification status

- Runtime shader loading chooses the artifact-owned entry point for each stage; callers provide no entry-point name.
- Shader sources import the root shared module and contain no Vulkan namespace, location, register, or physical binding syntax.
- `include/ez_gfx_api.h`, `bindings/bindings.xml`, and all production FFI exports describe ABI v18. The Win32 C textured cube builds against that header and exercises compute-written indexed-indirect graphics.
- Artifact tests cover format/version validation, malicious structurally valid archives exceeding semantic bounds, reflection failure across every backend, deterministic metallib compatibility selection, unique stage entry points, and generated example coverage.
- The CI backend matrix diagnoses the Windows SwiftShader SDK, tools, and ICD but compile-checks Vulkan native surfaces only: SwiftShader lacks required descriptor-indexing/update-after-bind capabilities. The hosted macOS row runs device-independent compiler, artifact, core, HAL, and runtime tests, then compile-checks Metal, `ez-gfx`, FFI, examples, and `metal_present`; Linux Vulkan and Windows DX12 also compile native tests without claiming hosted runtime coverage.
- Package CI builds and checks distinct runtime/compiler archives for Windows x64, Linux x64, and Apple Silicon.
- On the local Windows RTX 3080, one-frame Vulkan and DX12 example runs both pass.

## Hosted coverage limits

- GitHub-hosted Windows does not guarantee a D3D12 feature-level 12.1 adapter, so DX12 native GPU tests and the C example's DX12 path are compiled but not executed there.
- The Windows SwiftShader ICD is discoverable and `vulkaninfoSDK --summary` runs, but it reports `descriptorBindingSampledImageUpdateAfterBind=false`, `shaderSampledImageArrayNonUniformIndexing=false`, and zero update-after-bind pool capacity. The runtime remains fail-closed; live Vulkan software-adapter evidence is pending an adapter satisfying the required descriptor indexing.
- The current public Vulkan surface path is Win32-only, so Linux validates Lavapipe availability and compiles Vulkan tests without executing presentation.
- GitHub-hosted macOS runners provide the Apple compiler toolchain but no Metal device. Metal backend, FFI, example, and presentation tests compile there, but live Metal submission and presentation remain unverified pending a GPU-backed macOS runner.
- Hardware Vulkan and DX12 examples, including the DX12 logical-extent path, have local coverage; hosted runtime proof remains unavailable for Windows SwiftShader until a capable software adapter is available.

## Constraints

- Runtime packages must not contain or depend on Slang, DXC, compiler crates, or source/JIT fallback.
- Artifact loading selects an exact backend, profile, and stage. Each stage owns exactly one internal entry point; malformed, ambiguous, or mismatched data fails closed.
- The semantic floor is `ez-gfx-v1`; implemented DXIL requires Shader Model 6.5, not 6.6 direct heap indexing.
- Native handles and physical shader layouts remain backend-local. Public bindings use stable semantic IDs.
- FFI validates null/count/size/UTF-8 boundaries and catches panics. Non-null pointer validity remains the C caller's obligation.
- Snapshot-disabled presentation must not allocate or copy readback memory.

## Architecture backlog

### P0 — transient alias assignments are not lowered

- **Status:** Partial — compiled execution and backend synchronization are connected; alias lowering remains.
- **Evidence:** `crates/ez-gfx-runtime/src/frame.rs` retains compiled graphs and typed payloads; `render.rs::execute_compiled_graph` produces one immutable `FrameExecutionPlan`. `NativeFrameAdapter` validates waits and native resources, then lowers ordered transitions, pass boundaries, compute, graphics, readback, and present into one command buffer/list on Vulkan, DX12, and Metal. `crates/ez-gfx-runtime/tests/render.rs` proves reordered payload lookup, barriers, waits, coalescing, and atomic backend failure.
- **Impact:** Graph hazards, pass order, and submission atomicity reach native execution. Transient alias plans still cannot reduce memory.
- **Acceptance:** Lower compiled alias assignments into backend resource placement without changing executor ordering.

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

### P1 — thread-affine context cleanup remains explicit

- **Status:** Partial — Windows TLS abandonment is safe, but cross-thread explicit destruction and observable implicit cleanup failures remain unsupported.
- **Evidence:** `crates/ez-gfx/src/state/mod.rs` synchronizes only generational context-handle allocation. `ContextState` stays in creator-thread-local storage and operations remain creator-thread-affine. Normal thread TLS teardown invalidates handles synchronously; `ExitProcess` may terminate other threads without running their TLS destructors. Windows deliberately retains each entire abandoned context until process termination because DX12/COM cleanup under loader lock can deadlock, while non-Windows normal TLS teardown retains synchronous best-effort cleanup. State tests cover wrong-thread destruction, populated-state invalidation, and non-hanging creator-thread exit.
- **Impact:** Independent creator threads do not share a state mutex. Windows callers must explicitly destroy contexts for any reclamation before process exit and for cleanup error reporting; cross-thread destruction is rejected, and implicit cleanup cannot report failures.
- **Acceptance:** Support cross-thread explicit destruction and surface implicit cleanup failures without performing or joining native cleanup from Windows TLS destructors; preserve terminal handle invalidation and concurrent independent-context behavior.

### P1 — platform surface and validation gaps

- **Status:** Partial.
- **Evidence:** `ez-gfx-hal::FrameExecutionBackend` is the common execution seam, and all three adapters lower the same immutable plan. CI diagnoses the Windows SwiftShader SDK/tools/ICD but compile-checks Vulkan native surfaces because required descriptor indexing is unavailable; the hosted macOS row runs device-independent contracts and compile-checks Metal native surfaces because it has no Metal device; Linux Vulkan and Windows DX12 likewise compile native tests. Context creation still accepts Vulkan only with Win32, DX12 only on Windows/Win32, and Metal only on Apple/MetalLayer; example hosts cover Win32 and AppKit.
- **Impact:** Linux Vulkan surfaces remain unavailable. Hosted DX12 runtime execution remains unavailable because the runner does not guarantee the required adapter. Live Metal submission and presentation lack verification on a GPU-backed macOS runner. Live Vulkan software-adapter evidence also remains unavailable pending an adapter satisfying required descriptor indexing.
- **Acceptance:** Add Linux Vulkan surface support and execute its native presentation tests. Execute DX12 native tests on a runner with a guaranteed feature-level 12.1 adapter, Vulkan native tests on an adapter satisfying required descriptor indexing, and Metal native tests on a GPU-backed macOS runner.

### P1 — diagnostics are weak around cleanup and asynchronous work

- **Status:** Partial.
- **Evidence:** `frame_submit` records submit failures. Safe Rust `destroy_context` returns the first initialized-device wait or fallible release failure after terminal cleanup; pre-device Vulkan has no GPU work, so its `NotReady` wait is benign. Status-free resource releases still discard native destructor failures, and worker events remain unexposed.
- **Impact:** Resource leaks, upload failures, and partial submissions can be silent or lack correlation.
- **Acceptance:** Every failure has a correlated diagnostic/event, cleanup errors are observable, and queue overflow/cancellation/device-loss semantics are tested.

### P1 — FFI parity gaps

- **Status:** Partial.
- **Evidence:** `crates/ez-gfx-ffi/src/lib.rs:228-355,429-550` covers basic textures, pipelines, frames, and readback but has no texture update, target lifecycle, decoder callback, screenshot-save, graph authoring, or native compressed-upload API.
- **Impact:** Original integrations needing streaming, targets, callbacks, screenshots, or explicit graph control cannot migrate without ABI additions.
- **Acceptance:** Publish a parity matrix; add validated APIs and ABI tests for each required operation, including null/count/size/ownership/error boundaries.

### P2 — package and test coverage gaps

- **Status:** Partial.
- **Evidence:** Runtime suites cover retained graph execution, ordering, waits, transitions, pass coalescing, payload mapping, and failures. Artifact suites cover framed `rkyv` validation and startup-compiled example artifacts. Example smoke tests cover all six Rust scenes and snapshots. CI diagnoses SwiftShader capabilities, compile-checks Windows Vulkan native surfaces without GPU operations, and hosted macOS, Linux Vulkan, and Windows DX12 compile their native tests without claiming unavailable device coverage. The hosted macOS row still runs device-independent compiler and artifact contracts. CI builds/links the C textured cube on Windows without executing it. Package CI checks runtime/compiler separation, manifests, archives, export parity, and forbidden compiler-native imports. GPU performance and size baselines remain unverified.
- **Impact:** Managed targets, asynchronous uploads, viewport/scissor variation, Linux presentation, hosted Vulkan/DX12 runtime, and live Metal submission/presentation can still regress or remain unavailable.
- **Acceptance:** Add target/upload/viewport regression suites, Linux Vulkan presentation CI, Vulkan execution on an adapter satisfying required descriptor indexing, DX12 execution on guaranteed hardware, and Metal execution on a GPU-backed macOS runner.

## Ordered implementation sequence

1. Add the managed render-target manager, attachment views, resize/history, and multi-pass load/store behavior.
2. Connect bounded workers to asynchronous uploads; add texture updates, mip streaming, and native compressed-format paths.
3. Complete FFI APIs and enforce/document threading, destruction, ownership, and error propagation.
4. Expand diagnostics, managed-target/upload/viewport regression tests, Linux Vulkan presentation, DX12 hardware CI, packaging checks, and performance benchmarks.

## Tooling improvements

- setup nexttest
- setup miri and integrate with nexttest
