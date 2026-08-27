# P-019: Testing and golden snapshot harness

## Problem

Decide how to structure unit, integration, and visual snapshot tests across multiple backends, expanding pixel snapshot coverage for complex render graph shapes (fork/join topologies, scaled render areas, managed storage images, and history load/store).

## Prompt context

Full user prompt explicitly requires: "setup tests including snapshot tests."
Source evidence: `TODO.md` ("Add broader render target pixel snapshot coverage for more graph shapes. LoadTarget history and managed storage-image store/load now have snapshot tests, but fork/join and scaled-target render areas still need pixel assertions").
Original test evidence: `tests/snapshot.odin`, `tests/example6.odin`, `tests/snapshots/example6.expected.png`.

## Constraints and acceptance criteria

- Provide automated unit and integration tests for graph compilation, memory allocators, and reflection parsers.
- Provide a deterministic offscreen render and pixel readback harness comparing output against golden reference PNG images.
- Cover basic rendering, storage-image passes, scaled render targets, and fork/join render graph topologies.
- Explicit non-goals: manual interactive visual verification for CI/CD test runs.

## Dependencies

- Incoming dependency: `P-019` depends on `P-003` (HAL), `P-008` (Render Graph), and `P-016` (Draw submission).
- Outgoing dependency: `P-020` depends on `P-019` for test milestone verification and cutover acceptance.

## Unresolved questions

- How should cross-GPU rendering precision tolerances (e.g. perceptual hash or SSIM vs per-pixel tolerance) be handled across Vulkan, DX12, and Metal?
- Can snapshot tests run headless in CI environments lacking discrete GPUs (e.g. using Lavapipe/WARP/Metal software device)?

## Candidate solutions

### S-P-019-gpu-offscreen-pixel-goldens: Real backend offscreen rendering with per-pixel and tolerance-based goldens

#### Approach and integration

Run backend-specific integration tests against offscreen images. Vulkan uses an offscreen image or `VK_EXT_headless_surface`; DX12 uses an offscreen resource and readback heap; Metal uses a texture-backed render pass and blit readback. Store PNG goldens per backend/profile, and compare exact pixels where deterministic or configurable channel/region tolerances where floating-point output differs. Test fixtures explicitly construct history load/store, storage-image, fork/join, and scaled-target graphs.

#### Performance evidence

- **Test runtime:** Unknown until measured; depends on adapter startup, shader compilation, and readback synchronization. A test matrix spanning 3 backends multiplies setup cost by backend/profile count (`[INFERENCE]`).
- **Evidence boundary:** Vulkan documents headless surfaces, but equivalent CI availability and cross-driver pixel determinism for DX12/Metal remain unknown.

#### Tradeoffs and failure modes

- **Tradeoffs:** Exercises actual backend behavior and catches image-layout, format, and synchronization regressions.
- **Failure modes:** Driver differences, color-space conversion, and floating-point variation can cause false failures; backend-specific baselines increase storage and review burden. CI without a suitable adapter must report unavailable rather than silently pass.

#### Sources

- [Khronos VK_EXT_headless_surface](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_headless_surface.html) — headless Vulkan surface capability.
- [Vulkan rendering reference](https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderingInfo.html) — render-area semantics.
- `F:/Projects/oss/ez_gfx_api/tests/snapshot.odin` — incumbent snapshot harness.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — required graph-shape coverage.

### S-P-019-ir-deterministic-snapshots: Backend-independent graph/reflection snapshots plus selected GPU goldens

#### Approach and integration

Snapshot serialized shader reflection, graph topology, resource state transitions, allocation plans, and normalized draw commands with a text snapshot framework such as `insta`. Keep a smaller set of real GPU image goldens for representative backend smoke tests. Use canonical ordering and explicit numeric normalization before snapshotting.

#### Performance evidence

- **Test runtime:** IR snapshots avoid adapter startup and GPU readback; exact runtime is unknown until measured.
- **Coverage/scaling:** CPU-only snapshots scale with graph fixture count, while the reduced GPU suite scales with backend count (`[INFERENCE]`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Fast and reproducible for graph/compiler regressions; fewer GPU image assertions leave backend rasterization and synchronization defects less covered.
- **Failure modes:** Normalization can hide meaningful precision or ordering regressions; IR parity does not prove visual output.

#### Sources

- [insta documentation](https://docs.rs/insta/latest/insta/) — Rust snapshot storage and assertion model.
- `F:/Projects/oss/ez_gfx_api/tests/snapshot.odin` — incumbent image snapshot coverage.

### S-P-019-windowed-manual-screenshot-tests: Incumbent interactive screenshot capture

#### Approach and integration

Run interactive windowed examples and save screenshots to disk for manual inspection, retaining the original example-oriented workflow.

#### Performance evidence

- **Test runtime:** Unknown and operator-dependent; no reproducible CI timing or coverage measurement is available.

#### Tradeoffs and failure modes

- **Tradeoffs:** Simple and useful for exploratory visual debugging.
- **Failure modes:** Cannot automatically run in headless CI/CD pipelines; human review and desktop state make regressions easy to miss.

#### Sources

- `F:/Projects/oss/ez_gfx_api/tests/` — original test framework.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| --------- | ------------------------------- | -------------- | ---------------- | ----- |
| Real backend offscreen rendering with per-pixel/tolerance goldens | Highest backend coverage; runtime and cross-driver determinism unknown | High | Official Vulkan docs + direct source evidence | Adapter and driver matrix burden |
| Backend-independent IR snapshots plus selected GPU goldens | Fast CPU suite; limited GPU coverage; exact timing unknown | High | Official insta docs + direct source evidence | IR can miss visual defects |
| Interactive manual screenshot capture | Unbounded operator time; non-reproducible | Low | Direct source evidence | Cannot satisfy automated snapshot requirement alone |

## Selected solution

### Selection

`S-P-019-gpu-offscreen-pixel-goldens`: Real backend offscreen rendering with per-pixel and tolerance-based goldens.

### Selection rationale

`S-P-019-gpu-offscreen-pixel-goldens` directly fulfills the user prompt ("setup tests including snapshot tests") and expands graph coverage per TODO.md:
1. It validates end-to-end GPU rasterization, image format mapping, synchronization transitions, and shader execution against golden PNG images without requiring an interactive window or display server.
2. It expands coverage beyond simple rendering to include complex graph shapes: storage image read/write passes, scaled render target resolutions, history load/store (`LoadTarget`), and multi-node fork/join topologies.
3. It uses per-pixel comparison with configurable channel tolerance to account for floating-point rasterization variations across GPU hardware vendors.

### Rejected alternatives

- **`S-P-019-ir-deterministic-snapshots`**: Rejected as a standalone solution because text/IR snapshots cannot detect backend GPU driver bugs, incorrect attachment blending, format layout mismatches, or shader code-generation artifacts.
- **`S-P-019-windowed-manual-screenshot-tests`**: Rejected because manual screenshot inspection cannot run in headless CI/CD pipelines and fails the requirement for automated regression tests.

### Evidence summary

Headless offscreen rendering tests actual GPU command execution and memory readback without desktop environment dependencies (`[INFERENCE]` from Vulkan headless surface and offscreen rendering specifications).

### Key assumptions

- Target test environments have access to a GPU hardware device or software adapter (e.g. Lavapipe, WARP).
- Offscreen render targets can be read back to host memory via transfer copy staging buffers.

### Risks and mitigations

- **Risk:** Floating-point rounding and anti-aliasing differences across GPU vendors (NVIDIA, AMD, Intel, Apple) cause false-positive test failures.
- **Mitigation:** Implement configurable delta thresholds (e.g. max 1-2 least significant bits per channel or perceptual difference threshold) for image assertions.

### Validation actions

1. Implement headless golden snapshot tests for all 6 example scenes (Triangle, Cube, Compute, ImGui, Helmet, Sponza).
2. Add specific unit snapshot assertions for fork/join DAG shapes, scaled render targets, and storage-image read/write passes.
