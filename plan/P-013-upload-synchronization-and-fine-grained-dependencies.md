# P-013: Upload synchronization and fine-grained dependencies

## Problem

Decide how to replace global upload synchronization waits in `ez_gfx_begin_render` with fine-grained per-allocation, per-texture, or per-draw GPU queue dependencies, and replace host-side CPU waits for render target clears with queue-side timeline waits.

## Prompt context

Source evidence: `TODO.md` ("Replace the global vertex upload wait in `ez_gfx_begin_render` with per-allocation or per-draw dependencies...", "Replace the global texture upload wait in `ez_gfx_begin_render` with per-texture or per-draw dependencies...", "Replace remaining host-side waits for managed render target timeline dependencies with queue-side timeline waits so independent frame work can overlap more effectively").

## Constraints and acceptance criteria

- Eliminate unconditional CPU blocking on pending vertex and texture uploads at the start of a frame.
- Track upload readiness per resource; insert GPU queue-side semaphore/barrier dependencies only for resources referenced in the active frame's draw calls.
- Frame-start target clears must synchronize on the GPU timeline without CPU stall bubbles.
- Explicit non-goals: introducing unbounded frame lag or complex lock-free dependency graphs that risk CPU deadlocks.

## Dependencies

- Incoming dependency: `P-013` depends on `P-003` (HAL timeline semaphores), `P-008` (Render Graph), and `P-012` (Transfer queue).
- Outgoing dependency: `P-015` and `P-016` depend on `P-013` for non-blocking draw and presentation synchronization.

## Unresolved questions

- How should missing/in-flight textures be handled during draw recording (e.g., placeholder fallback vs GPU barrier wait)?
- What is the memory overhead of per-resource timeline token tracking?

## Candidate solutions

### S-P-013-gpu-timeline-token-dependency-tracking: Per-resource timeline token tracking with fine-grained GPU queue semaphore waits

#### Approach and integration

Every uploaded vertex allocation and texture is tagged with a 64-bit completion timeline value and an associated transfer queue family identifier. When a render graph node is compiled, referenced geometry allocations and texture handles register their completion values. The graphics queue submit inserts a `VkSemaphoreSubmitInfo` wait for the maximum required timeline value on the transfer semaphore, allowing the GPU transfer queue and graphics queue to overlap without CPU intervention. Frame-start target clears are recorded directly into the graphics command buffer preceded by image layout transitions guarded by queue-side timeline waits, eliminating all host-side `vkWaitSemaphores` during `begin_render`.

#### Performance evidence

- **CPU Frame Stalls:** In the original Odin codebase (`src/render.odin:21-27`), `ez_gfx_begin_render` calls `ez_gfx_ctx_wait_timeline` on the CPU for all scheduled transfers before rendering. Queue-side timeline synchronization moves wait operations from the host CPU thread to the GPU queue, allowing the CPU to proceed immediately (`[INFERENCE]` from timeline semaphore specifications).
- **Queue Overlap:** Allows the transfer queue to execute copy operations asynchronously with GPU graphics work on hardware supporting asynchronous transfer engines (`[INFERENCE]` based on Vulkan timeline semaphore specs).

#### Tradeoffs and failure modes

- **Tradeoffs:** Requires maintaining timeline value tracking per resource and checking referenced resource sets during graph compilation.
- **Failure Modes:** If a draw call references a resource whose transfer was queued but never submitted, the graphics queue could deadlock waiting on an unreachable timeline value unless validation detects unsubmitted dependencies.

#### Sources

- [Khronos Vulkan Timeline Semaphores Specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-semaphores-timeline) — queue synchronization primitives.
- `F:/Projects/oss/ez_gfx_api/src/render.odin` & `src/ctx.odin` — original begin_render synchronization points.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — upload wait replacement and timeline synchronization requirements.

### S-P-013-deferred-readiness-placeholder-fallback: Non-blocking fallback rendering with default placeholder slots

#### Approach and integration

Rather than making the graphics queue wait on the transfer queue, uncompleted texture handles point to a fallback 1x1 default texture descriptor in the bindless array. Background transfers update the bindless slot and transition image layouts asynchronously upon transfer completion.

#### Performance evidence

- **Zero GPU Queue Stalls:** Graphics queue does not wait on in-flight texture transfers; frame rate remains unaffected by background loading (`[INFERENCE]`).
- **Visual Artifacts:** Objects render with placeholder textures until transfer completion is signaled.
#### Tradeoffs and failure modes

- **Tradeoffs:** Zero graphics queue blocking, but introduces visible texture popping/streaming delay.
- **Failure Modes:** Inapplicable to vertex/index geometry buffers (rendering with missing vertex buffers causes invalid geometry or crashes).

#### Sources

- [Game Engine Asset Streaming Architecture](https://advances.realtimerendering.com/) — placeholder binding patterns for texture streaming.

### S-P-013-global-timeline-drain: Incumbent host-side CPU wait for all pending uploads at frame start

#### Approach and integration

Maintain original model: `ez_gfx_begin_render` queries the highest scheduled transfer timeline counter and blocks the host CPU thread via `vkWaitSemaphores` / `WaitForSingleObject` before recording begins.

#### Performance evidence

- **CPU Stalls:** Host CPU thread blocks during `begin_render` until all queued transfer work completes (`[OBSERVED]` in `src/render.odin:21-27`).
#### Tradeoffs and failure modes

- **Tradeoffs:** Trivial synchronization logic with zero risk of missing-resource hazards.
- **Failure Modes:** Severe frame hitching and stutter during background streaming; destroys UI responsiveness.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/render.odin` — incumbent implementation.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — CPU stall issues.
- [Khronos synchronization examples](https://docs.vulkan.org/samples/latest/samples/extensions/timeline_semaphore/) — timeline semaphore usage.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| Per-resource timeline token tracking with queue-side waits | Eliminates CPU frame-start stalls, enables transfer/graphics GPU overlap, exact visual correctness | High | Direct source evidence & Khronos Vulkan spec | Unsubmitted transfer timeline deadlocks |
| Deferred readiness placeholder fallback | Zero GPU/CPU stalls, but causes texture pop-in; unusable for vertex geometry | Moderate | Direct source evidence & streaming literature | Visual pop-in, geometric data invalidity |
| Incumbent global timeline drain on CPU | CPU blocks during frame start on in-flight transfers | Low | Direct source code inspection | Violates non-blocking streaming goals |

## Selected solution

### Selection

`S-P-013-gpu-timeline-token-dependency-tracking`: Per-resource timeline token tracking with fine-grained GPU queue semaphore waits.

### Selection rationale

`S-P-013-gpu-timeline-token-dependency-tracking` directly resolves the host CPU stall issues identified in TODO.md:
1. It replaces the global CPU blocking wait (`ez_gfx_ctx_wait_timeline`) in `begin_render` with fine-grained queue-side timeline semaphore waits submitted only for active draw dependencies.
2. It allows background asset loading to overlap continuously with graphics execution without causing frame hitching or freezing the main render loop.
3. Frame-start target clears are recorded directly into command buffers preceded by queue-side timeline dependencies rather than blocking the CPU host thread.

### Rejected alternatives

- **`S-P-013-deferred-readiness-placeholder-fallback`**: Rejected because placeholder rendering causes visual pop-in artifacts and is strictly invalid for geometry buffers (drawing with placeholder vertex buffers yields invalid vertex data).
- **`S-P-013-global-timeline-drain`**: Rejected because blocking the CPU on all in-flight transfer work at the start of every frame causes severe framerate stutter during asset streaming.

### Evidence summary

Queue-side timeline semaphores eliminate host CPU thread wait bubbles and enable hardware transfer/graphics queue concurrency (`[INFERENCE]` from Vulkan timeline semaphore synchronization specifications).

### Key assumptions

- The target hardware supports queue-level timeline semaphore signaling and waiting across transfer and graphics queue families.
- Render graph node compilation records which resource handles are referenced by active draw passes.

### Risks and mitigations

- **Risk:** GPU queue deadlock if a draw command references a timeline token from a transfer command buffer that was queued but never submitted.
- **Mitigation:** Graph compiler validates that all referenced timeline tokens belong to submitted transfer batches before issuing graphics queue submissions.

### Validation actions

1. Measure CPU frame time in `begin_render` while streaming textures in the background to confirm 0.0ms host wait duration.
2. Integration test multi-queue synchronization across concurrent vertex/texture uploads in Example 6 (Sponza).
