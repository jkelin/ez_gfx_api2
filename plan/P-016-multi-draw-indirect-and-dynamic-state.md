# P-016: Multi-draw indirect and dynamic state

## Problem

Decide how to provide ergonomic and zero-overhead Multi-Draw Indirect (MDI) buffer acquisition, population, and execution, while adding hardware per-draw scissor and viewport controls (replacing shader `discard` clipping for ImGui and UI renderers).

## Prompt context

Source evidence: `TODO.md` ("Add hardware per-draw scissor support for ImGui and other UI renderers. Example 4 currently enforces clip rectangles in the fragment shader via `discard`, which is correct but less efficient than dynamic scissor state", "Expose per-pipeline or per-draw viewport and scissor rectangle controls").

## Constraints and acceptance criteria

- Allow acquiring, filling (CPU or compute shader), and submitting MDI indexed draw buffers.
- Expose hardware dynamic viewport and scissor rectangle state per draw command or pipeline node.
- Provide efficient rendering path for Dear ImGui using hardware scissor rectangles instead of fragment shader discard.
- Explicit non-goals: fallback emulated draw-loops for hardware that lacks indirect draw capability.

## Dependencies

- Incoming dependency: `P-016` depends on `P-003` (HAL dynamic state), `P-007` (Pipeline binding), and `P-008` (Render Graph execution).
- Outgoing dependency: `P-019` depends on `P-016` for rendering test scenes and ImGui integration.

## Unresolved questions

- Should scissor/viewport rectangles be stored inline in extended indirect command buffers or bound via dynamic state arrays?
- How to preserve backward compatibility for simple pipeline nodes that do not specify custom scissors?

## Candidate solutions

### S-P-016-per-command-dynamic-scissor-and-batch-mdi: Structured MDI buffer with dynamic scissor/viewport batches

#### Approach and integration

Provide `acquire_indirect_buffer()` returning a mapped indirect buffer handle storing `DrawIndexedCommand` structs (`index_count`, `instance_count`, `first_index`, `vertex_offset`, `first_instance`). For rendering UI and dynamic scenes:
- The render graph node accepts an optional array of `Rect` scissor/viewport definitions corresponding to command offsets.
- During command buffer recording, the engine binds the pipeline with `VK_DYNAMIC_STATE_SCISSOR` / `VK_DYNAMIC_STATE_VIEWPORT`, iterates the draw command ranges, issues `vkCmdSetScissor` / `RSSetScissorRects` / `setScissorRect`, and executes `vkCmdDrawIndexedIndirect` / `ExecuteIndirect` / `drawIndexedPrimitives` for each distinct scissor batch.
- For ImGui (Example 4), consecutive UI draw calls with identical clip rectangles are merged into single indirect draws, setting the hardware scissor once per batch.

#### Performance evidence
- **UI Shading ALU Savings:** Using hardware rasterizer scissor clipping eliminates fragment shader evaluations for geometry outside the scissor rectangle (`[INFERENCE]` from fixed-function rasterizer scissor clipping specifications). Exact ALU savings depend on UI quad overlap and window clipping geometry.
- **Early-Z Preservation:** Discard instructions in fragment shaders can prevent hardware early depth tests on architectures requiring post-fragment depth updates; hardware rasterizer scissors avoid this penalty (`[INFERENCE]` from GPU rasterization pipeline architecture).

#### Tradeoffs and failure modes

- **Tradeoffs:** If every individual draw call has a distinct scissor rectangle, issuing multiple `vkCmdSetScissor` calls increases CPU command buffer recording time slightly compared to a single monolithic MDI call.
- **Failure Modes:** Scissor rectangles with negative coordinates or extents exceeding the framebuffer dimensions must be clamped to avoid Vulkan/DX12 validation layer errors.

#### Sources

- [Khronos Vulkan Dynamic State Guide](https://docs.vulkan.org/guide/latest/dynamic_state.html) — dynamic scissor and viewport specification.
- [NVIDIA Shader Execution & Discard Optimization](https://developer.nvidia.com/blog/optimizing-vulkan-fragment-shaders/) — impact of discard vs rasterizer scissor on Early-Z.
- `F:/Projects/oss/ez_gfx_api/src/indirect_buffer.odin` & `src/imgui.odin` — original MDI and ImGui implementations.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — hardware scissor and viewport control requirements.

### S-P-016-device-generated-commands-dgc: GPU-driven indirect command signatures with inline scissor state

#### Approach and integration

Use D3D12 `ExecuteIndirect` with a command signature containing both `D3D12_INDIRECT_ARGUMENT_TYPE_SCISSOR` and `D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED`, and `VK_EXT_device_generated_commands` on Vulkan.

#### Performance evidence

- **Single GPU Dispatch:** Executes all draws and scissor changes in a single GPU dispatch without CPU command interleaving.
- **Hardware Portability:** `VK_EXT_device_generated_commands` has limited cross-vendor support on Vulkan (primarily NVIDIA) and no native equivalent in Metal without compute shader preprocessing.

#### Tradeoffs and failure modes

- **Tradeoffs:** Extremely fast GPU-driven culling, but lacks universal hardware support across AMD/Intel/Apple GPUs.
- **Failure Modes:** Driver incompatibility and crashes on non-supporting Vulkan devices and macOS Metal.

#### Sources

- [Vulkan Device Generated Commands Proposal](https://vulkan.lunarg.com/doc/view/latest/mac/antora/features/latest/features/proposals/VK_EXT_device_generated_commands.html) — DGC specification and limitations.

### S-P-016-shader-discard-clipping: Incumbent fragment shader discard with static fullscreen scissors

#### Approach and integration

Maintain original Odin model: pipeline executes a single monolithic MDI draw call with a static fullscreen viewport and scissor, passing clip rectangles via push constants or vertex attributes and calling `discard` in the fragment shader.

#### Performance evidence

- **Overdraw Penalty:** Every clipped pixel executes fragment shading ALU, texture sampling, and register allocation before being discarded (`[OBSERVED]` in `src/imgui.odin:110-130`), wasting significant GPU shader core cycles on complex UI layouts.

#### Tradeoffs and failure modes

- **Tradeoffs:** Single draw command submission with zero CPU state switches.
- **Failure Modes:** Disables Early-Z; severely degrades performance when rendering multi-window UI scenes.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/imgui.odin` — incumbent implementation.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — known UI clipping inefficiency.
- [Microsoft D3D12 ExecuteIndirect](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-executeindirect) — indirect command execution constraints.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| Dynamic state MDI records with hardware scissor | Hardware rasterizer clipping avoids out-of-bounds fragment invocations, universal multi-GPU support | High | GPU architectural specifications & source inspection | Minor command recording overhead on fragmented scissors |
| GPU-driven Device Generated Commands (DGC) | Single GPU dispatch for draw+scissor, but limited to specific GPUs | Low-Moderate | LunarG & Khronos specifications | Vendor-locked, unsupported on Metal/older GPUs |
| Incumbent fragment shader discard clipping | Evaluates fragment shader before discard, disables early-Z on overlapping UI | Low | Direct source code inspection | Violates UI performance TODOs |

## Selected solution

### Selection

`S-P-016-per-command-dynamic-scissor-and-batch-mdi`: Structured MDI buffer with dynamic scissor/viewport batches.

### Selection rationale

`S-P-016-per-command-dynamic-scissor-and-batch-mdi` resolves the UI rasterization inefficiency in Example 4 (ImGui) and satisfies the dynamic state controls requirement from TODO.md:
1. It utilizes native hardware dynamic state (`VK_DYNAMIC_STATE_SCISSOR` / `RSSetScissorRects` / `setScissorRect`) to cull clipped UI quads at the fixed-function rasterizer stage before fragment shading.
2. By eliminating fragment-shader-based `discard` clipping, it preserves hardware Early-Z depth optimization and hierarchical Z-cull efficiency on all target GPUs.
3. It batches consecutive draw commands sharing identical clip rectangles, maximizing indirect draw command batching while keeping command recording overhead negligible.

### Rejected alternatives

- **`S-P-016-device-generated-commands-dgc`**: Rejected due to lack of cross-vendor hardware availability (`VK_EXT_device_generated_commands` is predominantly NVIDIA-only and lacks Metal support on macOS).
- **`S-P-016-shader-discard-clipping`**: Rejected because executing fragment shaders and evaluating `discard` for out-of-bounds pixels wastes significant GPU ALU and texture sampling bandwidth on dense UI layouts.

### Evidence summary

Hardware rasterizer clipping culls out-of-bounds geometry prior to fragment shading and avoids Early-Z disabling instructions (`[INFERENCE]` from GPU rasterization architecture).

### Key assumptions

- Vulkan, DX12, and Metal pipelines support dynamic scissor and viewport rectangle updates during command buffer recording.
- UI renderers (such as Dear ImGui) supply clip rectangles associated with contiguous draw command ranges.

### Risks and mitigations

- **Risk:** Passing invalid or negative scissor coordinates causes graphics validation layer panics.
- **Mitigation:** Intercept and clamp scissor rectangles against the active render target / framebuffer bounds during draw node recording.

### Validation actions

1. Profile GPU fragment shader invocations and render pass execution times in Example 4 (ImGui) comparing hardware scissor vs fragment discard.
2. Verify dynamic viewport and scissor overrides in offscreen snapshot test suites.
