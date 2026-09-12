# P-003: Multi-backend hardware abstraction

## Problem

Decide the internal hardware abstraction layer (HAL) design to support Vulkan, DirectX 12, and Metal across Windows, Linux, and macOS without compromising high-throughput bindless performance or introducing leaky backend-specific assumptions.

## Prompt context

Full user prompt explicitly requires "support for vulkan, dx12 and metal".
Source evidence: Original Odin codebase (`src/ctx.odin`, `src/render.odin`, `src/swapchain.odin`) was single-backend Vulkan using dynamic rendering (`VK_KHR_dynamic_rendering`), timeline semaphores, and bindless descriptor indexing.

## Constraints and acceptance criteria

- Support Vulkan (Windows/Linux), DX12 (Windows), and Metal (macOS) backends.
- Abstract device queues, timeline synchronization, bindless descriptor heaps/argument buffers, and dynamic render passes.
- Explicit non-goals: supporting legacy graphics APIs (OpenGL, DirectX 11) or software rasterizers.

## Dependencies

- Incoming dependency: `P-003` depends on `P-001` for workspace/crate structure.
- Outgoing dependency: `P-004`, `P-005`, `P-007`, `P-008`, `P-017` depend on `P-003` for device, queue, and swapchain abstraction.

## Unresolved questions

- Should the HAL be implemented via internal static generics (monomorphized traits) or dynamic dispatch (trait objects)?
- Should we use raw backend bindings (e.g. `ash`, `windows-rs`, `metal-rs`) or build on an existing mid-level HAL (such as `wgpu-hal`)?

## Candidate solutions

### S-P-003-custom-static-raw-hal

#### Architecture, integration, and applicability

Define a narrow internal backend contract for devices, queues, resources, synchronization, descriptors, render encoders, indirect draws, and presentation. Compile one concrete backend over `ash`, `windows` D3D12, or `objc2-metal`; keep backend-specific feature negotiation and barrier lowering behind that boundary. Vulkan timeline semaphores/dynamic rendering/descriptor indexing map conceptually to D3D12 fences/render-target state/descriptor heaps and Metal shared events/render encoders/argument buffers. This gives direct access to the incumbent bindless and MDI requirements.

#### Evidence, tradeoffs, and failure modes

Static dispatch removes virtual calls at the abstraction boundary by construction, but no workload shows that dispatch is material; binary size, CPU recording time, and GPU throughput are unknown. The cost is three unsafe implementations and a cross-backend semantic contract. Failures include assuming equivalence where APIs differ, incorrect resource-state translation, unsupported bindless tiers, queue/fence lifetime bugs, and backend divergence. It is disqualified only if implementation capacity cannot sustain three native backends.

#### Sources

- [Vulkan dynamic rendering](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_dynamic_rendering.html)
- [D3D12 descriptor heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview)
- [D3D12 fences](https://learn.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization)
- [Metal argument buffers](https://developer.apple.com/documentation/metal/buffers/about_argument_buffers)
- Original `src/ctx.odin`, `src/render.odin`, and `src/swapchain.odin`.

### S-P-003-wgpu-hal-foundation

#### Architecture, integration, and applicability

Place the ez-gfx graph and public API over `wgpu-hal`, using its Vulkan, DX12, and Metal implementations while bypassing the higher-level WebGPU API. Adapt ez-gfx resource/queue/pipeline operations to `wgpu_hal::Api` and capability queries; retain shader-declared targets and scheduling above it.

#### Evidence, tradeoffs, and failure modes

`wgpu-hal` documents itself as an unsafe, low-level, native graphics abstraction with explicit Vulkan/DX12/Metal backends. Its API is not promised as the stable public API of wgpu, so version upgrades and safety preconditions become project obligations. Reuse reduces duplicated backend plumbing, but required bindless arrays, indirect execution, resource aliasing, or synchronization controls may not map without backend escape hatches. No comparative ez-gfx benchmark exists; dispatch, compile time, and frame cost are unknown. Missing required capability or inaccessible native feature disqualifies it.

#### Sources

- [`wgpu-hal` crate documentation](https://docs.rs/wgpu-hal/latest/wgpu_hal/)
- [wgpu source backend implementations](https://github.com/gfx-rs/wgpu/tree/trunk/wgpu-hal/src)
- [wgpu limits](https://docs.rs/wgpu-types/latest/wgpu_types/struct.Limits.html)

### S-P-003-incumbent-vulkan-specialization

#### Architecture, integration, and applicability

Port the current Vulkan-only architecture directly and defer DX12/Metal. This preserves dynamic rendering, descriptor indexing, and timelines with minimal semantic translation.

#### Evidence, tradeoffs, and failure modes

The original repository is implementation evidence that this model runs its examples, but provides no controlled throughput baseline. It is useful as a migration baseline, not a complete answer. It is disqualified by the explicit three-API requirement.

#### Sources

- Original `src/ctx.odin`, `src/render.odin`, `src/swapchain.odin`, and example snapshots.

## Performance comparison

No comparable ez-gfx CPU/GPU benchmark exists. Hard feature reach and long-term control therefore precede speculative dispatch cost.

| Rank | Candidate | Vulkan/DX12/Metal | Bindless/MDI/alias control | Runtime/scaling | Reliability and implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Custom static raw HAL | Passes | Direct access to required native features | Dispatch removed by construction; actual benefit and code-size cost unknown | Highest implementation and unsafe-code burden | API capabilities sourced; project measurements missing |
| 2 | `wgpu-hal` foundation | Passes nominal backend coverage | Unverified for every required escape hatch and aliasing model | Runtime cost unknown | Reuses mature plumbing; unstable internal API/safety contract | Crate architecture sourced; workload fit incomplete |
| — | Vulkan incumbent | Hard failure: no DX12/Metal | Strong Vulkan-only fit | Existing functional evidence only | Lowest migration cost | Disqualified by prompt |

## Selected solution

**Selected: `S-P-003-custom-static-raw-hal`.**

Define the smallest backend-neutral contract needed by the public API and graph, then implement it directly over `ash`, Windows D3D12 bindings, and `objc2-metal`. Keep capability discovery and state/barrier lowering backend-local; compile a concrete backend rather than placing dynamic dispatch in hot recording paths.
The HAL adds `MeshStages`/`MeshPipelineState` plus allocation-free `validate_mesh_dispatch` (`MeshDispatchLimits`), `ShaderStage::Task = 5`/`Mesh = 6`, and one per-backend mesh pipeline create path; task stages without mesh support fail before allocation.

**Rejected:** the Vulkan incumbent fails the three-API requirement. `wgpu-hal` is rejected because the plan requires exact bindless, indirect, synchronization, and transient-aliasing controls that remain unverified through its evolving unsafe interface; it becomes viable if a capability spike proves every required path without private forks or leaky escape hatches.

**Assumptions and risks:** the project can sustain three unsafe backend implementations. Similar concepts are not semantically identical; synchronization and descriptor tiers can diverge. Static dispatch may not be performance-material and can increase code size.

**Validation:** implement no product code before a capability matrix/spike proves resource aliasing, bindless indexing, indirect drawing, timeline-equivalent synchronization, dynamic rendering, and presentation on each backend; benchmark identical command-recording workloads and record CPU, GPU, driver, backend, and binary size.
