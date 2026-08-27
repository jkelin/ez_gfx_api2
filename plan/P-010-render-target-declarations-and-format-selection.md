# P-010: Render target declarations and format selection

## Problem

Decide how to enforce explicit shader target attributes as the single source of truth for render target intent, dynamically query device-supported depth/stencil candidate formats (preferring D24/D16 when D32 is not needed), provide per-target clear values, and model rich image usage metadata.

## Prompt context

Source evidence: `TODO.md` ("Document and enforce that explicit target attributes are the source of truth for engine render target intent...", "Choose depth/stencil formats from device-supported candidates and prefer D24/D16 where the extra D32 precision is not required...", "Model render target declarations with richer Vulkan usage metadata...", "Allow per-target clear values for render target initialization each frame").

## Constraints and acceptance criteria

- Explicit shader target attributes (`ColorTarget`, `DepthTarget`, `RWTexture2D`) must dictate format, scale, load/store actions, and sampleability.
- Query device format properties across Vulkan, DX12, and Metal to select optimal supported depth/stencil formats (e.g. `D32_SFLOAT`, `D24_UNORM_S8_UINT`, `D16_UNORM`).
- Allow custom clear colors/depths per declared target.
- Explicit non-goals: fallback to unsupported legacy formats without user-facing diagnostic errors.

## Dependencies

- Incoming dependency: `P-010` depends on `P-003` (HAL format queries) and `P-006` (Reflection metadata).
- Outgoing dependency: `P-008` and `P-009` depend on `P-010` for target creation and format verification.

## Unresolved questions

- How should format fallback prioritization be configured (e.g., preference list vs engine default)?
- Should per-target clear values be declared inside the shader attribute or configured via the public API?

## Candidate solutions

### S-P-010-shader-authority-with-runtime-format-probe

#### Architecture, integration, and applicability

Compile shader attributes into canonical target intent: kind, usage, scale, sampleability, load/store policy, abstract format/candidate class, and default clear. At device creation, P-003 queries Vulkan format properties, D3D12 format support, or Metal pixel-format capabilities and resolves a supported physical format. The selected format enters pipeline keys and target allocation; unsupported intent produces a diagnostic rather than a silent semantic fallback.

#### Evidence, tradeoffs, and failure modes

All three APIs separate shader interfaces from attachment formats and expose device/pipeline validation. D16 uses 2 bytes/texel and D32 4 before implementation-specific compression/tiling; a raw storage/bandwidth reduction is arithmetic, while frame-time benefit is unmeasured inference. D24 layout/support varies. Startup probe latency and clear cost are unknown and must be measured by candidate count/device/backend. Failures include depth-only versus depth-stencil aspect loss, incompatible usage/sample count, reflection optimization dropping intent, pipeline/attachment mismatch, and backend format mapping drift.

#### Sources

- [Vulkan format-property query](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceFormatProperties2.html)
- [Vulkan dynamic-rendering pipeline formats](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineRenderingCreateInfo.html)
- [D3D12 format support](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-feature-levels#format-support)
- [Metal pixel formats](https://developer.apple.com/documentation/metal/mtlpixelformat)
- Original `src/shader.odin`, `src/render_target.odin`, and TODO lines 13, 19, 37-39.

### S-P-010-shader-intent-with-host-instance-policy

#### Architecture, integration, and applicability

Keep shader attributes authoritative for semantic target intent and allowed format class, but let host policy choose among declared candidates and supply per-frame clear values. A declaration default applies when the host omits a clear. Host values are validated for aspect/type/range and are accepted only when load action is clear; arbitrary format overrides outside declared intent fail.

#### Evidence, tradeoffs, and failure modes

Vulkan, D3D12, Metal, and wgpu configure attachment formats in pipeline/pass descriptors, while load/clear operations are pass-instance state. This supports dynamic sky/background clears without recompiling shaders. It risks weakening “source of truth” unless the override boundary is narrow and explicit. No benchmark shows a performance difference from declaration-only clears; clear bandwidth and tile behavior are backend/GPU dependent. Failures include non-finite/out-of-range values, mismatched depth/stencil aspects, pipeline-cache key omission, and policy fallback that changes precision or usage.

#### Sources

- [Vulkan rendering attachment clear/load state](https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderingAttachmentInfo.html)
- [D3D12 graphics PSO formats](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_graphics_pipeline_state_desc)
- [wgpu render-pass attachment operations](https://docs.rs/wgpu/latest/wgpu/struct.RenderPassColorAttachment.html)
- [Metal render pipeline descriptor](https://developer.apple.com/documentation/metal/mtlrenderpipelinedescriptor)

### S-P-010-incumbent-hardcoded-mapping

#### Architecture, integration, and applicability

Retain fixed mappings and hardcoded clear values. The original parser maps the `d32_float` intent to Vulkan `D24_UNORM_S8_UINT`, supports a small color set, and render-graph clears are constants.

#### Evidence, tradeoffs, and failure modes

This is implemented evidence, not portability or performance evidence. It can select unsupported formats, silently change requested precision/aspects, and cannot express per-target clears. It is disqualified by the explicit probing, rich metadata, and clear-value TODOs.

#### Sources

- Original `src/shader.odin`, `src/render_graph.odin`, `src/render_target.odin`, and TODO lines 19 and 37-39.

## Performance comparison

No candidate has measured format-probe, clear, bandwidth, or frame-time data. Shader authority and capability validation are hard constraints; raw bytes-per-texel arithmetic is not a frame-time benchmark.

| Rank | Candidate | Hard constraints | Startup/runtime | Memory/bandwidth | Reliability/implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Shader authority with runtime format probe | Passes authoritative declarations, device probing, defaults/clears | Probe and clear cost unknown | D16 has smaller raw storage than D32; realized benefit unknown | Central mapping and diagnostics | API queries sourced |
| 2 | Shader intent with host instance policy | Conditional: host override must remain within declared intent | Dynamic clear cost unknown | Same resolved formats | Flexible but authority boundary is easier to violate | Pass-instance pattern sourced |
| — | Hardcoded incumbent | Hard failure: no reliable probing/rich clears; can alter requested semantics | Existing functionality only | No portable claim | Lowest cost | Disqualified by TODO |

## Selected solution

**Selected: `S-P-010-shader-authority-with-runtime-format-probe`.**

Compile target attributes into canonical intent and defaults. Resolve abstract depth/stencil candidate classes through backend capability queries at device creation, and include the chosen physical format in pipeline and target keys. Treat unsupported intent as an explicit diagnostic. Keep per-target clear defaults in declaration metadata; a later API may override values only if the declaration explicitly permits it.

**Rejected:** hardcoded mapping fails the TODOs. General host instance policy is not selected because unrestricted overrides weaken the source-of-truth rule; it becomes viable as a narrowly declared dynamic-clear capability without changing format, usage, scale, or load/store intent.

**Assumptions and risks:** declarations can express candidate precision/aspect classes without ambiguous fallback; compiler reflection preserves them. D24 support/layout differs by backend. Mapping errors, aspect loss, pipeline mismatch, and unsupported sample/usage combinations remain risks. Performance is unknown.

**Validation:** build a cross-backend format capability matrix; test supported and unsupported candidate orders, depth-only/stencil, sample count, usage, and clear typing; snapshot canonical metadata and diagnostics; record probe count/time, allocated bytes, clear GPU time, and frame impact for equivalent D16/D24/D32 workloads.
