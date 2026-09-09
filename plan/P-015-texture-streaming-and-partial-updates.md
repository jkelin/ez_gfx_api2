# P-015: Texture streaming and partial updates

## Problem

Decide how to implement streamed texture uploads with minimal-required mip readiness, partial texture region updates (`ez_gfx_texture_update_region` for dynamic UI font atlases), per-texture sample-ready state with deferred descriptor updates, and upload performance profiling counters.

## Prompt context

Source evidence: `TODO.md` ("Add streamed texture upload with minimal-required mip readiness...", "Example 4 handles ImGui 1.92 font atlas uploads... add `ez_gfx_texture_update_region` for partial `Want_Updates` uploads instead of full atlas reloads", "Add per-texture sample-ready state and defer descriptor updates until the graphics handoff has transitioned textures to `SHADER_READ_ONLY_OPTIMAL`", "Add texture upload profiling counters...").

## Constraints and acceptance criteria

- Support progressive mip uploads allowing low-res mips to render immediately while higher mips stream in.
- Expose `ez_gfx_texture_update_region` for sub-rectangle updates (e.g. dynamic font atlas glyph updates).
- Defer bindless descriptor heap population until GPU layout transition to `SHADER_READ_ONLY_OPTIMAL` is completed.
- Provide structured diagnostic profiling counters for decode time, staging bytes, queue latency, and handoff latency.
- Explicit non-goals: building a complete virtual texturing / mega-texture runtime system.

## Dependencies

- Incoming dependency: `P-015` depends on `P-007` (Descriptors), `P-012` (Staging pools), and `P-014` (Texture formats).
- Outgoing dependency: `P-016` depends on `P-015` for dynamic font atlas rendering in UI passes.

## Unresolved questions

- How should fallback default textures (1x1 pink/grey placeholder) be mapped into bindless slots before mip streaming finishes?
- What is the lock-free synchronization cost of real-time profiling counters?

## Candidate solutions

### S-P-015-progressive-mip-streamer-and-subregion-updates: Progressive mip streamer with subregion staging copies and atomic telemetry

#### Approach and integration

Implement texture streaming with progressive mip levels:
1. **Mip Streaming:** A texture is initialized with full mip chain allocation in GPU memory. The coarsest mips (e.g. mips 3..N) are uploaded in the initial transfer batch. Once the coarsest mip transition to `SHADER_READ_ONLY_OPTIMAL` is recorded and signaled on the GPU timeline, the bindless descriptor slot is updated to reference the texture image view. Subsequent background batches stream mips 2, 1, 0, executing image subresource copies and layout transitions without reallocating descriptors.
2. **Partial Subregion Updates:** Expose `ez_gfx_texture_update_region(texture, mip_level, offset_x, offset_y, width, height, data, data_size)`. Writes region bytes to staging memory and records `vkCmdCopyBufferToImage` / `CopyTextureRegion` covering the specified bounding rectangle.
3. **Deferred Descriptors:** Descriptor indexing writes occur only after the graphics queue handoff transition completes, preventing shaders from sampling uninitialized or invalid layout image memory.
4. **Profiling Counters:** Maintain lock-free `AtomicU64` counters tracking decode latency (Âµs), staging bytes transferred, transfer submit latency (Âµs), and graphics handoff latency (Âµs).

#### Performance evidence
- **Font Atlas Transfer Reduction (Geometry Arithmetic):** In ImGui dynamic font atlas updates (Example 4), adding glyphs dynamically requires updating the atlas bitmap. Re-uploading a full 2048x2048 RGBA8 atlas transfers 2048 * 2048 * 4 = 16,777,216 bytes (~16.78 MB). A partial sub-rectangle update (e.g. 64x64 RGBA8 glyph quad) transfers 64 * 64 * 4 = 16,384 bytes (16 KB), yielding an exact 1024:1 reduction in transferred bytes for that update (`[INFERENCE]` from rectangular region area arithmetic).
- **Framerate Smoothness:** Coarser mips have lower byte volume and complete transfer faster than full resolution base mips (`[INFERENCE]`). Exact streaming latency depends on storage I/O and PCIe transfer bandwidth.

#### Tradeoffs and failure modes

- **Tradeoffs:** Requires managing per-mip image subresource layout transitions and tracking dirty sub-regions.
- **Failure Modes:** Partial texture uploads on compressed block textures (BCn/ASTC) must align offset and extent to 4x4 block boundaries.

#### Sources

- [Vulkan Subresource Layout Transitions & Mip Streaming](https://docs.vulkan.org/guide/latest/image_subresources.html) â€” subresource synchronization patterns.
- `F:/Projects/oss/ez_gfx_api/src/texture_manager.odin` â€” original texture manager and ImGui font atlas handling.
- `F:/Projects/oss/ez_gfx_api/TODO.md` â€” streaming, partial updates, deferred descriptors, and profiling counter requirements.

### S-P-015-dynamic-atlas-virtual-suballocation: Dynamic atlas virtual suballocator with page updates

#### Approach and integration

Manage a large virtual texture atlas where individual textures/glyphs are assigned sub-tiles. Updates commit virtual texture pages dynamically.

#### Performance evidence

- **Complexity Overhead:** High CPU metadata overhead for page table management and shader texture coordinate patching.

#### Tradeoffs and failure modes

- **Tradeoffs:** Highly flexible for huge virtual worlds, but disproportionately complex for an easy-to-use graphics library.
- **Failure Modes:** Shader sampling across tile borders causes seam clamping artifacts without custom shader border filtering.

#### Sources

- [Virtual Texturing in Modern Engines](https://advances.realtimerendering.com/) â€” virtual texture atlas architecture.
- [Microsoft DirectX texture streaming guidance](https://learn.microsoft.com/en-us/windows/uwp/gaming/complete-guide-to-directx12-programming) â€” resource update integration context.

### S-P-015-full-reload-synchronous-textures: Incumbent full atlas and mip reloads with immediate descriptor writes

#### Approach and integration

Maintain original Odin behavior: every texture upload transfers all mips in a single monolithic copy. ImGui font atlas changes reload the full atlas bitmap, reallocating staging buffers and issuing full-image copy commands.

#### Performance evidence
- **Bandwidth Overhead (Arithmetic):** Re-uploading an entire 2048x2048 atlas for small glyph modifications transfers ~16.78 MB of data per full atlas reload (`[INFERENCE]` from image dimension arithmetic), increasing PCIe staging bus traffic.

#### Tradeoffs and failure modes

- **Tradeoffs:** Simple implementation with no subresource state tracking.
- **Failure Modes:** UI stuttering, high VRAM staging churn, and potential race conditions when descriptors are updated before GPU image layout transition finishes.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/imgui.odin` & `src/texture_manager.odin` â€” incumbent implementation.
- `F:/Projects/oss/ez_gfx_api/TODO.md` â€” font atlas performance bottleneck notes.
- [Khronos image subresource reference](https://docs.vulkan.org/spec/latest/chapters/resources.html) â€” image update and subresource constraints.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| Progressive mip streamer + subregion updates + atomic telemetry | 1024:1 transfer data reduction for 64x64 sub-regions on 2k atlas; latency dependent on I/O | High | Image arithmetic & source code inspection | Subresource layout barrier complexity |
| Dynamic atlas virtual suballocator | High flexibility, but requires shader coordinate patching | Low-Moderate | Industry literature | High complexity, texture coordinate seams |
| Incumbent full atlas/mip reload | ~16.78 MB transferred per 2k atlas reload, PCIe churn | Low | Direct source code inspection | Violates UI and streaming performance TODOs |

## Selected solution

### Selection

`S-P-015-progressive-mip-streamer-and-subregion-updates`: Progressive mip streamer with subregion staging copies and atomic telemetry.

### Selection rationale

`S-P-015-progressive-mip-streamer-and-subregion-updates` provides:
1. validated partial sub-rectangle uploads through `Texture::update_region`;
2. progressive mip streaming, allowing coarse mips to become sample-ready before finer levels;
3. deferred bindless descriptor publication after transfer and frame safety; and
4. upload telemetry without changing the owning `Texture` lifecycle.

### Rejected alternatives

- **`S-P-015-dynamic-atlas-virtual-suballocation`**: Rejected due to high architectural complexity and texture coordinate seam artifacts, which would unnecessarily complicate the user-facing graphics API.
- **`S-P-015-full-reload-synchronous-textures`**: Rejected because re-uploading full atlas bitmaps on every UI glyph addition causes significant PCIe staging traffic (~16.78 MB per 2k atlas reload) and frame stuttering.

### Evidence summary

Updating a 64x64 glyph sub-region instead of a full 2048x2048 atlas reduces transferred bytes by 1024:1 (16 KB vs 16.78 MB) (`[INFERENCE]` based on geometric area calculations).

### Key assumptions

- Dynamic UI renderers (such as ImGui) track dirty sub-rectangles across font atlas updates.
- GPU drivers support per-mip subresource copy commands (`vkCmdCopyBufferToImage` / `CopyTextureRegion`).

### Risks and mitigations

- **Risk:** Partial updates on block-compressed textures (BCn/ASTC) fail if rectangle extents are not 4x4 block-aligned.
- **Mitigation:** Require block-aligned offsets and extents except at the logical mip's right/bottom edge; reject invalid requests rather than clamping caller intent. DX12 rounds staging footprints to physical blocks while preserving logical mip dimensions.

### Validation actions

1. Benchmark transfer byte reduction and frame time during dynamic font atlas glyph generation in Example 4 (ImGui).
2. Verify progressive mip readiness order and deferred descriptor binding in background streaming tests.

### Implementation and evidence status

The implementation allocates the complete mip chain, submits coarse levels first with distinct completion values, gates descriptor publication on transfer completion and frame safety, and exposes partial updates and telemetry. `Texture::set_residency` changes only the sampled mip range: native views retain all image storage. Logical eviction does not reclaim GPU memory or stop finer uploads.

RTX 3080 Vulkan/DX12 regressions verify format sampling, region updates, queued ordering, residency changes, and submitted-work retirement/reuse. These backend tests predate the owning-wrapper cutover and do not verify dropping `Texture`.

DX12 explicitly returns `Unsupported` for non-block-aligned BC base dimensions, including through the C ABI. Valid 28Ã—12 bases retain a supported 7Ã—3 mip at level two; both clipped edge regions are sampled and updated in the regression. No padded-resource UV workaround is used. Both RTX backends reject ASTC explicitly.

The selected 2048Ã—2048 atlas/64Ã—64 glyph workload completed eight warmup and 64 measured frames on both backends with identical final pixels: dirty regions staged 1 MiB versus 1 GiB for full-image updates. Wall times include CPU/presentation/completion costs and do not establish a general frame-time improvement. See [exact measurements and retained proof](../docs/textures.md#verification-and-remaining-evidence).

Apple M2 Pro Metal now passes BC1/BC3/BC7 and ASTC linear/sRGB sampling at 8Ã—4 and 7Ã—3 bases, clipped 7Ã—3 mip updates, explicit residency promotion/demotion, and public `NotReady`â†’published transitions after frame retirement. Native GPU gates prove coarse progress, hidden/visible updates and retirement; batching, cancellation and failure paths also execute. Layer/pipeline sRGB lowering was corrected to match the shared graph; exact midtone readback prevents a primary-color-only parity claim. [Evidence](../docs/textures.md#verification-and-remaining-evidence) is adapter-specific, not all-device validation. Per-batch GPU waits are removed on Metal texture owners (see P-012); Vulkan/DX12 keep documented exceptions. Logical eviction still retains full allocation and does not stop fine uploads, as selected.

This change adds: Metal `mipFilter` follows `min_filter` (a 4:1 minification probe fails pre-fix with mean 255 and passes post-fix); polling reaps completed Vulkan/Metal frame slots before the descriptor gate so sustained rendering unblocks publication without `wait_idle`; DX12 fence-gate refusals stay retryable while allocation/capability/device errors stay terminal; direct RGBA8 readback requires full residency and rejects compressed captures identically on all backends; safe and C sampler validation agree on finite `1.0..=16.0` anisotropy. A probe-exposed Vulkan defect (missing transfer-source usage, no shader-read restore) is fixed with validation-clean evidence. See [exact evidence](../docs/textures.md#verification-and-remaining-evidence).
