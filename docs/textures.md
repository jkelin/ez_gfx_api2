# Textures

`ez-gfx` splits texture loading across the context owner, a bounded Rayon pool, and a backend transfer queue. The caller receives a generational `TextureHandle` before decoding or GPU upload completes.

## Public Rust interface

The safe crate exports these functions at its root:

```rust
pub fn load_texture(
    context: ContextHandle,
    source: TextureSource,
    bytes: &[u8],
    generate_mips: bool,
    config: &TextureConfig,
) -> Result<TextureHandle, EzGfxResult>;

pub fn poll_texture_load(context: ContextHandle, texture: TextureHandle) -> EzGfxResult;
pub fn cancel_texture_load(context: ContextHandle, texture: TextureHandle) -> EzGfxResult;
pub fn texture_binding(
    context: ContextHandle,
    texture: TextureHandle,
) -> Result<u32, EzGfxResult>;
pub fn texture_residency(
    context: ContextHandle,
    texture: TextureHandle,
) -> Result<(u32, u32), EzGfxResult>;
pub fn set_texture_residency(
    context: ContextHandle,
    texture: TextureHandle,
    resident_mips: u32,
) -> EzGfxResult;
pub fn unload_texture(context: ContextHandle, texture: TextureHandle);
pub fn update_texture_region(
    context: ContextHandle,
    texture: TextureHandle,
    region: TextureRegion<'_>,
) -> EzGfxResult;
pub fn texture_upload_telemetry(
    context: ContextHandle,
) -> Result<TextureUploadTelemetrySnapshot, EzGfxResult>;
pub fn register_texture_decoder(
    source: u8,
    callback: TextureDecodeCallback,
) -> Result<(), TextureError>;
pub fn unregister_texture_decoder(source: u8) -> Result<(), TextureError>;
```

`load_texture` copies `bytes` and returns after bounded admission. The intended readiness contract is: `NotReady` is transient, `Ok` permits binding, and every other result is terminal for that request. `texture_binding` returns the stable bindless texture-heap index, not a `PublicBinding`.

**Readiness warning — source-inferred, not reproduced:** registry completion can advance before a frame-safe descriptor rewrite. `texture_binding` can then return the registry index without checking published mip count. Consequently, `Ok` alone may not establish initial descriptor publication during in-flight rendering. See [the completion/publication sequence](../crates/ez-gfx/src/state/texture.rs#L433-L543) and [registry binding lookup](../crates/ez-gfx-runtime/src/texture.rs#L870-L879). The examples below illustrate the intended contract, not proof that this case is safe.

## Rust load and render flow

This follows `examples/02_textured_cube`: buffer resources use `PublicBinding`, while the texture index travels in push constants to the shader’s `[BindlessTextureHeap]`.

```rust
use bytemuck::{Pod, Zeroable, bytes_of};
use ez_gfx::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    mvp: [[f32; 4]; 4],
    texture_id: u32,
    padding: [u32; 3],
}

struct StreamedTexture {
    handle: Option<TextureHandle>,
    binding: u32,
    fallback_binding: u32,
    ready: bool,
}

impl StreamedTexture {
    fn load(
        context: ContextHandle,
        png: &[u8],
        fallback_binding: u32,
    ) -> Result<Self, EzGfxResult> {
        let config = TextureConfig {
            width: 0,
            height: 0,
            mip_count: 0,
            destination: TextureDestination::Auto,
            sampler: TextureSamplerDesc {
                min_filter: SamplerFilter::Linear,
                mag_filter: SamplerFilter::Linear,
                max_anisotropy: 1.0,
                address_u: SamplerAddressMode::Clamp,
                address_v: SamplerAddressMode::Clamp,
                address_w: SamplerAddressMode::Clamp,
            },
        };
        let handle = load_texture(context, TextureSource::Png, png, true, &config)?;

        Ok(Self {
            handle: Some(handle),
            binding: fallback_binding,
            fallback_binding,
            ready: false,
        })
    }

    // Poll on the context creator thread before beginning this frame.
    fn poll(&mut self, context: ContextHandle) -> Result<(), EzGfxResult> {
        if self.ready {
            return Ok(());
        }
        let Some(handle) = self.handle else {
            return Ok(());
        };

        match poll_texture_load(context, handle) {
            EzGfxResult::Ok => {
                // Intended ready-use point; see the readiness warning above.
                self.binding = texture_binding(context, handle)?;
                self.ready = true;
                Ok(())
            }
            EzGfxResult::NotReady => {
                self.binding = self.fallback_binding;
                Ok(())
            }
            terminal => {
                // Keep rendering with the ready fallback and retire the failed handle.
                unload_texture(context, handle);
                self.handle = None;
                self.binding = self.fallback_binding;
                Err(terminal)
            }
        }
    }

    fn unload(&mut self, context: ContextHandle) {
        if let Some(handle) = self.handle.take() {
            unload_texture(context, handle);
        }
    }
}

fn render_frame(
    context: ContextHandle,
    surface: SurfaceHandle,
    shader: ShaderHandle,
    indirect: IndirectBufferHandle,
    positions: StructuredBufferHandle,
    pipeline_state: DynamicPipelineState,
    mvp: [[f32; 4]; 4],
    texture: &mut StreamedTexture,
) -> Result<(), EzGfxResult> {
    // A terminal error is reported, but the frame can still use the fallback.
    if let Err(error) = texture.poll(context) {
        eprintln!("texture load failed: {error:?}");
    }

    let push = Push {
        mvp,
        texture_id: texture.binding,
        padding: [0; 3],
    };
    let bindings = [PublicBinding {
        name: "positions".to_owned(),
        resource: ResourceIdentity::Structured(positions),
    }];

    status(begin_render(context, surface))?;
    status(render_add_graphics(
        context,
        shader,
        indirect,
        &bindings,
        pipeline_state,
        bytes_of(&push),
    ))?;
    status(finish_render(context))
}

fn status(value: EzGfxResult) -> Result<(), EzGfxResult> {
    match value {
        EzGfxResult::Ok => Ok(()),
        error => Err(error),
    }
}
```

The fallback must already be resident and have its binding cached. If initial `load_texture` returns `QueueFull`, no handle escaped: retain the fallback and retry admission in a later frame. After successful admission, `NotReady` retains the fallback. `InvalidArgument`, `InvalidContext`, `NativeFailure`, `Unsupported`, `DeviceLost`, `QueueFull`, or `Cancelled` is terminal when returned while progressing the request; log it, unload the handle when still owned, and decide whether the application can continue.

Under the intended contract, the ready-use point is the successful `texture_binding` call inside `poll`, before `begin_render`; the readiness warning above qualifies same-frame use in the current implementation. Do not add the texture to `PublicBinding`: that interface names structured, indirect, and render-target resources. The shader uses the index instead:

```slang
[BindlessTextureHeap(EZ_GFX_MAX_TEXTURES)]
ParameterBlock<TextureHeap> texture_heap;

TextureEntry entry = texture_heap.entries[push.texture_id];
float4 color = entry.texture.Sample(entry.sampler, uv);
```

## C ABI load and render flow

The C ABI uses the same ordering and shares the readiness warning above. This excerpt assumes `context`, `surface`, `png_bytes`, `png_size`, `fallback_binding`, `shader`, `indirect`, buffer `bindings`, `binding_count`, `dynamic_state`, and `current_mvp` are already valid:

```c
typedef struct TexturePush {
    float mvp[16];
    uint32_t texture_id;
    uint32_t padding[3];
} TexturePush;

EzGfxTextureDesc desc = {0};
desc.source_format = EzGfxSourceTextureFormat_Png;
desc.destination_format = EzGfxTextureDestinationFormat_Rgba8Unorm;
desc.generate_mips = 1;
desc.min_filter = EzGfxTextureFilter_Linear;
desc.mag_filter = EzGfxTextureFilter_Linear;
desc.max_anisotropy = 1.0f;
desc.address_mode_u = EzGfxTextureAddressMode_ClampToEdge;
desc.address_mode_v = EzGfxTextureAddressMode_ClampToEdge;
desc.address_mode_w = EzGfxTextureAddressMode_ClampToEdge;

EzGfxTexture texture = 0;
uint32_t texture_binding = fallback_binding;
int texture_ready = 0;

EzGfxResult result =
    ez_gfx_texture_load(png_bytes, png_size, &desc, &texture, context);
if (result == EzGfxResult_QueueFull) {
    /* No handle escaped. Keep the fallback and retry load in a later frame. */
} else if (result != EzGfxResult_Ok) {
    /* Invalid input/context is terminal; no texture handle was produced. */
}

/* Once per frame, before ez_gfx_begin_render: */
if (texture != 0 && !texture_ready) {
    result = ez_gfx_texture_poll(texture, context);
    if (result == EzGfxResult_Ok) {
        result = ez_gfx_texture_get_binding(texture, &texture_binding, context);
        if (result == EzGfxResult_Ok) {
            texture_ready = 1; /* Intended contract; see readiness warning. */
        }
    }
    if (result != EzGfxResult_Ok && result != EzGfxResult_NotReady) {
        fprintf(stderr, "texture load failed: %u\n", (unsigned)result);
        ez_gfx_texture_unload(texture, context);
        texture = 0;
        texture_binding = fallback_binding;
    }
}

TexturePush push = {0};
memcpy(push.mvp, current_mvp, sizeof(push.mvp));
push.texture_id = texture_binding;

result = ez_gfx_begin_render(surface, context);
if (result != EzGfxResult_Ok) goto cleanup;

result = ez_gfx_render_add_vertex_pipeline(
    shader,
    indirect,
    bindings,
    binding_count,
    &dynamic_state,
    &push,
    (uint32_t)sizeof(push),
    context);
if (result != EzGfxResult_Ok) goto cleanup;

result = ez_gfx_finish_render(context);
if (result != EzGfxResult_Ok) {
    /* Submission failed and presentation was skipped. */
    goto cleanup;
}
cleanup:

if (texture != 0) {
    ez_gfx_texture_unload(texture, context);
}
```

`ez_gfx_texture_load` writes `texture` only on success. The source bytes and descriptor need remain valid only through that call. `ez_gfx_update_texture_region` likewise copies validated region bytes before returning; compressed offsets and non-edge extents must align to the format’s 4×4 blocks. `ez_gfx_texture_get_binding` and telemetry queries write output only on success. As in Rust, C `EzGfxBinding` does not carry a sampled texture; pass the cached heap index through shader data such as push constants.

## Frame-graph readiness

Polling before binding is the public usage rule. Separately, once the owner thread has created the native texture and admitted its transfer, frame recording interns native-created textures with their texture-transfer completion tokens. The compiled graph lowers those tokens into transfer-to-graphics waits before sampled access. `frame_enqueue_readback` applies the same readiness token before transfer readback.

This is a submission-safety net, not an alternate readiness interface. A pending request has no public bindless index, so application code must not guess an index or select it in shader data. Continue selecting the fallback until `poll_texture_load` and `texture_binding` succeed. Recording graphics while another native-admitted texture is pending can still order GPU graphics behind that upload because the current graphics graph declares native-created textures as sampled resources; this does not block the CPU polling call.

GPU waits do not themselves publish a descriptor. They do not resolve the source-inferred initial-publication issue described above.

## Lifecycle

1. Call `load_texture` on the context creator thread.
2. Retain the returned handle. The input bytes may be released immediately; `load_texture` copies them before returning.
3. Call `poll_texture_load` regularly, normally once per frame. Polling moves completed CPU work into native texture creation and transfer submission, then checks GPU completion without waiting.
4. Continue rendering with a fallback texture while polling returns `NotReady`.
5. Under the intended readiness contract, after polling returns `Ok`, call `texture_binding` and use its stable bindless index; see the current publication warning above.
6. Call `unload_texture` for early reclamation, or let `destroy_context` reclaim the texture.

All context and resource calls are creator-thread-affine. A texture handle belongs to one context and remains invalid in every other context. Successful cancellation or unload invalidates it immediately. Native storage and its descriptor/view retire only after accepted transfer and submitted graphics dependencies complete; a retired binding slot cannot be reused early. Slot reuse increments the generation, so stale handles remain invalid.

`load_texture` is non-waiting, not allocation-free: it validates arguments, copies the source payload, reserves a handle, and attempts bounded CPU admission. It never waits for decode, transcode, mip generation, transfer submission, or GPU completion.

## Progress and render integration

`poll_texture_load(context, texture)` returns:

- `Ok`: the first coarse transfer completion was observed. Intended meaning: safe to sample while finer mips stream. Initial descriptor publication may lag; see the warning above.
- `NotReady`: CPU work, owner-thread submission, or first sample-ready GPU completion remains pending.
- A terminal error: decoding, validation, native allocation, transfer, device, context, or handle validation failed.

Polling is part of progress. CPU workers publish decoded results to a bounded channel; the context creator drains that channel and creates/submits native textures. An application that neither polls nor calls another texture-progressing operation will not advance a decoded request into GPU submission.

`texture_binding`, `texture_residency`, and `set_texture_residency` also make one nonblocking progress pass. Do not put a pending texture into frame-graph bindings: first observe `Ok` from `poll_texture_load`, then resolve its binding. The binding index is reserved with the handle and stays stable until unload.

`texture_residency` returns the currently exposed contiguous coarse mip count and immutable total after CPU decode and native transfer admission. It returns `NotReady` while CPU work is pending. `set_texture_residency` accepts `1..=total`: decreasing the count logically evicts finer levels without discarding their allocation, while increasing it publishes only transfer-complete levels and returns `NotReady` until the requested range and a frame-safe descriptor rewrite are available. Each backend submits decoded levels from the coarsest mip toward level zero under distinct completion values, so no sampled view exposes an uninitialized fine level.

Residency is a sampled-view limit, not a GPU-memory budget: Vulkan replaces an image view, DX12 rewrites an SRV, and Metal replaces a view of the retained parent texture. All three retain the full mip-chain allocation and uploaded bytes. Decreasing residency neither frees image storage nor cancels finer uploads; increasing it reuses completed data. Storage retires on unload/cancellation after outstanding GPU uses, not on logical mip eviction. This matches P-015's selected full-chain-allocation design; sparse allocation and physical mip reclamation are not implemented.

`wait_idle(context)` is the explicit blocking alternative. It drains accepted CPU texture work, submits decoded results, flushes transfer owners, and waits for native completion. Use it for loading screens, tests, or shutdown—not a latency-sensitive render loop.

## Cancellation and cleanup

`cancel_texture_load` succeeds in the CPU stage and after native transfer-worker admission while the first sample-ready completion remains pending. It atomically invalidates the public handle, marks uncommitted native jobs to be skipped, and emits a phase-specific cancellation record. A CPU decoder already executing may finish internally, and a native job already committed to the driver may complete, but neither can revive the handle; submitted resources remain deferred until transfer and graphics dependencies retire. Cancellation after the load is sample-ready returns `InvalidArgument`; use `unload_texture` instead.

Unloading a pending request cancels it. Unloading a native texture immediately invalidates the handle while deferring descriptor/view and storage reclamation until its last accepted transfer and all submitted graphics frames retire. The C unload function is `void`, so C callers do not receive reclamation failures; terminal context destruction performs dependency-ordered cleanup.

Destroying the context stops new CPU admission, drains or discards accepted work safely, joins worker ownership, waits for native work when initialized, and invalidates every child handle. Never retain a texture handle beyond its context.

## Backpressure

Admission is deliberately bounded:

- source payload: at most 64 MiB per request;
- CPU pool: at most 64 in-flight jobs and 64 MiB of aggregate admitted source bytes;
- decoded-result channel: 64 entries;
- each backend transfer-owner channel: 64 jobs.

The Rayon thread count is `max(1, available_parallelism - 1)`, leaving one logical CPU for the host/render thread. A full CPU budget makes `load_texture` return `QueueFull` immediately and no handle escapes. Retry in a later frame after polling existing requests. Do not busy-loop on `QueueFull`.

Transfer admission is also non-waiting. A full or failed native transfer worker turns the affected request into a terminal failure during polling; it does not block until queue space appears.

## Decode, transcode, and mip behavior

### Source and destination support

These are source-level capabilities, not a claim that every cell has native pixel evidence. The [decoder dispatch](../crates/ez-gfx-runtime/src/texture.rs#L274-L347), [KTX2 branches](../crates/ez-gfx-runtime/src/texture.rs#L412-L474), and [format mappings](../crates/ez-gfx-runtime/src/texture.rs#L590-L634) define admission.

| Input | Current destination behavior | Limits |
| --- | --- | --- |
| Raw RGB8/RGBA8; BMP, JPEG, PNG, TGA | RGBA8 linear or sRGB; `Auto` is linear RGBA8 | RGB expands to RGBA; encoded images convert to RGBA8. No runtime BC/ASTC encoding. |
| Direct native KTX2 | Preserves RGBA8, BC1, BC3, BC7, or ASTC 4×4, linear/sRGB | Requires no supercompression and an admitted compression family. An explicit destination must match exactly: no conversion or decompression fallback. |
| KTX2 UASTC/ETC1S | Dependency-selected BC/ASTC/RGBA8; `Auto` preserves color-space metadata | Accepts unsupercompressed UASTC or BasisLZ ETC1S only. Explicit BC1/BC3 transcode requests fail; see below. |
| Standalone `.basis` | Explicit RGBA8/BC1/BC3/BC7/ASTC 4×4, linear/sRGB | Requires `basis`. `Auto` chooses linear BC7, then linear ASTC, then linear RGBA8; it does not preserve sRGB automatically or choose ETC1S-specific BC1/BC3. |
| Custom source `128..=255` | Application returns validated mips in a supported native format | Can supply raw blocks indirectly; no built-in DDS or raw compressed source variant. |

`TextureDestination` exposes only `Auto` and linear/sRGB RGBA8, BC1, BC3, BC7, and ASTC 4×4. BC2/BC4/BC5/BC6H, other ASTC block shapes, HDR/float texture destinations, arrays, cubemaps, and 3D textures are absent from this loading contract. Inputs and aggregate decoded mip bytes are each limited to 64 MiB. KTX2 Zstd/Zlib supercompression is unsupported. These restrictions concern loaded textures, not separate render-target formats.

**KTX2 transcode gaps:** the [adapter](../crates/ez-gfx-runtime/src/texture.rs#L501-L553) passes a compression family, not an explicit target, to `basisu_c_sys`. Its selector emits BC7/BC4/BC5/ASTC/RGBA32, not BC1/BC3, so explicit BC1/BC3 requests fail the final format comparison despite enum mappings existing. Single/two-channel `Auto` can select BC4/BC5, which [the output mapping](../crates/ez-gfx-runtime/src/texture.rs#L611-L634) rejects. There is no retry to an admitted fallback. KTX2 `Auto` passes no sRGB override; [standalone Basis `Auto`](../crates/ez-gfx-runtime/src/texture/basis.rs#L30-L40) always selects linear storage. Use an explicit sRGB destination for standalone assets requiring it.

### Features and backend admission

The `basis` feature on `ez-gfx` or `ez-gfx-ffi` enables standalone `TextureSource::Basis` / C source code 7; disabled builds return `Unsupported`. It does **not** remove native transcoder linkage: [runtime dependencies](../crates/ez-gfx-runtime/Cargo.toml#L8-L19) include `image`, `ktx2`, and `basisu_c_sys` unconditionally. Optional KTX2/Basis decoder linkage required by [selected P-014](../plan/P-014-basis-universal-and-compressed-textures.md#selected-solution) remains incomplete. No runtime shader compiler is involved.

| Backend | Runtime compression admission | Qualification |
| --- | --- | --- |
| Vulkan | Union of advertised BC and ASTC-LDR device features | [Device probe](../crates/ez-gfx-backend-vulkan/src/device.rs#L835-L845); actual adapter-dependent support, not universal ASTC availability. |
| Direct3D 12 | BC only | [Admission](../crates/ez-gfx-backend-dx12/src/native/device.rs#L232); no ASTC native target mapping. |
| Metal | BC **or** ASTC | [Probe](../crates/ez-gfx-backend-metal/src/native/device.rs#L27-L31) selects BC when `supportsBCTextureCompression`, otherwise assumes ASTC. It neither positively queries ASTC nor reports both families; BC-capable Apple hardware cannot select ASTC through this capability contract. |

RGBA8 is the common uncompressed target. Explicit compressed destinations require the admitted family but still face the source-specific restrictions above. Native mappings alone do not establish execution evidence.

Application source codes `128..=255` use a registered `TextureDecodeCallback`. Rust callbacks are `Send + Sync + 'static` and may run concurrently. Registration snapshots an `Arc` into each accepted job, so unregistering prevents new admission without invalidating queued work. C registers paired decode/release callbacks plus `user_data`: successful output pointers remain valid until ez-gfx copies all mips and invokes release exactly once. The application must retain callback code and `user_data` until unregister and all accepted requests complete.

Raw inputs require nonzero `width` and `height`, and their byte length must match exactly. For encoded inputs, zero descriptor dimensions accept decoded dimensions; nonzero values are assertions checked after decode. A nonzero `mip_count` similarly asserts the decoded/generated count.

When mip generation is requested, a one-level RGBA8 source is box-filtered down to 1×1 in the Rayon job. Any existing valid multi-level chain is preserved, even if it stops before 1×1. A one-level compressed source with generation requested is rejected; an existing compressed chain passes through. Dimensions, block geometry, every mip length, ordering, count, and aggregate size are validated before native allocation. See [generation](../crates/ez-gfx-runtime/src/texture.rs#L354-L405) and [validation](../crates/ez-gfx-runtime/src/texture.rs#L641-L683).

### Original implementation parity

The [Odin manager](../reference/ez_gfx_api/src/texture_manager.odin#L55-L118) allowed replacing/clearing built-in BMP–KTX2 decoder callbacks; current registrations are custom IDs only and built-ins are always available. Original source regions remained borrowed until the [texture-loaded callback](../reference/ez_gfx_api/src/texture_manager.odin#L1047-L1050); current loading copies source bytes immediately and exposes polling/bounded runtime events, not an equivalent per-load completion callback.

Original [KTX2 decoding](../reference/ez_gfx_api/src/texture_manager.odin#L815-L841) expanded Basis content to RGBA32 and extracted level zero. Preserved source mip chains, native compressed storage, standalone Basis, progressive residency, and partial updates are additions, not original parity regressions. DDS, wider BC/ASTC formats, and optional linkage must be assessed against selected P-014 rather than credited to the original loader.

Original [mip generation](../reference/ez_gfx_api/src/texture_manager.odin#L1401-L1476) used Vulkan GPU blits with linear filtering; current generation uses CPU RGBA8 box filtering, so identical filtering results are not guaranteed.

## Transfer and staging contract

The shared transfer module owns:

- a bounded dedicated transfer-owner thread;
- adaptive batches flushed at 32 MiB, 64 copies, or a 200 µs collection deadline;
- power-of-two staging buckets from 64 KiB through 64 MiB;
- best-fit reuse only after the associated completion token retires;
- idle trimming after 256 pool epochs;
- monotonic queue-local completion tokens.

Texture staging is backend-owned because row pitch, block layout, image layout, and native allocation rules differ. Each backend copies decoded mip or update bytes into reusable host-visible buckets and retains buckets until completion. Geometry transfer uses the same policy and worker machinery but an independent queue timeline.

**Batching gap:** adaptive worker collection is not native cross-texture copy coalescing. [Vulkan](../crates/ez-gfx-backend-vulkan/src/transfer.rs#L34-L40) and [DX12](../crates/ez-gfx-backend-dx12/src/native/transfer.rs#L55-L61) group each texture mip/region by its unique completion value, deliberately preserving real per-mip readiness. [Metal](../crates/ez-gfx-backend-metal/src/native/transfer.rs#L25-L28) commits each prebuilt command buffer separately. The cross-texture native batching promised by [selected P-012](../plan/P-012-transfer-queue-staging-and-batching.md#selected-solution) remains a design gap; pooled staging and bounded transfer owners are implemented.

| Backend | Native transfer path | Completion and ownership |
| --- | --- | --- |
| Vulkan | Prefers a transfer-only queue family. It otherwise requests a secondary queue from the graphics family when available and falls back to the universal queue. Texture and geometry workers use separate queues when queue counts permit. | Timeline semaphores order graphics release, copy, and graphics acquire. Each ownership acquire repeats its release's exact old/new layouts. Four command-resource slots retire by completion. Presentation semaphores belong to swapchain images, not frame slots. |
| Direct3D 12 | A COPY queue records staged buffer/texture copies. A DIRECT queue waits on the copy fence and records state transitions required for graphics use. Texture and geometry have independent copy workers. | Native fences serialize copy-to-direct ownership and expose completion. Four allocator/list/event slots are recycled after their fence values complete. |
| Metal | A dedicated `MTLCommandQueue` creates blit command buffers and `MTLBlitCommandEncoder` copies. Submission ownership moves to the transfer thread. | GPU events order prior graphics reads before region writes and completed texture writes before subsequent draws/readbacks. Ordered command-buffer values and terminal status retain staging through completion. |

The fallback paths preserve correctness on adapters without a dedicated transfer family or enough independent queues; they reduce overlap, not semantics.

## Partial updates and telemetry

`update_texture_region(context, texture, region)` validates a resident texture, mip level, bounds, format-specific byte length, and compressed-block geometry before copying the caller’s bytes. Admission is nonblocking and bounded. Uncompressed rows are tightly packed. BC and ASTC offsets must be block-aligned; width and height must be multiples of four unless the rectangle reaches that mip’s right or bottom edge. Accepted updates enter the same backend transfer owner and graphics handoff as initial uploads.

Updates accepted before frame submission are included even when draws were recorded earlier. Submission refreshes texture dependencies. Vulkan and DX12 flush worker submission before enqueuing a graphics wait, preventing a wait from blocking the graphics queue needed to produce its own handoff signal. Metal commits a graphics release marker before returning update admission and encodes GPU event waits before frame encoders. These dependencies do not require an application idle wait before updating an already sampled texture.

`texture_upload_telemetry(context)` returns lock-free, context-wide, monotonically saturating totals: Rayon decode/transcode/mip time, bytes copied into texture staging, admission-to-native-submission latency, and native-submission-to-first-sample-ready handoff latency. Samples are diagnostic aggregates, not per-request timing guarantees.

## Runtime events

The existing runtime-event queue reports texture progress with the texture handle as `resource`:

- `Admission / Ok`: bounded CPU admission succeeded and the handle escaped.
- `Decode / Ok`: decoding, optional transcode, mip generation, and validation succeeded.
- `Upload / Ok`: the native transfer worker accepted the upload. This is not GPU completion.
- `Bind / Ok`: the transfer completion value was observed; this currently does not independently prove descriptor publication.
- `Decode / Cancelled`: cancellation succeeded before native transfer admission.
- `Upload / Cancelled`: cancellation succeeded after native transfer-worker admission.
- `Decode / <error>`: decoding, validation, or native admission failed terminally.

Poll events with `poll_runtime_event`; it also performs a texture-progress pass. Events are bounded diagnostics, not guaranteed completion callbacks. Use `poll_texture_load` for request state, subject to the initial-publication warning above.

## C ABI mapping

The C lifecycle is identical:

| Safe Rust | C ABI |
| --- | --- |
| `load_texture` | `ez_gfx_texture_load` |
| `poll_texture_load` | `ez_gfx_texture_poll` |
| `cancel_texture_load` | `ez_gfx_texture_cancel` |
| `texture_binding` | `ez_gfx_texture_get_binding` |
| `update_texture_region` | `ez_gfx_update_texture_region` |
| `texture_upload_telemetry` | `ez_gfx_texture_get_upload_telemetry` |
| `texture_residency` | `ez_gfx_texture_get_residency` |
| `set_texture_residency` | `ez_gfx_texture_set_residency` |
| `unload_texture` | `ez_gfx_texture_unload` |
| `wait_idle` | `ez_gfx_context_wait_idle` |

C pointer ranges are needed only for the duration of each call. `ez_gfx_texture_load` copies `data[0..data_size]`; neither the source bytes nor `EzGfxTextureDesc` must remain alive afterward. Output pointers are written only on success. The descriptor label is validated bounded UTF-8 metadata; it is not an asynchronous borrowed string.

Under the intended readiness contract, a typical render loop calls `ez_gfx_texture_poll`, retains the fallback on `NotReady`, and resolves/caches the binding on `Ok`; the publication warning above applies equally to C. Other polling results are terminal and should be logged before unloading the failed handle. On `QueueFull` from load, no valid output texture was produced; retry later.

## Verification and remaining evidence

The results below are previously recorded evidence, not rerun by this documentation audit. Current source support and known defects above must not be inferred away from passing narrower regressions.

RTX 3080 Vulkan/DX12 `texture_pixels` regressions sample BC1, BC3, BC7, and RGBA8 through the existing shader into a hidden 64×64 swapchain. Every pixel is checked: 2,048 left-half pixels change red→green and 2,048 blue pixels remain unchanged. Coverage includes updates after submitted draws, immediate update→draw without CPU polling, draw-record→update→submit, 32 unload-after-submit/reuse cycles per format, stale handles, admitted-upload unload, and final readback equality.

The Vulkan regression requires validation, captures and re-emits native output, and fails on any `VUID-` or `Validation Error`. It passes without validation errors after correcting paired ownership layouts and per-swapchain-image presentation semaphore reuse. DX12 passes the same pixel/order/lifetime checks. Native tests assert `IsWindowVisible == false`, absence of `WS_VISIBLE`, a non-foreground test window, and an exact client extent. No visible validation windows are required.

The examples smoke target includes a bounded three-frame hidden regression (32 tests). Hidden windows advance from the polling event loop without requesting visibility or activation. The shader-artifact target previously passed all five tests; its contract is unchanged.

Both RTX adapters advertised BC-only compression and explicitly rejected ASTC 4×4 with `Unsupported`; no ASTC hardware sampling is claimed. Metal GPU-event synchronization cross-checks for `aarch64-apple-darwin`, but Metal was not executed. Basis/KTX2 decoding is covered separately by residency tests, not by the custom-payload pixel regression. Compressed sRGB formats and partial edge blocks remain outside this pixel coverage.

P-015 remains incomplete in evidence: sustained coarse-to-fine rendering, deterministic unsignaled-fence unload/reuse, native Metal execution, and the dynamic-atlas transfer/frame-time benchmark remain in root `TODO.md`. Residency limits retain full allocations; no VRAM savings or streaming performance gains are claimed. Throwaway scenarios were removed; the self-contained native regression is retained, and immutable snapshots were not regenerated.

## Selected scope versus remaining work

- [P-014](../plan/P-014-basis-universal-and-compressed-textures.md#selected-solution): optional linkage, DDS/direct raw-block ingestion, source-aware native target selection, and the KTX2 target gaps above remain. Runtime GPU encoding of raw images into Basis is an explicit non-goal.
- [P-015](../plan/P-015-texture-streaming-and-partial-updates.md#selected-solution): coarse-first upload, region copies, residency views, and telemetry exist; initial descriptor publication needs reproduction and correction if confirmed. Full-chain allocation is selected: physical mip reclamation, sparse/virtual texturing, and stopping finer uploads on logical eviction are not completion requirements.
- [P-012](../plan/P-012-transfer-queue-staging-and-batching.md#selected-solution): staging reuse and independent completion streams exist; native cross-texture coalescing remains absent.
- [P-018](../plan/P-018-async-worker-and-task-architecture.md#selected-solution): bounded Rayon decode and dedicated transfer owners exist. Current polling/events differ from original completion callbacks; callback-dispatch parity/evidence must not be claimed. Neither Tokio nor the rejected custom-thread-pool alternative is required.

Remaining evidence includes Metal execution, ASTC and compressed-sRGB sampling, partial edge blocks, sustained coarse-first rendering, deterministic unsignaled-fence retirement/reuse, and the selected transcode, staging-reuse, multicore-load, and dynamic-atlas benchmarks. See [root TODO](../TODO.md). Implementation gaps, source-inferred defects, and unexecuted scenarios are distinct; none is closed by a format enum or a plan status claim.
