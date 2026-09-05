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
pub fn unload_texture(context: ContextHandle, texture: TextureHandle);
```

`load_texture` copies `bytes` and returns after bounded admission. `poll_texture_load` is the readiness source of truth: `NotReady` is transient, `Ok` permits binding, and every other result is terminal for that request. `texture_binding` returns the stable bindless texture-heap index, not a `PublicBinding`.

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
                // The requested texture can be selected in this same frame.
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

The ready-use point is the successful `texture_binding` call inside `poll`, before `begin_render`. The next `Push.texture_id` may select that index in the same frame. Do not add the texture to `PublicBinding`: that interface names structured, indirect, and render-target resources. The shader uses the index instead:

```slang
[BindlessTextureHeap(EZ_GFX_MAX_TEXTURES)]
ParameterBlock<TextureHeap> texture_heap;

TextureEntry entry = texture_heap.entries[push.texture_id];
float4 color = entry.texture.Sample(entry.sampler, uv);
```

## C ABI load and render flow

The C ABI uses the same ordering. This excerpt assumes `context`, `surface`, `png_bytes`, `png_size`, `fallback_binding`, `shader`, `indirect`, buffer `bindings`, `binding_count`, `dynamic_state`, and `current_mvp` are already valid:

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
            texture_ready = 1; /* Usable in the frame begun below. */
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

/* During early cleanup; context destruction also owns terminal cleanup. */
if (texture != 0) {
    ez_gfx_texture_unload(texture, context);
}
```

`ez_gfx_texture_load` writes `texture` only on success. The source bytes and descriptor need remain valid only through that call. `ez_gfx_texture_get_binding` writes its output only on success. As in Rust, C `EzGfxBinding` does not carry a sampled texture; pass the cached heap index through shader data such as push constants.

## Frame-graph readiness

Polling before binding is the public usage rule. Separately, once the owner thread has created the native texture and admitted its transfer, frame recording interns native-created textures with their texture-transfer completion tokens. The compiled graph lowers those tokens into transfer-to-graphics waits before sampled access. `frame_enqueue_readback` applies the same readiness token before transfer readback.

This is a submission-safety net, not an alternate readiness interface. A pending request has no public bindless index, so application code must not guess an index or select it in shader data. Continue selecting the fallback until `poll_texture_load` and `texture_binding` succeed. Recording graphics while another native-admitted texture is pending can still order GPU graphics behind that upload because the current graphics graph declares native-created textures as sampled resources; this does not block the CPU polling call.

## Lifecycle

1. Call `load_texture` on the context creator thread.
2. Retain the returned handle. The input bytes may be released immediately; `load_texture` copies them before returning.
3. Call `poll_texture_load` regularly, normally once per frame. Polling moves completed CPU work into native texture creation and transfer submission, then checks GPU completion without waiting.
4. Continue rendering with a fallback texture while polling returns `NotReady`.
5. After polling returns `Ok`, call `texture_binding` and use its stable bindless index.
6. Call `unload_texture` for early reclamation, or let `destroy_context` reclaim the texture.

All context and resource calls are creator-thread-affine. A texture handle belongs to one context and remains invalid in every other context. Cancellation or unload invalidates it. Slot reuse increments the generation, so stale handles remain invalid.

`load_texture` is non-waiting, not allocation-free: it validates arguments, copies the source payload, reserves a handle, and attempts bounded CPU admission. It never waits for decode, transcode, mip generation, transfer submission, or GPU completion.

## Progress and render integration

`poll_texture_load(context, texture)` returns:

- `Ok`: decode and the texture transfer timeline completed; the texture may be bound.
- `NotReady`: CPU work, owner-thread submission, or GPU completion remains pending.
- A terminal error: decoding, validation, native allocation, transfer, device, context, or handle validation failed.

Polling is part of progress. CPU workers publish decoded results to a bounded channel; the context creator drains that channel and creates/submits native textures. An application that neither polls nor calls another texture-progressing operation will not advance a decoded request into GPU submission.

`texture_binding` and `texture_residency` also make one nonblocking progress pass. Do not put a pending texture into frame-graph bindings: first observe `Ok` from `poll_texture_load`, then resolve its binding. The binding index is reserved with the handle and stays stable until unload.

`texture_residency` returns `(resident_mips, total_mips)` after CPU decode and native transfer admission. It returns `NotReady` while CPU work is pending. During GPU transfer it may report zero resident mips. The registry supports mip-count progression, but current Vulkan, DX12, and Metal texture uploads submit one complete mip chain and signal its final completion value; callers should not assume streaming one-mip-at-a-time residency.

`wait_idle(context)` is the explicit blocking alternative. It drains accepted CPU texture work, submits decoded results, flushes transfer owners, and waits for native completion. Use it for loading screens, tests, or shutdown—not a latency-sensitive render loop.

## Cancellation and cleanup

`cancel_texture_load` succeeds only while the request remains in the CPU/pending stage. Once the creator thread drains its decoded result and admits the native transfer job, cancellation is too late—even if the transfer owner has not submitted that job to the driver. Success marks the task cancelled, removes both registry and public handles, and emits a cancellation record. A worker already executing decoder code may finish internally, but its result is discarded and cannot revive the handle.

Cancellation after native transfer admission returns `InvalidArgument`. Use `unload_texture` instead. Unloading a pending request cancels it. Unloading a native texture waits for safe native reclamation before releasing its descriptor and storage. The C unload function is `void`, so C callers do not receive reclamation failures; terminal context destruction still performs dependency-ordered cleanup.

Destroying the context stops new CPU admission, drains or discards accepted work safely, joins worker ownership, waits for native work when initialized, and invalidates every child handle. Never retain a texture handle beyond its context.

## Backpressure

Admission is deliberately bounded:

- source payload: at most 64 MiB per request;
- CPU pool: at most 64 in-flight jobs and 64 MiB of aggregate admitted source bytes;
- decoded-result channel: 64 entries;
- each backend transfer-owner channel: 64 jobs.

The Rayon thread count is `max(1, available_parallelism - 1)`, leaving one logical CPU for the host/render thread. A full CPU budget makes `load_texture` return `QueueFull` immediately and no handle escapes. Retry in a later frame after polling existing requests. Do not busy-loop on `QueueFull`.

Transfer admission is also non-waiting. A full or failed native transfer worker turns the affected request into a terminal failure during polling; it does not block until queue space appears.

## Decode and mip behavior

Supported sources are raw RGB8, raw RGBA8, BMP, JPEG, PNG, TGA, and two-dimensional single-layer/single-face KTX2. KTX2 accepts RGBA8 payloads and BasisLZ ETC1S or UASTC payloads. Basis content is currently transcoded to RGBA8 on the CPU; compressed GPU destination formats are not yet exposed. The C descriptor therefore currently accepts only `EzGfxTextureDestinationFormat_Rgba8`.

Raw inputs require nonzero `width` and `height`, and their byte length must match exactly. For encoded inputs, zero descriptor dimensions accept decoded dimensions; nonzero values are assertions checked after decode. A nonzero `mip_count` similarly asserts the decoded/generated count.

When mip generation is requested, a one-level source is box-filtered down to 1×1 in the Rayon job. An existing valid mip chain is preserved. Decoded dimensions, every mip length, mip ordering, mip count, and aggregate RGBA8 size are validated before native allocation.

## Transfer and staging contract

The shared transfer module owns:

- a bounded dedicated transfer-owner thread;
- adaptive batches flushed at 32 MiB, 64 copies, or a 200 µs collection deadline;
- power-of-two staging buckets from 64 KiB through 64 MiB;
- best-fit reuse only after the associated completion token retires;
- idle trimming after 256 pool epochs;
- monotonic queue-local completion tokens.

Texture staging is backend-owned because row pitch, image layout, and native allocation rules differ. Each backend copies every decoded mip into one reusable host-visible bucket, records one native copy batch, returns the bucket to its pool with a retirement token, and frees only completed idle buckets. Geometry transfer uses the same policy and worker machinery but an independent queue timeline, preventing unrelated texture and geometry progress from sharing one global completion counter.

| Backend | Native transfer path | Completion and ownership |
| --- | --- | --- |
| Vulkan | Prefers a transfer-only queue family. It otherwise requests a secondary queue from the graphics family when available and falls back to the universal queue. Texture and geometry workers use separate queues when queue counts permit. | Timeline semaphores track queue-local completion. Separate-family image copies issue transfer release and graphics acquire barriers through a semaphore handoff before shader-read residency. Four command-resource slots are reused only after retirement. |
| Direct3D 12 | A COPY queue records staged buffer/texture copies. A DIRECT queue waits on the copy fence and records state transitions required for graphics use. Texture and geometry have independent copy workers. | Native fences serialize copy-to-direct ownership and expose completion. Four allocator/list/event slots are recycled after their fence values complete. |
| Metal | A dedicated `MTLCommandQueue` creates blit command buffers and `MTLBlitCommandEncoder` copies. Submission ownership moves to the transfer thread. | Ordered command-buffer values and terminal command status provide completion. Metal resource ownership is implicit; staging remains retained until its command completes. |

The fallback paths preserve correctness on adapters without a dedicated transfer family or enough independent queues; they reduce overlap, not semantics.

## Runtime events

The existing runtime-event queue reports texture progress with the texture handle as `resource`:

- `Admission / Ok`: bounded CPU admission succeeded and the handle escaped.
- `Decode / Ok`: decoding, optional transcode, mip generation, and validation succeeded.
- `Upload / Ok`: the native transfer worker accepted the upload. This is not GPU completion.
- `Bind / Ok`: the transfer completion value was observed and binding became ready.
- `Decode / Cancelled`: cancellation succeeded before native transfer admission.
- `Decode / <error>`: decode or request validation failed.

Poll events with `poll_runtime_event`; it also performs a texture-progress pass. Events are bounded diagnostics, not the source of truth for readiness. Use `poll_texture_load` for the request state.

## C ABI mapping

The C lifecycle is identical:

| Safe Rust | C ABI |
| --- | --- |
| `load_texture` | `ez_gfx_texture_load` |
| `poll_texture_load` | `ez_gfx_texture_poll` |
| `cancel_texture_load` | `ez_gfx_texture_cancel` |
| `texture_binding` | `ez_gfx_texture_get_binding` |
| `texture_residency` | `ez_gfx_texture_get_residency` |
| `unload_texture` | `ez_gfx_texture_unload` |
| `wait_idle` | `ez_gfx_context_wait_idle` |

C pointer ranges are needed only for the duration of each call. `ez_gfx_texture_load` copies `data[0..data_size]`; neither the source bytes nor `EzGfxTextureDesc` must remain alive afterward. Output pointers are written only on success. The descriptor label is validated bounded UTF-8 metadata; it is not an asynchronous borrowed string.

A typical render loop calls `ez_gfx_texture_poll`. On `EzGfxResult_NotReady`, it keeps the fallback binding. On `EzGfxResult_Ok`, it calls `ez_gfx_texture_get_binding` once and stores that index. Any other result is terminal for that request and should be logged before unloading the failed handle. On `EzGfxResult_QueueFull` from load, no valid output texture was produced; retry later.
