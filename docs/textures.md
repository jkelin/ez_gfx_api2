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

`load_texture` copies `bytes` and returns after bounded admission. `NotReady` is transient, `Ok` permits binding, and every other polling result is terminal for that request. `texture_binding` returns the stable bindless texture-heap index, not a `PublicBinding`.

Initial readiness requires both completed coarse transfers and a published, frame-safe descriptor. Completed GPU copies alone do not make polling, binding, or `Bind / Ok` succeed. Finer uploads may remain outstanding after initial readiness.

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
                // The published coarse view can be sampled in this frame.
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

The ready-use point is the successful `texture_binding` call inside `poll`, before `begin_render`. Do not add the texture to `PublicBinding`: that interface names structured, indirect, and render-target resources. The shader uses the index instead:

```slang
[BindlessTextureHeap(EZ_GFX_MAX_TEXTURES)]
ParameterBlock<TextureHeap> texture_heap;

TextureEntry entry = texture_heap.entries[push.texture_id];
float4 color = entry.texture.Sample(entry.sampler, uv);
```

## C ABI load and render flow

The C ABI uses the same readiness and ordering contract. This excerpt assumes `context`, `surface`, `png_bytes`, `png_size`, `fallback_binding`, `shader`, `indirect`, buffer `bindings`, `binding_count`, `dynamic_state`, and `current_mvp` are already valid:

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
            texture_ready = 1; /* The coarse descriptor is published. */
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

Both context creation descriptors (`EzGfxContextDesc`, `EzGfxBackendContextDesc`) carry `texture_decode_workers` as their trailing field (ABI 24): zero selects the default decode topology, a nonzero value pins exactly that many worker threads. Earlier fields keep their offsets. Always `ez_gfx_abi_version()`-gate callers against `EZ_GFX_ABI_VERSION` before touching the extended layout.

`ez_gfx_texture_load` writes `texture` only on success. The source bytes and descriptor need remain valid only through that call. `ez_gfx_update_texture_region` likewise copies validated region bytes before returning; compressed offsets and non-edge extents must align to the format’s 4×4 blocks. `ez_gfx_texture_get_binding` and telemetry queries write output only on success. As in Rust, C `EzGfxBinding` does not carry a sampled texture; pass the cached heap index through shader data such as push constants.

## Frame-graph readiness

Polling before binding is the public usage rule. Frame recording declares only published textures as sampled resources and attaches completion tokens for their exposed mip ranges. Submission refreshes dependencies, including updates accepted after recording. Hidden fine-mip writes do not replace a coarse view's readiness token. `frame_enqueue_readback` uses the corresponding texture dependency.

A pending request has no public bindless index: do not guess an index or select it in shader data. Continue selecting the fallback until polling and binding succeed. Unpublished uploads do not add unrelated sampling waits. Bindless draws conservatively declare all published textures, so an update to another published texture can still add a dependency.

GPU waits alone do not publish descriptors; readiness and frame-safe publication are checked separately.

## Lifecycle

1. Call `load_texture` on the context creator thread.
2. Retain the returned handle. The input bytes may be released immediately; `load_texture` copies them before returning.
3. Call `poll_texture_load` regularly, normally once per frame. Polling moves completed CPU work into native texture creation and transfer submission, then checks GPU completion without waiting.
4. Continue rendering with a fallback texture while polling returns `NotReady`.
5. After polling returns `Ok`, call `texture_binding` and use its stable bindless index.
6. Call `unload_texture` for early reclamation, or let `destroy_context` reclaim the texture.

All context and resource calls are creator-thread-affine. A texture handle belongs to one context and remains invalid in every other context. Successful cancellation or unload invalidates it immediately. Native storage and its descriptor/view retire only after accepted transfer and submitted graphics dependencies complete; a retired binding slot cannot be reused early. Slot reuse increments the generation, so stale handles remain invalid.

`load_texture` is non-waiting, not allocation-free: it validates arguments, copies the source payload, reserves a handle, and attempts bounded CPU admission. It never waits for decode, transcode, mip generation, transfer submission, or GPU completion.

## Progress and render integration

`poll_texture_load(context, texture)` returns:

- `Ok`: the exposed coarse view is transfer-complete and its descriptor is published; finer mips may still stream.
- `NotReady`: CPU work, owner-thread admission, exposed-range GPU completion, or frame-safe initial publication remains pending.
- A terminal error: decoding, validation, native allocation, transfer, device, context, or handle validation failed.

Polling is part of progress. CPU workers publish decoded results to a bounded channel; the context creator drains that channel and creates/submits native textures. An application that neither polls nor calls another texture-progressing operation will not advance a decoded request into GPU submission.

`texture_binding`, `texture_residency`, and `set_texture_residency` also make one nonblocking progress pass. Do not put a pending texture into frame-graph bindings: first observe `Ok` from `poll_texture_load`, then resolve its binding. The binding index is reserved with the handle and stays stable until unload.

Each polling pass first reaps completed frame slots without blocking: Vulkan checks fence status and Metal checks settled commands before consulting the descriptor gate, while DX12 already consults the live graphics fence on every check. Finished GPU work therefore unblocks publication during sustained rendering without a `wait_idle` round-trip. A transient publication refusal from the graphics gate (notably DX12's fence check racing an independently advanced fence) stays retryable and surfaces `NotReady`; allocation, validation, capability, and device errors stay terminal.

`texture_residency` returns the currently exposed contiguous coarse mip count and immutable total after CPU decode and native transfer admission. It returns `NotReady` while CPU work is pending. `set_texture_residency` accepts `1..=total`: decreasing the count logically evicts finer levels without discarding their allocation, while increasing it publishes only transfer-complete levels and returns `NotReady` until the requested range and a frame-safe descriptor rewrite are available. Each backend submits decoded levels from the coarsest mip toward level zero under distinct completion values, so no sampled view exposes an uninitialized fine level.

Residency is a sampled-view limit, not a GPU-memory budget: Vulkan replaces an image view, DX12 rewrites an SRV, and Metal replaces a view of the retained parent texture. All three retain the full mip-chain allocation and uploaded bytes. Decreasing residency neither frees image storage nor cancels finer uploads; increasing it reuses completed data. Storage retires on unload/cancellation after outstanding GPU uses, not on logical mip eviction. This matches P-015's selected full-chain-allocation design; sparse allocation and physical mip reclamation are not implemented.

`wait_idle(context)` is the explicit blocking alternative. It drains accepted CPU texture work, submits decoded results, flushes transfer owners, and waits for native completion. Use it for loading screens, tests, or shutdown—not a latency-sensitive render loop.

### Direct readback contract

`frame_enqueue_readback` captures the full stored image as RGBA8 through the frame graph; sampling through shaders is a separate path with its own filtering. The boundary rejects block-compressed textures with `InvalidArgument` on every backend (compressed blocks have no RGBA texel grid to capture), reports `NotReady` for pending or logically demoted textures (a demoted view exposes a smaller extent than the request), and records the transfer dependency covering the exposed range. Vulkan texture images carry transfer-source usage with a graph entry barrier and a matching shader-read restore; Metal clamps the blit to the published view's real level-zero extent.

## Cancellation and cleanup

`cancel_texture_load` succeeds in the CPU stage and after native transfer-worker admission while the first sample-ready completion remains pending. It atomically invalidates the public handle, marks uncommitted native jobs to be skipped, and emits a phase-specific cancellation record. A CPU decoder already executing may finish internally, and a native job already committed to the driver may complete, but neither can revive the handle; submitted resources remain deferred until transfer and graphics dependencies retire. Cancellation after the load is sample-ready returns `InvalidArgument`; use `unload_texture` instead.

Unloading a pending request cancels it. Unloading a native texture immediately invalidates the handle while deferring descriptor/view and storage reclamation until its last accepted transfer and all submitted graphics frames retire. The C unload function is `void`, so C callers do not receive reclamation failures; terminal context destruction performs dependency-ordered cleanup.

Destroying the context stops CPU admission, joins workers, waits for native work, and invalidates every child handle. Worker failure still runs native shutdown and drains actual submitted commands, not an unreachable final token. If a live device cannot establish drain completion, destruction returns the existing typed failure and deliberately retains the failed context's GPU-owned state rather than freeing potentially live resources or aborting the host. This bounded-per-failed-context safety retention is not normal reclamation; explicit device loss follows native release guarantees.

## Backpressure

Admission is deliberately bounded:

- source payload: at most 64 MiB per request;
- CPU pool: at most 64 in-flight jobs and 64 MiB of aggregate admitted source bytes;
- decoded-result channel: 64 entries;
- each backend transfer owner: 64 queued copy jobs, counting every member of an atomically admitted mip bundle.

The Rayon thread count defaults to `max(1, available_parallelism - 1)`, leaving one logical CPU for the host/render thread. `ContextOptions::texture_decode_workers` (and its `with_texture_decode_workers` builder) overrides the count: zero keeps the default topology and a nonzero value pins exactly that many worker threads. A full CPU budget makes `load_texture` return `QueueFull` immediately and no handle escapes. Retry in a later frame after polling existing requests. Do not busy-loop on `QueueFull`.

Transfer admission is also non-waiting. A full or failed worker makes the affected request fail terminally during polling. Rejected work does not advance completion high-water marks, so later idle waits cannot target work never admitted.

## Decode, transcode, and mip behavior

### Source and destination support

These are source-level capabilities, not a claim that every cell has native pixel evidence. [Decoder dispatch and KTX2 admission](../crates/ez-gfx-runtime/src/texture.rs), [DDS](../crates/ez-gfx-runtime/src/texture/dds.rs), and [raw mip admission](../crates/ez-gfx-runtime/src/texture/raw.rs) define the validated loading contract.

| Input | Current destination behavior | Limits |
| --- | --- | --- |
| Raw RGB8/RGBA8; BMP, JPEG, PNG, TGA | RGBA8 linear or sRGB; `Auto` is linear RGBA8 | RGB expands to RGBA; encoded images convert to RGBA8. No runtime BC/ASTC encoding. |
| Direct native KTX2 | Preserves RGBA8, BC1, BC3, BC7, or ASTC 4×4, linear/sRGB | Requires no supercompression and an admitted compression family. An explicit destination must match exactly: no conversion or decompression fallback. |
| KTX2 UASTC/ETC1S | Explicit RGBA8/BC1/BC3/BC7/ASTC 4×4, linear/sRGB; source-aware `Auto` | Requires `ktx2` + `basis`; accepts unsupercompressed UASTC or BasisLZ ETC1S. R/Rg restrictions below. |
| Standalone `.basis` | Same explicit targets and source-aware `Auto` | Requires `basis`; header sRGB is preserved by `Auto`. An unset sRGB flag means linear. |
| DDS | Preserves DXT1/BC1, DXT5/BC3, or DX10 RGBA8/BC1/BC3/BC7, including DX10 sRGB | Single 2D image; no transcoding. Legacy DXT3/BC2 and legacy BGR/BGRA conversions are not supported. |
| Raw native mip chain | Preserves RGBA8/BC1/BC3/BC7/ASTC 4×4, linear/sRGB | Explicit format, nonzero dimensions/count; contiguous tightly packed levels, largest first, exact total bytes. No private container header. |
| Custom source `128..=255` | Application returns validated mips in a supported native format | Explicit destinations must match callback output; compressed output requires adapter admission. |

`TextureDestination` exposes only `Auto` and linear/sRGB RGBA8, BC1, BC3, BC7, and ASTC 4×4. BC2/BC4/BC5/BC6H, other ASTC block shapes, HDR/float texture destinations, arrays, cubemaps, 3D textures, and KTX2 Zstd/Zlib supercompression are **evaluated exclusions of this contract**, not required extensions. These restrictions concern loaded textures, not separate render-target formats.

The [shared target policy](../crates/ez-gfx-runtime/src/texture/basis.rs) chooses ETC1S BC1 for opaque color or BC3 for alpha, and UASTC BC7 on BC-capable adapters; ASTC-only adapters use ASTC 4×4, otherwise RGBA8. `Auto` preserves source transfer metadata; explicit linear/sRGB requests override the label without gamma-converting the encoded values. Both universal containers pass an explicit target to the native transcoder, including BC1/BC3.

**R/Rg restriction:** KTX2 R/Rg `Auto` falls back to RGBA8 with source transfer metadata. Output samples are `(R,0,0,1)` or `(R,G,0,1)`, including second-channel data carried in an ETC1S/UASTC alpha plane. Explicit RGBA8 destinations work; explicit compressed R/Rg requests return `Unsupported` rather than exposing incorrect channels. No BC4/BC5 storage, backend view swizzle, or runtime re-encoding is added.

Input and aggregate decoded mip bytes are each limited to 64 MiB. DDS, raw, and direct KTX2 validate physical mip count, per-level geometry/size, and total output budget before copying mip payloads. DDS/raw additionally require an exact payload total; DDS accepts tightly packed native pitch or matching linear-size metadata, not padded row conversion. DDS unknown/straight/opaque alpha modes preserve bytes; premultiplied/custom modes are unsupported and reserved bits invalid. Universal paths bound output before native decompression/transcoding; malformed metadata fails closed.

### Sampler contract

The safe entry point validates samplers exactly like the C boundary: `max_anisotropy` must be finite and within `1.0..=16.0`, otherwise `load_texture` returns `InvalidArgument` before admission. Native lowering then differs only inside that range: Vulkan enables anisotropy above `1.0` and derives the mipmap mode from `min_filter`; DX12 selects anisotropic filtering above `1.0` and floors the level; Metal maps `min_filter` onto `mipFilter` (`Nearest`/`Linear`). The mapping matters because the Apple descriptor default is `notMipmapped`: without it every minified fragment would sample view level zero regardless of the published chain.

### Features and backend admission

Decoder features are opt-in on `ez-gfx`, `ez-gfx-ffi`, and `ez-gfx-runtime`:

| Features | KTX2 direct data | KTX2 universal data | Standalone Basis | Native Basis linkage |
| --- | --- | --- | --- | --- |
| Default / none | `Unsupported` | `Unsupported` | `Unsupported` | None |
| `ktx2` | Supported | `Unsupported` | `Unsupported` | None |
| `basis` | `Unsupported` | `Unsupported` | Supported | Included |
| `ktx2,basis` | Supported | Supported | Supported | Included |

**Build change:** KTX2 no longer works implicitly in default builds. Add `features = ["ktx2", "basis"]` to the safe-crate dependency, or build FFI with `cargo build -p ez-gfx-ffi --features ktx2,basis`. The examples opt in explicitly. Source IDs remain present in disabled builds and report `Unsupported`. DDS/raw/basic raster decoding needs neither feature. [Runtime dependencies](../crates/ez-gfx-runtime/Cargo.toml) keep native transcoding optional; normal dependency graphs remain shader-compiler-free, and runtime features do not enable the Basis encoder.

| Backend | Runtime compression admission | Qualification |
| --- | --- | --- |
| Vulkan | Union of advertised BC and ASTC-LDR device features | [Device probe](../crates/ez-gfx-backend-vulkan/src/device.rs#L835-L845); actual adapter-dependent support, not universal ASTC availability. |
| Direct3D 12 | BC only | BC base width and height must be multiples of four; unsupported base extents return `Unsupported` through native, safe, and C interfaces. Lower mips retain their logical edge dimensions. No ASTC mapping. |
| Metal | Union of independently queried BC and ASTC support | [Device probe](../crates/ez-gfx-backend-metal/src/native/device.rs) checks `supportsBCTextureCompression` and Apple-family ASTC support independently. An adapter may advertise both. See native proof below. |

RGBA8 is the common uncompressed target. Explicit compressed destinations require the admitted family but still face the source-specific restrictions above. Native mappings alone do not establish execution evidence.

For example, a 7×3 BC base is supported by the tested Vulkan adapter but rejected by DX12. A valid 28×12 DX12 BC base contains a supported 7×3 mip at level two, including 4×3 and 3×3 edge updates. Only the staging footprint rounds up to physical blocks; native logical dimensions and shader UVs are unchanged. See [Microsoft's BC resource and mip rules](https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression).

Application source codes `128..=255` use a registered `TextureDecodeCallback`. Rust callbacks are `Send + Sync + 'static` and may run concurrently. Registration snapshots an `Arc` into each accepted job, so unregistering prevents new admission without invalidating queued work. C registers paired decode/release callbacks plus `user_data`: successful output pointers remain valid until ez-gfx copies all mips and invokes release exactly once. The application must retain callback code and `user_data` until unregister and all accepted requests complete.

Raw RGB8/RGBA8 require nonzero `width` and `height`, with exact byte length. Native `Raw` additionally requires a nonzero `mip_count` and concrete source format. For encoded inputs, zero descriptor dimensions accept decoded dimensions; nonzero values are assertions checked after decode. A nonzero descriptor `mip_count` similarly asserts the decoded/generated count. Set `generate_mips = false` when supplying a single compressed level without an existing chain.

When mip generation is requested, a one-level RGBA8 source is box-filtered down to 1×1 in the Rayon job. Each output texel covers its proportional source extent with integer sectioning, so trailing odd rows and columns contribute instead of being dropped. sRGB channels decode to linear light, average, and re-encode (linear 0.5 encodes to 188); alpha stays a straight linear mean. Any existing valid multi-level chain is preserved, even if it stops before 1×1. A one-level compressed source with generation requested is rejected; an existing compressed chain passes through. The full generated aggregate is validated against the 64 MiB budget before any level allocation. Dimensions, block geometry, every mip length, ordering, count, and aggregate size are validated before native allocation. See [generation and validation](../crates/ez-gfx-runtime/src/texture.rs).

### Original implementation parity

**Chosen migration deltas:** the [Odin manager](../reference/ez_gfx_api/src/texture_manager.odin#L55-L118) allowed replacing/clearing built-in BMP–KTX2 decoder callbacks; current registration deliberately uses custom IDs only, with encoded-container availability controlled by features. Original source regions remained borrowed until the [texture-loaded callback](../reference/ez_gfx_api/src/texture_manager.odin#L1047-L1050); current loading copies source bytes immediately and uses polling/bounded diagnostic events. A per-load completion callback is not part of the selected current API; diagnostic events do not promise lossless callback delivery.

Original [KTX2 decoding](../reference/ez_gfx_api/src/texture_manager.odin#L815-L841) expanded Basis content to RGBA32 and extracted level zero. Preserved source mip chains, native compressed storage, standalone Basis, DDS/raw ingestion, progressive residency, and partial updates are additions, not original parity regressions.

Original [mip generation](../reference/ez_gfx_api/src/texture_manager.odin#L1401-L1476) used Vulkan GPU blits with linear filtering. CPU RGBA8 box filtering is the chosen portable migration policy, not a missing GPU-blit implementation; identical filtering results are not guaranteed.

## Transfer and staging contract

The shared transfer module owns:

- a bounded dedicated transfer-owner thread;
- adaptive batches flushed at 32 MiB, 64 copies, or a 200 µs collection deadline;
- power-of-two staging buckets from 64 KiB through 64 MiB;
- best-fit reuse only after the associated completion token retires;
- idle trimming after 256 pool epochs;
- monotonic queue-local completion tokens.

Texture staging is backend-owned because row pitch, block layout, image layout, and native allocation rules differ. Each backend copies decoded mip or update bytes into reusable host-visible buckets and retains buckets until completion. Geometry transfer uses the same policy and worker machinery but an independent queue timeline.

Initial mip chains enter atomically and preserve FIFO order. Adjacent copies at the same coarse-to-fine stage can share one native submission across textures; batches never reorder stages or combine successive writes to the same image unsafely. Each mip retains a real completion value, even when one native signal completes several same-stage copies. Vulkan and DX12 record shared copy batches; Metal encodes shared blit command buffers instead of committing one prebuilt buffer per texture.

| Backend | Native transfer path | Completion and ownership |
| --- | --- | --- |
| Vulkan | Dedicated transfer family where available; otherwise secondary graphics-family queue or universal queue. Texture and geometry use separate queues when possible. | Timeline semaphores order graphics release, copy, and acquire. Fine-copy completion waits run on the transfer owner before enqueueing graphics acquire, allowing an already-ready coarse frame to finish. Exact paired layouts, four retiring command slots, per-image presentation semaphores. |
| Direct3D 12 | Independent texture/geometry COPY workers; DIRECT queue performs graphics transitions. | The owner waits for copy completion before enqueueing DIRECT acquire. Coarse frames need not wait for later fine copies. Four allocator/list/event slots retire by actual completion. |
| Metal | Independent geometry/texture queues, shared texture blit command buffers, independent BC/ASTC queries. | GPU events order prior reads and later sampling. Texture owners check committed completion/status; cancellation-only batches commit a real signal. Geometry workers track ordered admission, and graphics waits only for the referenced accepted buffer command. |

The fallback paths preserve correctness on adapters without a dedicated transfer family or enough independent queues; they reduce overlap, not semantics.

Before graphics waits, all backends flush each referenced transfer owner only through the required accepted token. An already-submitted token returns without draining later stages. This is not a CPU-nonblocking guarantee: native callbacks may wait for the required GPU copy/command before their handoff is safe. Metal also waits for the referenced buffer command on the calling thread before graphics submission. These correctness-first waits differ from P-012's original no-per-submission-blocking goal; later fine copies do not block already-ready coarse frames. Measured wall costs appear below.

## Partial updates and telemetry

`update_texture_region(context, texture, region)` validates a resident texture, mip level, bounds, format-specific byte length, and compressed-block geometry before copying the caller’s bytes. Admission is nonblocking and bounded. Uncompressed rows are tightly packed. BC and ASTC offsets must be block-aligned; width and height must be multiples of four unless the rectangle reaches that mip’s right or bottom edge. Accepted updates enter the same backend transfer owner and graphics handoff as initial uploads.

Updates accepted before submission are included even when draws were recorded earlier. Targeted worker flushes precede graphics event/fence waits on all three backends, including buffer transfers; a failed producer is rejected before an unsignaled wait can poison future frames. Updates outside the exposed mip range preserve coarse sampling readiness. Applications need not call idle before updating a sampled texture.

`texture_upload_telemetry(context)` returns lock-free, context-wide, monotonically saturating totals: Rayon decode/transcode/mip time, bytes copied into texture staging, admission-to-native-submission latency, and native-submission-to-first-sample-ready handoff latency. Samples are diagnostic aggregates, not per-request timing guarantees.

## Runtime events

The existing runtime-event queue reports texture progress with the texture handle as `resource`:

- `Admission / Ok`: bounded CPU admission succeeded and the handle escaped.
- `Decode / Ok`: decoding, optional transcode, mip generation, and validation succeeded.
- `Upload / Ok`: the native transfer worker accepted the upload. This is not GPU completion.
- `Bind / Ok`: the sampled range's transfer completion was observed and a nonempty descriptor view is published.
- `Decode / Cancelled`: cancellation succeeded before native transfer admission.
- `Upload / Cancelled`: cancellation succeeded after native transfer-worker admission.
- `Decode / <error>`: decoding, validation, or native admission failed terminally.

Poll events with `poll_runtime_event`; it also performs a texture-progress pass. Events are bounded diagnostics, not guaranteed completion callbacks. Use `poll_texture_load` for request state.

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

ABI **23** adds source codes `8 = Dds` and `9 = Raw` without adding exports or changing descriptor layout. Regenerate/rebuild bindings against [the header](../include/ez_gfx_api.h) and [XML declarations](../bindings/bindings.xml); existing source numbers remain stable. `Raw` uses `destination_format` as its concrete native source/storage format (`Auto` is invalid), plus nonzero `width`, `height`, and `mip_count`. The existing load function receives the complete contiguous mip byte span and copies it before returning.

A typical C render loop calls `ez_gfx_texture_poll`, retains the fallback on `NotReady`, and resolves/caches the binding on `Ok`. Other polling results are terminal and should be logged before unloading the failed handle. On `QueueFull` from load, no valid output texture was produced; retry later.

## Verification and remaining evidence

Native proof uses RTX 3080 Vulkan/DX12 and Apple M2 Pro Metal; hosted CI compilation remains separate from hardware execution.

- This change, RTX 3080 Vulkan/DX12 and Apple M2 Pro Metal: `texture_pixels` gains direct-capture probes. RGBA8 graph readback returns stored texels byte-for-byte on Vulkan/DX12; compressed captures return `InvalidArgument` identically on all three backends; demoted captures return `NotReady` (compressed-terminal precedence pinned on DX12); a DX12 destroy-driven fence scenario stays retryable; Vulkan and Metal reap probes unblock publication by polling alone with no `wait_idle`. Metal gains a 4:1 minification probe (256px white/black chain on the 64px surface): it fails pre-fix with mean 255 and passes post-fix, plus demote/promote direct round-trip bytes.
- This change also fixed a real Vulkan defect the probes exposed: texture images lacked transfer-source usage and the graph readback left no shader-read restore, producing `VUID-VkImageMemoryBarrier`/`vkCmdCopyImageToBuffer`/`vkCmdDraw` validation errors. Both are fixed; the Vulkan parent run is validation-clean. Counts re-verified here: Vulkan native lib 16/16, `ez-gfx` lib 21/21, Metal lib 16/16, Apple `texture_pixels` 2/2.
- `texture_pixels`: BC1/BC3/BC7 linear and sRGB plus RGBA8 sampled into hidden 64×64 surfaces, every pixel checked. Covers changed/untouched regions, queued and record-before-update ordering, sustained residency changes, stale handles, cancellation, and submitted-work unload/reuse. Vulkan tests 7×3 bases; DX12 tests explicit safe/C rejection of those bases and real 7×3 mip-two edges from 28×12 bases.
- Deterministic native regressions: Vulkan six and DX12 five GPU scenarios prove coarse sampling while a submitted fine copy is GPU-blocked, native cross-texture batching, unsignaled graphics-fence retention, rejected full queues without hanging idle, and drain-safe partial submission failure. The DX12 admission unit adds base/mip geometry coverage.
- First-publication regression was reproduced before the fix through Vulkan public polling and a deterministic safe-state test. Completed copies no longer produce premature readiness.
- Final behavioral checks: HAL transfer 11, safe state 15, FFI residency two, and all 32 hidden example smoke tests passed. The expanded pixel matrix passed both backends; Vulkan captures native output and fails on `VUID-` or `Validation Error`. Hidden window helpers assert invisible/non-foreground windows and exact client extents. Shader-artifact contracts were unchanged and were not rerun.
- Hosted CI executes device-independent backend contracts and compiles texture GPU regressions separately where required hardware is unavailable; this is not native runtime proof. Run the unfiltered backend library suites on admitted GPU hardware.
- Both RTX backends explicitly reject ASTC with `Unsupported`; no ASTC sampling on those adapters is claimed.
- Apple M2 Pro, macOS 15.7.7, Rust 1.95, Xcode 26.3 and Slang 2025.23.2: Metal library tests 16, native compute two, FFI render/readback three, and the expanded pixel matrix pass headlessly using unattached `CAMetalLayer` objects. No window is created or activated.
- Metal pixels cover BC1/BC3/BC7 and ASTC 4×4, linear/sRGB, at 8×4 and 7×3 bases; clipped 7×3 mips in 28×12 chains; changed/untouched regions; promotion/demotion; public `NotReady` before frame retirement and successful publication afterward; and submitted-work unload/reuse. Seven native lifecycle regressions prove GPU-gated coarse/fine overlap, hidden/visible updates, retirement, rejected admission, partial failure, and accepted buffer waits. Existing transfer tests prove shared native batches and cancellation-only completion signals.
- Real ETC1S/UASTC `.basis` and KTX2 fixtures are transcoded and sampled as BC7/ASTC linear/sRGB, compared with independently RGBA-decoded images rendered through the same path. Raw native ingestion covers all eight compressed formats plus RGBA8; DDS DXT1 reaches sampled readback. This is GPU evidence, not target-enum inspection.
- Apple runtime texture/transcode tests pass 32+6 with combined features; ABI/streaming tests pass 13+3. The built feature-enabled dylib passes 50-export parity and actual Mach-O import auditing with zero forbidden compiler imports.
- Hardware execution exposed and fixed two Metal defects: pending admitted buffer copies were rejected instead of ordered before graphics, and linear BGRA surfaces contradicted the graph's BGRA8 sRGB contract. Layer creation/acquisition and pipeline attachment formats now agree. Exact midtone regressions verify linear 0.1 clears encode as 89, not 26, and BC7 low channels encode as 13, not 1; alpha is unchanged. Existing Metal output therefore becomes correctly sRGB-encoded.

### Measured workloads

Throwaway debug-profile scenarios were executed and removed. These are local wall-time observations, not GPU timestamps, general throughput guarantees, or full-scene benchmarks.

| Native upload workload: 64 independent 64×64 RGBA8 images, 1 MiB total | Forced serial completion | Queued uploads |
| --- | --- | --- |
| Vulkan, three runs | 64 native batches; 102.985–109.079 ms; one retained staging bucket | Two batches; 16.185–17.752 ms; 64 retained buckets |
| DX12, three runs | 64 native batches; 352.872–644.082 ms; one retained bucket | Five batches; 7.973–14.720 ms; 29 retained buckets |

Native queue signals/submission observations counted actual batches; sampled readback matched uploaded bytes. Serial mode waits after every image, whereas queued mode waits after all admissions. Different waiting/allocation costs and retained memory prevent treating this comparison as a frame-rate prediction.

The selected 2048×2048 RGBA8 atlas workload used 64×64 glyph updates, eight warmup frames and 64 measured hidden render frames. Both modes produced identical final shader readback. Dirty regions staged 1,048,576 bytes versus 1,073,741,824 for full-image updates: **1024× less transferred data**.

| Backend | Dirty-region wall µs/frame | Full-image wall µs/frame |
| --- | ---: | ---: |
| Vulkan | 6,954.561 | 6,730.919 |
| DX12 | 14,845.913 | 15,538.956 |

Presentation, CPU work, and completion waits are included. No Vulkan frame-time improvement was observed, and neither result isolates GPU transfer time.

## Selected scope versus remaining work

- [P-014](../plan/P-014-basis-universal-and-compressed-textures.md#implementation-evidence): optional linkage, DDS/raw ingestion, source-aware target selection, and explicit BC1/BC3 transcoding are implemented and decoder-tested. R/Rg compressed destinations and wider containers/formats remain explicit restrictions. Runtime GPU encoding of raw images into Basis is an explicit non-goal.
- [P-015](../plan/P-015-texture-streaming-and-partial-updates.md#implementation-and-evidence-status): published coarse readiness, partial mip-edge updates, logical residency, and telemetry are implemented with Vulkan/DX12/Metal GPU evidence. Polling reaps completed frames before the descriptor gate (Vulkan/Metal), DX12 fence-gate refusals stay retryable, direct RGBA8 readback requires full residency and rejects compressed captures, and Metal mip filtering follows `min_filter`. Full-chain allocation is selected: physical mip reclamation, sparse/virtual texturing, and cancelling finer uploads on logical eviction are not completion requirements.
- [P-012](../plan/P-012-transfer-queue-staging-and-batching.md#selected-solution): pooled staging, independent streams, and native cross-texture batching are implemented and GPU-tested. Removing per-batch GPU completion waits remains an open scheduling requirement in root TODO; targeted waits do not erase that goal. The new frame-slot reap only observes completion, it does not remove any wait.
- [P-018](../plan/P-018-async-worker-and-task-architecture.md#selected-solution): bounded Rayon decode and dedicated transfer owners exist. The decode worker count is configurable via `ContextOptions::texture_decode_workers` (zero keeps the default topology). Polling/events replace original completion callbacks by design; callback-dispatch parity is not claimed. Neither Tokio nor the rejected custom-thread-pool alternative is required.

Remaining evidence includes broader transcode/multicore/full-scene performance and Metal atlas/staging benchmarks; Windows timings are not extrapolated to Apple. Native evidence is specific to the tested adapters and scenarios, not every Metal device or full-plan completion. See [root TODO](../TODO.md).

Decoder proof is separate: default/no-default, `ktx2`-only, `basis`-only, and combined-feature regressions cover admission, explicit destinations, real ETC1S/UASTC R/Rg pixels, standalone/KTX2 sRGB metadata, and malformed bounds. Warm real-image decode measurements and their limitations are recorded in [P-014 implementation evidence](../plan/P-014-basis-universal-and-compressed-textures.md#implementation-evidence); they do not establish GPU upload or full-scene performance.
