# Textures

Texture loading is asynchronous. Each context/device owns one 1×1 opaque-magenta sampled texture. Every safe `Texture` is a typed view into the context-owned bindless heap; dropping the wrapper does not unload or recycle its texture. A manager-owned decode driver (`DecodeDriver`) performs decode/mip work on a lazily built Rayon pool and backend transfer owners submit device copies.

## Public path

`Context::load_texture` validates the request, copies caller source, reserves a generational lease, and publishes that stable slot as an alias of the shared fallback before returning. `TextureConfig` owns the texture source, the generate-mips option, and the minimum required coarse-prefix mip count as a plain `u32` (zero means optional with no frame CPU wait, `REQUIRED_MIPS_FULL` resolves to the decoded total at submission). Device initialization backfills loads queued before a frame-capable device existed. Every alias uses the fallback's one context-owned sampler; a uniform 1x1 texel is invariant under filtering, addressing, and anisotropy. Real publication installs the request's sampler with its texture. The manager then admits decode and native transfer waves under the shared transfer budget; decode, validation, and scheduling policy live in `ez-gfx-texture-manager` (see its README), linked to the selected backend through its `TextureBackendContext` traits.

Binding queries preserve terminal status precedence: failed textures return their typed failure and lost contexts return `DeviceLost`; the fallback never converts either condition into success. Recording sampling work drives every required-positive texture's prefix to a referenced, GPU-waited draw: required pending CPU decodes pump to native submission (waiting only on decode and admission, never on transfer completion), then transfer-independent descriptors install once prior submitted frames drain out of the slot under a graphics-completion-only bound. A required texture therefore never samples fallback in the recorded frame. Optional zero-mip textures never block recording; they decode and submit through ordinary nonblocking pumps while their bindings keep sampling fallback. Driving runs only when the merged pipeline layout requires the bindless texture heap (heap detection is declaration-based so every target agrees); heapless shaders skip it, and numeric binding IDs otherwise make loaded-texture references opaque. The Sponza example therefore streams interactively and pins fully-streamed snapshots (`REQUIRED_MIPS_FULL`) with no application waiting.

Events identify a typed texture, vertex allocation, or index allocation and report:

- `SourceStaged`: the runtime owns the required CPU data, so caller source memory may be released;
- `DeviceReady`: the resource is device-local and ready for rendering;
- `Failed(status)`: terminal failure;
- `Cancelled`: terminal cancellation.

The upload queue is unbounded and lossless. It is separate from bounded runtime diagnostics. Applications must retain a callback and regularly reach graphics safe points; otherwise queued records retain memory until context destruction.

```mermaid
sequenceDiagram
    participant App
    participant Gfx as ez-gfx
    participant CPU as Decode workers
    participant GPU as Backend transfer
    participant Queue as Upload queue
    App->>Gfx: load_texture(source bytes)
    Gfx->>Gfx: copy source, reserve binding, publish magenta fallback
    Gfx->>Queue: SourceStaged
    App->>Queue: drain once per frame
    Gfx->>CPU: admit bounded decode wave
    CPU-->>Gfx: decoded mip chain
    Gfx->>GPU: admit bounded staged upload
    alt upload and publication succeed
        GPU-->>Gfx: completion
        Gfx->>Queue: DeviceReady
        Gfx->>Gfx: atomically replace fallback with real view
        App->>Gfx: continue using stable binding
        App->>Gfx: destroy or drop Context
        Gfx->>GPU: destroy retained textures after completion
    else failure
        Gfx->>Queue: Failed
    else cancel before readiness
        App->>Gfx: Texture::cancel_load
        Gfx->>Queue: Cancelled
    end
```

## Texture lifecycle

Obtain the stable bindless index immediately after `load_texture` succeeds. Until `DeviceReady`, that index safely samples opaque magenta. CPU decode and native upload remain asynchronous. Recording sampling work first drives required pending decodes to native submission (optional zero-mip textures progress only through ordinary nonblocking pumps and keep sampling fallback), then installs the transfer-independent descriptor wherever no submitted frame can still observe the slot; at the latest, a graphics-safe descriptor-update gate changes the same slot from the fallback alias to the completed real coarse view once its transfer finishes. Binding lookup does not poll or wait for real uploads; residency operations retain their existing nonblocking progress behavior. `Context::wait_idle` still drains all manager work when explicitly requested.

Cancellation wins only before initial real readiness and emits `Cancelled`. Decode, upload, cancellation, and failure paths leave the slot pointing at fallback until safe reuse; reuse republishes fallback before returning the new texture. Dropping `Texture` releases only the Rust wrapper; the context retains its heap entry and native storage so stable bindless IDs remain valid until `Context::destroy` or context drop. Context teardown drains native work where possible, then destroys real textures and the shared fallback.

Device loss is terminal. Textures still in decode or awaiting transfer completion receive a terminal `Failed(DeviceLost)` event. Textures whose real readiness was already published are not in that pending set and do not receive a retroactive upload failure.

## Residency and updates

Initial readiness requires real coarse residency plus a frame-safe published descriptor. `required_mips: 0` means optional: frame recording never waits for the texture's CPU decode, and its binding samples fallback until the first real coarse mip is decoded, submitted, transfer-complete, and descriptor-safe; ordinary event polling then reports `DeviceReady` exactly once and finer mips continue in the background. `required_mips: 1` waits for the smallest mip only; larger values delay `DeviceReady` until that many coarse levels are resident. `REQUIRED_MIPS_FULL` (`u32::MAX`) waits for the decoded total. Requirements exceeding the decoded chain fail terminally with `InvalidArgument` instead of clamping. Finer mips may continue transferring after `DeviceReady`.

`Texture::residency` returns the exposed contiguous coarse mip count and immutable total. `Texture::set_residency` accepts `1..=total`; growth returns `NotReady` until required transfers and descriptor publication complete. `Texture::update_region` validates mip bounds, block alignment, row pitch, and byte length before scheduling a copy.

## Admission and memory

Generic texture policy lives in `ez-gfx-texture-manager`: ingestion and validation (`texture` module), required-prefix selection (`REQUIRED_MIPS_FULL`, `resolve_required_mips`, `required_completion_token`), FIFO decode batching (`BatchPlan`), decode worker ownership and lifecycle (`DecodeDriver`), the shared transfer budget (`SharedTransferPool`), fine-mip planning (`fine_residency_targets`, `PrefixTransferWork`), and the uniform staging-eviction interface (`ReclaimableStaging`, blanket-implemented for every `ReusableStagingPool`). Backend crates implement the manager's `TextureBackendTexture`/`TextureBackendContext` traits for their private native types; `ez-gfx` links the selected implementations with static dispatch and owns context lifetimes, fallback aliasing, and events.

Uploads submit in two phases. Phase one allocates full mip storage on every backend but submits the required coarse prefix (`create_texture_with_prefix`), at least the coarsest mip: an optional requirement still uploads one mechanical coarse mip in the background while its binding samples fallback, so required work never head-of-line blocks behind earlier textures' fine levels in the backend transfer FIFO. Finer levels stay manager-owned in admission order and submit coarsest-first through the validated region-update vehicle under background admission once the required prefix publishes; the decode reservation holds through the required-prefix completion token and releases on cancel, loss, or unload with the retained payloads. Decode and publication order across textures stays FIFO within each requirement class, with required-positive work bypassing undecoded optional heads, so scheduling is starvation-free without a priority lane in the transfer worker. Geometry heap logic lives in `ez-gfx-geometry-manager`. Both managers consult one `SharedTransferPool` per context, so texture decode reservations, geometry uploads, and buffer copies share one admission domain for the transfer queues: geometry and buffer pressure counts against texture admission, while an empty texture ledger still admits one required reservation alone so progress cannot deadlock. Transfer-queue tokens retire geometry and buffer bytes; texture reservations release at required-prefix completion, cancel, or loss.

Texture admission is nonblocking and FIFO. Accepted sources remain manager-owned while queued; callers submit textures naturally and never choose a batch size or retry routine `QueueFull`. The manager reserves the 64 MiB per-request maximum for each active decode and limits decoded plus in-flight native-transfer work to a 256 MiB window. Completed small uploads release reservations before later waves, while an otherwise-valid request that exceeds an empty window is admitted alone so progress cannot deadlock. The synchronous decoded-to-staging copy may transiently duplicate one admitted chain, but queued decoded results and in-flight staging cannot grow with the full submission set.

The result channel remains mechanically lossless and unbounded, but only manager-permitted jobs can publish into it, so payload occupancy is bounded by the same reservations. Backend transfer queues retain their existing all-or-none chain submission and adaptive coalescing. Vulkan, Direct3D 12, and Metal staging buckets are each capped at the shared 64 MiB request maximum; holding a manager reservation through the required-prefix completion token therefore bounds active native staging without a backend interface change.

Real request boundaries remain:

- texture payloads: at most 64 MiB per request;
- C ABI caller ranges: at most 16 MiB;
- decoder worker threads: `1..=256`, with zero selecting the default count;
- texture handle and bindless descriptor capacity: finite and validated.

Transfer workers coalesce adjacent compatible requests according to staging policy byte/copy/deadline thresholds. Those are batch-flush thresholds, not caller admission limits. Reusable mapped staging buckets remain completion-gated and are reused or trimmed by the backend; the manager separately bounds active decoded and transfer payloads.

### Sponza fixture encoding

`examples/shared/assets/sponza.glb` retains the immutable-reference fixture encoding recorded in the asset itself. Its GLB metadata names `glTF-Transform v4.4.1`; `KHR_texture_basisu` references 69 KTX2 images. Every image has undefined `vkFormat`, DFD color model 163 (ETC1S), and supercompression scheme 1 (BasisLZ). Every KTX2 payload names `ktx create v4.4.2 / libktx v4.4.2` as its writer. Embedded `KTXwriterScParams` are `--threads 2` for 45 images and `--no-endpoint-rdo --no-selector-rdo --threads 2` for 24 images. Only metadata embedded in the checked-in bytes is treated as provenance.

## Texture heap capacity

The maximum is currently 1024 textures. This is one synchronized semantic/runtime/backend contract: core capability admission, compiler and HAL reflection validation, runtime handle allocation, Vulkan and Direct3D 12 descriptor counts, Metal argument-buffer layout, and `EZ_GFX_MAX_TEXTURES` all agree.

Slang requires a compile-time static array length for the cross-target `ParameterBlock<TextureHeap>` layout, so the shader declaration cannot represent an independently larger runtime heap. The maximum is technically required by the current Vulkan, Direct3D 12, and Metal layouts; it is not an incidental shader-only cap. Raising it requires one authoritative contract change across every layer plus device-capability admission. Arbitrary runtime growth is unsupported while those static layouts remain.

## Copy ownership

C calls copy borrowed source bytes during the call, preserving asynchronous lifetime safety. Encoded input must remain decode-owned until decoding finishes. Decoded/native payloads are copied into mapped staging before device transfer. The current texture API does not expose a caller-writable mapped staging lease, so it must not be described as zero-copy.

## Frame ownership

Texture bindings are recorded only through `&mut Frame`; the context-owned texture heap requires no per-frame retain call. Surface recording uses `Surface::begin_frame()` and `Frame::configure_swapchain`; named targets use `Context::begin_frame()` and `Frame::configure_render_target`. `Frame::finish(self)` preserves exact errors, while dropping an unfinished frame aborts. `Buffer<T>`, `CounterBuffer<T>`, and single-value `ValueBuffer<T>` are one-frame values: their first execute use claims them, current bindings persist across same-frame execute calls, and terminal frame paths clear bindings and invalidate claimed public use while native backing remains completion-gated. `RenderTarget::prepare_readback(&mut frame)` creates and attaches an opaque owner-and-generation request, and completed metadata and bytes exist only during the registered callback.

## Render-target readback

Managed render-target readback supports only full-image `Rgba8Unorm`; `Bgra8Srgb`, `Rgba16Float`, compressed, and depth targets fail before graph recording because backend copies do not convert texels. Safe `RenderTarget::prepare_readback(&mut frame)` returns an opaque owner-and-generation `Readback`; ABI 41 `ez_gfx_frame_enqueue_render_target_readback(context, frame, render_target, out_request_id)` returns a stable request correlator instead. Presented images use the presented-capture path and report no source handle. The correlator is reserved in graph insertion order and written only on success; a failed insertion cancels the reservation. Callback events pair opaque `readback_source` with `readback_source_kind`: texture requests report `Texture`, render-target requests report `RenderTarget`, and unrequested snapshots report `None` with a zero handle. The extent and RGBA bytes remain valid only for the callback invocation.
## C ABI 42

Install `ez_gfx_context_register_callback(context, callback, user_data)`. The callback receives `EzGfxEventKind_Upload`, runtime, diagnostic, dropped-count, and readback events on the context creator thread at graphics safe points. `ez_gfx_texture_get_binding` succeeds immediately after a successful load; the returned slot samples fallback until `EzGfxUploadStatus_DeviceReady` identifies real publication. `EzGfxTextureDesc.required_mips` carries the coarse-prefix requirement (zero means optional with no frame CPU wait, `u32::MAX` requires the full decoded chain); over-sized values fail terminally. Copy readback bytes before the callback returns. Passing a null callback clears the registration. Convert result codes with `ez_gfx_error_print`.

C retains explicit context-first texture load, cancellation, binding, residency, region-update, telemetry, resource-diagnostics, and unload functions because RAII is available only through the safe Rust interface. `ez_gfx_context_get_resource_diagnostics` fills an `EzGfxResourceDiagnostics` struct with pending-upload counts, retained bytes, and cache sizes; it rejects null outputs and stale handles like every other context query.

## Pending-upload and cache diagnostics

`Context::resource_diagnostics` returns a point-in-time `ResourceDiagnostics` snapshot, distinct from the monotonic `texture_upload_telemetry` counters. `pending_textures` counts uploads queued for decode, holding decoded output, or awaiting their final transfer completion. `pending_texture_bytes` reports owned source bytes before decode completion, decoded bytes awaiting native admission, and decoded payload bytes after native submission. Staging sizes aggregate retained bucket capacity across the shared, buffer, and counter pools; `pipeline_entries` counts retained compiled pipelines and `readback_bytes` aggregates retained readback frames. Counts and bytes saturate instead of wrapping. The query observes creator-thread state without requiring device health, so teardown titles keep reporting after loss until destruction.

## Staging retention budgets and memory telemetry

Upload staging reuses best-fit buckets across the shared, per-stride buffer, and counter pools. Each pool carries a finite retention ceiling (32 MiB shared, 16 MiB per buffer-stride entry, 8 MiB counter); `put` never evicts, and only explicit trims enforce the ceiling. Per-pool ceilings cannot bound the stride map, which grows one pool per element width, so a 64 MiB context-wide aggregate ceiling caps every transfer cache including backend texture staging (`all_transfer_cache_retained`): enforcement evicts the largest completed bucket across every pool until the combined total fits, without touching in-flight buckets and without affecting best-fit reuse order. Point-in-time diagnostics keep reporting the three context pools alone via `aggregate_staging_retained`.

`Context::memory_telemetry` reports allocator telemetry (via one `generate_report` per query, never per frame), retained staging totals with the true aggregate high-water across pools, counter serialization capacity, and decode worker counts. Surface image counts, extents, formats, and depth bytes aggregate active safe surface state in one convention on every backend: extents prefer the safe surface extent, Vulkan contributes device swapchain fields, DX12 contributes back-buffer counts with its R8G8B8A8 format code, and Metal contributes drawable counts with its BGRA8 sRGB code. Zero images gates extent and format to unknown. The manager-owned decode driver builds its Rayon pool lazily before admission, so a rejected load leaves no registry, identity, or pending residue.

## Synchronization

Device initialization waits only for creation/publication of the shared fallback before any frame can sample. Later frames never globally wait for pending real texture uploads, and no submission path blocks the caller on transfer completion. Recording sampling work drives required pending CPU decodes to native submission first (optional textures never wait here; waiting only on decode and admission), installs each required-prefix descriptor wherever no submitted frame can still observe the slot, and contributes every referenced texture's required-prefix completion token as a graph dependency: Vulkan waits the texture timeline semaphore, Direct3D 12 enqueues `ID3D12CommandQueue::Wait` on the texture fence, and Metal encodes `waitForEvent` on the texture completion event. The remaining Direct3D 12 frame-fence CPU wait covers only readback coherence and present-error reclamation. Real descriptor/view replacement reuses each backend's graphics-completion publication gate whenever a submitted frame is still in flight.

## Verification and remaining evidence

Pure queue, allocator, and scheduling transitions are covered by texture-manager tests. ABI layout tests cover callback event records, typed heap/allocation handles, one-frame buffer signatures, presentation modes, the resource-diagnostics struct and export, and the ABI 42 contract. Native backend behavior requires the Linux Vulkan, Windows DX12, and macOS Metal remote matrices.
