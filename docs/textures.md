# Textures

Texture loading is asynchronous. Each safe `Texture` is a typed view into the context-owned bindless heap; dropping the wrapper does not unload or recycle its texture. A Rayon pool performs decode/mip work and backend transfer owners submit device copies.

## Public path

`Context::load_texture` validates the request, copies caller source before returning, reserves a generational lease, and schedules CPU work. Register one creator-thread callback with `Context::register_callback`; graphics safe points drain the lossless upload queue, advance decoded work and native transfers, publish completed views, and report failures.

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
    Gfx->>CPU: copy source and create decode job
    Gfx->>Queue: SourceStaged
    App->>Queue: drain once per frame
    CPU-->>Gfx: decoded mip chain
    Gfx->>GPU: upload staged mips
    alt upload and publication succeed
        GPU-->>Gfx: completion
        Gfx->>Queue: DeviceReady
        App->>Gfx: Texture::binding and use
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

After a texture `DeviceReady` event, obtain its stable bindless index from the owning `Texture`. Use a resident fallback until then. Binding, residency, and residency updates each perform a nonblocking owner-thread progress pass that drains completed decode work, admits transfers, and publishes completed residency. `Context::wait_idle` drains native work, then performs the same publication pass. None dequeues upload events or replaces event consumption.

Cancellation wins only before initial readiness and emits `Cancelled`. Dropping `Texture` releases only the Rust wrapper; the context retains its heap entry and native storage so stable bindless IDs remain valid until `Context::destroy` or context drop. Context teardown cancels queued decode work, drains native work where possible, then destroys retained textures, events, surfaces, and other resources.

Device loss is terminal. Textures still in decode or awaiting transfer completion receive a terminal `Failed(DeviceLost)` event. Textures whose readiness was already published are not in that pending set and do not receive a retroactive upload failure.

## Residency and updates

Initial readiness requires a completed coarse mip range and a frame-safe published descriptor. Finer mips may continue transferring after `DeviceReady`.

`Texture::residency` returns the exposed contiguous coarse mip count and immutable total. `Texture::set_residency` accepts `1..=total`; growth returns `NotReady` until required transfers and descriptor publication complete. `Texture::update_region` validates mip bounds, block alignment, row pitch, and byte length before scheduling a copy.

## Admission and memory

CPU jobs, the decoded-result channel, and backend transfer request channels have no fixed job-count or aggregate-byte backpressure. Scheduling is limited by actual allocation, counter, thread, or OS failure. `QueueFull` remains a representable ABI status for genuine counter/channel failure, not routine texture admission control.

Real request boundaries remain:

- runtime texture payloads: at most 64 MiB per request;
- C ABI caller ranges: at most 16 MiB;
- decoder worker threads: `1..=256`, with zero selecting the default count;
- texture handle and bindless descriptor capacity: finite and validated.

Transfer workers coalesce adjacent compatible requests according to staging policy byte/copy/deadline thresholds. Those are batch-flush thresholds, not admission limits. Reusable mapped staging buckets are reclaimed only after their completion token retires and trimmed after idle epochs.

## Texture heap capacity

The maximum is currently 1024 textures. This is one synchronized semantic/runtime/backend contract: core capability admission, compiler and HAL reflection validation, runtime handle allocation, Vulkan and Direct3D 12 descriptor counts, Metal argument-buffer layout, and `EZ_GFX_MAX_TEXTURES` all agree.

Slang requires a compile-time static array length for the cross-target `ParameterBlock<TextureHeap>` layout, so the shader declaration cannot represent an independently larger runtime heap. The maximum is technically required by the current Vulkan, Direct3D 12, and Metal layouts; it is not an incidental shader-only cap. Raising it requires one authoritative contract change across every layer plus device-capability admission. Arbitrary runtime growth is unsupported while those static layouts remain.

## Copy ownership

C calls copy borrowed source bytes during the call, preserving asynchronous lifetime safety. Encoded input must remain decode-owned until decoding finishes. Decoded/native payloads are copied into mapped staging before device transfer. The current texture API does not expose a caller-writable mapped staging lease, so it must not be described as zero-copy.

## Frame ownership

Texture bindings are recorded only through `&mut Frame`; the context-owned texture heap requires no per-frame retain call. Surface recording uses `Surface::begin_frame()` and `Frame::configure_swapchain`; named targets use `Context::begin_frame()` and `Frame::configure_render_target`. `Frame::finish(self)` preserves exact errors, while dropping an unfinished frame aborts. `Buffer<T>`, `CounterBuffer<T>`, and single-value `ValueBuffer<T>` are one-frame values: their first execute use claims them, current bindings persist across same-frame execute calls, and terminal frame paths clear bindings and invalidate claimed public use while native backing remains completion-gated. `RenderTarget::prepare_readback(&mut frame)` creates and attaches an opaque owner-and-generation request, and completed metadata and bytes exist only during the registered callback.

## C ABI 37

Install `ez_gfx_context_register_callback(context, callback, user_data)`. The callback receives `EzGfxEventKind_Upload`, runtime, diagnostic, dropped-count, and readback events on the context creator thread at graphics safe points. Compare upload resource handles, resolve bindings after `EzGfxUploadStatus_DeviceReady`, and copy readback bytes before the callback returns. Passing a null callback clears the registration. Convert result codes with `ez_gfx_error_print`.

C retains explicit context-first texture load, cancellation, binding, residency, region-update, telemetry, and unload functions because RAII is available only through the safe Rust interface.

## Synchronization

Frame recording imports only resources referenced by active work. Texture descriptor publication is guarded by transfer and graphics completion. Vulkan and Direct3D 12 still use conservative host waits in parts of the transfer handoff; Metal has nonblocking submission coverage. Removing those remaining waits is tracked separately and is not claimed complete here.

## Verification and remaining evidence

Pure queue and allocator transitions are covered by runtime tests. ABI layout tests cover callback event records, typed heap/allocation handles, one-frame buffer signatures, and the ABI 37 contract. Native backend behavior requires the Linux Vulkan, Windows DX12, and macOS Metal remote matrices.
