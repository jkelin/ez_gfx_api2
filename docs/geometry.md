# Geometry

Geometry uses owning auto-growing named vertex heaps and one lazy context-owned device-local `u32` index heap. Their allocation wrappers retain both the shared `Rc<ContextInner>` and the parent resource lease. Structured and counted buffers are separate persistent context-owned resources.

## Public path

1. Create each typed vertex heap with a unique semantic name through `Context::create_vertex_heap<T>(name)`; retain its owning `VertexHeap<T>`.
2. Call `VertexHeap::upload(&[T])` or `Context::upload_indices(&[u32])`; the index heap is created lazily, and each upload returns an owning generation-validated allocation.
3. Query `VertexAllocation::range` or `IndexAllocation::range` for indirect draw offsets.
4. Register one `Context` callback for upload, runtime, diagnostic, and readback events.
5. Drop heaps and allocations independently. Allocation leases keep their parent heap and context alive until cleanup is safe.

The C ABI keeps vertex heaps explicit because C has no RAII. Index storage remains a lazy context-owned singleton with upload, range-query, and allocation-release operations only.

```mermaid
sequenceDiagram
    participant App
    participant Gfx as ez-gfx
    participant Queue as Upload queue
    participant GPU as Backend transfer
    App->>Gfx: create owning heap
    App->>Gfx: upload typed slice
    Gfx->>Gfx: validate count/stride/bytes and reserve lease
    Gfx->>GPU: copy owned staging to device heap
    Gfx->>Queue: SourceStaged
    App->>Queue: drain once per frame
    alt transfer succeeds
        GPU-->>Gfx: completion
        Gfx->>Queue: DeviceReady
        App->>Gfx: query range and use
        App->>Gfx: drop allocation and heap
    else transfer fails or is cancelled
        Gfx->>Queue: Failed or Cancelled
        App->>Gfx: drop allocation
    end
```

## Shader reflection and frame binding

A shader declares a named heap explicitly:

```slang
[VertexHeap("positions")]
StructuredBuffer<float4> positions;
```

Reflection records `VertexHeap` separately from `StructuredBuffer`. Applications do not supply a `PublicBinding` for it. Frame recording imports the named heap automatically and each backend lowers it to its private descriptor layout. Vulkan, Direct3D 12, and Metal retain distinct physical layouts behind the same public contract.

The global index heap remains a native index-buffer binding. `DrawIndexedCommand::first_index` is the queried index allocation start plus any mesh-local index offset. Because a vertex heap is bound at byte offset zero, `vertex_offset` must include the queried vertex allocation start.

## Allocation and removal

Each heap uses an ordered range free list. Allocation validates stride, capacity, arithmetic, heap ownership, handle owner, resource kind, and generation. An allocation lease retains its parent heap, so stale, foreign, wrong-kind, and duplicate C releases are rejected while safe Rust cleanup remains single-owner `Drop`.

The index heap is a context singleton, not a freely creatable family of named heaps. Range retirement remains gated by native completion; parent resources remain leased until their children and recorded uses are gone.
## Upload events

`VertexHeap::upload(&[T])` and `Context::upload_indices(&[u32])` enqueue lossless typed transitions: `SourceStaged`, `DeviceReady`, `Failed(status)`, or `Cancelled`.

`Context::register_callback` is the sole safe event channel. The facade dispatches queued events at creator-thread operation seams; applications do not poll internal runtime queues.

A heap-level maximum readiness token is used when a frame imports a named heap. It may wait for a later allocation in the same heap, but never permits early use.

## Admission and copies

Geometry staging grows subject to allocator and OS failure, not a fixed slot count. Reusable mapped buckets retire by completion token and are reused only after completion.

The typed slice upload interface derives and checks count, stride, multiplication, and byte length before one caller-slice to mapped-staging copy, followed by the GPU copy. Raw pointer/count conversion and the 16 MiB caller boundary remain in `ez-gfx-ffi`. The interface does not provide a direct mapped lease.

## Persistent frame buffers

Applications allocate `Buffer<T>` and `CountedBuffer<T>` from `Context`, then populate them before beginning a frame. Recording imports their CPU-backed contents lazily when a binding or indirect draw first references them. The frame retains each imported buffer; mutation returns `NotReady` until the transaction finishes or aborts, after which the same owner can be updated and reused.

Compute-generated indirect bytes cannot update CPU publication metadata, so callers publish the known indirect count before compute and graphics share the buffer in one frame. Native per-frame storage remains completion-gated or, after indeterminate failure, quarantined, without invalidating the context-owned wrapper.

## Frame ownership

`Surface::begin_frame()` creates an owning target-less `Frame`; `Frame::configure_swapchain(size, format)` attaches presentation. Named targets use `Context::begin_frame()` and `Frame::configure_render_target(name, size, format)`. `Frame::finish(self)` preserves exact errors. `Drop` aborts an unfinished frame, so safe Rust exposes no frame-end, frame-abort, or transient-release functions.

## Examples

The shared `Example` host owns `Context`, `Surface`, resize/input/automation, callbacks, and consuming completion. Each procedural loop receives `WindowFrame`, calls `surface.begin_frame()`, explicitly configures the swapchain from `window_frame.size`, records, then passes both `Frame` and `RenderTarget` to `Example::handle_frame`.
