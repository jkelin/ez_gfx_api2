# P-012: Transfer queue staging and batching

## Problem

Decide how to implement reusable, bucketed staging buffer pools for both vertex and texture data, batch transfer-queue submissions to eliminate per-upload command buffer overhead, and replace global timeline signal serialization with decoupled per-stream, transfer-domain queue timelines.

## Prompt context

Source evidence: `TODO.md` ("Add a reusable vertex staging buffer pool...", "Batch transfer-queue vertex uploads...", "Add a reusable texture staging buffer pool...", "Batch transfer-queue texture submissions...", "Replace global timeline signal serialization with queue-side dependencies or per-manager ordered timelines").

## Constraints and acceptance criteria

- Provide bucketed staging buffer pools that recycle host-visible buffers across frames by timeline value.
- Coalesce multiple pending vertex, index, and texture copies into batched transfer command buffers.
- Decouple vertex and texture transfer workers so independent copy operations do not serialize on a shared monotonic timeline.
- Explicit non-goals: synchronous GPU blocking on individual transfer submissions.

## Dependencies

- Incoming dependency: `P-012` depends on `P-003` (HAL queues), `P-004` (Allocator), and `P-011` (Vertex manager).
- Outgoing dependency: `P-015` depends on `P-012` for staging memory and transfer batch execution; P-013 consumes its submitted aggregate prefixes.

## Unresolved questions

- How should staging pool bucket sizes (e.g. 64KB, 1MB, 16MB) be configured dynamically?
- Should transfer submission batching be triggered by frame-boundary flushes, size thresholds, or time intervals?

## Candidate solutions

### S-P-012-timeline-bucketed-staging-and-batching: Bucketed power-of-two recycling staging pools with adaptive batching transfer engine

#### Approach and integration

Implement a unified `StagingPool` managing power-of-two size-classed host-visible buffer blocks (e.g., 64KB, 256KB, 1MB, 4MB, 16MB). Each block records the GPU timeline semaphore value upon submission; subsequent allocation requests scan the appropriate size bucket and recycle any block whose associated GPU timeline has completed (`current_timeline >= block.last_submitted_timeline`). Transfer requests enqueue copy operations into an `AsyncTransferBatcher`. The batcher coalesces pending vertex, index, and texture copies into a single command buffer submitted at frame boundaries or when an accumulated byte threshold (e.g. 32MB) is reached, signaling an independent transfer timeline semaphore.

#### Performance evidence

- **Driver Submission Overhead:** In Vulkan/DX12, calling queue submit functions incurs driver and kernel transition overhead per submission call (`[INFERENCE]` from batching principles). Batching amortizes this overhead across multiple transfers.
- **PCIe Saturation:** Coalescing transfer batches reduces command recording overhead and enables larger continuous DMA copy dispatches (`[INFERENCE]` from GPU memory copy pipelining characteristics).
- **VRAM Allocation Churn:** Reusing staging blocks avoids continuous virtual memory buffer allocation and destruction (`[INFERENCE]` from memory pooling principles).

#### Tradeoffs and failure modes

- **Tradeoffs:** Requires maintaining size-classed free lists and periodic trimming of idle high-water-mark buckets to prevent VRAM staging bloat.
- **Failure Modes:** If transfer batching timeout is too long, initial texture readiness could be delayed by several frames unless explicit flush triggers exist.

#### Sources

- [NVIDIA Vulkan Memory and Transfer Guidelines](https://developer.nvidia.com/blog/vulkan-dos-donts/) — batching transfers and staging buffer recycling.
- `F:/Projects/oss/ez_gfx_api/src/vertex_manager.odin` & `src/texture_manager.odin` — original upload implementations.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — staging pools, batching, and timeline serialization requirements.

### S-P-012-ring-buffer-staging-monotonic-queue: Persistent mapped ring-buffer staging with monotonic submission queue

#### Approach and integration

Allocate a fixed-size (e.g. 64MB or 128MB) persistently mapped host-visible ring buffer. Uploads append data contiguously, advancing a write head. Submissions record copy commands and update a monotonic submission timeline. Staging memory before the GPU-completed timeline head is recycled continuously.

#### Performance evidence

- **Allocation Speed:** Fast pointer bump allocation for sub-allocations within active ring segments (`[INFERENCE]` from contiguous ring buffer structure).
- **Headroom Limits:** If a single asset upload exceeds ring buffer capacity, it requires a fallback allocation or stalls until space frees (`[INFERENCE]`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Zero bucket searching overhead, but rigid maximum single-allocation size bound.
- **Failure Modes:** High risk of CPU stalls when large asset streaming fills the ring buffer faster than the GPU transfer queue consumes it.

#### Sources

- [Vulkan Staging Ring Buffer Guide](https://docs.vulkan.org/guide/latest/transfer_queue.html) — streaming ring buffer architecture.

### S-P-012-per-upload-dedicated-staging: Incumbent dedicated staging allocation and individual command buffer submit

#### Approach and integration

Maintain the original Odin pattern: each vertex upload and texture load dynamically creates a dedicated `vma.Allocation` staging buffer, records a dedicated `vkCommandBuffer`, and immediately submits to the transfer queue.

#### Performance evidence

- **Driver & Memory Overhead:** 100 texture uploads generate 100 distinct buffer allocations and 100 `vkQueueSubmit` calls (`[OBSERVED]` in `src/texture_manager.odin`), causing significant driver submission spikes and heap fragmentation.

#### Tradeoffs and failure modes

- **Tradeoffs:** Simple isolate-and-forget lifecycle per upload.
- **Failure Modes:** Heavy GPU driver CPU hitching, memory fragmentation, and GPU queue stalls during loading sequences.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/texture_manager.odin` — original implementation.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — original implementation limitations.
- [Vulkan transfer queue guide](https://docs.vulkan.org/guide/latest/transfer_queue.html) — transfer submission and staging guidance.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| Bucketed power-of-two recycling staging pools + adaptive batcher | Amortizes driver submit overhead, eliminates staging allocation churn | High | Direct source evidence & GPU memory guidelines | Bucket high-water-mark memory management |
| Persistent mapped ring-buffer staging | Fast pointer allocation, simple ring pointers, but rigid capacity cap | Moderate | Direct source evidence & Vulkan guidelines | Stalls on uploads exceeding ring buffer size |
| Incumbent dedicated staging allocation + single submit | High driver submission overhead per asset, heavy allocation churn | Low | Direct source code inspection | Unusable for heavy asset loading screens |

## Selected solution

### Selection

`S-P-012-timeline-bucketed-staging-and-batching`: Bucketed power-of-two recycling staging pools with adaptive batching transfer engine.

### Selection rationale

`S-P-012-timeline-bucketed-staging-and-batching` resolves multiple inherited staging and submission bottlenecks from TODO.md:
1. It pools and recycles host-visible staging buffers across size buckets based on GPU timeline completion, eliminating continuous buffer allocation/destruction churn.
2. It amortizes driver submission overhead by coalescing multiple pending vertex, index, and texture copies into batched transfer command buffers.
3. It decouples transfer submissions across distinct timeline semaphores, removing global monotonic timeline serialization between vertex and texture managers.

### Rejected alternatives

- **`S-P-012-ring-buffer-staging-monotonic-queue`**: Rejected because a fixed-size ring buffer cannot accommodate large individual 4K texture uploads without stalling or requiring a secondary fallback allocator.
- **`S-P-012-per-upload-dedicated-staging`**: Rejected because recording and submitting one command buffer per asset upload creates severe driver submission overhead and memory fragmentation during bulk asset loading.

### Evidence summary

Reusing staging blocks by GPU timeline status eliminates allocation churn and amortizes driver submission overhead across batched transfers (`[INFERENCE]` from GPU memory pooling and submission batching principles).

### Key assumptions

- The GPU supports timeline semaphores (`VK_KHR_timeline_semaphore` on Vulkan, native timeline fences on DX12/Metal).
- Staging allocations are returned to the pool upon transfer batch submission.

### Risks and mitigations

- **Risk:** Memory bloat if high-water-mark staging buffers remain allocated indefinitely after large loading screens.
- **Mitigation:** Implement an idle bucket trimmer that releases staging buffers unused for more than a configurable number of frames.

### Validation actions

1. Benchmark staging buffer allocation churn and reuse rates under repeated upload stress tests.
2. Verify batched transfer execution and completion timeline signaling in multi-texture loading scenarios (Example 6 Sponza).

### Implementation and evidence status

Power-of-two staging reuse, independent geometry/texture completion streams, and native cross-texture batches are implemented on Vulkan, Direct3D 12, and Metal. CPU jobs, decoded-result channels, geometry staging growth, and transfer request channels have no fixed count or aggregate-byte admission limit; real allocation, OS, handle, and request-size limits remain.

Batch byte/copy/deadline values are flush thresholds, not backpressure. Adjacent equal stream stages coalesce without FIFO reordering; repeated writes remain separate. Per-mip completion remains truthful. Prior queue-full measurements describe the superseded bounded implementation and are not current evidence.

Metal has nonblocking submission coverage. Vulkan and Direct3D 12 retain targeted host waits; removing them needs a dedicated transition/acquire queue and remains tracked work. Fresh remote evidence for this cutover is required before claiming native completion.

The 2026-09-07 local workspace, lint, ABI, C-build, event, and Vulkan/DX12 pixel evidence passed. Fresh remote native evidence is absent: the exact Linux/macOS mise tasks failed during SSH-agent signing, and the Windows task lacks its required host/path variables.

The selected nonblocking-submission goal is implemented on Metal; Vulkan and Direct3D 12 retain targeted host waits as explicit exceptions (see below), recorded in root `TODO.md`.

Polling paths reap already-completed Vulkan/Metal frame slots before the texture descriptor gate. Reaping only observes fence/command status and never waits.

Metal texture owners commit without waiting for execution; an earlier errored buffer fails the next submission without blocking, and shutdown/wait-idle drains still join GPU work. Vulkan and Direct3D 12 keep their host waits: submitting transfer-completion work early pends on the shared graphics queue ahead of later frames (breaking coarse sampling while fine work is blocked, as the coarse regressions prove), and COPY lists reject shader-resource transitions at close time. A Metal overlap regression on Apple Silicon submits a second batch while the first is GPU-pending; the pre-fix code fails it by timeout. `flush_through` proves driver submission on Metal; Vulkan/DX12 callbacks additionally join GPU completion. A dedicated transition/acquire queue per worker would remove the remaining waits without breaking frame order; that device-init change is future work.
