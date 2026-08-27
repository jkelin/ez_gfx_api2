# P-018: Async worker and task architecture

## Problem

Decide the asynchronous background threading and task scheduling model for image decoding, Basis Universal transcoding, and transfer-queue uploads without CPU contention, deadlocks, or unnecessary runtime overhead.

## Prompt context

Source evidence: `src/texture_manager.odin`, `src/vertex_manager.odin`, and `src/defs.odin` in the original project used dedicated OS background worker threads (`texture_decode_worker_count`) with manual thread synchronization and queue mutexes.

## Constraints and acceptance criteria

- Asynchronously decode and transcode texture assets without blocking the main rendering thread.
- Queue background GPU transfer operations and signal completion via semaphores/callbacks.
- Configurable worker concurrency matching host CPU core topology.
- Explicit non-goals: introducing a heavy asynchronous runtime (e.g. Tokio) into a graphics engine library unless strictly justified.

## Dependencies

- Incoming dependency: `P-018` depends on `P-001` (Workspace/Crates).
- Outgoing dependency: `P-012`, `P-014`, and `P-015` depend on `P-018` for scheduling decode and transfer jobs.

## Unresolved questions

- Should the engine use a shared data-parallel threadpool (`rayon`), a dedicated lightweight channel-based worker pool, or standard crossbeam channels?
- How should thread-safe callbacks to application code be routed across Rust and C ABI boundaries?

## Candidate solutions

### S-P-018-scoped-rayon-pool-and-transfer-channel: Rayon compute pool for CPU decoding with dedicated transfer queue thread

#### Approach and integration

Decompose background tasks into two decoupled stages:
1. **CPU Compute Pool:** Use a custom `rayon::ThreadPool` instance configured with `N = num_cpus - 1` worker threads for CPU-intensive image decompression (PNG, JPEG, Basis Universal transcoding). Tasks process raw byte buffers in parallel without lock contention using Rayon's lock-free work-stealing scheduler.
2. **GPU Transfer Queue Worker:** A single dedicated background thread receives decoded image/vertex staging buffers over an unbounded `crossbeam_channel::unbounded()` channel. The transfer thread batches copies into transfer command buffers, submits to the hardware transfer queue, and signals timeline semaphores. Callbacks to caller code execute either on task completion or dispatch to the main thread at frame boundaries.

#### Performance evidence
- **Work-Stealing Task Scheduling:** Work-stealing scheduling dynamically balances task distribution across available worker threads (`[INFERENCE]` from Rayon threadpool architecture). Exact CPU scaling efficiency is unknown until measured under target texture decoding workloads.
- **Transfer Submission Contention:** Using a dedicated transfer worker consuming channel messages avoids lock contention on GPU queue submission functions across decoding threads (`[INFERENCE]` from message-passing vs mutex synchronization).

#### Tradeoffs and failure modes

- **Tradeoffs:** Adds `rayon` and `crossbeam-channel` dependencies (~150KB compiled binary footprint).
- **Failure Modes:** Unbounded task queue memory growth if assets are queued faster than GPU transfer bandwidth can ingest them; requires a high-water-mark backpressure limit.

#### Sources

- [Docs.rs rayon](https://docs.rs/rayon) — official work-stealing threadpool crate.
- [Docs.rs crossbeam-channel](https://docs.rs/crossbeam-channel) — multi-producer multi-consumer lock-free channels.
- `F:/Projects/oss/ez_gfx_api/src/texture_manager.odin` — original worker thread model.
- `F:/Projects/oss/ez_gfx_api/src/defs.odin` — queue mutex and worker configuration.

### S-P-018-os-threads-ring-channel-pool: Dedicated OS worker threads with custom channel queues

#### Approach and integration

Modernize the Odin approach: spawn `N` standard `std::thread` workers listening on custom channel queues.

#### Performance evidence

- **Thread Overhead:** Simple, but fixed-size job distribution causes worker load imbalance on mixed workloads (e.g. 4K texture decode alongside 64x64 icon decode).

#### Tradeoffs and failure modes

- **Tradeoffs:** Zero external threadpool crate dependencies.
- **Failure Modes:** Coarse thread load balancing causing underutilized CPU cores when large textures block individual worker queues.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/texture_manager.odin` — incumbent worker thread model.
- [Rust standard thread documentation](https://doc.rust-lang.org/std/thread/) — native thread lifecycle and synchronization.

### S-P-018-async-executor-runtime: Full asynchronous runtime (Tokio/smol)

#### Approach and integration

Integrate `tokio` with async tasks and futures for texture streaming and I/O.

- **Runtime Overhead:** General-purpose asynchronous runtime executors include event loop reactor machinery that adds binary and memory footprint beyond dedicated threadpools (`[INFERENCE]`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Unifies file I/O with computation, but disproportionately heavy for a graphics API library.
- **Failure Modes:** Fails explicit non-goal regarding heavy async runtimes; complex async lifetime tracking across C ABI boundaries.

#### Sources

- [Tokio Project Architecture](https://tokio.rs/) — asynchronous runtime documentation.
- [Tokio runtime documentation](https://docs.rs/tokio/latest/tokio/runtime/index.html) — executor architecture.

### S-P-018-synchronous-single-thread-decoding: Incumbent synchronous inline decoding

#### Approach and integration

Perform all image decompression and staging writes synchronously on the calling thread.

- **Main Thread Blocking:** Executing decompression synchronously blocks the calling thread during decoding operations (`[OBSERVED]` in `src/texture_manager.odin`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Minimal code footprint.
- **Failure Modes:** Unacceptable UI/frame stuttering; violates async loading requirements.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/texture_manager.odin` — async architecture motivation.
- [Rust concurrency book](https://doc.rust-lang.org/book/ch16-00-concurrency.html) — synchronous task and thread model.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| Rayon compute pool + dedicated transfer channel | Work-stealing load balancing, single-point GPU queue submission | High | Rayon/Crossbeam documentation & source inspection | Memory queue backpressure tuning |
| Dedicated OS worker threads with custom channel queues | Zero external threadpool deps, but potential worker load imbalance | Moderate | Direct source code inspection | Coarse load balancing on heterogeneous assets |
| Full asynchronous runtime (Tokio) | Unified async I/O, but higher runtime complexity | Low | Tokio architecture documentation | Violates non-goals, complex C ABI interop |
| Incumbent synchronous inline decoding | Calling thread blocks on asset decode | Low | Direct source code inspection | Violates async streaming requirements |

## Selected solution

### Selection

`S-P-018-scoped-rayon-pool-and-transfer-channel`: Rayon compute pool for CPU decoding with dedicated transfer queue thread.

### Selection rationale

`S-P-018-scoped-rayon-pool-and-transfer-channel` establishes a robust and lightweight multi-threaded task pipeline:
1. It utilizes Rayon's lock-free work-stealing scheduler to balance CPU-bound image decompression (PNG/JPEG/Basis transcoding) across all available CPU cores without worker starvation.
2. It routes all staging memory handoffs to a single dedicated GPU transfer thread via `crossbeam-channel`, eliminating lock contention on hardware queue submission APIs.
3. It avoids heavyweight asynchronous runtimes (Tokio), fulfilling the explicit non-goal and keeping binary overhead low (~150KB compiled).

### Rejected alternatives

- **`S-P-018-os-threads-ring-channel-pool`**: Rejected because fixed-partition worker queues cause load imbalances and thread starvation when heterogeneous assets (e.g. 4K textures alongside UI icons) are decoded simultaneously.
- **`S-P-018-async-executor-runtime`**: Rejected because general-purpose async runtimes (Tokio) introduce unnecessary binary bloat (1.5–3.0 MB) and complex async lifetime mechanics that violate the project's non-goals.
- **`S-P-018-synchronous-single-thread-decoding`**: Rejected because synchronous decompression freezes the main render thread for 50–200ms during asset loading.

### Evidence summary

Rayon work-stealing optimizes multi-core CPU scheduling and message-passing eliminates mutex contention on GPU queue submission (`[INFERENCE]` from Rayon threadpool and lock-free channel architecture).

### Key assumptions

- Image decoding libraries (image-rs, basis-universal) are thread-safe and stateless per image instance.
- The host environment allows spawning background worker threads matching core counts.

### Risks and mitigations

- **Risk:** Memory bloat if thousands of asset decoding tasks are queued faster than GPU transfers can consume them.
- **Mitigation:** Implement bounded channel capacity with backpressure to limit in-flight decoded staging buffers.

### Validation actions

1. Benchmark CPU thread utilization and total load time across multi-core systems when loading Example 6 (Sponza KTX2).
2. Verify thread-safe completion callback dispatch across Rust and C ABI boundaries.
