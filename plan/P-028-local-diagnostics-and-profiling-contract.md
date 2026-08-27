# P-028: Local diagnostics and profiling contract

## Problem

Define one local, structured contract for errors, warnings, graph/backend diagnostics, asynchronous job failures, and requested performance observations. Existing decisions produce typed statuses, schedule diagnostics, upload telemetry, adapter metadata, and measurements, but no selected schema defines severity, correlation, ownership, bounded retention, or composition without losing cause.

## Prompt context

The runtime spans compiler artifacts, HAL, graph compilation, workers, transfers, textures, presentation, and tests. Remote telemetry is not requested. Public failures must remain actionable through Rust and FFI.

## Constraints and acceptance criteria

- Typed public failures remain distinct from logs.
- One operation has stable local correlation across worker, transfer, graph, queue, and callback boundaries.
- Counters/timestamps declare units, clock/domain, scope, availability, and collection overhead.
- Collection and retention are bounded and locally controlled; no remote upload or hidden persistence.
- Fatal, recoverable, warning, and profiling semantics are distinct.

## Dependencies

- Depends on P-002 status/FFI, P-003 backend errors, P-008 graph diagnostics, P-015 upload phases, P-018/P-026 async events, and P-019 test evidence.
- Feeds P-020 cutover reporting.

## Unresolved questions

- What portable envelope and taxonomy spans sync/async failures?
- How are operation/resource/frame/node/job/submission/backend messages correlated?
- Are observations pulled, streamed, capture-scoped, or combined?
- Which bounded queues/rings own retention and overflow behavior?

## Candidate solutions

### S-P-028-pull-snapshot-diagnostics: Explicit local diagnostic snapshots

#### Approach and integration

Each context owns bounded counters and a latest-state diagnostic snapshot. API consumers call `read_diagnostics`/`read_profile` to obtain typed errors, warnings, queue depths, phase timings, and backend identifiers with operation/resource/frame correlation fields.

#### Performance evidence

Steady-state overhead is bounded by counter writes and snapshot copying; exact CPU cost, memory, and readout latency are unknown until measured with diagnostics disabled/enabled. Event history can be lost between reads.

#### Tradeoffs and failure modes

Simple FFI and no worker-to-host push thread; polling hosts may miss transient events. Overflow is replaced by explicit dropped-count fields rather than unbounded storage.

#### Sources

- [Rust `Instant`](https://doc.rust-lang.org/std/time/struct.Instant.html) — monotonic duration measurement.
- [Vulkan timestamp queries](https://docs.vulkan.org/spec/latest/chapters/queries.html) — GPU timing query semantics.

### S-P-028-bounded-local-event-stream: Bounded structured event stream

#### Approach and integration

All components emit structured diagnostic events into a bounded per-context ring/channel. Events include severity, correlation IDs, timestamps/domains, component, backend details, and typed payload. Consumers poll or subscribe through the existing delivery boundary; overflow emits a synthetic loss event/counter.

#### Performance evidence

Per-event allocation/copy and synchronization overhead, queue memory, overflow rate, and readout latency are unknown until measured under upload/render stress with event rates declared.

#### Tradeoffs and failure modes

Preserves causal ordering and transient failures, but ordering across queues requires sequence/domain metadata. Overflow and consumer starvation are explicit operational concerns.

#### Sources

- [Rust `sync_channel`](https://doc.rust-lang.org/std/sync/mpsc/fn.sync_channel.html) — bounded ordered event transport.
- [Vulkan debug utils](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_debug_utils.html) — native diagnostic callback context.

### S-P-028-scoped-capture-session: Opt-in local capture sessions

#### Approach and integration

Normal runtime exposes typed errors and low-cost counters. A caller starts a bounded capture session selecting components/frames; the session records detailed events, CPU/GPU timestamps, queue transitions, and upload phases into a local file or in-memory report, then closes explicitly.

#### Performance evidence

Disabled steady-state cost can be near counter checks but is unknown; enabled overhead, capture memory, file I/O, and readout latency depend on event volume and session bounds and require measurement.

#### Tradeoffs and failure modes

Limits profiling perturbation and report size, but failures outside a session are not fully reconstructable. Session start/stop and file-write errors must not affect rendering correctness.

#### Sources

- [Rust `tracing` spans](https://docs.rs/tracing/latest/tracing/span/) — scoped structured context.
- [D3D12 timestamp queries](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing) — GPU timing evidence.

## Performance comparison

| Candidate | Runtime overhead | Causality/retention | Constraint fit | Evidence |
|---|---|---|---|---|
| Pull diagnostic snapshots | Bounded steady-state cost; exact cost unknown | Loses transient events between reads | High | Rust/Vulkan timing docs |
| Bounded local event stream | Per-event cost and overflow unknown | Best continuous causal retention within bound | High | Rust/Vulkan diagnostics docs |
| Scoped capture session | Low expected disabled cost; enabled cost unknown | Detailed but only during selected session | High | tracing/API timing docs |

## Selected solution

### Selection

`S-P-028-bounded-local-event-stream`: Bounded structured event stream.

### Selection rationale

The event stream best preserves causal context across worker, transfer, graph, queue, backend, and callback boundaries, which is the central unresolved requirement. A bounded local stream can carry typed failures and profiling observations without remote telemetry or hidden persistence; pull snapshots remain a derived readout for low-frequency consumers.

### Rejected alternatives and reversal conditions

- **`S-P-028-pull-snapshot-diagnostics`**: Rejected as the sole contract because transient async failures and event ordering can be lost between reads. Reconsider for minimal deployments where only latest-state diagnostics are required.
- **`S-P-028-scoped-capture-session`**: Rejected as the sole contract because failures outside an active capture are not reconstructable. Retain as an optional higher-detail mode if stream overhead is measured to perturb workloads.

### Evidence and unknowns

Bounded channel semantics and monotonic CPU/GPU timing primitives provide the transport/timing basis. Event allocation/copy cost, overflow rate, timestamp cost, readout latency, and capture perturbation remain unknown until measured with diagnostics enabled and disabled.

### Assumptions and risks

- A bounded per-context ring/channel is sufficient for local retention and does not require durable or remote storage.
- Every event carries severity, typed category, correlation IDs, units, clock/domain, and availability.
- Overflow emits a loss marker/counter; it never blocks rendering indefinitely.

### Validation actions

1. Exercise fatal/recoverable/warning/profiling events across worker, transfer, graph, backend, and callback boundaries.
2. Force queue overflow and verify bounded memory, loss reporting, and continued rendering.
3. Compare diagnostics-disabled/enabled CPU cost, timestamp/readout latency, queue contention, and memory on named workloads.
