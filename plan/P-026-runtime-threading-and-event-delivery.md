# P-026: Runtime threading and event delivery contract

## Problem

Select the public threading model for contexts/resources and delivery of asynchronous completion, errors, cancellation, and shutdown across Rust and C/C#. P-018 provides workers and a transfer owner, but callers still need deterministic affinity, ordering, ownership, backpressure, and finalizer behavior without a heavyweight async runtime.

## Prompt context

The migrated library has safe Rust and FFI surfaces, CPU workers, a dedicated transfer owner, bounded queues, asynchronous uploads, cancellation, and host-owned window/event loops. It must remain lightweight and must not require Tokio.

## Constraints and acceptance criteria

- Every public type and operation has explicit thread-affinity or concurrent-use rules.
- No callback runs under an internal lock or after its owner is destroyed.
- Event payload ownership/lifetime remains valid through FFI and C# finalization.
- Backpressure, reentrancy, cancellation races, shutdown, and a host that stops pumping events are deterministic.
- No solution requires Tokio or ownership of the host event loop.

## Dependencies

- Depends on P-002 for Rust/FFI ownership and callback representation.
- Depends on P-012 for transfer submission serialization, P-015 for readiness, P-018 for worker production, and P-017 for external event-loop ownership.

## Unresolved questions

- Which handles are thread-affine or concurrently usable?
- Is delivery polling, host-dispatch, dedicated callback thread, or a combination?
- What ordering and overflow policy applies to progress/completion/cancel/failure/shutdown?
- How does registration prevent callback-after-free and finalizer races?

## Candidate solutions

### S-P-026-host-polled-event-queue: Host-polled bounded event queue

#### Approach and integration

Workers publish owned events to a bounded per-context queue. The host calls `poll_events` or `drain_events` from its chosen thread; event payloads remain valid until the poll result is released. Resource operations are thread-safe only where documented, while frame recording remains context-affine.

#### Performance evidence

Event latency depends on host pump frequency; queue memory is explicitly bounded by event count/bytes. Exact latency, throughput, wakeups, and contention are unknown until measured with mixed upload workloads on named hosts.

#### Tradeoffs and failure modes

Maximum host control and straightforward FFI semantics. A host that stops polling delays notifications and can fill the queue; overflow must produce a typed status without blocking workers.

#### Sources

- [Rust `sync_channel`](https://doc.rust-lang.org/std/sync/mpsc/fn.sync_channel.html) — bounded ordered channel semantics.
- [Rust Nomicon FFI](https://doc.rust-lang.org/nomicon/ffi.html) — FFI ownership and callback concerns.

### S-P-026-host-supplied-dispatcher: Host-supplied callback dispatcher

#### Approach and integration

The host registers a dispatcher/executor contract. Runtime workers enqueue event closures or opaque event records to that dispatcher; callbacks execute only when the host invokes the dispatcher. Registration uses an explicit lifetime token and shutdown barrier.

#### Performance evidence

Adds one dispatch hop and host scheduling latency; queue depth, callback latency, and dispatcher overhead are unknown until measured under callback-heavy loads.

#### Tradeoffs and failure modes

Integrates with GUI/game loops and avoids runtime-owned callback threads, but a missing or malicious dispatcher can starve events. FFI must not retain borrowed callback state after unregister.

#### Sources

- [Rust `Send` and `Sync`](https://doc.rust-lang.org/nomicon/send-and-sync.html) — thread-safety contracts.
- [Rust panic/unwind FFI guidance](https://doc.rust-lang.org/nomicon/ffi.html) — callback boundary risks.

### S-P-026-dedicated-callback-thread: Runtime-owned callback thread

#### Approach and integration

A runtime-owned event thread drains a bounded queue and invokes registered callbacks outside internal locks. Callbacks receive owned payloads and an explicit shutdown notification; Rust callers may opt into a receiver instead.

#### Performance evidence

Callback delivery is independent of host pump frequency, but adds a thread, wakeups, and context switches. Exact memory, throughput, and callback latency are unknown until measured.

#### Tradeoffs and failure modes

Reliable delivery for hosts without an event loop, but callback thread affinity may conflict with UI frameworks and creates reentrancy/finalizer hazards. Shutdown must drain or cancel deterministically.

#### Sources

- [Rust `std::thread`](https://doc.rust-lang.org/std/thread/) — thread lifecycle.
- [Rust channel documentation](https://doc.rust-lang.org/std/sync/mpsc/) — receiver ownership and ordering.

## Performance comparison

| Candidate | Relevant dimensions | Constraint fit | Evidence quality | Risks |
|---|---|---|---|---|
| Host-polled bounded event queue | Bounded memory; latency equals pump interval; overhead unknown | High | Rust bounded-channel docs | Host neglect/overflow |
| Host-supplied dispatcher | One dispatch hop; latency/overhead unknown | High | Rust thread-safety/FFI docs | Dispatcher lifetime/starvation |
| Dedicated callback thread | Independent delivery; extra thread/wakeups; unknown latency | Moderate | Rust std docs | UI affinity/reentrancy |

## Selected solution

### Selection

`S-P-026-host-polled-event-queue`: Host-polled bounded event queue.

### Selection rationale

This best fits the lightweight library and host-owned event-loop constraints. It gives Rust and C# callers explicit control over when callbacks/events are observed, keeps payload ownership deterministic, and reuses a bounded queue rather than adding a runtime callback thread. Public affinity rules can keep frame recording context-affine while permitting documented resource queries from other threads.

### Rejected alternatives and reversal conditions

- **`S-P-026-host-supplied-dispatcher`**: Rejected as the primary contract because dispatcher registration and callback lifetime are more complex across C# finalizers and hosts with no stable dispatcher. Reconsider if GUI integrations require callbacks on a specific UI thread.
- **`S-P-026-dedicated-callback-thread`**: Rejected because it adds runtime thread/reentrancy and UI-affinity costs. Reconsider only if polling cannot meet a documented latency requirement.

### Evidence and unknowns

Rust bounded channels provide ordered, bounded transport; event latency, overflow frequency, and callback cost remain unknown for the target workloads.

### Assumptions and risks

- Hosts pump events at a documented cadence and release owned event payloads.
- Queue overflow must be observable and non-blocking; a stopped host may delay notifications.
- Registration/unregistration uses a shutdown barrier to prevent callback-after-free.

### Validation actions

1. Test ordering, overflow, cancellation, shutdown, reentrancy, and a host that stops polling.
2. Exercise Rust and C# callback/payload ownership through finalization and context destruction.
3. Measure event latency, queue bytes, wakeups, contention, and render-thread impact under bounded mixed upload workloads.
