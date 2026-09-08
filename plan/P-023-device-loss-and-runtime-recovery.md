# P-023: Device loss and runtime recovery

## Problem

Define behavior when Vulkan, DX12, or Metal reports device loss/removal. Pending jobs, callbacks, handles, caches, and GPU-owned state need one safe transition.

## Prompt context

The Rust runtime owns devices, queues, allocations, graphs, workers, transfers, swapchains, and generation-checked Rust/C handles.

## Constraints and acceptance criteria

- Loss must not deadlock waits, reuse unfinished memory, unwind through FFI, or strand callbacks.
- Public resources and operations must enter deterministic states.
- Teardown/recovery must order workers, transfers, surfaces, caches, allocator, and native objects safely.
- Old handles and completion tokens must never appear valid after loss.
- Portable errors may retain bounded native diagnostics.

## Dependencies

- P-002, P-003, P-004, P-007, P-012, P-017, P-018, and P-020.

## Unresolved questions

- Is loss terminal for one runtime, or does the library recreate a new device generation?
- Which CPU descriptions/payloads survive?
- How are pending jobs, unreachable tokens, leases, and callbacks completed?

## Candidate solutions

### S-P-023-terminal-instance: Poison the runtime; host recreates

#### Approach and integration

Atomically transition `Running -> Lost` once. Reject new work, fail every unsignaled token, cancel leases and queued payloads, deliver one terminal result per callback, preserve bounded diagnostics, and destroy CPU/native wrappers without waiting for impossible GPU progress. Handles remain recognizable but permanently invalid. The host creates a fresh runtime and resources.

#### Constraint applicability

This is the smallest common Rust/FFI contract and requires no retained asset bytes. It normalizes Vulkan errors, D3D12 removal, and Metal command-buffer failures without claiming a lost device can resume.

#### Performance evidence

Steady-state cost is an unmeasured state check plus optional diagnostics. Microsoft's DRED specification reports 2–5% loss for automatic breadcrumbs on typical AAA D3D12 engines; rich diagnostics therefore cannot be assumed free. Measure disabled/enabled overhead, detection-to-notification, fan-out, and teardown under injected loss.

#### Tradeoffs and failure modes

Simple and deterministic, but interrupts the application. Double callback completion, fence waits during teardown, or treating lost-device memory as completed are disqualifiers.

#### Sources

- [Vulkan lost device](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-lost-device) defines post-loss behavior.
- [D3D12 `GetDeviceRemovedReason`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getdeviceremovedreason) retrieves removal cause.
- [D3D12 DRED](https://microsoft.github.io/DirectX-Specs/d3d/DeviceRemovedExtendedData.html) documents diagnostics and its contextual 2–5% breadcrumb cost.
- [WebGPU device loss](https://www.w3.org/TR/webgpu/#device-lost) is mature cross-backend evidence for permanent old-device loss.

### S-P-023-managed-recreation: Rebuild from retained descriptions

#### Approach and integration

Terminate the failed native device, increment a device epoch, enumerate/select again, then rebuild artifacts, pipelines, resources, descriptors, uploads, and surfaces from retained descriptions or host reload callbacks. Old graph work and callbacks fail rather than replay; new objects publish only after readiness.

#### Constraint applicability

This provides continuity only if resource recreation ownership and FFI callback lifetimes are explicit. Backend changes require portable artifacts, formats, and intent.

#### Performance evidence

No general recovery timing exists; it scales with device creation, pipeline count, retained bytes, and reuploads. Measure retained CPU memory, bytes reuploaded, pipeline rebuild/cache hits, and time to first valid frame on named backends.

#### Tradeoffs and failure modes

Central recovery deeply couples every resource module and may duplicate host asset memory. Missing callbacks, unavailable adapters, invalid caches, and format changes can abort recovery. Reusing old handles/timelines or replaying ambiguous mutations is disqualifying; terminal failure remains fallback.

#### Sources

- [DXGI device-lost handling](https://learn.microsoft.com/en-us/windows/uwp/gaming/handling-device-lost-scenarios) demonstrates releasing and recreating device-dependent resources.
- [Vulkan lost device](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-lost-device) requires a new logical device rather than revival.
- [Metal command-buffer status](https://developer.apple.com/documentation/metal/mtlcommandbuffer/status) exposes terminal command outcomes.

## Performance comparison

| Rank | Candidate | Hard-constraint result | Normalized evidence | Reliability/memory cost |
| ---- | --------- | ---------------------- | ------------------- | ----------------------- |
| 1 | Terminal instance; host recreates | Passes deterministic errors, stale-handle safety, FFI containment, and bounded ownership | Backend loss semantics support permanent old-device failure; state-check and teardown costs unknown | Lowest retained memory and implementation coupling; visible interruption |
| 2 | Managed recreation | Passes only with durable payload/recreation ownership and epoch invalidation | New-device recreation is supported; recovery time and retained-memory cost are wholly workload-dependent | Highest cross-module coupling and duplicated/reloadable asset state |

Managed recreation has no comparable performance evidence and depends on requirements the prompt does not state. Terminal failure is the necessary baseline even if later recovery is added.

## Selected solution

**Select S-P-023-terminal-instance: poison the lost runtime and require host recreation.**

The first fatal result atomically transitions the runtime to `Lost`. New work fails; unsignaled tokens, leases, queued transfers, and callbacks complete exactly once with typed loss/cancellation; no teardown path waits for GPU progress. All existing handles remain permanently invalid, while bounded portable and backend diagnostics survive until destruction. The host explicitly creates a new runtime and reloads resources.

Reject managed recreation because it requires retained asset bytes or reload callbacks, cross-module replay semantics, and transparent handle replacement not requested by the migration. Reconsider only if continuity becomes an explicit requirement and the public API adopts durable resource descriptions, epoch handles, bounded retained memory, and defined failure fallback.

Evidence: Vulkan and WebGPU treat the old device as permanently lost; D3D12 and Metal expose terminal removal/command outcomes. DRED reports 2–5% loss for automatic breadcrumbs on typical AAA D3D12 engines, so enhanced diagnostics are optional and measured. State-check, notification, and teardown costs remain unknown.

Assumptions: applications can rebuild state after receiving loss. Risks are double completion, callback races, deadlock during shutdown, missing diagnostics, and mistaken completion of GPU-owned memory. Inject loss at acquire, submit, present, transfer, and idle paths; assert bounded completion and zero GPU waits; fuzz concurrent calls/destruction; verify stale Rust/FFI handles; measure disabled/enabled diagnostic overhead, detection-to-notification, fan-out, retained bytes, and teardown time on each backend.
