# ez-gfx-texture-manager

Backend-neutral texture policy: decode, upload scheduling, and residency
tracking behind backend-defined traits. This crate never touches native
handles and never depends on a backend, the runtime, or `ez-gfx`.

## Architecture

- `texture`: source ingestion (`TextureSource`), validated `DecodedTexture`
  chains, `TextureDecoder` with process-wide custom decoders, RGBA mip
  generation, and lock-free upload telemetry. Codec features: `basis`,
  `ktx2` (both off by default).
- Scheduling: required-prefix token selection (`required_completion_token`),
  FIFO decode batching (`BatchPlan`), the shared transfer budget
  (`SharedTransferPool`), and phase-two fine-mip planning
  (`fine_residency_targets`, `PrefixTransferWork`).
- `pipeline`: the backend-neutral upload state machine (`TexturePipeline`)
  plus generic transitions over `TextureBackendContext`
  (`submit_ready_uploads`, `pump_fine_uploads`, `advance_residency`,
  `reference_required_prefix`, `reclaim_transfer_work`, `observe_ready`,
  `cancel_all_pending`, `drop_device_state`). Outcomes and
  [`UploadFailure`] values let the owning runtime map failures to its own
  errors and emit its own events; the shared pool and the native textures
  stay caller-owned and are passed down, never duplicated.
- Registry: generational `TextureRegistry`/`TextureId` slots, residency
  states, descriptor bindings, and unload events.
- `decode_driver`: manager-owned CPU decode execution (`DecodeDriver`) over
  a lazily built Rayon pool. It owns the worker policy, the result channel,
  the active-job count, and shutdown; it pops pipeline FIFO order,
  reserves transfer bytes atomically with each spawn, and collects terminal
  results back into the pipeline. The runtime holds one driver per context
  and only calls `ensure_started`/`dispatch`/`collect`/`shutdown`/
  `worker_count`.
- `ReclaimableStaging`: uniform eviction interface over heterogeneous staging
  caches, blanket-implemented for every HAL `ReusableStagingPool`.

`ez-gfx` owns `ContextState` (identity, fallback aliasing, upload events,
thread affinity) and drives this policy; backend crates own allocations,
copies, views, and timelines.

## Interfaces

- `REQUIRED_MIPS_FULL: u32 = u32::MAX` requires the whole decoded chain.
  `required_mips` is a plain `u32`: `0` means optional (no frame CPU wait;
  fallback samples until first real coarse residency publishes),
  over-sized values fail terminally once the decoded total is known
  (`resolve_required_mips`), never clamp.
- `TextureBackendTexture` (`MipTransferValues` + `last_transfer_value`) and
  `TextureBackendContext` (create-prefix, region upload, prefix
  reference/publication, descriptor readiness, retirement, destruction,
  completion polling, cancellation) are implemented by each backend crate
- Ordering warning: `MipTransferValues` slices run largest-to-smallest
  (required prefix is the tail); `required_completion_token` takes
  coarse-first submission order (a positive prefix ends at index
  `required - 1`, zero returns no token). Passing one ordering to the other
  gates the wrong mip.

## Synchronization

Submission uploads in two phases. Phase one allocates full storage but
submits at least the coarsest mip (the mechanical single-mip prefix behind
an optional requirement, since backends reject prefix zero), so required
work never blocks behind fine levels in the backend FIFO. The required
decode reservation holds through the required-prefix completion token.
Phase two submits retained fine mips coarsest-first through region updates
under background admission once the required prefix publishes. Descriptor
rewrites wait only for prior submitted frames to drain (graphics completion);
transfer completion stays GPU-gated through required tokens attached to
frame work. Cancellation and device loss release both ledgers and retire
slots to the fallback alias.

## Priorities and batching

Admission order across textures is FIFO within each requirement class, so
publication stays deterministic. Required-positive decodes and submissions
bypass undecoded optional heads so optional work never head-of-line blocks
required work; optional-only workloads still dispatch and submit through
every ordinary nonblocking pump. Required work outranks background work only through
admission: required reservations keep the empty-ledger progress exception
while background reservations never force admission. Each active decode
reserves the 64 MiB per-request maximum inside a 256 MiB window; an
otherwise-valid request exceeding an empty window is admitted alone so
progress cannot deadlock.

## Shared transfer and cache behavior

One `SharedTransferPool` per context is the single admission domain for
texture, geometry, and buffer bytes. Texture reservations release at mip
completion tokens; geometry/buffer staging retires by transfer-queue token.
Staging buckets stay backend-owned behind `ReclaimableStaging`: eviction
trims the largest completed bucket across every pool under one aggregate
ceiling without touching in-flight buckets.

## Lifecycle

`begin_upload` → `mark_submitted`/`mark_mips_submitted` → `poll` emits
`Resident` → `retire` withholds the binding → `release_retired` recycles
it; `cancel_upload`/`unload`/`clear` invalidate generations. Handles are
generational: stale use fails instead of aliasing recycled slots.

## Examples

## Testing

Unit tests cover sentinel resolution, optional-zero semantics, token
selection order, required-bypass submission and dispatch priority, range
completion, batch admission, ledger separation, registry generations,
pool admission bounds, lazy construction, and gated worker dispatch. Integration
tests (`tests/decode_driver.rs`) cover worker policy validation, result
collection, cancellation delivery, panic containment, and shutdown. Integration tests (`tests/textures.rs`,
`tests/texture_transcode.rs` with `basis`/`ktx2` features) cover decode
contracts. Native behavior is verified through the backend matrix (Linux
Vulkan, Windows DX12, macOS Metal).

## Backend adapter responsibilities

Implement `TextureBackendTexture` + `TextureBackendContext` for the
private native types in the backend crate (orphan rule: the trait is
foreign, the types are local). Keep signatures HAL/core-only, preserve
exact token order (coarse-first) and value order (largest-first), and keep
`cfg` gates in the backend/`ez-gfx` layers — this crate stays
platform-agnostic.
