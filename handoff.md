# Handoff — texture workstreams (in-flight snapshot)

Base: `03345a0` (`feat(texture): close audited parity gaps and expose decode workers in C ABI`).
Status: worktree UNCOMMITTED at write time; host `cargo check -p ez-gfx --all-targets` clean.
No push.

## 1. Managed render-target interfaces

### Landed: probing + allocation/lifecycle (prior turns, in tree)
- Native single-mip sampled `create_render_target` + `probe_target_formats` on
  Vulkan (`crates/ez-gfx-backend-vulkan/src/texture.rs` ~463-630),
  DX12 (`crates/ez-gfx-backend-dx12/src/native/texture.rs` ~492-575, incl. per-RT
  single-entry RTV heap), Metal (`crates/ez-gfx-backend-metal/src/native/texture.rs`
  ~318-450, view built directly with `pixel_format`).
- Safe lifecycle `crates/ez-gfx/src/state/render_target.rs` (new, ~280 lines):
  `create_render_target` (~47), `destroy_render_target` (~148),
  `render_target_format/extent/clear` (~166-230), `destroy_all_render_targets`
  teardown (~223), top-down binding free-list (~248), `render_target_clear_color`
  helper (~16, `None` clears map to transparent black).
- RTs stored under `RenderTargetHandle` in `ContextState.render_targets`
  (`crates/ez-gfx/src/state/mod.rs` ~253-255); never enter upload/update/readback paths.
- `begin_render_target` (`crates/ez-gfx/src/state/context.rs` ~596): color-only gate,
  sets `frame_render_target`, clears `active_surface`.
- Residency settled: eviction is descriptor/view narrowing only on all three
  backends; full image allocation retained; memory retires on unload, never on
  demotion (`docs/textures.md` verification bullet).

### In progress: pass attachment + per-target clear (this slice, UNFINISHED)
- Backend actions reshaped: `BeginPass { pass, colors: Vec<PassAttachment> }`
  (`PassAttachment { resource, clear: [f32;4] }`), `NativeFrameResource::RenderTarget`
  added, barrier arms shared with Texture, `uses_surface` attachment-precise,
  area/draw validation per-attachment extent, depth+RT and samples!=1 rejected,
  surface clear preserved via `SURFACE_DEFAULT_CLEAR` (`crates/ez-gfx-hal/src/lib.rs` ~1128).
  - Vulkan: `backend-vulkan/src/lib.rs` ~194-235, `frame.rs` `begin_render_pass` ~156,
    preflight ~972/~1033, depth guard ~1160.
  - DX12: `backend-dx12/src/native/mod.rs` ~159-200, `frame.rs` `begin_pass` ~571,
    `pass_target` tracking for EndPass discard, preflight ~48/~92.
  - Metal: `backend-metal/src/native/mod.rs` ~147-188, `frame.rs` `begin_pass` ~107,
    validation ~650/~773, record dispatch ~1131. NEVER COMPILED (see risks).
- State: `intern_render_target_resource` (no initial state; first barrier from
  undefined), `graphics_node` RT override with depth→`Unsupported`
  (`state/frame/mod.rs` ~344/~395), `frame_submit` skips present for target-only
  frames (~716), executors resolve colors + RT extent fallback
  (`frame/vulkan.rs` ~309/~448, `frame/dx12.rs` ~195/~351, `frame/metal.rs` ~413/~474).
- Native GPU proof tests (clear→readback exact bytes + rejection pins):
  Vulkan `texture_tests.rs` ~773 (GREEN this turn, 3/3 with allocation+probe),
  DX12 `native/texture_tests.rs` ~849 (GREEN earlier this turn, 3/3),
  Metal `native/texture_tests.rs` ~735 (NOT RUN — needs Mac).
- Safe tests `state/tests.rs` ~400 (validation, prior) and ~467
  (`begin_render_target_rejects_foreign_handles`: phantom + wrong-kind texture
  handle → `InvalidContext`; COMPILE-CHECKED ONLY, not run).

### Remains (render targets)
- FFI/ABI: no C creation/probe/clear/destroy/begin (deliberately deferred).
- Heap unification: interim top-down RT binding allocator; unify with texture slots.
- Depth/storage/MSAA RTs, resize/history, sampling an RT in a later pass, screenshot-save.
- `frame_begin` does not clear `frame_render_target`; `destroy_render_target` does
  not clear a bound override (dangling target risk).
- Safe depth+RT (`graphics_node` `Unsupported`) has no headless test (needs shader);
  pinned natively only.

## 2. Texture device-loss handling
Open. Root `TODO.md` P0-adjacent item: terminal device-loss for queued CPU jobs,
transfers, staging leases, pending handles/waits; cross-thread context destruction
under Windows loader lock (P-023, P-026). No work started this session.

## 3. Adapter texture limits/diagnostics
Open. `TODO.md` P1: deterministic adapter enumeration, explicit stable adapter
selection through Rust and FFI, admitted limits/formats, rejection diagnostics
(P-003, P-022). Capability probing itself (compression union, `probe_target_formats`)
is implemented per backend; selection/diagnostics API is not.

## 4. Remaining Metal texture perf workloads
Open. Atlas/staging wall-time benchmarks on Apple hardware are listed as remaining
evidence (`docs/textures.md` "Measured workloads" + "Selected scope versus remaining
work"); Windows timings must not be extrapolated. Requires Mac runs.

## 5. Pending follow-up commit set
In tree, uncommitted, mixed authorship:
- Worker cap: `ez-gfx-assets/src/lib.rs` (`MAX_CPU_POOL_THREADS`, `CpuPool`) +
  `state/mod.rs` admission-cap validation + topology test (peer lane; builds clean).
- P-012 nonblocking: `plan/P-012-*.md` + `TODO.md` reworded toward a dedicated
  transition/acquire queue; Metal nonblocking done, Vulkan/DX12 targeted waits remain.
- Overlap hardening: prior-turn texture proofs (see `docs/textures.md` evidence).

## Worktree file list (`git status --short` at write time)
31 modified + 1 untracked (`crates/ez-gfx/src/state/render_target.rs`).
Touches: `TODO.md`, `docs/textures.md`, `plan/P-012-*.md`, `plan/P-015-*.md`,
`ez-gfx-assets/src/lib.rs`, `ez-gfx-hal/src/lib.rs`,
`ez-gfx-backend-{vulkan,dx12,metal}/Cargo.toml`,
`ez-gfx-backend-vulkan/src/{frame.rs,lib.rs,texture.rs,texture_tests.rs}`,
`ez-gfx-backend-dx12/src/native/{frame.rs,mod.rs,texture.rs,texture_tests.rs}`,
`ez-gfx-backend-metal/src/native/{device.rs,frame.rs,mod.rs,texture.rs,texture_tests.rs,transfer.rs}`,
`ez-gfx/src/state/{context.rs,mod.rs,native.rs,tests.rs}`,
`ez-gfx/src/state/frame/{dx12.rs,metal.rs,mod.rs,vulkan.rs}`.
(DX12/Metal backend + assets + plan edits predate this slice; verified untouched
except DX12 `texture.rs`/`mod.rs` RTV work and import repairs.)

## Test evidence so far (this session)
- Vulkan native RT: 3/3 green on RTX 3080 (probe, allocation, clear→readback exact
  red bytes + Texture-as-color/depth/samples pins). Reran in `target/vk-rt`.
- DX12 native RT: 3/3 green on RTX 3080 (probe, allocation, clear→readback exact
  green bytes + depth pin). Reran in `target/dx-rt`.
- `cargo check -p ez-gfx --all-targets`: clean (covers new safe test compile).
- Prior (kept): `ez-gfx` lib 24/24, example smoke 32/32, shader artifacts 5/5,
  host BC pixel suites 2/2 (Vulkan+DX12, ASTC `Unsupported` on both).
- NOT run: Metal anything (remote pending), new safe foreign-handle test, DX12/Vulkan
  suites after final import repairs (check-only).

## Known risks
- Metal backend + `state/frame/metal.rs` do not compile-check on Windows (Apple-gated);
  first Mac `cargo test -p ez-gfx-backend-metal --lib` may surface type errors
  (esp. `&*texture.texture` coercion, `Target` enum paths, `width` field visibility).
- `cargo fmt` was run mid-slice on some files; several edit-tool operations dropped
  adjacent lines (all caught via immediate `git diff` audit + restore, but the
  BeginPass/barrier arms in all three executors deserve reviewer eyes).
- `target/debug` Vulkan test exe link-locks (stale test process); use a fresh
  `--target-dir` for backend test runs.
- `begin_render` clears the RT override but `frame_begin` does not; destroyed RTs
  leave a dangling `frame_render_target` until the next begin call.
