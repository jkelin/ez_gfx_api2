# P-013: Frame-level upload readiness policy

## Decision

`Surface::begin_frame()` and `Context::begin_frame()` create target-less owning frames. `Frame::configure_swapchain` or `Frame::configure_render_target` attaches one target. All graph, render, texture, geometry, buffer, and counter-buffer recording operates through `&mut Frame`.

At `Frame::finish(self)`, only reachable recorded work is submitted. A prior recording error aborts and is returned unchanged. Submission errors return unchanged and skip presentation; presentation errors return unchanged after successful submission.

Dropped unfinished frames abort. Finish, recording failure, submission failure, presentation failure, and abort consume every claimed buffer. Native backing remains completion-gated or quarantined internally; wrappers never return to writable state.

## Retired compatibility design

The former handle-based interface split frame begin, submit, presentation, and transient release across public safe calls. The ownership cutover replaces it rather than retaining aliases.

## C ABI

ABI 34 represents `EzGfxFrame` as an opaque generational `u64`. `ez_gfx_frame_begin` creates surface frames; `ez_gfx_render_target_frame_begin` creates managed-target frames. The validated FFI exposes explicit `ez_gfx_frame_end` and `ez_gfx_frame_abort`; both invalidate the frame and its claimed buffer handles on every result.

## Validation

Cover recording, submit, and present error identity; implicit abort; one-frame buffer claim, same-frame reuse, terminal invalidation, stale/foreign/double-completed C frames and buffers; and Rust/C/header/XML/export/layout parity.
