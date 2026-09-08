# P-013: Frame-level upload readiness policy

## Decision

`Surface::begin_frame()` and `Context::begin_frame()` create target-less owning frames. `Frame::configure_swapchain` or `Frame::configure_render_target` attaches one target. All graph, render, texture, geometry, buffer, and counted-buffer recording operates through `&mut Frame`.

At `Frame::finish(self)`, only reachable recorded work is submitted. A prior recording error aborts and is returned unchanged. Submission errors return unchanged and skip presentation; presentation errors return unchanged after successful submission.

Dropped unfinished frames abort. Finish, recording failure, submission failure, presentation failure, and abort all end the recording transaction, release imported buffers for reuse, and keep native backing completion-gated or quarantined internally.

## Retired compatibility design

The former handle-based interface split frame begin, submit, presentation, and transient release across public safe calls. The ownership cutover replaces it rather than retaining aliases.

## C ABI

ABI 32 represents `EzGfxFrame` as an opaque generational `u64`. `ez_gfx_frame_begin` creates surface frames; `ez_gfx_render_target_frame_begin` creates managed-target frames. The validated FFI exposes explicit `ez_gfx_frame_end` and `ez_gfx_frame_abort`; both invalidate on every result, and context destruction aborts descendant frames.

## Validation

Cover recording, submit, and present error identity; implicit abort; persistent-buffer import exclusion and reuse after every terminal path; stale, foreign, and double-completed C frames; and Rust/C/header/XML/export/layout parity.
