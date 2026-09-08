# P-013: Frame-level upload readiness policy

## Problem

Define one predictable frame-start upload policy without coupling asset readiness to render-graph node order, batching, or resource discovery.

## Decision

Add an immutable `FrameBeginConfig` shared by `frame_begin`, `begin_render`, and `begin_render_target`.

- `wait_for_geometry_uploads` defaults to `true` and covers named vertex heaps plus the global index heap. When disabled, applications must use lossless `DeviceReady` events to schedule visible geometry and must not draw or overwrite allocations whose transfers remain pending.
- `TextureMipWait` defaults to `Coarsest`, preserving current initial readiness. `None` disables frame submission texture-upload waits only; applications gate first visibility through `DeviceReady` or a fallback and finer residency through `set_texture_residency`/`texture_residency`. `ThroughLevel(level)` requires readiness through the selected minimum mip: mip 0 is finest, higher numeric levels are coarser and upload first, and levels beyond a texture's chain resolve to that texture's coarsest level.

At submission, collect only reachable, submitted work selected by the geometry and texture policies and apply at most one aggregate completion prefix per enabled active transfer timeline domain to the whole frame. Uploads admitted during recording join that frozen submission snapshot. Asset readiness never creates waits on particular textures, allocations, draws, graph resources, or graph nodes.

`SourceStaged` remains the caller-memory ownership transition. `DeviceReady` remains the application visibility transition. `TextureMipWait::None` suppresses only the frame wait: polling, completion-gated descriptor publication, staging recycling, progressive uploads, events, and device-loss cleanup continue. Transfer completion tokens also remain internal inputs for removal retirement. Generic graph ordering, hazard barriers, render-target transitions, and frame-local resource synchronization remain unchanged.

## Superseded design

The former selected design attached per-allocation, per-texture, and per-draw completion tokens during graph compilation. That design is retired because named heap imports, indirect draws, graph reordering, and transfer batching make exact asset-to-node dependency discovery complex and fragile.

Current code still attaches heap-maximum and texture readiness to first resource access. Heap-maximum waiting is safe but conservative; texture uploads already submit higher/coarser mips first and publish progressively. The new frame policy replaces asset-specific graph readiness without changing those historical implementation facts.

## Dependencies

P-013 depends on P-003 for comparable timeline domains, P-012 for submitted transfer prefixes, and P-015 for coarse-first mip completion and descriptor publication. P-016 consumes the selected frame policy. P-011 range retirement and P-012's Vulkan/DX12 transition/acquire queue remain independent work.

## Validation

Safe Rust validates `TextureMipWait`, including `ThroughLevel` ranges, and snapshots one immutable config per frame. The C ABI validates canonical booleans, texture-policy tags and payloads, reserved fields, layouts, and defaults before delegating to the safe implementation.

Cover default behavior, manual geometry scheduling through `wait_for_geometry_uploads = false`, manual texture scheduling through `TextureMipWait::None`, `DeviceReady`/fallback first visibility, finer-residency polling, mixed mip counts, `Coarsest`, middle levels, level 0, uploads admitted during recording, unreachable or failed transfer work, delayed descriptor publication, continued polling/recycling/progressive upload under `None`, device loss, and all three begin paths. Keep Rust/C/header/XML/export/layout parity and canonical docs synchronized. Vulkan, Direct3D 12, and Metal evidence must prove at most one frame-level aggregate prefix per enabled active transfer domain and no asset-upload wait actions tied to graph nodes or resources.
