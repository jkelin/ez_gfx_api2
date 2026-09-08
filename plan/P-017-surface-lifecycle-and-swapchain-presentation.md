# P-017: Surface lifecycle and swapchain presentation

## Problem

Decide how to handle OS native window interoperability (`raw-window-handle` integration across Win32, Cocoa, X11/Wayland), swapchain resizing and minimization handling, non-blocking presentation synchronization, and enforcement of the presentation swapchain shader write-only policy.

## Prompt context

Source evidence: `TODO.md` ("Document the swapchain shader policy in public-facing docs: swapchain is a presentation target and is shader write-only. Readback remains available through transfer-based screenshots, not shader resource reads", "Support arbitrary sampled and writable access to managed render targets... The swapchain is intentionally shader write-only through `ColorTarget(\"swapchain\", \"write\")`").
Additional evidence: `AGENTS.md` ("Window lifetime, input, resize/minimize observation, and event polling belong to the parent C# application...").

## Constraints and acceptance criteria

- Integrate with standard Rust windowing ecosystems using `raw-window-handle` / `raw-display-handle` (v0.6).
- Handle swapchain recreation during window resize, minimize (zero extent), and DPI scale changes gracefully without driver panics.
- Enforce the presentation swapchain shader write-only rule (readback via transfer-based screenshots only).
- Explicit non-goals: taking ownership of OS window event loops or input handling.

## Dependencies

- Incoming dependency: `P-017` depends on `P-003` (HAL surface & swapchain).
- Outgoing dependency: `P-008` and `P-019` depend on `P-017` for frame presentation and screenshot readback.

## Resolved interface

Zero extent returns `NotReady` from `begin_frame`. Presentation mode remains a validated `Surface` construction option.

## Candidate solutions

### S-P-017-raw-window-handle-surface-and-guarded-swapchain: Rust raw-window-handle 0.6 integration with guarded presentation lifecycle

#### Approach and integration

Construct an owning `Surface` from borrowed host-native handles. The wrapper retains `Rc<ContextInner>` and its resource lease until `Drop`; the host keeps the actual window/display objects alive for that interval. Construction is atomic: native surface creation, device initialization, and initial resize either all succeed or destroy the unpublished raw surface and return the original error without retaining a safe wrapper.

`begin_frame(&Context, &Surface)` returns `NotReady` without recording or acquisition for zero drawable extent. Recording uses `&mut Frame`; `Frame::finish(self)` submits and presents, while `Frame::drop` aborts. Resize/out-of-date handling stays behind the surface seam. Graph validation rejects shader reads from the presentation target; screenshots use transfer readback.

#### Performance evidence

- **Zero-Flicker Resize:** Reusing `old_swapchain` avoids black frame flashes during window resizing (`[INFERENCE]` from Vulkan/DXGI swapchain specs).
- **Minimization CPU Savings:** Returning `NotReady` on 0x0 extent eliminates 100% of GPU rendering and swapchain acquire overhead while a window is minimized or occluded (`[OBSERVED]` in `src/swapchain.odin:210-230`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Parent application retains responsibility for polling window events and notifying the surface of resize events.
- **Failure Modes:** Rapid continuous window resizing on certain X11/Wayland compositors can yield `VK_SUBOPTIMAL_KHR` or `VK_ERROR_OUT_OF_DATE_KHR`, requiring deferred swapchain rebuild retry loops.

#### Sources

- [Rust raw-window-handle Specification](https://docs.rs/raw-window-handle/latest/raw_window_handle/) — standard window handle abstraction.
- [Vulkan Swapchain Recreation Best Practices](https://docs.vulkan.org/guide/latest/swapchain_recreation.html) — handling resize, minimize, and presentation modes.
- `F:/Projects/oss/ez_gfx_api/src/window.odin` & `src/swapchain.odin` — original surface and swapchain code.
- `F:/Projects/oss/ez_gfx_api/AGENTS.md` — guidelines on parent-owned window lifecycle.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — swapchain write-only policy requirements.

### S-P-017-winit-window-integrated-surface: Embedded window creation and event loop ownership

#### Approach and integration

Bundle `winit` directly into `ez_gfx_api` so calling `ez_gfx_create_window()` creates and owns the OS window and drives the event loop.

#### Performance evidence

- **Convenience vs Interop:** High convenience for standalone toy demos, but completely breaks embedding in external engines, C# GUI applications, or native UI frameworks.

#### Tradeoffs and failure modes

- **Tradeoffs:** Violates explicit non-goals and architectural rules in `AGENTS.md` regarding external parent window ownership.
- **Failure Modes:** Severe event loop conflicts when host applications (such as WPF/WinForms, GLFW, or custom game engines) manage the main loop.

#### Sources

- `F:/Projects/oss/ez_gfx_api/AGENTS.md` — explicit constraint that window ownership belongs to the parent app.
- [raw-window-handle crate documentation](https://docs.rs/raw-window-handle/latest/raw_window_handle/) — standard native-handle traits.

### S-P-017-incumbent-c-platform-descriptor: Incumbent C platform pointer descriptor table

#### Approach and integration

Maintain original Odin descriptor: `EzGfxSurfaceDesc` with raw `void* window`, `void* display`, and an enum `EzGfxSurfacePlatform (Win32, GLFW)`.

#### Performance evidence

- **Ecosystem Incompatibility:** Lacks support for Rust windowing libraries (`winit`, `sdl2`, `egui-winit`) without manual unsafe pointer extraction.

#### Tradeoffs and failure modes

- **Tradeoffs:** Matches C ABI directly, but is unidiomatic and fragile for Rust callers.
- **Failure Modes:** Unsafe dangling pointer dereferences if parent window is closed before surface destruction.

#### Sources

- `F:/Projects/oss/ez_gfx_api/bindings/bindings.xml` — incumbent surface descriptor definition.
- [Microsoft DXGI surface documentation](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/dxgi-overviews) — native swapchain/surface model.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| --------- | ------------------------------- | -------------- | ---------------- | ----- |
| raw-window-handle 0.6 integration with guarded presentation | Standard Rust ecosystem fit, zero-flicker resize, safe minimization, strict write-only enforcement | High | Official docs & direct source inspection | Linux compositor out-of-date swapchain retries |
| Embedded winit window and event loop ownership | Standalone demo convenience, but blocks external engine integration | Low | Direct architectural guidelines | Violates parent window ownership constraints |
| Incumbent C platform pointer descriptor table | Direct C ABI match, but fragile and unidiomatic in Rust | Moderate | Direct source code inspection | Unsafe raw pointer lifetimes |

## Selected solution

### Selection

`S-P-017-raw-window-handle-surface-and-guarded-swapchain`: Rust raw-window-handle 0.6 integration with guarded presentation lifecycle.

### Selection rationale

The selected surface lifecycle:
1. accepts standard borrowed native handles while an owning `Surface` retains its context/resource lease;
2. leaves OS window, event-loop, resize observation, and input ownership with the host;
3. destroys the unpublished raw surface after any native-creation, initialization, or initial-resize failure;
4. returns `NotReady` from `begin_frame` for zero drawable extent; and
5. presents only through `Frame::finish(self)`, preserving the exact submit or present error.

### Rejected alternatives

- **`S-P-017-winit-window-integrated-surface`**: Hard-constraint failure: taking internal ownership of the window and event loop violates the explicit architectural rule that window lifetime and message polling belong to the host application.
- **`S-P-017-incumbent-c-platform-descriptor`**: Rejected because raw `void*` pointer descriptors are unidiomatic and unsafe in Rust, preventing clean integration with modern Rust window crates.

### Evidence summary

Minimization detection eliminates rendering and swapchain acquire work while the window has a 0x0 extent (`[OBSERVED]` in `src/swapchain.odin:210-230`), while swapchain recreation and native handle interoperability are covered by official API documentation. Exact present latency is unknown until measured.

### Key assumptions

- The host keeps native window/display objects alive while `Surface` exists and forwards observed resize state.
- The underlying display subsystem supports the selected backend's presentation mechanism.

### Risks and mitigations

- **Risk:** partial surface construction leaks a native object or publishes a half-initialized wrapper.
- **Mitigation:** inject failure after native creation, initialization, and initial resize; verify cleanup and the original error.

### Validation actions

1. Test construction rollback, wrapper drop-order permutations, resize, minimization, restoration, and DPI changes.
2. Verify exact `Frame::finish` errors and rejection of presentation-target shader reads.

### Native surface color evidence

The frame graph advertises BGRA8 sRGB. Metal layer creation/acquisition and graphics pipeline attachments now lower that same format; sampled-texture formats remain independent. Captured RGBA bytes preserve hardware sRGB encoding, with only BGRA channel reordering and no second gamma conversion.

Apple M2 Pro headless render/readback tests verify a linear 0.1 clear becomes `[89, 89, 89, 255]`, not the former linear `[26, 26, 26, 255]`, alongside separated color/depth passes and compressed midtone sampling. This corrects visible Metal output; it does not establish resize/minimize or every presentation-mode requirement.
