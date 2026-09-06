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

## Unresolved questions

- How should minimized window state (`0x0` dimensions) signal `NotReady` back to callers across safe Rust and C ABI boundaries?
- What swapchain present modes (`Fifo`, `Mailbox`, `Immediate`) should be exposed as defaults?

## Candidate solutions

### S-P-017-raw-window-handle-surface-and-guarded-swapchain: Rust raw-window-handle 0.6 integration with guarded presentation lifecycle

#### Approach and integration

Implement surface creation in Rust accepting types implementing `raw_window_handle::HasDisplayHandle + raw_window_handle::HasWindowHandle` (with C FFI adapters accepting raw `HWND`/`NSWindow`/`wl_surface` pointers). Surface management:
1. **Minimization & Resizing:** When framebuffer dimensions query as `0x0` (minimized window), `begin_render()` returns `EzGfxResult::NotReady` immediately, skipping GPU command recording and swapchain acquire calls. On size change, swapchain recreation uses `old_swapchain` recycling to prevent window flickering.
2. **Swapchain Shader Write-Only Enforcement:** Graph validation scans all shader target declarations; if any node references `"swapchain"` with read access (`Sampled`, `RWTexture2D` read), compilation fast-fails with `InvalidShaderResourceUsage`. Screenshot readbacks use a transfer blit (`vkCmdCopyImage`/`CopyResource`) from swapchain image to a host-visible staging buffer.
3. **Presentation Modes:** Negotiate `Mailbox` (uncapped low latency) or `Fifo` (vsync locked) based on caller configuration and hardware support.

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

`S-P-017-raw-window-handle-surface-and-guarded-swapchain` directly adheres to all project windowing constraints and TODO policies:
1. It implements standard Rust window interoperability via `raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle` (v0.6) for seamless integration with `winit`, `sdl2`, or custom windowing layers, while providing C FFI wrappers for raw HWND/NSWindow pointers.
2. It respects parent application window ownership (as required by `AGENTS.md`), avoiding internal event loop hijacking.
3. It detects 0x0 minimization state and returns `EzGfxResult::NotReady`, preventing unnecessary GPU rendering work.
4. It enforces the swapchain shader write-only policy during graph compilation, directing readback requests to transfer screenshot commands.

### Rejected alternatives

- **`S-P-017-winit-window-integrated-surface`**: Hard-constraint failure: taking internal ownership of the window and event loop violates the explicit architectural rule that window lifetime and message polling belong to the host application.
- **`S-P-017-incumbent-c-platform-descriptor`**: Rejected because raw `void*` pointer descriptors are unidiomatic and unsafe in Rust, preventing clean integration with modern Rust window crates.

### Evidence summary

Minimization detection eliminates rendering and swapchain acquire work while the window has a 0x0 extent (`[OBSERVED]` in `src/swapchain.odin:210-230`), while swapchain recreation and native handle interoperability are covered by official API documentation. Exact present latency is unknown until measured.

### Key assumptions

- Parent applications handle OS window event polling and forward resize events to the surface API.
- The underlying display subsystem supports `VK_KHR_swapchain` / `IDXGISwapChain` / `CAMetalLayer`.

### Risks and mitigations

- **Risk:** Continuous window dragging on Linux compositors triggers out-of-date swapchain errors.
- **Mitigation:** Handle `VK_SUBOPTIMAL_KHR` or `VK_ERROR_OUT_OF_DATE_KHR` with guarded deferred swapchain recreation.

### Validation actions

1. Test window resizing, minimization, and restoration across all 6 examples with zero GPU validation layer warnings.
2. Verify that attempting to bind the swapchain as a shader sampled read produces an explicit `InvalidShaderResourceUsage` error during graph compilation.

### Native surface color evidence

The frame graph advertises BGRA8 sRGB. Metal layer creation/acquisition and graphics pipeline attachments now lower that same format; sampled-texture formats remain independent. Captured RGBA bytes preserve hardware sRGB encoding, with only BGRA channel reordering and no second gamma conversion.

Apple M2 Pro headless render/readback tests verify a linear 0.1 clear becomes `[89, 89, 89, 255]`, not the former linear `[26, 26, 26, 255]`, alongside separated color/depth passes and compressed midtone sampling. This corrects visible Metal output; it does not establish resize/minimize or every presentation-mode requirement.
