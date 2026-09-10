# Presentation

`Frame::configure_swapchain(size, format, presentation_mode)` selects presentation behavior for that frame. `Surface::presentation_modes()` returns the modes available for the initialized surface; Vulkan support is surface-dependent and therefore cannot be inferred from context or adapter options. `Surface::resolve_presentation_mode(requested)` exposes the same deterministic resolution used during frame configuration.

Modes:

- `Fifo`: ordered, tear-free presentation at vertical blank.
- `Mailbox`: newest pending frame at vertical blank, tear-free.
- `Immediate`: no vertical-blank wait; tearing is permitted.
- `Relaxed`: FIFO while on time, with an immediate late presentation after a missed blank.
- `Paced`: latest-ready queued frame at vertical blank, tear-free. This is Vulkan FIFO-latest-ready behavior, not ordinary FIFO.

Every real presentation surface must report `Fifo`. A Vulkan headless surface reports only logical `Fifo`; it has no presentable native swapchain, so finishing a surface frame remains `NotReady`. Unsupported requests resolve in this order:

- `Fifo` → `Fifo`
- `Mailbox` → `Mailbox`, `Paced`, `Fifo`
- `Immediate` → `Immediate`, `Mailbox`, `Paced`, `Fifo`
- `Relaxed` → `Relaxed`, `Fifo`
- `Paced` → `Paced`, `Mailbox`, `Fifo`

Vulkan reports the surface's `VkPresentModeKHR` set directly. `Paced` is advertised only when the FIFO-latest-ready device extension and feature were enabled and the surface reports that mode. Changing the effective mode recreates the swapchain.

DX12 reports `Fifo` and `Paced`, plus `Immediate` when DXGI tearing support is available. `Fifo` uses sync interval 1. `Paced` uses sync interval 0 without tearing. `Immediate` uses sync interval 0 with `DXGI_PRESENT_ALLOW_TEARING`; the swapchain is created with the matching allow-tearing flag.

Metal reports `Fifo` and `Immediate`. The backend applies `CAMetalLayer.displaySyncEnabled` before acquiring the drawable and restores the host layer's original value when the surface is destroyed.

ABI 40 exposes `EzGfxPresentationMode`, `EzGfxPresentationModes`, `ez_gfx_surface_get_presentation_modes`, and the presentation-mode argument on `ez_gfx_frame_begin`. Mode bits use the enum values as bit indices. Invalid encoded modes return `EzGfxResult_InvalidArgument`; unsupported valid modes use the fallback order above.
