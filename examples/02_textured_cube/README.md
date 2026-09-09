# Textured Cube

Showcases indexed geometry, an orbit camera, push-constant MVP data, PNG texture upload, and bindless texture selection.

`main.rs` reads top-to-bottom: it creates owning vertex/index allocations, a texture, and a shader, then enters a linear frame loop. Each iteration handles input, acquires a one-frame counter buffer for its indexed-indirect draw, explicitly configures the swapchain, records textured graphics, and passes the frame and target to `Example::handle_frame`. Resource wrappers release through `Drop`.

Run:

```text
mise run example-2
```

![Rendered textured cube](../snapshots/02_textured_cube.png)

Expected result: a shaded, textured cube viewed from an elevated orbit-camera angle.
