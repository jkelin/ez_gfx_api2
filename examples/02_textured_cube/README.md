# Textured Cube

Showcases indexed geometry, an orbit camera, push-constant MVP data, PNG texture upload, and bindless texture selection.

`main.rs` reads top-to-bottom: it creates owning vertex/index allocations, a texture, a shader, and a persistent counted buffer, then enters a linear frame loop. Each iteration handles input, explicitly configures the swapchain, records textured graphics, and passes the frame and target to `Example::handle_frame`. Resource wrappers release through `Drop`.

Run:

```text
mise run example-2
```

![Rendered textured cube](../snapshots/02_textured_cube.png)

Expected result: a shaded, textured cube viewed from an elevated orbit-camera angle.
