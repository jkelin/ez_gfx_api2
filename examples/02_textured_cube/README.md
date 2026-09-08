# Textured Cube

Showcases indexed geometry, an orbit camera, push-constant MVP data, PNG texture upload, and bindless texture selection.

`main.rs` reads top-to-bottom through one setup closure: it creates owning vertex/index allocations, a texture, and a shader, then returns concise recording logic. The shared host supplies each configured frame; the closure writes a `CountedBuffer`, records textured graphics, and passes ownership to `Example::handle_frame`. Resource wrappers release through `Drop`.

Run:

```text
mise run example-2
```

![Rendered textured cube](../snapshots/02_textured_cube.png)

Expected result: a shaded, textured cube viewed from an elevated orbit-camera angle.
