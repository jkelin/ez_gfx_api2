# Textured Cube

Showcases indexed geometry, an orbit camera, push-constant MVP data, PNG texture upload, and bindless texture selection.

`main.rs` retains typed geometry handles and the texture, then acquires a fresh indirect handle after each frame begins. Batched `write_indirect` publishes the draw; `texture_binding`, push constants, and `DynamicPipelineState` complete graphics recording.

Run:

```text
mise run example-2
```

![Rendered textured cube](../snapshots/02_textured_cube.png)

Expected result: a shaded, textured cube viewed from an elevated orbit-camera angle.
