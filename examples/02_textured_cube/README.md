# Textured Cube

Showcases indexed geometry, an orbit camera, push-constant MVP data, PNG texture upload, and bindless texture selection.

`main.rs` is the complete renderer. It uploads cube positions and indices, creates an indirect draw with `acquire_indirect`, `write_indirect`, and `set_indirect_count`, and loads `cube.png` with `load_texture`. `texture_binding` supplies the bindless texture ID carried in the push constants; `render_add_graphics` records the draw with `DynamicPipelineState`.

Run:

```text
mise run example-2
```

![Rendered textured cube](../snapshots/02_textured_cube.png)

Expected result: a shaded, textured cube viewed from an elevated orbit-camera angle.
