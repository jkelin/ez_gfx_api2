# Sponza KTX2

Showcases compressed KTX2 material textures, bindless per-primitive texture IDs, structured GLB data, and a compute-to-graphics indirect rendering flow.

`main.rs` is the complete renderer. It loads the shared `../shared/assets/sponza.glb`, validates embedded base-color images as KTX2, and loads their mip chains with `load_texture`; `texture_binding` maps each image into the bindless heap, with a fallback texture for missing assignments. It uploads positions, normals, UVs, and primitive records, then uses `render_add_compute` to generate indirect commands and `render_add_graphics` to shade the textured scene with depth, lighting, and alpha discard. Textures are released with `unload_texture` during teardown.

Run:

```text
mise run example-6
```

![Sponza rendered with KTX2 textures](../snapshots/06_sponza_ktx2.png)

Expected result: a textured Sponza atrium with normal-based lighting and transparent material cutouts.
