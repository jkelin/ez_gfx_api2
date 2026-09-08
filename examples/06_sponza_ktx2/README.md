# Sponza KTX2

Showcases compressed KTX2 material textures, bindless per-primitive texture IDs, structured GLB data, and a compute-to-graphics indirect rendering flow.

`main.rs` reads top-to-bottom through the shared host setup closure: it loads `sponza.glb`, validates embedded KTX2 images, and creates owning texture, geometry, and shader wrappers. The returned per-frame closure calls `begin_frame`, acquires structured and indirect buffers through `&mut Frame`, records compute-generated commands and textured graphics, then returns the owning `Frame` to `Example::handle_frame`. Missing materials use the retained fallback texture; every persistent wrapper releases through `Drop`.

Run:

```text
mise run example-6
```

![Sponza rendered with KTX2 textures](../snapshots/06_sponza_ktx2.png)

Expected result: a textured Sponza atrium with normal-based lighting and transparent material cutouts.
