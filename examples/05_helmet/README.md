# Helmet

Showcases loading a GLB model into named position and normal vertex heaps plus structured primitive buffers, then using compute-generated indirect commands for depth-tested graphics.

`main.rs` reads top-to-bottom: setup loads `helmet.glb` and creates owning geometry and shader wrappers. The per-frame closure receives a configured frame, acquires `Buffer` and `CountedBuffer` transients, records compute then graphics, and passes ownership to `Example::handle_frame`.

Run:

```text
mise run example-5
```

![Rendered helmet](../snapshots/05_helmet.png)

Expected result: a centered helmet model shaded from its transformed normals.
