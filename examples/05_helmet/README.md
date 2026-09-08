# Helmet

Showcases loading a GLB model into named position and normal vertex heaps plus structured primitive buffers, then using compute-generated indirect commands for depth-tested graphics.

`main.rs` is the complete renderer. It loads `helmet.glb`, uploads positions and normals with `create_vertex_heap`/`upload_vertices`, fills the global index heap with `create_index_heap`/`upload_indices` returning typed allocation handles, stages primitive records with `acquire_structured`/`write_structured`, and allocates the indirect buffer. `render_add_compute` builds one indexed draw command per primitive; `render_add_graphics` consumes those commands with structured bindings and MVP push constants. The universal artifact is loaded with `load_shader`.

Run:

```text
mise run example-5
```

![Rendered helmet](../snapshots/05_helmet.png)

Expected result: a centered helmet model shaded from its transformed normals.
