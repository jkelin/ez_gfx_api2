# Helmet

Showcases loading a GLB model into structured position, normal, and primitive buffers, then using compute-generated indirect commands for depth-tested graphics.

`main.rs` is the complete renderer. It loads `helmet.glb`, uploads mesh data with `acquire_structured`/`write_structured` and `create_index_heap`/`upload_indices`, and allocates the indirect buffer. `render_add_compute` builds one indexed draw command per primitive; `render_add_graphics` consumes those commands with structured bindings and MVP push constants. The universal artifact is loaded with `load_shader`.

Run:

```text
mise run example-5
```

![Rendered helmet](../snapshots/05_helmet.png)

Expected result: a centered helmet model shaded from its transformed normals.
