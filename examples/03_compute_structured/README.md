# Compute Structured

Showcases a GLB scene rendered from structured buffers, with a compute pass generating indexed-indirect draw commands for the graphics pass.

`main.rs` is the complete renderer. It loads the shared `../shared/assets/sponza.glb`, uploads positions, normals, primitive records, and indices, and allocates the indirect buffer. Each frame, `render_add_compute` dispatches one workgroup per primitive to write draw commands; `render_add_graphics` then consumes that indirect buffer with the same structured bindings and MVP push constants. The artifact is loaded with `load_shader`.

Run:

```text
mise run example-3
```

![Structured compute scene](../snapshots/03_compute_structured.png)

Expected result: the normalized Sponza scene rendered with normal-derived colors and depth testing.
