# Triangle

Showcases the smallest safe `ez-gfx` graphics path: structured vertex data, an index heap, one indexed-indirect draw, and a compiled shader artifact.

`main.rs` is the complete renderer. It uploads positions with `acquire_structured`/`write_structured`, creates and fills an index heap with `create_index_heap`/`upload_indices`, then creates an indirect buffer with `acquire_indirect`, `write_indirect`, and `set_indirect_count`. It loads the `.ezgfx` artifact with `load_shader` and records the draw with `render_add_graphics`.

Run:

```text
mise run example-1
```

![Rendered triangle](../snapshots/01_triangle.png)

Expected result: a solid red triangle centered in the 640×480 presentation surface.
