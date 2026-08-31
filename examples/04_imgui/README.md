# Dear ImGui

Showcases Dear ImGui integration: font-atlas upload, CPU draw-data flattening, resizable structured buffers, bindless texture sampling, clip rectangles, and per-command indirect draws.

`main.rs` is the complete renderer. It loads the font atlas with `load_texture` and resolves its `texture_binding`. ImGui draw lists are copied into structured buffers using `write_structured`; buffers grow with new `acquire_structured` allocations when needed. It writes one `DrawIndexedCommand` per UI command with `write_indirect`, sets the count with `set_indirect_count`, and records graphics with `render_add_graphics`. Each command carries its clip rectangle through the vertex shader, and the fragment shader discards pixels outside it.

Run:

```text
mise run example-4
```

![Dear ImGui demo](../snapshots/04_imgui.png)

Expected result: the Dear ImGui demo window and controls rendered over the presentation surface.
