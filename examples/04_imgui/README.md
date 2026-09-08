# Dear ImGui

Showcases Dear ImGui integration: font-atlas upload, CPU draw-data flattening, named geometry heaps, bindless texture sampling, clip rectangles, and per-command indirect draws.

`main.rs` procedurally loads the font atlas, resolves its texture binding, and creates typed heaps. Each linear-loop iteration flattens ImGui draw data into owning heap allocations and context-owned buffers, explicitly configures the swapchain, records graphics, and passes the frame and target to `Example::handle_frame`.

Run:

```text
mise run example-4
```

![Dear ImGui demo](../snapshots/04_imgui.png)

Expected result: the Dear ImGui demo window and controls rendered over the presentation surface.
