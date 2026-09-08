# Triangle

Showcases the smallest safe `ez-gfx` graphics path: named vertex-heap data, an index heap, one indexed-indirect draw, and a compiled shader artifact.

`main.rs` reads top-to-bottom: it creates typed geometry, a persistent counted buffer, and a shader, then enters a linear frame loop. Each iteration explicitly begins and configures the swapchain, records graphics, and passes the frame and target to `Example::handle_frame`.

Run:

```text
mise run example-1
```

![Rendered triangle](../snapshots/01_triangle.png)

Expected result: a solid red triangle centered in the 640×480 presentation surface.
