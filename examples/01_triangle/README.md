# Triangle

Showcases the smallest safe `ez-gfx` graphics path: named vertex-heap data, an index heap, one indexed-indirect draw, and a compiled shader artifact.

`main.rs` retains the typed `positions` heap handle and geometry allocation, then acquires a fresh indirect handle after each frame begins. One batched `write_indirect` call writes and publishes the draw before graphics records it.

Run:

```text
mise run example-1
```

![Rendered triangle](../snapshots/01_triangle.png)

Expected result: a solid red triangle centered in the 640×480 presentation surface.
