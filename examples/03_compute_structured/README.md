# Compute Structured

Showcases a GLB scene rendered from reflected named vertex heaps, with structured primitive records and compute-generated indexed-indirect commands.

`main.rs` retains typed position/normal heaps and allocations plus persistent `Buffer<BasicPrimitive>` and `CountedBuffer<DrawIndexedCommand>` values. Each linear-loop iteration updates input, explicitly configures the swapchain, imports both buffers for compute and graphics, and passes the frame and target to `Example::handle_frame`.

Run:

```text
mise run example-3
```

![Structured compute scene](../snapshots/03_compute_structured.png)

Expected result: the normalized Sponza scene rendered with normal-derived colors and depth testing.
