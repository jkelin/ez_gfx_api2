# Compute Structured

Showcases a GLB scene rendered from reflected named vertex heaps, with structured primitive records and compute-generated indexed-indirect commands.

`main.rs` uploads typed position/normal geometry once. Each linear-loop iteration updates input, acquires one-frame `Buffer<BasicPrimitive>` and `CounterBuffer<DrawIndexedCommand>` values, explicitly configures the swapchain, records compute-generated commands and graphics, and passes the frame and target to `Example::handle_frame`.

Run:

```text
mise run example-3
```

![Structured compute scene](../snapshots/03_compute_structured.png)

Expected result: the normalized Sponza scene rendered with normal-derived colors and depth testing.
