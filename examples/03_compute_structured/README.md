# Compute Structured

Showcases a GLB scene rendered from reflected named vertex heaps, with structured primitive records and compute-generated indexed-indirect commands.

`main.rs` retains typed position/normal heaps and allocations. For each host-configured frame it acquires fresh `Buffer<BasicPrimitive>` and `CountedBuffer<DrawIndexedCommand>` values, writes primitive records, publishes the CPU-known output count, and shares both from compute to graphics.

Run:

```text
mise run example-3
```

![Structured compute scene](../snapshots/03_compute_structured.png)

Expected result: the normalized Sponza scene rendered with normal-derived colors and depth testing.
