# Compute Structured

Showcases a GLB scene rendered from reflected named vertex heaps, with structured primitive records and compute-generated indexed-indirect commands.

`main.rs` retains typed position/normal heap and allocation handles. Each frame acquires fresh structured primitive and indirect buffers, writes the primitive records, explicitly publishes the CPU-known compute output count, then shares both handles from compute to graphics in that frame.

Run:

```text
mise run example-3
```

![Structured compute scene](../snapshots/03_compute_structured.png)

Expected result: the normalized Sponza scene rendered with normal-derived colors and depth testing.
