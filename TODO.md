# Handoff

## Completed baseline

- Cargo workspace with compiler-free runtime, safe Rust contracts, C ABI v17, opaque generational handles, validated semantic/artifact/capability data, and panic containment.
- Vulkan, DX12, and Metal backend crates use a backend-neutral HAL and `gpu-allocator` 0.28.
- Offline Slang compilation emits SPIR-V 1.5, Shader Model 6.5 DXIL, and Metal products in bounded `.ezgfx` artifacts.
- Vulkan and DX12 resource upload, compute, indexed-indirect draw, presentation, and opt-in readback paths are implemented. DX12 hardware compute and graphics proofs passed on the local RTX 3080.
- KTX2/Basis decoding, progressive texture residency, bounded streaming/events/diagnostics, six migrated examples, external PNG comparison, runtime/compiler package isolation, and canonical plan updates are present.
- `cargo clippy --workspace --all-targets -- -D warnings` passed before the final `cargo fmt --all`.

## Remaining verification

The priority stop cancelled these post-format checks before completion. Run them unchanged:

```text
cargo test -p ez-gfx-examples --test smoke -- --nocapture
cargo test -p ez-gfx-ffi --test abi dx12_frame_uploads_indirect_compiles_graph_and_reads_back_texture -- --exact --nocapture
cargo check -p ez-gfx-ffi --target aarch64-apple-darwin
cargo run -p xtask -- package x86_64-pc-windows-msvc 0.1.0 target/package-smoke
```

No final-tree failure is known. Before the final refactors/format, the snapshot suite passed 2/2, the exact DX12 readback test passed, the Metal backend crate cross-checked for `aarch64-apple-darwin`, and package isolation completed.

## Platform gaps

- Metal was compile-checked only; no Apple host was available for device, metallib, CAMetalLayer, presentation, or readback execution.
- Vulkan and DX12 were exercised locally; other adapters/drivers remain unobserved.

## Constraints

- Runtime packages must not contain or depend on Slang, DXC, compiler crates, or source/JIT fallback.
- Artifact loading selects an exact backend, stage, entry, and semantic profile; malformed or mismatched data fails closed.
- The semantic floor is `ez-gfx-v1`; implemented DXIL requires Shader Model 6.5, not 6.6 direct heap indexing.
- Native handles and physical shader layouts remain backend-local. Public bindings use stable semantic IDs.
- FFI validates null/count/size/UTF-8 boundaries and catches panics. Non-null pointer validity remains the C caller's obligation.
- Snapshot-disabled presentation must not allocate or copy readback memory.
