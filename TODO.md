# Handoff

## Completed baseline

- Cargo workspace with compiler-free runtime, safe Rust contracts, C ABI v17, opaque generational handles, validated semantic/artifact/capability data, and panic containment.
- Vulkan, DX12, and Metal backend crates use a backend-neutral HAL and `gpu-allocator` 0.28.
- Offline Slang compilation emits SPIR-V 1.5, Shader Model 6.5 DXIL, and Metal products in bounded `.ezgfx` artifacts.
- Vulkan and DX12 resource upload, compute, indexed-indirect draw, presentation, and opt-in readback paths are implemented. DX12 hardware compute and graphics proofs passed on the local RTX 3080.
- KTX2/Basis decoding, progressive texture residency, bounded streaming/events/diagnostics, six migrated examples, external PNG comparison, runtime/compiler package isolation, and canonical plan updates are present.
- `cargo clippy --workspace --all-targets -- -D warnings` passed before the final `cargo fmt --all`.

## Verification status

- Six standalone programs own their host setup, `ApplicationHandler`, scene, required helpers, shader inputs, and artifacts; no shared Rust example tree or library target remains.
- All six one-frame Metal Validation runs pass with zero diagnostics and zero dropped observations.
- `cargo build -p ez-gfx-examples --bins` passes; the seven-test examples smoke passes 7/7.
- Example snapshots were regenerated and verified.
- Full workspace `cargo clippy --workspace --all-targets -- -D warnings` passed, followed by `cargo fmt --all`.

## Remaining verification

```text
cargo run -p xtask -- package x86_64-pc-windows-msvc 0.1.0 target/package-smoke
  blocked: the x86_64-pc-windows-msvc Rust target is not installed

DX12 native target execution/type-check
  unavailable: this macOS Homebrew Rust host cannot execute or type-check the native Windows DX12 target
```

## Constraints

- Runtime packages must not contain or depend on Slang, DXC, compiler crates, or source/JIT fallback.
- Artifact loading selects an exact backend, stage, entry, and semantic profile; malformed or mismatched data fails closed.
- The semantic floor is `ez-gfx-v1`; implemented DXIL requires Shader Model 6.5, not 6.6 direct heap indexing.
- Native handles and physical shader layouts remain backend-local. Public bindings use stable semantic IDs.
- FFI validates null/count/size/UTF-8 boundaries and catches panics. Non-null pointer validity remains the C caller's obligation.
- Snapshot-disabled presentation must not allocate or copy readback memory.
