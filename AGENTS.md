# AGENTS.md

## Scope

This repository is a Rust/Cargo migration of `ez_gfx_api`. Preserve the recognizable C/C# integration path while keeping the safe Rust API authoritative. Do not claim a feature is complete unless its implementation and required evidence exist.

## Architecture rules

- Keep `ez-gfx` safe: it owns runtime behavior, resource lifetimes, graph semantics, and typed errors. `ez-gfx-ffi` is a narrow boundary that validates, converts, delegates, and contains panics; it must not reimplement behavior.
- Treat Vulkan, DX12, and Metal as one synchronized contract. Any backend-facing abstraction, implementation, test, or documented behavior change must be assessed and updated consistently across all three, or explicitly record why a backend is unsupported.
- Keep backend-native handles, state lowering, and physical shader layouts private. Public Rust resources use typed handles; the stable C ABI uses opaque `uint64_t`/`u64` handles with generation/owner validation.
- Keep compiler/runtime dependency isolation absolute: runtime crates and distributions contain no Slang, DXC, compiler crates, native compiler libraries, source compilation, JIT, or shader fallback. The non-distributed Rust examples are development compiler clients that compile adjacent Slang source paths with target lists and development mode, then load owned validated artifact bytes; runtime packages remain compiler-free.

## Shader and artifact contracts

- Use one root shared, backend-agnostic Slang module and one source convention; user shaders contain no physical Vulkan syntax or backend-specific binding assumptions.
- `.ezgfxshader` is a versioned, bounded, validated `rkyv` container. Validate sizes, counts, ranges, versions, digests, reflection, target coverage, and entry/stage identity before allocation or backend calls. Require exactly one entrypoint per stage; reject malformed, incompatible, ambiguous, or mismatched artifacts closed. Runtime selects precompiled target data and never compiles.
- Preserve canonical semantic resource IDs while retaining target-native layouts and specializations. DXIL semantics target Shader Model 6.5; do not introduce Shader Model 6.6 direct-heap requirements.

## ABI and boundary changes

- Change `include/ez_gfx_api.h`, `bindings/bindings.xml`, Rust exports/bindings, ABI version, export parity, layout probes, tests, and relevant docs atomically. Validate null/count/size pairs, arithmetic, UTF-8, enums/layouts, out-pointers, ownership, handles, and async lifetimes; no panic or unwind crosses C.
- Prefer fail-fast typed errors. Treat invalid external data, stale handles, lost devices, unsupported capabilities, and unavailable required adapters as explicit failures, never silent defaults.

## Dependencies and packaging

- Keep Slang/DXC/Apple compiler tooling in compiler packages and explicit development clients only. Audit dependency trees and native imports for runtime packages. Do not add a runtime fallback to make a build pass.
- Preserve the existing Cargo workspace and backend-local native dependencies. Do not add a new abstraction layer when the existing HAL/API seam is sufficient.

## Docs, TODOs, and personal-project discipline

- Keep architecture decisions and status evidence current in the existing plan documents and `TODO.md`; do not duplicate or silently rewrite canonical decisions. Record new cross-module architectural work in root `TODO.md` and remove it when resolved.
- In personal-project implementations, comment edge cases local to the function being changed. Keep comments operational and specific; avoid speculative completion claims.

## Verification

- Test the changed contract within its blast radius, then run applicable source-line checks, Clippy, and formatting in that order at handoff. Use backend-matrix tests where behavior crosses HAL boundaries; include ABI and artifact validation tests for corresponding contract changes. Do not regenerate immutable snapshots without an explicit requirement.

## UI exception

If this repository gains a UI, every interactive element must have stable semantic kebab-case `data-testid` plus stable `id` or semantic class, forward selectors through shared components, and expose `pointer`, `text`, or `not-allowed` cursor affordances appropriately. Keep decorative labels unselectable but meaningful content selectable.
