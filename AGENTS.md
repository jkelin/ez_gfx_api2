# AGENTS.md

The agent WILL NOT update README.md unless explicitly asked to. The README.md is managed by humans.

## Scope

This repository is a Rust/Cargo migration of `ez_gfx_api`. Preserve the recognizable C/C# integration path while keeping the safe Rust API authoritative. Do not claim a feature is complete unless its implementation and required evidence exist.

## Architecture rules

- Keep `ez-gfx` safe: it owns runtime behavior, resource lifetimes, graph semantics, and typed errors. `ez-gfx-ffi` is a narrow boundary that validates, converts, delegates, and contains panics; it must not reimplement behavior.
- Treat Vulkan, DX12, and Metal as one synchronized contract. Any backend-facing abstraction, implementation, test, or documented behavior change must be assessed and updated consistently across all three, or explicitly record why a backend is unsupported.
- Keep backend-native handles, state lowering, and physical shader layouts private. Public Rust resources are owning wrappers; raw handles are doc-hidden and reserved for the stable C ABI, which uses opaque `uint64_t`/`u64` values with generation/owner validation.
- Keep compiler/runtime dependency isolation absolute: runtime crates and distributions contain no Slang, DXC, compiler crates, native compiler libraries, source compilation, JIT, or shader fallback. The non-distributed Rust examples are development compiler clients that compile adjacent Slang source paths with target lists and development mode, then load owned validated artifact bytes; runtime packages remain compiler-free.
- Keep texture and geometry policy in their backend-neutral manager crates (`ez-gfx-texture-manager`, `ez-gfx-geometry-manager`) behind manager-defined traits; backend crates implement those traits for their private native types, and `ez-gfx` links the selected implementations with static dispatch (generics), never `dyn`. Manager crates must never depend on backends, the runtime, or `ez-gfx`.
- Every crate keeps a current `README.md` describing its ownership, interfaces, and trait contracts; keep `docs/textures.md`, `docs/geometry.md`, and each manager README current with every related change.

## Shader and artifact contracts

- Use one root shared, backend-agnostic Slang module and one source convention; user shaders contain no physical Vulkan syntax or backend-specific binding assumptions.
- `.ezgfxshader` is a versioned, bounded, validated `rkyv` container. Validate sizes, counts, ranges, versions, digests, reflection, target coverage, and `(stage, entry-point name)` identity before allocation or backend calls. Permit multiple names per stage, require exact stage/name selection, and reject duplicate products, malformed, incompatible, ambiguous, or mismatched artifacts closed. Runtime selects precompiled target data and never compiles.
- Preserve canonical semantic resource IDs while retaining target-native layouts and specializations. DXIL semantics target Shader Model 6.5; do not introduce Shader Model 6.6 direct-heap requirements.

## ABI and boundary changes

- Treat `crates/ez-gfx-ffi` declarations and docs as the binding authority. Record only non-Rust facts such as managed overrides, validation, nullability, and access semantics in `crates/ez-gfx-ffi/src/bindings-metadata.json`. Run `cargo run -p ez-gfx-bindgen -- all`; never edit generated `bindings/bindings.xml` or `bindings/c/include/ez_gfx_api.h`. Change Rust exports, ABI version, metadata, generated outputs, layout probes, tests, and relevant docs atomically.
  Public exports use `ez_gfx_{object}_{operation}`. Context-bound functions put `context` first; operations on an existing object put that object second, including every frame operation as `(context, frame, ...)`. Operations on a frame use `frame`, never `render` or `graph`, terminology. Constructors and queries without an existing object still put context first when context-bound. Context operations put context first. Global ABI, error, adapter, decoder, handle, and semantic utilities omit context. Do not retain compatibility aliases or shims.
- Prefer fail-fast typed errors. Treat invalid external data, stale handles, lost devices, unsupported capabilities, and unavailable required adapters as explicit failures, never silent defaults.

## Dependencies and packaging

- Keep Slang/DXC/Apple compiler tooling in compiler packages and explicit development clients only. Audit dependency trees and native imports for runtime packages. Do not add a runtime fallback to make a build pass.
- Preserve the existing Cargo workspace and backend-local native dependencies. Do not add a new abstraction layer when the existing HAL/API boundary is sufficient.

## Docs, TODOs, and personal-project discipline

- Keep architecture decisions and status evidence current in the existing plan documents and `TODO.md`; do not duplicate or silently rewrite canonical decisions. Record new cross-module architectural work in root `TODO.md` and remove it when resolved.
- Treat `docs/textures.md` as the canonical texture contract. Update it with every texture machinery, public API, backend behavior, synchronization, or lifecycle change.
- Treat `docs/geometry.md` as the canonical living vertex and geometry contract. Update it with every vertex/index heap, structured vertex buffer, upload, shader binding, render usage, backend behavior, synchronization, lifecycle, or parity-status change.
- In personal-project implementations, comment edge cases local to the function being changed. Keep comments operational and specific; avoid speculative completion claims.
- Examples propagate routine failures with direct `?`. They must not use `.context(...)`; avoid `.with_context(...)` as well when direct propagation or a concise standalone `anyhow!` keeps the call readable.
- You can also read `VERIFICATION_HOSTS.md` to get addresses for ssh boxes to use for cross platform verification.
- Keep volatile contract numbers out of README files: never state specific C ABI versions, shader artifact format versions, or artifact magic values. Those belong in the `ez-gfx-ffi` binding authority, generated outputs, and canonical plan documents as appropriate.

## Verification

- Test the changed contract within its blast radius, then run applicable source-line checks, Clippy, and formatting in that order at handoff. Use backend-matrix tests where behavior crosses HAL boundaries; include ABI and artifact validation tests for corresponding contract changes. Do not regenerate immutable snapshots without an explicit requirement.
- Backend changes MUST be verified through the backend matrix, which runs each libtest case in its own `cargo test` process with a 60-second default timeout: Vulkan-facing changes use `remote-test-linux` and local Windows Vulkan suites; DX12 uses local Windows suites; Metal uses `remote-test-macos`; cross-backend texture or HAL changes use all three. When the development workstation itself hosts a backend (Windows for Vulkan/Direct3D 12), running the corresponding suites locally satisfies that backend's matrix requirement; Linux and macOS coverage remains remote. The remote tasks sync the current working tree first; Windows remotes additionally require POSIX-capable rsync locally and remotely plus a POSIX SSH shell.
- All tests and agentic smoke/verification processes MUST run hidden/headless, without showing or activating windows or taking focus. NEVER open a user-visible window from a test or agent unless the user explicitly requests it. If hidden automation stalls, fix the harness or report the blocker; NEVER fall back to visible windows.

## UI exception

If this repository gains a UI, every interactive element must have stable semantic kebab-case `data-testid` plus stable `id` or semantic class, forward selectors through shared components, and expose `pointer`, `text`, or `not-allowed` cursor affordances appropriately. Keep decorative labels unselectable but meaningful content selectable.
