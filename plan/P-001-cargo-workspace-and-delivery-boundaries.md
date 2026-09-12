# P-001: Cargo workspace and delivery boundaries

## Problem

Decide the Rust workspace packaging, feature flags, and deliverable boundaries for migrating `ez_gfx_api` into a modular Cargo workspace. The crate topology must decouple offline shader compilation and optional decoders from runtime consumption, supporting desktop platforms across Vulkan, DX12, and Metal.

## Prompt context

Full user prompt: “migration of ez-gfx-api from `..\..\oss\ez_gfx_api\` tho this folder. the new version should use rust, cargo and stuff, including support for vulkan, dx12 and metal, including support for basisu tex compression and slang shaders. the slang shaders should be universal across all 3 gfx apis. also split up the compiler and runtime section, so that you don't need to bundle the slang compiler everywhere (there is a section about this inside TODOs.md inside the original project). prepare implementation for the different todos. use the memory-allocator rust crate instead of vma. setup tests including snapshot tests. roughly maintain the original api”

Source evidence inspected: original `TODO.md` (precompiled shaders, optional KTX2 linking), `README.md`, `CONTEXT.md`, and `bindings/bindings.xml`.

## Constraints and acceptance criteria

- The workspace packaging must separate shader compilation from the core runtime so shipping applications do not bundle Slang compiler shared libraries or compiler logic.
- Decoders (KTX2, Basis Universal, image formats) must be configurable via Cargo feature flags.
- Support multi-target builds across Windows (Vulkan/DX12), Linux (Vulkan), and macOS (Metal).
- Explicit non-goals: rewriting application assets, creating a custom windowing library, or changing shader syntax away from Slang.

## Dependencies

- Outgoing dependency: `P-002` depends on `P-001` for crate public API export boundaries.
- Outgoing dependency: `P-003` depends on `P-001` for backend crate modularization.
- Outgoing dependency: `P-005` and `P-006` depend on `P-001` for compiler vs runtime crate separation.

## Unresolved questions

- Should backends be separated into distinct crates or feature-gated modules in a single backend crate?
- Should the C ABI layer live in the core runtime crate or a dedicated FFI crate?

## Candidate solutions

### S-P-001-strict-workspace-boundary

#### Architecture, integration, and applicability

Use a virtual workspace with `resolver = "2"` (or `3` if the eventual MSRV permits it): `ez-gfx-core` owns API/graph/reflection types; `ez-gfx-runtime` consumes precompiled artifacts and selected backends; `ez-gfx-compiler` owns in-process `shader-slang`/slang-rs bindings and Apple postprocessing; `ez-gfx-ffi` owns `cdylib`/`staticlib` exports; optional decoder/backend packages isolate native dependencies. The runtime dependency graph must not contain the compiler crate. This directly addresses the original runtime Slang session and unconditional Odin KTX import while preserving dependencies on P-002, P-003, P-005, and P-006.

#### Evidence, tradeoffs, and failure modes

- Cargo workspaces share a lockfile/output directory, not features. Features are additive and unioned for a package; resolver v2 only avoids some build/dev/inactive-target unification. Separate packages therefore establish a stronger delivery boundary than features. Costs are more manifests, internal API boundaries, package selection, and schema compatibility. Failures include a runtime-to-compiler dependency, `--workspace`/`--all-features` being mistaken for a minimal-runtime check, conflicting native `links` packages, or an artifact lacking compiler/schema identity. No controlled build-time, binary-size, or incremental-build measurement exists; those remain unknown. Existing `out/` binaries have uncontrolled build conditions and are not evidence.

#### Sources

- [Cargo workspaces](https://doc.rust-lang.org/stable/cargo/reference/workspaces.html)
- [Cargo feature unification](https://doc.rust-lang.org/stable/cargo/reference/features.html#feature-unification)
- [Cargo resolver v2 and native `links`](https://doc.rust-lang.org/stable/cargo/reference/resolver.html#feature-resolver-version-2)
- [Slang compilation API and Rust bindings](https://shader-slang.org/docs/compilation-api/) and [shader-slang crate](https://crates.io/crates/shader-slang)
- Original `TODO.md` lines 45-47 and 61-63; `src/ctx.odin`, `src/shader.odin`, and `src/texture_manager.odin`.

### S-P-001-single-package-feature-matrix

#### Architecture, integration, and applicability

Keep one library package with modules gated by `compiler`, `ffi`, `basisu`, `ktx2`, `vulkan`, `dx12`, and `metal`; target-specific dependencies use `cfg(...)`. A runtime-only consumer selects no compiler feature. This minimizes cross-crate interfaces and can preserve a cohesive Rust API.

#### Evidence, tradeoffs, and failure modes

Cargo documents that every dependency edge contributes to the union of enabled features and that default features can be re-enabled by another edge. Resolver v2 does not isolate ordinary dependencies. One workspace test, example, or downstream consumer can therefore compile/link Slang or a decoder into the shared package; `--all-features` necessarily does so. This hypothesis is disqualified if “does not bundle Slang” means a structurally impossible runtime dependency rather than a documented feature combination. Runtime cost is otherwise expected to be identical after dead-code elimination only where native linking and build scripts do not retain artifacts; that is an inference, not a measurement. Compile time, binary size, and DLL dependencies require per-target measurements.

#### Sources

- [Cargo features and default-feature caveats](https://doc.rust-lang.org/stable/cargo/reference/features.html)
- [Cargo platform-specific dependencies](https://doc.rust-lang.org/stable/cargo/reference/specifying-dependencies.html#platform-specific-dependencies)
- [Cargo resolver limitations](https://doc.rust-lang.org/stable/cargo/reference/resolver.html)

## Performance comparison

No candidate has measured repository-specific build or binary data; runtime throughput is not a differentiator because both can expose the same compiled runtime code. Ranking therefore follows the hard delivery constraint first.

| Rank | Candidate | Hard constraints | Startup/runtime | Build, delivery, and reliability | Implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
- | 1 | Strict workspace boundary | Passes: runtime dependency graph can exclude Slang; platform and decoder packages remain selectable | Runtime effect unknown; no compiler startup path | Strongest protection against feature unification and accidental native linkage | More packages and schema boundaries | Cargo semantics sourced; project measurements missing |
| 2 | Single-package feature matrix | Conditional failure: an ordinary dependency edge can re-enable compiler/native features | Runtime effect unknown | Smaller manifest surface, but feature union can violate the no-Slang deliverable | Lower initial cost; higher configuration risk | Cargo semantics sourced; project measurements missing |

## Selected solution

**Selected: `S-P-001-strict-workspace-boundary`.**

The prompt makes compiler/runtime separation a shipping constraint, not a convenience. Package boundaries make the minimal runtime dependency graph inspectable and prevent ordinary feature unification from being the only safeguard. Use `ez-gfx-core`, `ez-gfx-runtime`, `ez-gfx-compiler`/CLI, and `ez-gfx-ffi`; keep backend and decoder native dependencies outside core. P-002, P-003, P-005, and P-006 consume these boundaries.

**Rejected:** the single-package matrix is not selected because Cargo feature union can reintroduce compiler dependencies through another edge. It becomes viable only if “do not bundle Slang” is relaxed to a documented build configuration rather than an enforced package boundary.

**Assumptions and risks:** the eventual MSRV determines resolver 2 versus 3; internal reflection/artifact types must remain versioned across packages; more packages may increase clean-build orchestration. No build-time or binary-size claim is made.

**Validation:** inspect `cargo tree` for a runtime-only package on Windows/Linux/macOS and fail if Slang/compiler/native compiler libraries appear; record clean/incremental build times and stripped runtime sizes for the same toolchain and target; verify decoder-disabled and each backend package independently.
