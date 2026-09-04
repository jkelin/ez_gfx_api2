# P-002: Public API and C ABI bindings

## Problem

Decide how to expose an idiomatic Rust public API while evaluating backward compatibility with the existing C ABI defined in `bindings/bindings.xml` and its consuming C# bindings.

## Prompt context

Full user prompt requires to "roughly maintain the original api" while transitioning to Rust.
Source evidence: `F:/Projects/oss/ez_gfx_api/bindings/bindings.xml`, `F:/Projects/oss/ez_gfx_api/AGENTS.md`, and C# interop bindings in `F:/Projects/oss/ez_gfx_api/csharp/`.

## Constraints and acceptance criteria

- Provide a safe, idiomatic Rust API with RAII handle ownership, typed error results, and slice-based buffer views.
- Roughly maintain the original API surface semantics so existing rendering concepts, handles, and pipeline flow remain recognizable.
- Explicit non-goals: forcing a complete redesign of the user-facing drawing model without architectural necessity.

## Dependencies

- Incoming dependency: `P-002` depends on `P-001` for crate layout.
- Outgoing dependency: `P-008`, `P-011`, `P-014`, `P-016` depend on `P-002` for API signatures and handle definitions.

## Unresolved questions

- Must the migration preserve exact binary/C ABI compatibility with `bindings/bindings.xml` for existing C# wrappers, or is a Rust-first API with conceptual parity sufficient?
- Should C ABI symbols be exported from the main library or a standalone FFI crate/module?

## Candidate solutions

### S-P-002-layered-rust-facade-and-ffi

#### Architecture, integration, and applicability

Expose safe RAII Rust types over a private core and keep every `extern "C"` export in `ez-gfx-ffi`, built as `cdylib`/`staticlib`. Preserve `bindings.xml` symbol names, ABI probe, fixed-width layouts, packed `u64` handle semantics, and explicit destroy/status behavior for existing C# `SafeHandle` consumers. Rust-only methods use slices and typed `Result`; FFI uses `#[repr(C)]`, pointer/count views, validated out-parameters, and integer status codes. Validate nullability, alignment, count multiplication, address range, handle generation/owner, UTF-8, and asynchronous ownership at the boundary. Panics must not unwind into foreign code.

#### Evidence, tradeoffs, and failure modes

The Rust Nomicon supports converting pointer/count pairs to slices behind a checked wrapper; it does not quantify overhead. Thin calls need not copy buffer contents, but call, validation, string-marshalling, and handle-check costs are unknown. Measure empty calls, uploads, and draw submission separately with compiler, CPU, and workload recorded. Dual surfaces add generator and ABI verification work. Failures include Rust-default layouts, changed enum widths/calling convention, null zero-length slice mishandling, borrowed upload data outliving the call, stale/cross-context handles, finalizer races, panic/abort behavior, and platform DLL dependencies.

#### Sources

- [Rust Nomicon: FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [Rust Reference: type layout](https://doc.rust-lang.org/reference/type-layout.html)
- [Rust `catch_unwind` limits](https://doc.rust-lang.org/std/panic/fn.catch_unwind.html)
- Original `bindings/bindings.xml` (ABI version, layouts, limits) and `csharp/` wrappers.

### S-P-002-rust-only-conceptual-parity

#### Architecture, integration, and applicability

Publish only an idiomatic Rust crate. Private-field resource types own or reference their context; `Drop` performs deferred release; frame/encoder borrows prevent recording against destroyed owners; constructors return `Result`; uploads accept slices. Backend selection remains private through an enum, trait object, or monomorphized implementation. Concepts and call flow remain recognizable without preserving binary compatibility.

#### Evidence, tradeoffs, and failure modes

Rust specifies deterministic destructor invocation, and generics are monomorphized while trait objects use runtime vtables. No ez-gfx measurement establishes dispatch overhead, binary-size growth, allocation count, or GPU throughput; all are unknown. RAII improves Rust-side lifetime checking but asynchronous GPU completion still needs deferred destruction. `Send`/`Sync` must reflect backend rules. This candidate is disqualified if existing C# or C consumers must work: the incumbent ABI uses exported functions, numeric handles, and managed `SafeHandle` releases.

#### Sources

- [Rust `Drop`](https://doc.rust-lang.org/std/ops/trait.Drop.html)
- [Rust Book: trait objects and dynamic dispatch](https://doc.rust-lang.org/book/ch18-02-trait-objects.html)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [wgpu resource ownership examples](https://docs.rs/wgpu/latest/wgpu/)
- Original `bindings/bindings.xml` and `csharp/EzGfx.Native/`.

## Performance comparison

No candidate has measured ABI-call, allocation, submission, or GPU-throughput data. The hard discriminator is continuity with the original API and existing bindings.

| Rank | Candidate | Hard constraints | Runtime and memory | Reliability/operations | Implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Rust facade plus dedicated FFI | Passes conceptual parity and preserves existing C/C# path | Boundary overhead and validation cost unknown; bulk slices need not copy | Requires ABI/layout/export tests; isolates unsafe foreign inputs | Dual surface and generator maintenance | Rust ABI rules and incumbent contract sourced |
| 2 | Rust-only parity | Fails if “roughly maintain” includes existing C/C# consumers | Dispatch/refcount costs unknown; removes FFI call path | Strong Rust lifetimes, but no existing managed deployment | Lower surface cost | Rust ownership properties sourced; project measurements missing |

## Selected solution

**Selected: `S-P-002-layered-rust-facade-and-ffi`.**

Expose an idiomatic RAII Rust API while preserving the recognizable handle/function contract through a separate `ez-gfx-ffi` package. Keep the core independent of ABI details. Every stable string uses an adjacent explicit byte length: required strings are non-null and nonzero, optional strings are either null+zero or non-null+nonzero, all ranges are capped UTF-8 without embedded NUL, and no terminator scan occurs. Validate every pointer/count, handle owner/generation, arithmetic bound, and out-parameter; map typed Rust errors to fixed C statuses; prevent unwinding across the boundary.

**Rejected:** Rust-only parity is rejected because the source repository contains a maintained C ABI and C# consumers, and removing them is a larger compatibility break than “roughly maintain” supports. It becomes viable only if the user explicitly drops non-Rust compatibility.

**Assumptions and risks:** exact binary compatibility is preferred but not explicitly demanded; preserving it requires target-specific layout and export checks. Async uploads cannot borrow caller memory after return. Finalizer order, panic mode, and deferred GPU destruction remain risks.

**Validation:** diff generated exports/layouts/constants against `bindings.xml`; run existing C# smoke consumers against the Rust library; fuzz invalid FFI inputs; benchmark empty calls, checked handles, uploads, and submission separately with CPU/toolchain/workload recorded.
