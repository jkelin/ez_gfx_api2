# P-006: Precompiled shader container and reflection

## Problem

Decide the format and serialization schema for precompiled shader modules containing target binaries (SPIR-V, DXIL, MSL) and reflection metadata (target declarations, load/store actions, bindings, push constants) so the runtime does not require Slang at startup, resolving conflicts between explicit target attributes and compiler optimizations.

## Prompt context

Full user prompt: "split up the compiler and runtime section, so that you don't need to bundle the slang compiler everywhere (there is a section about this inside TODOs.md inside the original project)."
Source evidence: `TODO.md` ("Add precompiled shader modules with reflection metadata so applications do not need to ship Slang source or compile reflection at runtime", "Document and enforce that explicit target attributes are the source of truth").

## Constraints and acceptance criteria

- Binary/container format containing multi-backend bytecode and complete reflection metadata.
- Runtime loader must parse precompiled containers with zero compiler dependency.
- Explicit target declarations in shader source must remain the authoritative source of truth for graph scheduling regardless of compiler optimization passes.
- Explicit non-goals: runtime JIT recompilation without explicit compiler activation.

## Dependencies

- Incoming dependency: `P-006` depends on `P-005` for compilation output.
- Outgoing dependency: `P-007` and `P-008` depend on `P-006` for runtime shader reflection and pipeline creation.

## Unresolved questions

- Should the container format be binary (e.g. Bincode/FlatBuffers) or human-readable (e.g. JSON metadata alongside raw bytecode)?
- Should offline compilation be exposed via a CLI tool, build.rs integration, or both?

## Candidate solutions

### S-P-006-versioned-sectioned-bundle

#### Architecture, integration, and applicability

Emit one `.ezgfxshader` with a fixed little-endian frame around a bounded `rkyv` payload. The payload contains canonical reflection, stage-grouped target products, compiler/toolchain provenance, and one internal entry point per stage. Runtime verifies framing, digest, bytechecked archive structure, and semantic coverage before selecting a target without linking Slang.

#### Evidence, tradeoffs, and failure modes

DXIL uses a mature FourCC/version/size/hash/part-offset container; KTX2 uses a fixed identifier and bounded offset/length indices. Apply checked arithmetic, count/size ceilings, alignment/non-overlap, unique required sections, hashes, schema validation, and backend bytecode validation. Unknown optional sections can be skipped; incompatible required versions are rejected. Memory mapping can avoid a file read copy, but backend module creation and reflection ownership may still copy; no load-time, RSS, or artifact-size measurement exists. Corruption, overflow, stale schema, reflection/blob mismatch, and non-atomic replacement are failure modes. Avoid a Rust-native bincode layout as the permanent unversioned wire contract.

#### Sources

- [DXIL container header](https://github.com/microsoft/DirectXShaderCompiler/blob/main/include/dxc/DxilContainer/DxilContainer.h)
- [KTX 2.0 specification](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html)
- [Vulkan shader module validation](https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateInfo.html)
- Original `src/shader.odin` and `TODO.md` lines 45-47.

### S-P-006-manifest-and-sidecar-binaries

#### Architecture, integration, and applicability

Emit a versioned JSON or CBOR manifest with reflection, hashes, target/profile metadata, and paths to `.spv`, `.dxil`, and `.metallib`/MSL files. Runtime validates the manifest and each hash before selecting a sidecar. Compiler and runtime still share a documented schema but not Slang.

#### Evidence, tradeoffs, and failure modes

Human-readable JSON improves inspection and independent binary replacement; loose files simplify native tool invocation. It increases file opens and creates partial-deployment and path/case hazards. Atomic deployment requires a containing package or versioned directory plus final manifest rename. Claims that JSON is materially slower are unsupported here; parse latency, file-system cost, memory, and package size are unknown and need measurements across target storage. Missing, stale, swapped, or traversal-capable paths are disqualifiers unless hashes and path confinement are enforced.

#### Sources

- [Slang JSON reflection](https://shader-slang.org/slang/user-guide/reflection)
- [RFC 8259 JSON](https://www.rfc-editor.org/rfc/rfc8259)
- [RFC 8949 CBOR](https://www.rfc-editor.org/rfc/rfc8949)

### S-P-006-schema-generated-single-file

#### Architecture, integration, and applicability

Use a schema-defined format such as FlatBuffers for reflection and an outer blob table for shader binaries. Generated readers provide explicit field evolution; the outer header supplies magic, size ceilings, integrity, and required-section semantics.

#### Evidence, tradeoffs, and failure modes

FlatBuffers documents direct access without unpacking, but that does not prove lower end-to-end startup cost for this workload. Generated code and schema evolution reduce accidental Rust-layout coupling while adding a tool dependency and verifier requirements. Missing bounds verification, unsupported required fields, and bytecode lifetime assumptions remain failures. Performance is unknown.

#### Sources

- [FlatBuffers schema evolution](https://flatbuffers.dev/evolution/)
- [FlatBuffers Rust usage](https://flatbuffers.dev/languages/rust/)

## Performance comparison

No candidate has comparable load-time, RSS, file-open, or artifact-size measurements. Atomic delivery, bounded parsing, and schema ownership drive ranking.

| Rank | Candidate | Hard constraints | Startup/memory | Reliability/operations | Implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Framed, validated `rkyv` | Passes zero-Slang runtime and complete multi-target artifact | Requires one bounded aligned validation copy | One hashable artifact; explicit format cutover | Small fixed frame plus maintained archive crate | Implemented and contract-tested |
| 2 | Schema-generated bundle | Passes | Direct access claimed by tool, end-to-end benefit unknown | Generated evolution contract; verifier/tool dependency | Schema/codegen overhead | Format properties sourced; workload missing |
| 3 | Manifest plus sidecars | Passes only with confined, hashed, atomic packaging | Extra file/parse cost unknown and incomparable | Highest partial-deployment/path risk | Easiest inspection and native-tool integration | Standards sourced; workload missing |

## Selected solution

**Selected: `S-P-006-versioned-sectioned-bundle`.**

Define a fixed 56-byte little-endian frame containing magic, format version, reserved flags, payload length, and BLAKE3 digest around one bounded `rkyv` payload. Format v3 bytechecks aligned bytes and validates archived collection, string, metadata, provenance, and variant ceilings before owned deserialization. Runtime then validates every required backend/stage reflection once, carries typed binding and pipeline-layout products, and rejects missing, ambiguous, malformed, conflicting, or invalid texture-heap data before native shader creation. It never silently invokes Slang.

**Rejected:** sidecars remain rejected because partial deployment and synchronization undermine a shipping asset boundary. A custom section parser and generated schema pipeline add owned evolution machinery already covered by framed `rkyv`.

**Assumptions and risks:** format evolution uses explicit version cutovers; hashes detect corruption, not authenticity. Archive validation, integer bounds, stale reflection, and backend product mismatch remain critical.

**Validation:** property/fuzz test truncated, overlapping, duplicated, oversized, unknown, and corrupted sections; round-trip compiler artifacts; reject incompatible required versions; load every target without Slang present; benchmark cold/warm load, opens, allocations, peak RSS, and artifact size on specified OS/storage.
