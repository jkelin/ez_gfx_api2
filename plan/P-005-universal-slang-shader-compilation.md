# P-005: Universal Slang shader compilation

## Problem

Decide how to structure the Slang shader compilation pipeline to generate universal multi-target shader code (SPIR-V for Vulkan, DXIL for DX12, MSL/AIR for Metal) from a single Slang source file, maintaining consistent bindless layout, entry points, and shader-declared render targets.

## Prompt context

Full user prompt explicitly requires: "support for slang shaders. the slang shaders should be universal across all 3 gfx apis."
Source evidence: Original Odin code (`src/shader.odin`) loaded Slang source at runtime, targeting SPIR-V exclusively for Vulkan.

## Constraints and acceptance criteria

- A single universal Slang shader source file must compile correctly to SPIR-V, DXIL, and MSL/AIR.
- Uniform bindless resource access models and entry point conventions across all 3 targets.
- Extraction of shader reflection metadata (render target attributes, push constants, resource bindings) during compilation.
- Explicit non-goals: introducing a custom DSL or supporting legacy HLSL/GLSL files.

## Dependencies

- Incoming dependency: `P-005` depends on `P-001` for offline compiler crate boundaries.
- Outgoing dependency: `P-006` depends on `P-005` for precompiled container generation and reflection extraction.
- Outgoing dependency: `P-007`, `P-008`, `P-010` depend on `P-005` for pipeline reflection data.

## Unresolved questions

- How should Metal binding indices and argument buffers be mapped to Slang's auto-generated bindings?
- How to ensure Slang compiler optimization does not strip required reflection metadata?

## Candidate solutions

### S-P-005-slang-native-multi-target

#### Architecture, integration, and applicability

The offline compiler creates Slang targets for `SLANG_SPIRV`, `SLANG_DXIL`, and `SLANG_METAL` or `SLANG_METAL_LIB`, loads one module/entry-point set, links a composite program, extracts canonical reflection, and emits target code. `ParameterBlock` is designed to map to Vulkan descriptor sets, D3D12 descriptor tables, and Metal argument buffers. Preserve custom target attributes before optimization and serialize them as authoritative metadata.

#### Evidence, tradeoffs, and failure modes

Slang officially supports these targets. DXIL production/signing can require DXC libraries on Windows; `METAL_LIB` requires Apple/Xcode tools, while MSL text can be generated elsewhere and finalized on macOS. Metal legalization packs entry inputs and has target-specific binding behavior. No representative compile-time, output-size, or runtime performance measurement exists for this shader set. Failures include target-specific language features, inconsistent explicit bindings, stripped unused parameters/attributes, matrix/layout differences, unavailable DXC/Xcode tools, and compiler-version drift.

#### Sources

- [Slang supported targets](https://shader-slang.org/slang/user-guide/targets)
- [Slang compilation API](https://shader-slang.org/docs/compilation-api/)
- [Slang reflection](https://shader-slang.org/slang/user-guide/reflection)
- [Slang Metal-specific behavior](https://shader-slang.org/slang/user-guide/metal-target-specific)
- Original `src/shader.odin` and `examples/6_sponza_ktx2/draw.slang`.

### S-P-005-spirv-pivot-and-translation

#### Architecture, integration, and applicability

Compile Slang once to canonical SPIR-V, retain Slang-derived metadata separately, translate SPIR-V to MSL and HLSL with SPIRV-Cross or Naga, then use Apple Metal tools and DXC for final binaries. Explicit binding-remap tables become part of the artifact.

#### Evidence, tradeoffs, and failure modes

SPIRV-Cross emits MSL and HLSL source, not DXIL; Naga likewise has HLSL/MSL writers, so DXC and Apple tools remain. Translation can remap resource indices, and Slang user attributes are not preserved as general SPIR-V reflection. This adds parser/translator/tool versions and can drift from canonical metadata. No contextual compile/runtime benchmark was found. It is disqualified if all target outputs and reflection must come directly from one Slang semantic pipeline without remapping.

#### Sources

- [SPIRV-Cross README](https://github.com/KhronosGroup/SPIRV-Cross)
- [Naga backends](https://github.com/gfx-rs/wgpu/tree/trunk/naga/src/back)
- [Microsoft DXC](https://github.com/microsoft/DirectXShaderCompiler)

### S-P-005-incumbent-runtime-vulkan-compilation

#### Architecture, integration, and applicability

Keep the original model: initialize Slang in every runtime, compile source to SPIR-V at startup, and reflect there. It proves the custom attributes and Vulkan flow but has no DXIL/Metal artifact path.

#### Evidence, tradeoffs, and failure modes

The original implementation and snapshots are functional evidence only; no controlled startup benchmark exists. It bundles Slang, exposes compiler/toolchain failures to applications, and is disqualified by compiler/runtime separation and three-backend requirements.

#### Sources

- Original `src/shader.odin`, `src/ctx.odin`, examples, and `TODO.md` lines 45-47.

## Performance comparison

No comparable compile-time, artifact-size, startup, or shader-runtime measurements exist. Semantic fidelity to one Slang source and canonical reflection is the first ranking criterion.

| Rank | Candidate | Hard constraints | Compiler/startup | Runtime fidelity and reliability | Tool/implementation cost | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Native Slang multi-target | Passes one-source SPIR-V/DXIL/Metal and offline compiler boundary | Compile cost unknown; runtime compiler absent through P-006 | One frontend/reflection authority; target-specific legalization remains | Slang plus DXC/Xcode where required | Official targets sourced; project measurements missing |
| 2 | SPIR-V translation pivot | Conditional failure: target outputs no longer all direct Slang products | Extra translation stages; cost unknown | Binding/reflection drift risk; still needs DXC/Xcode | Largest toolchain | Translator capabilities sourced; workload fit incomplete |
| — | Runtime Vulkan incumbent | Hard failure: Vulkan-only and bundles Slang | Existing startup compile, unmeasured | Proven Vulkan path only | Lowest migration effort | Disqualified by prompt |

## Selected solution

**Selected: `S-P-005-slang-native-multi-target`.**

Compile the same module and entry points offline with Slang's SPIR-V, DXIL, and Metal targets. Extract target declarations and common interface metadata before optimization can erase intent, then pass target blobs and canonical reflection to P-006. Use target-specific binding validation rather than a second semantic compiler pipeline.

**Rejected:** the incumbent fails backend and packaging constraints. The SPIR-V pivot is rejected because SPIRV-Cross/Naga add remapping and do not directly emit DXIL; it becomes a fallback only if a native Slang target proves unusable for a required shader feature.

**Assumptions and risks:** the shared Slang subset covers all current shaders; DXC/signing and Xcode Metal tools are available in compiler environments; Metal legalization and explicit bindings can be reconciled. Compilation and artifact-size costs remain unknown.

**Validation:** compile every shader/entry point for all targets in CI runners with required native tools; compare canonical reflection and explicit target attributes across outputs; create pipelines and render snapshot fixtures on each backend; record compiler version/options, wall time, peak RSS, and blob sizes.
