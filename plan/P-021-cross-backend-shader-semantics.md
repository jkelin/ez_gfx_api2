# P-021: Cross-backend shader execution semantics

## Problem

Define one observable shader ABI across Vulkan, DX12, and Metal so universal Slang source produces equivalent rendering rather than merely compiling on all three backends. The plan does not yet select canonical clip/depth coordinates, framebuffer origin and winding, matrix/layout rules, resource binding namespaces, push/root/constant data layout, or specialization behavior.

## Prompt context

Migrate `ez_gfx_api` to Rust/Cargo with Vulkan, DX12, and Metal support. Slang shaders must be universal across all three APIs, and the runtime must consume precompiled artifacts without bundling Slang while roughly preserving the original API.

## Constraints and acceptance criteria

- One documented source-level contract must produce equivalent observable results on all three backends.
- Canonical reflection and artifact metadata must fully describe any backend lowering.
- Backend corrections must not require three user-authored shader variants.
- The contract must cover raster coordinates, winding/culling, matrix and constant layout, bindings, specialization, and texture/sampler semantics.
- Unsupported semantics must fail during offline compilation or pipeline creation, not silently diverge at draw time.

## Dependencies

- P-003 supplies backend lowering and capabilities.
- P-005 supplies multi-target Slang compilation.
- P-006 persists canonical reflection and target metadata.
- P-007 consumes binding and pipeline-layout metadata.
- P-010 consumes render-target format intent.

## Unresolved questions

- Which coordinate, depth, origin, winding, matrix, and packing conventions are canonical?
- Which differences are normalized by Slang options, generated entry wrappers, pipeline state, or viewport transforms?
- What is the stable logical binding namespace, including bindless resources, samplers, constants, and specialization values?
- Which cross-target reflection equivalence checks are release-blocking?

## Candidate solutions

### S-P-021-logical-vulkan-compatible-abi: Canonical logical ABI with generated target adapters

#### Approach and integration

Keep the recognizable Vulkan incumbent semantics as the public logical contract: explicit row- or column-major matrix policy, $0..1$ depth, declared front-face convention, separate texture/sampler identities, stable logical bindless spaces, and fixed scalar/aggregate packing rules. The offline compiler generates or injects target entry wrappers and records every transform/remap in canonical reflection. Vulkan uses native conventions where possible; DX12 and Metal receive viewport/front-face state, entry-point legalization, argument-buffer/root-signature mapping, and constant-layout adaptation. Cross-target reflection is compared by logical identity rather than raw target binding numbers.

Slang already models type/variable declarations separately from target-specific layouts and exposes layout reflection per compilation target. Its Metal backend flattens and packs entry parameters, translates system semantics, maps specialization constants to function constants, and supports explicit binding rules; those transformations make artifact-recorded target adapters feasible. This candidate preserves one source while refusing target-native layout leakage through the public API.

#### Constraint applicability

This directly preserves incumbent concepts and provides one observable ABI. It requires every supported shader feature to have a defined lowering on all three targets; Metal-specific reflection limitations, such as packed varying-use metadata, must be rejected or normalized in canonical metadata. Unsupported target semantics fail offline.

#### Performance evidence

No source reports an end-to-end cost for this exact three-target ABI. A viewport/front-face adjustment is fixed pipeline or dynamic state; generated arithmetic wrappers can add instructions only where a convention cannot be expressed in pipeline state. Slang documents target legalization but publishes no general instruction-count or GPU-time bound. Required measurements are target disassembly size, wrapper instructions, pipeline variants, artifact/reflection bytes, compiler time, pipeline creation time, and identical scene GPU time on named adapters.

#### Tradeoffs and failure modes

The stable ABI simplifies callers and snapshots but creates owned adapter code and schema. Double-flips, mismatched winding, matrix transposition, incorrect struct packing, sampler namespace drift, and specialization-ID collisions can yield valid pipelines with wrong pixels. A Vulkan-shaped rule may unnecessarily burden DX12/Metal. Any feature without semantics-preserving lowering disqualifies it from the portable subset.

#### Sources

- [Slang reflection](https://shader-slang.org/slang/user-guide/reflection) separates declarations from target-specific layouts and exposes binding/layout data per target.
- [Slang supported targets](https://shader-slang.org/slang/user-guide/targets) describes distinct D3D12, Vulkan, and Metal parameter-binding models.
- [Slang Metal-specific functionality](https://shader-slang.org/slang/user-guide/metal-target-specific) documents entry legalization, argument buffers, matrix layout, explicit bindings, and specialization-to-function-constant translation.
- [Vulkan vertex post-processing](https://docs.vulkan.org/spec/latest/chapters/vertexpostproc.html) defines clip and framebuffer coordinate transforms.
- [D3D12 viewports and clipping](https://learn.microsoft.com/en-us/windows/win32/direct3d12/viewports-and-clipping) defines Direct3D viewport/depth behavior.

### S-P-021-target-native-layouts: Shared source with target-native ABI and canonical semantic IDs

#### Approach and integration

Guarantee one Slang source and equivalent resource semantics, but let each target own its physical parameter layout, binding numbers, root/argument-buffer structure, specialization representation, and legal entry signature. The artifact stores a canonical semantic resource graph plus complete per-target reflection tables. Runtime pipeline builders consume the selected target table; public resource handles bind by stable semantic ID/name hash rather than by a universal physical slot. Coordinate and winding differences use backend pipeline/viewport state; shader code follows one documented HLSL-style math convention.

This treats Slang's per-target `ProgramLayout` as authoritative and avoids forcing unlike APIs into one byte-level ABI. It resembles mature cross-API compilers that retain shared high-level declarations while emitting backend-specific binding layouts.

#### Constraint applicability

It satisfies universal source and compiler/runtime separation, but “universal” means semantic equivalence rather than identical physical ABI. It fits only if rough API continuity does not promise existing raw Vulkan binding numbers or constant-buffer bytes to all backends. Canonical artifact validation must prove that every required semantic resource exists with compatible type/access on every target.

#### Performance evidence

No comparable benchmark isolates target-native layout versus generated universal-layout adapters. Expected advantages—fewer wrapper instructions and more native root/argument-buffer layouts—are inference, not measurements. Costs include larger reflection/artifacts and runtime semantic-ID translation. Measure target blob/reflection size, lookup CPU time, descriptor/root updates, pipeline creation, generated instruction counts, and scene GPU time.

#### Tradeoffs and failure modes

This minimizes artificial backend constraints and follows Slang reflection directly, but expands artifact schema and testing. Reflection drift can connect the wrong resource despite valid code; semantic IDs must be collision-safe and deterministic. CPU structure bytes cannot be blindly shared unless a separate canonical serializer emits each target layout. Raw binding exposure would disqualify this candidate.

#### Sources

- [Slang reflection](https://shader-slang.org/slang/user-guide/reflection) states that one type may have different layouts depending on use and target.
- [Slang compilation](https://shader-slang.org/slang/user-guide/compiling) describes target-indexed layout/reflection from the compilation API.
- [Slang parameter blocks](https://shader-slang.org/docs/parameter-blocks/) describes a cross-platform parameter abstraction over D3D12, Vulkan, and Metal.
- [Metal-specific Slang behavior](https://shader-slang.org/slang/user-guide/metal-target-specific) shows why Metal entry and resource layouts are not byte-for-byte SPIR-V layouts.
- [D3D12 root signatures](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures) defines Direct3D's binding contract.

## Performance comparison

| Rank | Candidate | Hard-constraint result | Normalized evidence | Reliability/implementation cost |
| ---- | --------- | ---------------------- | ------------------- | ------------------------------- |
| 1 | Target-native layouts with canonical semantic IDs | Passes universal-source, no-runtime-Slang, backend-neutrality, and conceptual API continuity | Slang directly exposes target-indexed layouts; relative lookup, artifact-size, and GPU-time costs are unknown | Less lowering risk; requires canonical-ID validation and per-target CPU packing |
| 2 | Vulkan-compatible logical/physical ABI with generated adapters | Passes only if every Vulkan-shaped rule has semantics-preserving DX12/Metal lowering | No comparative performance data; wrapper instruction and pipeline-variant costs are unknown | Highest remapper/schema ownership and silent double-transform risk |

The candidates have no comparable workload measurement. The target-native option ranks first because the prompt requires one source and three native APIs, not identical physical bindings; Slang's documented target layouts are direct evidence. Forcing a Vulkan-shaped physical ABI adds unproven work and conflicts with the selected backend-neutral HAL direction.

## Selected solution

**Select S-P-021-target-native-layouts: shared Slang source, canonical semantic ABI, target-native physical layouts.**

One documented source convention fixes math, depth, winding, texture/sampler, and semantic resource identity. Offline compilation emits SPIR-V, DXIL, and Metal products plus a canonical semantic resource graph and complete per-target layouts. Runtime binds stable, collision-safe semantic IDs to the selected target layout; it never assumes identical binding numbers or aggregate byte layouts. Pipeline/viewport state handles coordinate differences where possible, and offline compilation rejects semantics without equivalent lowering.

Reject the Vulkan-compatible physical ABI because identical backend slots/packing are not requested, target APIs differ materially, and generated remapping adds unmeasured code and correctness risk. Reconsider it only if incumbent C/C# compatibility proves callers persist raw Vulkan binding numbers or constant-buffer bytes that cannot migrate to semantic IDs.

Evidence: Slang reflection explicitly separates declarations from layouts and exposes target-indexed layout data; its Metal backend documents entry legalization, binding, matrix, and function-constant differences. No source establishes a performance winner, so runtime lookup, artifact size, target instruction counts, compile/pipeline times, and scene GPU time remain unknown.

Assumptions: “roughly maintain” preserves concepts, not raw Vulkan physical layout; all required shader features have equivalent target semantics. Risks are reflection drift, ID collision, CPU packing mismatch, coordinate double transforms, and larger artifacts. Validate collision-free deterministic IDs; compare canonical type/access/resource sets across all targets; snapshot reflection and generated code; test matrix/packing/specialization/texture-sampler/coordinate cases; run identical backend goldens; measure lookup, artifact bytes, compilation, pipeline creation, instruction counts, and GPU time on named adapters.
