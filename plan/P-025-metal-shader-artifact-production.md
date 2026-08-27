# P-025: Metal shader artifact production and deployment

## Problem

Select the Metal product stored in `.ezshader`: MSL source, offline metallib variants, or both. The choice changes runtime compilation, Apple-tool requirements, compatibility, size, and fallback behavior.

## Prompt context

One Slang source targets Vulkan, DX12, and Metal. Shipping runtimes must not bundle Slang; compiler environments may use Apple tools.

## Constraints and acceptance criteria

- Runtime never invokes or links Slang.
- Product covers the declared Apple OS/architecture matrix.
- Apple compilation occurs in a valid tool environment.
- Metadata identifies platform/SDK/language/library/compiler compatibility inputs.
- Variant selection is deterministic and absence fails before pipeline creation.

## Dependencies

- P-001, P-005, P-006, P-007, P-020, and P-029.

## Unresolved questions

- Store MSL, AIR/metallib, or multiple products?
- Which Apple SDK/platform/GPU variants are required?
- Is runtime compilation of pre-generated MSL acceptable?
- Which reflection/debug data remains?

## Candidate solutions

### S-P-025-offline-metallib: Apple-toolchain metallib variants only

#### Approach and integration

Slang emits MSL offline; the compiler product invokes Apple's `metal` compiler to IR and links one or more `.metallib` products. `.ezshader` sections carry metallib bytes plus SDK/platform, minimum OS, Metal language/compiler, architecture/variant, entry-point, and source/interface hashes. Runtime selects a compatible section and calls Metal library-from-data/URL APIs; it never compiles source. Metal binary archives remain a separate derived PSO cache owned by P-024.

Apple documents `metal -> .ir`, optional `metal-ar`, then `metal -> .metallib`; it also notes command-line Metal tools for Windows use the same options, though supported SDK/licensing/distribution must still be proven.

#### Constraint applicability

Strongest compiler/runtime split and most deterministic shipping failure boundary. It requires an Apple-supported Metal toolchain for every release target and enough variants to cover the selected deployment matrix.

#### Performance evidence

Apple states the generated metallib is loadable at runtime, but supplies no general load-versus-source compile timing. Its binary-archive example (a separate PSO-binary product) shows an all-GPU archive of 696 KiB versus an 8 KiB shader metallib, and thinned groups of 124 KiB Apple, 180 KiB AMD, and 396 KiB Intel; these exact figures apply only to that documented sample and show variant multiplication, not this project's size. Measure build time, metallib bytes, library load, pipeline creation, and first-frame stutter per Apple target.

#### Tradeoffs and failure modes

Fast predictable runtime and no source exposure, but tighter SDK/OS compatibility and release infrastructure. Confusing metallib with GPU-specific binary archives, omitting variant metadata, or accepting an incompatible library at runtime are disqualifiers.

#### Sources

- [Apple precompiled shader libraries](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files.md) documents MSL-to-IR-to-metallib commands and runtime loading.
- [Slang Metal target behavior](https://shader-slang.org/slang/user-guide/metal-target-specific) documents generated MSL legalization and bindings.
- [Apple binary archive manipulation](https://developer.apple.com/documentation/metal/manipulating-metal-binary-archives) distinguishes metallib from GPU binary slices and provides the cited sample sizes.

### S-P-025-runtime-msl: Store MSL and compile with Metal at runtime

#### Approach and integration

Slang emits final, validated MSL and canonical reflection offline. `.ezshader` stores MSL, entry points, required Metal language/version/capabilities, hashes, and compile options. Runtime uses `MTLDevice` source-library creation, reports compiler diagnostics, then creates pipelines; it ships no Slang. P-024 may persist Metal binary archives to reduce later pipeline work.

#### Constraint applicability

Satisfies the literal no-Slang runtime rule and may span more OS/GPU combinations with one source product. It still bundles a runtime compilation path, so it fails if “precompiled shader” means no shader compiler work in shipping applications or if deployment policy forbids source compilation.

#### Performance evidence

Runtime compilation latency and memory are shader/device/OS dependent; no gathered primary source gives a portable number. Apple recommends offline GPU binaries to reduce first-launch/load stutter, so runtime MSL is expected to have worse cold behavior, but magnitude is unknown. Measure source bytes, compile latency/RSS, pipeline creation, cache warm-up, and first-frame stalls on named Apple systems.

#### Tradeoffs and failure modes

Simpler release matrix and useful diagnostics, but slower/non-deterministic cold start, source exposure, runtime compiler failures, and driver-dependent code generation. Treating successful Slang MSL emission as proof that every deployment OS accepts it is disqualifying.

#### Sources

- [Metal `newLibraryWithSource`](https://developer.apple.com/documentation/metal/mtldevice/newlibrarywithsource:options:completionhandler:) defines asynchronous runtime source compilation.
- [Apple precompiled shader libraries](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files.md) provides the offline alternative.
- [Slang supported targets](https://shader-slang.org/slang/user-guide/targets) describes Metal textual target generation.

### S-P-025-metallib-with-msl-fallback: Preferred binaries plus explicit source fallback

#### Approach and integration

Carry selected metallib variants and one MSL fallback. Runtime tries only metadata-compatible metallibs; if none match and policy explicitly permits compilation, it compiles MSL and records the chosen path. Release/runtime-only builds may disable fallback. Artifact hashes bind both products to the same canonical interface.

#### Constraint applicability

Maximizes deployment reach without Slang, but weakens deterministic “fully precompiled” behavior and increases artifact size. It is valid only if fallback is observable, policy-controlled, and snapshot/cache identity records it.

#### Performance evidence

Costs combine duplicated source/binary bytes with best-case metallib loading and worst-case runtime compilation. No representative project data exists. Measure artifact growth, binary hit rate, load/compile latency, and fallback frequency across the declared OS/GPU matrix.

#### Tradeoffs and failure modes

Useful transition strategy, but highest schema/test complexity. Silent fallback can hide broken packaging; mismatched source/binary reflection can produce divergent pipelines. Mandatory no-runtime-compiler deployments disqualify it.

#### Sources

- [Apple Metal libraries](https://developer.apple.com/documentation/metal/metal-libraries) documents compiled library products and linking.
- [Apple runtime source compilation](https://developer.apple.com/documentation/metal/mtldevice/newlibrarywithsource:options:completionhandler:) provides the fallback mechanism.
- [Apple precompiled libraries](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files.md) provides the preferred offline path.

## Performance comparison

| Rank | Candidate | Hard-constraint result | Normalized evidence | Startup/deployment cost |
| ---- | --------- | ---------------------- | ------------------- | ----------------------- |
| 1 | Offline metallib variants only | Passes no-runtime-Slang and strongest interpretation of precompiled/runtime separation | Apple documents offline build/load; project build, size, load, and compatibility data are unknown | Apple toolchain and variant matrix; no runtime source compile |
| — | Runtime MSL | **Hard failure** if inherited precompiled-shader TODO excludes runtime shader compilation | API exists, but comparative latency/RSS/stutter are unknown | Simplest artifact matrix; cold compiler work and runtime failure |
| — | Metallib plus MSL fallback | **Hard failure** under the same no-runtime-compiler interpretation; also permits silent packaging regression unless strictly gated | Combined size, binary-hit rate, and worst-case compile cost are unknown | Largest artifact/schema/test matrix |

The Apple archive size example is not a comparable metallib benchmark and is not used to rank performance. Prompt emphasis on compiler/runtime separation and precompiled shaders ranks metallib-only first; the other candidates fail before performance ranking.

## Selected solution

**Select S-P-025-offline-metallib: store Apple-toolchain-built metallib variants only.**

Offline tooling asks Slang for MSL, compiles it to IR, and links `.metallib` products with Apple tools. `.ezshader` carries compatible variants and platform/SDK/minimum-OS/Metal-language/compiler/entry/interface metadata. Runtime deterministically selects and loads a compatible library; absence fails before pipeline creation. Metal binary archives remain derived PSO caches under P-024, not shader libraries.

Reject runtime MSL and hybrid fallback because they move shader compilation and related nondeterministic failure into shipping runtime, contrary to the strongest reading of the inherited precompiled-shader/compiler split. Reconsider only if Apple deployment coverage cannot be achieved with supported offline variants and the user explicitly permits Metal's runtime compiler while still excluding Slang.

Evidence: Apple documents MSL-to-IR-to-metallib and runtime loading; Slang documents Metal source legalization. Apple's separate binary-archive sample shows variant size can be significant but does not predict this project's metallib size or timing. Offline build time, library bytes, compatibility coverage, load time, pipeline creation, and first-frame stutter remain unknown.

Assumptions: release infrastructure can run supported Apple Metal tools for every target and metallib compatibility metadata can be defined. Risks are SDK/OS incompatibility, missing variants, confusing metallib with GPU binary archives, stripped reflection/debug data, and non-Apple build-host constraints. Compile/load every entry on the declared Apple matrix with Slang absent at runtime; reject mismatched metadata; compare canonical reflection and backend goldens; measure offline time/RSS, bytes per variant, load/pipeline time, and first frame; audit runtime linkage for compiler libraries.
