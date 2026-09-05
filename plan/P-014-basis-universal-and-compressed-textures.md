# P-014: Basis Universal and compressed textures

## Problem

Decide how to integrate Basis Universal texture transcoding (.basis and KTX2 UASTC/ETC1S) into native compressed GPU formats (BC1-BC7 on desktop, ASTC on mobile/macOS) and support direct block-compressed texture ingestion, with opt-in build-time feature flags to avoid mandatory decoder linkage.

## Prompt context

Full user prompt explicitly requires: "including support for basisu tex compression...".
Source evidence: `TODO.md` ("Support compressed texture formats. Color target parsing is still limited to uncompressed formats such as rgba8 and rgba16f, with no BC/ASTC/block-compressed texture handling", "Make KTX2 optional at link time for static builds...").

## Constraints and acceptance criteria

- Support Basis Universal (.basis) and KTX2 container formats with runtime transcoding to optimal GPU block formats (BC7, BC3, BC1, ASTC).
- Support direct loading of raw block-compressed texture data (BCn, ASTC).
- Make KTX2 and Basis Universal decoders optional via Cargo feature flags (e.g. `feature = "basis"`).
- Explicit non-goals: runtime GPU encoding/compression of raw images into Basis format on the client.

## Dependencies

- Incoming dependency: `P-014` depends on `P-001` (Feature flags), `P-003` (HAL format support), and `P-004` (Allocator).
- Outgoing dependency: `P-015` depends on `P-014` for texture upload pathways and mip streaming.

## Unresolved questions

- Which Rust Basis Universal transcoder crate (e.g. `basis-universal`, `basisu-sys`, or pure Rust transcoder) provides the best performance and cross-platform reliability?
- How should target format selection be mapped when a platform lacks BC7 or ASTC support?

## Candidate solutions

### S-P-014-basis-universal-feature-transcoder: Native C++ transcoder wrapper (`basis-universal` crate) behind Cargo feature flag

#### Approach and integration

Integrate the `basis-universal` crate (wrapping Binomial's official C++ Basis Universal transcoder) under a `basis` Cargo feature flag. At engine initialization, query GPU device format support tables across Vulkan, DX12, and Metal:
- If BC7 is supported (Desktop DX12/Vulkan/Metal): transcode UASTC -> BC7 (high quality) and ETC1S -> BC1/BC3.
- If ASTC is supported (Apple Silicon / Mobile): transcode UASTC -> ASTC 4x4.
- Fallback: transcode to BC3/BC1 or uncompressed RGBA8.
Transcoding occurs directly into staging buffer memory before transfer copy recording. Direct BCn/ASTC DDS/KTX2 files bypass transcoding and copy directly into matching compressed GPU image layouts.

#### Performance evidence
- **Transcoding Rate:** Transcoder throughput varies across host CPU architectures, SIMD instruction availability, and target block formats; exact throughput is unknown until benchmarked in the target runtime environment.
- **VRAM Savings (Bit-Rate Arithmetic):** Compared to 32-bit uncompressed RGBA8 (32 bits per pixel), standard block-compressed formats have fixed nominal bit-rates:
  - BC1 (RGB / 1-bit alpha): 4 bits per pixel (8:1 nominal compression vs RGBA8) (`[INFERENCE]` from 64-bit per 4x4 block format specification).
  - BC3 / BC7 / ASTC 4x4 (RGBA): 8 bits per pixel (4:1 nominal compression vs RGBA8) (`[INFERENCE]` from 128-bit per 4x4 block format specification).
  - ASTC 8x8 (RGBA): 2 bits per pixel (16:1 nominal compression vs RGBA8) (`[INFERENCE]` from 128-bit per 8x8 block format specification).
- **Disk Footprint:** Supercompressed container size depends on image entropy and texture content; exact compression ratio is unknown until measured on project assets.
#### Tradeoffs and failure modes

- **Tradeoffs:** `basis-universal` compiles a native C++ transcoder submodule via `cc-rs` in `build.rs`, requiring a C++ compiler in the build environment when the feature is enabled.
- **Failure Modes:** Transcoding fails if image dimensions are not a multiple of 4 on block-compressed targets unless padding is applied.

#### Sources

- [Basis Universal GitHub & Transcoder Specification](https://github.com/BinomialLLC/basis_universal) — transcoding targets and format speeds.
- [Docs.rs basis-universal](https://docs.rs/basis-universal) — official Rust transcoder bindings.
- `F:/Projects/oss/ez_gfx_api/src/texture_manager.odin` — original KTX2 / decoder linkage.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — compressed texture and optional KTX2 linking requirements.

### S-P-014-direct-compressed-ktx2-ingestion: Pure Rust KTX2 parser with pre-compressed block textures

#### Approach and integration

Implement pure-Rust KTX2 container reading using `ktx2` crate for directly pre-encoded BCn/ASTC textures, with runtime Basis transcoding disabled by default. Transcoding is available only if an optional external dynamic transcoder plugin is registered at runtime.

#### Performance evidence

- **Zero C++ Build Dependency:** Pure Rust build toolchain.
- **Transcode Availability:** Cannot unpack generic `.basis` or universal ETC1S/UASTC textures without external pre-transcoding or plugin.

#### Tradeoffs and failure modes

- **Tradeoffs:** Zero C++ compiler requirements for build, but loses runtime universal asset transcoding unless pre-baked into platform-specific block formats.
- **Failure Modes:** Incompatible with universal multi-platform `.basis` assets.

#### Sources

- [Docs.rs ktx2](https://docs.rs/ktx2) — pure Rust KTX2 container reader.

### S-P-014-uncompressed-only-decoder: Incumbent uncompressed RGBA8 software expansion

#### Approach and integration

Maintain the original Odin architecture: unpack all compressed textures to uncompressed 32-bit RGBA or 64-bit RGBA16F in CPU memory prior to GPU transfer.

- **VRAM Footprint (Bit-Rate Arithmetic):** Uncompressed 32-bit RGBA8 allocates 32 bpp (e.g. 4096x4096x4 bytes = 64 MB base level), compared to 8 bpp (16 MB base level) for BC7/ASTC 4x4 (`[INFERENCE]` based on format block size definitions).
- **Memory Bandwidth:** Higher bit-rates increase memory bandwidth consumption during texture sampling (`[INFERENCE]`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Avoids block format alignment constraints.
- **Failure Modes:** Severe memory pressure; fails the explicit prompt requirement for Basis Universal and compressed textures.

#### Sources

- `F:/Projects/oss/ez_gfx_api/src/texture_manager.odin` — incumbent implementation.
- `F:/Projects/oss/ez_gfx_api/TODO.md` — format limitations.
- [Khronos KTX specification](https://github.khronos.org/KTX-Specification/) — compressed texture container semantics.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| Native C++ transcoder wrapper (`basis-universal`) behind feature flag | Fixed bit-rate VRAM reduction (4:1 to 16:1 vs RGBA8); transcode speed unknown until benchmarked | High | Format specifications & crate documentation | C++ toolchain needed when feature enabled |
| Pure Rust KTX2 parser with pre-compressed block textures | Fast direct GPU copies for pre-compressed formats, lacks runtime universal transcode | Moderate | Rust ecosystem docs | Requires pre-baked platform-specific DDS/KTX2 assets |
| Incumbent uncompressed RGBA8 software expansion | 32 bpp uncompressed VRAM footprint, high memory bandwidth | Low | Direct source code inspection | Violates explicit prompt requirements |
## Selected solution

### Selection

`S-P-014-basis-universal-feature-transcoder`: Native C++ transcoder wrapper (`basis-universal` crate) behind Cargo feature flag.

### Selection rationale

`S-P-014-basis-universal-feature-transcoder` directly satisfies the prompt's explicit requirement for Basis Universal texture compression and the inherited TODO for optional linkage:
1. It transcodes universal `.basis` and KTX2 UASTC/ETC1S supercompressed textures at runtime to optimal GPU native block formats (BC7 on desktop Vulkan/DX12/Metal, ASTC on Apple Silicon/mobile, with BC1/BC3 fallback).
2. Placing the transcoder behind the `basis` Cargo feature flag ensures static/embedded builds do not incur C++ compilation overhead or binary bloat when compressed textures are unneeded.
3. It supports direct loading of pre-compressed BCn/ASTC DDS and KTX2 textures without redundant transcoding.

### Rejected alternatives

- **`S-P-014-direct-compressed-ktx2-ingestion`**: Rejected because a container-only parser cannot transcode universal `.basis` or UASTC/ETC1S texture assets, requiring developers to pre-bake format-specific assets for every GPU architecture.
- **`S-P-014-uncompressed-only-decoder`**: Hard constraint failure; decompressing textures to uncompressed 32-bit RGBA8 wastes VRAM (4:1 nominal bloat vs BC7) and fails the explicit prompt requirement.

### Evidence summary

Block-compressed texture formats reduce VRAM footprint by 4:1 (BC7/ASTC 4x4) to 8:1 (BC1) compared to uncompressed RGBA8 (`[INFERENCE]` from fixed nominal bit-rate specifications). Runtime transcoding throughput will be benchmarked on target hardware.

### Key assumptions

- Host build environment has a standard C++ compiler available when building with `--features basis`.
- Target GPU drivers expose BC7 or ASTC block-compressed texture capability.

### Risks and mitigations

- **Risk:** Non-multiple-of-4 image dimensions fail hardware block compression requirements.
- **Mitigation:** Implement image padding/clamping during decode/transcode to ensure valid 4x4 block extents.

### Validation actions

1. Benchmark transcode throughput (MB/s) for UASTC -> BC7 and ETC1S -> BC1 on target CPU.
2. Integration test loading `.basis` and KTX2 assets in Example 6 (Sponza KTX2) on Vulkan, DX12, and Metal.

## Implementation evidence

Status on 2026-09-05; the historical selected solution above is unchanged.

- [Runtime decoding](../crates/ez-gfx-runtime/src/texture.rs) now passes explicit native targets for universal KTX2. [Shared selection](../crates/ez-gfx-runtime/src/texture/basis.rs) preserves source sRGB and chooses ETC1S BC1/BC3 or UASTC BC7 on BC devices, ASTC on ASTC-only devices, and RGBA8 otherwise. The private standalone wrapper reads encoding/alpha/sRGB metadata; native BC1/BC3 build support is enabled.
- [Optional features](../crates/ez-gfx-runtime/Cargo.toml) separate `ktx2` parsing from `basis` native transcoding. Default normal dependencies include neither; combined features include both without a shader compiler or Basis encoder. Native transcoder implementation uses `basisu_c_sys` plus the private standalone bridge rather than the candidate's named `basis-universal` crate.
- [DDS](../crates/ez-gfx-runtime/src/texture/dds.rs) and [raw](../crates/ez-gfx-runtime/src/texture/raw.rs) ingest supported native mip layouts. ABI 23 reserves sources 8/9; no new export or descriptor layout is required. Direct ingestion validates the whole chain before payload copies. Universal output bounds are checked before native decompression/transcoding.
- R/Rg KTX2 `Auto` selects canonical RGBA8, preserving channel values and transfer metadata. Explicit compressed R/Rg remains unsupported without expanding the backend swizzle/re-encoding contract. Zstd/Zlib, wider BC/ASTC formats, HDR, arrays/cubes/3D, and native-container conversion are evaluated exclusions; [the texture contract](../docs/textures.md) lists exact restrictions.
- The runtime still transcodes into owned CPU mip buffers before staging, not directly into mapped staging as the candidate proposed. The existing owner-thread allocation/transfer seam is retained.

### Decoder proof

Scoped runtime texture/transcode tests passed with no features (27 tests), `ktx2` only (29), `basis` only (28), and combined features (38). Coverage includes valid/malformed DDS and raw chains, direct KTX2 geometry, exact explicit targets, actual ETC1S/UASTC R/Rg decoded pixels, source color metadata across standalone/KTX2 containers, and pre-native output bounds. Shared target-policy/unit tests passed (3); standalone metadata validation passed (1). FFI streaming/layout/version tests verify the ABI cutover separately. No GPU sampling claim follows from these decoder results.

### Warm decode measurement

Measured 2026-09-05 on AMD Ryzen 9 5950X, Windows x64, Rust 1.88.0 / LLVM 20.1.5 (`x86_64-pc-windows-msvc`), default Cargo release profile. An isolated development encoder converted the repository's `examples/02_textured_cube/cube.png` (1024×1024) using `basisu_c_sys` 0.9.0 sRGB defaults: UASTC LDR 4×4 without Zstd and ETC1S. Encoding and image loading were outside timing; the encoder feature was not added to runtime dependencies.

Each row timed 128 decodes of retained input bytes on a Rayon pool after one warm-up decode. Times include decode/transcode, output allocation, and scheduling. Output MiB/s means total native mip bytes produced divided by 1,048,576 and elapsed wall time, not compressed-input throughput. UASTC input/output per job: 1,048,768/1,048,576 bytes; ETC1S input/BC1 output: 100,345/524,288 bytes.

| Decode target | Threads | Jobs | Wall ms | Output MiB/s |
| --- | ---: | ---: | ---: | ---: |
| UASTC → BC7 | 1 | 128 | 1437.868 | 89.021 |
| UASTC → BC7 | 4 | 128 | 432.785 | 295.759 |
| UASTC → BC7 | 8 | 128 | 255.881 | 500.232 |
| ETC1S → BC1 | 1 | 128 | 444.621 | 143.943 |
| ETC1S → BC1 | 4 | 128 | 127.984 | 500.061 |
| ETC1S → BC1 | 8 | 128 | 72.867 | 878.313 |

These are warm, single-machine measurements, not cold-load results, baseline speedup claims, or full Sponza/streaming performance. GPU upload, staging, frame time, and cross-platform integration remain separate evidence. The throwaway harness was removed; two 32×32 derived UASTC fixtures remain for color/channel regressions with generation provenance in the test source.
