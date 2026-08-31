# ez-gfx-compiler

Offline shader compiler using `shader-slang` in-process sessions and a `clap` CLI. It emits format-v3 `.ezgfxshader` artifacts containing SPIR-V, DXIL, Metal products, canonical reflection, provenance, and target-specific compatibility. Apple metallib builds record macOS platform, architecture, deployment minimum, SDK, Metal language, metallib contract, and Apple toolchain identity gathered at build time; non-Apple development builds emit MSL with an explicit language contract. Each stage owns one artifact-internal entry point.

The Rust examples call this library from Cargo `build.rs`; the C structured cube invokes the CLI from CMake. Both generate artifacts in build output rather than tracking compiled shader files.

Compiler and build environments require the native Slang shared library. `shader-slang` discovers a Vulkan SDK installation or `SLANG_DIR`/`SLANG_LIB_DIR`; DXIL additionally requires `dxil.dll` and `dxcompiler.dll`. Runtime packages depend on `ez-gfx-artifact`, not this crate or native compiler libraries.
