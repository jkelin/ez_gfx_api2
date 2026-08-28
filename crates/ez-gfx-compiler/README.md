# ez-gfx-compiler

Offline shader compiler using the `shader-slang` Rust bindings for in-process Slang sessions. It maps validated requests to SPIR-V, DXIL, and Metal targets, captures typed reflection/provenance, and emits `ez-gfx-artifact` data. Apple `xcrun metal` postprocesses Slang-generated MSL into metallib; no Slang executable is invoked.

The compiler requires the native Slang shared library at build/runtime. `shader-slang` discovers a Vulkan SDK installation or `SLANG_DIR`/`SLANG_LIB_DIR`; DXIL additionally requires `dxil.dll` and `dxcompiler.dll`. Shipping runtime packages must depend only on `ez-gfx-artifact`, not this crate or Slang libraries.
