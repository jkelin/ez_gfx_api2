# ez-gfx-artifact

Compiler-free `.ezshader` container types shared by the offline compiler and runtime. It validates bounded sections, target variants, semantic metadata, provenance, and BLAKE3 execution digests. Runtime depends only on this crate; it never links Slang.
