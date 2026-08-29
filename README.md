# ez-gfx-api2

Rust/Cargo migration of the original ez_gfx_api with Vulkan, Direct3D 12, and Metal backends; universal Slang shaders; compressed textures; and a compiler-free runtime.

## Workspace flow

`ez-gfx-core` defines backend/compiler-neutral semantic IDs, layouts, handles, and capability contracts. `ez-gfx-hal` defines native-lowering boundaries. Backend crates implement HAL with `ash`, `windows`, and `objc2-metal`. `ez-gfx-compiler` uses `shader-slang`/slang-rs in-process bindings to compile one source to SPIR-V, DXIL, and Metal; Apple `xcrun metal` postprocesses MSL into metallib. It emits versioned `ez-gfx-artifact` bundles. `ez-gfx-runtime` loads those bundles without Slang or JIT compilation and decodes KTX2 UASTC and BasisLZ ETC1S through `basisu_c_sys`. `ez-gfx-assets` validates compressed texture payloads. `ez-gfx-ffi` exposes the validated C ABI.

Runtime dependencies must not include the compiler or Slang native libraries. Hosts own artifact authenticity, filesystem policy, and event polling. See `plan/COMPONENTS.md` and `plan/SOLUTIONS.md` for canonical decisions and validation gates.

## Tasks and examples

Install [mise](https://mise.jdx.dev/) and run `mise tasks` for the portable build, test, example, and packaging commands. `mise run examples-smoke` exercises all six migrated Rust example paths; `mise run example-3` runs one example. See `examples/README.md` for the original-source inventory and behavior mapping.

Each program owns its winit raw-handle host code. Win32 passes HWND/HINSTANCE to Vulkan or DX12. macOS attaches and retains a scale-aware CAMetalLayer, selects Metal, then releases the layer after surface destruction. Set `EZ_GFX_BACKEND` to `vulkan`, `dx12`, or (on macOS) `metal`; unsupported host/backend combinations fail at the boundary.

`mise run package` packages the host target. Append `-- <target> <version> <output>` to override its defaults. The Rust `xtask` implementation replaces the former platform-specific shell scripts, verifies runtime/compiler isolation, requires native compiler libraries from `SLANG_DIR` or `VULKAN_SDK`, writes sorted SHA-256 manifests, and creates ZIP archives on Windows or `.tar.gz` archives elsewhere.
