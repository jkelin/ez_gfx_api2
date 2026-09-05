# C textured cube

Minimal ABI v23 example: static cube positions, normals, indices, and one primitive record are uploaded through `include/ez_gfx_api.h`. Compute writes the indexed-indirect command; graphics consumes it and presents a normal-colored cube.

## Build

From the repository root:

```powershell
cargo build -p ez-gfx-ffi -p ez-gfx-compiler
cmake -S examples/c/textured_cube -B target/c-examples/textured_cube-build
cmake --build target/c-examples/textured_cube-build --config Debug
```

CMake defaults to workspace products. External packages may set:

- `EZ_GFX_INCLUDE_DIR`: directory containing `ez_gfx_api.h`
- `EZ_GFX_FFI_LIBRARY`: Windows import library
- `EZ_GFX_FFI_DLL`: runtime DLL copied beside the executable
- `EZ_GFX_COMPILER`: packaged `ez-gfx-compile.exe`
- `EZ_GFX_SHADER_ARTIFACT`: generated artifact path

For extracted Windows packages:

```powershell
$Runtime = "C:\sdk\ez-gfx-runtime-x86_64-pc-windows-msvc-0.1.0"
$Compiler = "C:\sdk\ez-gfx-compiler-x86_64-pc-windows-msvc-0.1.0"
$Build = Join-Path $PWD "target\c-examples\textured-cube-package"
cmake -S examples/c/textured_cube -B $Build `
  -DEZ_GFX_INCLUDE_DIR="$Runtime" `
  -DEZ_GFX_FFI_LIBRARY="$Runtime\ez_gfx_ffi.dll.lib" `
  -DEZ_GFX_FFI_DLL="$Runtime\ez_gfx_ffi.dll" `
  -DEZ_GFX_COMPILER="$Compiler\ez-gfx-compile.exe" `
  -DEZ_GFX_SHADER_ARTIFACT="$Build\textured_cube.ezgfxshader"
cmake --build $Build --config Release
```

CMake invokes `ez-gfx-compile` directly on `textured_cube.slang` with SPIR-V, DXIL, and Metal targets in development mode; no shader manifest or artifact is tracked.

Run one bounded frame and require a nonempty raw 640×480 RGBA snapshot:

```powershell
target/c-examples/textured_cube-build/Debug/textured_cube.exe --backend vulkan --artifact target/c-examples/textured_cube-build/Debug/textured_cube.ezgfxshader --max-frames 1 --snapshot target/c-examples/vulkan.rgba
target/c-examples/textured_cube-build/Debug/textured_cube.exe --backend dx12 --artifact target/c-examples/textured_cube-build/Debug/textured_cube.ezgfxshader --max-frames 1 --snapshot target/c-examples/dx12.rgba
```

The host is intentionally Win32-only. The C ABI accepts a borrowed `CAMetalLayer`, but creating and retaining one requires Objective-C; a plain portable C Metal host would fake ownership. CI compiles the ABI header as C11 and C++17 on every OS and builds/links this sample on Windows. The Vulkan row runs with its configured software ICD. GitHub-hosted Windows does not guarantee a D3D12 feature-level 12.1 adapter, so that row is explicitly compile-only; run the DX12 command above on a hardware-capable host.
