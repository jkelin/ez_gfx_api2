# C textured cube

Minimal Vulkan ABI 39 example: GLFW owns a no-client-API window while ez-gfx creates its surface from tagged Win32, Xlib, or Wayland handles and queries the initial framebuffer extent. Typed geometry heaps persist; structured and counter buffers are reacquired per frame. A creator-thread callback copies borrowed snapshot bytes from the terminal frame only.

## Build

From the repository root:

```powershell
cargo build -p ez-gfx-ffi -p ez-gfx-compiler
cmake -S examples/02_textured_cube_c -B target/c-examples/textured_cube-build
cmake --build target/c-examples/textured_cube-build --config Debug
```

CMake defaults to workspace products. External packages may set:

- `EZ_GFX_INCLUDE_DIR`: directory containing `ez_gfx_api.h`
- `EZ_GFX_FFI_LIBRARY`: platform link library
- `EZ_GFX_FFI_RUNTIME`: runtime library copied beside the executable
- `EZ_GFX_COMPILER`: packaged shader compiler
- `EZ_GFX_SHADER_ARTIFACT`: generated artifact path

For extracted Windows packages:

```powershell
$Runtime = "C:\sdk\ez-gfx-runtime-x86_64-pc-windows-msvc-0.1.0"
$Compiler = "C:\sdk\ez-gfx-compiler-x86_64-pc-windows-msvc-0.1.0"
$Build = Join-Path $PWD "target\c-examples\textured-cube-package"
cmake -S examples/02_textured_cube_c -B $Build `
  -DEZ_GFX_INCLUDE_DIR="$Runtime" `
  -DEZ_GFX_FFI_LIBRARY="$Runtime\ez_gfx_ffi.dll.lib" `
  -DEZ_GFX_FFI_RUNTIME="$Runtime\ez_gfx_ffi.dll" `
  -DEZ_GFX_COMPILER="$Compiler\ez-gfx-compile.exe" `
  -DEZ_GFX_SHADER_ARTIFACT="$Build\textured_cube.ezgfxshader"
cmake --build $Build --config Release
```

CMake fetches pinned GLFW and invokes `ez-gfx-compile` for SPIR-V, DXIL, and Metal development variants.

Run one bounded frame and require a nonempty raw 640×480 RGBA snapshot:

```powershell
target/c-examples/textured_cube-build/Debug/textured_cube.exe --backend vulkan --artifact target/c-examples/textured_cube-build/Debug/textured_cube.ezgfxshader --max-frames 1 --snapshot target/c-examples/vulkan.rgba --hidden
```

The example is Vulkan-only because GLFW exposes the portable native handles required by Vulkan without adding any window-system dependency to library crates. Hidden runs remain non-activating. Minimized zero-size framebuffers pause rendering until restored; nonzero size changes use `ez_gfx_surface_resize`.
