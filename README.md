# Easy Graphics API

Abstraction layer over modern graphics APIs like Vulkan, DX12 and Metal. It provides unified interface to high-performance graphics using modern techniques within a compact API.

The intent of this library is to limit the user to only the high performance techniques and APIs which allows us to provide more convenience and safety. You will write fast code and you will like it.

See [Examples](examples) for some code.

## Features

- Integrated [Slang shading language](https://shader-slang.org/) compiler with type-safe named buffer bindings
  - Render targets and buffers are addressed by names rather than binding slots
  - Shader reflection is used to automatically bind resources to the correct slots
  - Shaders can be precompiled including reflected information
- [Texture management](docs/textures.md)
  - Managed growing texture heap for read only textures (not render targets) and bindless rendering
  - Textures are loaded asynchronously with a parallel decoder supporting multiple common formats
  - Textures are uploaded over a dedicated hardware transfer queue to the GPU to avoid stalling the graphics queue
  - Support for KTX2 Basis Universal texture compression
  - Asynchronous callback system notifiying the user that they can unload textures from memory after they have been uploaded to the GPU
- [Geometry management](docs/geometry.md)
  - Similiar to textures, geometry is loaded asynchronously and uploaded to the GPU over a dedicated transfer queue
  - Single global index heap and multiple user-defined vertex heaps (ideally one for each vertex layout)
  - Allows for efficient geometry (mesh/model) management and rendering
- Render graph
  - Each frame the user defines draw/compute operations in sequence and the library figures out dependencies based on render target and buffer usage
  - Automatic synchronization of resources between passes
  - (not yet implemented) Render graph allows for render target aliasing reducing memory usage
- Unified buffer management
  - User defined transient buffers each frame, these then rely on BAR to be synchronized to the GPU
  - This is useful for dynamic data as well as Multidraw Indirect buffers
- Multidraw Indirect bindless rendering only
  - Dramatically reduced rendering API surface while retainig the high performace path
  - MDI buffers can be filled on both CPU and GPU with simple APIs
  - Strong distinction between Texture and Render Target enables further API simplification
- Somewhat safe Rust API and C bindings


## Prior work

- [No Graphics API article](https://www.sebastianaaltonen.com/blog/no-graphics-api) and it's Vulkan based implementation [No GFX project](https://github.com/leotmp/no_gfx_api)
  - More elegant and simpler abstraction over modern graphics hardware. However the Vulkan implementation relies on some very recent Vulkan features with poor hardware support. It's also less safe (because it's more general purpose) and has worse tooling support because of reliance of GPU pointers.
- Established RHI abstractions like [bgfx](https://github.com/bkaradzic/bgfx), [wgpu](https://github.com/gfx-rs/wgpu), [Diligent Engine](https://github.com/DiligentGraphics/DiligentEngine) are more general purpose and focus on older features such as CPU driven rendering and resource management. Frequently with these libraries bindless rendering is a cutting edge capability with poor support, while in reality this feature has been cornerstone of high performance graphics for the last 15 years. Specifically for WGPU this is also influced by the fact that browsers have been dragging their feet with bindless support and as of 2026 it's still not available within WebGPU.

## Abstractins

- Context: The main object which owns the graphics device and all resources
- Surface: Binding between Context and window or headless surface
- Frame: A single frame of rendering for a particular surface. Encapsulates the render graph which is executed when the frame is submitted.
  - Framerate is not necesserily tied to window refresh rate, in fact it is a good practice to run the render loop on a separate thread.
  - The Frame is technically thread safe, but if you are using it from multiplle threads you are doing something wrong. You can fill up buffers from multiple threads before you begin a frame, but during frame construction you should just bind those buffers
  - Each frame produces one swap of the swapchain
- Buffer: transient array of data, consumed by a frame, that will be synchronized to the GPU when the frame is submitted.
  - Buffers are automatically recycled across frames when it's safe to do so
  - Counted Buffers also have a count u32 at the start of the buffer, this is ideal for Multidraw Indirect buffers which can be filled on the GPU and then consumed by the GPU.
- Resource heaps
  - Automatically growing collections of resources asynchronously filled on the CPU and consumed in a bindless fashion inside shaders
  - Indexed by handles which are then reusable on the GPU (you pass the handles through buffers)
  - Texture heap: stores read-only texture data including mip chains and samplers
  - Index heap: used for storing index buffers for uploaded geometry
  - Vertex heaps: user defined heaps for storing individual vertex arrays for geometry
- Render targets: GPU resident transient images into which rendering is done
  - Bound to individual frames and reused
  - Can request read back onto the CPU
