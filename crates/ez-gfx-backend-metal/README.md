# ez-gfx-backend-metal

Apple Metal HAL using `objc2-metal`. It owns admitted devices, command queues, synchronization, heaps, allocator-backed textures, blit upload/readback, compiler-produced metallib libraries, and CAMetalLayer drawable presentation while retaining its CAMetalLayer through the surface lifetime. Display synchronization is disabled so presentation throughput is not capped to vertical blank.

Surfaces and graphics pipelines use BGRA8 sRGB, matching the frame graph. Shader colors and clears are linear; captured RGBA bytes retain sRGB encoding and unchanged alpha. Runtime loading accepts offline metallib, not development MSL.

Headless Apple M2 Pro validation covers compressed sampling, partial updates, coarse/fine readiness, retirement, batching, cancellation, and failed producers. See [exact texture evidence and remaining limits](../../docs/textures.md#verification-and-remaining-evidence).