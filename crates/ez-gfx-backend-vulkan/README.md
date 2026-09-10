# ez-gfx-backend-vulkan

Vulkan 1.3 HAL using `ash`. It owns instance/device/queues, timeline uploads, allocator-backed images, update-after-bind sampled-image descriptors, layout transitions, low-latency swapchains, and GPU readback. Presentation prefers mailbox, then immediate, with required FIFO as the compatibility fallback. Native handles stay behind runtime/FFI boundaries.