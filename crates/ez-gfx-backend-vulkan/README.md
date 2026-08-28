# ez-gfx-backend-vulkan

Vulkan 1.3 HAL using `ash`. It owns instance/device/queues, timeline uploads, allocator-backed images, update-after-bind sampled-image descriptors, layout transitions, FIFO swapchains, and GPU readback. Native handles stay behind runtime/FFI boundaries.