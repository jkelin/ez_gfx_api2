# ez-gfx-backend-dx12

Direct3D 12 HAL using `windows`. It owns devices/queues/fences, placed resources, shader-visible SRV descriptors, texture transitions, flip-model swapchains, indirect-buffer uploads, and GPU readback. Hosts provide HWND handles; native handles remain backend-local.