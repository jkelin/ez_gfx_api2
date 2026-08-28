# ez-gfx-hal

Backend-neutral contracts for capabilities, resources, synchronization, allocation, barriers, presentation, and transfer. It depends on `ez-gfx-core`; native backend crates implement its boundary, and runtime/FFI consume the neutral contracts without native handles. Compiler artifacts supply target-native layouts to runtime/HAL.