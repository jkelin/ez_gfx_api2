# ez-gfx-runtime

Compiler-free execution boundary for contexts, frame recording, indirect commands, compiled frame graphs, shader artifacts, native texture decoding/residency, descriptors, render targets, caches, uploads, and host-polled events. It depends on `ez-gfx-core`, `ez-gfx-hal`, and `ez-gfx-artifact`; native backends implement HAL. It never links Slang or performs JIT compilation.