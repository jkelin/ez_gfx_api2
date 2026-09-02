# TODO

## P0

- Lower compiled transient alias assignments into backend resource placement without changing executor ordering.
- Add managed render-target creation, description, release, binding, resize/history, and graph attachment support with validated ownership, dimensions, and formats.

## P1

- Connect bounded asset workers to the public Rust and FFI APIs for context-validated, cancellable asynchronous mip/region uploads, completion events, GPU completion tracking, eviction, and native compressed formats.
- Add validated viewport and scissor state across the Rust API, ABI, Metal, Vulkan, and DX12.
- Support cross-thread explicit context destruction and observable implicit cleanup failures without native cleanup or joins from Windows TLS destructors.
- Add Linux Vulkan surfaces and native presentation coverage.
- Run Vulkan native tests on an adapter supporting required descriptor indexing, DX12 native tests on a guaranteed feature-level 12.1 adapter, and Metal native tests on a GPU-backed macOS runner.
- Correlate submission, cleanup, resource-release, upload, queue-overflow, cancellation, and device-loss failures through diagnostics or events.
- Publish an FFI parity matrix and add validated texture-update, target-lifecycle, decoder-callback, screenshot-save, graph-authoring, and native compressed-upload APIs with ABI boundary tests.

## P2

- Add managed-target, asynchronous-upload, and viewport/scissor regression suites.
- Establish GPU performance and package-size baselines.

## Tooling

- Configure Miri and integrate it with nextest.
