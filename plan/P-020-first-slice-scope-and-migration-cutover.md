# P-020: First-slice scope and migration cutover

## Problem

Decide the minimum viable first-slice scope for initial implementation and define the staged milestone plan for full migration, verifying cutover criteria from the Odin source codebase to the Rust destination.

## Prompt context

Full user prompt: "prepare implementation for the different todos... setup tests including snapshot tests. roughly maintain the original api".
Source evidence: The original project contains 6 examples (Triangle, Textured Cube, Compute Structured Buffer, ImGui, Helmet, Sponza KTX2) and 25 distinct TODO items in `TODO.md`.

## Constraints and acceptance criteria

- Define clear milestone boundaries and verifiable acceptance criteria covering all prompt requirements and original functionality.
- Retain first-slice vs. deferred scope definition as competing hypotheses to be compared in Step 2.
- Explicit non-goals: untracked scope creep or monolithic big-bang porting without incremental verification.

## Dependencies

- Incoming dependency: `P-020` depends on all problem files (`P-001` through `P-019`) for scope mapping and milestone sequencing.
- Outgoing dependency: None (terminal planning coordinator).

## Unresolved questions

- Which feature subset constitutes the most effective first vertical slice (e.g. Vulkan-only core vs cross-backend HAL vs headless render graph)?
- Which examples serve as cutover gates for each progressive milestone?
- Should cutover require all original examples, the C# smoke test, and every TODO, or can migration acceptance be staged by capability and platform?

## Candidate solutions

### S-P-020-vulkan-first-vertical-slice: Vulkan-first vertical slice with staged backend expansion

#### Approach and integration

Treat a Vulkan-capable end-to-end path as the initial integration hypothesis: connect a Rust workspace, one compiler/runtime boundary, one backend, representative resource and graph paths, and executable snapshot tests before adding other backends and deferred TODO capabilities. Subsequent milestones would expand platform coverage and optimization work.

#### Performance evidence

- **Delivery latency:** Unknown until measured; evaluate time from workspace bootstrap to first passing render snapshot.
- **Regression feedback:** Expected to be faster than a full matrix, but this is an inference requiring measurement against the other sequencing hypotheses.

#### Tradeoffs and failure modes

- **Tradeoffs:** Early vertical proof may expose integration defects sooner while postponing cross-backend parity.
- **Disqualifiers:** If the first backend's abstractions prevent universal Slang output or force incompatible resource semantics, the sequencing hypothesis fails.

#### Sources

- `F:/Projects/oss/ez_gfx_api/examples/` â€” 6 original example applications.
- `F:/Projects/oss/ez_gfx_api/TODO.md` â€” complete backlog of feature improvements.
- [Cargo test documentation](https://doc.rust-lang.org/cargo/commands/cargo-test.html) â€” staged test execution and package boundaries.

### S-P-020-cross-backend-horizontal-slice: Cross-backend foundation before higher-level features

#### Approach and integration

Treat all three required backend targets and universal shader outputs as the first integration boundary, validating device creation, resource formats, synchronization, and a minimal common API before implementing the full render graph and asset pipeline. Higher-level features follow after backend parity evidence.

#### Performance evidence

- **Delivery latency:** Unknown; implementation and test effort likely increases with platform matrix size (`[INFERENCE]`).
- **Coverage:** Earliest backend parity tests can expose abstraction gaps, but visual feature coverage is delayed (`[INFERENCE]`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Reduces risk of building Vulkan-only abstractions that cannot map to DX12/Metal.
- **Disqualifiers:** Requires usable hardware/toolchains for all target platforms before meaningful end-to-end validation.

#### Sources

- User prompt requirement for Vulkan, DX12, and Metal.
- `F:/Projects/oss/ez_gfx_api/README.md` â€” shader-declared pipeline and render-graph concepts.
- [Vulkan overview](https://docs.vulkan.org/guide/latest/what_is_vulkan.html) â€” backend capability context.

### S-P-020-minimal-headless-core-slice: Headless compiler/graph core before device execution

#### Approach and integration

Treat compiler output validation, serialized reflection, API-shape checks, and CPU-side render graph planning as the first slice. Attach real backend execution and image snapshots only after the intermediate representations and cutover contracts stabilize.

#### Performance evidence

- **Test runtime:** Expected to be lowest because CPU-only fixtures avoid adapter startup and GPU readback, but exact timing is unknown until measured.
- **Risk:** Hardware integration defects are detected later (`[INFERENCE]`).

#### Tradeoffs and failure modes

- **Tradeoffs:** Fast deterministic feedback and platform-independent CI.
- **Disqualifiers:** Cannot establish that resource transitions, shader binaries, or snapshots work on actual Vulkan/DX12/Metal devices.

#### Sources

- `F:/Projects/oss/ez_gfx_api/README.md` â€” render graph and shader-declared pipeline concepts.
- `F:/Projects/oss/ez_gfx_api/CONTEXT.md` â€” render graph terminology and resource edges.
- [Cargo workspaces documentation](https://doc.rust-lang.org/cargo/reference/workspaces.html) â€” workspace milestone organization.

## Performance comparison

| Candidate | Relevant performance dimensions | Constraint fit | Evidence quality | Risks |
| --------- | ------------------------------- | -------------- | ----------------- | ----- |
| Vulkan-first vertical slice with staged expansion | Earliest likely GPU snapshot; delivery and regression timing unknown | Moderate-High | Direct source evidence; timing is unknown | Vulkan-first abstractions may constrain other backends |
| Cross-backend foundation before higher-level features | Earliest backend parity; matrix cost and delivery timing unknown | High | Direct prompt evidence; timing is inference | Requires all platform toolchains/hardware |
| Headless compiler/graph core before device execution | Lowest expected test setup cost; exact timing unknown | Moderate | Direct source evidence; timing is inference | Delays hardware and visual validation |

## Selected solution

### Selection

`S-P-020-vulkan-first-vertical-slice`, followed by a completed clean ownership cutover.

### Selection rationale

The initial vertical slice established an end-to-end executable path before backend expansion. Final Rust examples use the shared `Example` host for process options, winit inversion, native window, context/surface creation, resize, input, pacing, benchmark, capture, and reporting. `Example::new` returns `(Example, Context, Surface)` so each main directly owns both graphics objects. Each procedural loop passes `&Surface` to receive `WindowFrame`, explicitly begins and configures the swapchain transaction, records through `&mut Frame`, then calls `Example::handle_frame(frame, swapchain_target)`.

The safe cutover is ownership-only: resources release through `Drop`, `Frame::finish(self)` is consuming, unfinished frames abort on `Drop`, and no compatibility aliases retain manual safe destruction or the former multiple begin/end paths. ABI 34 keeps explicit C lifecycle functions over opaque generational handles.

### Rejected alternatives

- **`S-P-020-cross-backend-horizontal-slice`**: Rejected for the first slice because requiring all three platform toolchains and usable adapters delays executable feedback; retained as the expansion gate for cross-backend parity.
- **`S-P-020-minimal-headless-core-slice`**: Rejected as the sole first slice because CPU-only graph/reflection tests cannot validate actual GPU resource transitions, shader binaries, or visual output; retained as a supporting preflight test layer.

### Evidence summary

Delivery latency and defect-rate comparisons remain unknown until measured. The selection is based on prompt-derived delivery criteria and the original project's six examples and snapshot harness, not invented performance measurements.

### Key assumptions

- A Vulkan development/test environment is available for the first executable vertical path.
- Slice boundaries can be staged without preventing universal Slang metadata and API contracts from being tested early.

### Risks and mitigations

- **Risk:** Vulkan-first abstractions constrain later DX12/Metal implementations.
- **Mitigation:** Require backend-independent shader reflection and render-graph contracts as slice entry criteria, then run cross-backend conformance before declaring cutover.

### Validation actions

1. Exercise all six renderers through the shared `Example` host with caller-owned `Context` and `Surface`, inner resource scopes, explicit surface drop, error-propagating `Context::close`, and host-only publication in `Example::drop`; publication failure exits nonzero except during an active unwind.
2. Gate ABI 34 frame end/abort, one-frame buffer invalidation, opaque-handle validation, Rust wrapper drop order, and required Vulkan/DX12/Metal behavior.
