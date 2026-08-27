# P-029: Cross-platform native build and distribution

## Problem

Select how Rust packages, backend bindings, Slang tooling, Basis Universal, DXC, Apple tools, C ABI libraries, and runtime-only artifacts are built and distributed per supported platform. P-001 separates Cargo packages and target dependencies, but no delivery policy defines build hosts, linkage, redistributable contents, feature leakage proof, or signing/notarization ownership.

## Prompt context

The new project uses Rust/Cargo and must support Vulkan, DX12, Metal, optional Basis Universal compression, universal offline Slang compilation, and a runtime that does not bundle Slang. It retains a C/C# path where practical.

## Constraints and acceptance criteria

- Runtime distributions exclude Slang/compiler-only native libraries by construction and audit.
- Each target has a documented build-host/toolchain matrix and reproducible Cargo package/feature invocation.
- Vulkan, DX12, Metal, Basis, compiler, and FFI dependencies have explicit static/dynamic/platform-supplied linkage and redistribution policies.
- Feature-off runtime builds exclude optional decoder/C++ linkage.
- C ABI names, architectures, runtime dependencies, symbols, and package layout are deterministic.
- Apple-only shader tools are separated from runtime distribution.

## Dependencies

- Depends on P-001 workspace, P-002 FFI, P-003 backend bindings, P-005 compiler tools, P-014 Basis linkage, P-025 Metal products, and P-027 artifact provenance.
- Feeds P-019 CI profiles and P-020 cutover.

## Unresolved questions

- Initial target triples, minimum OS versions, architectures, and build hosts?
- Which dependencies are static, dynamic, delay-loaded, or platform-supplied?
- Are compiler tools standalone archives, Cargo-installed binaries, or host-built packages?
- How are runtime-only, compiler, Basis-enabled, backend-specific, and FFI artifacts named without feature leakage?
- Which licenses, notices, redistributables, symbols, signing, and notarization records accompany each package?

## Candidate solutions

### S-P-029-source-built-per-target: Reproducible source builds per target and separate runtime/compiler packages

#### Approach and integration

Build each Cargo package from pinned source/lockfile on a target-appropriate CI host. Produce separate runtime, FFI, compiler/tool, and optional Basis artifacts; use target-native SDKs/linkers and package manifests. Runtime artifacts undergo dependency-tree and binary-import audits.

#### Performance evidence

Build duration, cache hit rate, package size, startup/load time, and CI matrix cost are unknown until measured per target and clean/incremental condition. Source builds maximize reproducibility control but require every native toolchain in CI.

#### Tradeoffs and failure modes

Strong feature/linkage control and clear licenses; higher CI/toolchain maintenance. Failures include SDK drift, missing platform tools, non-reproducible native builds, and accidental host-target linkage.

#### Sources

- [Cargo build scripts](https://doc.rust-lang.org/cargo/reference/build-scripts.html) — native build integration.
- [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html) — reproducible build configuration.
- [Rust platform support](https://doc.rust-lang.org/rustc/platform-support.html) — target/toolchain matrix.

### S-P-029-prebuilt-toolchain-artifacts: Centrally built signed archives and platform installers

#### Approach and integration

CI produces versioned archives/installers containing runtime/FFI binaries, manifests, notices, and separately downloadable compiler/tool packages. Native dependencies are bundled or explicitly platform-supplied per package. Consumers verify package signatures/digests before installation.

#### Performance evidence

Consumer build time approaches download/install time; installed size, extraction time, and archive matrix cost are unknown until measured. Central builds can improve reproducibility but add release-pipeline and artifact-retention cost.

#### Tradeoffs and failure modes

Predictable consumer experience and controlled toolchain provenance; platform matrix and signing/notarization burden increase. Failures include stale platform packages, wrong architecture, incomplete runtime redistributables, and signature/key rotation errors.

#### Sources

- [Cargo package](https://doc.rust-lang.org/cargo/commands/cargo-package.html) — package creation.
- [Microsoft Artifact Signing](https://learn.microsoft.com/en-us/azure/artifact-signing/overview) — signed artifact model.
- [Apple code signing](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution) — macOS distribution/notarization.

### S-P-029-platform-package-manager: Native ecosystem packages with host-supplied graphics/runtime dependencies

#### Approach and integration

Publish Rust crates and platform-native packages (NuGet, vcpkg/Homebrew-like outputs) that depend on installed Vulkan loaders, DX12/Windows components, Metal frameworks, and separately installed compiler tools. Package metadata declares feature/backend requirements and ABI assets.

#### Performance evidence

Package-manager resolution/install time, dependency reuse, and runtime startup are unknown and depend on host caches and package ecosystem. Host-supplied libraries can reduce package size but increase environment variability.

#### Tradeoffs and failure modes

Best ecosystem integration and less bundled duplication; difficult cross-platform reproducibility and version coordination. Missing loaders/SDKs, package conflicts, and host ABI drift must fail with actionable diagnostics.

#### Sources

- [Cargo publishing](https://doc.rust-lang.org/cargo/reference/publishing.html) — crate publication boundary.
- [Vulkan loader documentation](https://github.com/KhronosGroup/Vulkan-Loader) — platform loader model.
- [Microsoft D3D12 deployment](https://learn.microsoft.com/en-us/windows/win32/direct3d12/directx-12-programming-guide) — platform API boundary.

## Performance comparison

| Candidate | Build/install/startup | Reproducibility and operations | Constraint fit | Evidence |
|---|---|---|---|---|
| Source-built per target | Consumer install small; CI build/matrix cost unknown | Strong source/lockfile control; native toolchain burden | High | Cargo/Rust official docs |
| Prebuilt signed archives/installers | Consumer build fast; download/install size/time unknown | Strong central provenance; release/signing burden | High | Cargo/Microsoft/Apple docs |
| Platform package managers | Host dependency reuse; resolution/startup unknown | Variable host state and version conflicts | Moderate | Cargo/Vulkan/Microsoft docs |

## Selected solution

### Selection

`S-P-029-prebuilt-toolchain-artifacts`: Centrally built signed archives and platform installers.

### Selection rationale

This best satisfies deterministic C ABI delivery and the requirement that runtime consumers do not receive compiler tooling. A release pipeline can publish separate runtime/FFI, compiler, and optional Basis packages with explicit architecture, dependency, notices, and feature manifests. Signing/notarization and import/dependency audits happen before publication, while source/lockfile builds remain the reproducibility input.

### Rejected alternatives and reversal conditions

- **`S-P-029-source-built-per-target`**: Rejected as the consumer delivery model because every consumer would need the complete native toolchain matrix and would face greater environment variability. Retain as the release-build method and reconsider as the distribution model for expert integrators.
- **`S-P-029-platform-package-manager`**: Rejected as the primary model because host-supplied loaders/SDKs and package conflicts make deterministic cross-platform runtime delivery weaker. Reconsider for ecosystem-specific secondary packages after the canonical archives are stable.

### Evidence and unknowns

Cargo supports package/build boundaries; Microsoft and Apple documents establish platform signing/distribution mechanisms. Archive size, install time, CI matrix cost, native dependency count, startup time, and signing/notarization latency remain unknown until measured per target and architecture.

### Assumptions and risks

- Release CI can access target-native SDKs, linkers, Apple signing/notarization, and Windows signing credentials.
- Signed archives include license notices, symbols policy, dependency manifests, and a compiler-free runtime proof.
- Risks include wrong architecture, stale packages, SDK drift, incomplete redistributables, key rotation, and accidental feature leakage.

### Validation actions

1. Build runtime-only, FFI, compiler, and Basis-enabled artifacts for each declared target; inspect dependency trees and binary imports.
2. Install each archive in a clean target environment and run ABI smoke, backend startup, artifact loading, and snapshot gates.
3. Verify signatures/notarization, notices, architecture labels, and reproducibility metadata; measure build/install/startup/package metrics with named environments.
