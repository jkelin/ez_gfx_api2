# P-027: Artifact integrity and provenance policy

## Problem

Define the trust contract for `.ezgfxshader` and related precompiled runtime inputs beyond structural parsing. P-006 records hashes and compiler identity, but no policy defines authenticity ownership, trusted provenance, key handling, or behavior for absent/invalid signatures.

## Prompt context

Compiler and runtime are separate deliverables. Runtime loads multi-backend artifacts without Slang, validates external binary inputs, and must fail before unsafe backend calls. No remote telemetry or service is requested.

## Constraints and acceptance criteria

- Distinguish structural integrity, compatibility, and authenticity.
- Digests cover every execution-relevant section and canonical metadata.
- Versioned provenance supports reproducibility and diagnostics.
- Runtime never implies authenticity from an unkeyed hash.
- Any signature/host-verification path defines key ownership, failure behavior, optionality, and offline deployment.

## Dependencies

- Depends on P-001 deployment, P-005 compiler identity, P-006 artifact sections, and P-025 Metal provenance.
- Informs P-007 cache invalidation, P-019 reproducibility, and P-020 release gates.

## Unresolved questions

- Is authenticity host-owned, runtime-verified, or deployment-selectable?
- What compiler/tool/include/source/environment identities are required?
- Does a signature cover the container, section manifest, or external deployment manifest?
- What happens when provenance is absent, unknown, unverifiable, or incompatible?

## Candidate solutions

### S-P-027-host-owned-integrity: Runtime structural validation and unkeyed content digests; host owns authenticity

#### Approach and integration

Runtime validates bounds, schema, hashes, target/interface compatibility, and compiler identity. Package managers or application deployment systems authenticate the complete artifact externally; runtime exposes provenance for diagnostics but does not verify signatures.

#### Performance evidence

Avoids cryptographic verification and key-store I/O at runtime; exact startup savings and package overhead are unknown. Validation cost depends on artifact size and hash algorithm and must be measured with representative bundles.

#### Tradeoffs and failure modes

Small offline-capable runtime and simple key ownership. A compromised or careless host can replace a validly structured artifact; this candidate is disqualified where runtime-level authenticity is a hard security requirement.

#### Sources

- [Rust `sha2` crate documentation](https://docs.rs/sha2/latest/sha2/) — digest API boundary.
- [Vulkan shader module validation](https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateInfo.html) — structural input validation context.

### S-P-027-runtime-signature-verification: Optional runtime signature verification over canonical artifact bytes

#### Approach and integration

Compiler signs the canonical `.ezgfxshader` container or deployment manifest. Runtime accepts a configured trust store/key policy, verifies the signature before parsing execution data, then validates archive structure and hashes. Offline deployments package public keys with the application.

#### Performance evidence

Adds signature verification CPU time and key-material storage; exact latency and artifact-size overhead are unknown until measured for representative artifact sizes and chosen algorithm/hardware.

#### Tradeoffs and failure modes

Provides a standalone authenticity boundary, but key rotation, revocation, compromise recovery, and platform-specific secure storage become product responsibilities. Missing/invalid signatures must fail closed when policy requires them.

#### Sources

- [Microsoft Artifact Signing overview](https://learn.microsoft.com/en-us/azure/artifact-signing/overview) — artifact signing trust model.
- [RustCrypto signature traits](https://docs.rs/signature/latest/signature/) — pluggable signature verification interface.

### S-P-027-signed-deployment-manifest: Host-signed manifest authenticating artifact set and provenance

#### Approach and integration

A deployment manifest lists artifact digests, schema/compiler/tool identities, target products, and policy. The host/package installer signs the manifest; runtime verifies the manifest and checks each local artifact digest before use.

#### Performance evidence

One manifest signature amortizes verification across artifacts, but adds manifest I/O and deployment coupling. Exact verification/startup overhead is unknown until measured for package sizes and artifact counts.

#### Tradeoffs and failure modes

Coordinates shader, native library, and cache versions; however, partial installs, manifest/artifact mismatch, and key distribution are failure modes. Runtime must reject missing or extra execution artifacts according to policy.

#### Sources

- [in-toto specification](https://github.com/in-toto/attestation) — signed supply-chain metadata model.
- [SLSA provenance](https://slsa.dev/spec/v1.0/provenance) — build provenance fields and verification context.

## Performance comparison

| Candidate | Runtime/startup | Deployment/scaling | Constraint fit | Evidence |
|---|---|---|---|---|
| Host-owned integrity/digests | Lowest runtime crypto overhead; exact cost unknown | Simple, but host trust required | Moderate; fails standalone authenticity requirement | Rust digest/API docs |
| Runtime signature verification | Added verification latency/bytes unknown; scales per artifact | Strong standalone trust; key lifecycle burden | High where authenticity is required | Microsoft/Rust signature docs |
| Signed deployment manifest | One verification amortized across package; I/O unknown | Coordinates multi-artifact releases; tighter packaging coupling | High for package-level trust | in-toto/SLSA specs |

## Selected solution

### Selection

`S-P-027-host-owned-integrity`: Runtime structural validation and unkeyed content digests; host owns authenticity.

### Selection rationale

The prompt requires compiler/runtime separation and validated binary inputs, but does not require a new key-management service or remote trust system. Host-owned authenticity plus mandatory runtime structural validation keeps the runtime offline, small, and independent of signing infrastructure while explicitly refusing to treat an unkeyed digest as authentication. Full execution-content hashes and provenance still protect compatibility/reproducibility checks.

### Rejected alternatives and reversal conditions

- **`S-P-027-runtime-signature-verification`**: Rejected as default because it introduces key distribution, rotation, revocation, secure storage, and cryptographic runtime dependencies not required by the prompt. Reconsider if deployments require the runtime itself to reject artifacts from an untrusted host.
- **`S-P-027-signed-deployment-manifest`**: Rejected as default because it couples runtime acceptance to a package installer/manifest lifecycle. Reconsider if the release must authenticate coordinated shader, native-library, and cache sets.

### Evidence and unknowns

Bounded parsing and full content digests are mandatory; digest/parse latency, artifact-size overhead, and host package verification cost remain unknown until measured on representative bundles.

### Assumptions and risks

- The host/package deployment boundary is trusted to authenticate artifacts when authenticity is needed.
- Runtime reports authenticity as “host-verified” or “not verified,” never “signed” merely because a digest matches.
- A compromised host can replace structurally valid artifacts; this selection is invalid if standalone runtime authenticity becomes a hard requirement.

### Validation actions

1. Fuzz malformed/truncated/overlapping artifacts and verify rejection before backend calls.
2. Round-trip artifacts across compiler/runtime versions and verify every execution section is covered by the digest.
3. Test explicit absent/unknown/incompatible provenance states and record startup parsing/hash costs.
