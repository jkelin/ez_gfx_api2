//! Binary artifact encoding and validation contract tests.
use ez_gfx_artifact::{
    AppleArchitecture, ApplePlatform, Artifact, ArtifactError, CompatibilityVersion,
    MAX_ARTIFACT_BYTES, MetalCompatibility, Provenance, Stage, Target, TargetCompatibility,
    TargetVariant,
};

fn compatibility(target: Target) -> TargetCompatibility {
    match target {
        Target::Metallib => TargetCompatibility::MetalLibrary {
            metal: MetalCompatibility {
                platform: ApplePlatform::MacOs,
                architecture: AppleArchitecture::Aarch64,
                minimum_os: CompatibilityVersion::new(14, 0),
                sdk: CompatibilityVersion::new(15, 0),
                language: CompatibilityVersion::new(3, 0),
                library: CompatibilityVersion::new(1, 0),
                toolchain: "apple-clang-16".into(),
            },
        },
        _ => TargetCompatibility::portable(target).unwrap(),
    }
}

fn variant(target: Target, stage: Stage, entry: &str, profile: &str, bytes: u8) -> TargetVariant {
    TargetVariant::new(
        target,
        stage,
        entry,
        profile,
        compatibility(target),
        vec![bytes],
    )
    .unwrap()
}

fn sample() -> Artifact {
    Artifact::new(
        br#"{"schema":1}"#.to_vec(),
        Provenance::new("slangc", "2026.16", vec!["-O2".into()], "linux-x64"),
        vec![
            variant(Target::Spirv, Stage::Vertex, "vs", "spirv_1_5", 1),
            variant(Target::Dxil, Stage::Vertex, "vs", "sm_6_5", 2),
            variant(Target::Metallib, Stage::Vertex, "vs", "metallib_3_0", 3),
        ],
    )
    .unwrap()
}

#[test]
fn round_trip_preserves_digest_and_variants() {
    let artifact = sample();
    let bytes = artifact.encode().unwrap();
    assert_eq!(Artifact::decode(&bytes).unwrap(), artifact);
}

#[test]
fn artifact_allows_only_one_entry_point_per_stage() {
    let variants = vec![
        variant(Target::Spirv, Stage::Vertex, "vs_a", "spirv_1_5", 1),
        variant(Target::Dxil, Stage::Vertex, "vs_b", "sm_6_5", 2),
        variant(Target::Metallib, Stage::Vertex, "vs_a", "metallib_3_0", 3),
    ];

    assert!(matches!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            variants
        ),
        Err(ArtifactError::DuplicateStage(Stage::Vertex))
    ));
}

#[test]
fn graphics_vertex_fragment_entries_require_all_backends() {
    let mut variants = Vec::new();
    for (stage, entry, base) in [(Stage::Vertex, "vs", 1), (Stage::Fragment, "fs", 4)] {
        variants.extend([
            variant(Target::Spirv, stage, entry, "spirv_1_5", base),
            variant(Target::Dxil, stage, entry, "sm_6_5", base + 1),
            variant(Target::Metallib, stage, entry, "metallib_3_0", base + 2),
        ]);
    }
    assert!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            variants
        )
        .is_ok()
    );

    let missing = vec![variant(Target::Spirv, Stage::Vertex, "vs", "spirv_1_5", 1)];
    assert!(matches!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            missing
        ),
        Err(ArtifactError::MissingCoverage { .. })
    ));
}

#[test]
fn rejects_duplicate_exact_variant_and_invalid_bounds() {
    let duplicate = vec![
        variant(Target::Spirv, Stage::Vertex, "vs", "spirv_1_5", 1),
        variant(Target::Spirv, Stage::Vertex, "vs", "spirv_1_5", 2),
        variant(Target::Dxil, Stage::Vertex, "vs", "sm_6_5", 3),
        variant(Target::Metallib, Stage::Vertex, "vs", "metallib_3_0", 4),
    ];
    assert!(matches!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            duplicate
        ),
        Err(ArtifactError::DuplicateVariant)
    ));
    assert!(matches!(
        TargetVariant::new(
            Target::Spirv,
            Stage::Vertex,
            "",
            "p",
            compatibility(Target::Spirv),
            vec![1],
        ),
        Err(ArtifactError::InvalidEntryPoint)
    ));
    assert!(matches!(
        Artifact::decode(&vec![0; MAX_ARTIFACT_BYTES + 1]),
        Err(ArtifactError::TooLarge)
    ));
}

#[test]
fn framed_payload_rejects_truncation_version_length_and_invalid_archive() {
    let bytes = sample().encode().unwrap();
    assert!(matches!(
        Artifact::decode(&bytes[..bytes.len() - 1]),
        Err(ArtifactError::Truncated)
    ));

    let mut wrong_version = bytes.clone();
    wrong_version[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        Artifact::decode(&wrong_version),
        Err(ArtifactError::UnsupportedVersion(u32::MAX))
    ));

    let mut wrong_length = bytes.clone();
    wrong_length[16..24].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        Artifact::decode(&wrong_length),
        Err(ArtifactError::InvalidHeader)
    ));

    let mut invalid_archive = bytes;
    invalid_archive[56..].fill(0xff);
    let digest = blake3::hash(&invalid_archive[56..]);
    invalid_archive[24..56].copy_from_slice(digest.as_bytes());
    assert!(matches!(
        Artifact::decode(&invalid_archive),
        Err(ArtifactError::InvalidArchive)
    ));
}

#[test]
fn digest_covers_the_complete_archived_payload() {
    let original = sample().encode().unwrap();
    let mut bytes = original.clone();
    bytes[56] ^= 1;
    assert!(matches!(
        Artifact::decode(&bytes),
        Err(ArtifactError::DigestMismatch)
    ));

    let mut changed = sample();
    changed.provenance.options.push("-g".into());
    assert_ne!(changed.encode().unwrap()[24..56], original[24..56]);
}
