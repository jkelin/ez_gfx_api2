//! Binary artifact encoding and validation contract tests.
use ez_gfx_artifact::{
    ARTIFACT_FORMAT_VERSION, AppleArchitecture, ApplePlatform, Artifact, ArtifactError,
    CompatibilityVersion, MAX_ARTIFACT_BYTES, MetalCompatibility, Provenance, Stage, Target,
    TargetCompatibility, TargetVariant,
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
fn artifact_rejects_duplicate_entry_target_products() {
    let variants = vec![
        variant(Target::Spirv, Stage::Vertex, "vs_a", "spirv_1_5", 1),
        variant(Target::Spirv, Stage::Vertex, "vs_a", "spirv_1_5", 2),
    ];

    assert!(matches!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            variants
        ),
        Err(ArtifactError::DuplicateVariant)
    ));
}

#[test]
fn global_target_subsets_round_trip_for_every_stage() {
    let variants = vec![
        variant(Target::Spirv, Stage::Vertex, "vs", "spirv_1_5", 1),
        variant(Target::Msl, Stage::Vertex, "vs", "metal_3_0", 2),
        variant(Target::Spirv, Stage::Fragment, "fs", "spirv_1_5", 3),
        variant(Target::Msl, Stage::Fragment, "fs", "metal_3_0", 4),
    ];
    let artifact = Artifact::new(
        br"{}".to_vec(),
        Provenance::new("s", "v", vec![], "t"),
        variants,
    )
    .unwrap();

    let decoded = Artifact::decode(&artifact.encode().unwrap()).unwrap();

    assert_eq!(decoded, artifact);
}

#[test]
fn every_single_stage_target_subset_is_valid() {
    for (target, profile) in [
        (Target::Spirv, "spirv_1_5"),
        (Target::Dxil, "sm_6_5"),
        (Target::Msl, "metal_3_0"),
        (Target::Metallib, "metallib_3_0"),
    ] {
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            vec![variant(target, Stage::Compute, "cs", profile, 1)],
        )
        .unwrap();
    }
}

#[test]
fn rejects_disjoint_stage_target_coverage() {
    let variants = vec![
        variant(Target::Spirv, Stage::Vertex, "vs", "spirv_1_5", 1),
        variant(Target::Dxil, Stage::Fragment, "fs", "sm_6_5", 2),
    ];

    assert!(matches!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            variants
        ),
        Err(ArtifactError::InconsistentTargetCoverage { .. })
    ));
}

#[test]
fn rejects_overlapping_but_uneven_stage_target_coverage() {
    let variants = vec![
        variant(Target::Spirv, Stage::Vertex, "vs", "spirv_1_5", 1),
        variant(Target::Dxil, Stage::Vertex, "vs", "sm_6_5", 2),
        variant(Target::Spirv, Stage::Fragment, "fs", "spirv_1_5", 3),
    ];

    assert!(matches!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            variants
        ),
        Err(ArtifactError::InconsistentTargetCoverage { .. })
    ));
}

#[test]
fn rejects_mixed_msl_and_metallib_stage_coverage() {
    let variants = vec![
        variant(Target::Msl, Stage::Vertex, "vs", "metal_3_0", 1),
        variant(Target::Metallib, Stage::Fragment, "fs", "metallib_3_0", 2),
    ];

    assert!(matches!(
        Artifact::new(
            br"{}".to_vec(),
            Provenance::new("s", "v", vec![], "t"),
            variants
        ),
        Err(ArtifactError::InconsistentTargetCoverage { .. })
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
fn encoding_uses_format_v5_and_rejects_v4() {
    assert_eq!(ARTIFACT_FORMAT_VERSION, 5);

    let bytes = sample().encode().unwrap();
    assert_eq!(&bytes[..8], b"EZSHDR05");
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 5);

    let mut version_four = bytes;
    version_four[..8].copy_from_slice(b"EZSHDR04");
    version_four[8..12].copy_from_slice(&4_u32.to_le_bytes());
    assert!(matches!(
        Artifact::decode(&version_four),
        Err(ArtifactError::InvalidHeader)
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
