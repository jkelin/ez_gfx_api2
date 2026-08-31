//! Binary artifact encoding and validation contract tests.
use ez_gfx_artifact::{
    Artifact, ArtifactError, MAX_ARTIFACT_BYTES, Provenance, Stage, Target, TargetVariant,
};

fn variant(target: Target, stage: Stage, entry: &str, profile: &str, bytes: u8) -> TargetVariant {
    TargetVariant::new(target, stage, entry, profile, vec![bytes]).unwrap()
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
        TargetVariant::new(Target::Spirv, Stage::Vertex, "", "p", vec![1]),
        Err(ArtifactError::InvalidEntryPoint)
    ));
    assert!(matches!(
        Artifact::decode(&vec![0; MAX_ARTIFACT_BYTES + 1]),
        Err(ArtifactError::TooLarge)
    ));
}

#[test]
fn rejects_truncation_and_overlap() {
    let bytes = sample().encode().unwrap();
    assert!(matches!(
        Artifact::decode(&bytes[..bytes.len() - 1]),
        Err(ArtifactError::Truncated)
    ));
    let mut overlap = bytes;
    overlap[52..60].copy_from_slice(&0u64.to_le_bytes());
    assert!(matches!(
        Artifact::decode(&overlap),
        Err(ArtifactError::InvalidSection)
    ));
}

#[test]
fn digest_covers_metadata_and_provenance() {
    let original = sample().encode().unwrap();
    for field in [0usize, 1usize] {
        let mut bytes = original.clone();
        let section = 52 + field * 16;
        let offset = usize::try_from(u64::from_le_bytes(
            bytes[section..section + 8].try_into().unwrap(),
        ))
        .unwrap();
        bytes[offset + if field == 0 { 1 } else { 5 }] ^= 1;
        assert!(matches!(
            Artifact::decode(&bytes),
            Err(ArtifactError::DigestMismatch)
        ));
    }
    let mut changed = sample();
    changed.provenance.options.push("-g".into());
    assert_ne!(changed.encode().unwrap()[20..52], original[20..52]);
}
