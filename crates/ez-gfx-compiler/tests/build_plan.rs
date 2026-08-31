//! Build-time shader manifest planning tests.

use ez_gfx_artifact::{Stage, Target};
use ez_gfx_compiler::{MetalOutput, plan_manifest};
use std::path::Path;

const MANIFEST: &[u8] = br#"{
  "source":"examples/demo/demo.slang",
  "output":"examples/demo/demo.ezgfxshader",
  "required_version":"2026",
  "include_dirs":["."],
  "defines":["FEATURE=1"],
  "semantic_metadata":{"abi":1},
  "toolchain":"slang",
  "development":true,
  "targets":[
    {"target":"spirv","stage":"compute","entry":"computemain","profile":"spirv_1_5"},
    {"target":"dxil","stage":"compute","entry":"computemain","profile":"sm_6_5"},
    {"target":"metallib","stage":"compute","entry":"computemain","profile":"metal_3_0"}
  ]
}"#;

#[test]
fn manifest_rewrite_resolves_inputs_and_uses_requested_output_directory() {
    let plan = plan_manifest(
        MANIFEST,
        Path::new("/workspace"),
        Path::new("/cargo/out"),
        MetalOutput::Source,
    )
    .unwrap();

    assert_eq!(plan.config.required_version, "2026");
    assert_eq!(
        plan.request.source,
        Path::new("/workspace/examples/demo/demo.slang")
    );
    assert_eq!(plan.request.output_dir, Path::new("/cargo/out"));
    assert_eq!(plan.request.include_dirs, [Path::new("/workspace")]);
    assert_eq!(plan.request.defines, ["FEATURE=1"]);
    assert!(!plan.request.release_complete);
}

#[test]
fn non_apple_builds_emit_msl_without_invoking_xcrun() {
    let plan = plan_manifest(
        MANIFEST,
        Path::new("/workspace"),
        Path::new("/cargo/out"),
        MetalOutput::Source,
    )
    .unwrap();

    assert!(
        plan.request
            .targets
            .iter()
            .any(|target| { target.target == Target::Msl && target.stage == Stage::Compute })
    );
    assert!(
        !plan
            .request
            .targets
            .iter()
            .any(|target| target.target == Target::Metallib)
    );
}

#[test]
fn apple_builds_require_metallib_products() {
    let plan = plan_manifest(
        MANIFEST,
        Path::new("/workspace"),
        Path::new("/cargo/out"),
        MetalOutput::Library,
    )
    .unwrap();

    assert!(plan.request.release_complete);
    assert!(
        plan.request
            .targets
            .iter()
            .any(|target| { target.target == Target::Metallib && target.stage == Stage::Compute })
    );
    assert!(
        !plan
            .request
            .targets
            .iter()
            .any(|target| target.target == Target::Msl)
    );
}

#[test]
fn malformed_manifest_is_rejected_before_native_compiler_use() {
    assert!(
        plan_manifest(
            br#"{"source":7}"#,
            Path::new("/workspace"),
            Path::new("/cargo/out"),
            MetalOutput::Source,
        )
        .is_err()
    );
}
