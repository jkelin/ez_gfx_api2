//! Apple toolchain provenance: host macOS version, Metal deployment minimums,
//! and Apple tool output capture.

use std::process::Command;

use ez_gfx_artifact::{CompatibilityVersion, Stage};

use super::{CompilerError, parse_compatibility_version, parse_deployment_version};

pub(super) fn metal_minimum_os(
    deployment_target: Option<&str>,
    host_os: CompatibilityVersion,
    stage: Stage,
) -> Result<CompatibilityVersion, CompilerError> {
    // Explicit deployment targets remain authoritative. Task and mesh products
    // fail closed when either an explicit target or the concrete host provenance is too old.
    let minimum_os = deployment_target
        .map(parse_deployment_version)
        .transpose()?
        .unwrap_or(host_os);
    if matches!(stage, Stage::Task | Stage::Mesh) && minimum_os < CompatibilityVersion::new(13, 0) {
        return Err(CompilerError::InvalidRequest(
            "task/mesh Metal deployment target",
        ));
    }
    Ok(minimum_os)
}

pub(super) fn host_macos_version() -> Result<CompatibilityVersion, CompilerError> {
    let output = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .map_err(CompilerError::Io)?;
    if !output.status.success() {
        return Err(CompilerError::ToolFailed(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    parse_compatibility_version(
        std::str::from_utf8(&output.stdout)
            .map_err(|_| CompilerError::InvalidRequest("macOS version"))?,
    )
}

pub(super) fn apple_tool_output(args: &[&str]) -> Result<String, CompilerError> {
    let output = Command::new("xcrun")
        .args(args)
        .output()
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => CompilerError::AppleToolNotFound("xcrun".into()),
            _ => CompilerError::Io(error),
        })?;
    if !output.status.success() {
        return Err(CompilerError::ToolFailed(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| CompilerError::InvalidRequest("Apple tool output"))
}
