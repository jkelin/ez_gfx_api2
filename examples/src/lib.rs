mod presented;
mod scenes;

pub use presented::PresentedReport;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Example {
    Triangle,
    TexturedCube,
    ComputeStructured,
    ImGui,
    Helmet,
    SponzaKtx2,
}

/// Runs an externally presented example until close, unless the validated frame-limit variable is set.
pub fn run(example: Example) -> Result<(), String> {
    let frame_limit = match std::env::var("EZ_GFX_EXAMPLE_MAX_FRAMES") {
        Ok(value) => {
            let value = value
                .parse::<u32>()
                .map_err(|_| "EZ_GFX_EXAMPLE_MAX_FRAMES must be a positive integer".to_owned())?;
            if value == 0 {
                return Err("EZ_GFX_EXAMPLE_MAX_FRAMES must be positive".to_owned());
            }
            Some(value)
        }
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.to_string()),
    };
    let report = presented::run_example(example, frame_limit)?;
    let report_requested = std::env::var_os("EZ_GFX_EXAMPLE_REPORT").is_some();
    let snapshot_path = std::env::var_os("EZ_GFX_EXAMPLE_SNAPSHOT");
    if report_requested || snapshot_path.is_some() {
        let report =
            report.ok_or_else(|| "snapshot output requires a finite frame limit".to_owned())?;
        if let Some(path) = snapshot_path {
            let update = match std::env::var("EZ_GFX_UPDATE_SNAPSHOTS") {
                Ok(value) if value == "1" => true,
                Ok(_) => return Err("EZ_GFX_UPDATE_SNAPSHOTS must be 1 when set".to_owned()),
                Err(std::env::VarError::NotPresent) => false,
                Err(error) => return Err(error.to_string()),
            };
            compare_or_update_snapshot(&report, std::path::Path::new(&path), update)?;
        }
        if report_requested {
            println!(
                "ez-gfx-snapshot {} {} {} {} {} {} {}",
                report.width,
                report.height,
                report.frames,
                blake3::hash(&report.rgba8),
                report.runtime_events,
                report.diagnostics,
                report.dropped_observations,
            );
        }
    }
    Ok(())
}

/// Validation still creates a visible native window, surface, swapchain, draw, presentation, and readback.
pub fn run_for_validation(example: Example, frames: u32) -> Result<PresentedReport, String> {
    presented::run_example(example, Some(frames))?
        .ok_or_else(|| "validation ended without a captured frame".to_owned())
}

// Snapshot updates create parent directories; comparisons reject absent, malformed, dimension-mismatched, or changed pixels.
fn compare_or_update_snapshot(
    report: &PresentedReport,
    path: &std::path::Path,
    update: bool,
) -> Result<(), String> {
    if update {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create snapshot directory: {error}"))?;
        }
        return image::save_buffer_with_format(
            path,
            &report.rgba8,
            report.width,
            report.height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .map_err(|error| format!("update snapshot {}: {error}", path.display()));
    }

    let expected = image::open(path)
        .map_err(|error| format!("open snapshot {}: {error}", path.display()))?
        .into_rgba8();
    if expected.dimensions() != (report.width, report.height) {
        return Err(format!(
            "snapshot {} dimensions changed from {:?} to {:?}",
            path.display(),
            expected.dimensions(),
            (report.width, report.height)
        ));
    }
    let expected = expected.into_raw();
    if expected != report.rgba8 {
        let changed = expected
            .chunks_exact(4)
            .zip(report.rgba8.chunks_exact(4))
            .filter(|(left, right)| left != right)
            .count();
        return Err(format!(
            "snapshot {} changed {changed} pixels (expected {}, actual {})",
            path.display(),
            blake3::hash(&expected),
            blake3::hash(&report.rgba8)
        ));
    }
    Ok(())
}
