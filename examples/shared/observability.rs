use core::fmt::Write as _;

/// One-line legend for the compressed window title, printed once at startup.
/// Stderr keeps `ez-gfx-snapshot`/benchmark stdout machine-parseable.
pub const FRAME_TITLE_LEGEND: &str = "window title: {id} [backend] {fps}fps Tn/bytes Vn/bytes In/bytes Sn/bytes Pn Rbytes (T/V/I pending texture/vertex/index uploads, S staging buckets, P pipelines, R readback)";

/// Prints [`FRAME_TITLE_LEGEND`] once at example startup.
pub fn print_frame_title_legend() {
    // Human hint only; stderr avoids breaking snapshot-report stdout parsing.
    eprintln!("{FRAME_TITLE_LEGEND}");
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ObservationCounts {
    pub runtime_events: u32,
    pub diagnostics: u32,
    pub dropped: u64,
}

pub fn drain_bounded<E>(
    limit: u32,
    mut event: impl FnMut() -> Result<(bool, u64), E>,
    mut diagnostic: impl FnMut() -> Result<(bool, u64), E>,
) -> Result<ObservationCounts, E> {
    let mut counts = ObservationCounts::default();
    for _ in 0..limit {
        let (present, dropped) = event()?;
        counts.dropped = counts.dropped.saturating_add(dropped);
        if !present {
            break;
        }
        counts.runtime_events += 1;
    }
    for _ in 0..limit {
        let (present, dropped) = diagnostic()?;
        counts.dropped = counts.dropped.saturating_add(dropped);
        if !present {
            break;
        }
        counts.diagnostics += 1;
    }
    Ok(counts)
}

/// Compact per-frame window title shared by every example.
///
/// Format: `{identity} [{backend}] {fps:.0}fps T{tex}/{tex_bytes} V{vup}/{v_bytes} I{iup}/{i_bytes} S{buckets}/{stage_bytes} P{pipelines} R{readback_bytes}`.
/// Counts are outstanding upload allocations; byte counts use [`push_compact_bytes`].
/// A missing snapshot (stale or lost context query) renders as `diag ?` so title
/// updates never break the render loop.
pub fn push_frame_title(
    dst: &mut String,
    identity: &str,
    backend: &str,
    fps: f32,
    diagnostics: Option<&ez_gfx::ResourceDiagnostics>,
) {
    // FPS is display-only telemetry; clamp degenerate deltas instead of failing.
    let fps = fps.max(0.0);
    let _ = write!(dst, "{identity} [{backend}] {fps:.0}fps ");
    let Some(diagnostics) = diagnostics else {
        let _ = write!(dst, "diag ?");
        return;
    };
    let _ = write!(dst, "T{}/", diagnostics.pending_textures,);
    push_compact_bytes(dst, diagnostics.pending_texture_bytes);
    let _ = write!(dst, " V{}/", diagnostics.pending_vertex_uploads);
    push_compact_bytes(dst, diagnostics.pending_vertex_bytes);
    let _ = write!(dst, " I{}/", diagnostics.pending_index_uploads);
    push_compact_bytes(dst, diagnostics.pending_index_bytes);
    let _ = write!(dst, " S{}/", diagnostics.staging_buckets);
    push_compact_bytes(dst, diagnostics.staging_bytes);
    let _ = write!(dst, " P{} R", diagnostics.pipeline_entries);
    push_compact_bytes(dst, diagnostics.readback_bytes);
}

/// Appends a compact byte count (`0B`, `1.5KB`, `4.0MB`) without allocating.
pub fn push_compact_bytes(dst: &mut String, bytes: u64) {
    // Binary units match staging and transfer accounting; one decimal keeps
    // titles short while distinguishing sub-megabyte streaming deltas.
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        let _ = write!(dst, "{bytes}B");
    } else {
        let _ = write!(dst, "{value:.1}{}", UNITS[unit]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_stops_on_empty_and_saturates_drops() {
        let mut events = [(true, u64::MAX), (false, 1)].into_iter();
        let counts =
            drain_bounded(4, || Ok::<_, ()>(events.next().unwrap()), || Ok((false, 2))).unwrap();
        assert_eq!(
            counts,
            ObservationCounts {
                runtime_events: 1,
                diagnostics: 0,
                dropped: u64::MAX
            }
        );
    }

    #[test]
    fn drain_honors_each_queue_limit() {
        let counts = drain_bounded(2, || Ok::<_, ()>((true, 0)), || Ok((true, 0))).unwrap();
        assert_eq!((counts.runtime_events, counts.diagnostics), (2, 2));
    }

    #[test]
    fn compact_bytes_use_binary_units_with_one_decimal() {
        for (bytes, expected) in [
            (0, "0B"),
            (1, "1B"),
            (1023, "1023B"),
            (1024, "1.0KB"),
            (1536, "1.5KB"),
            (5 * 1024 * 1024, "5.0MB"),
            (3 * 1024 * 1024 * 1024, "3.0GB"),
        ] {
            let mut dst = String::new();
            push_compact_bytes(&mut dst, bytes);
            assert_eq!(dst, expected, "{bytes}");
        }
    }

    #[test]
    fn frame_title_counts_uploads_and_degrades_without_snapshot() {
        let mut title = String::new();
        push_frame_title(
            &mut title,
            "02_textured_cube",
            "vulkan",
            60.4,
            Some(&ez_gfx::ResourceDiagnostics {
                pending_textures: 2,
                pending_texture_bytes: 1536,
                pending_vertex_uploads: 1,
                pending_vertex_bytes: 16,
                pending_index_uploads: 0,
                pending_index_bytes: 0,
                staging_buckets: 3,
                staging_bytes: 4 * 1024 * 1024,
                pipeline_entries: 7,
                readback_bytes: 256,
                ..Default::default()
            }),
        );
        assert_eq!(
            title,
            "02_textured_cube [vulkan] 60fps T2/1.5KB V1/16B I0/0B S3/4.0MB P7 R256B"
        );

        title.clear();
        push_frame_title(&mut title, "01_triangle", "dx12", 0.0, None);
        assert_eq!(title, "01_triangle [dx12] 0fps diag ?");
    }
}
