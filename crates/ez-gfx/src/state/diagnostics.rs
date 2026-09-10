//! Point-in-time resource and cache diagnostics.
//!
//! Unlike the monotonic [`TextureUploadTelemetrySnapshot`](ez_gfx_runtime::texture::TextureUploadTelemetrySnapshot)
//! counters, this snapshot describes what is outstanding right now: decode- and
//! transfer-pending uploads with their retained byte sizes, plus retained
//! staging, pipeline, and readback cache sizes.

use super::{ContextHandle, map_lifecycle, with_context_mut};
use crate::{Error, ResourceDiagnostics, Result};

/// Returns pending-upload counts with retained bytes plus retained cache sizes.
///
/// Decode-pending textures report admitted caller source bytes; transfer-pending
/// textures report decoded staging bytes recorded at native submission. Vertex
/// and index counts are outstanding upload allocations carrying the byte size
/// reserved by their original upload. Staging sizes aggregate the shared pool
/// with the buffer and counter pools; readback sizes aggregate retained frames.
///
/// # Errors
///
/// Returns an error for an invalid or stale context handle, or when called from
/// a thread other than the context creator. Device loss does not fail this
/// observation: teardown titles keep reporting until the context is destroyed.
pub fn resource_diagnostics(context: ContextHandle) -> Result<ResourceDiagnostics> {
    with_context_mut(context, |context| {
        context.identity.check_thread().map_err(map_lifecycle)?;
        let mut diagnostics = ResourceDiagnostics::default();
        for pending in context.pending_textures.values() {
            // Decode-pending entries always carry their admitted size; the owned
            // bytes live on the decode closure, so this count is the only record.
            diagnostics.pending_textures = diagnostics.pending_textures.saturating_add(1);
            diagnostics.pending_texture_bytes = diagnostics
                .pending_texture_bytes
                .saturating_add(pending.source_bytes);
        }
        for (handle, bytes) in &context.texture_transfer_bytes {
            // Sizes stay in lockstep with `texture_ready`; publication, cancel,
            // loss, and teardown remove both, so every entry here is outstanding.
            if context.texture_ready.contains_key(handle) {
                diagnostics.pending_textures = diagnostics.pending_textures.saturating_add(1);
                diagnostics.pending_texture_bytes =
                    diagnostics.pending_texture_bytes.saturating_add(*bytes);
            }
        }
        for handle in context.geometry_uploads.keys() {
            // A retired-but-unswept transfer keeps its pending key after its live
            // range is reclaimed; it no longer has a reportable size, so skip it
            // rather than fail the whole observation for a normal teardown race.
            let Ok((kind, bytes)) = context.geometry.upload_summary(*handle) else {
                continue;
            };
            // Each pending key is one outstanding upload allocation; element
            // totals stay behind the typed range queries, not this summary.
            match kind {
                ez_gfx_runtime::geometry::GeometryHeapKind::Vertex => {
                    diagnostics.pending_vertex_uploads =
                        diagnostics.pending_vertex_uploads.saturating_add(1);
                    diagnostics.pending_vertex_bytes =
                        diagnostics.pending_vertex_bytes.saturating_add(bytes);
                }
                ez_gfx_runtime::geometry::GeometryHeapKind::Index => {
                    diagnostics.pending_index_uploads =
                        diagnostics.pending_index_uploads.saturating_add(1);
                    diagnostics.pending_index_bytes =
                        diagnostics.pending_index_bytes.saturating_add(bytes);
                }
            }
        }
        let mut staging_buckets = context.staging.len();
        let mut staging_bytes = context.staging.retained_bytes();
        for pool in context.buffer_pool.values() {
            staging_buckets = staging_buckets.saturating_add(pool.len());
            staging_bytes = staging_bytes.saturating_add(pool.retained_bytes());
        }
        staging_buckets = staging_buckets.saturating_add(context.counter_pool.len());
        staging_bytes = staging_bytes.saturating_add(context.counter_pool.retained_bytes());
        diagnostics.staging_buckets = u32::try_from(staging_buckets).unwrap_or(u32::MAX);
        diagnostics.staging_bytes = staging_bytes;
        // Pipeline and readback map lengths always fit `u32`; the fallback only
        // guards the conversion.
        diagnostics.pipeline_entries =
            u32::try_from(context.pipelines.len()).map_err(|_| Error::NativeFailure)?;
        diagnostics.readback_bytes = context
            .last_readbacks
            .iter()
            .fold(0_u64, |total, readback| {
                total.saturating_add(readback.len() as u64)
            });
        Ok(diagnostics)
    })
}
