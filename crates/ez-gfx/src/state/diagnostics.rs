//! Point-in-time resource and cache diagnostics.
//!
//! Unlike the monotonic [`TextureUploadTelemetrySnapshot`](ez_gfx_runtime::texture::TextureUploadTelemetrySnapshot)
//! counters, this snapshot describes what is outstanding right now: decode- and
//! transfer-pending uploads with their retained byte sizes, plus retained
//! staging, pipeline, and readback cache sizes.

use super::{
    ContextHandle, NativeContext, aggregate_staging_retained, map_lifecycle, with_context_mut,
};
use crate::{Error, MemoryTelemetryReport, ResourceDiagnostics, Result};

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
            diagnostics.pending_textures = diagnostics.pending_textures.saturating_add(1);
            diagnostics.pending_texture_bytes = diagnostics
                .pending_texture_bytes
                .saturating_add(pending.decoded_bytes.unwrap_or(pending.source_bytes));
        }
        for (handle, bytes) in &context.texture_transfer_bytes {
            // Manager uploads hold a transfer reservation; region rewrites only
            // re-arm texture_ready, so both gates report outstanding transfers.
            if context.texture_transfer_work.contains_key(handle)
                || context.texture_ready.contains_key(handle)
            {
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

/// Returns on-demand allocator and memory telemetry without dispatching events.
///
/// Backend allocators report through `generate_report` exactly once per call;
/// callers must query explicitly and never per frame. Staging sizes aggregate
/// the shared pool with the buffer and counter pools; worker and slot counts
/// come from live context state. Unlike [`resource_diagnostics`], this never
/// dispatches callbacks, so a dispatch failure cannot mask the snapshot.
///
/// # Errors
///
/// Returns an error for an invalid or stale context handle, or when called from
/// a thread other than the context creator.
pub fn memory_telemetry(context: ContextHandle) -> Result<MemoryTelemetryReport> {
    with_context_mut(context, |context| {
        context.identity.check_thread().map_err(map_lifecycle)?;
        let backend = match &context.native {
            NativeContext::Vulkan(native) => native.memory_telemetry(),
            #[cfg(windows)]
            NativeContext::Dx12(native) => native.memory_telemetry(),
            #[cfg(target_vendor = "apple")]
            NativeContext::Metal(native) => native.memory_telemetry(),
        };
        // Pool telemetry only reads lengths, so aggregation never allocates;
        // counts accumulate with saturation and bucket totals cap at `u32::MAX`.
        let mut buckets = context.staging.len();
        for pool in context.buffer_pool.values() {
            buckets = buckets.saturating_add(pool.len());
        }
        buckets = buckets.saturating_add(context.counter_pool.len());
        let bytes = aggregate_staging_retained(context);
        let high_water = context.staging_high_water_bytes;
        let surface = aggregate_surface_telemetry(context, &backend);
        let mut report = backend;
        report.swapchain_images = surface.images;
        report.swapchain_extent = surface.extent;
        report.swapchain_format = surface.format;
        report.swapchain_bytes =
            ez_gfx_hal::rgba8_image_bytes(surface.images, surface.extent.0, surface.extent.1);
        report.depth_bytes = surface.depth_extent.map_or(0, |(width, height)| {
            ez_gfx_hal::rgba8_image_bytes(1, width, height)
        });
        Ok(MemoryTelemetryReport {
            backend: report,
            staging_buckets: u32::try_from(buckets).unwrap_or(u32::MAX),
            staging_bytes: bytes,
            staging_high_water: high_water,
            // Retained serialization capacity, not live occupancy; bounded by
            // the write-path trim.
            counter_scratch_bytes: u64::try_from(context.counter_scratch.capacity())
                .unwrap_or(u64::MAX),
            // Pool sizes always fit `u32`; the fallback only guards the conversion.
            decode_workers: u32::try_from(context.async_textures.worker_count())
                .unwrap_or(u32::MAX),
        })
    })
}

/// Surface image counts, extents, formats, and depth aggregated in one convention.
struct SurfaceTelemetry {
    images: u32,
    extent: (u32, u32),
    format: u32,
    depth_extent: Option<(u32, u32)>,
}

/// Aggregates surface memory telemetry from active safe surface state.
///
/// Extents prefer the safe `SurfaceState` every backend maintains, so headless
/// and window surfaces share one source; the backend contributes only what safe
/// state cannot observe (image counts, format codes, depth presence). Vulkan
/// swapchain fields are device-global, so Vulkan-backed surfaces reuse the
/// backend report once instead of once per surface.
fn aggregate_surface_telemetry(
    context: &super::ContextState,
    backend: &ez_gfx_hal::BackendMemoryTelemetry,
) -> SurfaceTelemetry {
    let mut images = 0_u32;
    let mut extent = (0_u32, 0_u32);
    let mut format = 0_u32;
    let mut depth_extent = None;
    let mut vulkan_counted = false;
    // The active surface owns presentation; without one, the largest surface
    // still describes retention better than zeros.
    let mut preferred = context.active_surface;
    if preferred.is_none() {
        let mut largest = 0_u64;
        for (handle, record) in &context.surfaces {
            let area = record.state.extent().map_or(0, |(width, height)| {
                u64::from(width).saturating_mul(u64::from(height))
            });
            if area > largest {
                largest = area;
                preferred = Some(*handle);
            }
        }
    }
    for (handle, record) in &context.surfaces {
        let extent_here = record.state.extent();
        match (&context.native, &record.native) {
            (NativeContext::Vulkan(_), super::NativeSurface::Vulkan(_)) => {
                // Device-global swapchain: count once however many surfaces exist.
                if !vulkan_counted {
                    vulkan_counted = true;
                    images = images.saturating_add(backend.swapchain_images);
                }
                if Some(*handle) == preferred {
                    extent = extent_here.unwrap_or(backend.swapchain_extent);
                    format = backend.swapchain_format;
                    // Depth matches the swapchain; the backend already resolved it.
                    if backend.depth_bytes > 0 {
                        depth_extent = Some(extent);
                    }
                }
            }
            #[cfg(windows)]
            (NativeContext::Dx12(_), super::NativeSurface::Dx12(surface)) => {
                images = images.saturating_add(surface.telemetry_images());
                if Some(*handle) == preferred {
                    extent = extent_here.unwrap_or_else(|| surface.telemetry_extent());
                    format = surface.telemetry_format();
                    if surface.telemetry_has_depth() {
                        depth_extent = Some(extent);
                    }
                }
            }
            #[cfg(target_vendor = "apple")]
            (NativeContext::Metal(_), super::NativeSurface::Metal(surface)) => {
                images = images.saturating_add(surface.telemetry_images());
                if Some(*handle) == preferred {
                    // Metal has no backend extent query without mutating the
                    // layer, so safe state is the only extent source here.
                    extent = extent_here.unwrap_or((0, 0));
                    format = surface.telemetry_format();
                    depth_extent = surface.telemetry_depth_extent();
                }
            }
            #[cfg(any(windows, target_vendor = "apple"))]
            _ => {}
        }
    }
    // Without images nothing is retained; a stale format or extent would mislead.
    if images == 0 {
        extent = (0, 0);
        format = 0;
    }
    SurfaceTelemetry {
        images,
        extent,
        format,
        depth_extent,
    }
}
