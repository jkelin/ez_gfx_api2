use super::{
    Access, BufferRange, Format, FrameGraph, HazardKind, ImageRange, LoadOp, NodeDesc, NodeId,
    PassInfo, QueueKind, ResourceAccess, ResourceDesc, ResourceId, ResourceRange, ResourceRecord,
    ResourceShape, StoreOp,
};
use arrayvec::ArrayVec;

pub(crate) fn validate_pass(
    graph: &FrameGraph,
    node: &NodeDesc,
    pass: &PassInfo,
) -> Result<(), GraphError> {
    if node.queue != QueueKind::Graphics {
        return Err(GraphError::InvalidPass);
    }
    let area_end = [
        pass.area[0]
            .checked_add(pass.area[2])
            .ok_or(GraphError::InvalidPass)?,
        pass.area[1]
            .checked_add(pass.area[3])
            .ok_or(GraphError::InvalidPass)?,
    ];
    for (resource, depth) in pass
        .colors
        .iter()
        .copied()
        .map(|resource| (resource, false))
        .chain(pass.depth.map(|resource| (resource, true)))
    {
        let record = graph.resource(resource)?;
        let ResourceShape::Image {
            width,
            height,
            mips: _,
            layers: _,
            format,
            samples,
        } = record.desc.shape
        else {
            return Err(GraphError::InvalidPass);
        };
        if samples != pass.samples
            || area_end[0] > width
            || area_end[1] > height
            || depth != (format == Format::Depth32Float)
        {
            return Err(GraphError::InvalidPass);
        }
        let valid_access = node.accesses.iter().any(|access| {
            access.resource == resource
                && matches!(
                    access.range,
                    ResourceRange::Image(ImageRange {
                        first_mip: 0,
                        mip_count,
                        first_layer: 0,
                        layer_count,
                    }) if mip_count > 0 && layer_count > 0
                )
                && if depth {
                    matches!(
                        access.state.access,
                        ResourceAccess::DepthStencilRead | ResourceAccess::DepthStencilWrite
                    )
                } else {
                    access.state.access == ResourceAccess::ColorAttachmentWrite
                }
        });
        if !valid_access {
            return Err(GraphError::InvalidPass);
        }
    }
    Ok(())
}

/// Checks that an access range matches the resource shape and remains within its bounds.
///
/// # Errors
///
/// Returns `GraphError::RangeTypeMismatch` when the range kind does not match the resource, or `GraphError::InvalidRange` when the range overflows or exceeds the resource bounds.
pub(crate) fn validate_access(
    resource: &ResourceRecord,
    access: &Access,
) -> Result<(), GraphError> {
    match (&resource.desc.shape, access.range) {
        (ResourceShape::Buffer { size, .. }, ResourceRange::Buffer(range))
            if range
                .offset
                .checked_add(range.size)
                .is_some_and(|end| end <= *size) =>
        {
            Ok(())
        }
        (ResourceShape::Image { mips, layers, .. }, ResourceRange::Image(range))
            if range
                .first_mip
                .checked_add(range.mip_count)
                .is_some_and(|end| end <= *mips)
                && range
                    .first_layer
                    .checked_add(range.layer_count)
                    .is_some_and(|end| end <= *layers) =>
        {
            Ok(())
        }
        (ResourceShape::Buffer { .. }, ResourceRange::Image(_))
        | (ResourceShape::Image { .. }, ResourceRange::Buffer(_)) => {
            Err(GraphError::RangeTypeMismatch)
        }
        _ => Err(GraphError::InvalidRange),
    }
}

/// Reports whether two adjacent graphics passes can share one render pass.
pub(crate) fn pass_compatible(previous: &PassInfo, next: &PassInfo) -> bool {
    previous.colors == next.colors
        && previous.depth == next.depth
        && previous.area == next.area
        && previous.samples == next.samples
        && previous.store == StoreOp::Store
        && next.load == LoadOp::Load
}

/// Returns the overlapping buffer span or image subresource rectangle, if any.
pub(crate) fn intersection(a: ResourceRange, b: ResourceRange) -> Option<ResourceRange> {
    match (a, b) {
        (ResourceRange::Buffer(a), ResourceRange::Buffer(b)) => {
            let start = a.offset.max(b.offset);
            let end = (a.offset + a.size).min(b.offset + b.size);
            if start < end {
                Some(ResourceRange::Buffer(BufferRange {
                    offset: start,
                    size: end - start,
                }))
            } else {
                None
            }
        }
        (ResourceRange::Image(a), ResourceRange::Image(b)) => {
            let first_mip = a.first_mip.max(b.first_mip);
            let mip_end = (a.first_mip + a.mip_count).min(b.first_mip + b.mip_count);
            let first_layer = a.first_layer.max(b.first_layer);
            let layer_end = (a.first_layer + a.layer_count).min(b.first_layer + b.layer_count);
            if first_mip < mip_end && first_layer < layer_end {
                Some(ResourceRange::Image(ImageRange {
                    first_mip,
                    mip_count: mip_end - first_mip,
                    first_layer,
                    layer_count: layer_end - first_layer,
                }))
            } else {
                None
            }
        }
        _ => None,
    }
}
/// Splits a resource range into the at most four regions left after removing an overlapping cut.
pub(crate) fn subtract(range: ResourceRange, cut: ResourceRange) -> ArrayVec<ResourceRange, 4> {
    let Some(overlap) = intersection(range, cut) else {
        return ArrayVec::from_iter([range]);
    };
    match (range, overlap) {
        (ResourceRange::Buffer(range), ResourceRange::Buffer(overlap)) => {
            let mut result = ArrayVec::new();
            if range.offset < overlap.offset {
                result.push(ResourceRange::Buffer(BufferRange {
                    offset: range.offset,
                    size: overlap.offset - range.offset,
                }));
            }
            let range_end = range.offset + range.size;
            let overlap_end = overlap.offset + overlap.size;
            if overlap_end < range_end {
                result.push(ResourceRange::Buffer(BufferRange {
                    offset: overlap_end,
                    size: range_end - overlap_end,
                }));
            }
            result
        }
        (ResourceRange::Image(range), ResourceRange::Image(overlap)) => {
            let mut result = ArrayVec::new();
            let range_mip_end = range.first_mip + range.mip_count;
            let overlap_mip_end = overlap.first_mip + overlap.mip_count;
            let range_layer_end = range.first_layer + range.layer_count;
            let overlap_layer_end = overlap.first_layer + overlap.layer_count;
            if range.first_mip < overlap.first_mip {
                result.push(ResourceRange::Image(ImageRange {
                    first_mip: range.first_mip,
                    mip_count: overlap.first_mip - range.first_mip,
                    first_layer: range.first_layer,
                    layer_count: range.layer_count,
                }));
            }
            if overlap_mip_end < range_mip_end {
                result.push(ResourceRange::Image(ImageRange {
                    first_mip: overlap_mip_end,
                    mip_count: range_mip_end - overlap_mip_end,
                    first_layer: range.first_layer,
                    layer_count: range.layer_count,
                }));
            }
            if range.first_layer < overlap.first_layer {
                result.push(ResourceRange::Image(ImageRange {
                    first_mip: overlap.first_mip,
                    mip_count: overlap.mip_count,
                    first_layer: range.first_layer,
                    layer_count: overlap.first_layer - range.first_layer,
                }));
            }
            if overlap_layer_end < range_layer_end {
                result.push(ResourceRange::Image(ImageRange {
                    first_mip: overlap.first_mip,
                    mip_count: overlap.mip_count,
                    first_layer: overlap_layer_end,
                    layer_count: range_layer_end - overlap_layer_end,
                }));
            }
            result
        }
        _ => unreachable!("intersection only returns ranges of the same kind"),
    }
}
/// Reports whether two buffer spans or image subresource rectangles intersect.
pub(crate) fn overlaps(a: ResourceRange, b: ResourceRange) -> bool {
    match (a, b) {
        (ResourceRange::Buffer(a), ResourceRange::Buffer(b)) => {
            a.offset < b.offset + b.size && b.offset < a.offset + a.size
        }
        (ResourceRange::Image(a), ResourceRange::Image(b)) => {
            a.first_mip < b.first_mip + b.mip_count
                && b.first_mip < a.first_mip + a.mip_count
                && a.first_layer < b.first_layer + b.layer_count
                && b.first_layer < a.first_layer + a.layer_count
        }
        _ => false,
    }
}
/// Reports whether an access mode can modify resource contents.
pub(crate) fn is_write(access: ResourceAccess) -> bool {
    matches!(
        access,
        ResourceAccess::StorageWrite
            | ResourceAccess::StorageReadWrite
            | ResourceAccess::IndirectStorageReadWrite
            | ResourceAccess::ColorAttachmentWrite
            | ResourceAccess::DepthStencilWrite
            | ResourceAccess::TransferWrite
    )
}
/// Classifies a dependency from the read/write modes of two ordered accesses.
pub(crate) fn hazard_kind(before: ResourceAccess, after: ResourceAccess) -> Option<HazardKind> {
    match (is_write(before), is_write(after)) {
        (true, false) => Some(HazardKind::Raw),
        (false, true) => Some(HazardKind::War),
        (true, true) => Some(HazardKind::Waw),
        _ => None,
    }
}
/// Returns a range covering the entire buffer or every image mip and layer.
pub(crate) fn full_range(desc: &ResourceDesc) -> ResourceRange {
    match desc.shape {
        ResourceShape::Buffer { size, .. } => {
            ResourceRange::Buffer(BufferRange { offset: 0, size })
        }
        ResourceShape::Image { mips, layers, .. } => ResourceRange::Image(ImageRange {
            first_mip: 0,
            mip_count: mips,
            first_layer: 0,
            layer_count: layers,
        }),
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AliasClass {
    Buffer { alignment: u64 },
    Image { format: Format, samples: u8 },
}
/// Groups buffers by alignment and images by format and sample count for aliasing.
pub(crate) fn alias_class(desc: &ResourceDesc) -> AliasClass {
    match desc.shape {
        ResourceShape::Buffer { alignment, .. } => AliasClass::Buffer { alignment },
        ResourceShape::Image {
            format, samples, ..
        } => AliasClass::Image { format, samples },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Describes why frame-graph construction or validation failed.
pub enum GraphError {
    /// A resource description is invalid.
    InvalidResource,
    /// A resource range lies outside its bounds or is otherwise invalid.
    InvalidRange,
    /// A buffer was addressed with an image range, or an image with a buffer range.
    RangeTypeMismatch,
    /// A node description is invalid.
    InvalidNode,
    /// A graphics pass has incompatible attachments, area, samples, or access modes.
    InvalidPass,
    /// A resource access uses a queue that does not support its requested state.
    QueueMismatch,
    /// A referenced resource identifier is not present in the graph.
    UnknownResource,
    /// A referenced node identifier is not present in the graph.
    UnknownNode,
    /// History access was requested for a resource not configured for history.
    NotHistory,
    /// The graph exceeded a fixed resource, node, or dependency capacity.
    CapacityExhausted,
    /// Overlapping accesses create a feedback dependency on one resource.
    /// A resource has a feedback dependency.
    Feedback {
        /// Feedback resource.
        resource: ResourceId,
    },
    /// The graph contains a cycle.
    Cycle {
        /// Nodes participating in the cycle.
        nodes: Vec<NodeId>,
    },
}

#[cfg(test)]
mod tests {
    use super::subtract;
    use crate::graph::{ImageRange, ResourceRange};
    use ez_gfx_hal::BufferRange;

    #[test]
    fn subtraction_is_inline_and_bounded_by_four_fragments() {
        let image = ResourceRange::Image(ImageRange {
            first_mip: 0,
            mip_count: 4,
            first_layer: 0,
            layer_count: 4,
        });
        let center = ResourceRange::Image(ImageRange {
            first_mip: 1,
            mip_count: 2,
            first_layer: 1,
            layer_count: 2,
        });
        let fragments = subtract(image, center);

        assert_eq!(fragments.len(), 4);
        assert_eq!(fragments.capacity(), 4);
        assert_eq!(
            subtract(
                ResourceRange::Buffer(BufferRange { offset: 0, size: 4 }),
                ResourceRange::Buffer(BufferRange { offset: 8, size: 4 }),
            )
            .as_slice(),
            &[ResourceRange::Buffer(BufferRange { offset: 0, size: 4 })]
        );
    }
}
