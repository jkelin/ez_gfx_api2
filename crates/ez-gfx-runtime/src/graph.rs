use compact_str::CompactString;
use smallvec::SmallVec;
use std::{cmp::Reverse, collections::BinaryHeap, mem::size_of};

use ez_gfx_hal::{BufferRange, CompletionToken, QueueKind, ResourceAccess, ResourceState};

pub use crate::target::{Format, LoadOp, StoreOp};
mod cache;
mod compiler;
use cache::GRAPH_TEMPLATE_SCHEMA;
pub(crate) use cache::GraphTemplateCache;
pub use cache::GraphTemplateCacheStats;
pub(crate) use compiler::GraphWorkspace;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Stable handle identifying a graph resource.
pub struct ResourceId(u32);
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Stable handle identifying a graph node.
pub struct NodeId(u32);
impl ResourceId {
    /// Returns the zero-based resource index.
    pub const fn index(self) -> u32 {
        self.0
    }
    /// Creates a resource handle from a zero-based index.
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}
impl NodeId {
    /// Returns the zero-based node index.
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes how a resource persists and who owns its storage.
pub enum ResourceLifetime {
    /// Uses graph-managed storage that may be aliased after its final use.
    Transient,
    /// Preserves resource state for reuse across frames.
    PersistentHistory,
    /// Refers to storage managed outside the graph.
    External,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ResourceShape {
    Buffer {
        size: u64,
        alignment: u64,
    },
    Image {
        width: u32,
        height: u32,
        mips: u32,
        layers: u32,
        format: Format,
        samples: u8,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Describes a buffer or image and its storage lifetime.
pub struct ResourceDesc {
    /// Defines the buffer size and alignment or the image geometry and format.
    shape: ResourceShape,
    /// Determines storage ownership and persistence.
    lifetime: ResourceLifetime,
}
impl ResourceDesc {
    /// Creates a buffer description with nonzero size and power-of-two alignment.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidResource` if `size` is zero or `alignment` is not a power of two.
    pub fn buffer(
        size: u64,
        alignment: u64,
        lifetime: ResourceLifetime,
    ) -> Result<Self, GraphError> {
        if size == 0 || !alignment.is_power_of_two() {
            return Err(GraphError::InvalidResource);
        }
        Ok(Self {
            shape: ResourceShape::Buffer { size, alignment },
            lifetime,
        })
    }

    /// Dimensions/counts are nonzero, mip levels stop at 1x1, and sample count is closed.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidResource` for zero dimensions/counts, mip levels beyond 1x1, or unsupported samples.
    pub fn image(
        width: u32,
        height: u32,
        mips: u32,
        layers: u32,
        format: Format,
        samples: u8,
        lifetime: ResourceLifetime,
    ) -> Result<Self, GraphError> {
        let max_mips = u32::BITS - width.max(height).leading_zeros();
        if width == 0
            || height == 0
            || mips == 0
            || mips > max_mips
            || layers == 0
            || !matches!(samples, 1 | 2 | 4 | 8)
        {
            return Err(GraphError::InvalidResource);
        }
        Ok(Self {
            shape: ResourceShape::Image {
                width,
                height,
                mips,
                layers,
                format,
                samples,
            },
            lifetime,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Selects a contiguous mip and array-layer region of an image.
pub struct ImageRange {
    /// First mip level included in the region.
    pub first_mip: u32,
    /// Number of mip levels included in the region.
    pub mip_count: u32,
    /// First array layer included in the region.
    pub first_layer: u32,
    /// Number of array layers included in the region.
    pub layer_count: u32,
}
impl ImageRange {
    /// Creates a nonempty image region whose mip and layer bounds do not overflow.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidRange` if either count is zero or either range end overflows `u32`.
    pub fn new(
        first_mip: u32,
        mip_count: u32,
        first_layer: u32,
        layer_count: u32,
    ) -> Result<Self, GraphError> {
        if mip_count == 0
            || layer_count == 0
            || first_mip.checked_add(mip_count).is_none()
            || first_layer.checked_add(layer_count).is_none()
        {
            return Err(GraphError::InvalidRange);
        }
        Ok(Self {
            first_mip,
            mip_count,
            first_layer,
            layer_count,
        })
    }
    /// Selects every mip and array layer from zero.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidRange` if `mips` or `layers` is zero.
    pub fn all(mips: u32, layers: u32) -> Result<Self, GraphError> {
        Self::new(0, mips, 0, layers)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Selects either a byte region of a buffer or subresources of an image.
pub enum ResourceRange {
    /// Selects a byte range within a buffer.
    Buffer(BufferRange),
    /// Selects mip levels and array layers within an image.
    Image(ImageRange),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Declares a resource region and the state required by a node.
pub struct Access {
    /// Resource accessed by the node.
    pub resource: ResourceId,
    /// Buffer bytes or image subresources being accessed.
    pub range: ResourceRange,
    /// Queue, access mode, and layout required during the access.
    pub state: ResourceState,
}
impl Access {
    /// Declares access to a buffer byte range in the requested state.
    pub const fn buffer(resource: ResourceId, range: BufferRange, state: ResourceState) -> Self {
        Self {
            resource,
            range: ResourceRange::Buffer(range),
            state,
        }
    }
    /// Declares access to an image subresource region in the requested state.
    pub const fn image(resource: ResourceId, range: ImageRange, state: ResourceState) -> Self {
        Self {
            resource,
            range: ResourceRange::Image(range),
            state,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Describes render attachments, render area, sampling, and load/store behavior.
pub struct PassInfo {
    /// Color attachments in binding order.
    colors: SmallVec<[ResourceId; 2]>,
    /// Optional depth attachment.
    depth: Option<ResourceId>,
    /// Render area as `[x, y, width, height]`.
    area: [u32; 4],
    /// Multisample count shared by all attachments.
    samples: u8,
    /// Operation applied to attachment contents when the pass begins.
    load: LoadOp,
    /// Operation applied to attachment contents when the pass ends.
    store: StoreOp,
}
impl PassInfo {
    /// Creates a render-pass description with valid attachments, area, and sample count.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidPass` if there are no attachments, the area is empty, the sample count is unsupported, or an attachment is duplicated.
    pub fn new(
        colors: Vec<ResourceId>,
        depth: Option<ResourceId>,
        area: [u32; 4],
        samples: u8,
        load: LoadOp,
        store: StoreOp,
    ) -> Result<Self, GraphError> {
        if colors.is_empty() && depth.is_none()
            || area[2] == 0
            || area[3] == 0
            || !matches!(samples, 1 | 2 | 4 | 8)
            || colors
                .iter()
                .enumerate()
                .any(|(index, color)| colors[index + 1..].contains(color))
            || depth.is_some_and(|value| colors.contains(&value))
        {
            return Err(GraphError::InvalidPass);
        }
        Ok(Self {
            colors: colors.into_iter().collect(),
            depth,
            area,
            samples,
            load,
            store,
        })
    }
    /// Creates the common one-color render-pass description without temporary heap storage.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidPass` if the area is empty or the sample count is unsupported.
    pub fn single_color(
        color: ResourceId,
        area: [u32; 4],
        samples: u8,
        load: LoadOp,
        store: StoreOp,
    ) -> Result<Self, GraphError> {
        if area[2] == 0 || area[3] == 0 || !matches!(samples, 1 | 2 | 4 | 8) {
            return Err(GraphError::InvalidPass);
        }
        let mut colors = SmallVec::new();
        colors.push(color);
        Ok(Self {
            colors,
            depth: None,
            area,
            samples,
            load,
            store,
        })
    }
    /// Returns the color attachments in binding order.
    pub fn colors(&self) -> &[ResourceId] {
        &self.colors
    }

    /// Returns the optional depth attachment.
    pub const fn depth(&self) -> Option<ResourceId> {
        self.depth
    }

    /// Returns the render area as `[x, y, width, height]`.
    pub const fn area(&self) -> [u32; 4] {
        self.area
    }

    /// Returns the multisample count.
    pub const fn samples(&self) -> u8 {
        self.samples
    }

    /// Returns the attachment load operation.
    pub const fn load(&self) -> LoadOp {
        self.load
    }

    /// Returns the attachment store operation.
    pub const fn store(&self) -> StoreOp {
        self.store
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Describes a queued graph node, its accesses, dependencies, and optional render pass.
pub struct NodeDesc {
    /// Human-readable node name used for diagnostics.
    name: CompactString,
    /// Queue on which the node executes.
    queue: QueueKind,
    /// Resource accesses performed by the node.
    accesses: SmallVec<[Access; 8]>,
    /// Nodes that must complete before this node.
    dependencies: SmallVec<[NodeId; 2]>,
    /// Optional render-pass metadata for attachment work.
    pass: Option<PassInfo>,
}
impl NodeDesc {
    /// Creates an empty node description for the named queue operation.
    pub fn new(name: impl AsRef<str>, queue: QueueKind) -> Self {
        Self {
            name: CompactString::new(name),
            queue,
            accesses: SmallVec::new(),
            dependencies: SmallVec::new(),
            pass: None,
        }
    }
    /// Appends a resource access to the node.
    #[must_use]
    pub fn access(mut self, access: Access) -> Self {
        self.accesses.push(access);
        self
    }
    /// Adds a node that must complete before this node.
    #[must_use]
    pub fn depends_on(mut self, node: NodeId) -> Self {
        self.dependencies.push(node);
        self
    }
    /// Associates render-pass metadata with the node.
    #[must_use]
    pub fn pass(mut self, pass: PassInfo) -> Self {
        self.pass = Some(pass);
        self
    }
}

#[derive(Default)]
/// Collects resources, queued work, dependencies, and cross-frame states for compilation.
pub struct FrameGraph {
    /// Registered resource descriptions and readiness metadata.
    resources: Vec<ResourceRecord>,
    /// Nodes in insertion order.
    nodes: Vec<NodeDesc>,
    /// Explicit prerequisite-to-dependent edges.
    explicit_edges: Vec<(NodeId, NodeId)>,
    /// Previously recorded states indexed by dense resource ID.
    history: Vec<Option<ResourceState>>,
}

pub(crate) const FRAME_WORKSPACE_BYTE_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
struct ResourceRecord {
    desc: ResourceDesc,
    ready: Option<CompletionToken>,
    initial: Option<ResourceState>,
}

impl FrameGraph {
    /// Creates an empty frame graph.
    pub fn new() -> Self {
        Self::default()
    }
    /// Registers a resource description and returns its handle.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::CapacityExhausted` if the resource count cannot be represented by `ResourceId`.
    pub fn add_resource(&mut self, desc: ResourceDesc) -> Result<ResourceId, GraphError> {
        let id = ResourceId(
            u32::try_from(self.resources.len()).map_err(|_| GraphError::CapacityExhausted)?,
        );
        self.resources.push(ResourceRecord {
            desc,
            ready: None,
            initial: None,
        });
        self.history.push(None);
        Ok(id)
    }

    /// External and persistent resources must declare the state established before this frame.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnknownResource` for an unknown handle or `GraphError::InvalidResource` for a transient resource.
    pub fn set_resource_initial_state(
        &mut self,
        resource: ResourceId,
        state: ResourceState,
    ) -> Result<(), GraphError> {
        let record = self.resource_mut(resource)?;
        if record.desc.lifetime == ResourceLifetime::Transient {
            return Err(GraphError::InvalidResource);
        }
        record.initial = Some(state);
        Ok(())
    }
    /// Records an external completion token awaited before the resource's first access.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnknownResource` if `resource` is not registered.
    pub fn set_resource_ready(
        &mut self,
        resource: ResourceId,
        token: CompletionToken,
    ) -> Result<(), GraphError> {
        self.resource_mut(resource)?.ready = Some(token);
        Ok(())
    }
    /// Records the state carried into this frame by a persistent history resource.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnknownResource` for an unknown handle or `GraphError::NotHistory` for a non-history resource.
    pub fn set_history_state(
        &mut self,
        resource: ResourceId,
        state: ResourceState,
    ) -> Result<(), GraphError> {
        if self.resource(resource)?.desc.lifetime != ResourceLifetime::PersistentHistory {
            return Err(GraphError::NotHistory);
        }
        self.history[resource.0 as usize] = Some(state);
        Ok(())
    }

    /// Access shape/bounds, node queue, attachment IDs, and same-node feedback are validated eagerly.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid name, resource access, render pass, dependency, or handle, a queue mismatch, overlapping same-node access involving a write, or exhausted node handles.
    pub fn add_node(&mut self, mut node: NodeDesc) -> Result<NodeId, GraphError> {
        if node.name.is_empty() || node.name.len() > 255 {
            return Err(GraphError::InvalidNode);
        }
        for access in &node.accesses {
            if access.state.queue != node.queue {
                return Err(GraphError::QueueMismatch);
            }
            validate_access(self.resource(access.resource)?, access)?;
        }
        for left in 0..node.accesses.len() {
            for right in left + 1..node.accesses.len() {
                let a = node.accesses[left];
                let b = node.accesses[right];
                if a.resource == b.resource
                    && overlaps(a.range, b.range)
                    && (is_write(a.state.access) || is_write(b.state.access))
                {
                    return Err(GraphError::Feedback {
                        resource: a.resource,
                    });
                }
            }
        }
        if let Some(pass) = &node.pass {
            validate_pass(self, &node, pass)?;
        }
        node.dependencies.sort_unstable();
        node.dependencies.dedup();
        let id =
            NodeId(u32::try_from(self.nodes.len()).map_err(|_| GraphError::CapacityExhausted)?);
        for dependency in &node.dependencies {
            if dependency.0 as usize >= self.nodes.len() {
                return Err(GraphError::UnknownNode);
            }
            let edge = (*dependency, id);
            if !self.explicit_edges.contains(&edge) {
                self.explicit_edges.push(edge);
            }
        }
        self.nodes.push(node);
        Ok(id)
    }

    /// Adds `prerequisite -> dependent`; both nodes must already exist.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnknownNode` if either handle is unknown or `GraphError::Cycle` if a node depends on itself.
    pub fn add_dependency(
        &mut self,
        prerequisite: NodeId,
        dependent: NodeId,
    ) -> Result<(), GraphError> {
        self.node(prerequisite)?;
        self.node(dependent)?;
        if prerequisite == dependent {
            return Err(GraphError::Cycle {
                nodes: vec![prerequisite],
            });
        }
        let edge = (prerequisite, dependent);
        if !self.explicit_edges.contains(&edge) {
            self.explicit_edges.push(edge);
        }
        Ok(())
    }

    /// Returns the record for a known resource handle.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnknownResource` if `id` is not registered.
    fn resource(&self, id: ResourceId) -> Result<&ResourceRecord, GraphError> {
        self.resources
            .get(id.0 as usize)
            .ok_or(GraphError::UnknownResource)
    }
    /// Returns mutable metadata for a known resource handle.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnknownResource` if `id` is not registered.
    fn resource_mut(&mut self, id: ResourceId) -> Result<&mut ResourceRecord, GraphError> {
        self.resources
            .get_mut(id.0 as usize)
            .ok_or(GraphError::UnknownResource)
    }
    /// Returns the description for a known node handle.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnknownNode` if `id` is not registered.
    fn node(&self, id: NodeId) -> Result<&NodeDesc, GraphError> {
        self.nodes.get(id.0 as usize).ok_or(GraphError::UnknownNode)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// Classifies an ordering hazard between two overlapping accesses.
pub enum HazardKind {
    /// A read must follow an earlier write.
    Raw,
    /// A write must follow an earlier read.
    War,
    /// A write must follow an earlier write.
    Waw,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Records a resource hazard that orders two nodes.
pub struct HazardEdge {
    /// Earlier node in the required execution order.
    pub from: NodeId,
    /// Later node in the required execution order.
    pub to: NodeId,
    /// Resource whose overlapping accesses create the hazard.
    pub resource: ResourceId,
    /// Read/write relationship that requires ordering.
    pub kind: HazardKind,
}
impl HazardEdge {
    /// Creates a hazard edge between two nodes for a resource.
    pub const fn new(from: NodeId, to: NodeId, resource: ResourceId, kind: HazardKind) -> Self {
        Self {
            from,
            to,
            resource,
            kind,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes synchronization required before a node can execute.
pub struct QueueWait {
    /// Node that must wait.
    pub node: NodeId,
    /// Optional graph node whose completion is awaited.
    pub source: Option<NodeId>,
    /// Optional external completion token awaited by the node.
    pub external: Option<CompletionToken>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Describes a resource-range state change required before a node access.
pub struct Transition {
    /// Node before which the state change occurs.
    pub node: NodeId,
    /// Resource being transitioned.
    pub resource: ResourceId,
    /// Buffer bytes or image subresources being transitioned.
    pub range: ResourceRange,
    /// Previously tracked state, or `None` when uninitialized.
    pub before: Option<ResourceState>,
    /// State required by the upcoming access.
    pub after: ResourceState,
}
#[derive(Clone, Debug, Eq, PartialEq)]
/// Groups adjacent compatible render nodes into one render pass.
pub struct CompiledPass {
    /// Render-pass metadata shared by the grouped nodes.
    pub info: PassInfo,
    /// Scheduled nodes executed within the pass.
    pub nodes: Vec<NodeId>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Identifies the transient storage slot assigned to a resource.
pub struct AliasAssignment {
    /// Reusable storage-slot index.
    pub slot: u32,
}

/// Contains the schedule and synchronization metadata produced from a frame graph.
#[derive(Default)]
pub struct CompiledGraph {
    /// Nodes in dependency-respecting execution order.
    order: Vec<NodeId>,
    /// Derived ordering hazards between resource accesses.
    hazards: Vec<HazardEdge>,
    /// External and cross-queue waits required by scheduled nodes.
    waits: Vec<QueueWait>,
    /// Resource state changes required before accesses.
    transitions: Vec<Transition>,
    /// Coalesced render passes in execution order.
    passes: Vec<CompiledPass>,
    /// Transient storage assignments indexed by dense resource ID.
    aliases: Vec<Option<AliasAssignment>>,
    /// Final uniform states indexed by dense history-resource ID.
    history_states: Vec<Option<ResourceState>>,
}
impl CompiledGraph {
    /// Returns nodes in dependency-respecting execution order.
    pub fn order(&self) -> &[NodeId] {
        &self.order
    }
    /// Returns the derived resource hazards.
    pub fn hazards(&self) -> &[HazardEdge] {
        &self.hazards
    }
    /// Returns the external and cross-queue waits.
    pub fn waits(&self) -> &[QueueWait] {
        &self.waits
    }
    /// Returns the required resource state transitions.
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }
    /// Returns the coalesced render passes.
    pub fn passes(&self) -> &[CompiledPass] {
        &self.passes
    }
    /// Returns the transient storage assignment for a resource, if any.
    pub fn alias(&self, resource: ResourceId) -> Option<AliasAssignment> {
        self.aliases.get(resource.0 as usize).copied().flatten()
    }
    /// Returns the final uniform state retained for a history resource, if any.
    pub fn history_state(&self, resource: ResourceId) -> Option<ResourceState> {
        self.history_states
            .get(resource.0 as usize)
            .copied()
            .flatten()
    }
}

impl CompiledGraph {
    pub(crate) fn retained_bytes(&self) -> usize {
        vec_bytes(&self.order)
            .saturating_add(vec_bytes(&self.hazards))
            .saturating_add(vec_bytes(&self.waits))
            .saturating_add(vec_bytes(&self.transitions))
            .saturating_add(vec_bytes(&self.passes))
            .saturating_add(
                self.passes
                    .iter()
                    .map(|pass| vec_bytes(&pass.nodes))
                    .sum::<usize>(),
            )
            .saturating_add(vec_bytes(&self.aliases))
            .saturating_add(vec_bytes(&self.history_states))
    }
}

fn vec_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity().saturating_mul(size_of::<T>())
}

fn nested_vec_bytes<T>(values: &Vec<Vec<T>>) -> usize {
    vec_bytes(values).saturating_add(values.iter().map(|value| vec_bytes(value)).sum::<usize>())
}

mod validation;
pub use validation::GraphError;
use validation::{
    AliasClass, alias_class, full_range, hazard_kind, intersection, is_write, overlaps,
    pass_compatible, subtract, validate_access, validate_pass,
};

#[cfg(test)]
mod tests {
    use super::{LoadOp, PassInfo, ResourceId, StoreOp};

    #[test]
    fn single_color_pass_validates_boundaries_without_spilling() {
        let color = ResourceId(0);
        let pass =
            PassInfo::single_color(color, [0, 0, 1, 1], 1, LoadOp::Clear, StoreOp::Store).unwrap();

        assert_eq!(pass.colors(), &[color]);
        assert!(
            PassInfo::single_color(color, [0, 0, 0, 1], 1, LoadOp::Clear, StoreOp::Store).is_err()
        );
        assert!(
            PassInfo::single_color(color, [0, 0, 1, 1], 3, LoadOp::Clear, StoreOp::Store).is_err()
        );
    }
}
