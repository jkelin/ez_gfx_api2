use std::collections::{BTreeMap, BTreeSet};

use ez_gfx_hal::{BufferRange, CompletionToken, QueueKind, ResourceAccess, ResourceState};

pub use crate::target::{Format, LoadOp, StoreOp};

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

    /// Dimensions/counts are nonzero, sample count is closed, and byte-size arithmetic must fit u64.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidResource` if any dimension or count is zero or `samples` is not 1, 2, 4, or 8.
    pub fn image(
        width: u32,
        height: u32,
        mips: u32,
        layers: u32,
        format: Format,
        samples: u8,
        lifetime: ResourceLifetime,
    ) -> Result<Self, GraphError> {
        if width == 0
            || height == 0
            || mips == 0
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
    colors: Vec<ResourceId>,
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
        {
            return Err(GraphError::InvalidPass);
        }
        let mut unique = colors.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != colors.len() || depth.is_some_and(|value| colors.contains(&value)) {
            return Err(GraphError::InvalidPass);
        }
        Ok(Self {
            colors,
            depth,
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
    name: String,
    /// Queue on which the node executes.
    queue: QueueKind,
    /// Resource accesses performed by the node.
    accesses: Vec<Access>,
    /// Nodes that must complete before this node.
    dependencies: Vec<NodeId>,
    /// Optional render-pass metadata for attachment work.
    pass: Option<PassInfo>,
}
impl NodeDesc {
    /// Creates an empty node description for the named queue operation.
    pub fn new(name: impl Into<String>, queue: QueueKind) -> Self {
        Self {
            name: name.into(),
            queue,
            accesses: Vec::new(),
            dependencies: Vec::new(),
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
    explicit_edges: BTreeSet<(NodeId, NodeId)>,
    /// Previously recorded states for persistent history resources.
    history: BTreeMap<ResourceId, ResourceState>,
}
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
        self.history.insert(resource, state);
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
            self.explicit_edges.insert((*dependency, id));
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
        self.explicit_edges.insert((prerequisite, dependent));
        Ok(())
    }

    /// Validates and schedules the graph, producing synchronization, transitions, passes, and aliases.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::Cycle` if the dependency graph is cyclic or `GraphError::InvalidPass` if a transient attachment is loaded before initialization.
    pub fn compile(&self) -> Result<CompiledGraph, GraphError> {
        let hazards = self.build_hazards();
        let mut edges = self.explicit_edges.clone();
        edges.extend(hazards.iter().map(|edge| (edge.from, edge.to)));
        let order = stable_topological(self.nodes.len(), &edges)?;
        self.validate_transient_attachment_loads(&order)?;
        let positions: BTreeMap<_, _> = order
            .iter()
            .enumerate()
            .map(|(index, node)| (*node, index))
            .collect();
        let waits = self.build_waits(&order, &edges);
        let (transitions, history_states) = self.build_transitions(&order);
        let passes = self.coalesce_passes(&order, &transitions);
        let aliases = self.assign_aliases(&positions, &edges);
        Ok(CompiledGraph {
            order,
            hazards,
            waits,
            transitions,
            passes,
            aliases,
            history_states,
        })
    }

    /// Rejects loading a transient attachment before any scheduled write initializes it.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidPass` if a transient attachment is loaded before an earlier scheduled write initializes it.
    fn validate_transient_attachment_loads(&self, order: &[NodeId]) -> Result<(), GraphError> {
        let mut initialized = BTreeSet::new();
        for node in order {
            let node = &self.nodes[node.0 as usize];
            if node
                .pass
                .as_ref()
                .is_some_and(|pass| pass.load == LoadOp::Load)
            {
                let pass = node.pass.as_ref().expect("pass was checked");
                for attachment in pass.colors.iter().copied().chain(pass.depth) {
                    // Transient images have no preserved contents before the graph's first write.
                    if self.resources[attachment.0 as usize].desc.lifetime
                        == ResourceLifetime::Transient
                        && !initialized.contains(&attachment)
                    {
                        return Err(GraphError::InvalidPass);
                    }
                }
            }
            initialized.extend(
                node.accesses
                    .iter()
                    .filter(|access| is_write(access.state.access))
                    .map(|access| access.resource),
            );
        }
        Ok(())
    }

    /// Derives ordered read/write hazard edges for overlapping resource accesses.
    fn build_hazards(&self) -> Vec<HazardEdge> {
        let mut result = Vec::new();
        for later in 0..self.nodes.len() {
            for earlier in 0..later {
                for before in &self.nodes[earlier].accesses {
                    for after in &self.nodes[later].accesses {
                        if before.resource != after.resource || !overlaps(before.range, after.range)
                        {
                            continue;
                        }
                        if let Some(kind) = hazard_kind(before.state.access, after.state.access) {
                            result.push(HazardEdge::new(
                                NodeId(
                                    u32::try_from(earlier)
                                        .expect("node count is bounded by handle space"),
                                ),
                                NodeId(
                                    u32::try_from(later)
                                        .expect("node count is bounded by handle space"),
                                ),
                                before.resource,
                                kind,
                            ));
                        }
                    }
                }
            }
        }
        result.sort_unstable_by_key(|edge| (edge.from, edge.to, edge.resource, edge.kind));
        result.dedup();
        result
    }

    /// Builds external readiness waits and cross-queue dependency waits.
    fn build_waits(&self, order: &[NodeId], edges: &BTreeSet<(NodeId, NodeId)>) -> Vec<QueueWait> {
        let mut waits = Vec::new();
        let mut first = BTreeSet::new();
        for node in order {
            for access in &self.nodes[node.0 as usize].accesses {
                if first.insert(access.resource)
                    && let Some(token) = self.resources[access.resource.0 as usize].ready
                {
                    waits.push(QueueWait {
                        node: *node,
                        source: None,
                        external: Some(token),
                    });
                }
            }
        }
        for (from, to) in edges {
            let source_queue = self.nodes[from.0 as usize].queue;
            let target_queue = self.nodes[to.0 as usize].queue;
            let wait = QueueWait {
                node: *to,
                source: Some(*from),
                external: None,
            };
            if source_queue != target_queue && !waits.contains(&wait) {
                waits.push(wait);
            }
        }
        waits
    }

    /// Derives per-range state transitions and final uniform history states.
    fn build_transitions(
        &self,
        order: &[NodeId],
    ) -> (Vec<Transition>, BTreeMap<ResourceId, ResourceState>) {
        let mut tracked: BTreeMap<ResourceId, Vec<(ResourceRange, ResourceState)>> =
            BTreeMap::new();
        for (index, resource) in self.resources.iter().enumerate() {
            if let Some(state) = resource.initial {
                tracked.insert(
                    ResourceId(u32::try_from(index).expect("validated index fits u32")),
                    vec![(full_range(&resource.desc), state)],
                );
            }
        }
        for (resource, state) in &self.history {
            tracked.entry(*resource).or_default().push((
                full_range(&self.resources[resource.0 as usize].desc),
                *state,
            ));
        }
        let mut transitions = Vec::new();
        for node in order {
            for access in &self.nodes[node.0 as usize].accesses {
                let states = tracked.entry(access.resource).or_default();
                let mut uncovered = vec![access.range];
                for (range, state) in states.iter() {
                    let Some(overlap) = intersection(*range, access.range) else {
                        continue;
                    };
                    if *state != access.state {
                        transitions.push(Transition {
                            node: *node,
                            resource: access.resource,
                            range: overlap,
                            before: Some(*state),
                            after: access.state,
                        });
                    }
                    uncovered = uncovered
                        .into_iter()
                        .flat_map(|range| subtract(range, overlap))
                        .collect();
                }
                transitions.extend(uncovered.into_iter().map(|range| Transition {
                    node: *node,
                    resource: access.resource,
                    range,
                    before: None,
                    after: access.state,
                }));
                let mut updated = Vec::new();
                for (range, state) in states.drain(..) {
                    updated.extend(
                        subtract(range, access.range)
                            .into_iter()
                            .map(|remainder| (remainder, state)),
                    );
                }
                updated.push((access.range, access.state));
                *states = updated;
            }
        }
        let mut history = BTreeMap::new();
        for (resource, states) in tracked {
            if self.resources[resource.0 as usize].desc.lifetime
                != ResourceLifetime::PersistentHistory
            {
                continue;
            }
            if let Some(state) = states.first().map(|(_, state)| *state)
                && states.iter().all(|(_, candidate)| *candidate == state)
            {
                history.insert(resource, state);
            }
        }
        (transitions, history)
    }

    /// Merges adjacent compatible render nodes when no transition interrupts them.
    fn coalesce_passes(&self, order: &[NodeId], transitions: &[Transition]) -> Vec<CompiledPass> {
        let mut passes: Vec<CompiledPass> = Vec::new();
        let mut previous_was_pass = false;
        for node in order {
            let Some(info) = self.nodes[node.0 as usize].pass.clone() else {
                previous_was_pass = false;
                continue;
            };
            let requires_transition = transitions
                .iter()
                .any(|transition| transition.node == *node);
            match passes.last_mut() {
                Some(pass)
                    if previous_was_pass
                        && !requires_transition
                        && pass_compatible(&pass.info, &info) =>
                {
                    pass.info.store = info.store;
                    pass.nodes.push(*node);
                }
                _ => passes.push(CompiledPass {
                    info,
                    nodes: vec![*node],
                }),
            }
            previous_was_pass = true;
        }
        passes
    }

    /// Assigns reusable storage slots to nonoverlapping compatible transient resources.
    fn assign_aliases(
        &self,
        positions: &BTreeMap<NodeId, usize>,
        edges: &BTreeSet<(NodeId, NodeId)>,
    ) -> BTreeMap<ResourceId, AliasAssignment> {
        let mut intervals = Vec::new();
        for (index, resource) in self.resources.iter().enumerate() {
            if resource.desc.lifetime != ResourceLifetime::Transient {
                continue;
            }
            let id = ResourceId(u32::try_from(index).expect("validated index fits u32"));
            let uses: Vec<_> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.accesses.iter().any(|access| access.resource == id))
                .map(|(node, _)| NodeId(u32::try_from(node).expect("validated index fits u32")))
                .collect();
            if let (Some(first), Some(last)) = (
                uses.iter().min_by_key(|node| positions[node]),
                uses.iter().max_by_key(|node| positions[node]),
            ) {
                intervals.push((id, *first, *last));
            }
        }
        intervals.sort_unstable_by_key(|(id, first, _)| (positions[first], *id));
        let mut slots: Vec<(AliasClass, NodeId)> = Vec::new();
        let mut result = BTreeMap::new();
        for (id, first, last) in intervals {
            let class = alias_class(&self.resources[id.0 as usize].desc);
            let slot = slots
                .iter()
                .position(|(existing, prior_last)| {
                    let same_queue = self.nodes[prior_last.0 as usize].queue
                        == self.nodes[first.0 as usize].queue;
                    *existing == class && (same_queue || reachable(*prior_last, first, edges))
                })
                .unwrap_or_else(|| {
                    slots.push((class.clone(), first));
                    slots.len() - 1
                });
            slots[slot].1 = last;
            result.insert(
                id,
                AliasAssignment {
                    slot: u32::try_from(slot).expect("validated index fits u32"),
                },
            );
        }
        result
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
    /// Transient storage assignments indexed by resource.
    aliases: BTreeMap<ResourceId, AliasAssignment>,
    /// Final uniform states retained for persistent history resources.
    history_states: BTreeMap<ResourceId, ResourceState>,
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
        self.aliases.get(&resource).copied()
    }
    /// Returns the final uniform state retained for a history resource, if any.
    pub fn history_state(&self, resource: ResourceId) -> Option<ResourceState> {
        self.history_states.get(&resource).copied()
    }
}

/// Reports whether directed edges connect one node to another.
fn reachable(from: NodeId, to: NodeId, edges: &BTreeSet<(NodeId, NodeId)>) -> bool {
    let mut pending = vec![from];
    let mut visited = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if !visited.insert(node) {
            continue;
        }
        for (_, next) in edges.range((node, NodeId(0))..=(node, NodeId(u32::MAX))) {
            if *next == to {
                return true;
            }
            pending.push(*next);
        }
    }
    false
}

/// Produces a deterministic topological order, preferring lower node indices.
///
/// # Errors
///
/// Returns `GraphError::Cycle` if the directed edges contain a cycle.
fn stable_topological(
    count: usize,
    edges: &BTreeSet<(NodeId, NodeId)>,
) -> Result<Vec<NodeId>, GraphError> {
    let mut indegree = vec![0_u32; count];
    let mut outgoing: Vec<Vec<NodeId>> = vec![Vec::new(); count];
    for (from, to) in edges {
        indegree[to.0 as usize] += 1;
        outgoing[from.0 as usize].push(*to);
    }
    let mut ready: BTreeSet<_> = indegree
        .iter()
        .enumerate()
        .filter(|(_, value)| **value == 0)
        .map(|(index, _)| NodeId(u32::try_from(index).expect("validated index fits u32")))
        .collect();
    let mut order = Vec::with_capacity(count);

    while let Some(node) = ready.pop_first() {
        order.push(node);
        for next in &outgoing[node.0 as usize] {
            indegree[next.0 as usize] -= 1;
            if indegree[next.0 as usize] == 0 {
                ready.insert(*next);
            }
        }
    }
    if order.len() != count {
        return Err(GraphError::Cycle {
            nodes: indegree
                .iter()
                .enumerate()
                .filter(|(_, value)| **value > 0)
                .map(|(index, _)| NodeId(u32::try_from(index).expect("validated index fits u32")))
                .collect(),
        });
    }
    Ok(order)
}
mod validation;
pub use validation::GraphError;
use validation::{
    AliasClass, alias_class, full_range, hazard_kind, intersection, is_write, overlaps,
    pass_compatible, subtract, validate_access, validate_pass,
};
