use std::collections::{BTreeMap, BTreeSet};

use ez_gfx_hal::{BufferRange, CompletionToken, QueueKind, ResourceAccess, ResourceState};

pub use crate::target::{Format, LoadOp, StoreOp};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceId(u32);
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLifetime {
    Transient,
    PersistentHistory,
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
pub struct ResourceDesc {
    shape: ResourceShape,
    lifetime: ResourceLifetime,
}
impl ResourceDesc {
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
pub struct ImageRange {
    pub first_mip: u32,
    pub mip_count: u32,
    pub first_layer: u32,
    pub layer_count: u32,
}
impl ImageRange {
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
    pub fn all(mips: u32, layers: u32) -> Result<Self, GraphError> {
        Self::new(0, mips, 0, layers)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceRange {
    Buffer(BufferRange),
    Image(ImageRange),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Access {
    pub resource: ResourceId,
    pub range: ResourceRange,
    pub state: ResourceState,
}
impl Access {
    pub const fn buffer(resource: ResourceId, range: BufferRange, state: ResourceState) -> Self {
        Self {
            resource,
            range: ResourceRange::Buffer(range),
            state,
        }
    }
    pub const fn image(resource: ResourceId, range: ImageRange, state: ResourceState) -> Self {
        Self {
            resource,
            range: ResourceRange::Image(range),
            state,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PassInfo {
    colors: Vec<ResourceId>,
    depth: Option<ResourceId>,
    area: [u32; 4],
    samples: u8,
    load: LoadOp,
    store: StoreOp,
}
impl PassInfo {
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeDesc {
    name: String,
    queue: QueueKind,
    accesses: Vec<Access>,
    dependencies: Vec<NodeId>,
    pass: Option<PassInfo>,
}
impl NodeDesc {
    pub fn new(name: impl Into<String>, queue: QueueKind) -> Self {
        Self {
            name: name.into(),
            queue,
            accesses: Vec::new(),
            dependencies: Vec::new(),
            pass: None,
        }
    }
    pub fn access(mut self, access: Access) -> Self {
        self.accesses.push(access);
        self
    }
    pub fn depends_on(mut self, node: NodeId) -> Self {
        self.dependencies.push(node);
        self
    }
    pub fn pass(mut self, pass: PassInfo) -> Self {
        self.pass = Some(pass);
        self
    }
}

#[derive(Default)]
pub struct FrameGraph {
    resources: Vec<ResourceRecord>,
    nodes: Vec<NodeDesc>,
    explicit_edges: BTreeSet<(NodeId, NodeId)>,
    history: BTreeMap<ResourceId, ResourceState>,
}
#[derive(Clone, Debug)]
struct ResourceRecord {
    desc: ResourceDesc,
    ready: Option<CompletionToken>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_resource(&mut self, desc: ResourceDesc) -> Result<ResourceId, GraphError> {
        let id = ResourceId(
            u32::try_from(self.resources.len()).map_err(|_| GraphError::CapacityExhausted)?,
        );
        self.resources.push(ResourceRecord { desc, ready: None });
        Ok(id)
    }
    pub fn set_resource_ready(
        &mut self,
        resource: ResourceId,
        token: CompletionToken,
    ) -> Result<(), GraphError> {
        self.resource_mut(resource)?.ready = Some(token);
        Ok(())
    }
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
            for attachment in pass.colors.iter().copied().chain(pass.depth) {
                self.resource(attachment)?;
            }
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

    pub fn compile(&self) -> Result<CompiledGraph, GraphError> {
        let hazards = self.build_hazards();
        let mut edges = self.explicit_edges.clone();
        edges.extend(hazards.iter().map(|edge| (edge.from, edge.to)));
        let order = stable_topological(self.nodes.len(), &edges)?;
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
                                NodeId(earlier as u32),
                                NodeId(later as u32),
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

    fn build_waits(&self, order: &[NodeId], edges: &BTreeSet<(NodeId, NodeId)>) -> Vec<QueueWait> {
        let mut waits = Vec::new();
        let mut first = BTreeSet::new();
        for node in order {
            for access in &self.nodes[node.0 as usize].accesses {
                if first.insert(access.resource) {
                    if let Some(token) = self.resources[access.resource.0 as usize].ready {
                        waits.push(QueueWait {
                            node: *node,
                            source: None,
                            external: Some(token),
                        });
                    }
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

    fn build_transitions(
        &self,
        order: &[NodeId],
    ) -> (Vec<Transition>, BTreeMap<ResourceId, ResourceState>) {
        let mut tracked: BTreeMap<ResourceId, Vec<(ResourceRange, ResourceState)>> =
            BTreeMap::new();
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
            if let Some(state) = states.first().map(|(_, state)| *state) {
                if states.iter().all(|(_, candidate)| *candidate == state) {
                    history.insert(resource, state);
                }
            }
        }
        (transitions, history)
    }

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
            let id = ResourceId(index as u32);
            let uses: Vec<_> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.accesses.iter().any(|access| access.resource == id))
                .map(|(node, _)| NodeId(node as u32))
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
            result.insert(id, AliasAssignment { slot: slot as u32 });
        }
        result
    }

    fn resource(&self, id: ResourceId) -> Result<&ResourceRecord, GraphError> {
        self.resources
            .get(id.0 as usize)
            .ok_or(GraphError::UnknownResource)
    }
    fn resource_mut(&mut self, id: ResourceId) -> Result<&mut ResourceRecord, GraphError> {
        self.resources
            .get_mut(id.0 as usize)
            .ok_or(GraphError::UnknownResource)
    }
    fn node(&self, id: NodeId) -> Result<&NodeDesc, GraphError> {
        self.nodes.get(id.0 as usize).ok_or(GraphError::UnknownNode)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HazardKind {
    Raw,
    War,
    Waw,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HazardEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub resource: ResourceId,
    pub kind: HazardKind,
}
impl HazardEdge {
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
pub struct QueueWait {
    pub node: NodeId,
    pub source: Option<NodeId>,
    pub external: Option<CompletionToken>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Transition {
    pub node: NodeId,
    pub resource: ResourceId,
    pub range: ResourceRange,
    pub before: Option<ResourceState>,
    pub after: ResourceState,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledPass {
    pub info: PassInfo,
    pub nodes: Vec<NodeId>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AliasAssignment {
    pub slot: u32,
}

pub struct CompiledGraph {
    order: Vec<NodeId>,
    hazards: Vec<HazardEdge>,
    waits: Vec<QueueWait>,
    transitions: Vec<Transition>,
    passes: Vec<CompiledPass>,
    aliases: BTreeMap<ResourceId, AliasAssignment>,
    history_states: BTreeMap<ResourceId, ResourceState>,
}
impl CompiledGraph {
    pub fn order(&self) -> &[NodeId] {
        &self.order
    }
    pub fn hazards(&self) -> &[HazardEdge] {
        &self.hazards
    }
    pub fn waits(&self) -> &[QueueWait] {
        &self.waits
    }
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }
    pub fn passes(&self) -> &[CompiledPass] {
        &self.passes
    }
    pub fn alias(&self, resource: ResourceId) -> Option<AliasAssignment> {
        self.aliases.get(&resource).copied()
    }
    pub fn history_state(&self, resource: ResourceId) -> Option<ResourceState> {
        self.history_states.get(&resource).copied()
    }
}

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
        .map(|(index, _)| NodeId(index as u32))
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
                .map(|(index, _)| NodeId(index as u32))
                .collect(),
        });
    }
    Ok(order)
}

fn validate_access(resource: &ResourceRecord, access: &Access) -> Result<(), GraphError> {
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

fn pass_compatible(previous: &PassInfo, next: &PassInfo) -> bool {
    previous.colors == next.colors
        && previous.depth == next.depth
        && previous.area == next.area
        && previous.samples == next.samples
        && previous.store == StoreOp::Store
        && next.load == LoadOp::Load
}

fn intersection(a: ResourceRange, b: ResourceRange) -> Option<ResourceRange> {
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

fn subtract(range: ResourceRange, cut: ResourceRange) -> Vec<ResourceRange> {
    let Some(overlap) = intersection(range, cut) else {
        return vec![range];
    };
    match (range, overlap) {
        (ResourceRange::Buffer(range), ResourceRange::Buffer(overlap)) => {
            let mut result = Vec::with_capacity(2);
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
            let mut result = Vec::with_capacity(4);
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
fn overlaps(a: ResourceRange, b: ResourceRange) -> bool {
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
fn is_write(access: ResourceAccess) -> bool {
    matches!(
        access,
        ResourceAccess::StorageWrite
            | ResourceAccess::StorageReadWrite
            | ResourceAccess::ColorAttachmentWrite
            | ResourceAccess::DepthStencilWrite
            | ResourceAccess::TransferWrite
    )
}
fn hazard_kind(before: ResourceAccess, after: ResourceAccess) -> Option<HazardKind> {
    match (is_write(before), is_write(after)) {
        (true, false) => Some(HazardKind::Raw),
        (false, true) => Some(HazardKind::War),
        (true, true) => Some(HazardKind::Waw),
        _ => None,
    }
}
fn full_range(desc: &ResourceDesc) -> ResourceRange {
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
enum AliasClass {
    Buffer { alignment: u64 },
    Image { format: Format, samples: u8 },
}
fn alias_class(desc: &ResourceDesc) -> AliasClass {
    match desc.shape {
        ResourceShape::Buffer { alignment, .. } => AliasClass::Buffer { alignment },
        ResourceShape::Image {
            format, samples, ..
        } => AliasClass::Image { format, samples },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphError {
    InvalidResource,
    InvalidRange,
    RangeTypeMismatch,
    InvalidNode,
    InvalidPass,
    QueueMismatch,
    UnknownResource,
    UnknownNode,
    NotHistory,
    CapacityExhausted,
    Feedback { resource: ResourceId },
    Cycle { nodes: Vec<NodeId> },
}
