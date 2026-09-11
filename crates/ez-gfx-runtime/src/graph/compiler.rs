use super::{
    Access, AliasAssignment, AliasClass, BinaryHeap, CompiledGraph, CompiledPass,
    FRAME_WORKSPACE_BYTE_LIMIT, FrameGraph, GRAPH_TEMPLATE_SCHEMA, GraphError, GraphTemplateCache,
    HazardEdge, LoadOp, NodeId, QueueWait, ResourceAccess, ResourceId, ResourceLifetime,
    ResourceRange, ResourceShape, ResourceState, Reverse, StoreOp, Transition, alias_class,
    full_range, hazard_kind, intersection, is_write, nested_vec_bytes, overlaps, pass_compatible,
    size_of, subtract, vec_bytes,
};
#[derive(Default)]
pub(crate) struct GraphWorkspace {
    explicit_order: Vec<NodeId>,
    edges: Vec<(NodeId, NodeId)>,
    positions: Vec<usize>,
    indegree: Vec<u32>,
    outgoing: Vec<Vec<NodeId>>,
    ready: BinaryHeap<Reverse<NodeId>>,
    initialized: Vec<Vec<ResourceRange>>,
    tracked: Vec<Vec<(ResourceRange, ResourceState)>>,
    tracked_spare: Vec<Vec<(ResourceRange, ResourceState)>>,
    uncovered: Vec<ResourceRange>,
    remainders: Vec<ResourceRange>,
    updated_states: Vec<(ResourceRange, ResourceState)>,
    pass_nodes: Vec<Vec<NodeId>>,
    intervals: Vec<(ResourceId, NodeId, NodeId)>,
    slots: Vec<(AliasClass, NodeId)>,
    traversal_pending: Vec<NodeId>,
    traversal_seen: Vec<bool>,
    structure: Vec<u8>,
}

impl FrameGraph {
    // Names, ready tokens, initial states, and history are deliberately omitted: none changes the
    // stable schedule identity, and the dynamic values are recomputed on every cache hit.
    fn encode_structure(&self, output: &mut Vec<u8>) {
        output.clear();
        push_u32(output, GRAPH_TEMPLATE_SCHEMA);
        push_usize(output, self.resources.len());
        for resource in &self.resources {
            match resource.desc.shape {
                ResourceShape::Buffer { size, alignment } => {
                    output.push(0);
                    push_u64(output, size);
                    push_u64(output, alignment);
                }
                ResourceShape::Image {
                    width,
                    height,
                    mips,
                    layers,
                    format,
                    samples,
                } => {
                    output.push(1);
                    push_u32(output, width);
                    push_u32(output, height);
                    push_u32(output, mips);
                    push_u32(output, layers);
                    output.push(format as u8);
                    output.push(samples);
                }
            }
            output.push(resource.desc.lifetime as u8);
        }

        push_usize(output, self.nodes.len());
        for node in &self.nodes {
            output.push(node.queue as u8);
            push_usize(output, node.accesses.len());
            for access in &node.accesses {
                push_u32(output, access.resource.0);
                match access.range {
                    ResourceRange::Buffer(range) => {
                        output.push(0);
                        push_u64(output, range.offset);
                        push_u64(output, range.size);
                    }
                    ResourceRange::Image(range) => {
                        output.push(1);
                        push_u32(output, range.first_mip);
                        push_u32(output, range.mip_count);
                        push_u32(output, range.first_layer);
                        push_u32(output, range.layer_count);
                    }
                }
                output.push(access.state.queue as u8);
                output.push(access.state.stage as u8);
                output.push(access.state.access as u8);
            }
            push_usize(output, node.dependencies.len());
            for dependency in &node.dependencies {
                push_u32(output, dependency.0);
            }
            match &node.pass {
                Some(pass) => {
                    output.push(1);
                    push_usize(output, pass.colors.len());
                    for color in &pass.colors {
                        push_u32(output, color.0);
                    }
                    push_u32(output, pass.depth.map_or(u32::MAX, |depth| depth.0));
                    for value in pass.area {
                        push_u32(output, value);
                    }
                    output.push(pass.samples);
                    output.push(pass.load as u8);
                    output.push(pass.store as u8);
                }
                None => output.push(0),
            }
        }

        push_usize(output, self.explicit_edges.len());
        for (prerequisite, dependent) in &self.explicit_edges {
            push_u32(output, prerequisite.0);
            push_u32(output, dependent.0);
        }
    }

    /// Validates and schedules the graph, producing synchronization, transitions, passes, and aliases.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::Cycle` for cyclic dependencies, `GraphError::InvalidPass`
    /// for an invalid transient load, or `GraphError::CapacityExhausted` before the
    /// bounded reusable workspace can exceed its byte ceiling.
    pub fn compile(&self) -> Result<CompiledGraph, GraphError> {
        let mut output = CompiledGraph::default();
        let mut workspace = GraphWorkspace::default();
        self.compile_into(&mut output, &mut workspace)?;
        Ok(output)
    }

    pub(crate) fn compile_into(
        &self,
        output: &mut CompiledGraph,
        workspace: &mut GraphWorkspace,
    ) -> Result<(), GraphError> {
        self.compile_internal(output, workspace, None)
    }

    pub(crate) fn compile_cached_into(
        &self,
        output: &mut CompiledGraph,
        workspace: &mut GraphWorkspace,
        cache: &mut GraphTemplateCache,
    ) -> Result<(), GraphError> {
        self.compile_internal(output, workspace, Some(cache))
    }

    fn compile_internal(
        &self,
        output: &mut CompiledGraph,
        workspace: &mut GraphWorkspace,
        cache: Option<&mut GraphTemplateCache>,
    ) -> Result<(), GraphError> {
        let access_count = self
            .nodes
            .iter()
            .try_fold(0_usize, |count, node| {
                count.checked_add(node.accesses.len())
            })
            .ok_or(GraphError::CapacityExhausted)?;
        preflight_workspace_bytes(
            self.resources.len(),
            self.nodes.len(),
            access_count,
            self.explicit_edges.len(),
        )?;

        let result = self.compile_mutating(output, workspace, cache);
        if matches!(result, Err(GraphError::CapacityExhausted)) {
            // Every capacity failure after mutation releases the entire attempted
            // compile atomically; no partial high-water allocation survives.
            *workspace = GraphWorkspace::default();
            *output = CompiledGraph::default();
        }
        result
    }

    fn compile_mutating(
        &self,
        output: &mut CompiledGraph,
        workspace: &mut GraphWorkspace,
        mut cache: Option<&mut GraphTemplateCache>,
    ) -> Result<(), GraphError> {
        recycle_pass_nodes(&mut output.passes, &mut workspace.pass_nodes);
        output.order.clear();
        output.hazards.clear();
        output.waits.clear();
        output.transitions.clear();
        output.aliases.clear();
        output.history_states.clear();
        workspace.edges.clear();
        workspace.positions.clear();

        let digest = if cache.is_some() {
            self.encode_structure(&mut workspace.structure);
            structure_digest(&workspace.structure)
        } else {
            0
        };
        let cache_hit = cache.as_deref_mut().is_some_and(|cache| {
            cache.restore(
                &workspace.structure,
                digest,
                output,
                &mut workspace.edges,
                &mut workspace.positions,
            )
        });

        if !cache_hit {
            output
                .aliases
                .try_reserve(self.resources.len())
                .map_err(|_| GraphError::CapacityExhausted)?;
            output.aliases.resize(self.resources.len(), None);

            // Dense IDs let topological and resource state storage reuse flat vectors.
            stable_topological_into(
                self.nodes.len(),
                &self.explicit_edges,
                &mut workspace.indegree,
                &mut workspace.outgoing,
                &mut workspace.ready,
                &mut workspace.explicit_order,
            )?;
            let retained_before_hazards = workspace
                .retained_bytes()
                .saturating_add(output.retained_bytes());
            let remaining_bytes = FRAME_WORKSPACE_BYTE_LIMIT
                .checked_sub(retained_before_hazards)
                .ok_or(GraphError::CapacityExhausted)?;
            self.build_hazards_into(
                &workspace.explicit_order,
                &mut output.hazards,
                remaining_bytes,
            )?;

            workspace
                .edges
                .try_reserve(
                    self.explicit_edges
                        .len()
                        .saturating_add(output.hazards.len()),
                )
                .map_err(|_| GraphError::CapacityExhausted)?;
            workspace.edges.extend_from_slice(&self.explicit_edges);
            workspace
                .edges
                .extend(output.hazards.iter().map(|edge| (edge.from, edge.to)));
            workspace.edges.sort_unstable();
            workspace.edges.dedup();

            stable_topological_into(
                self.nodes.len(),
                &workspace.edges,
                &mut workspace.indegree,
                &mut workspace.outgoing,
                &mut workspace.ready,
                &mut output.order,
            )?;
            self.validate_transient_attachment_loads(
                &output.order,
                &mut workspace.initialized,
                &mut workspace.uncovered,
                &mut workspace.remainders,
            )?;

            workspace
                .positions
                .try_reserve(self.nodes.len())
                .map_err(|_| GraphError::CapacityExhausted)?;
            workspace.positions.resize(self.nodes.len(), usize::MAX);
            for (position, node) in output.order.iter().enumerate() {
                workspace.positions[node.0 as usize] = position;
            }

            self.assign_aliases_into(
                &workspace.positions,
                &workspace.edges,
                &mut workspace.intervals,
                &mut workspace.slots,
                &mut workspace.traversal_pending,
                &mut workspace.traversal_seen,
                &mut output.aliases,
            );
        }

        output
            .history_states
            .try_reserve(self.resources.len())
            .map_err(|_| GraphError::CapacityExhausted)?;
        output.history_states.resize(self.resources.len(), None);
        self.build_waits_into(
            &output.order,
            &workspace.edges,
            &mut workspace.traversal_seen,
            &mut output.waits,
        );
        self.build_transitions_into(
            &output.order,
            &mut workspace.tracked,
            &mut workspace.tracked_spare,
            &mut workspace.uncovered,
            &mut workspace.remainders,
            &mut workspace.updated_states,
            &mut output.transitions,
            &mut output.history_states,
        )?;
        self.coalesce_passes_into(
            &output.order,
            &output.transitions,
            &mut workspace.pass_nodes,
            &mut output.passes,
        );

        if workspace
            .retained_bytes()
            .saturating_add(output.retained_bytes())
            > FRAME_WORKSPACE_BYTE_LIMIT
        {
            return Err(GraphError::CapacityExhausted);
        }
        if !cache_hit {
            if let Some(cache) = cache {
                cache.insert(
                    &workspace.structure,
                    digest,
                    output,
                    &workspace.edges,
                    &workspace.positions,
                );
            }
        }
        Ok(())
    }

    /// Clears one recording while retaining all flat collection capacities.
    pub(crate) fn clear(&mut self) {
        self.resources.clear();
        self.nodes.clear();
        self.explicit_edges.clear();
        self.history.clear();
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        vec_bytes(&self.resources)
            .saturating_add(vec_bytes(&self.nodes))
            .saturating_add(
                self.nodes
                    .iter()
                    .map(|node| {
                        if node.name.is_heap_allocated() {
                            node.name.capacity()
                        } else {
                            0
                        }
                        .saturating_add(
                            node.accesses.capacity().saturating_mul(size_of::<Access>()),
                        )
                        .saturating_add(
                            node.dependencies
                                .capacity()
                                .saturating_mul(size_of::<NodeId>()),
                        )
                        .saturating_add(node.pass.as_ref().map_or(0, |pass| {
                            pass.colors
                                .capacity()
                                .saturating_mul(size_of::<ResourceId>())
                        }))
                    })
                    .sum::<usize>(),
            )
            .saturating_add(vec_bytes(&self.explicit_edges))
            .saturating_add(vec_bytes(&self.history))
    }

    fn validate_transient_attachment_loads(
        &self,
        order: &[NodeId],
        initialized: &mut Vec<Vec<ResourceRange>>,
        uncovered: &mut Vec<ResourceRange>,
        remainders: &mut Vec<ResourceRange>,
    ) -> Result<(), GraphError> {
        prepare_nested(initialized, self.resources.len())?;
        for node_id in order {
            let node = &self.nodes[node_id.0 as usize];
            if let Some(pass) = &node.pass {
                let attachments = pass.colors.iter().copied().chain(pass.depth);
                if pass.load == LoadOp::Load {
                    for attachment in attachments.clone() {
                        if self.resources[attachment.0 as usize].desc.lifetime
                            != ResourceLifetime::Transient
                        {
                            continue;
                        }
                        for access in node.accesses.iter().filter(|access| {
                            access.resource == attachment
                                && matches!(
                                    access.state.access,
                                    ResourceAccess::ColorAttachmentWrite
                                        | ResourceAccess::DepthStencilRead
                                        | ResourceAccess::DepthStencilWrite
                                )
                        }) {
                            uncovered.clear();
                            uncovered.push(access.range);
                            for covered in &initialized[attachment.0 as usize] {
                                remainders.clear();
                                for range in uncovered.drain(..) {
                                    remainders.extend(subtract(range, *covered));
                                }
                                core::mem::swap(uncovered, remainders);
                            }
                            if !uncovered.is_empty() {
                                return Err(GraphError::InvalidPass);
                            }
                        }
                    }
                }
            }

            for access in node
                .accesses
                .iter()
                .filter(|access| is_write(access.state.access))
            {
                initialized[access.resource.0 as usize].push(access.range);
            }

            if let Some(pass) = &node.pass
                && pass.store == StoreOp::Discard
            {
                for attachment in pass.colors.iter().copied().chain(pass.depth) {
                    let ranges = &mut initialized[attachment.0 as usize];
                    for discard in node
                        .accesses
                        .iter()
                        .filter(|access| {
                            access.resource == attachment
                                && matches!(
                                    access.state.access,
                                    ResourceAccess::ColorAttachmentWrite
                                        | ResourceAccess::DepthStencilRead
                                        | ResourceAccess::DepthStencilWrite
                                )
                        })
                        .map(|access| access.range)
                    {
                        remainders.clear();
                        for range in ranges.drain(..) {
                            remainders.extend(subtract(range, discard));
                        }
                        core::mem::swap(ranges, remainders);
                    }
                }
            }
        }
        Ok(())
    }

    fn build_hazards_into(
        &self,
        order: &[NodeId],
        result: &mut Vec<HazardEdge>,
        mut remaining_bytes: usize,
    ) -> Result<(), GraphError> {
        result.clear();
        for later_position in 0..order.len() {
            let later = order[later_position];
            for earlier in order[..later_position].iter().copied() {
                for before in &self.nodes[earlier.0 as usize].accesses {
                    for after in &self.nodes[later.0 as usize].accesses {
                        if before.resource == after.resource
                            && overlaps(before.range, after.range)
                            && let Some(kind) = hazard_kind(before.state.access, after.state.access)
                        {
                            if result.len() == result.capacity() {
                                // Existing capacity is already charged in the aggregate baseline.
                                // Geometric growth is capped by the remaining shared byte budget.
                                let element_bytes = core::mem::size_of::<HazardEdge>();
                                let affordable = remaining_bytes / element_bytes;
                                if affordable == 0 {
                                    return Err(GraphError::CapacityExhausted);
                                }
                                let additional = result.capacity().max(64).min(affordable);
                                let reserved_bytes = additional
                                    .checked_mul(element_bytes)
                                    .ok_or(GraphError::CapacityExhausted)?;
                                remaining_bytes = remaining_bytes
                                    .checked_sub(reserved_bytes)
                                    .ok_or(GraphError::CapacityExhausted)?;
                                let previous_bytes = vec_bytes(result);
                                result
                                    .try_reserve_exact(additional)
                                    .map_err(|_| GraphError::CapacityExhausted)?;
                                let growth = vec_bytes(result).saturating_sub(previous_bytes);
                                remaining_bytes = remaining_bytes
                                    .checked_sub(growth.saturating_sub(reserved_bytes))
                                    .ok_or(GraphError::CapacityExhausted)?;
                            }
                            result.push(HazardEdge::new(
                                earlier,
                                later,
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
        Ok(())
    }

    fn build_waits_into(
        &self,
        order: &[NodeId],
        edges: &[(NodeId, NodeId)],
        seen: &mut Vec<bool>,
        waits: &mut Vec<QueueWait>,
    ) {
        waits.clear();
        seen.clear();
        seen.resize(self.resources.len(), false);
        for node in order {
            for access in &self.nodes[node.0 as usize].accesses {
                let first = &mut seen[access.resource.0 as usize];
                if !*first {
                    *first = true;
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
        for &(from, to) in edges {
            if self.nodes[from.0 as usize].queue != self.nodes[to.0 as usize].queue {
                let wait = QueueWait {
                    node: to,
                    source: Some(from),
                    external: None,
                };
                if !waits.contains(&wait) {
                    waits.push(wait);
                }
            }
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "typed scratch vectors keep hot-path storage reusable"
    )]
    fn build_transitions_into(
        &self,
        order: &[NodeId],
        tracked: &mut Vec<Vec<(ResourceRange, ResourceState)>>,
        tracked_spare: &mut Vec<Vec<(ResourceRange, ResourceState)>>,
        uncovered: &mut Vec<ResourceRange>,
        remainders: &mut Vec<ResourceRange>,
        updated: &mut Vec<(ResourceRange, ResourceState)>,
        transitions: &mut Vec<Transition>,
        history_states: &mut [Option<ResourceState>],
    ) -> Result<(), GraphError> {
        prepare_nested_retained(tracked, tracked_spare, self.resources.len())?;
        for (index, resource) in self.resources.iter().enumerate() {
            if let Some(state) = resource.initial {
                tracked[index].push((full_range(&resource.desc), state));
            }
            if let Some(state) = self.history[index] {
                tracked[index].push((full_range(&resource.desc), state));
            }
        }

        transitions.clear();
        for node in order {
            for access in &self.nodes[node.0 as usize].accesses {
                let states = &mut tracked[access.resource.0 as usize];
                uncovered.clear();
                uncovered.push(access.range);
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
                    remainders.clear();
                    for range in uncovered.drain(..) {
                        remainders.extend(subtract(range, overlap));
                    }
                    core::mem::swap(uncovered, remainders);
                }
                transitions.extend(uncovered.drain(..).map(|range| Transition {
                    node: *node,
                    resource: access.resource,
                    range,
                    before: None,
                    after: access.state,
                }));

                updated.clear();
                for (range, state) in states.drain(..) {
                    updated.extend(
                        subtract(range, access.range)
                            .into_iter()
                            .map(|remainder| (remainder, state)),
                    );
                }
                updated.push((access.range, access.state));
                core::mem::swap(states, updated);
            }
        }

        for (index, states) in tracked.iter().enumerate() {
            if self.resources[index].desc.lifetime == ResourceLifetime::PersistentHistory
                && let Some(state) = states.first().map(|(_, state)| *state)
                && states.iter().all(|(_, candidate)| *candidate == state)
            {
                history_states[index] = Some(state);
            }
        }
        Ok(())
    }

    fn coalesce_passes_into(
        &self,
        order: &[NodeId],
        transitions: &[Transition],
        reusable_nodes: &mut Vec<Vec<NodeId>>,
        passes: &mut Vec<CompiledPass>,
    ) {
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
                _ => {
                    let mut nodes = reusable_nodes.pop().unwrap_or_default();
                    nodes.push(*node);
                    passes.push(CompiledPass { info, nodes });
                }
            }
            previous_was_pass = true;
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "typed alias scratch vectors avoid per-frame allocation"
    )]
    fn assign_aliases_into(
        &self,
        positions: &[usize],
        edges: &[(NodeId, NodeId)],
        intervals: &mut Vec<(ResourceId, NodeId, NodeId)>,
        slots: &mut Vec<(AliasClass, NodeId)>,
        pending: &mut Vec<NodeId>,
        seen: &mut Vec<bool>,
        aliases: &mut [Option<AliasAssignment>],
    ) {
        intervals.clear();
        for (index, resource) in self.resources.iter().enumerate() {
            if resource.desc.lifetime != ResourceLifetime::Transient {
                continue;
            }
            let id = ResourceId(u32::try_from(index).expect("validated index fits u32"));
            let mut first = None;
            let mut last = None;
            for (node_index, node) in self.nodes.iter().enumerate() {
                if node.accesses.iter().any(|access| access.resource == id) {
                    let node = NodeId(u32::try_from(node_index).expect("validated index fits u32"));
                    if first.is_none_or(|candidate: NodeId| {
                        positions[node.0 as usize] < positions[candidate.0 as usize]
                    }) {
                        first = Some(node);
                    }
                    if last.is_none_or(|candidate: NodeId| {
                        positions[node.0 as usize] > positions[candidate.0 as usize]
                    }) {
                        last = Some(node);
                    }
                }
            }
            if let (Some(first), Some(last)) = (first, last) {
                intervals.push((id, first, last));
            }
        }
        intervals.sort_unstable_by_key(|(id, first, _)| (positions[first.0 as usize], *id));
        slots.clear();
        for &(id, first, last) in intervals.iter() {
            let class = alias_class(&self.resources[id.0 as usize].desc);
            let slot = slots
                .iter()
                .position(|(existing, prior_last)| {
                    let same_queue = self.nodes[prior_last.0 as usize].queue
                        == self.nodes[first.0 as usize].queue;
                    *existing == class
                        && (same_queue
                            || reachable_dense(
                                *prior_last,
                                first,
                                edges,
                                pending,
                                seen,
                                self.nodes.len(),
                            ))
                })
                .unwrap_or_else(|| {
                    slots.push((class.clone(), first));
                    slots.len() - 1
                });
            slots[slot].1 = last;
            aliases[id.0 as usize] = Some(AliasAssignment {
                slot: u32::try_from(slot).expect("validated index fits u32"),
            });
        }
    }
}
impl GraphWorkspace {
    pub(crate) fn retained_bytes(&self) -> usize {
        vec_bytes(&self.explicit_order)
            .saturating_add(vec_bytes(&self.edges))
            .saturating_add(vec_bytes(&self.positions))
            .saturating_add(vec_bytes(&self.indegree))
            .saturating_add(nested_vec_bytes(&self.outgoing))
            .saturating_add(
                self.ready
                    .capacity()
                    .saturating_mul(size_of::<Reverse<NodeId>>()),
            )
            .saturating_add(nested_vec_bytes(&self.initialized))
            .saturating_add(nested_vec_bytes(&self.tracked))
            .saturating_add(nested_vec_bytes(&self.tracked_spare))
            .saturating_add(vec_bytes(&self.uncovered))
            .saturating_add(vec_bytes(&self.remainders))
            .saturating_add(vec_bytes(&self.updated_states))
            .saturating_add(nested_vec_bytes(&self.pass_nodes))
            .saturating_add(vec_bytes(&self.intervals))
            .saturating_add(vec_bytes(&self.slots))
            .saturating_add(vec_bytes(&self.traversal_pending))
            .saturating_add(vec_bytes(&self.traversal_seen))
            .saturating_add(vec_bytes(&self.structure))
    }
}

fn preflight_workspace_bytes(
    resources: usize,
    nodes: usize,
    accesses: usize,
    explicit_edges: usize,
) -> Result<usize, GraphError> {
    // This is only the graph's linear floor. Actual hazard candidates are
    // counted and reserved fallibly after explicit topological ordering.
    let total = resources
        .checked_mul(64)
        .and_then(|bytes| bytes.checked_add(nodes.checked_mul(128)?))
        .and_then(|bytes| bytes.checked_add(accesses.checked_mul(64)?))
        .and_then(|bytes| bytes.checked_add(explicit_edges.checked_mul(32)?))
        .ok_or(GraphError::CapacityExhausted)?;
    if total > FRAME_WORKSPACE_BYTE_LIMIT {
        Err(GraphError::CapacityExhausted)
    } else {
        Ok(total)
    }
}

fn prepare_nested<T>(values: &mut Vec<Vec<T>>, len: usize) -> Result<(), GraphError> {
    values
        .try_reserve(len.saturating_sub(values.len()))
        .map_err(|_| GraphError::CapacityExhausted)?;
    values.resize_with(len, Vec::new);
    for value in values.iter_mut() {
        value.clear();
    }
    Ok(())
}

fn prepare_nested_retained<T>(
    values: &mut Vec<Vec<T>>,
    spare: &mut Vec<Vec<T>>,
    len: usize,
) -> Result<(), GraphError> {
    while values.len() > len {
        let mut value = values.pop().expect("length checked");
        value.clear();
        spare.push(value);
    }
    values
        .try_reserve(len.saturating_sub(values.len()))
        .map_err(|_| GraphError::CapacityExhausted)?;
    while values.len() < len {
        values.push(spare.pop().unwrap_or_default());
    }
    for value in values.iter_mut() {
        value.clear();
    }
    Ok(())
}

fn recycle_pass_nodes(passes: &mut Vec<CompiledPass>, reusable: &mut Vec<Vec<NodeId>>) {
    reusable.extend(passes.drain(..).map(|mut pass| {
        pass.nodes.clear();
        pass.nodes
    }));
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_usize(output: &mut Vec<u8>, value: usize) {
    push_u64(
        output,
        u64::try_from(value).expect("supported target pointer width fits u64"),
    );
}

fn structure_digest(structure: &[u8]) -> u64 {
    // FNV-1a is a bounded lookup accelerator only; exact encoded bytes are compared on every hit.
    structure
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |digest, byte| {
            (digest ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}
fn reachable_dense(
    from: NodeId,
    to: NodeId,
    edges: &[(NodeId, NodeId)],
    pending: &mut Vec<NodeId>,
    seen: &mut Vec<bool>,
    node_count: usize,
) -> bool {
    pending.clear();
    pending.push(from);
    seen.clear();
    seen.resize(node_count, false);
    while let Some(node) = pending.pop() {
        let visited = &mut seen[node.0 as usize];
        if *visited {
            continue;
        }
        *visited = true;
        for &(_, next) in edges.iter().filter(|(source, _)| *source == node) {
            if next == to {
                return true;
            }
            pending.push(next);
        }
    }
    false
}

#[allow(
    clippy::too_many_arguments,
    reason = "caller-owned vectors retain topological scratch capacity"
)]
fn stable_topological_into(
    count: usize,
    edges: &[(NodeId, NodeId)],
    indegree: &mut Vec<u32>,
    outgoing: &mut Vec<Vec<NodeId>>,
    ready: &mut BinaryHeap<Reverse<NodeId>>,
    order: &mut Vec<NodeId>,
) -> Result<(), GraphError> {
    indegree.clear();
    indegree.resize(count, 0);
    prepare_nested(outgoing, count)?;
    for &(from, to) in edges {
        indegree[to.0 as usize] = indegree[to.0 as usize]
            .checked_add(1)
            .ok_or(GraphError::CapacityExhausted)?;
        outgoing[from.0 as usize].push(to);
    }
    ready.clear();
    for (index, value) in indegree.iter().enumerate() {
        if *value == 0 {
            ready.push(Reverse(NodeId(
                u32::try_from(index).map_err(|_| GraphError::CapacityExhausted)?,
            )));
        }
    }
    order.clear();
    order
        .try_reserve(count)
        .map_err(|_| GraphError::CapacityExhausted)?;

    while let Some(Reverse(node)) = ready.pop() {
        order.push(node);
        for next in &outgoing[node.0 as usize] {
            indegree[next.0 as usize] -= 1;
            if indegree[next.0 as usize] == 0 {
                ready.push(Reverse(*next));
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{GraphWorkspace, preflight_workspace_bytes};
    use crate::graph::{
        Access, CompiledGraph, FRAME_WORKSPACE_BYTE_LIMIT, FrameGraph, GraphError, NodeDesc,
        ResourceDesc, ResourceLifetime,
    };
    use ez_gfx_hal::{BufferRange, QueueKind, ResourceAccess, ResourceState, ShaderStage};

    #[test]
    fn workspace_preflight_rejects_overflow_and_declared_limit() {
        assert!(preflight_workspace_bytes(2, 3, 4, 1).unwrap() < FRAME_WORKSPACE_BYTE_LIMIT);
        assert!(preflight_workspace_bytes(usize::MAX, 1, 1, 1).is_err());
        assert!(preflight_workspace_bytes(1, 1, 2_000_000, 1).is_err());
        assert_eq!(FRAME_WORKSPACE_BYTE_LIMIT, 64 * 1024 * 1024);
    }

    #[test]
    fn large_read_only_graph_is_not_rejected_as_quadratic() {
        let mut graph = FrameGraph::new();
        let resource = graph
            .add_resource(ResourceDesc::buffer(4, 4, ResourceLifetime::External).unwrap())
            .unwrap();
        let state = ResourceState::new(
            QueueKind::Graphics,
            ShaderStage::Vertex,
            ResourceAccess::SampledRead,
        )
        .unwrap();
        graph.set_resource_initial_state(resource, state).unwrap();
        for _ in 0..1025 {
            graph
                .add_node(
                    NodeDesc::new("read", QueueKind::Graphics).access(Access::buffer(
                        resource,
                        BufferRange::new(0, 4).unwrap(),
                        state,
                    )),
                )
                .unwrap();
        }

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.order().len(), 1025);
        assert!(compiled.hazards().is_empty());
    }

    #[test]
    fn hazard_overflow_discards_all_mutated_compile_storage() {
        let mut graph = FrameGraph::new();
        let resource = graph
            .add_resource(ResourceDesc::buffer(4, 4, ResourceLifetime::External).unwrap())
            .unwrap();
        let state = ResourceState::new(
            QueueKind::Compute,
            ShaderStage::Compute,
            ResourceAccess::StorageWrite,
        )
        .unwrap();
        graph.set_resource_initial_state(resource, state).unwrap();
        for _ in 0..4096 {
            graph
                .add_node(
                    NodeDesc::new("write", QueueKind::Compute).access(Access::buffer(
                        resource,
                        BufferRange::new(0, 4).unwrap(),
                        state,
                    )),
                )
                .unwrap();
        }
        let mut output = CompiledGraph::default();
        let mut workspace = GraphWorkspace::default();

        assert_eq!(
            graph.compile_into(&mut output, &mut workspace),
            Err(GraphError::CapacityExhausted)
        );
        assert_eq!(workspace.retained_bytes(), 0);
        assert_eq!(output.retained_bytes(), 0);
        assert!(output.order().is_empty());
        assert!(output.hazards().is_empty());
    }
}
