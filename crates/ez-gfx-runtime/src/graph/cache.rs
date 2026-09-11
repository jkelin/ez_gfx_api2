use super::{AliasAssignment, CompiledGraph, HazardEdge, NodeId, size_of, vec_bytes};
pub(super) const GRAPH_TEMPLATE_SCHEMA: u32 = 1;
const GRAPH_TEMPLATE_CACHE_ENTRY_LIMIT: usize = 8;
const GRAPH_TEMPLATE_CACHE_BYTE_LIMIT: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Observable state of one recorder's bounded stable-graph template cache.
pub struct GraphTemplateCacheStats {
    /// Successful collision-safe structural lookups.
    pub hits: u64,
    /// Structural lookups that required compilation.
    pub misses: u64,
    /// Successfully compiled templates considered for admission.
    pub compiles: u64,
    /// Templates removed by deterministic least-recently-used eviction.
    pub evictions: u64,
    /// Explicit cache invalidations.
    pub invalidations: u64,
    /// Currently retained template count.
    pub entries: usize,
    /// Current retained cache capacity in bytes.
    pub retained_bytes: usize,
    /// Largest retained cache capacity observed.
    pub high_water_bytes: usize,
    /// Cache generation changed by invalidation.
    pub generation: u64,
    /// Stable graph-structure encoding schema.
    pub schema: u32,
    /// Maximum retained template count.
    pub entry_limit: usize,
    /// Maximum retained cache bytes.
    pub byte_limit: usize,
}

struct GraphTemplate {
    schema: u32,
    generation: u64,
    digest: u64,
    structure: Vec<u8>,
    order: Vec<NodeId>,
    hazards: Vec<HazardEdge>,
    edges: Vec<(NodeId, NodeId)>,
    positions: Vec<usize>,
    aliases: Vec<Option<AliasAssignment>>,
    last_used: u64,
}

pub(crate) struct GraphTemplateCache {
    entries: Vec<GraphTemplate>,
    generation: u64,
    clock: u64,
    hits: u64,
    misses: u64,
    compiles: u64,
    evictions: u64,
    invalidations: u64,
    high_water_bytes: usize,
}

impl Default for GraphTemplateCache {
    fn default() -> Self {
        let entries = Vec::with_capacity(GRAPH_TEMPLATE_CACHE_ENTRY_LIMIT);
        let high_water_bytes = entries
            .capacity()
            .saturating_mul(size_of::<GraphTemplate>());
        Self {
            entries,
            generation: 1,
            clock: 0,
            hits: 0,
            misses: 0,
            compiles: 0,
            evictions: 0,
            invalidations: 0,
            high_water_bytes,
        }
    }
}

impl GraphTemplateCache {
    pub(crate) fn invalidate(&mut self) {
        self.entries.clear();
        // Clearing first makes generation wrap safe: no entry from an earlier generation survives.
        self.generation = self.generation.wrapping_add(1).max(1);
        self.invalidations = self.invalidations.saturating_add(1);
    }

    pub(crate) fn stats(&self) -> GraphTemplateCacheStats {
        GraphTemplateCacheStats {
            hits: self.hits,
            misses: self.misses,
            compiles: self.compiles,
            evictions: self.evictions,
            invalidations: self.invalidations,
            entries: self.entries.len(),
            retained_bytes: self.retained_bytes(),
            high_water_bytes: self.high_water_bytes,
            generation: self.generation,
            schema: GRAPH_TEMPLATE_SCHEMA,
            entry_limit: GRAPH_TEMPLATE_CACHE_ENTRY_LIMIT,
            byte_limit: GRAPH_TEMPLATE_CACHE_BYTE_LIMIT,
        }
    }

    pub(super) fn restore(
        &mut self,
        structure: &[u8],
        digest: u64,
        output: &mut CompiledGraph,
        edges: &mut Vec<(NodeId, NodeId)>,
        positions: &mut Vec<usize>,
    ) -> bool {
        let Some(index) = self.entries.iter().position(|entry| {
            entry.schema == GRAPH_TEMPLATE_SCHEMA
                && entry.generation == self.generation
                && entry.digest == digest
                && entry.structure == structure
        }) else {
            self.misses = self.misses.saturating_add(1);
            return false;
        };

        self.clock = self.clock.saturating_add(1);
        let entry = &mut self.entries[index];
        entry.last_used = self.clock;
        output.order.extend_from_slice(&entry.order);
        output.hazards.extend_from_slice(&entry.hazards);
        edges.extend_from_slice(&entry.edges);
        positions.extend_from_slice(&entry.positions);
        output.aliases.extend_from_slice(&entry.aliases);
        self.hits = self.hits.saturating_add(1);
        true
    }

    pub(super) fn insert(
        &mut self,
        structure: &[u8],
        digest: u64,
        output: &CompiledGraph,
        edges: &[(NodeId, NodeId)],
        positions: &[usize],
    ) {
        self.compiles = self.compiles.saturating_add(1);
        let entry_bytes = structure
            .len()
            .saturating_add(output.order.len().saturating_mul(size_of::<NodeId>()))
            .saturating_add(output.hazards.len().saturating_mul(size_of::<HazardEdge>()))
            .saturating_add(edges.len().saturating_mul(size_of::<(NodeId, NodeId)>()))
            .saturating_add(positions.len().saturating_mul(size_of::<usize>()))
            .saturating_add(
                output
                    .aliases
                    .len()
                    .saturating_mul(size_of::<Option<AliasAssignment>>()),
            );
        if self.retained_container_bytes().saturating_add(entry_bytes)
            > GRAPH_TEMPLATE_CACHE_BYTE_LIMIT
        {
            return;
        }

        while self.entries.len() >= GRAPH_TEMPLATE_CACHE_ENTRY_LIMIT
            || self.retained_bytes().saturating_add(entry_bytes) > GRAPH_TEMPLATE_CACHE_BYTE_LIMIT
        {
            let Some((index, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_used, *index))
            else {
                return;
            };
            self.entries.remove(index);
            self.evictions = self.evictions.saturating_add(1);
        }

        self.clock = self.clock.saturating_add(1);
        self.entries.push(GraphTemplate {
            schema: GRAPH_TEMPLATE_SCHEMA,
            generation: self.generation,
            digest,
            structure: structure.to_vec(),
            order: output.order.clone(),
            hazards: output.hazards.clone(),
            edges: edges.to_vec(),
            positions: positions.to_vec(),
            aliases: output.aliases.clone(),
            last_used: self.clock,
        });
        while self.retained_bytes() > GRAPH_TEMPLATE_CACHE_BYTE_LIMIT {
            let Some((index, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_used, *index))
            else {
                break;
            };
            self.entries.remove(index);
            self.evictions = self.evictions.saturating_add(1);
        }
        self.high_water_bytes = self.high_water_bytes.max(self.retained_bytes());
    }

    fn retained_container_bytes(&self) -> usize {
        self.entries
            .capacity()
            .saturating_mul(size_of::<GraphTemplate>())
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_container_bytes().saturating_add(
            self.entries
                .iter()
                .map(GraphTemplate::retained_bytes)
                .sum::<usize>(),
        )
    }
}

impl GraphTemplate {
    fn retained_bytes(&self) -> usize {
        vec_bytes(&self.structure)
            .saturating_add(vec_bytes(&self.order))
            .saturating_add(vec_bytes(&self.hazards))
            .saturating_add(vec_bytes(&self.edges))
            .saturating_add(vec_bytes(&self.positions))
            .saturating_add(vec_bytes(&self.aliases))
    }
}

#[cfg(test)]
mod tests {
    use super::{GRAPH_TEMPLATE_SCHEMA, GraphTemplate, GraphTemplateCache};
    use crate::graph::CompiledGraph;

    #[test]
    fn template_digest_collision_requires_exact_structure_match() {
        let mut cache = GraphTemplateCache::default();
        cache.entries.push(GraphTemplate {
            schema: GRAPH_TEMPLATE_SCHEMA,
            generation: cache.generation,
            digest: 7,
            structure: vec![1],
            order: Vec::new(),
            hazards: Vec::new(),
            edges: Vec::new(),
            positions: Vec::new(),
            aliases: Vec::new(),
            last_used: 0,
        });
        let mut output = CompiledGraph::default();
        let mut edges = Vec::new();
        let mut positions = Vec::new();

        assert!(!cache.restore(&[2], 7, &mut output, &mut edges, &mut positions));
        assert_eq!((cache.hits, cache.misses), (0, 1));
    }
}
