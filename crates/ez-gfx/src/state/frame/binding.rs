use ez_gfx_runtime::binding::{
    BindingError, BindingKind, PublicBinding, ReflectedBindings, ResourceIdentity,
};

/// Allocation-free view of the bindings relevant to one reflected layout.
#[derive(Clone, Copy)]
pub(in crate::state) struct BindingProjection<'a> {
    layout: &'a ReflectedBindings,
    bindings: &'a [PublicBinding],
}

pub(in crate::state) struct ProjectedResources<'a> {
    requirements: core::slice::Iter<'a, ez_gfx_runtime::binding::BindingRequirement>,
    bindings: &'a [PublicBinding],
}

impl Iterator for ProjectedResources<'_> {
    type Item = ResourceIdentity;

    fn next(&mut self) -> Option<Self::Item> {
        self.requirements
            .find(|requirement| requirement.kind != BindingKind::VertexHeap)
            .map(|requirement| {
                self.bindings
                    .iter()
                    .find(|binding| binding.name == requirement.name)
                    .expect("validated projection contains every reflected binding")
                    .resource
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ProjectedResources<'_> {
    fn len(&self) -> usize {
        self.requirements
            .clone()
            .filter(|requirement| requirement.kind != BindingKind::VertexHeap)
            .count()
    }
}

impl<'a> BindingProjection<'a> {
    pub(in crate::state) const fn new(
        layout: &'a ReflectedBindings,
        bindings: &'a [PublicBinding],
    ) -> Self {
        Self { layout, bindings }
    }

    pub(in crate::state) fn iter(self) -> impl Iterator<Item = &'a PublicBinding> + Clone {
        self.bindings.iter().filter(move |binding| {
            self.layout.requirements().iter().any(|requirement| {
                requirement.kind != BindingKind::VertexHeap && requirement.name == binding.name
            })
        })
    }

    pub(in crate::state) fn resources(self) -> ProjectedResources<'a> {
        ProjectedResources {
            requirements: self.layout.requirements().iter(),
            bindings: self.bindings,
        }
    }

    pub(in crate::state) fn validate(self) -> Result<(), BindingError> {
        for (index, binding) in self.iter().enumerate() {
            if self
                .iter()
                .take(index)
                .any(|prior| prior.name == binding.name)
            {
                return Err(BindingError::Duplicate(binding.name.clone()));
            }
        }
        for requirement in self
            .layout
            .requirements()
            .iter()
            .filter(|requirement| requirement.kind != BindingKind::VertexHeap)
        {
            let binding = self
                .bindings
                .iter()
                .find(|binding| binding.name == requirement.name)
                .ok_or_else(|| BindingError::Missing(requirement.name.clone()))?;
            if binding.resource.kind() != requirement.kind {
                return Err(BindingError::KindMismatch(requirement.name.clone()));
            }
        }
        Ok(())
    }

    pub(in crate::state) fn arenas_are_read_only(
        self,
        arenas: &std::collections::HashSet<ez_gfx_core::handle::PackedHandle>,
    ) -> bool {
        self.layout.requirements().iter().all(|requirement| {
            if !requirement.writable || requirement.kind != BindingKind::Buffer {
                return true;
            }
            self.bindings
                .iter()
                .find(|binding| binding.name == requirement.name)
                .is_none_or(|binding| match binding.resource {
                    ResourceIdentity::Buffer(handle) => !arenas.contains(&handle.packed()),
                    ResourceIdentity::Counter(_) | ResourceIdentity::RenderTarget(_) => true,
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use ez_gfx_artifact::Stage;
    use ez_gfx_core::{
        Backend,
        handle::{BufferHandle, CounterBufferHandle, LocalHandle, PackedHandle},
    };
    use ez_gfx_runtime::binding::{
        BindingError, PublicBinding, ReflectedBindings, ResourceIdentity,
    };

    use super::BindingProjection;

    const METADATA: &[u8] = br#"{"reflections":[{"target":"Spirv","entry":"main","stage":"Compute","reflection":{"parameters":[{"semantic_name":"instances","api_kind":"buffer","binding_index":0,"binding_space":0},{"semantic_name":"draws","api_kind":"counter_buffer","binding_index":1,"binding_space":0,"descriptor_count":2}]}}]}"#;

    fn layout() -> ReflectedBindings {
        ReflectedBindings::parse(METADATA, Backend::Vulkan, "main", Stage::Compute).unwrap()
    }

    fn packed(slot: u32) -> PackedHandle {
        PackedHandle::child(
            LocalHandle::new(1, 1).unwrap(),
            LocalHandle::new(slot, 1).unwrap(),
        )
        .unwrap()
    }

    fn buffer(name: &str, slot: u32) -> PublicBinding {
        PublicBinding {
            name: name.into(),
            resource: ResourceIdentity::Buffer(BufferHandle::from_packed(packed(slot)).unwrap()),
        }
    }

    fn counter(name: &str, slot: u32) -> PublicBinding {
        PublicBinding {
            name: name.into(),
            resource: ResourceIdentity::Counter(
                CounterBufferHandle::from_packed(packed(slot)).unwrap(),
            ),
        }
    }

    #[test]
    fn unrelated_persisted_binding_is_ignored_across_shader_switch() {
        let current = layout();
        let bindings = [
            buffer("prior-shader-only", 1),
            buffer("instances", 2),
            counter("draws", 3),
        ];

        let projection = BindingProjection::new(&current, &bindings);

        assert_eq!(
            projection.iter().collect::<Vec<_>>(),
            bindings[1..].iter().collect::<Vec<_>>()
        );
        assert_eq!(projection.validate(), Ok(()));
    }

    #[test]
    fn required_binding_missing_still_fails() {
        let current = layout();
        let bindings = [buffer("prior-shader-only", 1)];
        let projection = BindingProjection::new(&current, &bindings);

        assert_eq!(
            projection.validate(),
            Err(BindingError::Missing("draws".into()))
        );
    }

    #[test]
    fn wrong_type_for_required_binding_still_fails() {
        let current = layout();
        let bindings = [counter("instances", 1), counter("draws", 2)];
        let projection = BindingProjection::new(&current, &bindings);

        assert_eq!(
            projection.validate(),
            Err(BindingError::KindMismatch("instances".into()))
        );
    }

    #[test]
    fn matching_bindings_and_current_layout_duplicates_are_preserved() {
        let current = layout();
        let bindings = [
            buffer("instances", 1),
            counter("draws", 2),
            buffer("instances", 3),
        ];

        let projection = BindingProjection::new(&current, &bindings);

        assert_eq!(
            projection.iter().collect::<Vec<_>>(),
            bindings.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            projection.validate(),
            Err(BindingError::Duplicate("instances".into()))
        );
    }

    #[test]
    fn arena_buffers_must_target_read_only_requirements() {
        let read = br#"{"reflections":[{"target":"Spirv","entry":"main","stage":"Compute","reflection":{"parameters":[{"semantic_name":"arena","api_kind":"buffer","binding_index":0,"binding_space":0,"resource_access":"Read"}]}}]}"#;
        let write = br#"{"reflections":[{"target":"Spirv","entry":"main","stage":"Compute","reflection":{"parameters":[{"semantic_name":"arena","api_kind":"buffer","binding_index":0,"binding_space":0,"resource_access":"ReadWrite"}]}}]}"#;
        let read = ReflectedBindings::parse(read, Backend::Vulkan, "main", Stage::Compute).unwrap();
        let write =
            ReflectedBindings::parse(write, Backend::Vulkan, "main", Stage::Compute).unwrap();
        let bindings = [buffer("arena", 7)];
        let arenas = std::collections::HashSet::from([packed(7)]);

        assert!(BindingProjection::new(&read, &bindings).arenas_are_read_only(&arenas));
        assert!(!BindingProjection::new(&write, &bindings).arenas_are_read_only(&arenas));
    }
}
