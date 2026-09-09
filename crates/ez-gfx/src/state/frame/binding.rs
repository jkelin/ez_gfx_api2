use ez_gfx_runtime::binding::{BindingKind, PublicBinding, ReflectedBindings};

/// Projects a persistent frame binding set onto one shader layout.
///
/// Unrelated names belong to earlier or later shader executions and stay dormant. Selection is
/// name-only so exact validation still rejects a required name with the wrong type or duplicates.
pub(super) fn select_bindings(
    layout: &ReflectedBindings,
    bindings: &[PublicBinding],
) -> Vec<PublicBinding> {
    bindings
        .iter()
        .filter(|binding| {
            layout.requirements().iter().any(|requirement| {
                // Vertex heaps are context-managed; a same-named public binding remains unrelated.
                requirement.kind != BindingKind::VertexHeap && requirement.name == binding.name
            })
        })
        .cloned()
        .collect()
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

    use super::select_bindings;

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

        let selected = select_bindings(&current, &bindings);

        assert_eq!(selected, bindings[1..]);
        assert_eq!(current.validate(&selected), Ok(()));
    }

    #[test]
    fn required_binding_missing_still_fails() {
        let current = layout();
        let selected = select_bindings(&current, &[buffer("prior-shader-only", 1)]);

        assert_eq!(
            current.validate(&selected),
            Err(BindingError::Missing("draws".into()))
        );
    }

    #[test]
    fn wrong_type_for_required_binding_still_fails() {
        let current = layout();
        let selected = select_bindings(&current, &[counter("instances", 1), counter("draws", 2)]);

        assert_eq!(
            current.validate(&selected),
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

        let selected = select_bindings(&current, &bindings);

        assert_eq!(selected, bindings);
        assert_eq!(
            current.validate(&selected),
            Err(BindingError::Duplicate("instances".into()))
        );
    }
}
