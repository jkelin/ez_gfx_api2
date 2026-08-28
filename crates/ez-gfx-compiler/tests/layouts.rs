use ez_gfx_compiler::{CompilationContractError, validate_target_layouts};
use ez_gfx_core::{
    Backend, ResourceAccess, ResourceKind, SemanticGraph, SemanticResource, TargetBinding,
    TargetLayout,
};

#[test]
fn compilation_contract_requires_all_native_targets() {
    let graph = SemanticGraph::new(vec![
        SemanticResource::new(
            "frame.scene",
            ResourceKind::ConstantBuffer,
            ResourceAccess::Read,
            1,
        )
        .unwrap(),
    ])
    .unwrap();
    let id = graph.resources()[0].id();
    let vulkan =
        TargetLayout::new(Backend::Vulkan, vec![TargetBinding::descriptor(id, 0, 0)]).unwrap();

    assert_eq!(
        validate_target_layouts(&graph, &[vulkan]),
        Err(CompilationContractError::MissingBackend(Backend::Dx12))
    );
}

#[test]
fn compilation_contract_accepts_distinct_native_bindings() {
    let graph = SemanticGraph::new(vec![
        SemanticResource::new(
            "frame.scene",
            ResourceKind::ConstantBuffer,
            ResourceAccess::Read,
            1,
        )
        .unwrap(),
    ])
    .unwrap();
    let id = graph.resources()[0].id();
    let layouts = [
        TargetLayout::new(Backend::Vulkan, vec![TargetBinding::descriptor(id, 0, 3)]).unwrap(),
        TargetLayout::new(Backend::Dx12, vec![TargetBinding::descriptor(id, 2, 7)]).unwrap(),
        TargetLayout::new(Backend::Metal, vec![TargetBinding::argument(id, 9)]).unwrap(),
    ];

    validate_target_layouts(&graph, &layouts).unwrap();
}
