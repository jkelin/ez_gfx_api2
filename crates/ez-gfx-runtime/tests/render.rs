use ez_gfx_runtime::render::{CullMode, DynamicPipelineState, PrimitiveTopology};

#[test]
fn dynamic_pipeline_state_validates_every_discriminant() {
    let state = DynamicPipelineState::from_abi(2, 1, 0, 1).unwrap();
    assert_eq!(state.cull, CullMode::Back);
    assert_eq!(state.topology, PrimitiveTopology::TriangleList);

    for values in [[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 6, 0], [0, 0, 0, 2]] {
        assert!(
            DynamicPipelineState::from_abi(values[0], values[1], values[2], values[3]).is_err()
        );
    }
}
