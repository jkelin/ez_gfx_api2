use super::*;
use std::collections::HashSet;

#[test]
fn pipeline_layout_keys_ignore_reflection_order() {
    let first = ez_gfx_hal::ShaderBufferLayout::new(0, 3, 1, false).unwrap();
    let second = ez_gfx_hal::ShaderBufferLayout::new(0, 1, 2, true).unwrap();

    assert_eq!(
        pipeline_layout_key(&[first, second]),
        pipeline_layout_key(&[second, first])
    );
}

#[test]
fn graphics_pipeline_keys_include_state_attachment_and_texture_interface() {
    let state = DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap();
    let key = PipelineKey::Graphics {
        backend: Backend::Vulkan,
        shader: 7,
        shader_digest: [1; 32],
        vertex_product: 0,
        vertex_entry: "vertexmain".to_owned(),
        fragment_product: 1,
        fragment_entry: "fragmentmain".to_owned(),
        texture_heap: None,
        layouts: Vec::new(),
        state,
        depth_required: false,
        color_format: 44,
        depth_format: 0,
        sample_count: 1,
    };
    let mut changed_state = key.clone();
    let PipelineKey::Graphics { state, .. } = &mut changed_state else {
        unreachable!()
    };
    state.blend = ez_gfx_hal::BlendMode::Alpha;
    let mut changed_format = key.clone();
    let PipelineKey::Graphics { color_format, .. } = &mut changed_format else {
        unreachable!()
    };
    *color_format = 50;
    let mut changed_heap = key.clone();
    let PipelineKey::Graphics { texture_heap, .. } = &mut changed_heap else {
        unreachable!()
    };
    *texture_heap = Some(ez_gfx_hal::ShaderTextureHeapLayout::new(0, 4, 16, 2, 0, 1).unwrap());

    assert_eq!(
        HashSet::from([key, changed_state, changed_format, changed_heap]).len(),
        4
    );
}
