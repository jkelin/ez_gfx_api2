//! Native Metal rendering and presentation tests through the C ABI.
#![cfg(target_vendor = "apple")]

use std::{ffi::CString, path::Path};

use ez_gfx_artifact::{Artifact, Provenance, Stage, Target, TargetVariant};
use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxDrawIndexedCommand, EzGfxResult, EzGfxSurfaceDesc,
    EzGfxTextureDesc, ez_gfx_acquire_indirect, ez_gfx_begin_render, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_init_device, ez_gfx_context_wait_idle,
    ez_gfx_finish_render, ez_gfx_frame_begin, ez_gfx_frame_readback, ez_gfx_frame_submit,
    ez_gfx_graph_enqueue_texture_readback, ez_gfx_index_heap_create, ez_gfx_index_heap_destroy,
    ez_gfx_indirect_release, ez_gfx_indirect_set_draw_count, ez_gfx_indirect_write_draw,
    ez_gfx_render_add_compute_pipeline, ez_gfx_render_add_vertex_pipeline, ez_gfx_shader_destroy,
    ez_gfx_shader_load_artifact, ez_gfx_surface_create, ez_gfx_surface_destroy,
    ez_gfx_texture_load, ez_gfx_texture_unload, ez_gfx_vertex_upload_indices,
};
use objc2::rc::Retained;
use objc2_core_foundation::CGSize;
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::CAMetalLayer;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

#[test]
fn metal_separated_passes_preserve_color_and_depth() {
    let root = std::env::temp_dir().join(format!("ez-gfx-metal-present-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let artifact = graphics_artifact(&root);

    let cached = render(&artifact, true);
    assert_eq!(cached.len(), WIDTH as usize * HEIGHT as usize * 4);
    assert_eq!(pixel(&cached, 2, 2), [26, 26, 26, 255]);
    assert_eq!(pixel(&cached, WIDTH / 4, HEIGHT / 2), [255, 0, 0, 255]);
    assert_eq!(pixel(&cached, WIDTH * 3 / 4, HEIGHT / 2), [0, 255, 0, 255]);

    assert!(render(&artifact, false).is_empty());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn metal_texture_readback_submits_without_a_surface() {
    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 2,
        backend: 3,
    };
    let texture_desc = EzGfxTextureDesc {
        source_format: 1,
        destination_format: 0,
        width: 2,
        height: 1,
        mip_count: 1,
        generate_mips: 0,
        min_filter: 0,
        mag_filter: 0,
        max_anisotropy: 1.0,
        address_mode_u: 0,
        address_mode_v: 0,
        address_mode_w: 0,
        debug_label: core::ptr::null(),
    };
    let expected = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let mut context = 0;
    let mut texture = 0;

    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&context_desc, &mut context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_texture_load(
                    expected.as_ptr(),
                    expected.len(),
                    &texture_desc,
                    &mut texture,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_begin(context), EzGfxResult::Ok);
    assert_eq!(
        ez_gfx_graph_enqueue_texture_readback(texture, context),
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_submit(context), EzGfxResult::Ok);

    let mut size = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_frame_readback(core::ptr::null_mut(), 0, &mut size, context) }
        },
        EzGfxResult::Ok
    );
    let mut actual = vec![0; size];
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_frame_readback(actual.as_mut_ptr(), actual.len(), &mut size, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(actual, expected);

    ez_gfx_texture_unload(texture, context);
    ez_gfx_context_destroy(context);
}

#[test]
fn metal_compute_submits_without_a_surface() {
    let root = std::env::temp_dir().join(format!("ez-gfx-metal-compute-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let artifact = graphics_artifact(&root);
    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 2,
        backend: 3,
    };
    let mut context = 0;
    let mut shader = 0;

    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&context_desc, &mut context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_shader_load_artifact(artifact.as_ptr(), artifact.len(), &mut shader, context)
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_begin(context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_compute_pipeline(
                    shader,
                    1,
                    1,
                    1,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    0,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_submit(context), EzGfxResult::Ok);

    ez_gfx_shader_destroy(shader, context);
    ez_gfx_context_destroy(context);
    let _ = std::fs::remove_dir_all(root);
}

// The retained layer outlives surface destruction; disabled caching must leave readback empty.
fn render(artifact: &[u8], cache_presented_snapshots: bool) -> Vec<u8> {
    let layer = CAMetalLayer::new();
    layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    layer.setDrawableSize(CGSize {
        width: f64::from(WIDTH),
        height: f64::from(HEIGHT),
    });

    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 2,
        backend: 3,
    };
    let surface_desc = EzGfxSurfaceDesc {
        window: Retained::as_ptr(&layer).cast_mut().cast(),
        display: core::ptr::null_mut(),
        platform: 2,
        width: WIDTH,
        height: HEIGHT,
        cache_presented_snapshots: u8::from(cache_presented_snapshots),
    };
    let mut context = 0;
    let mut surface = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&context_desc, &mut context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_surface_create(&surface_desc, &mut surface, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_context_init_device(surface, context),
        EzGfxResult::Ok
    );

    let mut shader = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_shader_load_artifact(artifact.as_ptr(), artifact.len(), &mut shader, context)
            }
        },
        EzGfxResult::Ok
    );

    let label = CString::new("metal-present").unwrap();
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_index_heap_create(3 * size_of::<u32>() as u64, label.as_ptr(), context)
            }
        },
        EzGfxResult::Ok
    );
    let indices = [0_u32, 1, 2];
    let mut first_index = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_vertex_upload_indices(
                    indices.as_ptr().cast(),
                    indices.len() as u32,
                    &mut first_index,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let mut indirect = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_acquire_indirect(1, label.as_ptr(), &mut indirect, context) }
        },
        EzGfxResult::Ok
    );
    let command = EzGfxDrawIndexedCommand {
        index_count: 3,
        instance_count: 1,
        first_index,
        vertex_offset: 0,
        first_instance: 0,
    };
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_indirect_write_draw(indirect, 0, &command, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_indirect_set_draw_count(indirect, 1, context),
        EzGfxResult::Ok
    );

    let left = [-0.45_f32, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 1.0];
    let right = [0.45_f32, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 1.0];
    let occluded = [-0.45_f32, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0];
    assert_eq!(ez_gfx_begin_render(surface, context), EzGfxResult::Ok);
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_vertex_pipeline(
                    shader,
                    indirect,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    left.as_ptr().cast(),
                    core::mem::size_of_val(&left) as u32,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    // This node separates the graphics passes. The second pass must load both attachments.
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_compute_pipeline(
                    shader,
                    1,
                    1,
                    1,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    0,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_vertex_pipeline(
                    shader,
                    indirect,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    right.as_ptr().cast(),
                    core::mem::size_of_val(&right) as u32,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    // A farther blue draw overlaps the first pass. Preserved depth keeps the first red pixel.
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_vertex_pipeline(
                    shader,
                    indirect,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    occluded.as_ptr().cast(),
                    core::mem::size_of_val(&occluded) as u32,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_finish_render(context), EzGfxResult::Ok);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);

    let mut size = 0;
    let status = {
        // SAFETY: Non-null output pointers reference writable storage of the declared capacity and alignment for this call.
        unsafe { ez_gfx_frame_readback(core::ptr::null_mut(), 0, &mut size, context) }
    };
    let bytes = if cache_presented_snapshots {
        assert_eq!(status, EzGfxResult::Ok);
        assert_eq!(size, WIDTH as usize * HEIGHT as usize * 4);
        let mut bytes = vec![0; size];
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe {
                    ez_gfx_frame_readback(bytes.as_mut_ptr(), bytes.len(), &mut size, context)
                }
            },
            EzGfxResult::Ok
        );
        bytes
    } else {
        assert_eq!(status, EzGfxResult::NotReady);
        assert_eq!(size, 0);
        Vec::new()
    };

    ez_gfx_indirect_release(indirect, context);
    ez_gfx_index_heap_destroy(context);
    ez_gfx_shader_destroy(shader, context);
    ez_gfx_surface_destroy(surface, context);
    ez_gfx_context_destroy(context);
    drop(layer);
    bytes
}

// The test requires Xcode's offline Metal compiler; runtime code still loads metallib bytes only.
fn graphics_artifact(root: &Path) -> Vec<u8> {
    let source = root.join("present.metal");
    let library = root.join("present.metallib");
    std::fs::write(
        &source,
        r#"#include <metal_stdlib>
using namespace metal;

struct VertexOut { float4 position [[position]]; };
struct Params { float2 offset; float depth; float padding; float4 color; };

vertex VertexOut vertexmain(
    uint vertex_id [[vertex_id]],
    constant Params& params [[buffer(0)]]
) {
    constexpr float2 positions[] = {
        float2(-0.35, -0.6),
        float2(0.35, -0.6),
        float2(0.0, 0.6),
    };
    VertexOut output;
    output.position = float4(positions[vertex_id] + params.offset, params.depth, 1.0);
    return output;
}

fragment float4 fragmentmain(constant Params& params [[buffer(0)]]) {
    return params.color;
}

kernel void computemain(uint thread_id [[thread_position_in_grid]]) {
    (void)thread_id;
}
"#,
    )
    .unwrap();
    ez_gfx_compiler::build_metallib(source, library.clone()).unwrap();
    let metallib = std::fs::read(library).unwrap();
    let metadata = br#"{"reflections":[{"target":"Metallib","entry":"vertexmain","stage":"Vertex","reflection":{"parameters":[]}},{"target":"Metallib","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[],"depth_required":true}},{"target":"Metallib","entry":"computemain","stage":"Compute","reflection":{"parameters":[]}}]}"#.to_vec();
    let mut variants = Vec::new();
    for (stage, entry) in [
        (Stage::Vertex, "vertexmain"),
        (Stage::Fragment, "fragmentmain"),
        (Stage::Compute, "computemain"),
    ] {
        variants
            .push(TargetVariant::new(Target::Spirv, stage, entry, "ez-gfx-v1", vec![1]).unwrap());
        variants
            .push(TargetVariant::new(Target::Dxil, stage, entry, "ez-gfx-v1", vec![1]).unwrap());
        variants.push(
            TargetVariant::new(
                Target::Metallib,
                stage,
                entry,
                "ez-gfx-v1",
                metallib.clone(),
            )
            .unwrap(),
        );
    }
    Artifact::new(
        metadata,
        Provenance::new("xcrun metal", "test", Vec::new(), "macosx"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap()
}

// Test coordinates are fixed in bounds of the complete packed frame.
fn pixel(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * WIDTH + x) * 4) as usize;
    bytes[offset..offset + 4].try_into().unwrap()
}
