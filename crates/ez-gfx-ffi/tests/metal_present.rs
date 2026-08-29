#![cfg(target_vendor = "apple")]

use std::{ffi::CString, path::Path};

use ez_gfx_artifact::{Artifact, Provenance, Stage, Target, TargetVariant};
use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxDrawIndexedCommand, EzGfxResult, EzGfxShaderEntry,
    EzGfxSurfaceDesc, ez_gfx_acquire_indirect, ez_gfx_begin_render, ez_gfx_context_create_backend,
    ez_gfx_context_destroy, ez_gfx_context_init_device, ez_gfx_context_wait_idle,
    ez_gfx_finish_render, ez_gfx_frame_readback, ez_gfx_index_heap_create,
    ez_gfx_index_heap_destroy, ez_gfx_indirect_release, ez_gfx_indirect_set_draw_count,
    ez_gfx_indirect_write_draw, ez_gfx_render_add_vertex_pipeline, ez_gfx_shader_destroy,
    ez_gfx_shader_load_artifact, ez_gfx_surface_create, ez_gfx_surface_destroy,
    ez_gfx_vertex_upload_indices,
};
use objc2::rc::Retained;
use objc2_core_foundation::CGSize;
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::CAMetalLayer;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

#[test]
fn metal_frame_presents_and_reads_back_rgba8() {
    let root = std::env::temp_dir().join(format!("ez-gfx-metal-present-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let artifact = graphics_artifact(&root);

    let cached = render(&artifact, true);
    assert_eq!(cached.len(), WIDTH as usize * HEIGHT as usize * 4);
    assert_eq!(pixel(&cached, 2, 2), [0, 0, 0, 255]);
    assert_eq!(pixel(&cached, WIDTH / 2, HEIGHT / 2), [255, 0, 0, 255]);

    assert!(render(&artifact, false).is_empty());
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
        ez_gfx_context_create_backend(&context_desc, &mut context),
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_surface_create(&surface_desc, &mut surface, context),
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_context_init_device(surface, context),
        EzGfxResult::Ok
    );

    let vertex = CString::new("vertexmain").unwrap();
    let fragment = CString::new("fragmentmain").unwrap();
    let entries = [
        EzGfxShaderEntry {
            entry: vertex.as_ptr(),
            stage: 1,
            _padding: [0; 7],
        },
        EzGfxShaderEntry {
            entry: fragment.as_ptr(),
            stage: 2,
            _padding: [0; 7],
        },
    ];
    let mut shader = 0;
    assert_eq!(
        ez_gfx_shader_load_artifact(
            artifact.as_ptr(),
            artifact.len(),
            entries.as_ptr(),
            entries.len(),
            &mut shader,
            context,
        ),
        EzGfxResult::Ok
    );

    let label = CString::new("metal-present").unwrap();
    assert_eq!(
        ez_gfx_index_heap_create(3 * size_of::<u32>() as u64, label.as_ptr(), context),
        EzGfxResult::Ok
    );
    let indices = [0_u32, 1, 2];
    let mut first_index = 0;
    assert_eq!(
        ez_gfx_vertex_upload_indices(
            indices.as_ptr().cast(),
            indices.len() as u32,
            &mut first_index,
            context,
        ),
        EzGfxResult::Ok
    );
    let mut indirect = 0;
    assert_eq!(
        ez_gfx_acquire_indirect(1, label.as_ptr(), &mut indirect, context),
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
        ez_gfx_indirect_write_draw(indirect, 0, &command, context),
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_indirect_set_draw_count(indirect, 1, context),
        EzGfxResult::Ok
    );

    assert_eq!(ez_gfx_begin_render(surface, context), EzGfxResult::Ok);
    assert_eq!(
        ez_gfx_render_add_vertex_pipeline(
            shader,
            indirect,
            core::ptr::null(),
            0,
            core::ptr::null(),
            core::ptr::null(),
            0,
            context,
        ),
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_finish_render(context), EzGfxResult::Ok);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);

    let mut size = 0;
    let status = ez_gfx_frame_readback(core::ptr::null_mut(), 0, &mut size, context);
    let bytes = if cache_presented_snapshots {
        assert_eq!(status, EzGfxResult::Ok);
        assert_eq!(size, WIDTH as usize * HEIGHT as usize * 4);
        let mut bytes = vec![0; size];
        assert_eq!(
            ez_gfx_frame_readback(bytes.as_mut_ptr(), bytes.len(), &mut size, context),
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

vertex VertexOut vertexmain(uint vertex_id [[vertex_id]]) {
    constexpr float2 positions[] = {
        float2(-0.8, -0.8),
        float2(0.8, -0.8),
        float2(0.0, 0.8),
    };
    VertexOut output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    return output;
}

fragment float4 fragmentmain() { return float4(1.0, 0.0, 0.0, 1.0); }
"#,
    )
    .unwrap();
    ez_gfx_compiler::build_metallib(source, library.clone()).unwrap();
    let metallib = std::fs::read(library).unwrap();
    let metadata = br#"{"reflections":[{"target":"Metallib","entry":"vertexmain","stage":"Vertex","reflection":{"parameters":[]}},{"target":"Metallib","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[]}}]}"#.to_vec();
    let mut variants = Vec::new();
    for (stage, entry) in [
        (Stage::Vertex, "vertexmain"),
        (Stage::Fragment, "fragmentmain"),
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
