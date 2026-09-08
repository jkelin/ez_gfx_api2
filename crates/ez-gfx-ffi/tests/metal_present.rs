//! Native Metal rendering and presentation tests through the C ABI.
#![cfg(target_vendor = "apple")]

use std::path::Path;

use ez_gfx_artifact::{
    AppleArchitecture, ApplePlatform, Artifact, CompatibilityVersion, MetalCompatibility,
    Provenance, Stage, Target, TargetCompatibility, TargetVariant,
};
use ez_gfx_ffi::{
    EzGfxBackendContextDesc, EzGfxDrawIndexedCommand, EzGfxDynamicState, EzGfxEvent,
    EzGfxEventKind, EzGfxRenderTargetDesc, EzGfxResult, EzGfxSurfaceDesc, EzGfxTextureDesc,
    ez_gfx_callback_register, ez_gfx_context_create_backend, ez_gfx_context_destroy,
    ez_gfx_context_init_device, ez_gfx_context_wait_idle, ez_gfx_counted_buffer_acquire,
    ez_gfx_counted_buffer_write_draws, ez_gfx_frame_begin, ez_gfx_frame_end,
    ez_gfx_graph_enqueue_texture_readback, ez_gfx_index_allocation_get_range,
    ez_gfx_index_heap_create, ez_gfx_index_heap_destroy, ez_gfx_render_add_compute_pipeline,
    ez_gfx_render_add_vertex_pipeline, ez_gfx_render_target_create, ez_gfx_render_target_destroy,
    ez_gfx_render_target_frame_begin, ez_gfx_shader_destroy, ez_gfx_shader_load_artifact,
    ez_gfx_surface_create, ez_gfx_surface_destroy, ez_gfx_texture_load, ez_gfx_texture_unload,
    ez_gfx_vertex_upload_indices,
};
use objc2::rc::Retained;
use objc2_core_foundation::CGSize;
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::CAMetalLayer;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
#[derive(Default)]
struct Collected {
    readback: Option<Vec<u8>>,
}

/// Copies callback-scoped readback bytes into test-owned storage.
///
/// # Safety
///
/// `user_data` must point to a live `Collected` while registered.
unsafe extern "C" fn collect_event(event: *const EzGfxEvent, user_data: *mut core::ffi::c_void) {
    // SAFETY: registration keeps both pointers valid for this callback invocation.
    let (event, collected) = unsafe { (&*event, &mut *user_data.cast::<Collected>()) };
    if event.kind == EzGfxEventKind::Readback {
        let bytes = if event.readback_byte_count == 0 {
            Vec::new()
        } else {
            // SAFETY: nonempty readback bytes are callback-scoped and copied before returning.
            unsafe {
                core::slice::from_raw_parts(event.readback_bytes, event.readback_byte_count)
                    .to_vec()
            }
        };
        collected.readback = Some(bytes);
    }
}

fn begin_offscreen_frame(context: u64) -> (u64, u64) {
    let name = b"metal-target";
    let format = 1_u8;
    let desc = EzGfxRenderTargetDesc {
        name: name.as_ptr(),
        name_length: name.len(),
        usage: 0,
        relative_scale: 1.0,
        samples: 1,
        candidate_formats: &raw const format,
        candidate_count: 1,
        sampleable: 0,
        use_clear: 0,
        clear_color: [0.0; 4],
    };
    let mut target = 0;
    // SAFETY: descriptor, format, and output storage remain live through the call.
    assert_eq!(
        unsafe { ez_gfx_render_target_create(&raw const desc, 1, 1, &raw mut target, context) },
        EzGfxResult::Ok
    );
    let mut frame = 0;
    // SAFETY: frame output storage is live and aligned.
    assert_eq!(
        unsafe { ez_gfx_render_target_frame_begin(context, target, &raw mut frame) },
        EzGfxResult::Ok
    );
    (frame, target)
}

#[test]
fn metal_separated_passes_preserve_color_and_depth() {
    let root = std::env::temp_dir().join(format!("ez-gfx-metal-present-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let artifact = graphics_artifact(&root);

    let cached = render(&artifact, true);
    assert_eq!(cached.len(), WIDTH as usize * HEIGHT as usize * 4);
    // The surface contract encodes a linear 0.1 clear as sRGB, without changing alpha.
    assert_eq!(pixel(&cached, 2, 2), [89, 89, 89, 255]);
    let left = pixel(&cached, WIDTH / 4, HEIGHT / 2);
    assert!(left[0] > left[1]);
    assert_eq!(left[3], 255);
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
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
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
        debug_label_length: 0,
    };
    let expected = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let mut context = 0;
    let mut texture = 0;

    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const context_desc, &raw mut context) }
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
                    &raw const texture_desc,
                    &raw mut texture,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    // Admission is asynchronous; readback requires the decoded native texture to be ready.
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    let mut collected = Collected::default();
    assert_eq!(
        // SAFETY: `collected` remains alive until registration is explicitly cleared.
        unsafe {
            ez_gfx_callback_register(context, Some(collect_event), (&raw mut collected).cast())
        },
        EzGfxResult::Ok
    );
    let (frame, target) = begin_offscreen_frame(context);
    assert_eq!(
        ez_gfx_graph_enqueue_texture_readback(texture, frame),
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_end(frame), EzGfxResult::Ok);
    ez_gfx_render_target_destroy(target, context);

    let actual = collected.readback.take().expect("readback event delivered");
    assert_eq!(actual, expected);

    ez_gfx_texture_unload(texture, context);
    assert_eq!(
        // SAFETY: clearing a live registration retains no user-data pointer.
        unsafe { ez_gfx_callback_register(context, None, core::ptr::null_mut()) },
        EzGfxResult::Ok
    );
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
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
    };
    let mut context = 0;
    let mut shader = 0;

    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_context_create_backend(&raw const context_desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_shader_load_artifact(
                    artifact.as_ptr(),
                    artifact.len(),
                    &raw mut shader,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let (frame, target) = begin_offscreen_frame(context);
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
                    frame,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_end(frame), EzGfxResult::Ok);
    ez_gfx_render_target_destroy(target, context);

    ez_gfx_shader_destroy(shader, context);
    ez_gfx_context_destroy(context);
    let _ = std::fs::remove_dir_all(root);
}

fn submit_render_nodes(context: u64, frame: u64, shader: u64, indirect: u64) {
    let left = [-0.45_f32, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.5];
    let right = [0.45_f32, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 1.0];
    let occluded = [-0.45_f32, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0];
    let alpha_blend = EzGfxDynamicState {
        cull_mode: 0,
        front_face: 0,
        primitive_type: 0,
        blend_mode: 1,
    };
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_render_add_vertex_pipeline(
                    shader,
                    indirect,
                    core::ptr::null(),
                    0,
                    &raw const alpha_blend,
                    left.as_ptr().cast(),
                    u32::try_from(core::mem::size_of_val(&left)).unwrap(),
                    frame,
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
                    frame,
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
                    u32::try_from(core::mem::size_of_val(&right)).unwrap(),
                    frame,
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
                    u32::try_from(core::mem::size_of_val(&occluded)).unwrap(),
                    frame,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(ez_gfx_frame_end(frame), EzGfxResult::Ok);
    assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
}

// The retained layer outlives surface destruction; disabled caching must leave readback empty.
fn render(artifact: &[u8], cache_presented_snapshots: bool) -> Vec<u8> {
    let layer = CAMetalLayer::new();
    layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
    layer.setDrawableSize(CGSize {
        width: f64::from(WIDTH),
        height: f64::from(HEIGHT),
    });

    let context_desc = EzGfxBackendContextDesc {
        enable_debug: 0,
        enable_validation: 0,
        surface_platform: 2,
        backend: 3,
        texture_decode_workers: 0,
        adapter_count: 0,
        adapter: core::ptr::null(),
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
            unsafe { ez_gfx_context_create_backend(&raw const context_desc, &raw mut context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe { ez_gfx_surface_create(&raw const surface_desc, &raw mut surface, context) }
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        ez_gfx_context_init_device(surface, context),
        EzGfxResult::Ok
    );
    let mut collected = Collected::default();
    assert_eq!(
        // SAFETY: `collected` remains alive until registration is explicitly cleared.
        unsafe {
            ez_gfx_callback_register(context, Some(collect_event), (&raw mut collected).cast())
        },
        EzGfxResult::Ok
    );

    let mut shader = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_shader_load_artifact(
                    artifact.as_ptr(),
                    artifact.len(),
                    &raw mut shader,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );

    let label = b"metal-present";
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_index_heap_create(
                    3 * size_of::<u32>() as u64,
                    label.as_ptr(),
                    label.len(),
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let indices = [0_u32, 1, 2];
    let mut index_allocation = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_vertex_upload_indices(
                    indices.as_ptr().cast(),
                    u32::try_from(indices.len()).unwrap(),
                    &raw mut index_allocation,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    let mut first_index = 0;
    let mut index_count = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_index_allocation_get_range(
                    index_allocation,
                    &raw mut first_index,
                    &raw mut index_count,
                    context,
                )
            }
        },
        EzGfxResult::Ok
    );
    assert_eq!(index_count, 3);
    let mut frame = 0;
    assert_eq!(
        // SAFETY: frame output storage is live and aligned.
        unsafe { ez_gfx_frame_begin(context, surface, &raw mut frame) },
        EzGfxResult::Ok
    );
    let mut indirect = 0;
    assert_eq!(
        {
            // SAFETY: Non-null arguments use live test-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
            unsafe {
                ez_gfx_counted_buffer_acquire(
                    1,
                    label.as_ptr(),
                    label.len(),
                    &raw mut indirect,
                    frame,
                )
            }
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
            unsafe { ez_gfx_counted_buffer_write_draws(indirect, 0, &raw const command, 1, frame) }
        },
        EzGfxResult::Ok
    );

    submit_render_nodes(context, frame, shader, indirect);

    let bytes = if cache_presented_snapshots {
        let bytes = collected
            .readback
            .take()
            .expect("presented readback event delivered");
        assert_eq!(bytes.len(), WIDTH as usize * HEIGHT as usize * 4);
        bytes
    } else {
        assert!(collected.readback.is_none());
        Vec::new()
    };
    assert_eq!(
        // SAFETY: clearing a live registration retains no user-data pointer.
        unsafe { ez_gfx_callback_register(context, None, core::ptr::null_mut()) },
        EzGfxResult::Ok
    );

    ez_gfx_index_heap_destroy(context);
    ez_gfx_shader_destroy(shader, context);
    ez_gfx_surface_destroy(surface, context);
    ez_gfx_context_destroy(context);
    drop(layer);
    bytes
}

// The test requires Xcode's offline Metal compiler; runtime code still loads metallib bytes only.
fn graphics_artifact(root: &Path) -> Vec<u8> {
    // Portable products use canonical compatibility; the exercised Metal product is a metallib,
    // not source MSL, so it carries the concrete host architecture and offline library contract.
    let compatibility = |target| match target {
        Target::Metallib => TargetCompatibility::MetalLibrary {
            metal: MetalCompatibility {
                platform: ApplePlatform::MacOs,
                architecture: match std::env::consts::ARCH {
                    "aarch64" => AppleArchitecture::Aarch64,
                    "x86_64" => AppleArchitecture::X86_64,
                    _ => panic!("unsupported Apple architecture"),
                },
                minimum_os: CompatibilityVersion::new(14, 0),
                sdk: CompatibilityVersion::new(15, 0),
                language: CompatibilityVersion::new(3, 0),
                library: CompatibilityVersion::new(1, 0),
                toolchain: "apple-clang-16".into(),
            },
        },
        _ => TargetCompatibility::portable(target).unwrap(),
    };
    let source = root.join("present.metal");
    let library = root.join("present.metallib");
    std::fs::write(
        &source,
        r"#include <metal_stdlib>
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
",
    )
    .unwrap();
    ez_gfx_compiler::build_metallib(source, library.clone()).unwrap();
    let metallib = std::fs::read(library).unwrap();
    let metadata = br#"{"reflections":[{"target":"Metallib","entry":"vertexmain","stage":"Vertex","reflection":{"parameters":[]}},{"target":"Metallib","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[],"depth_required":true}},{"target":"Metallib","entry":"computemain","stage":"Compute","reflection":{"parameters":[],"workgroup_size":[8,2,1]}}]}"#.to_vec();
    let mut variants = Vec::new();
    for (stage, entry) in [
        (Stage::Vertex, "vertexmain"),
        (Stage::Fragment, "fragmentmain"),
        (Stage::Compute, "computemain"),
    ] {
        variants.push(
            TargetVariant::new(
                Target::Spirv,
                stage,
                entry,
                "ez-gfx-v1",
                compatibility(Target::Spirv),
                vec![1],
            )
            .unwrap(),
        );
        variants.push(
            TargetVariant::new(
                Target::Dxil,
                stage,
                entry,
                "ez-gfx-v1",
                compatibility(Target::Dxil),
                vec![1],
            )
            .unwrap(),
        );
        variants.push(
            TargetVariant::new(
                Target::Metallib,
                stage,
                entry,
                "ez-gfx-v1",
                compatibility(Target::Metallib),
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
