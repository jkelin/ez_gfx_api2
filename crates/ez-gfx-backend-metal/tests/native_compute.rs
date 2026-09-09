//! GPU-backed Metal compute behavior tests.
#![cfg(target_vendor = "apple")]

use ez_gfx_backend_metal::native::{
    NativeBufferBinding, NativeComputeDispatch, NativeContext, NativeFrameAction,
};
use ez_gfx_compiler::{Target, compile_shader};
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_hal::{
    AllocationRequest, ImageMip, MemoryAllocator, MemoryClass, SamplerAddressMode, SamplerFilter,
    ShaderTextureHeapLayout, TextureFormat, TextureSamplerDesc,
};
use ez_gfx_runtime::shader::RuntimeShader;

#[test]
fn reflected_workgroup_size_controls_metal_invocation_count() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("workgroup.slang");
    std::fs::write(
        &source,
        r#"[__AttributeUsage(_AttributeTargets.Var)]
struct BufferAttribute { string name; };

[Buffer("values")]
RWStructuredBuffer<uint> values;

[shader("compute")]
[numthreads(8, 2, 1)]
void computemain(uint3 id : SV_DispatchThreadID) {
    values[id.y * 8 + id.x] = id.y * 8 + id.x + 1;
}
"#,
    )
    .unwrap();

    let artifact = compile_shader(&source, &[Target::Metal], false).unwrap();
    let runtime = RuntimeShader::load(&artifact, Backend::Metal, SemanticProfile::V1).unwrap();
    let (product_index, _, entry) = runtime.compute_product().unwrap();
    let threads_per_group = runtime.compute_workgroup_size().unwrap();
    assert_eq!(threads_per_group, [8, 2, 1]);

    let products = runtime
        .products()
        .map(|(_, bytes)| bytes)
        .collect::<Vec<_>>();
    let bindings = runtime.bindings(ez_gfx_artifact::Stage::Compute).unwrap();
    let binding_index = bindings.requirements()[0].binding as usize;

    let mut context = NativeContext::create_default().unwrap();
    let shader = context.create_shader(&products).unwrap();
    let pipeline = context
        .create_compute_pipeline(&shader, product_index, entry, None)
        .unwrap();
    let mut output = context
        .allocate(AllocationRequest::new(16 * 4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    context.mapped_slice_mut(&mut output).unwrap().fill(0);
    context.flush(&mut output, 0, 16 * 4).unwrap();

    let native_binding = NativeBufferBinding {
        allocation: &output,
        offset: 0,
        index: binding_index,
    };
    let action = NativeFrameAction::Compute(NativeComputeDispatch {
        pipeline: &pipeline,
        groups: [1, 1, 1],
        threads_per_group,
        bindings: std::slice::from_ref(&native_binding),
        texture_heap: None,
        textures: &[],
    });
    context.execute_frame(None, &[action], false).unwrap();
    context.wait_idle().unwrap();
    context.invalidate(&mut output, 0, 16 * 4).unwrap();

    let actual = context
        .mapped_slice(&output)
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, (1..=16).collect::<Vec<_>>());

    context.destroy_pipeline(pipeline);
    context.destroy_shader(shader);
    context.free(output).unwrap();
    context.wait_idle().unwrap();
}

#[test]
fn compute_stage_samples_the_metal_texture_heap() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("compute_texture.slang");
    std::fs::write(
        &source,
        r#"[__AttributeUsage(_AttributeTargets.Var)]
struct BufferAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)]
struct BindlessTextureHeapAttribute { int capacity; };
struct TextureEntry { Texture2D<float4> texture; SamplerState sampler; };
struct TextureHeap { TextureEntry entries[1024]; };

[Buffer("values")]
RWStructuredBuffer<uint> values;
[BindlessTextureHeap(1024)]
ParameterBlock<TextureHeap> texture_heap;

[shader("compute")]
[numthreads(1, 1, 1)]
void computemain(uint3 id : SV_DispatchThreadID) {
    values[0] = texture_heap.entries[0].texture.SampleLevel(
        texture_heap.entries[0].sampler, float2(0.5, 0.5), 0).r > 0.9 ? 77 : 11;
}
"#,
    )
    .unwrap();

    let artifact = compile_shader(&source, &[Target::Metal], false).unwrap();
    let runtime = RuntimeShader::load(&artifact, Backend::Metal, SemanticProfile::V1).unwrap();
    let (product_index, _, entry) = runtime.compute_product().unwrap();
    let pipeline_layout = runtime
        .pipeline_layout(ez_gfx_artifact::Stage::Compute)
        .unwrap();
    let reflected_heap = pipeline_layout.texture_heap().unwrap();
    let texture_heap = ShaderTextureHeapLayout::new(
        reflected_heap.space,
        reflected_heap.binding,
        reflected_heap.capacity,
        reflected_heap.argument_stride,
        reflected_heap.texture_argument_offset,
        reflected_heap.sampler_argument_offset,
    )
    .unwrap();
    let products = runtime
        .products()
        .map(|(_, bytes)| bytes)
        .collect::<Vec<_>>();
    let bindings = runtime.bindings(ez_gfx_artifact::Stage::Compute).unwrap();
    let binding_index = bindings.requirements()[0].binding as usize;

    let mut context = NativeContext::create_default().unwrap();
    let shader = context.create_shader(&products).unwrap();
    let pipeline = context
        .create_compute_pipeline(&shader, product_index, entry, Some(texture_heap))
        .unwrap();
    let pixels = [255_u8, 0, 0, 255];
    let mip = ImageMip {
        width: 1,
        height: 1,
        bytes: &pixels,
    };
    let sampler = TextureSamplerDesc {
        min_filter: SamplerFilter::Nearest,
        mag_filter: SamplerFilter::Nearest,
        max_anisotropy: 1.0,
        address_u: SamplerAddressMode::Clamp,
        address_v: SamplerAddressMode::Clamp,
        address_w: SamplerAddressMode::Clamp,
    };
    let (mut texture, completions) = context
        .create_texture(TextureFormat::Rgba8Unorm, &[mip], 0, sampler)
        .unwrap();
    let completion = completions.last().unwrap().value;
    while context.completed_texture_transfer_value().unwrap() < completion {
        std::thread::yield_now();
    }
    context.publish_texture_mips(&mut texture, 1).unwrap();
    let mut output = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    context.mapped_slice_mut(&mut output).unwrap().fill(0);
    let native_binding = NativeBufferBinding {
        allocation: &output,
        offset: 0,
        index: binding_index,
    };
    let textures = [&texture];
    let action = NativeFrameAction::Compute(NativeComputeDispatch {
        pipeline: &pipeline,
        groups: [1, 1, 1],
        threads_per_group: runtime.compute_workgroup_size().unwrap(),
        bindings: std::slice::from_ref(&native_binding),
        texture_heap: Some(texture_heap),
        textures: &textures,
    });
    context.execute_frame(None, &[action], false).unwrap();
    context.wait_idle().unwrap();
    context.invalidate(&mut output, 0, 4).unwrap();
    assert_eq!(
        u32::from_ne_bytes(
            context.mapped_slice(&output).unwrap()[..4]
                .try_into()
                .unwrap()
        ),
        77
    );

    context.destroy_texture(texture).unwrap();
    context.destroy_pipeline(pipeline);
    context.destroy_shader(shader);
    context.free(output).unwrap();
    context.wait_idle().unwrap();
}
