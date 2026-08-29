[__AttributeUsage(_AttributeTargets.Var)] struct StructuredBufferAttribute { string name; }
[__AttributeUsage(_AttributeTargets.Var)] struct ColorTargetAttribute { string name; string access; }
[__AttributeUsage(_AttributeTargets.Var)] struct BindlessTextureHeapAttribute { int capacity; }
struct Push { row_major float4x4 mvp; uint texture_id; uint padding0; uint padding1; uint padding2; };
[[vk::push_constant]] ConstantBuffer<Push> push;
[StructuredBuffer("positions")] [[vk::binding(0, 0)]] StructuredBuffer<float4> positions;
static const uint EZ_GFX_MAX_TEXTURES = 1024;
struct TextureEntry { Texture2D<float4> texture; SamplerState sampler; };
struct TextureHeap { TextureEntry entries[EZ_GFX_MAX_TEXTURES]; };
[BindlessTextureHeap(EZ_GFX_MAX_TEXTURES)] [[vk::binding(0, 1)]] ParameterBlock<TextureHeap> texture_heap;
struct V2P { float4 position : SV_Position; [vk_location(0)] float2 uv : TEXCOORD0; };
[shader("vertex")] V2P vertexmain(uint vertex_index : SV_VertexID) {
    V2P output;
    output.position = mul(push.mvp, float4(positions[vertex_index].xyz, 1.0));
    uint corner = vertex_index & 3;
    output.uv = float2((corner == 1 || corner == 2) ? 1.0 : 0.0, (corner == 0 || corner == 1) ? 1.0 : 0.0);
    return output;
}
struct Output { [vk_location(0)] [ColorTarget("swapchain", "write")] float4 color : SV_Target0; };
[shader("fragment")] Output fragmentmain(V2P input) {
    TextureEntry entry = texture_heap.entries[push.texture_id]; Output output; output.color = entry.texture.Sample(entry.sampler, input.uv); return output;
}
