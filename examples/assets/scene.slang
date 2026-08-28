[__AttributeUsage(_AttributeTargets.Var)] struct StructuredBufferAttribute { string name; }
[__AttributeUsage(_AttributeTargets.Var)] struct ColorTargetAttribute { string name; string access; }
struct Push { float4 tint; uint use_texture; uint texture_id; uint2 padding; };
[[vk::push_constant]] ConstantBuffer<Push> push;
[StructuredBuffer("positions")] [[vk::binding(0, 0)]] StructuredBuffer<float4> positions;
[StructuredBuffer("colors")] [[vk::binding(1, 0)]] RWStructuredBuffer<float4> colors;
static const uint EZ_GFX_MAX_TEXTURES = 1024;
[[vk::binding(0, 1)]] Sampler2D textures[EZ_GFX_MAX_TEXTURES];
struct V2P { float4 pos : SV_Position; [vk_location(0)] float4 color : COLOR0; [vk_location(1)] float2 uv : TEXCOORD0; };
[shader("vertex")] V2P vertexmain(uint vertex_index : SV_VertexID) {
    V2P output;
    output.pos = positions[vertex_index];
    output.color = colors[vertex_index] * push.tint;
    uint corner = vertex_index & 3;
    output.uv = float2((corner == 1 || corner == 2) ? 1.0 : 0.0, (corner == 0 || corner == 1) ? 1.0 : 0.0);
    return output;
}
struct PSOutput { [vk_location(0)] [ColorTarget("swapchain", "write")] float4 color : SV_Target0; };
[shader("fragment")] PSOutput fragmentmain(V2P input) {
    PSOutput output;
    output.color = input.color;
    if (push.use_texture != 0) output.color *= textures[push.texture_id].Sample(input.uv);
    return output;
}
[shader("compute")] [numthreads(1, 1, 1)] void computemain(uint3 id : SV_DispatchThreadID) {
    colors[id.x] = float4(0.15, 0.75, 1.0, 1.0);
}
