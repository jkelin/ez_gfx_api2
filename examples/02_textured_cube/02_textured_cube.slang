import ez_gfx_api;

struct Push {
    row_major float4x4 mvp;
    uint texture_id;
    uint padding0;
    uint padding1;
    uint padding2;
};

[push_constant]
ConstantBuffer<Push> push;

[StructuredBuffer("positions")]
StructuredBuffer<float4> positions;

[BindlessTextureHeap(EZ_GFX_MAX_TEXTURES)]
ParameterBlock<TextureHeap> texture_heap;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexmain(uint vertex_index : SV_VertexID) {
    VertexOutput output;
    output.position = mul(push.mvp, float4(positions[vertex_index].xyz, 1.0));

    uint corner = vertex_index & 3;
    output.uv = float2(
        (corner == 1 || corner == 2) ? 1.0 : 0.0,
        (corner == 0 || corner == 1) ? 1.0 : 0.0);
    return output;
}

struct FragmentOutput {
    [ColorTarget("swapchain", "write")]
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragmentmain(VertexOutput input) {
    TextureEntry entry = texture_heap.entries[push.texture_id];

    FragmentOutput output;
    output.color = entry.texture.Sample(entry.sampler, input.uv);
    return output;
}
