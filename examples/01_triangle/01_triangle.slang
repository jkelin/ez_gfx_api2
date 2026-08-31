import ez_gfx_api;

[StructuredBuffer("positions")]
StructuredBuffer<float4> positions;

struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertexmain(uint vertex_index : SV_VertexID) {
    VertexOutput output;
    output.position = float4(positions[vertex_index].xyz, 1.0);
    return output;
}

struct FragmentOutput {
    [ColorTarget("swapchain", "write")]
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragmentmain(VertexOutput input) {
    FragmentOutput output;
    output.color = float4(1.0, 0.0, 0.0, 1.0);
    return output;
}
