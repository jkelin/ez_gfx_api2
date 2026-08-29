[__AttributeUsage(_AttributeTargets.Var)] struct StructuredBufferAttribute { string name; }
[__AttributeUsage(_AttributeTargets.Var)] struct ColorTargetAttribute { string name; string access; }
[StructuredBuffer("positions")] [[vk::binding(0, 0)]] StructuredBuffer<float4> positions;
struct V2P { float4 position : SV_Position; };
[shader("vertex")] V2P vertexmain(uint vertex_index : SV_VertexID) {
    V2P output; output.position = float4(positions[vertex_index].xyz, 1.0); return output;
}
struct Output { [vk_location(0)] [ColorTarget("swapchain", "write")] float4 color : SV_Target0; };
[shader("fragment")] Output fragmentmain(V2P input) {
    Output output; output.color = float4(1.0, 0.0, 0.0, 1.0); return output;
}
