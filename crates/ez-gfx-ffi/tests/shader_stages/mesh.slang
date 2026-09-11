import ez_gfx_api;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

// Procedural traditional stage for background and mixed-traditional draws.
// Vertices 0-2 cover the frame in blue; vertices 3-5 emit a green corner
// triangle that the pixel assertions locate modulo the backend Y convention;
// vertices 6-8 cover the frame in exact black for the mesh pixel background.
[shader("vertex")]
VertexOutput vertexmain(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {
    float2 full[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 corner[3] = { float2(-1.0, 0.5), float2(-0.5, 0.5), float2(-1.0, 1.0) };
    float2 p;
    float3 c;
    uint tri = vertex_id / 3;
    uint v = vertex_id % 3;
    if (tri == 0) {
        p = full[v];
        c = float3(0.0, 0.0, 1.0);
    } else if (tri == 1) {
        p = corner[v];
        c = float3(0.0, 1.0, 0.0);
    } else {
        p = full[v];
        c = float3(0.0, 0.0, 0.0);
    }
    VertexOutput output;
    output.position = float4(p, 0.5, 1.0);
    output.color = c;
    return output;
}

// Mesh-only triangle with a nontrivial workgroup shape. Only thread zero emits
// so every dispatched group reproduces the same red triangle.
[shader("mesh")]
[numthreads(4, 2, 1)]
[outputtopology("triangle")]
void meshmain(
    uint group_index : SV_GroupIndex,
    OutputVertices<VertexOutput, 3> vertices,
    OutputIndices<uint3, 1> triangles)
{
    if (group_index == 0) {
        SetMeshOutputCounts(3, 1);
        vertices[0] = { float4(-0.75, -0.5, 0.5, 1.0), float3(1.0, 0.0, 0.0) };
        vertices[1] = { float4(0.75, -0.5, 0.5, 1.0), float3(1.0, 0.0, 0.0) };
        vertices[2] = { float4(0.0, 0.5, 0.5, 1.0), float3(1.0, 0.0, 0.0) };
        triangles[0] = uint3(0, 1, 2);
    }
}

struct FragmentOutput {
    [ColorTarget("swapchain", "write")]
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragmentmain(VertexOutput input) {
    FragmentOutput output;
    output.color = float4(input.color, 1.0);
    return output;
}

struct TargetFragmentOutput {
    [ColorTarget("mesh-black-target", "write")]
    float4 color : SV_Target0;
};

[shader("fragment")]
TargetFragmentOutput fragmentmain_target(VertexOutput input) {
    TargetFragmentOutput output;
    output.color = float4(input.color, 1.0);
    return output;
}
