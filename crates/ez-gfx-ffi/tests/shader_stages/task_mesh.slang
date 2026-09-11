import ez_gfx_api;

// Triangle parameters written by a compute stage and consumed by the task
// stage through the same buffer. Each frame runs a different compute entry
// point, so the task input buffer changes both the triangle and its color
// with no CPU upload between the compute write and the mesh read. Color
// leads so every packing convention agrees on both member offsets.
struct DrawParams {
    float4 color;
    float scale;
};

struct MeshPayload {
    float4 color;
    float scale;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

[Buffer("draws")]
RWStructuredBuffer<DrawParams> draws;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_red() {
    DrawParams params;
    params.color = float4(1.0, 0.0, 0.0, 1.0);
    params.scale = 0.4;
    draws[0] = params;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_green() {
    DrawParams params;
    params.color = float4(0.0, 1.0, 0.0, 1.0);
    params.scale = 1.0;
    draws[0] = params;
}

[shader("amplification")]
[numthreads(1, 1, 1)]
void taskmain() {
    MeshPayload payload;
    payload.color = draws[0].color;
    payload.scale = draws[0].scale;
    DispatchMesh(1, 1, 1, payload);
}

[shader("amplification")]
[numthreads(1, 1, 1)]
void taskmain_fixed() {
    MeshPayload payload;
    payload.color = float4(0.0, 0.0, 1.0, 1.0);
    payload.scale = draws[0].scale;
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void meshmain_fixed(
    in payload MeshPayload payload,
    OutputVertices<VertexOutput, 3> vertices,
    OutputIndices<uint3, 1> triangles)
{
    float s = payload.scale;
    SetMeshOutputCounts(3, 1);
    vertices[0] = { float4(-0.75 * s, -0.5 * s, 0.5, 1.0), float3(1.0, 1.0, 0.0) };
    vertices[1] = { float4(0.75 * s, -0.5 * s, 0.5, 1.0), float3(1.0, 1.0, 0.0) };
    vertices[2] = { float4(0.0, 0.5 * s, 0.5, 1.0), float3(1.0, 1.0, 0.0) };
    triangles[0] = uint3(0, 1, 2);
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void meshmain(
    in payload MeshPayload payload,
    OutputVertices<VertexOutput, 3> vertices,
    OutputIndices<uint3, 1> triangles)
{
    float s = payload.scale;
    float3 c = payload.color.xyz;
    SetMeshOutputCounts(3, 1);
    vertices[0] = { float4(-0.75 * s, -0.5 * s, 0.5, 1.0), c };
    vertices[1] = { float4(0.75 * s, -0.5 * s, 0.5, 1.0), c };
    vertices[2] = { float4(0.0, 0.5 * s, 0.5, 1.0), c };
    triangles[0] = uint3(0, 1, 2);
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
