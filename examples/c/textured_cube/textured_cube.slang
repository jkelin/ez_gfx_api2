import "../../../ez_gfx_api";

struct Primitive {
    uint first_index;
    uint index_count;
    uint vertex_offset;
    uint normal_offset;
};

struct Draw {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

[StructuredBuffer("primitives")]
StructuredBuffer<Primitive> primitives;

[IndirectBuffer("draw_commands")]
RWStructuredBuffer<Draw> draw_commands;

[shader("compute")]
[numthreads(1, 1, 1)]
void computemain(uint3 id : SV_DispatchThreadID) {
    if (id.x != 0) {
        return;
    }

    Primitive primitive = primitives[0];
    Draw command;
    command.index_count = primitive.index_count;
    command.instance_count = 1;
    command.first_index = primitive.first_index;
    command.vertex_offset = int(primitive.vertex_offset);
    command.first_instance = 0;
    draw_commands[0] = command;
}

[StructuredBuffer("positions")]
StructuredBuffer<float4> positions;

[StructuredBuffer("normals")]
StructuredBuffer<float4> normals;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertexmain(uint vertex_index : SV_VertexID) {
    float3 position = positions[vertex_index].xyz;
    float3 normal = normals[vertex_index].xyz;

    VertexOutput output;
    output.position = float4(
        position.x * 0.45 + position.z * 0.20,
        position.y * 0.55 + position.z * 0.12,
        position.z * 0.08 + 0.50,
        1.0
    );
    output.color = float4(normal * 0.35 + 0.65, 1.0);
    return output;
}

struct FragmentOutput {
    [ColorTarget("swapchain", "write")]
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragmentmain(VertexOutput input) {
    FragmentOutput output;
    output.color = input.color;
    return output;
}
