import "../../ez_gfx_api";

struct Primitive {
    uint first_index;
    uint index_count;
    uint vertex_offset;
    uint normal_offset;
    row_major float4x4 transform;
};

struct Draw {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

struct SceneParams {
    row_major float4x4 mvp;
    uint primitive_count;
    uint padding0;
    uint padding1;
    uint padding2;
};

[DepthPipeline]
[Buffer("params")]
StructuredBuffer<SceneParams> params;

[Buffer("primitives")]
StructuredBuffer<Primitive> primitives;

[CounterBuffer("draw_commands")]
CounterBuffer<Draw> draw_commands;

[shader("compute")]
[numthreads(1, 1, 1)]
void computemain(uint3 id : SV_DispatchThreadID) {
    if (id.x == 0) {
        draw_commands.set_count(params[0].primitive_count);
    }
    if (id.x >= params[0].primitive_count) {
        return;
    }

    Primitive primitive = primitives[id.x];
    Draw command;
    command.index_count = primitive.index_count;
    command.instance_count = 1;
    command.first_index = primitive.first_index;
    command.vertex_offset = int(primitive.vertex_offset);
    command.first_instance = id.x;
    draw_commands.set(id.x, command);
}

[VertexHeap("positions")]
StructuredBuffer<float4> positions;

[VertexHeap("normals")]
StructuredBuffer<float4> normals;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertexmain(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) {
    Primitive primitive = primitives[instance_index];
    VertexOutput output;
    output.position = mul(params[0].mvp, mul(primitive.transform, float4(positions[vertex_index].xyz, 1.0)));

    uint local_vertex = vertex_index - primitive.vertex_offset;
    float3 normal = normalize(mul((float3x3)primitive.transform, normals[local_vertex + primitive.normal_offset].xyz));
    output.color = float4(normal * 0.5 + 0.5, 1.0);
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
