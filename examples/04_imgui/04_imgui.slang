import "../../ez_gfx_api";

struct Push {
    float2 display_size;
    float vertical_sign;
    float padding;
};

struct ImGuiVertex {
    float2 pos;
    float2 uv;
    uint col;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct ImGuiCommand {
    float4 clip_rect;
    uint texture_id;
    uint idx_offset;
    uint vtx_offset;
    uint pad;
};

[push_constant]
ConstantBuffer<Push> push;

[StructuredBuffer("imgui_vertices")]
StructuredBuffer<ImGuiVertex> vertices;

[StructuredBuffer("imgui_indices")]
StructuredBuffer<uint> indices;

[StructuredBuffer("imgui_commands")]
StructuredBuffer<ImGuiCommand> commands;

[BindlessTextureHeap(EZ_GFX_MAX_TEXTURES)]
ParameterBlock<TextureHeap> texture_heap;

struct VertexOutput {
    float4 position : SV_Position;
    float2 screen_pos : TEXCOORD0;
    float2 uv : TEXCOORD1;
    float4 color : COLOR0;
    float4 clip_rect : TEXCOORD2;
    nointerpolation uint texture_id : TEXCOORD3;
};

[shader("vertex")]
VertexOutput vertexmain(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) {
    ImGuiCommand command = commands[instance_index];
    uint index = indices[command.idx_offset + vertex_index];
    ImGuiVertex vertex = vertices[command.vtx_offset + index];

    VertexOutput output;
    output.position = float4(
        (vertex.pos.x / push.display_size.x) * 2.0 - 1.0,
        ((vertex.pos.y / push.display_size.y) * 2.0 - 1.0) * push.vertical_sign,
        0.0,
        1.0);
    output.screen_pos = vertex.pos;
    output.uv = vertex.uv;
    output.color = ez_gfx_unpack_rgba8(vertex.col);
    output.clip_rect = command.clip_rect;
    output.texture_id = command.texture_id;
    return output;
}

struct FragmentOutput {
    [ColorTarget("swapchain", "write")]
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragmentmain(VertexOutput input) {
    TextureEntry entry = texture_heap.entries[input.texture_id];
    if (input.screen_pos.x < input.clip_rect.x || input.screen_pos.x >= input.clip_rect.z ||
        input.screen_pos.y < input.clip_rect.y || input.screen_pos.y >= input.clip_rect.w) {
        discard;
    }

    FragmentOutput output;
    output.color = entry.texture.Sample(entry.sampler, input.uv) * input.color;
    return output;
}
