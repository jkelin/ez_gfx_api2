[__AttributeUsage(_AttributeTargets.Var)] struct StructuredBufferAttribute { string name; }
[__AttributeUsage(_AttributeTargets.Var)] struct ColorTargetAttribute { string name; string access; }
[__AttributeUsage(_AttributeTargets.Var)] struct BindlessTextureHeapAttribute { int capacity; }
struct Push { float2 display_size; float vertical_sign; float padding; };
struct ImGuiVertex { float2 pos; float2 uv; uint col; uint pad0; uint pad1; uint pad2; };
struct ImGuiCommand { float4 clip_rect; uint texture_id; uint idx_offset; uint vtx_offset; uint pad; };
[[vk::push_constant]] ConstantBuffer<Push> push;
[StructuredBuffer("imgui_vertices")] [[vk::binding(0, 0)]] StructuredBuffer<ImGuiVertex> vertices;
[StructuredBuffer("imgui_indices")] [[vk::binding(1, 0)]] StructuredBuffer<uint> indices;
[StructuredBuffer("imgui_commands")] [[vk::binding(2, 0)]] StructuredBuffer<ImGuiCommand> commands;
static const uint EZ_GFX_MAX_TEXTURES = 1024;
struct TextureEntry { Texture2D<float4> texture; SamplerState sampler; };
struct TextureHeap { TextureEntry entries[EZ_GFX_MAX_TEXTURES]; };
[BindlessTextureHeap(EZ_GFX_MAX_TEXTURES)] [[vk::binding(0, 1)]] ParameterBlock<TextureHeap> texture_heap;
struct V2P { float4 position : SV_Position; [vk_location(0)] float2 screen_pos : TEXCOORD0; [vk_location(1)] float2 uv : TEXCOORD1; [vk_location(2)] float4 color : COLOR0; [vk_location(3)] float4 clip_rect : TEXCOORD2; [vk_location(4)] nointerpolation uint texture_id : TEXCOORD3; };
[shader("vertex")] V2P vertexmain(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) {
    ImGuiCommand command = commands[instance_index]; uint index = indices[command.idx_offset + vertex_index]; ImGuiVertex vertex = vertices[command.vtx_offset + index]; V2P output;
    output.position = float4((vertex.pos.x / push.display_size.x) * 2.0 - 1.0, ((vertex.pos.y / push.display_size.y) * 2.0 - 1.0) * push.vertical_sign, 0.0, 1.0);
    output.screen_pos = vertex.pos; output.uv = vertex.uv;
    output.color = float4((vertex.col & 255) / 255.0, ((vertex.col >> 8) & 255) / 255.0, ((vertex.col >> 16) & 255) / 255.0, ((vertex.col >> 24) & 255) / 255.0);
    output.clip_rect = command.clip_rect; output.texture_id = command.texture_id; return output;
}
struct Output { [vk_location(0)] [ColorTarget("swapchain", "write")] float4 color : SV_Target0; };
[shader("fragment")] Output fragmentmain(V2P input) {
    TextureEntry entry = texture_heap.entries[input.texture_id];
    if (input.screen_pos.x < input.clip_rect.x || input.screen_pos.x >= input.clip_rect.z || input.screen_pos.y < input.clip_rect.y || input.screen_pos.y >= input.clip_rect.w) discard;
    Output output; output.color = entry.texture.Sample(entry.sampler, input.uv) * input.color; return output;
}
