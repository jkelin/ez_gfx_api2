// Shared semantic declarations for every ez-gfx shader target.
[__AttributeUsage(_AttributeTargets.Var)]
struct StructuredBufferAttribute { string name; }

[__AttributeUsage(_AttributeTargets.Var)]
struct IndirectBufferAttribute { string name; }

[__AttributeUsage(_AttributeTargets.Var)]
struct ColorTargetAttribute { string name; string access; }

[__AttributeUsage(_AttributeTargets.Var)]
struct BindlessTextureHeapAttribute { int capacity; }

[__AttributeUsage(_AttributeTargets.Var)]
struct DepthPipelineAttribute {}

static const uint EZ_GFX_MAX_TEXTURES = 1024;

struct TextureEntry {
    Texture2D<float4> texture;
    SamplerState sampler;
};

struct TextureHeap {
    TextureEntry entries[EZ_GFX_MAX_TEXTURES];
};

// Decodes ImGui's packed RGBA8 color without target-specific operations.
float4 ez_gfx_unpack_rgba8(uint packed) {
    return float4(
        (packed & 255) / 255.0,
        ((packed >> 8) & 255) / 255.0,
        ((packed >> 16) & 255) / 255.0,
        ((packed >> 24) & 255) / 255.0);
}
