// Shared semantic declarations for every ez-gfx shader target.
[__AttributeUsage(_AttributeTargets.Var)]
struct BufferAttribute { string name; }

[__AttributeUsage(_AttributeTargets.Var)]
struct VertexHeapAttribute { string name; }

[__AttributeUsage(_AttributeTargets.Var)]
struct CounterBufferAttribute { string name; }

[__AttributeUsage(_AttributeTargets.Var)]
struct ColorTargetAttribute { string name; string access; }

[__AttributeUsage(_AttributeTargets.Var)]
struct BindlessTextureHeapAttribute { int capacity; }

[__AttributeUsage(_AttributeTargets.Var)]
struct DepthPipelineAttribute {}

/// Two-descriptor counter storage: an atomic count followed by typed elements.
struct CounterBuffer<T> {
    RWStructuredBuffer<uint> count;
    RWStructuredBuffer<T> elements;

    [ForceInline]
    void set_count(uint value) {
        count[0] = value;
    }

    [ForceInline]
    uint add_count(uint value) {
        uint previous;
        InterlockedAdd(count[0], value, previous);
        return previous;
    }

    [ForceInline]
    void set(uint index, T value) {
        elements[index] = value;
    }

    [ForceInline]
    T get(uint index) {
        return elements[index];
    }
};

static const uint EZ_GFX_MAX_TEXTURES = 1024;

struct TextureEntry {
    Texture2D<float4> texture;
    SamplerState sampler;
};

struct TextureHeap {
    TextureEntry entries[EZ_GFX_MAX_TEXTURES];
};

// Instance-index inputs consumed only through `ez_gfx_instance_index`.
// Slang drops unreferenced global varyings, so importing shaders that never
// call it gain no extra inputs.
in uint ez_gfx_instance_id : SV_InstanceID;
in uint ez_gfx_vulkan_instance_index : SV_VulkanInstanceID;

// DX12 per-draw root constant holding the indirect record's first_instance.
// Emitted as raw HLSL so it never appears in reflection; the DX12 backend
// appends this b0/space0 constant to every graphics root signature.
uint ez_gfx_dx12_base_instance() {
    __target_switch {
    case hlsl:
        __requirePrelude("cbuffer EzGfxDrawConstants : register(b0, space0) { uint ez_gfx_base_instance; }");
        __intrinsic_asm "(ez_gfx_base_instance)";
    }
}

/// Absolute instance index with Vulkan `gl_InstanceIndex` semantics: the
/// draw's first_instance plus the instance within the draw, on every backend.
/// Vertex stage only; use instead of `SV_InstanceID`, whose base-instance
/// handling differs per target (Slang subtracts it on SPIR-V and D3D omits
/// it, while Metal's `[[instance_id]]` includes it).
uint ez_gfx_instance_index() {
    __target_switch {
    case spirv:
        return ez_gfx_vulkan_instance_index;
    case metal:
        return ez_gfx_instance_id;
    case hlsl:
        // SM 6.5 lacks SV_StartInstanceLocation (SM 6.8).
        return ez_gfx_instance_id + ez_gfx_dx12_base_instance();
    }
}

// Decodes ImGui's packed RGBA8 color without target-specific operations.
float4 ez_gfx_unpack_rgba8(uint packed) {
    return float4(
        (packed & 255) / 255.0,
        ((packed >> 8) & 255) / 255.0,
        ((packed >> 16) & 255) / 255.0,
        ((packed >> 24) & 255) / 255.0);
}
