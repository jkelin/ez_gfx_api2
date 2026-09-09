#ifndef EZ_GFX_API_H
#define EZ_GFX_API_H

#include <stdint.h>
#include <stddef.h>

#define EZ_GFX_ABI_VERSION 33u

#if defined(__clang__)
#  if __has_attribute(access)
#    define EZ_GFX_ACCESS(...) __attribute__((access(__VA_ARGS__)))
#  else
#    define EZ_GFX_ACCESS(...)
#  endif
#elif defined(__GNUC__) && (__GNUC__ >= 10)
#  define EZ_GFX_ACCESS(...) __attribute__((access(__VA_ARGS__)))
#else
#  define EZ_GFX_ACCESS(...)
#endif

/* ABI string contract: each const char* has an explicit byte length, denotes exactly that many UTF-8 bytes without scanning for a terminator, and rejects embedded NUL. */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * EzGfxContext: Opaque packed u64 context handle (slot+1/generation in bits 0-39).
 */
typedef uint64_t EzGfxContext;
/**
 * EzGfxSurface: Opaque packed u64 surface handle; child bits index the context identity arena and resolve only as Surface.
 */
typedef uint64_t EzGfxSurface;
/**
 * EzGfxFrame: Opaque generation-, kind-, and owner-validated live frame handle.
 */
typedef uint64_t EzGfxFrame;
/** Stable process-unique identity for one enqueued readback request. */
typedef uint64_t EzGfxReadbackRequest;
/**
 * EzGfxShader: Opaque packed u64 shader handle; child bits index the context identity arena and resolve only as Shader.
 */
typedef uint64_t EzGfxShader;
/**
 * EzGfxCounterBuffer: Opaque packed u64 one-frame counter-buffer handle; child bits index the context identity arena and resolve only as Indirect.
 */
typedef uint64_t EzGfxCounterBuffer;
/**
 * EzGfxBuffer: Opaque packed u64 one-frame buffer handle; child bits index the context identity arena and resolve only as Structured.
 */
typedef uint64_t EzGfxBuffer;
/** Opaque named vertex heap; owner and generation are validated. */
typedef uint64_t EzGfxVertexHeap;
/** Opaque allocation within a named vertex heap; owner and generation are validated. */
typedef uint64_t EzGfxVertexAllocation;
/** Opaque allocation within the global index heap; owner and generation are validated. */
typedef uint64_t EzGfxIndexAllocation;
/**
 * EzGfxTexture: Opaque packed u64 texture handle; child bits index the context identity arena and resolve only as Texture.
 */
typedef uint64_t EzGfxTexture;
/**
 * EzGfxRenderTarget: Opaque packed u64 render-target handle; child bits index the context identity arena and resolve only as RenderTarget.
 */
typedef uint64_t EzGfxRenderTarget;

/**
 * EzGfxResult:
 * @EzGfxResult_Ok: The operation completed successfully.
 * @EzGfxResult_InvalidArgument: A public argument failed validation.
 * @EzGfxResult_InvalidContext: The context is null, stale, or does not own the resource.
 * @EzGfxResult_NativeFailure: The native graphics operation failed.
 * @EzGfxResult_NotReady: The operation is temporarily unavailable, such as a minimized surface or pending texture.
 * @EzGfxResult_QueueFull: Asynchronous scheduling or staging capacity is unavailable.
 * @EzGfxResult_Cancelled: The asynchronous operation was cancelled before completion.
 *
 * Result returned by context, surface, shader, buffer, and render operations.
 */
typedef uint8_t EzGfxResult;
enum {
    EzGfxResult_Ok = 0,
    EzGfxResult_InvalidArgument = 1,
    EzGfxResult_InvalidContext = 2,
    EzGfxResult_NativeFailure = 3,
    EzGfxResult_NotReady = 4,
    EzGfxResult_Unsupported = 5,
    EzGfxResult_DeviceLost = 6,
    EzGfxResult_QueueFull = 7,
    EzGfxResult_Cancelled = 8,
};

/**
 * EzGfxTextureError:
 * @EzGfxTextureError_None: The operation completed successfully.
 * @EzGfxTextureError_InvalidContext: The context is null, stale, or does not own the resource.
 * @EzGfxTextureError_InvalidArguments: Texture data or description failed validation.
 * @EzGfxTextureError_UnsupportedFormat: No registered decoder supports the source format.
 * @EzGfxTextureError_OutOfTextureHandles: The texture handle table is full.
 * @EzGfxTextureError_OutOfMemory: Texture allocation failed.
 * @EzGfxTextureError_DecodeFailed: The image decoder rejected the data.
 * @EzGfxTextureError_VulkanFailed: A Vulkan texture operation failed.
 * @EzGfxTextureError_WorkerUnavailable: No texture upload worker is available.
 * @EzGfxTextureError_NotFound: The texture handle is not loaded.
 *
 * Retained texture-specific code set and type for ABI compatibility. Current texture exports return EzGfxResult or void.
 */
typedef uint8_t EzGfxTextureError;
enum {
    EzGfxTextureError_None = 0,
    EzGfxTextureError_InvalidContext = 1,
    EzGfxTextureError_InvalidArguments = 2,
    EzGfxTextureError_UnsupportedFormat = 3,
    EzGfxTextureError_OutOfTextureHandles = 4,
    EzGfxTextureError_OutOfMemory = 5,
    EzGfxTextureError_DecodeFailed = 6,
    EzGfxTextureError_VulkanFailed = 7,
    EzGfxTextureError_WorkerUnavailable = 8,
    EzGfxTextureError_NotFound = 9,
};

/**
 * EzGfxShaderKind:
 * @EzGfxShaderKind_Graphics: Vertex and fragment shader pair.
 * @EzGfxShaderKind_Compute: Compute shader.
 *
 * Shader stage family.
 */
typedef uint8_t EzGfxShaderKind;
enum {
    EzGfxShaderKind_Graphics = 0,
    EzGfxShaderKind_Compute = 1,
};

/** Explicit native backend selection for EzGfxBackendContextDesc. */
typedef uint8_t EzGfxBackend;
enum {
    EzGfxBackend_Vulkan = 1,
    EzGfxBackend_Dx12 = 2,
    EzGfxBackend_Metal = 3,
};
/** Stable adapter classification for EzGfxAdapterInfo. */
typedef uint8_t EzGfxAdapterClass;
enum {
    EzGfxAdapterClass_Software = 0,
    EzGfxAdapterClass_Other = 1,
    EzGfxAdapterClass_Integrated = 2,
    EzGfxAdapterClass_Discrete = 3,
};

/** Runtime operation phase carried by EzGfxRuntimeRecord. */
typedef uint8_t EzGfxRuntimePhase;
enum {
    EzGfxRuntimePhase_Admission = 1,
    EzGfxRuntimePhase_Decode = 2,
    EzGfxRuntimePhase_Upload = 3,
    EzGfxRuntimePhase_Bind = 4,
    EzGfxRuntimePhase_Submit = 5,
    EzGfxRuntimePhase_Present = 6,
    EzGfxRuntimePhase_Readback = 7,
    EzGfxRuntimePhase_Device = 8,
};

/** Bounded local diagnostic severity. */
typedef uint8_t EzGfxDiagnosticLevel;
enum {
    EzGfxDiagnosticLevel_Info = 1,
    EzGfxDiagnosticLevel_Warning = 2,
    EzGfxDiagnosticLevel_Error = 3,
};

/** Resource family carried by EzGfxUploadEvent. */
typedef uint8_t EzGfxUploadResourceKind;
enum {
    EzGfxUploadResourceKind_Texture = 1,
    EzGfxUploadResourceKind_VertexAllocation = 2,
    EzGfxUploadResourceKind_IndexAllocation = 3,
};

/** Lossless upload transition carried by EzGfxUploadEvent. */
typedef uint8_t EzGfxUploadStatus;
enum {
    EzGfxUploadStatus_SourceStaged = 1,
    EzGfxUploadStatus_DeviceReady = 2,
    EzGfxUploadStatus_Failed = 3,
    EzGfxUploadStatus_Cancelled = 4,
};

/**
 * EzGfxSurfacePlatform:
 * @EzGfxSurfacePlatform_Win32: Win32 HWND and HINSTANCE handles.
 * @EzGfxSurfacePlatform_GLFW: GLFW native window handle.
 * @EzGfxSurfacePlatform_Headless: Windowless Vulkan headless surface; no native handles. Requires ABI 27.
 *
 * Native surface platform.
 */
typedef uint8_t EzGfxSurfacePlatform;
enum {
    EzGfxSurfacePlatform_Win32 = 0,
    EzGfxSurfacePlatform_GLFW = 1,
};
/* Borrowed CAMetalLayer pointer in EzGfxSurfaceDesc.window. */
#define EzGfxSurfacePlatform_MetalLayer 2
/* Windowless headless surface; EzGfxSurfaceDesc.window and EzGfxSurfaceDesc.display are null. Requires ABI 27. */
#define EzGfxSurfacePlatform_Headless 3

/**
 * EzGfxSourceTextureFormat:
 * @EzGfxSourceTextureFormat_Rgb: Raw RGB pixels.
 * @EzGfxSourceTextureFormat_Rgba: Raw RGBA pixels.
 * @EzGfxSourceTextureFormat_Bmp: BMP image bytes.
 * @EzGfxSourceTextureFormat_Jpeg: JPEG image bytes.
 * @EzGfxSourceTextureFormat_Png: PNG image bytes.
 * @EzGfxSourceTextureFormat_Tga: TGA image bytes.
 * @EzGfxSourceTextureFormat_Ktx2: KTX2 image bytes.
 * @EzGfxSourceTextureFormat_Basis: Standalone Basis Universal bytes when built with optional Basis support.
 * @EzGfxSourceTextureFormat_Dds: DDS image bytes with a legacy or DX10 header.
 * @EzGfxSourceTextureFormat_Raw: Tightly packed mip bytes in the destination storage format; the destination must name a concrete format.
 *
 * Source image encoding.
 */
typedef uint8_t EzGfxSourceTextureFormat;
enum {
    EzGfxSourceTextureFormat_Rgb = 0,
    EzGfxSourceTextureFormat_Rgba = 1,
    EzGfxSourceTextureFormat_Bmp = 2,
    EzGfxSourceTextureFormat_Jpeg = 3,
    EzGfxSourceTextureFormat_Png = 4,
    EzGfxSourceTextureFormat_Tga = 5,
    EzGfxSourceTextureFormat_Ktx2 = 6,
    EzGfxSourceTextureFormat_Basis = 7,
    EzGfxSourceTextureFormat_Dds = 8,
    EzGfxSourceTextureFormat_Raw = 9,
};

/**
 * EzGfxTextureFilter:
 * @EzGfxTextureFilter_Nearest: Nearest-neighbor filtering.
 * @EzGfxTextureFilter_Linear: Linear filtering.
 *
 * Texture sampling filter.
 */
typedef uint8_t EzGfxTextureFilter;
enum {
    EzGfxTextureFilter_Nearest = 0,
    EzGfxTextureFilter_Linear = 1,
};

/**
 * EzGfxTextureAddressMode:
 * @EzGfxTextureAddressMode_Repeat: Repeat coordinates.
 * @EzGfxTextureAddressMode_ClampToEdge: Clamp coordinates to the edge.
 *
 * Texture addressing mode.
 */
typedef uint8_t EzGfxTextureAddressMode;
enum {
    EzGfxTextureAddressMode_Repeat = 0,
    EzGfxTextureAddressMode_ClampToEdge = 1,
};

/**
 * EzGfxTextureDestinationFormat:
 * @EzGfxTextureDestinationFormat_Rgba8Unorm: Linear 8-bit normalized RGBA.
 * @EzGfxTextureDestinationFormat_Auto: Prefer an admitted native compressed format.
 * @EzGfxTextureDestinationFormat_Rgba8Srgb: sRGB 8-bit normalized RGBA.
 * @EzGfxTextureDestinationFormat_Bc1Unorm: Linear BC1.
 * @EzGfxTextureDestinationFormat_Bc1Srgb: sRGB BC1.
 * @EzGfxTextureDestinationFormat_Bc3Unorm: Linear BC3.
 * @EzGfxTextureDestinationFormat_Bc3Srgb: sRGB BC3.
 * @EzGfxTextureDestinationFormat_Bc7Unorm: Linear BC7.
 * @EzGfxTextureDestinationFormat_Bc7Srgb: sRGB BC7.
 * @EzGfxTextureDestinationFormat_Astc4x4Unorm: Linear ASTC 4x4.
 * @EzGfxTextureDestinationFormat_Astc4x4Srgb: sRGB ASTC 4x4.
 *
 * Destination texture format.
 */
typedef uint8_t EzGfxTextureDestinationFormat;
enum {
    EzGfxTextureDestinationFormat_Rgba8Unorm = 0,
    EzGfxTextureDestinationFormat_Auto = 1,
    EzGfxTextureDestinationFormat_Rgba8Srgb = 2,
    EzGfxTextureDestinationFormat_Bc1Unorm = 3,
    EzGfxTextureDestinationFormat_Bc1Srgb = 4,
    EzGfxTextureDestinationFormat_Bc3Unorm = 5,
    EzGfxTextureDestinationFormat_Bc3Srgb = 6,
    EzGfxTextureDestinationFormat_Bc7Unorm = 7,
    EzGfxTextureDestinationFormat_Bc7Srgb = 8,
    EzGfxTextureDestinationFormat_Astc4x4Unorm = 9,
    EzGfxTextureDestinationFormat_Astc4x4Srgb = 10,
};
/**
 * EzGfxRenderTargetFormat:
 * @EzGfxRenderTargetFormat_Rgba8Unorm: Four-channel 8-bit normalized RGBA.
 * @EzGfxRenderTargetFormat_Bgra8Srgb: Four-channel 8-bit sRGB BGRA.
 * @EzGfxRenderTargetFormat_Rgba16Float: Four-channel 16-bit floating-point RGBA.
 * @EzGfxRenderTargetFormat_Depth32Float: 32-bit floating-point depth.
 * @EzGfxRenderTargetFormat_Bc7Unorm: BC7-compressed 8-bit normalized RGBA.
 * @EzGfxRenderTargetFormat_Astc4x4Unorm: ASTC-compressed 4x4 texel 8-bit normalized RGBA.
 *
 * Render-target storage format. Discriminants mirror the runtime format codes.
 */
typedef uint8_t EzGfxRenderTargetFormat;
enum {
    EzGfxRenderTargetFormat_Rgba8Unorm = 1,
    EzGfxRenderTargetFormat_Bgra8Srgb = 2,
    EzGfxRenderTargetFormat_Rgba16Float = 3,
    EzGfxRenderTargetFormat_Depth32Float = 4,
    EzGfxRenderTargetFormat_Bc7Unorm = 5,
    EzGfxRenderTargetFormat_Astc4x4Unorm = 6,
};

/**
 * EzGfxRenderTargetUsage:
 * @EzGfxRenderTargetUsage_Color: Color-rendering output.
 * @EzGfxRenderTargetUsage_Depth: Depth-rendering output; creation stays unsupported.
 * @EzGfxRenderTargetUsage_Storage: Storage access; creation stays unsupported.
 * @EzGfxRenderTargetUsage_Sampled: Texture sampling; creation stays unsupported.
 *
 * Render-target access mode.
 */
typedef uint8_t EzGfxRenderTargetUsage;
enum {
    EzGfxRenderTargetUsage_Color = 0,
    EzGfxRenderTargetUsage_Depth = 1,
    EzGfxRenderTargetUsage_Storage = 2,
    EzGfxRenderTargetUsage_Sampled = 3,
};

/**
 * EzGfxCullMode:
 * @EzGfxCullMode_None: No face culling.
 * @EzGfxCullMode_Front: Cull front-facing primitives.
 * @EzGfxCullMode_Back: Cull back-facing primitives.
 *
 * Pipeline cull mode.
 */
typedef uint8_t EzGfxCullMode;
enum {
    EzGfxCullMode_None = 0,
    EzGfxCullMode_Front = 1,
    EzGfxCullMode_Back = 2,
};

/**
 * EzGfxFrontFace:
 * @EzGfxFrontFace_CounterClockwise: Counter-clockwise winding.
 * @EzGfxFrontFace_Clockwise: Clockwise winding.
 *
 * Front-face winding order.
 */
typedef uint8_t EzGfxFrontFace;
enum {
    EzGfxFrontFace_CounterClockwise = 0,
    EzGfxFrontFace_Clockwise = 1,
};

/**
 * EzGfxPrimitiveType:
 * @EzGfxPrimitiveType_TriangleList: Triangle list.
 * @EzGfxPrimitiveType_PointList: Point list.
 * @EzGfxPrimitiveType_LineList: Line list.
 * @EzGfxPrimitiveType_LineStrip: Line strip.
 * @EzGfxPrimitiveType_TriangleStrip: Triangle strip.
 * @EzGfxPrimitiveType_TriangleFan: Triangle fan.
 *
 * Pipeline primitive topology.
 */
typedef uint8_t EzGfxPrimitiveType;
enum {
    EzGfxPrimitiveType_TriangleList = 0,
    EzGfxPrimitiveType_PointList = 1,
    EzGfxPrimitiveType_LineList = 2,
    EzGfxPrimitiveType_LineStrip = 3,
    EzGfxPrimitiveType_TriangleStrip = 4,
    EzGfxPrimitiveType_TriangleFan = 5,
};

/**
 * EzGfxBlendMode:
 * @EzGfxBlendMode_None: Opaque blending.
 * @EzGfxBlendMode_Alpha: Source-alpha blending.
 *
 * Pipeline blend mode.
 */
typedef uint8_t EzGfxBlendMode;
enum {
    EzGfxBlendMode_None = 0,
    EzGfxBlendMode_Alpha = 1,
};

/**
 * EzGfxAdapterDesc:
 * @stable_id: Stable 128-bit adapter identity from enumeration.
 * @allow_software: Non-zero accepts a software-class adapter; zero or one.
 *
 * Explicit adapter request selected by stable identity. Requires ABI 26.
 */
typedef struct EzGfxAdapterDesc {
    uint8_t stable_id[16];
    uint8_t allow_software;
} EzGfxAdapterDesc;

/**
 * EzGfxAdapterInfo:
 * @stable_id: Stable 128-bit adapter identity from enumeration.
 * @backend: Value from EzGfxBackend.
 * @adapter_class: Value from EzGfxAdapterClass.
 * @admitted: Non-zero when the adapter passes admission under the queried policy.
 * @software_rejected: Non-zero when software policy alone rejects the adapter.
 * @error_count: Count of unmet profile requirements; zero when admitted.
 *
 * Enumerated adapter identity with admission diagnostics. Requires ABI 26.
 */
typedef struct EzGfxAdapterInfo {
    uint8_t stable_id[16];
    EzGfxBackend backend;
    EzGfxAdapterClass adapter_class;
    uint8_t admitted;
    uint8_t software_rejected;
    uint32_t error_count;
} EzGfxAdapterInfo;

/**
 * EzGfxContextDesc:
 * @enable_debug: Non-zero enables debug utilities.
 * @enable_validation: Non-zero enables validation layers.
 * @surface_platform: Value from EzGfxSurfacePlatform.
 * @texture_decode_workers: Async texture decode threads; zero selects the default topology.
 * @adapter_count: Explicit adapter requests; zero keeps default ranking, one selects by identity. Requires ABI 26.
 * @adapter (nullable) (array length=adapter_count): Exactly @adapter_count explicit requests; null if and only if zero. Requires ABI 26.
 *
 * Context creation options.
 */
typedef struct EzGfxContextDesc {
    uint8_t enable_debug;
    uint8_t enable_validation;
    EzGfxSurfacePlatform surface_platform;
    uint32_t texture_decode_workers;
    uint32_t adapter_count;
    const EzGfxAdapterDesc *adapter;
} EzGfxContextDesc;

/** Backend-selecting context creation options. */
typedef struct EzGfxBackendContextDesc {
    uint8_t enable_debug;
    uint8_t enable_validation;
    EzGfxSurfacePlatform surface_platform;
    EzGfxBackend backend;
    uint32_t texture_decode_workers;
    uint32_t adapter_count;
    const EzGfxAdapterDesc *adapter;
} EzGfxBackendContextDesc;

/**
 * EzGfxSurfaceDesc:
 * @window (nullable for Headless): Native HWND or CAMetalLayer pointer; null for Headless.
 * @display (nullable): Native HINSTANCE; null for Metal and Headless, permitted for GLFW.
 * @platform: Value from EzGfxSurfacePlatform.
 * @width: Initial framebuffer width.
 * @height: Initial framebuffer height.
 * @cache_presented_snapshots: Non-zero caches presented images.
 *
 * Caller-owned native window or Metal layer used to create a presentation surface.
 */
typedef struct EzGfxSurfaceDesc {
    void * window;
    void * display;
    EzGfxSurfacePlatform platform;
    uint32_t width;
    uint32_t height;
    uint8_t cache_presented_snapshots;
} EzGfxSurfaceDesc;

/**
 * EzGfxShaderDesc:
 * @path (not nullable): Exactly @path_length UTF-8 bytes.
 * @path_length: Non-zero byte length of @path.
 * @vertex_entry (nullable): Exactly @vertex_entry_length UTF-8 bytes, or null when its length is zero.
 * @vertex_entry_length: Byte length of @vertex_entry.
 * @fragment_entry (nullable): Exactly @fragment_entry_length UTF-8 bytes, or null when its length is zero.
 * @fragment_entry_length: Byte length of @fragment_entry.
 * @compute_entry (nullable): Exactly @compute_entry_length UTF-8 bytes, or null when its length is zero.
 * @compute_entry_length: Byte length of @compute_entry.
 * @kind: Value from EzGfxShaderKind.
 *
 * Shader source and entry-point metadata. String ranges are not NUL-terminated
 * and embedded NUL bytes are invalid.
 */
typedef struct EzGfxShaderDesc {
    const char * path;
    size_t path_length;
    const char * vertex_entry;
    size_t vertex_entry_length;
    const char * fragment_entry;
    size_t fragment_entry_length;
    const char * compute_entry;
    size_t compute_entry_length;
    EzGfxShaderKind kind;
} EzGfxShaderDesc;

/**
 * EzGfxTextureDesc:
 * @source_format: Value from EzGfxSourceTextureFormat.
 * @destination_format: Value from EzGfxTextureDestinationFormat.
 * @width: Decoded width for raw pixels and Raw ingestion.
 * @height: Decoded height for raw pixels and Raw ingestion.
 * @mip_count: Number of mip levels, zero for decoder defaults; Raw requires an explicit nonzero count.
 * @generate_mips: Non-zero requests mip generation.
 * @min_filter: Value from EzGfxTextureFilter.
 * @mag_filter: Value from EzGfxTextureFilter.
 * @max_anisotropy: Requested anisotropy; zero uses the default.
 * @address_mode_u: Value from EzGfxTextureAddressMode.
 * @address_mode_v: Value from EzGfxTextureAddressMode.
 * @address_mode_w: Value from EzGfxTextureAddressMode.
 * @debug_label (nullable): Exactly @debug_label_length UTF-8 bytes, or null when its length is zero.
 * @debug_label_length: Byte length of @debug_label.
 *
 * Texture decoding and sampling metadata.
 */
typedef struct EzGfxTextureDesc {
    EzGfxSourceTextureFormat source_format;
    EzGfxTextureDestinationFormat destination_format;
    uint32_t width;
    uint32_t height;
    uint32_t mip_count;
    uint8_t generate_mips;
    EzGfxTextureFilter min_filter;
    EzGfxTextureFilter mag_filter;
    float max_anisotropy;
    EzGfxTextureAddressMode address_mode_u;
    EzGfxTextureAddressMode address_mode_v;
    EzGfxTextureAddressMode address_mode_w;
    const char * debug_label;
    size_t debug_label_length;
} EzGfxTextureDesc;

/** One decoded custom texture mip retained until the release callback. */
typedef struct EzGfxDecodedTextureMip {
    uint32_t width;
    uint32_t height;
    const uint8_t *data;
    size_t data_size;
} EzGfxDecodedTextureMip;

/** Custom decoder output copied before its release callback is invoked. */
typedef struct EzGfxDecodedTexture {
    EzGfxTextureDestinationFormat format;
    uint32_t mip_count;
    const EzGfxDecodedTextureMip *mips;
} EzGfxDecodedTexture;

/** Borrowed bytes and destination rectangle for one asynchronous texture update. */
typedef struct EzGfxTextureRegionDesc {
    uint32_t mip_level;
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    const uint8_t *data;
    size_t data_size;
} EzGfxTextureRegionDesc;

/**
 * EzGfxRenderTargetDesc:
 * @name (not nullable): Exactly @name_length UTF-8 bytes, 1..=255 bytes.
 * @name_length: Byte length of @name.
 * @usage: Value from EzGfxRenderTargetUsage.
 * @relative_scale: Scale factor relative to the reference dimensions; finite and positive.
 * @samples: Requested multisample count; one of 1, 2, 4, or 8.
 * @candidate_formats (not nullable) (array length=candidate_count): Exactly @candidate_count format codes from EzGfxRenderTargetFormat.
 * @candidate_count: Candidate format count, 1..=16.
 * @sampleable: Non-zero requires texture-sampling support; zero or one.
 * @use_clear: Non-zero stores @clear_color; zero or one.
 * @clear_color: Color clear value read only when @use_clear is non-zero.
 *
 * Render-target name, usage, format candidates, and clear value. String and
 * candidate ranges are borrowed only for the creating call.
 */
typedef struct EzGfxRenderTargetDesc {
    const uint8_t *name;
    size_t name_length;
    EzGfxRenderTargetUsage usage;
    float relative_scale;
    uint8_t samples;
    const uint8_t *candidate_formats;
    uint32_t candidate_count;
    uint8_t sampleable;
    uint8_t use_clear;
    float clear_color[4];
} EzGfxRenderTargetDesc;

/** Monotonic context-wide asynchronous texture pipeline counters. */
typedef struct EzGfxTextureUploadTelemetry {
    uint64_t decode_microseconds;
    uint64_t staging_bytes;
    uint64_t queue_latency_microseconds;
    uint64_t handoff_latency_microseconds;
} EzGfxTextureUploadTelemetry;

/**
 * Concurrent custom decoder callback. `compression_support` uses bit 0 for BC and bit 1 for ASTC.
 * On success, output pointers must remain valid until `EzGfxTextureDecoderReleaseCallback`.
 */
typedef EzGfxResult (*EzGfxTextureDecoderCallback)(
    const uint8_t *data,
    size_t data_size,
    uint8_t compression_support,
    EzGfxDecodedTexture *out_texture,
    void *user_data);

/** Releases one successful custom decoder output after ez-gfx copies it. */
typedef void (*EzGfxTextureDecoderReleaseCallback)(
    const EzGfxDecodedTexture *texture,
    void *user_data);

/**
 * EzGfxBinding:
 * @name (not nullable): Exactly @name_length UTF-8 shader-binding-name bytes.
 * @name_length: Non-zero byte length of @name.
 * @buffer: Optional buffer handle.
 * @counter_buffer: Optional one-frame counter-command buffer handle.
 * @render_target: Optional render-target handle.
 *
 * Shader resource binding. Names are not NUL-terminated and embedded NUL bytes
 * are invalid.
 */
typedef struct EzGfxBinding {
    const char * name;
    size_t name_length;
    EzGfxBuffer buffer;
    EzGfxCounterBuffer counter_buffer;
    EzGfxRenderTarget render_target;
} EzGfxBinding;

/**
 * EzGfxDynamicState:
 * @cull_mode: Cull mode override.
 * @front_face: Front-face winding override.
 * @primitive_type: Primitive topology override.
 * @blend_mode: Blend mode override.
 *
 * Optional dynamic state overrides.
 */
typedef struct EzGfxDynamicState {
    EzGfxCullMode cull_mode;
    EzGfxFrontFace front_face;
    EzGfxPrimitiveType primitive_type;
    EzGfxBlendMode blend_mode;
} EzGfxDynamicState;

/**
 * EzGfxDrawIndexedCommand:
 * @index_count: Number of indices.
 * @instance_count: Number of instances.
 * @first_index: First index.
 * @vertex_offset: Signed vertex offset.
 * @first_instance: First instance.
 *
 * One indexed indirect draw command.
 */
typedef struct EzGfxDrawIndexedCommand {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t first_instance;
} EzGfxDrawIndexedCommand;

/** One host-polled runtime outcome. Correlation IDs are context-local and nonzero. */
typedef struct EzGfxRuntimeRecord {
    uint64_t correlation_id;
    uint64_t resource;
    EzGfxBackend backend;
    EzGfxRuntimePhase phase;
    EzGfxResult status;
    uint8_t _padding[5];
} EzGfxRuntimeRecord;

/** One lossless typed upload transition. Poll until out_present is zero each frame. */
typedef struct EzGfxUploadEvent {
    uint64_t resource;
    EzGfxUploadResourceKind resource_kind;
    EzGfxUploadStatus status;
    EzGfxResult error;
    uint8_t _padding[5];
} EzGfxUploadEvent;

/** One bounded local diagnostic with its causal runtime record. */
typedef struct EzGfxDiagnostic {
    EzGfxRuntimeRecord record;
    EzGfxDiagnosticLevel level;
    uint8_t _padding[7];
} EzGfxDiagnostic;
/**
 * EzGfxEventKind: selects the valid payload of EzGfxEvent.
 */
typedef uint8_t EzGfxEventKind;
enum {
    EzGfxEventKind_Upload = 1,
    EzGfxEventKind_Runtime = 2,
    EzGfxEventKind_Diagnostic = 3,
    EzGfxEventKind_ObservationsDropped = 4,
    EzGfxEventKind_Readback = 5,
    EzGfxEventKind_Snapshot = 6,
};

/**
 * EzGfxEvent: one tagged graphics event for the registered context callback.
 * Only the payload selected by kind is valid. Readback bytes are borrowed for
 * the callback invocation and must not be retained afterwards.
 */
typedef struct EzGfxEvent {
    EzGfxEventKind kind;
    uint8_t _pad_kind[7];
    EzGfxUploadEvent upload;
    EzGfxRuntimeRecord record;
    uint8_t level;
    uint8_t _pad_level[7];
    uint64_t dropped;
    EzGfxReadbackRequest readback_request_id;
    EzGfxTexture readback_texture;
    uint32_t readback_width;
    uint32_t readback_height;
    size_t readback_byte_count;
    const uint8_t *readback_bytes;
} EzGfxEvent;

/**
 * EzGfxEventCallback: context event callback invoked on the owner thread at
 * graphics safe points. `event` borrows its payload for the invocation only.
 * `user_data` is the registration value. Callbacks must not reenter graphics
 * operations or unwind; a reentrant dispatch attempt fails and a panic
 * unregisters the callback.
 */
typedef void (*EzGfxEventCallback)(const EzGfxEvent *event, void *user_data);

/**
 * EzGfxByteBuffer:
 * @length: Number of bytes in data.
 * @data (nullable) (array length=length): Byte range; nullable only when length is zero.
 *
 * Pointer-plus-length byte range for ABI consumers.
 */
typedef struct EzGfxByteBuffer {
    size_t length;
    const uint8_t * data;
} EzGfxByteBuffer;

/**
 * EzGfxHandleParts:
 * @context_slot: Context arena slot.
 * @context_generation: Context slot generation.
 * @child_slot: Child-resource arena slot; zero for a context handle.
 * @child_generation: Child-resource slot generation; zero for a context handle.
 * @is_context: Non-zero when the handle refers directly to a context.
 *
 * Decoded fields of an opaque packed handle.
 */
typedef struct EzGfxHandleParts {
    uint32_t context_slot;
    uint32_t context_generation;
    uint32_t child_slot;
    uint32_t child_generation;
    uint8_t is_context;
    uint8_t _padding[3];
} EzGfxHandleParts;

/**
 * ez_gfx_abi_version:
 *
 * Returns: (transfer none): ABI version; no context is required.
 */
uint32_t ez_gfx_abi_version(void);

/**
 * ez_gfx_print_error:
 * @result: EzGfxResult byte to describe; unknown values produce "unknown error".
 * @buffer: (array length=capacity) (nullable): Caller-owned UTF-8 output buffer.
 * @capacity: Writable bytes in @buffer, or zero with a null buffer to query.
 * @out_required: (out): Receives required bytes including the trailing NUL.
 *
 * Insufficient capacity returns EzGfxResult_InvalidArgument without modifying
 * @buffer. The returned message is stable for the ABI revision.
 *
 * Returns: EzGfxResult_Ok, or EzGfxResult_InvalidArgument for invalid pointers,
 * capacity, or insufficient storage.
 */
EzGfxResult ez_gfx_print_error(
    EzGfxResult result,
    char *buffer,
    size_t capacity,
    size_t *out_required)
    EZ_GFX_ACCESS(write_only, 2, 3)
    EZ_GFX_ACCESS(write_only, 4);

/**
 * ez_gfx_context_create:
 * @desc (in) (not nullable): Context creation options.
 * @out_context (out caller-allocates): Receives the opaque context handle.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or a creation error.
 */
EzGfxResult ez_gfx_context_create(const EzGfxContextDesc * desc, EzGfxContext * out_context) EZ_GFX_ACCESS(write_only, 2);

/** Creates a context for an explicit native backend. */
EzGfxResult ez_gfx_context_create_backend(const EzGfxBackendContextDesc * desc, EzGfxContext * out_context) EZ_GFX_ACCESS(write_only, 2);
/**
 * ez_gfx_adapter_count:
 * @out_count (out caller-allocates): Receives the number of enumerated adapters.
 *
 * Counts adapters without creating anything; no context is required. Requires ABI 26.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or EzGfxResult_InvalidArgument.
 */
EzGfxResult ez_gfx_adapter_count(uint32_t * out_count) EZ_GFX_ACCESS(write_only, 1);
/**
 * ez_gfx_adapters_query:
 * @allow_software: Non-zero admits software-class adapters in the diagnosis; zero or one.
 * @out_adapters (out caller-allocates) (array length=capacity): Receives at most @capacity entries; null if and only if @capacity is zero.
 * @capacity: Writable entry count of @out_adapters; zero queries the total with a null buffer.
 * @out_written (out caller-allocates): Receives entries written, or the total when @capacity is zero.
 *
 * Enumerates adapters with admission diagnostics without creating anything; no context is required. Requires ABI 26.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or EzGfxResult_InvalidArgument.
 */
EzGfxResult ez_gfx_adapters_query(uint8_t allow_software, EzGfxAdapterInfo * out_adapters, uint32_t capacity, uint32_t * out_written) EZ_GFX_ACCESS(write_only, 2, 3) EZ_GFX_ACCESS(write_only, 4);

/**
 * ez_gfx_context_wait_idle:
 * @context (not nullable): Owning context.
 *
 * Waits without destroying resources. Returns: (transfer none): EzGfxResult_Ok or an invalid-context, not-ready, native-failure, or device-loss result.
 */
EzGfxResult ez_gfx_context_wait_idle(EzGfxContext context);

/**
 * ez_gfx_context_destroy:
 * @context: Context to destroy on its creator thread; null, stale, repeated, and wrong-thread calls are ignored.
 *
 * Waits for an initialized device to become idle, then destroys every resource owned by the context. A context without a device is destroyed without waiting. Cleanup is terminal once it begins.
 * Returns: (transfer none): No return value; wait or release failures cannot be reported through the stable ABI.
 */
void ez_gfx_context_destroy(EzGfxContext context);

/**
 * ez_gfx_surface_create:
 * @desc (in) (not nullable): Window and initial extent.
 * @out_surface (out caller-allocates): Receives the opaque surface handle.
 * @context (not nullable): Context that owns the surface.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or a validation/native error.
 */
EzGfxResult ez_gfx_surface_create(const EzGfxSurfaceDesc * desc, EzGfxSurface * out_surface, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2);

/**
 * ez_gfx_context_init_device:
 * @surface: Surface owned by context.
 * @context (not nullable): Owning context.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or a native failure.
 */
EzGfxResult ez_gfx_context_init_device(EzGfxSurface surface, EzGfxContext context);

/**
 * ez_gfx_surface_resize:
 * @surface: Surface to resize.
 * @width: New width; zero is valid only with zero height for minimized state.
 * @height: New height; zero is valid only with zero width for minimized state.
 * @context (not nullable): Owning context.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok, EzGfxResult_NotReady, or an error.
 */
EzGfxResult ez_gfx_surface_resize(EzGfxSurface surface, uint32_t width, uint32_t height, EzGfxContext context);

/**
 * ez_gfx_surface_get_extent:
 * @surface: Surface to query.
 * @out_width (out caller-allocates): Receives width.
 * @out_height (out caller-allocates): Receives height.
 * @context (not nullable): Owning context.
 *
 * Returns: (transfer none): Returns EzGfxResult_NotReady while minimized.
 */
EzGfxResult ez_gfx_surface_get_extent(EzGfxSurface surface, uint32_t * out_width, uint32_t * out_height, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);

/**
 * ez_gfx_surface_resize_pending:
 * @surface: Surface to query.
 * @out_pending (out caller-allocates): Receives 0 or 1.
 * @context (not nullable): Owning context.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or an error.
 */
EzGfxResult ez_gfx_surface_resize_pending(EzGfxSurface surface, int32_t * out_pending, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2);

/** Updates the snapshot-cache preference; enabled must be zero or one. */
EzGfxResult ez_gfx_surface_set_snapshot_cache(EzGfxSurface surface, int32_t enabled, EzGfxContext context);
/** Loads a validated precompiled shader artifact; runtime packages contain no compiler. */
EzGfxResult ez_gfx_shader_load_artifact(const uint8_t *data, size_t data_size, EzGfxShader *out_shader, EzGfxContext context) EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);
/** Destroys a shader and its backend-native modules/libraries. */
void ez_gfx_shader_destroy(EzGfxShader shader, EzGfxContext context);
/** Decodes validated image/KTX2 bytes, including BasisLZ ETC1S and UASTC payloads, then starts an asynchronous GPU upload. */
EzGfxResult ez_gfx_texture_load(const uint8_t *data, size_t data_size, const EzGfxTextureDesc *desc, EzGfxTexture *out_texture, EzGfxContext context) EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(read_only, 3) EZ_GFX_ACCESS(write_only, 4);
/** Registers one concurrent custom decoder for a source code in 128..=255. */
EzGfxResult ez_gfx_texture_decoder_register(EzGfxSourceTextureFormat source_format, EzGfxTextureDecoderCallback callback, EzGfxTextureDecoderReleaseCallback release, void *user_data);
/** Unregisters a custom decoder; already accepted requests retain their callback. */
EzGfxResult ez_gfx_texture_decoder_unregister(EzGfxSourceTextureFormat source_format);
/** Copies and asynchronously uploads one validated texture sub-rectangle. */
EzGfxResult ez_gfx_update_texture_region(EzGfxTexture texture, const EzGfxTextureRegionDesc *desc, EzGfxContext context) EZ_GFX_ACCESS(read_only, 2);
/** Returns monotonic context-wide asynchronous texture pipeline counters. */
EzGfxResult ez_gfx_texture_get_upload_telemetry(EzGfxTextureUploadTelemetry *out_telemetry, EzGfxContext context) EZ_GFX_ACCESS(write_only, 1);
/** Cancels a texture before or after transfer-worker admission without waiting for GPU completion. */
EzGfxResult ez_gfx_texture_cancel(EzGfxTexture texture, EzGfxContext context);
/** Returns the stable bindless index once upload completion makes the texture resident. */
EzGfxResult ez_gfx_texture_get_binding(EzGfxTexture texture, uint32_t *out_binding, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2);
/** Reports completed mip residency and the immutable decoded mip count; resident may be zero while transfers are pending. */
EzGfxResult ez_gfx_texture_get_residency(EzGfxTexture texture, uint32_t *out_resident_mips, uint32_t *out_total_mips, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Sets the contiguous coarse mip count exposed through the stable binding; returns NotReady while required uploads or safe descriptor rewrites remain pending. */
EzGfxResult ez_gfx_texture_set_residency(EzGfxTexture texture, uint32_t resident_mips, EzGfxContext context);
/** Invalidates the handle and releases owned GPU storage. */
void ez_gfx_texture_unload(EzGfxTexture texture, EzGfxContext context);
/** Creates a managed render target from a declaration and explicit extents; depth, storage, and multisample stay unsupported. */
EzGfxResult ez_gfx_render_target_create(const EzGfxRenderTargetDesc *desc, uint32_t width, uint32_t height, EzGfxRenderTarget *out_target, EzGfxContext context) EZ_GFX_ACCESS(read_only, 1) EZ_GFX_ACCESS(write_only, 4);
/** Destroys a render target and clears any bound override; stale handles are ignored. */
void ez_gfx_render_target_destroy(EzGfxRenderTarget target, EzGfxContext context);
/** Reports the resolved storage format code of a live render target. */
EzGfxResult ez_gfx_render_target_get_format(EzGfxRenderTarget target, uint8_t *out_format, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2);
/** Reports the extents of a live render target. */
EzGfxResult ez_gfx_render_target_get_extent(EzGfxRenderTarget target, uint32_t *out_width, uint32_t *out_height, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Reports the stored clear value: one zero-or-one flag plus four color components. */
EzGfxResult ez_gfx_render_target_get_clear(EzGfxRenderTarget target, uint8_t *out_use_clear, float *out_color, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Probes whether one format admits a sampled color target at the given sample count; the status is the answer. */
EzGfxResult ez_gfx_render_target_probe_format(uint8_t format, uint8_t samples, EzGfxContext context);
/** Begins frame recording against a managed render target. */
EzGfxResult ez_gfx_render_target_frame_begin(EzGfxContext context, EzGfxRenderTarget target, EzGfxFrame *out_frame) EZ_GFX_ACCESS(write_only, 3);
/** Begins one presented-surface frame and returns its explicit owner handle. */
EzGfxResult ez_gfx_frame_begin(EzGfxContext context, EzGfxSurface surface, EzGfxFrame *out_frame) EZ_GFX_ACCESS(write_only, 3);

/** Acquires a context-owned one-frame counter buffer with an explicit element stride. */
EzGfxResult ez_gfx_counter_buffer_acquire(uint32_t element_size, uint32_t element_count, const char *debug_name, size_t debug_name_length, EzGfxCounterBuffer *out_buffer, EzGfxContext context) EZ_GFX_ACCESS(read_only, 3, 4) EZ_GFX_ACCESS(write_only, 5);
/** Writes a contiguous draw-command range before a counter buffer is consumed. */
EzGfxResult ez_gfx_counter_buffer_write_draws(EzGfxCounterBuffer buffer, uint32_t start_index, const EzGfxDrawIndexedCommand *commands, uint32_t command_count, EzGfxContext context) EZ_GFX_ACCESS(read_only, 3, 4);
/** Publishes a CPU-known count before consumption by a frame. */
EzGfxResult ez_gfx_counter_buffer_publish_count(EzGfxCounterBuffer buffer, uint32_t count, EzGfxContext context);
/** Releases an unconsumed counter buffer; zero and stale values are ignored. */
void ez_gfx_counter_buffer_release(EzGfxCounterBuffer buffer, EzGfxContext context);
/** Records a validated graphics pipeline and indexed-indirect draw, consuming the counter buffer into this frame. */
EzGfxResult ez_gfx_render_add_vertex_pipeline(EzGfxShader shader, EzGfxCounterBuffer buffer, const EzGfxBinding *bindings, uint32_t binding_count, const EzGfxDynamicState *dynamic_state, const void *push_constants, uint32_t push_constant_size, EzGfxFrame frame) EZ_GFX_ACCESS(read_only, 3, 4) EZ_GFX_ACCESS(read_only, 6, 7);
/** Records a validated compute pipeline dispatch. */
EzGfxResult ez_gfx_render_add_compute_pipeline(EzGfxShader shader, uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z, const EzGfxBinding *bindings, uint32_t binding_count, const void *push_constants, uint32_t push_constant_size, EzGfxFrame frame) EZ_GFX_ACCESS(read_only, 5, 6) EZ_GFX_ACCESS(read_only, 7, 8);
/** Compiles and enqueues a texture-readback graph node and returns its callback correlator. */
EzGfxResult ez_gfx_graph_enqueue_texture_readback(EzGfxTexture texture, EzGfxFrame frame, EzGfxReadbackRequest *out_request_id) EZ_GFX_ACCESS(write_only, 3);
/** Consumes a live frame, submitting it and presenting surface frames. */
EzGfxResult ez_gfx_frame_end(EzGfxFrame frame);
/** Consumes a live frame without submitting it. */
EzGfxResult ez_gfx_frame_abort(EzGfxFrame frame);
/** Replaces the context event callback and delivers pending events. Null clears. */
EzGfxResult ez_gfx_callback_register(EzGfxContext context, EzGfxEventCallback callback, void *user_data);
/** Creates an auto-growing named vertex heap. */
EzGfxResult ez_gfx_vertex_heap_create(const char *name, size_t name_length, uint64_t stride, EzGfxVertexHeap *out_heap, EzGfxContext context) EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 4);

/** Destroys the vertex heap selected by its typed handle. */
void ez_gfx_vertex_heap_destroy(EzGfxVertexHeap heap, EzGfxContext context);


/** Uploads u32 indices and returns an owner-validated allocation handle. */
EzGfxResult ez_gfx_vertex_upload_indices(const void * data, uint32_t count, EzGfxIndexAllocation * out_allocation, EzGfxContext context) EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);

/** Uploads vertices to a typed heap and returns an owner-validated allocation handle. */
EzGfxResult ez_gfx_vertex_upload(EzGfxVertexHeap heap, const void *data, uint32_t element_count, uint64_t element_size, EzGfxVertexAllocation *out_allocation, EzGfxContext context) EZ_GFX_ACCESS(read_only, 2, 3) EZ_GFX_ACCESS(write_only, 5);
/** Queries the live range represented by a vertex allocation. */
EzGfxResult ez_gfx_vertex_allocation_get_range(EzGfxVertexAllocation allocation, uint32_t * out_first, uint32_t * out_count, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Queries the live range represented by an index allocation. */
EzGfxResult ez_gfx_index_allocation_get_range(EzGfxIndexAllocation allocation, uint32_t * out_first, uint32_t * out_count, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Removes one vertex allocation using its recorded heap owner after establishing GPU safety. */
EzGfxResult ez_gfx_vertex_allocation_remove(EzGfxVertexAllocation allocation, EzGfxContext context);
/** Removes one index allocation after establishing GPU safety. */
EzGfxResult ez_gfx_index_allocation_remove(EzGfxIndexAllocation allocation, EzGfxContext context);

/** Acquires a context-owned one-frame buffer before frame recording. */
EzGfxResult ez_gfx_buffer_acquire(uint32_t element_size, uint32_t element_count, const char *debug_name, size_t debug_name_length, EzGfxBuffer *out_buffer, EzGfxContext context) EZ_GFX_ACCESS(read_only, 3, 4) EZ_GFX_ACCESS(write_only, 5);

/** Copies a validated element range before the buffer is consumed. */
EzGfxResult ez_gfx_buffer_write(EzGfxBuffer buffer, uint32_t start_index, const void *data, uint32_t element_count, uint32_t element_size, EzGfxContext context) EZ_GFX_ACCESS(read_only, 3, 4);

/** Releases an unconsumed buffer; zero and stale values are ignored. */
void ez_gfx_buffer_release(EzGfxBuffer buffer, EzGfxContext context);

/**
 * ez_gfx_surface_destroy:
 * @surface: Surface to destroy.
 * @context (not nullable): Owning context.
 *
 * Returns: (transfer none): No return value; null handles are ignored.
 */
void ez_gfx_surface_destroy(EzGfxSurface surface, EzGfxContext context);

/**
 * ez_gfx_handle_inspect:
 * @handle: Opaque packed context or resource handle.
 * @out_parts (out caller-allocates): Receives the decoded slot and generation fields.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or EzGfxResult_InvalidContext.
 */
EzGfxResult ez_gfx_handle_inspect(uint64_t handle, EzGfxHandleParts * out_parts) EZ_GFX_ACCESS(write_only, 2);

/**
 * ez_gfx_semantic_id:
 * @name (in) (not nullable): Exact canonical semantic-name bytes; no terminator is read.
 * @length: Name byte count in the inclusive range 1..255.
 * @out_id (out caller-allocates): Receives the 16-byte semantic identifier.
 *
 * A semantic name is ASCII dot-separated. Every non-empty segment starts with
 * an ASCII letter and continues with only ASCII letters, digits, or underscores.
 * Embedded NUL, empty segments, non-ASCII bytes, and all other shapes are invalid.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or EzGfxResult_InvalidArgument.
 */
EzGfxResult ez_gfx_semantic_id(const uint8_t * name, size_t length, uint8_t * out_id) EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);

#ifdef __cplusplus
}
#endif

#endif
