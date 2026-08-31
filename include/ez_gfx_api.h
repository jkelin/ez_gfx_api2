#ifndef EZ_GFX_API_H
#define EZ_GFX_API_H

#include <stdint.h>
#include <stddef.h>

#define EZ_GFX_ABI_VERSION 18u

#if defined(__clang__)
#  if __has_attribute(access)
#    define EZ_GFX_ACCESS(...) __attribute__((access(__VA_ARGS__)))
#  else
#    define EZ_GFX_ACCESS(...)
#  endif
#  if __has_attribute(counted_by)
#    define EZ_GFX_COUNTED_BY(field) __attribute__((counted_by(field)))
#  else
#    define EZ_GFX_COUNTED_BY(field)
#  endif
#elif defined(__GNUC__) && (__GNUC__ >= 10)
#  define EZ_GFX_ACCESS(...) __attribute__((access(__VA_ARGS__)))
#  define EZ_GFX_COUNTED_BY(field)
#else
#  define EZ_GFX_ACCESS(...)
#  define EZ_GFX_COUNTED_BY(field)
#endif

/* ABI string contract: every const char* is UTF-8 and NUL-terminated for the duration of the call. */

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
 * EzGfxShader: Opaque packed u64 shader handle; child bits index the context identity arena and resolve only as Shader.
 */
typedef uint64_t EzGfxShader;
/**
 * EzGfxIndirectBuffer: Opaque packed u64 indirect-buffer handle; child bits index the context identity arena and resolve only as Indirect.
 */
typedef uint64_t EzGfxIndirectBuffer;
/**
 * EzGfxStructuredBuffer: Opaque packed u64 structured-buffer handle; child bits index the context identity arena and resolve only as Structured.
 */
typedef uint64_t EzGfxStructuredBuffer;
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
 * @EzGfxResult_NotReady: The operation is temporarily unavailable, such as a minimized surface.
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

/**
 * EzGfxSurfacePlatform:
 * @EzGfxSurfacePlatform_Win32: Win32 HWND and HINSTANCE handles.
 * @EzGfxSurfacePlatform_GLFW: GLFW native window handle.
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

/**
 * EzGfxSourceTextureFormat:
 * @EzGfxSourceTextureFormat_Rgb: Raw RGB pixels.
 * @EzGfxSourceTextureFormat_Rgba: Raw RGBA pixels.
 * @EzGfxSourceTextureFormat_Bmp: BMP image bytes.
 * @EzGfxSourceTextureFormat_Jpeg: JPEG image bytes.
 * @EzGfxSourceTextureFormat_Png: PNG image bytes.
 * @EzGfxSourceTextureFormat_Tga: TGA image bytes.
 * @EzGfxSourceTextureFormat_Ktx2: KTX2 image bytes.
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
 * @EzGfxTextureDestinationFormat_Rgba8Unorm: 8-bit normalized RGBA.
 *
 * Destination texture format.
 */
typedef uint8_t EzGfxTextureDestinationFormat;
enum {
    EzGfxTextureDestinationFormat_Rgba8Unorm = 0,
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
 * EzGfxContextDesc:
 * @enable_debug: Non-zero enables debug utilities.
 * @enable_validation: Non-zero enables validation layers.
 * @surface_platform: Value from EzGfxSurfacePlatform.
 *
 * Context creation options.
 */
typedef struct EzGfxContextDesc {
    uint8_t enable_debug;
    uint8_t enable_validation;
    EzGfxSurfacePlatform surface_platform;
} EzGfxContextDesc;

/** Backend-selecting context creation options. */
typedef struct EzGfxBackendContextDesc {
    uint8_t enable_debug;
    uint8_t enable_validation;
    EzGfxSurfacePlatform surface_platform;
    EzGfxBackend backend;
} EzGfxBackendContextDesc;

/**
 * EzGfxSurfaceDesc:
 * @window (not nullable): Native HWND or CAMetalLayer pointer.
 * @display (nullable): Native HINSTANCE; null for Metal and permitted for GLFW.
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
 * @path (not nullable): UTF-8, NUL-terminated shader path.
 * @vertex_entry (nullable): Optional UTF-8 vertex entry point.
 * @fragment_entry (nullable): Optional UTF-8 fragment entry point.
 * @compute_entry (nullable): Optional UTF-8 compute entry point.
 * @kind: Value from EzGfxShaderKind.
 *
 * Shader source and entry-point metadata.
 */
typedef struct EzGfxShaderDesc {
    const char * path;
    const char * vertex_entry;
    const char * fragment_entry;
    const char * compute_entry;
    EzGfxShaderKind kind;
} EzGfxShaderDesc;

/**
 * EzGfxTextureDesc:
 * @source_format: Value from EzGfxSourceTextureFormat.
 * @destination_format: Value from EzGfxTextureDestinationFormat.
 * @width: Decoded width for raw pixels.
 * @height: Decoded height for raw pixels.
 * @mip_count: Number of mip levels, or zero for decoder defaults.
 * @generate_mips: Non-zero requests mip generation.
 * @min_filter: Value from EzGfxTextureFilter.
 * @mag_filter: Value from EzGfxTextureFilter.
 * @max_anisotropy: Requested anisotropy; zero uses the default.
 * @address_mode_u: Value from EzGfxTextureAddressMode.
 * @address_mode_v: Value from EzGfxTextureAddressMode.
 * @address_mode_w: Value from EzGfxTextureAddressMode.
 * @debug_label (nullable): Optional UTF-8 debug label.
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
} EzGfxTextureDesc;

/**
 * EzGfxBinding:
 * @name (not nullable): UTF-8 shader binding name.
 * @structured: Optional structured buffer handle.
 * @indirect: Optional indirect buffer handle.
 * @render_target: Optional render-target handle.
 *
 * Shader resource binding.
 */
typedef struct EzGfxBinding {
    const char * name;
    EzGfxStructuredBuffer structured;
    EzGfxIndirectBuffer indirect;
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

/** One bounded local diagnostic with its causal runtime record. */
typedef struct EzGfxDiagnostic {
    EzGfxRuntimeRecord record;
    EzGfxDiagnosticLevel level;
    uint8_t _padding[7];
} EzGfxDiagnostic;

/**
 * EzGfxByteBuffer:
 * @length: Number of bytes in data.
 * @data (nullable) (array length=length): Byte range; nullable only when length is zero.
 *
 * Pointer-plus-length byte range for ABI consumers.
 */
typedef struct EzGfxByteBuffer {
    size_t length;
    const uint8_t * data EZ_GFX_COUNTED_BY(length);
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
 * ez_gfx_context_wait_idle:
 * @context (not nullable): Owning context.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or EzGfxResult_InvalidContext.
 */
EzGfxResult ez_gfx_context_wait_idle(EzGfxContext context);

/**
 * ez_gfx_context_destroy:
 * @context: Context to destroy; null is ignored.
 *
 * Returns: (transfer none): No return value; null handles are ignored.
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
/** Returns the stable bindless index once upload completion makes the texture resident. */
EzGfxResult ez_gfx_texture_get_binding(EzGfxTexture texture, uint32_t *out_binding, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2);
/** Reports completed mip residency and the immutable decoded mip count; resident may be zero while transfers are pending. */
EzGfxResult ez_gfx_texture_get_residency(EzGfxTexture texture, uint32_t *out_resident_mips, uint32_t *out_total_mips, EzGfxContext context) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Invalidates the handle and releases owned GPU storage. */
void ez_gfx_texture_unload(EzGfxTexture texture, EzGfxContext context);
/** Begins and resets one presented-surface frame; use instead of ez_gfx_frame_begin. */
EzGfxResult ez_gfx_begin_render(EzGfxSurface surface, EzGfxContext context);
/** Begins one non-presented/headless frame; do not call both begin functions for one frame. */
EzGfxResult ez_gfx_frame_begin(EzGfxContext context);
/** Creates a bounded indexed-indirect command buffer. */
EzGfxResult ez_gfx_acquire_indirect(uint32_t capacity, const char *debug_name, EzGfxIndirectBuffer *out_indirect, EzGfxContext context) EZ_GFX_ACCESS(write_only, 3);
/** Writes one standard indexed indirect command. */
EzGfxResult ez_gfx_indirect_write_draw(EzGfxIndirectBuffer indirect, uint32_t index, const EzGfxDrawIndexedCommand *command, EzGfxContext context) EZ_GFX_ACCESS(read_only, 3);
/** Sets the submitted prefix of the indirect command buffer. */
EzGfxResult ez_gfx_indirect_set_draw_count(EzGfxIndirectBuffer indirect, uint32_t count, EzGfxContext context);
/** Invalidates an indirect handle; zero is ignored. */
void ez_gfx_indirect_release(EzGfxIndirectBuffer indirect, EzGfxContext context);
/** Records a validated graphics pipeline and indexed-indirect draw. */
EzGfxResult ez_gfx_render_add_vertex_pipeline(EzGfxShader shader, EzGfxIndirectBuffer indirect, const EzGfxBinding *bindings, uint32_t binding_count, const EzGfxDynamicState *dynamic_state, const void *push_constants, uint32_t push_constant_size, EzGfxContext context) EZ_GFX_ACCESS(read_only, 3, 4) EZ_GFX_ACCESS(read_only, 6, 7);
/** Records a validated compute pipeline dispatch. */
EzGfxResult ez_gfx_render_add_compute_pipeline(EzGfxShader shader, uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z, const EzGfxBinding *bindings, uint32_t binding_count, const void *push_constants, uint32_t push_constant_size, EzGfxContext context) EZ_GFX_ACCESS(read_only, 5, 6) EZ_GFX_ACCESS(read_only, 7, 8);
/** Compiles and enqueues a texture-readback graph node for the active frame. */
EzGfxResult ez_gfx_graph_enqueue_texture_readback(EzGfxTexture texture, EzGfxContext context);
/** Submits the active recording without native presentation and completes any recorded readback. */
EzGfxResult ez_gfx_frame_submit(EzGfxContext context);
/** Polls one bounded runtime event. out_present is zero when no event is queued; out_dropped reports overflow since the prior poll. */
EzGfxResult ez_gfx_poll_runtime_event(EzGfxRuntimeRecord *out_record, uint8_t *out_present, uint64_t *out_dropped, EzGfxContext context) EZ_GFX_ACCESS(write_only, 1) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Polls one bounded local diagnostic. out_present is zero when none is queued; out_dropped reports overflow since the prior poll. */
EzGfxResult ez_gfx_poll_diagnostic(EzGfxDiagnostic *out_diagnostic, uint8_t *out_present, uint64_t *out_dropped, EzGfxContext context) EZ_GFX_ACCESS(write_only, 1) EZ_GFX_ACCESS(write_only, 2) EZ_GFX_ACCESS(write_only, 3);
/** Submits the active frame, then presents; does not present if submission fails. */
EzGfxResult ez_gfx_finish_render(EzGfxContext context);
/** Copies a readback only after an enqueued readback's submission completes; capacity zero queries required size. */
EzGfxResult ez_gfx_frame_readback(uint8_t *data, size_t capacity, size_t *out_size, EzGfxContext context) EZ_GFX_ACCESS(write_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);
/** Creates a named device-local vertex heap. */
EzGfxResult ez_gfx_vertex_heap_create(const char * name, uint64_t capacity, uint64_t stride, EzGfxContext context);

/** Destroys a named vertex heap; invalid names are ignored. */
void ez_gfx_vertex_heap_destroy(const char * name, EzGfxContext context);

/** Creates the context's device-local u32 index heap. */
EzGfxResult ez_gfx_index_heap_create(uint64_t capacity, const char * debug_name, EzGfxContext context);

/** Destroys the context's index heap. */
void ez_gfx_index_heap_destroy(EzGfxContext context);

/** Uploads u32 indices through pooled staging and returns the first index. */
EzGfxResult ez_gfx_vertex_upload_indices(const void * data, uint32_t count, uint32_t * out_start_index, EzGfxContext context) EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);

/** Uploads vertices through pooled staging and returns the first vertex. */
EzGfxResult ez_gfx_vertex_upload(const char * heap_name, const void * data, uint32_t element_count, uint64_t element_size, uint32_t * out_start_index, EzGfxContext context) EZ_GFX_ACCESS(read_only, 2, 3) EZ_GFX_ACCESS(write_only, 5);

/** Acquires a real mapped upload buffer owned by an initialized context. */
EzGfxResult ez_gfx_structured_acquire(uint32_t element_size, uint32_t element_count, const char * debug_name, EzGfxStructuredBuffer * out_structured, EzGfxContext context) EZ_GFX_ACCESS(write_only, 4);

/** Copies bytes into a mapped buffer and flushes non-coherent memory. */
EzGfxResult ez_gfx_structured_write(EzGfxStructuredBuffer structured, const void * data, uint64_t data_size, EzGfxContext context) EZ_GFX_ACCESS(read_only, 2, 3);

/** Destroys the native buffer and frees its allocation; null is ignored. */
void ez_gfx_structured_release(EzGfxStructuredBuffer structured, EzGfxContext context);

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
 * @name (in) (not nullable): UTF-8 name bytes.
 * @length: Number of bytes in name.
 * @out_id (out caller-allocates): Receives the 16-byte semantic identifier.
 *
 * Returns: (transfer none): Returns EzGfxResult_Ok or EzGfxResult_InvalidArgument.
 */
EzGfxResult ez_gfx_semantic_id(const uint8_t * name, size_t length, uint8_t * out_id) EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);

#ifdef __cplusplus
}
#endif

#endif
