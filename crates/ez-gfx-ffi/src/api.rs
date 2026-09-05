use core::ffi::c_void;

/// Opaque 64-bit identifier used for graphics resources across the C ABI.
pub type EzGfxHandle = u64;
/// Opaque identifier for a graphics context.
pub type EzGfxContext = EzGfxHandle;
/// Opaque identifier for a presentation surface.
pub type EzGfxSurface = EzGfxHandle;
/// Opaque identifier for a compiled shader resource.
pub type EzGfxShader = EzGfxHandle;
/// Opaque identifier for a buffer containing indirect draw commands.
pub type EzGfxIndirectBuffer = EzGfxHandle;
/// Opaque identifier for a shader-accessible structured buffer.
pub type EzGfxStructuredBuffer = EzGfxHandle;
/// Opaque identifier for a sampled texture resource.
pub type EzGfxTexture = EzGfxHandle;
/// Opaque identifier for a render-target resource.
pub type EzGfxRenderTarget = EzGfxHandle;

pub use ez_gfx::EzGfxResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// C ABI result enumeration.
pub enum EzGfxTextureError {
    /// Indicates that no texture error occurred.
    None = 0,
    /// Indicates that the context identifier is invalid or stale.
    InvalidContext = 1,
    /// Indicates that one or more texture arguments are invalid.
    InvalidArguments = 2,
    /// Indicates that the requested texture format cannot be used.
    UnsupportedFormat = 3,
    /// Indicates that no texture identifier slots remain.
    OutOfTextureHandles = 4,
    /// Indicates that texture allocation exhausted available memory.
    OutOfMemory = 5,
    /// Indicates that the source image could not be decoded.
    DecodeFailed = 6,
    /// Indicates that a Vulkan texture request failed.
    VulkanFailed = 7,
    /// Indicates that no texture worker is available to process the request.
    WorkerUnavailable = 8,
    /// Indicates that the requested texture could not be found.
    NotFound = 9,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Describes graphics-context creation parameters for the C ABI.
pub struct EzGfxContextDesc {
    /// Enables graphics-backend debugging when nonzero.
    pub enable_debug: u8,
    /// Enables graphics API validation when nonzero.
    pub enable_validation: u8,
    /// Selects the native platform used to create presentation surfaces.
    pub surface_platform: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Describes graphics-context creation parameters with an explicit backend selector.
pub struct EzGfxBackendContextDesc {
    /// Enables graphics-backend debugging when nonzero.
    pub enable_debug: u8,
    /// Enables graphics API validation when nonzero.
    pub enable_validation: u8,
    /// Selects the native platform used to create presentation surfaces.
    pub surface_platform: u8,
    /// Selects the graphics backend by its C ABI numeric code.
    pub backend: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Describes a native presentation surface and its initial extent.
pub struct EzGfxSurfaceDesc {
    /// Points to the platform-native window object.
    pub window: *mut c_void,
    /// Points to the platform-native display or connection object.
    pub display: *mut c_void,
    /// Identifies the native window-system platform by its C ABI numeric code.
    pub platform: u8,
    /// Specifies the initial surface width in pixels.
    pub width: u32,
    /// Specifies the initial surface height in pixels.
    pub height: u32,
    /// Enables caching of presented surface snapshots when nonzero.
    pub cache_presented_snapshots: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Describes shader source and stage entry points for resource creation.
pub struct EzGfxShaderDesc {
    /// Points to exactly `path_length` UTF-8 bytes.
    pub path: *const u8,
    /// Specifies the nonzero byte length available through `path`.
    pub path_length: usize,
    /// Points to exactly `vertex_entry_length` UTF-8 bytes when present.
    pub vertex_entry: *const u8,
    /// Specifies the vertex-entry byte length, or zero when absent.
    pub vertex_entry_length: usize,
    /// Points to exactly `fragment_entry_length` UTF-8 bytes when present.
    pub fragment_entry: *const u8,
    /// Specifies the fragment-entry byte length, or zero when absent.
    pub fragment_entry_length: usize,
    /// Points to exactly `compute_entry_length` UTF-8 bytes when present.
    pub compute_entry: *const u8,
    /// Specifies the compute-entry byte length, or zero when absent.
    pub compute_entry_length: usize,
    /// Identifies the shader kind by its C ABI numeric code.
    pub kind: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Describes texture dimensions, formats, mipmapping, sampling, and labeling.
pub struct EzGfxTextureDesc {
    /// Identifies the source texel format by its C ABI numeric code.
    pub source_format: u8,
    /// Identifies the GPU destination format by its C ABI numeric code.
    pub destination_format: u8,
    /// Specifies the base mip level width in pixels.
    pub width: u32,
    /// Specifies the base mip level height in pixels.
    pub height: u32,
    /// Specifies the number of mip levels in the texture.
    pub mip_count: u32,
    /// Requests automatic mip generation when nonzero.
    pub generate_mips: u8,
    /// Selects the minification filter by its C ABI numeric code.
    pub min_filter: u8,
    /// Selects the magnification filter by its C ABI numeric code.
    pub mag_filter: u8,
    /// Specifies the maximum anisotropic filtering ratio.
    pub max_anisotropy: f32,
    /// Selects the texture-addressing mode for the U coordinate.
    pub address_mode_u: u8,
    /// Selects the texture-addressing mode for the V coordinate.
    pub address_mode_v: u8,
    /// Selects the texture-addressing mode for the W coordinate.
    pub address_mode_w: u8,
    /// Points to exactly `debug_label_length` UTF-8 bytes when present.
    pub debug_label: *const u8,
    /// Specifies the debug-label byte length, or zero when absent.
    pub debug_label_length: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Decoded bytes for one custom-decoder mip, retained until the release callback.
pub struct EzGfxDecodedTextureMip {
    /// Mip width in texels.
    pub width: u32,
    /// Mip height in texels.
    pub height: u32,
    /// Readable bytes retained until the release callback.
    pub data: *const u8,
    /// Exact byte length at `data`.
    pub data_size: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Custom-decoder output copied before its release callback is invoked.
pub struct EzGfxDecodedTexture {
    /// Destination-format code; `Auto` is invalid for decoded output.
    pub format: u8,
    /// Number of entries at `mips`.
    pub mip_count: u32,
    /// Readable array retained until the release callback.
    pub mips: *const EzGfxDecodedTextureMip,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Borrowed bytes and destination rectangle for one asynchronous texture update.
pub struct EzGfxTextureRegionDesc {
    /// Destination mip level.
    pub mip_level: u32,
    /// Destination X offset in texels.
    pub x: u32,
    /// Destination Y offset in texels.
    pub y: u32,
    /// Region width in texels.
    pub width: u32,
    /// Region height in texels.
    pub height: u32,
    /// Readable tightly packed texel or compressed-block bytes.
    pub data: *const u8,
    /// Exact byte length at `data`.
    pub data_size: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Monotonic context-wide asynchronous texture pipeline counters.
pub struct EzGfxTextureUploadTelemetry {
    /// Aggregate CPU decode/transcode/mip-generation time.
    pub decode_microseconds: u64,
    /// Aggregate bytes admitted to native staging.
    pub staging_bytes: u64,
    /// Aggregate transfer-owner queue latency.
    pub queue_latency_microseconds: u64,
    /// Aggregate transfer-to-graphics handoff latency.
    pub handoff_latency_microseconds: u64,
}

/// Custom image decoder invoked concurrently by texture workers.
pub type EzGfxTextureDecoderCallback = Option<
    unsafe extern "C" fn(
        data: *const u8,
        data_size: usize,
        compression_support: u8,
        out_texture: *mut EzGfxDecodedTexture,
        user_data: *mut c_void,
    ) -> EzGfxResult,
>;

/// Releases a custom decoder result after ez-gfx copies every mip.
pub type EzGfxTextureDecoderReleaseCallback =
    Option<unsafe extern "C" fn(texture: *const EzGfxDecodedTexture, user_data: *mut c_void)>;
#[derive(Clone, Copy)]
#[repr(C)]
/// Associates a named shader binding with buffer and render-target resources.
pub struct EzGfxBinding {
    /// Points to exactly `name_length` UTF-8 shader-binding-name bytes.
    pub name: *const u8,
    /// Specifies the nonzero byte length available through `name`.
    pub name_length: usize,
    /// Identifies the structured buffer assigned to the binding.
    pub structured: EzGfxStructuredBuffer,
    /// Identifies the indirect-command buffer assigned to the binding.
    pub indirect: EzGfxIndirectBuffer,
    /// Identifies the render target assigned to the binding.
    pub render_target: EzGfxRenderTarget,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Encodes dynamic rasterization, topology, and blending state.
pub struct EzGfxDynamicState {
    /// Selects the polygon culling mode by its C ABI numeric code.
    pub cull_mode: u8,
    /// Selects which vertex winding is considered front-facing.
    pub front_face: u8,
    /// Selects the primitive topology by its C ABI numeric code.
    pub primitive_type: u8,
    /// Selects the color blending mode by its C ABI numeric code.
    pub blend_mode: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Encodes one indexed, optionally instanced draw command.
pub struct EzGfxDrawIndexedCommand {
    /// Specifies the number of indices drawn per instance.
    pub index_count: u32,
    /// Specifies the number of instances to draw.
    pub instance_count: u32,
    /// Specifies the starting index within the index buffer.
    pub first_index: u32,
    /// Specifies the signed offset added to each fetched vertex index.
    pub vertex_offset: i32,
    /// Specifies the first instance identifier used by the draw.
    pub first_instance: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Describes a contiguous read-only byte range across the C ABI.
pub struct EzGfxByteBuffer {
    /// Specifies the byte count available through `data`.
    pub length: usize,
    /// Points to the first byte of the range and may be null only when `length` is zero.
    pub data: *const u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
#[allow(
    clippy::pub_underscore_fields,
    reason = "Named public padding preserves ABI layout and zero-initialization for callers."
)]
/// Exposes the slot and generation components encoded in an opaque identifier.
pub struct EzGfxHandleParts {
    /// Identifies the context slot in the context arena.
    pub context_slot: u32,
    /// Records the generation of the context slot.
    pub context_generation: u32,
    /// Identifies the child-resource slot within the context.
    pub child_slot: u32,
    /// Records the generation of the child-resource slot.
    pub child_generation: u32,
    /// Is nonzero when the identifier refers directly to a context.
    pub is_context: u8,
    /// Reserves bytes that keep the C ABI layout stable.
    pub _padding: [u8; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
#[allow(
    clippy::pub_underscore_fields,
    reason = "Named public padding preserves ABI layout and zero-initialization for callers."
)]
/// Records correlated runtime progress for a graphics resource.
pub struct EzGfxRuntimeRecord {
    /// Identifies the request or event associated with this record.
    pub correlation_id: u64,
    /// Identifies the graphics resource associated with this record.
    pub resource: EzGfxHandle,
    /// Identifies the graphics backend by its C ABI numeric code.
    pub backend: u8,
    /// Identifies the runtime phase by its C ABI numeric code.
    pub phase: u8,
    /// Identifies the phase status by its C ABI numeric code.
    pub status: u8,
    /// Reserves bytes that keep the C ABI layout stable.
    pub _padding: [u8; 5],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
#[allow(
    clippy::pub_underscore_fields,
    reason = "Named public padding preserves ABI layout and zero-initialization for callers."
)]
/// Couples a runtime record with its diagnostic severity.
pub struct EzGfxDiagnostic {
    /// Contains the correlated runtime record being diagnosed.
    pub record: EzGfxRuntimeRecord,
    /// Identifies the diagnostic severity by its C ABI numeric code.
    pub level: u8,
    /// Reserves bytes that keep the C ABI layout stable.
    pub _padding: [u8; 7],
}
#[cfg(test)]
mod tests {
    use crate::{EzGfxResult, texture::sampler_address_from_abi};
    use ez_gfx::SamplerAddressMode;

    #[test]
    fn texture_address_discriminants_match_the_public_abi() {
        assert_eq!(sampler_address_from_abi(0), Ok(SamplerAddressMode::Repeat));
        assert_eq!(sampler_address_from_abi(1), Ok(SamplerAddressMode::Clamp));
        assert_eq!(
            sampler_address_from_abi(2),
            Err(EzGfxResult::InvalidArgument)
        );
    }
}
