use core::ffi::c_void;

/// Opaque 64-bit identifier used for graphics resources across the C ABI.
pub type EzGfxHandle = u64;
/// Opaque packed `u64` context handle; slot+1/generation occupy bits 0-39.
pub type EzGfxContext = EzGfxHandle;
/// Opaque packed `u64` surface handle; child bits index the context identity arena and resolve only as a surface.
pub type EzGfxSurface = EzGfxHandle;
/// Opaque generation-, kind-, owner-, state-, and frame-serial-validated live frame handle.
pub type EzGfxFrame = EzGfxHandle;
/// Stable process-unique identity for one enqueued readback request.
pub type EzGfxReadbackRequest = u64;
/// Opaque packed `u64` shader handle; child bits index the context identity arena and resolve only as a shader.
pub type EzGfxShader = EzGfxHandle;
/// Opaque packed `u64` one-frame counter-buffer handle; child bits resolve only as indirect commands.
pub type EzGfxCounterBuffer = EzGfxHandle;
/// Opaque packed `u64` one-frame buffer handle; child bits resolve only as structured data.
pub type EzGfxBuffer = EzGfxHandle;
/// Opaque named vertex heap whose owner and generation are validated.
pub type EzGfxVertexHeap = EzGfxHandle;
/// Opaque allocation within a named vertex heap whose owner and generation are validated.
pub type EzGfxVertexAllocation = EzGfxHandle;
/// Opaque allocation within the singleton index heap whose owner and generation are validated.
pub type EzGfxIndexAllocation = EzGfxHandle;
/// Opaque packed `u64` texture handle; child bits index the context identity arena and resolve only as a texture.
pub type EzGfxTexture = EzGfxHandle;
/// Opaque packed `u64` render-target handle; child bits index the context identity arena and resolve only as a render target.
pub type EzGfxRenderTarget = EzGfxHandle;

/// Stable C ABI result code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EzGfxResult {
    /// Operation completed successfully.
    Ok = 0,
    /// An argument violates the operation contract.
    InvalidArgument = 1,
    /// A context or resource handle is invalid or stale.
    InvalidContext = 2,
    /// The native graphics backend failed.
    NativeFailure = 3,
    /// Completion or output is not yet available.
    NotReady = 4,
    /// The requested capability is unavailable.
    Unsupported = 5,
    /// The graphics device was lost.
    DeviceLost = 6,
    /// Asynchronous scheduling or staging capacity is unavailable.
    QueueFull = 7,
    /// An asynchronous operation was cancelled before completion.
    Cancelled = 8,
}

impl From<ez_gfx::Error> for EzGfxResult {
    #[allow(
        clippy::match_same_arms,
        reason = "current variants stay explicit while the non-exhaustive facade retains a safe fallback"
    )]
    fn from(error: ez_gfx::Error) -> Self {
        match error {
            ez_gfx::Error::InvalidArgument => Self::InvalidArgument,
            ez_gfx::Error::InvalidContext => Self::InvalidContext,
            ez_gfx::Error::Lifecycle(
                ez_gfx::LifecycleError::DeviceLost | ez_gfx::LifecycleError::AlreadyLost,
            ) => Self::DeviceLost,
            ez_gfx::Error::Lifecycle(_) => Self::InvalidContext,
            ez_gfx::Error::NativeFailure => Self::NativeFailure,
            ez_gfx::Error::NotReady => Self::NotReady,
            ez_gfx::Error::Unsupported | ez_gfx::Error::Capability(_) => Self::Unsupported,
            ez_gfx::Error::DeviceLost => Self::DeviceLost,
            ez_gfx::Error::QueueFull => Self::QueueFull,
            ez_gfx::Error::Cancelled => Self::Cancelled,
            ez_gfx::Error::ReentrantCallback => Self::InvalidArgument,
            ez_gfx::Error::CallbackPanicked => Self::NativeFailure,
            _ => Self::NativeFailure,
        }
    }
}

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
    /// Async texture decode worker threads; zero selects the default topology.
    pub texture_decode_workers: u32,
    /// Explicit adapter requests; zero keeps default ranking, one selects by identity.
    pub adapter_count: u32,
    /// Exactly `adapter_count` explicit requests; null if and only if zero.
    pub adapter: *const EzGfxAdapterDesc,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Explicit adapter request selected by stable identity.
pub struct EzGfxAdapterDesc {
    /// Stable 128-bit adapter identity from enumeration.
    pub stable_id: [u8; 16],
    /// Whether a software-class adapter is acceptable; zero or one.
    pub allow_software: u8,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// C ABI adapter classification; discriminants mirror core `AdapterClass`.
pub enum EzGfxAdapterClass {
    /// Software renderer.
    Software = 0,
    /// Adapter with no stronger classification.
    Other = 1,
    /// Integrated GPU.
    Integrated = 2,
    /// Discrete GPU.
    Discrete = 3,
}
#[derive(Clone, Copy)]
#[repr(C)]
/// Enumerated adapter identity with admission diagnostics under one software policy.
pub struct EzGfxAdapterInfo {
    /// Stable 128-bit adapter identity from enumeration.
    pub stable_id: [u8; 16],
    /// Backend code from `EzGfxBackend`.
    pub backend: u8,
    /// Class code from `EzGfxAdapterClass`.
    pub adapter_class: u8,
    /// Whether the adapter passes admission under the queried policy; zero or one.
    pub admitted: u8,
    /// Whether software policy alone rejects the adapter; zero or one.
    pub software_rejected: u8,
    /// Count of unmet profile requirements; zero when admitted.
    pub error_count: u32,
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
    /// Async texture decode worker threads; zero selects the default topology.
    pub texture_decode_workers: u32,
    /// Explicit adapter requests; zero keeps default ranking, one selects by identity.
    pub adapter_count: u32,
    /// Exactly `adapter_count` explicit requests; null if and only if zero.
    pub adapter: *const EzGfxAdapterDesc,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// C ABI render-target storage format; discriminants mirror runtime `Format`.
pub enum EzGfxRenderTargetFormat {
    /// Four-channel 8-bit normalized RGBA format.
    Rgba8Unorm = 1,
    /// Four-channel 8-bit sRGB BGRA format.
    Bgra8Srgb = 2,
    /// Four-channel 16-bit floating-point RGBA format.
    Rgba16Float = 3,
    /// 32-bit floating-point depth format.
    Depth32Float = 4,
    /// BC7-compressed 8-bit normalized RGBA format.
    Bc7Unorm = 5,
    /// ASTC-compressed 4-by-4 texel 8-bit normalized RGBA format.
    Astc4x4Unorm = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// C ABI render-target access mode.
pub enum EzGfxRenderTargetUsage {
    /// Supports color-rendering output.
    Color = 0,
    /// Supports depth-rendering output; creation stays unsupported.
    Depth = 1,
    /// Supports storage access; creation stays unsupported.
    Storage = 2,
    /// Supports texture sampling; creation stays unsupported.
    Sampled = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Describes render-target name, usage, format candidates, and clear value.
pub struct EzGfxRenderTargetDesc {
    /// Points to exactly `name_length` UTF-8 bytes; required, 1..=255 bytes.
    pub name: *const u8,
    /// Specifies the target-name byte length.
    pub name_length: usize,
    /// Selects the access mode by its C ABI numeric code.
    pub usage: u8,
    /// Scale factor relative to the reference dimensions; finite and positive.
    pub relative_scale: f32,
    /// Requested multisample count; one of 1, 2, 4, or 8.
    pub samples: u8,
    /// Points to exactly `candidate_count` format codes; required, 1..=16 entries.
    pub candidate_formats: *const u8,
    /// Specifies the candidate format count.
    pub candidate_count: u32,
    /// Non-zero requires texture-sampling support; must be zero or one.
    pub sampleable: u8,
    /// Non-zero stores `clear_color`; must be zero or one.
    pub use_clear: u8,
    /// Color clear value read only when `use_clear` is non-zero.
    pub clear_color: [f32; 4],
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
    /// Identifies the buffer assigned to the binding.
    pub buffer: EzGfxBuffer,
    /// Identifies the counter command buffer assigned to the binding.
    pub counter_buffer: EzGfxCounterBuffer,
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
/// One lossless typed upload transition.
pub struct EzGfxUploadEvent {
    /// Opaque texture, vertex-allocation, or index-allocation handle.
    pub resource: EzGfxHandle,
    /// Resource kind: 1 texture, 2 vertex allocation, 3 index allocation.
    pub resource_kind: u8,
    /// Status: 1 source staged, 2 device ready, 3 failed, 4 cancelled.
    pub status: u8,
    /// `RuntimeStatus` code for failed events, zero otherwise.
    pub error: u8,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// Discriminates the payload carried by [`EzGfxEvent`].
pub enum EzGfxEventKind {
    /// A typed upload state transition; `upload` is valid.
    Upload = 1,
    /// A runtime observation; `record` is valid.
    Runtime = 2,
    /// A diagnostic observation; `record` and `level` are valid.
    Diagnostic = 3,
    /// Bounded observability storage discarded records; `dropped` is valid.
    ObservationsDropped = 4,
    /// Completed explicitly requested readback bytes; the `readback_*` fields are valid.
    Readback = 5,
    /// Unrequested persistent presentation snapshot; the `readback_*` fields are valid.
    Snapshot = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
#[allow(
    clippy::pub_underscore_fields,
    reason = "Named public padding preserves ABI layout and zero-initialization for callers."
)]
/// One tagged graphics event delivered to the registered context callback.
///
/// Only the payload selected by `kind` is valid. Readback bytes are borrowed
/// for the callback invocation and must not be retained afterwards.
pub struct EzGfxEvent {
    /// Selects the valid payload by its C ABI numeric code.
    pub kind: EzGfxEventKind,
    /// Reserves bytes that keep the C ABI layout stable.
    pub _pad_kind: [u8; 7],
    /// Upload transition; valid when `kind` is `Upload`.
    pub upload: EzGfxUploadEvent,
    /// Runtime or diagnostic record; valid when `kind` is `Runtime` or `Diagnostic`.
    pub record: EzGfxRuntimeRecord,
    /// Diagnostic severity; valid when `kind` is `Diagnostic`.
    pub level: u8,
    /// Reserves bytes that keep the C ABI layout stable.
    pub _pad_level: [u8; 7],
    /// Discarded-record count; valid when `kind` is `ObservationsDropped`.
    pub dropped: u64,
    /// Stable request correlator, or zero for an unrequested snapshot.
    pub readback_request_id: EzGfxReadbackRequest,
    /// Readback source texture, or zero for a surface-presented snapshot.
    pub readback_texture: EzGfxTexture,
    /// Readback image width in pixels; valid for `Readback` or `Snapshot`.
    pub readback_width: u32,
    /// Readback image height in pixels; valid for `Readback` or `Snapshot`.
    pub readback_height: u32,
    /// Readback byte count; valid for `Readback` or `Snapshot`.
    pub readback_byte_count: usize,
    /// Readback RGBA bytes borrowed for this callback invocation only.
    pub readback_bytes: *const u8,
}

/// Context event callback invoked on the owner thread at graphics safe points.
///
/// `event` borrows its payload, including readback bytes, for the invocation
/// only. `user_data` is the registration value. Callbacks must not reenter
/// graphics operations or unwind; a reentrant dispatch attempt fails and a
/// panic unregisters the callback.
pub type EzGfxEventCallback =
    Option<unsafe extern "C" fn(event: *const EzGfxEvent, user_data: *mut c_void)>;
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

#[cfg(test)]
mod result_tests {
    use super::*;
    use ez_gfx::{CapabilityError, Error};
    use ez_gfx_runtime::LifecycleError;

    #[test]
    fn every_safe_error_maps_to_a_stable_abi_status() {
        for (error, expected) in [
            (Error::InvalidArgument, EzGfxResult::InvalidArgument),
            (Error::InvalidContext, EzGfxResult::InvalidContext),
            (Error::NativeFailure, EzGfxResult::NativeFailure),
            (Error::NotReady, EzGfxResult::NotReady),
            (Error::Unsupported, EzGfxResult::Unsupported),
            (Error::DeviceLost, EzGfxResult::DeviceLost),
            (Error::QueueFull, EzGfxResult::QueueFull),
            (Error::Cancelled, EzGfxResult::Cancelled),
            (
                Error::Lifecycle(LifecycleError::WrongThread),
                EzGfxResult::InvalidContext,
            ),
            (
                Error::Lifecycle(LifecycleError::AlreadyLost),
                EzGfxResult::DeviceLost,
            ),
            (
                Error::Capability(CapabilityError::MissingCompression),
                EzGfxResult::Unsupported,
            ),
        ] {
            assert_eq!(EzGfxResult::from(error), expected);
        }
    }
}
