//! C binding enums represented as bytes in the executable ABI.
//! `tools/bindgen` parses these declarations; runtime ABI fields retain their byte representation.

/// Shader stage family.
#[repr(u8)]
pub enum EzGfxShaderKind {
    /// Vertex and fragment shader pair.
    Graphics = 0,
    /// Compute shader.
    Compute = 1,
}

/// Selected native backend.
#[repr(u8)]
pub enum EzGfxBackend {
    /// Vulkan 1.3 backend.
    Vulkan = 1,
    /// Direct3D 12 backend.
    Dx12 = 2,
    /// Metal backend.
    Metal = 3,
}

/// Runtime operation phase.
#[repr(u8)]
pub enum EzGfxRuntimePhase {
    /// Adapter or device admission.
    Admission = 1,
    /// Asset decoding.
    Decode = 2,
    /// Resource upload.
    Upload = 3,
    /// Resource binding.
    Bind = 4,
    /// Queue submission.
    Submit = 5,
    /// Presentation.
    Present = 6,
    /// GPU readback.
    Readback = 7,
    /// Device lifecycle.
    Device = 8,
}

/// Resource family carried by an upload event.
#[repr(u8)]
pub enum EzGfxUploadResourceKind {
    /// Texture upload.
    Texture = 1,
    /// Vertex allocation upload.
    VertexAllocation = 2,
    /// Index allocation upload.
    IndexAllocation = 3,
}

/// Lossless upload transition.
#[repr(u8)]
pub enum EzGfxUploadStatus {
    /// Caller source bytes are no longer needed.
    SourceStaged = 1,
    /// Resource is ready for rendering.
    DeviceReady = 2,
    /// Upload terminated with error.
    Failed = 3,
    /// Upload was cancelled.
    Cancelled = 4,
}

/// Bounded local diagnostic severity.
#[repr(u8)]
pub enum EzGfxDiagnosticLevel {
    /// Informational diagnostic.
    Info = 1,
    /// Recoverable warning.
    Warning = 2,
    /// Operation failure.
    Error = 3,
}

/// Source image encoding; values 128 through 255 are application decoders.
#[repr(u8)]
pub enum EzGfxSourceTextureFormat {
    /// Raw RGB pixels.
    Rgb = 0,
    /// Raw RGBA pixels.
    Rgba = 1,
    /// BMP image bytes.
    Bmp = 2,
    /// JPEG image bytes.
    Jpeg = 3,
    /// PNG image bytes.
    Png = 4,
    /// TGA image bytes.
    Tga = 5,
    /// KTX2 image bytes.
    Ktx2 = 6,
    /// Optional standalone Basis Universal bytes.
    Basis = 7,
    /// DDS image bytes with a legacy or DX10 header.
    Dds = 8,
    /// Tightly packed mip bytes in the destination storage format.
    Raw = 9,
}

/// Texture sampling filter.
#[repr(u8)]
pub enum EzGfxTextureFilter {
    /// Nearest-neighbor filtering.
    Nearest = 0,
    /// Linear filtering.
    Linear = 1,
}

/// Texture addressing mode.
#[repr(u8)]
pub enum EzGfxTextureAddressMode {
    /// Repeat coordinates.
    Repeat = 0,
    /// Clamp coordinates to the edge.
    ClampToEdge = 1,
}

/// Requested GPU texture storage.
#[repr(u8)]
pub enum EzGfxTextureDestinationFormat {
    /// Linear 8-bit normalized RGBA.
    Rgba8Unorm = 0,
    /// Prefer admitted native compression.
    Auto = 1,
    /// sRGB 8-bit normalized RGBA.
    Rgba8Srgb = 2,
    /// Linear BC1.
    Bc1Unorm = 3,
    /// sRGB BC1.
    Bc1Srgb = 4,
    /// Linear BC3.
    Bc3Unorm = 5,
    /// sRGB BC3.
    Bc3Srgb = 6,
    /// Linear BC7.
    Bc7Unorm = 7,
    /// sRGB BC7.
    Bc7Srgb = 8,
    /// Linear ASTC 4x4.
    Astc4x4Unorm = 9,
    /// sRGB ASTC 4x4.
    Astc4x4Srgb = 10,
}

/// Pipeline cull mode.
#[repr(u8)]
pub enum EzGfxCullMode {
    /// No face culling.
    None = 0,
    /// Cull front-facing primitives.
    Front = 1,
    /// Cull back-facing primitives.
    Back = 2,
}

/// Front-face winding order.
#[repr(u8)]
pub enum EzGfxFrontFace {
    /// Counter-clockwise winding.
    CounterClockwise = 0,
    /// Clockwise winding.
    Clockwise = 1,
}

/// Pipeline primitive topology.
#[repr(u8)]
pub enum EzGfxPrimitiveType {
    /// Triangle list.
    TriangleList = 0,
    /// Point list.
    PointList = 1,
    /// Line list.
    LineList = 2,
    /// Line strip.
    LineStrip = 3,
    /// Triangle strip.
    TriangleStrip = 4,
    /// Triangle fan.
    TriangleFan = 5,
}

/// Pipeline blend mode.
#[repr(u8)]
pub enum EzGfxBlendMode {
    /// Opaque blending.
    None = 0,
    /// Source-alpha blending.
    Alpha = 1,
}
