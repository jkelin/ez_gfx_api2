use core::ffi::{c_char, c_void};

pub type EzGfxHandle = u64;
pub type EzGfxContext = EzGfxHandle;
pub type EzGfxSurface = EzGfxHandle;
pub type EzGfxShader = EzGfxHandle;
pub type EzGfxIndirectBuffer = EzGfxHandle;
pub type EzGfxStructuredBuffer = EzGfxHandle;
pub type EzGfxTexture = EzGfxHandle;
pub type EzGfxRenderTarget = EzGfxHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EzGfxResult {
    Ok = 0,
    InvalidArgument = 1,
    InvalidContext = 2,
    NativeFailure = 3,
    NotReady = 4,
    Unsupported = 5,
    DeviceLost = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EzGfxTextureError {
    None = 0,
    InvalidContext = 1,
    InvalidArguments = 2,
    UnsupportedFormat = 3,
    OutOfTextureHandles = 4,
    OutOfMemory = 5,
    DecodeFailed = 6,
    VulkanFailed = 7,
    WorkerUnavailable = 8,
    NotFound = 9,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxContextDesc {
    pub enable_debug: u8,
    pub enable_validation: u8,
    pub surface_platform: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxBackendContextDesc {
    pub enable_debug: u8,
    pub enable_validation: u8,
    pub surface_platform: u8,
    pub backend: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxSurfaceDesc {
    pub window: *mut c_void,
    pub display: *mut c_void,
    pub platform: u8,
    pub width: u32,
    pub height: u32,
    pub cache_presented_snapshots: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxShaderDesc {
    pub path: *const c_char,
    pub vertex_entry: *const c_char,
    pub fragment_entry: *const c_char,
    pub compute_entry: *const c_char,
    pub kind: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxShaderEntry {
    pub entry: *const c_char,
    pub stage: u8,
    pub _padding: [u8; 7],
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxTextureDesc {
    pub source_format: u8,
    pub destination_format: u8,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub generate_mips: u8,
    pub min_filter: u8,
    pub mag_filter: u8,
    pub max_anisotropy: f32,
    pub address_mode_u: u8,
    pub address_mode_v: u8,
    pub address_mode_w: u8,
    pub debug_label: *const c_char,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxBinding {
    pub name: *const c_char,
    pub structured: EzGfxStructuredBuffer,
    pub indirect: EzGfxIndirectBuffer,
    pub render_target: EzGfxRenderTarget,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxDynamicState {
    pub cull_mode: u8,
    pub front_face: u8,
    pub primitive_type: u8,
    pub blend_mode: u8,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxDrawIndexedCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct EzGfxByteBuffer {
    pub length: usize,
    pub data: *const u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct EzGfxHandleParts {
    pub context_slot: u32,
    pub context_generation: u32,
    pub child_slot: u32,
    pub child_generation: u32,
    pub is_context: u8,
    pub _padding: [u8; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct EzGfxRuntimeRecord {
    pub correlation_id: u64,
    pub resource: EzGfxHandle,
    pub backend: u8,
    pub phase: u8,
    pub status: u8,
    pub _padding: [u8; 5],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct EzGfxDiagnostic {
    pub record: EzGfxRuntimeRecord,
    pub level: u8,
    pub _padding: [u8; 7],
}
