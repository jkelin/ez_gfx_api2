use crate::{BACKEND, TEXTURE_DESCRIPTOR_CAPACITY};
use core::{ffi::c_void, ptr};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use ez_gfx_core::capability::{
    AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport, SemanticProfile,
};
use ez_gfx_hal::{
    AllocationError, AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BlendMode,
    BufferTransfer, CompletionToken, CullMode, DEFAULT_ALLOCATION_BLOCK_POLICY,
    DynamicPipelineState, ExecutionBarrier, ExecutionPass, FrontFace, HalError, ImageMip,
    MemoryAllocator, MemoryClass, PrimitiveTopology, QueueKind, ResourceAccess, SamplerAddressMode,
    SamplerFilter, ShaderBufferLayout, TextureFormat, TextureRegion, TextureSamplerDesc,
    TransferWorker, validate_texture_mips, validate_texture_region,
};
use gpu_allocator::{
    AllocationSizes, MemoryLocation,
    d3d12::{
        Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc, ID3D12DeviceVersion,
    },
};
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D_SHADER_MODEL_6_5, D3D12_BLEND_DESC,
    D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_BLEND_SRC_ALPHA,
    D3D12_BLEND_ZERO, D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0,
    D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COMMAND_SIGNATURE_DESC, D3D12_COMPARISON_FUNC_ALWAYS,
    D3D12_COMPARISON_FUNC_LESS, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CULL_MODE_BACK, D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_NONE,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DEPTH_STENCIL_DESC, D3D12_DEPTH_STENCIL_VALUE,
    D3D12_DEPTH_STENCILOP_DESC, D3D12_DEPTH_WRITE_MASK_ALL, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_RANGE,
    D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND, D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
    D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_FEATURE_DATA_SHADER_MODEL, D3D12_FEATURE_SHADER_MODEL,
    D3D12_FILL_MODE_SOLID, D3D12_FILTER_ANISOTROPIC, D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
    D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_FILTER_MIN_MAG_MIP_POINT,
    D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT, D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    D3D12_INDEX_BUFFER_VIEW, D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_DESC_0,
    D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, D3D12_LOGIC_OP_NOOP,
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
    D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC, D3D12_RESOURCE_BARRIER,
    D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_READ,
    D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_INDEX_BUFFER,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER,
    D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
    D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SAMPLER_DESC, D3D12_SHADER_BYTECODE, D3D12_SHADER_RESOURCE_VIEW_DESC,
    D3D12_SHADER_RESOURCE_VIEW_DESC_0, D3D12_SHADER_VISIBILITY_ALL, D3D12_SRV_DIMENSION_TEXTURE2D,
    D3D12_STENCIL_OP_KEEP, D3D12_TEX2D_SRV, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
    D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_VIEWPORT, D3D12SerializeRootSignature,
    ID3D12CommandSignature, ID3D12DescriptorHeap, ID3D12PipelineState, ID3D12RootSignature,
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, HWND, RECT, WAIT_FAILED, WAIT_OBJECT_0},
        Graphics::{
            Direct3D::{D3D_FEATURE_LEVEL_12_1, ID3DBlob},
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_DIRECT,
                D3D12_COMMAND_QUEUE_DESC, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS,
                D3D12_FENCE_FLAG_NONE, D3D12_RANGE, D3D12_RENDER_TARGET_VIEW_DESC,
                D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RTV_DIMENSION_TEXTURE2D,
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12CreateDevice, ID3D12CommandAllocator,
                ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
                ID3D12GraphicsCommandList, ID3D12Resource,
            },
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
                    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_UNKNOWN,
                    DXGI_SAMPLE_DESC,
                },
                CreateDXGIFactory1, DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_ERROR_NOT_FOUND,
                DXGI_ERROR_UNSUPPORTED, DXGI_PRESENT, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
                DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_EFFECT_FLIP_DISCARD,
                DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIAdapter4, IDXGIFactory4, IDXGISwapChain4,
            },
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
        UI::WindowsAndMessaging::GetClientRect,
    },
    core::Interface,
};

/// D3D12 buffer resource and its allocator ownership record.
pub struct NativeAllocation {
    resource: ID3D12Resource,
    allocation: Allocation,
    mapped_address: usize,
}

/// Root buffer binding resolved for one dispatch or draw.
pub struct NativeBufferBinding<'a> {
    /// Allocation supplying the root descriptor.
    pub allocation: &'a NativeAllocation,
    /// Byte offset added to the resource GPU address.
    pub offset: u64,
    /// Whether the root descriptor permits unordered writes.
    pub writable: bool,
}

/// Fully resolved indexed draw consumed by command-list recording.
pub struct NativeDrawIndexed<'a> {
    /// Render width in pixels.
    pub width: u32,
    /// Render height in pixels.
    pub height: u32,
    /// Graphics pipeline used by the draw.
    pub pipeline: &'a NativePipeline,
    /// Buffer containing 32-bit indices.
    pub index_buffer: &'a NativeAllocation,
    /// Logical byte length of the index resource.
    pub index_size: u64,
    /// Buffer containing indexed indirect commands.
    pub indirect_buffer: &'a NativeAllocation,
    /// Logical byte length of the indirect resource.
    pub indirect_size: u64,
    /// Number of indirect commands to execute.
    pub draw_count: u32,
    /// Reflected root buffer bindings.
    pub bindings: &'a [NativeBufferBinding<'a>],
}

/// Fully resolved compute dispatch consumed by command-list recording.
pub struct NativeComputeDispatch<'a> {
    /// Compute pipeline used by the dispatch.
    pub pipeline: &'a NativePipeline,
    /// Workgroup count for each dimension.
    pub groups: [u32; 3],
    /// Reflected root buffer bindings.
    pub bindings: &'a [NativeBufferBinding<'a>],
}

/// D3D12 resource referenced by a compiled frame barrier.
pub enum NativeFrameResource<'a> {
    /// Buffer resource.
    Buffer(&'a NativeAllocation),
    /// Texture resource.
    Texture(&'a NativeTexture),
    /// Current swapchain back buffer.
    Surface,
    /// Current depth resource.
    Depth,
    /// Managed single-mip color render target.
    RenderTarget(&'a NativeTexture),
}

/// One resolved pass color attachment: its native resource plus the clear
/// value applied when the pass load op clears. Surfaces carry the legacy
/// default; render targets carry their stored declaration clear.
pub struct PassAttachment<'a> {
    /// Resolved native color resource.
    pub resource: NativeFrameResource<'a>,
    /// Clear color applied for a clearing load op.
    pub clear: [f32; 4],
}

/// Validated D3D12 action emitted by the frame-plan adapter.
pub enum NativeFrameAction<'a> {
    /// Wait for external transfer completion.
    Wait(CompletionToken),
    /// Transition or order access to a resource.
    Barrier {
        /// Backend-neutral barrier description.
        barrier: ExecutionBarrier,
        /// Resolved resource affected by the barrier.
        resource: NativeFrameResource<'a>,
    },
    /// Begin the declared render pass with resolved color attachments.
    BeginPass {
        /// Backend-neutral pass description.
        pass: &'a ExecutionPass,
        /// One attachment per pass color, in order.
        colors: Vec<PassAttachment<'a>>,
    },
    /// Encode a compute dispatch.
    Compute(NativeComputeDispatch<'a>),
    /// Encode indexed indirect graphics work.
    Graphics(NativeDrawIndexed<'a>),
    /// Copy a texture into a readback allocation.
    TextureReadback {
        /// Texture to copy.
        texture: &'a NativeTexture,
        /// Readback width in pixels.
        width: u32,
        /// Readback height in pixels.
        height: u32,
    },
    /// End the active render pass.
    EndPass,
    /// Present the active swapchain buffer.
    Present,
}

/// Validated DXIL products retained until pipeline creation.
pub struct NativeShader {
    products: Vec<Vec<u8>>,
}

impl NativeShader {
    /// Returns the DXIL products in artifact order.
    pub fn products(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.products.iter().map(Vec::as_slice)
    }
}

/// D3D12 pipeline, root signature, and indirect execution metadata.
pub struct NativePipeline {
    state: ID3D12PipelineState,
    root: ID3D12RootSignature,
    topology: Option<D3D_PRIMITIVE_TOPOLOGY>,
    signature: Option<ID3D12CommandSignature>,
    buffer_writable: Vec<bool>,
}

/// Multisampled render storage owned by a managed render target.
///
/// The single-sample `NativeTexture` resource stays the resolve destination,
/// so barriers, readback, descriptors, and destruction keep working unchanged;
/// only the render pass binds this storage and resolves into it at end of pass.
pub struct MsaaStorage {
    resource: ID3D12Resource,
    allocation: Allocation,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    /// Render sample count; passes must request exactly this count.
    samples: u8,
}

/// Texture resource and allocator ownership record.
pub struct NativeTexture {
    resource: ID3D12Resource,
    allocation: Allocation,
    format: TextureFormat,
    width: u32,
    height: u32,
    mip_count: u32,
    resident_mips: u32,
    mip_completions: Vec<u64>,
    cancellation: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Slot in the shader-visible texture descriptor heap.
    pub binding: u32,
    /// Render-target view for managed color targets; `None` for uploads.
    /// The retaining heap keeps the CPU handle valid until destruction.
    pub rtv: Option<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE)>,
    /// Multisampled render storage plus its view and allocation; `None` for
    /// uploads and single-sample targets. Only render-target entry points
    /// touch this; the resource above stays the resolve destination.
    msaa: Option<MsaaStorage>,
}

impl NativeTexture {
    /// Returns the last transfer value that may reference this texture.
    pub fn last_transfer_value(&self) -> u64 {
        self.mip_completions.iter().copied().max().unwrap_or(0)
    }

    /// Latest copy values ordered from finest to coarsest mip; zero means never submitted.
    pub fn mip_transfer_values(&self) -> &[u64] {
        &self.mip_completions
    }
}

struct RetiredAllocation {
    allocation: NativeAllocation,
    completion: CompletionToken,
}

const FRAMES_IN_FLIGHT: usize = 3;

struct FrameSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_value: u64,
    garbage: Vec<NativeAllocation>,
}

enum DeferredResource {
    Allocation(NativeAllocation),
    Pipeline(NativePipeline),
    Texture(NativeTexture),
}

struct DeferredNativeResource {
    fence_value: u64,
    resource: DeferredResource,
}

struct SurfaceDepth {
    resource: ID3D12Resource,
    allocation: Allocation,
    heap: ID3D12DescriptorHeap,
}

/// Borrowed HWND, swapchain state, and optional captured frame.
pub struct NativeSurface {
    window: usize,
    swapchain: Option<IDXGISwapChain4>,
    buffers: Vec<ID3D12Resource>,
    rtv_heap: Option<ID3D12DescriptorHeap>,
    width: u32,
    height: u32,
    presented: Vec<u8>,
    depth: Option<SurfaceDepth>,
}

impl NativeSurface {
    /// Extracts a validated HWND from a matched Windows/Win32 raw handle pair.
    ///
    /// # Safety
    ///
    /// `display` and `window` must belong to the same live Windows host, and this function must run
    /// on the host's creator thread. The host must remain alive until the returned surface is
    /// destroyed by [`NativeContext::destroy_surface`] or native teardown is abandoned, after
    /// which it must remain alive for the process lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::InvalidArgument`] for every non-Windows or mismatched pair.
    pub unsafe fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<Self, HalError> {
        let (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) = (display, window)
        else {
            return Err(HalError::InvalidArgument);
        };
        Ok(Self {
            // Preserve the opaque HWND bit pattern even when its pointer-sized integer is negative.
            window: window.hwnd.get().cast_unsigned(),
            swapchain: None,
            width: 0,
            height: 0,
            buffers: Vec::new(),
            rtv_heap: None,
            presented: Vec::new(),
            depth: None,
        })
    }

    /// Returns the borrowed HWND value.
    pub const fn window(&self) -> usize {
        self.window
    }

    /// Reads the current drawable size from the borrowed HWND.
    ///
    /// # Errors
    ///
    /// Returns [`HalError::NativeFailure`] when Win32 rejects the query.
    pub fn window_extent(&self) -> Result<Option<(u32, u32)>, HalError> {
        let mut rect = RECT::default();
        // SAFETY: `self.window` remains borrowed and valid for this surface's lifetime.
        unsafe { GetClientRect(HWND(self.window as *mut _), &raw mut rect) }
            .map_err(|_| HalError::NativeFailure)?;
        let width = rect.right.saturating_sub(rect.left);
        let height = rect.bottom.saturating_sub(rect.top);
        if width == 0 || height == 0 {
            return Ok(None);
        }
        Ok(Some((
            u32::try_from(width).map_err(|_| HalError::NativeFailure)?,
            u32::try_from(height).map_err(|_| HalError::NativeFailure)?,
        )))
    }

    /// Returns the most recently captured RGBA8 frame, or an empty slice before capture.
    pub fn presented_rgba8(&self) -> &[u8] {
        &self.presented
    }
}

/// Admitted D3D12 device, queues, allocators, descriptors, and frame state.
pub struct NativeContext {
    /// Adapter selected during context creation.
    pub adapter: IDXGIAdapter4,
    /// Feature-level 12.1 device created from the selected adapter.
    pub device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    transfer_fence: ID3D12Fence,
    texture_fence: ID3D12Fence,
    transfer_worker: Option<TransferWorker<transfer::Dx12TransferJob>>,
    texture_worker: Option<TransferWorker<transfer::Dx12TransferJob>>,
    fence_event: HANDLE,
    next_fence: u64,
    idle_drained: bool,
    #[cfg(test)]
    wait_idle_failure: Option<HalError>,
    next_transfer_fence: u64,
    next_texture_fence: u64,
    allocator: Option<Allocator>,
    retired: Vec<RetiredAllocation>,
    texture_staging: ez_gfx_hal::ReusableStagingPool<NativeAllocation>,
    frame_slots: Vec<FrameSlot>,
    frame_cursor: usize,
    deferred: Vec<DeferredNativeResource>,
    adapter_info: AdapterInfo,
    descriptors: ID3D12DescriptorHeap,
    descriptor_stride: u32,
    samplers: ID3D12DescriptorHeap,
    sampler_stride: u32,
}

// SAFETY: D3D12/DXGI interfaces are agile and the event handle is process-wide; higher layers
// serialize mutation and enforce frame-recording thread affinity.
unsafe impl Send for NativeContext {}

mod commands;
use commands::{
    bind_dx12_compute_buffers, bind_dx12_graphics_buffers, copy_texture_to_readback,
    create_frame_slots, dx12_resource_state, record_resource_barriers, transition_barrier,
    uav_barrier,
};
mod device;
mod frame;
mod memory;
use memory::{adapter_id, map_allocation_windows, map_allocator, map_allocator_hal, map_windows};
mod pipeline;
mod surface;
mod texture;
mod transfer;

#[cfg(test)]
mod texture_tests;

#[cfg(test)]
mod surface_tests {
    use std::num::NonZeroIsize;

    use raw_window_handle::{
        RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
        XlibDisplayHandle,
    };

    use super::*;

    #[test]
    fn surface_accepts_only_windows_win32_pairs() {
        let window = RawWindowHandle::Win32(Win32WindowHandle::new(NonZeroIsize::new(1).unwrap()));

        let surface =
            // SAFETY: the synthetic matched pair is inspected only; no native call dereferences it.
            unsafe {
                NativeSurface::new(
                    RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
                    window,
                )
            }
            .unwrap();
        assert_eq!(surface.window(), 1);

        let mismatch = RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0));
        assert_eq!(
            // SAFETY: mismatched handles are rejected before either synthetic value is used.
            unsafe { NativeSurface::new(mismatch, window) }.err(),
            Some(HalError::InvalidArgument)
        );
    }
}
