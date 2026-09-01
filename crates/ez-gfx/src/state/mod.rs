use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

#[cfg(windows)]
use ez_gfx_backend_dx12::native::{NativeContext as Dx12Context, NativeSurface as Dx12Surface};
#[cfg(target_vendor = "apple")]
use ez_gfx_backend_metal::native::{NativeContext as MetalContext, NativeSurface as MetalSurface};
use ez_gfx_backend_vulkan::{
    NativeContext as VulkanContext, NativeSurface as VulkanSurface,
    SurfacePlatform as VulkanPlatform,
};
use ez_gfx_core::{
    Backend,
    handle::{
        ContextHandle, GenerationalArena, HandleParts, IndirectBufferHandle, LocalHandle,
        PackedHandle, ShaderHandle, StructuredBufferHandle, SurfaceHandle, TextureHandle,
    },
};
use ez_gfx_hal::{
    AllocationRequest, BufferRange, BufferTransfer, CompletionToken, DynamicPipelineState,
    ExecutionAction, FrameExecutionBackend, FrameExecutionPlan, HalError, ImageMip,
    MemoryAllocator, MemoryClass, QueueKind, ResourceAccess, ResourceState, ShaderStage,
};
use ez_gfx_runtime::render::{ExecutionError, execute_compiled_graph};
use ez_gfx_runtime::{
    ContextIdentity, ContextOptions, LifecycleError, ResourceKind, SurfaceOptions, SurfacePlatform,
    SurfaceState,
    frame::{ExecutableNode, FrameRecorder},
    geometry::{GeometryError, GeometryManager},
    graph::{
        Access, ImageRange, LoadOp, NodeDesc, PassInfo, ResourceDesc, ResourceId, ResourceLifetime,
        StoreOp,
    },
    indirect::{DrawIndexedCommand, IndexedIndirectBuffer},
    observability::{DiagnosticLevel, Observability, RuntimePhase, RuntimeRecord, RuntimeStatus},
    target::Format,
    texture::{
        TextureDecoder, TextureError, TextureId, TextureRegistry, TextureSource, generate_mips,
    },
};

use crate::EzGfxResult;

enum NativeContext {
    Vulkan(Box<VulkanContext>),
    #[cfg(windows)]
    Dx12(Box<Dx12Context>),
    #[cfg(target_vendor = "apple")]
    Metal(Box<MetalContext>),
}

enum NativeSurface {
    Vulkan(VulkanSurface),
    #[cfg(windows)]
    Dx12(Dx12Surface),
    #[cfg(target_vendor = "apple")]
    Metal(MetalSurface),
}

enum NativeAllocation {
    Vulkan(ez_gfx_backend_vulkan::NativeAllocation),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativeAllocation),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativeAllocation),
}

enum NativeShader {
    Vulkan(ez_gfx_backend_vulkan::NativeShader),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativeShader),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativeShader),
}

enum NativePipeline {
    Vulkan(ez_gfx_backend_vulkan::NativePipeline),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativePipeline),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativePipeline),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum PipelineKey {
    Compute {
        backend: Backend,
        shader: ShaderHandle,
        shader_digest: [u8; 32],
        product: usize,
        entry: String,
        layouts: Vec<ez_gfx_hal::ShaderBufferLayout>,
    },
    Graphics {
        backend: Backend,
        shader: ShaderHandle,
        shader_digest: [u8; 32],
        vertex_product: usize,
        vertex_entry: String,
        fragment_product: usize,
        fragment_entry: String,
        layouts: Vec<ez_gfx_hal::ShaderBufferLayout>,
        texture_heap: Option<ez_gfx_hal::ShaderTextureHeapLayout>,
        state: DynamicPipelineState,
        depth_required: bool,
        color_format: u32,
        depth_format: u32,
        sample_count: u8,
    },
}

impl PipelineKey {
    fn shader(&self) -> ShaderHandle {
        // Both variants always carry their owning generational shader handle.
        match self {
            Self::Compute { shader, .. } | Self::Graphics { shader, .. } => *shader,
        }
    }
}

const MAX_PIPELINE_CACHE_ENTRIES: usize = 1024;

struct ShaderRecord {
    native: NativeShader,
    digest: [u8; 32],
    graphics: Option<(usize, String, usize, String)>,
    compute: Option<(usize, String)>,
    runtime: ez_gfx_runtime::shader::RuntimeShader,
    graphics_layout: Option<ez_gfx_runtime::binding::PipelineLayout>,
}

enum NativeTexture {
    Vulkan(ez_gfx_backend_vulkan::NativeTexture),
    #[cfg(windows)]
    Dx12(ez_gfx_backend_dx12::native::NativeTexture),
    #[cfg(target_vendor = "apple")]
    Metal(ez_gfx_backend_metal::native::NativeTexture),
}

struct SurfaceRecord {
    native: NativeSurface,
    state: SurfaceState,
}

struct GeometryAllocation {
    allocation: NativeAllocation,
    ready: Option<CompletionToken>,
    size: u64,
}

struct StagingAllocation {
    capacity: u64,
    allocation: NativeAllocation,
    retirement: Option<CompletionToken>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameNativeResource {
    Buffer(PackedHandle),
    Texture(TextureHandle),
    Surface(SurfaceHandle),
    Depth,
    Index,
}
struct ContextState {
    identity: ContextIdentity,
    options: ContextOptions,
    native: NativeContext,
    surfaces: HashMap<SurfaceHandle, SurfaceRecord>,
    allocations: HashMap<PackedHandle, (u64, NativeAllocation)>,
    shaders: HashMap<ShaderHandle, ShaderRecord>,
    indirects: HashMap<IndirectBufferHandle, IndexedIndirectBuffer>,
    textures: HashMap<TextureHandle, (TextureId, NativeTexture, u32, u32, u32)>,
    pipelines: HashMap<PipelineKey, NativePipeline>,
    graphics_format: Option<u32>,
    texture_registry: TextureRegistry,
    texture_ready: HashMap<TextureHandle, CompletionToken>,
    geometry: GeometryManager,
    vertex_heaps: HashMap<String, GeometryAllocation>,
    index_heap: Option<GeometryAllocation>,
    staging: Vec<StagingAllocation>,
    frame: FrameRecorder,
    frame_resources: HashMap<PackedHandle, ResourceId>,
    frame_native_resources: HashMap<ResourceId, FrameNativeResource>,
    frame_index: Option<ResourceId>,
    frame_surface: Option<ResourceId>,
    frame_depth: Option<ResourceId>,
    frame_has_graphics: bool,
    last_readback: Vec<u8>,
    active_surface: Option<SurfaceHandle>,
    frame_presented: bool,
    observability: Observability,
}

type ContextHandleArena = GenerationalArena<()>;
static CONTEXT_HANDLES: LazyLock<Mutex<ContextHandleArena>> =
    LazyLock::new(|| Mutex::new(GenerationalArena::new()));
struct ThreadContexts {
    states: HashMap<LocalHandle, ContextState>,
}

impl ThreadContexts {
    fn cleanup_for_thread_exit(&mut self) -> EzGfxResult {
        // Handles invalidate synchronously before platform-specific abandonment handling.
        let result = match CONTEXT_HANDLES.lock() {
            Ok(mut handles) => {
                let mut result = EzGfxResult::Ok;
                for local in self.states.keys() {
                    if handles.remove(*local).is_err() {
                        result = EzGfxResult::NativeFailure;
                    }
                }
                result
            }
            Err(_) => EzGfxResult::NativeFailure,
        };

        #[cfg(windows)]
        {
            for (_, state) in self.states.drain() {
                // Windows TLS destructors run under loader lock; native cleanup or joining can deadlock.
                std::mem::forget(state);
            }
            result
        }

        #[cfg(not(windows))]
        {
            let mut result = result;
            for (_, state) in self.states.drain() {
                let cleanup = context::cleanup_context_state(state, None);
                if result == EzGfxResult::Ok {
                    result = cleanup;
                }
            }
            result
        }
    }
}

impl Drop for ThreadContexts {
    fn drop(&mut self) {
        let _ = self.cleanup_for_thread_exit();
    }
}

thread_local! {
    static CONTEXTS: RefCell<ThreadContexts> = RefCell::new(ThreadContexts {
        states: HashMap::new(),
    });
}

mod buffers;
mod context;
mod frame;
mod native;
#[cfg(windows)]
use native::dx12_bindings;
#[cfg(target_vendor = "apple")]
use native::metal_bindings;
use native::{
    allocate_native, completed_transfer_native, copy_native, destroy_native_texture,
    free_native_allocation, map_allocation, map_frame, map_geometry, map_hal, map_lifecycle,
    map_native_loss, map_texture, native_layouts, pipeline_layout_key, result_status,
    vulkan_bindings, wait_native_idle, write_native,
};
mod shader;
mod texture;

pub use buffers::*;
pub use context::*;
pub use frame::*;
pub use shader::*;
pub use texture::*;

fn with_surface_mut<T>(
    context: ContextHandle,
    surface: SurfaceHandle,
    operation: impl FnOnce(&mut SurfaceRecord) -> Result<T, EzGfxResult>,
) -> Result<T, EzGfxResult> {
    with_context_mut(context, |context| {
        context
            .identity
            .check_thread_and_health()
            .map_err(map_lifecycle)?;
        let handle = surface.packed();
        context
            .identity
            .resolve(handle, ResourceKind::Surface)
            .map_err(map_lifecycle)?;
        operation(
            context
                .surfaces
                .get_mut(&surface)
                .ok_or(EzGfxResult::InvalidContext)?,
        )
    })
}
fn with_context_mut<T>(
    context: ContextHandle,
    operation: impl FnOnce(&mut ContextState) -> Result<T, EzGfxResult>,
) -> Result<T, EzGfxResult> {
    let (local, _) = context_local(context)?;
    CONTEXTS.with(|contexts| {
        let mut contexts = contexts
            .try_borrow_mut()
            .map_err(|_| EzGfxResult::NativeFailure)?;
        let context = contexts
            .states
            .get_mut(&local)
            .ok_or(EzGfxResult::InvalidContext)?;
        operation(context)
    })
}
fn context_local(handle: ContextHandle) -> Result<(LocalHandle, PackedHandle), EzGfxResult> {
    let packed = handle.packed();

    match packed.parts().map_err(|_| EzGfxResult::InvalidContext)? {
        HandleParts::Context(local) => Ok((local, packed)),
        HandleParts::Child { .. } => Err(EzGfxResult::InvalidContext),
    }
}

#[cfg(test)]
mod tests;
