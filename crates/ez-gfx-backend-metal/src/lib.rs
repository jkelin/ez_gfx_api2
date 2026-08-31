#![deny(unsafe_op_in_unsafe_fn)]

use ez_gfx_core::{Backend, capability::MAX_BINDLESS_SAMPLED_TEXTURES};

pub const BACKEND: Backend = Backend::Metal;
pub const SUPPORTED_ON_TARGET: bool = cfg!(target_vendor = "apple");
pub const TEXTURE_DESCRIPTOR_CAPACITY: u32 = MAX_BINDLESS_SAMPLED_TEXTURES;

#[cfg(target_vendor = "apple")]
pub mod native {
    use core::ffi::c_void;

    use crate::{BACKEND, TEXTURE_DESCRIPTOR_CAPACITY};
    use ez_gfx_core::capability::{
        AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport, SemanticProfile,
    };
    use ez_gfx_hal::{
        AllocationError, AllocationRequest, AttachmentLoadOp, AttachmentStoreOp, BlendMode,
        BufferTransfer, CompletionToken, CullMode, DynamicPipelineState, ExecutionBarrier,
        ExecutionPass, FrontFace, HalError, ImageMip, MemoryAllocator, MemoryClass,
        PrimitiveTopology, QueueKind, SamplerAddressMode, SamplerFilter, ShaderTextureHeapLayout,
        TextureSamplerDesc, validate_rgba8_mips,
    };
    use gpu_allocator::{
        MemoryLocation,
        metal::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    };
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLArgumentBuffersTier, MTLArgumentEncoder, MTLBlendFactor, MTLBlitCommandEncoder,
        MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
        MTLCommandQueue, MTLCompareFunction, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLCreateSystemDefaultDevice, MTLCullMode, MTLDepthStencilDescriptor, MTLDepthStencilState,
        MTLDevice, MTLFunction, MTLHeap, MTLIndexType, MTLLibrary, MTLLoadAction, MTLOrigin,
        MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
        MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLRenderStages, MTLResource,
        MTLResourceOptions, MTLResourceUsage, MTLSamplerAddressMode, MTLSamplerDescriptor,
        MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStorageMode, MTLStoreAction,
        MTLTexture, MTLTextureDescriptor, MTLTextureUsage, MTLWinding,
    };
    use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

    pub struct NativeAllocation {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        allocation: Allocation,
        mapped_address: usize,
    }
    pub struct NativeBufferBinding<'a> {
        pub allocation: &'a NativeAllocation,
        pub offset: usize,
        pub index: usize,
    }

    pub struct NativeGraphicsDraw<'a> {
        pub shader: &'a NativeShader,
        pub graphics: &'a (usize, String, usize, String),
        pub depth_required: bool,
        pub texture_heap: Option<ShaderTextureHeapLayout>,
        pub state: DynamicPipelineState,
        pub index: &'a NativeAllocation,
        pub indirect: &'a NativeAllocation,
        pub draw_count: u32,
        pub push_constants: &'a [u8],
        pub bindings: &'a [NativeBufferBinding<'a>],
        pub textures: &'a [&'a NativeTexture],
    }

    pub struct NativeComputeDispatch<'a> {
        pub shader: &'a NativeShader,
        pub product_index: usize,
        pub entry: &'a str,
        pub groups: [u32; 3],
        pub push_constants: &'a [u8],
        pub bindings: &'a [NativeBufferBinding<'a>],
    }

    pub enum NativeFrameResource<'a> {
        Buffer(&'a NativeAllocation),
        Texture(&'a NativeTexture),
        Surface,
        Depth,
    }

    pub enum NativeFrameAction<'a> {
        Wait(CompletionToken),
        Barrier {
            barrier: ExecutionBarrier,
            resource: NativeFrameResource<'a>,
        },
        BeginPass(&'a ExecutionPass),
        Compute(NativeComputeDispatch<'a>),
        Graphics(NativeGraphicsDraw<'a>),
        TextureReadback {
            texture: &'a NativeTexture,
            width: u32,
            height: u32,
        },
        EndPass,
        Present,
    }

    pub struct NativeShader {
        libraries: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>>,
    }
    pub struct NativeTexture {
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
        allocation: Allocation,
        sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
        pub binding: u32,
    }

    pub struct NativePipeline {
        state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    // SAFETY: Metal resources and immutable libraries support cross-thread use. Higher layers
    // serialize mutation and destruction, so these owners are never accessed concurrently.
    unsafe impl Send for NativeAllocation {}
    unsafe impl Send for NativeShader {}
    unsafe impl Send for NativeTexture {}
    unsafe impl Send for NativeSurface {}
    type PreparedGraphics = (
        Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
        Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    );

    enum PreparedFrameAction {
        None,
        Compute(NativePipeline),
        Graphics {
            pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
            argument_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        },
    }

    struct RetiredAllocation {
        allocation: NativeAllocation,
        completion: CompletionToken,
    }

    struct SurfaceDepth {
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
        state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
        extent: (u32, u32),
    }

    pub struct NativeSurface {
        layer: usize,
        presented_rgba8: Vec<u8>,
        depth: Option<SurfaceDepth>,
    }
    impl NativeSurface {
        /// The CAMetalLayer pointer is borrowed; this crate never releases the host's layer.
        pub fn new(layer: *mut c_void, capture_presented: bool) -> Result<Self, HalError> {
            if layer.is_null() {
                return Err(HalError::InvalidArgument);
            }
            // Capture needs shader-readable drawable textures; set this before nextDrawable.
            if capture_presented {
                let layer = unsafe { &*(layer as *const CAMetalLayer) };
                layer.setFramebufferOnly(false);
            }
            Ok(Self {
                layer: layer as usize,
                presented_rgba8: Vec::new(),
                depth: None,
            })
        }

        /// Empty before the first cached presentation; successful captures replace the full frame.
        pub fn presented_rgba8(&self) -> &[u8] {
            &self.presented_rgba8
        }
    }

    pub struct NativeContext {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        allocator: Option<Allocator>,
        retired: Vec<RetiredAllocation>,
        adapter: AdapterInfo,
        next_transfer_value: u64,
        completed_transfer_value: u64,
    }

    // SAFETY: Metal devices and command queues support cross-thread use. Higher layers serialize
    // context mutation and enforce frame-recording thread affinity.
    unsafe impl Send for NativeContext {}

    impl NativeContext {
        pub fn create_default() -> Result<Self, HalError> {
            let device = MTLCreateSystemDefaultDevice().ok_or(HalError::Unsupported)?;
            let queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
            let tier_two = device.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2;
            let sampler_capacity =
                u32::try_from(device.maxArgumentBufferSamplerCount()).unwrap_or(u32::MAX);
            let compression = if device.supportsBCTextureCompression() {
                CompressionSupport::BC
            } else {
                CompressionSupport::ASTC
            };
            let capabilities = AdapterCapabilities {
                bindless_sampled_textures: if tier_two {
                    TEXTURE_DESCRIPTOR_CAPACITY.min(sampler_capacity)
                } else {
                    0
                },
                bindless_storage_resources: if tier_two { 1024 } else { 0 },
                bindless_samplers: sampler_capacity,
                max_indirect_draw_count: u32::MAX,
                shader_model: 0x0605,
                timeline_synchronization: true,
                resource_aliasing: true,
                dynamic_rendering: true,
                presentation: true,
                compression,
            };
            SemanticProfile::V1
                .admit(&capabilities)
                .map_err(|_| HalError::Unsupported)?;
            let registry = device.registryID();
            let mut stable_id = [0_u8; 16];
            stable_id[..8].copy_from_slice(&registry.to_le_bytes());
            stable_id[8..].copy_from_slice(&device.maxBufferLength().to_le_bytes());
            let name = device.name().to_string();
            let adapter = AdapterInfo::new(
                BACKEND,
                stable_id,
                name,
                format!("registry-{registry:016x}"),
                AdapterClass::Integrated,
                capabilities,
            )
            .map_err(|_| HalError::Unsupported)?;
            let allocator = Allocator::new(&AllocatorCreateDesc {
                device: device.clone(),
                debug_settings: Default::default(),
                allocation_sizes: Default::default(),
                create_residency_set: false,
            })
            .map_err(map_allocator_hal)?;
            Ok(Self {
                device,
                queue,
                allocator: Some(allocator),
                retired: Vec::new(),
                adapter,
                next_transfer_value: 1,
                completed_transfer_value: 0,
            })
        }

        pub fn adapter_info(&self) -> &AdapterInfo {
            &self.adapter
        }

        pub fn init_device(&self, surface: &NativeSurface) -> Result<AdapterInfo, HalError> {
            if surface.layer == 0 {
                return Err(HalError::InvalidArgument);
            }
            Ok(self.adapter.clone())
        }

        pub fn wait_idle(&self) -> Result<(), HalError> {
            let command = self.queue.commandBuffer().ok_or(HalError::NativeFailure)?;
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(HalError::NativeFailure);
            }
            Ok(())
        }

        fn ensure_surface_depth(
            &self,
            surface: &mut NativeSurface,
            extent: (u32, u32),
        ) -> Result<(), HalError> {
            if surface
                .depth
                .as_ref()
                .is_some_and(|depth| depth.extent == extent)
            {
                return Ok(());
            }
            let descriptor = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    MTLPixelFormat::Depth32Float,
                    extent.0 as usize,
                    extent.1 as usize,
                    false,
                )
            };
            descriptor.setStorageMode(MTLStorageMode::Private);
            descriptor.setUsage(MTLTextureUsage::RenderTarget);
            let texture = self
                .device
                .newTextureWithDescriptor(&descriptor)
                .ok_or(HalError::NativeFailure)?;
            let state_descriptor = MTLDepthStencilDescriptor::new();
            state_descriptor.setDepthCompareFunction(MTLCompareFunction::Less);
            state_descriptor.setDepthWriteEnabled(true);
            let state = self
                .device
                .newDepthStencilStateWithDescriptor(&state_descriptor)
                .ok_or(HalError::NativeFailure)?;
            surface.depth = Some(SurfaceDepth {
                texture,
                state,
                extent,
            });
            Ok(())
        }

        pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
            &self.device
        }

        /// Loads precompiled metallib products directly; source compilation is intentionally absent.
        pub fn create_shader(&self, products: &[&[u8]]) -> Result<NativeShader, HalError> {
            if products.is_empty() {
                return Err(HalError::InvalidArgument);
            }
            let mut libraries = Vec::with_capacity(products.len());
            for product in products {
                if product.is_empty() {
                    return Err(HalError::InvalidArgument);
                }
                let data = dispatch2::DispatchData::from_bytes(product);
                libraries.push(
                    self.device
                        .newLibraryWithData_error(&data)
                        .map_err(|_| HalError::NativeFailure)?,
                );
            }
            Ok(NativeShader { libraries })
        }

        pub fn destroy_shader(&self, shader: NativeShader) {
            drop(shader.libraries);
        }

        /// Resolves a named function from a precompiled metallib and creates its compute PSO.
        pub fn create_compute_pipeline(
            &self,
            shader: &NativeShader,
            product_index: usize,
            entry: &str,
        ) -> Result<NativePipeline, HalError> {
            if entry.is_empty() || entry.as_bytes().contains(&0) {
                return Err(HalError::InvalidArgument);
            }
            let library = shader
                .libraries
                .get(product_index)
                .ok_or(HalError::InvalidArgument)?;
            let name = NSString::from_str(entry);
            let function = library
                .newFunctionWithName(&name)
                .ok_or(HalError::InvalidArgument)?;
            let state = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| HalError::NativeFailure)?;
            Ok(NativePipeline { state })
        }

        fn prepare_graphics_draw(
            &self,
            draw: &NativeGraphicsDraw<'_>,
        ) -> Result<PreparedGraphics, HalError> {
            if draw.draw_count == 0
                || draw.push_constants.len() > 128
                || !draw.push_constants.len().is_multiple_of(4)
                || draw.state.topology == PrimitiveTopology::TriangleFan
                || draw
                    .bindings
                    .iter()
                    .any(|binding| binding.offset as u64 >= binding.allocation.allocation.size())
            {
                return Err(HalError::InvalidArgument);
            }
            let vertex_library = draw
                .shader
                .libraries
                .get(draw.graphics.0)
                .ok_or(HalError::InvalidArgument)?;
            let fragment_library = draw
                .shader
                .libraries
                .get(draw.graphics.2)
                .ok_or(HalError::InvalidArgument)?;
            let vertex = vertex_library
                .newFunctionWithName(&NSString::from_str(&draw.graphics.1))
                .ok_or(HalError::InvalidArgument)?;
            let fragment = fragment_library
                .newFunctionWithName(&NSString::from_str(&draw.graphics.3))
                .ok_or(HalError::InvalidArgument)?;
            let descriptor = MTLRenderPipelineDescriptor::new();
            descriptor.setVertexFunction(Some(&vertex));
            descriptor.setFragmentFunction(Some(&fragment));
            let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
            color.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            if draw.depth_required {
                descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
            }
            if draw.state.blend == BlendMode::Alpha {
                color.setBlendingEnabled(true);
                color.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                color.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            }
            let pipeline = self
                .device
                .newRenderPipelineStateWithDescriptor_error(&descriptor)
                .map_err(|_| HalError::NativeFailure)?;
            let argument_buffer = if let Some(heap) = draw.texture_heap {
                if draw
                    .textures
                    .iter()
                    .any(|texture| texture.binding >= heap.capacity)
                {
                    return Err(HalError::InvalidArgument);
                }
                let encoder =
                    unsafe { fragment.newArgumentEncoderWithBufferIndex(heap.binding as usize) };
                let buffer = self
                    .device
                    .newBufferWithLength_options(
                        encoder.encodedLength(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(HalError::NativeFailure)?;
                unsafe { encoder.setArgumentBuffer_offset(Some(&buffer), 0) };
                for texture in draw.textures {
                    let texture_index = texture.binding as usize * heap.argument_stride as usize
                        + heap.texture_argument_offset as usize;
                    let sampler_index = texture.binding as usize * heap.argument_stride as usize
                        + heap.sampler_argument_offset as usize;
                    unsafe {
                        encoder.setTexture_atIndex(Some(&texture.texture), texture_index);
                        encoder.setSamplerState_atIndex(Some(&texture.sampler), sampler_index);
                    }
                }
                Some(buffer)
            } else {
                None
            };
            Ok((pipeline, argument_buffer))
        }

        fn allocate_frame_readback(
            &mut self,
            width: u32,
            height: u32,
        ) -> Result<(NativeAllocation, u64, u64, u64, u32), HalError> {
            let tight_row = u64::from(width)
                .checked_mul(4)
                .ok_or(HalError::InvalidArgument)?;
            let row_stride = tight_row
                .checked_add(255)
                .map(|value| value & !255)
                .ok_or(HalError::InvalidArgument)?;
            let size = row_stride
                .checked_mul(u64::from(height))
                .ok_or(HalError::InvalidArgument)?;
            let request = AllocationRequest::new(size, 256, MemoryClass::Readback, true, None)
                .map_err(|_| HalError::InvalidArgument)?;
            let allocation = self.allocate(request).map_err(map_allocation_hal)?;
            Ok((allocation, tight_row, row_stride, size, height))
        }

        /// Records one complete frame into one Metal command buffer and presents only after every
        /// action has been encoded successfully.
        pub fn execute_frame(
            &mut self,
            surface: Option<(&mut NativeSurface, (u32, u32))>,
            actions: &[NativeFrameAction<'_>],
            capture_presented: bool,
        ) -> Result<Option<Vec<u8>>, HalError> {
            if actions.is_empty() {
                return Err(HalError::InvalidArgument);
            }
            let (mut surface, extent) = match surface {
                Some((surface, extent)) if extent.0 != 0 && extent.1 != 0 => {
                    (Some(surface), extent)
                }
                Some(_) => return Err(HalError::InvalidArgument),
                None => (None, (0, 0)),
            };
            let presents = actions
                .iter()
                .any(|action| matches!(action, NativeFrameAction::Present));
            let uses_surface = actions.iter().any(|action| {
                matches!(
                    action,
                    NativeFrameAction::BeginPass(_)
                        | NativeFrameAction::Graphics(_)
                        | NativeFrameAction::Present
                        | NativeFrameAction::Barrier {
                            resource: NativeFrameResource::Surface | NativeFrameResource::Depth,
                            ..
                        }
                )
            });
            if uses_surface && surface.is_none() || (uses_surface || capture_presented) && !presents
            {
                return Err(HalError::InvalidArgument);
            }
            for action in actions {
                if let NativeFrameAction::Wait(token) = action
                    && (token.queue != QueueKind::Transfer
                        || token.value > self.completed_transfer_value)
                {
                    return Err(HalError::InvalidArgument);
                }
            }
            if actions.iter().any(|action| {
                matches!(
                    action,
                    NativeFrameAction::BeginPass(pass) if pass.depth.is_some()
                )
            }) {
                self.ensure_surface_depth(
                    surface.as_deref_mut().ok_or(HalError::InvalidArgument)?,
                    extent,
                )?;
            }
            let mut prepared = Vec::with_capacity(actions.len());
            let mut readbacks = Vec::new();
            let mut pass_active = false;
            let mut saw_present = false;
            for action in actions {
                if saw_present {
                    for (allocation, _, _, _, _) in readbacks {
                        let _ = self.free(allocation);
                    }
                    return Err(HalError::InvalidArgument);
                }
                let item = match action {
                    NativeFrameAction::Wait(_) => Ok(PreparedFrameAction::None),
                    NativeFrameAction::Barrier { barrier, resource } => {
                        let valid = match resource {
                            NativeFrameResource::Buffer(allocation) => {
                                matches!(barrier.range, ez_gfx_hal::ExecutionRange::Buffer(_))
                                    && allocation.allocation.size() != 0
                            }
                            NativeFrameResource::Texture(texture) => {
                                matches!(barrier.range, ez_gfx_hal::ExecutionRange::Image(_))
                                    && texture.allocation.size() != 0
                            }
                            NativeFrameResource::Surface | NativeFrameResource::Depth => {
                                matches!(barrier.range, ez_gfx_hal::ExecutionRange::Image(_))
                            }
                        };
                        if pass_active || !valid {
                            Err(HalError::InvalidArgument)
                        } else {
                            Ok(PreparedFrameAction::None)
                        }
                    }
                    NativeFrameAction::BeginPass(pass) => {
                        let invalid = pass_active
                            || pass.colors.len() != 1
                            || pass.samples != 1
                            || pass.area[0]
                                .checked_add(pass.area[2])
                                .is_none_or(|end| end > extent.0)
                            || pass.area[1]
                                .checked_add(pass.area[3])
                                .is_none_or(|end| end > extent.1);
                        if invalid {
                            Err(HalError::InvalidArgument)
                        } else {
                            pass_active = true;
                            Ok(PreparedFrameAction::None)
                        }
                    }
                    NativeFrameAction::Compute(dispatch) => {
                        if pass_active
                            || dispatch.groups.contains(&0)
                            || dispatch.push_constants.len() > 128
                            || !dispatch.push_constants.len().is_multiple_of(4)
                            || dispatch.bindings.iter().any(|binding| {
                                binding.offset as u64 >= binding.allocation.allocation.size()
                            })
                        {
                            Err(HalError::InvalidArgument)
                        } else {
                            self.create_compute_pipeline(
                                dispatch.shader,
                                dispatch.product_index,
                                dispatch.entry,
                            )
                            .map(PreparedFrameAction::Compute)
                        }
                    }
                    NativeFrameAction::Graphics(draw) => {
                        if !pass_active
                            || draw.depth_required
                                && surface
                                    .as_ref()
                                    .is_none_or(|surface| surface.depth.is_none())
                        {
                            Err(HalError::InvalidArgument)
                        } else {
                            self.prepare_graphics_draw(draw)
                                .map(
                                    |(pipeline, argument_buffer)| PreparedFrameAction::Graphics {
                                        pipeline,
                                        argument_buffer,
                                    },
                                )
                        }
                    }
                    NativeFrameAction::TextureReadback { width, height, .. } => {
                        if pass_active || *width == 0 || *height == 0 {
                            Err(HalError::InvalidArgument)
                        } else {
                            self.allocate_frame_readback(*width, *height)
                                .map(|readback| {
                                    readbacks.push(readback);
                                    PreparedFrameAction::None
                                })
                        }
                    }
                    NativeFrameAction::EndPass => {
                        if !pass_active {
                            Err(HalError::InvalidArgument)
                        } else {
                            pass_active = false;
                            Ok(PreparedFrameAction::None)
                        }
                    }
                    NativeFrameAction::Present => {
                        if pass_active || saw_present {
                            Err(HalError::InvalidArgument)
                        } else {
                            saw_present = true;
                            if capture_presented {
                                match self.allocate_frame_readback(extent.0, extent.1) {
                                    Ok(readback) => readbacks.push(readback),
                                    Err(error) => {
                                        for (allocation, _, _, _, _) in readbacks {
                                            let _ = self.free(allocation);
                                        }
                                        return Err(error);
                                    }
                                }
                            }
                            Ok(PreparedFrameAction::None)
                        }
                    }
                };
                match item {
                    Ok(item) => prepared.push(item),
                    Err(error) => {
                        for (allocation, _, _, _, _) in readbacks {
                            let _ = self.free(allocation);
                        }
                        return Err(error);
                    }
                }
            }
            if pass_active || saw_present != presents {
                for (allocation, _, _, _, _) in readbacks {
                    let _ = self.free(allocation);
                }
                return Err(HalError::InvalidArgument);
            }
            let drawable = (|| {
                if uses_surface {
                    let surface = surface.as_ref().ok_or(HalError::InvalidArgument)?;
                    let layer = unsafe { &*(surface.layer as *const CAMetalLayer) };
                    layer.setDevice(Some(&self.device));
                    layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                    let drawable = layer.nextDrawable().ok_or(HalError::NotReady)?;
                    let texture = drawable.texture();
                    if texture.width() != extent.0 as usize || texture.height() != extent.1 as usize
                    {
                        return Err(HalError::NotReady);
                    }
                    Ok(Some(drawable))
                } else {
                    Ok(None)
                }
            })();
            let drawable = match drawable {
                Ok(drawable) => drawable,
                Err(error) => {
                    for (allocation, _, _, _, _) in readbacks {
                        let _ = self.free(allocation);
                    }
                    return Err(error);
                }
            };
            let drawable_texture = drawable.as_ref().map(|drawable| drawable.texture());
            let command = match self.queue.commandBuffer() {
                Some(command) => command,
                None => {
                    for (allocation, _, _, _, _) in readbacks {
                        let _ = self.free(allocation);
                    }
                    return Err(HalError::NativeFailure);
                }
            };
            let mut render_encoder = None;
            let mut render_area = [0_u32; 4];
            let mut presented = false;
            let mut readback_index = 0_usize;
            let encoded = (|| -> Result<(), HalError> {
                for (action_index, action) in actions.iter().enumerate() {
                    match action {
                        NativeFrameAction::Wait(_) => {}
                        NativeFrameAction::Barrier { barrier, resource } => {
                            if render_encoder.is_some() {
                                return Err(HalError::InvalidArgument);
                            }
                            match resource {
                                NativeFrameResource::Buffer(allocation) => {
                                    if !matches!(
                                        barrier.range,
                                        ez_gfx_hal::ExecutionRange::Buffer(_)
                                    ) || allocation.allocation.size() == 0
                                    {
                                        return Err(HalError::InvalidArgument);
                                    }
                                }
                                NativeFrameResource::Texture(texture) => {
                                    if !matches!(
                                        barrier.range,
                                        ez_gfx_hal::ExecutionRange::Image(_)
                                    ) || texture.allocation.size() == 0
                                    {
                                        return Err(HalError::InvalidArgument);
                                    }
                                }
                                NativeFrameResource::Surface | NativeFrameResource::Depth => {
                                    if !matches!(
                                        barrier.range,
                                        ez_gfx_hal::ExecutionRange::Image(_)
                                    ) {
                                        return Err(HalError::InvalidArgument);
                                    }
                                }
                            }
                            // Separate encoders in one command buffer are ordered Metal hazard
                            // boundaries. Every transition action is therefore lowered by requiring
                            // the prior encoder to be closed before the next node is opened.
                        }
                        NativeFrameAction::BeginPass(pass) => {
                            if render_encoder.is_some()
                                || pass.colors.len() != 1
                                || pass.samples != 1
                                || pass.area[0]
                                    .checked_add(pass.area[2])
                                    .is_none_or(|end| end > extent.0)
                                || pass.area[1]
                                    .checked_add(pass.area[3])
                                    .is_none_or(|end| end > extent.1)
                            {
                                return Err(HalError::InvalidArgument);
                            }
                            let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
                            let color = unsafe {
                                descriptor.colorAttachments().objectAtIndexedSubscript(0)
                            };
                            color.setTexture(Some(
                                drawable_texture.as_ref().ok_or(HalError::InvalidArgument)?,
                            ));
                            color.setLoadAction(match pass.load {
                                AttachmentLoadOp::Load => MTLLoadAction::Load,
                                AttachmentLoadOp::Clear => MTLLoadAction::Clear,
                                AttachmentLoadOp::Discard => MTLLoadAction::DontCare,
                            });
                            color.setStoreAction(match pass.store {
                                AttachmentStoreOp::Store => MTLStoreAction::Store,
                                AttachmentStoreOp::Discard => MTLStoreAction::DontCare,
                            });
                            color.setClearColor(MTLClearColor {
                                red: 0.1,
                                green: 0.1,
                                blue: 0.1,
                                alpha: 1.0,
                            });
                            if pass.depth.is_some() {
                                let depth = surface
                                    .as_ref()
                                    .and_then(|surface| surface.depth.as_ref())
                                    .ok_or(HalError::NotReady)?;
                                let attachment = descriptor.depthAttachment();
                                attachment.setTexture(Some(&depth.texture));
                                attachment.setLoadAction(match pass.load {
                                    AttachmentLoadOp::Load => MTLLoadAction::Load,
                                    AttachmentLoadOp::Clear => MTLLoadAction::Clear,
                                    AttachmentLoadOp::Discard => MTLLoadAction::DontCare,
                                });
                                attachment.setStoreAction(match pass.store {
                                    AttachmentStoreOp::Store => MTLStoreAction::Store,
                                    AttachmentStoreOp::Discard => MTLStoreAction::DontCare,
                                });
                                attachment.setClearDepth(1.0);
                            }
                            render_area = pass.area;
                            render_encoder = Some(
                                command
                                    .renderCommandEncoderWithDescriptor(&descriptor)
                                    .ok_or(HalError::NativeFailure)?,
                            );
                        }
                        NativeFrameAction::Compute(dispatch) => {
                            if render_encoder.is_some()
                                || dispatch.groups.contains(&0)
                                || dispatch.push_constants.len() > 128
                                || !dispatch.push_constants.len().is_multiple_of(4)
                            {
                                return Err(HalError::InvalidArgument);
                            }
                            let PreparedFrameAction::Compute(pipeline) = &prepared[action_index]
                            else {
                                return Err(HalError::InvalidArgument);
                            };
                            let encoder = command
                                .computeCommandEncoder()
                                .ok_or(HalError::NativeFailure)?;
                            encoder.setComputePipelineState(&pipeline.state);
                            for binding in dispatch.bindings {
                                if binding.offset as u64 >= binding.allocation.allocation.size() {
                                    return Err(HalError::InvalidArgument);
                                }
                                unsafe {
                                    encoder.setBuffer_offset_atIndex(
                                        Some(&binding.allocation.buffer),
                                        binding.offset,
                                        binding.index,
                                    )
                                };
                            }
                            if let Some(bytes) = core::ptr::NonNull::new(
                                dispatch.push_constants.as_ptr() as *mut c_void,
                            ) && !dispatch.push_constants.is_empty()
                            {
                                unsafe {
                                    encoder.setBytes_length_atIndex(
                                        bytes,
                                        dispatch.push_constants.len(),
                                        0,
                                    )
                                };
                            }
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                MTLSize {
                                    width: dispatch.groups[0] as usize,
                                    height: dispatch.groups[1] as usize,
                                    depth: dispatch.groups[2] as usize,
                                },
                                MTLSize {
                                    width: 1,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            encoder.endEncoding();
                        }
                        NativeFrameAction::Graphics(draw) => {
                            let encoder =
                                render_encoder.as_ref().ok_or(HalError::InvalidArgument)?;
                            let PreparedFrameAction::Graphics {
                                pipeline,
                                argument_buffer,
                            } = &prepared[action_index]
                            else {
                                return Err(HalError::InvalidArgument);
                            };
                            encoder.setRenderPipelineState(pipeline);
                            if draw.depth_required {
                                let depth = surface
                                    .as_ref()
                                    .and_then(|surface| surface.depth.as_ref())
                                    .ok_or(HalError::InvalidArgument)?;
                                encoder.setDepthStencilState(Some(&depth.state));
                            }
                            encoder.setCullMode(match draw.state.cull {
                                CullMode::None => MTLCullMode::None,
                                CullMode::Front => MTLCullMode::Front,
                                CullMode::Back => MTLCullMode::Back,
                            });
                            encoder.setFrontFacingWinding(match draw.state.front_face {
                                FrontFace::CounterClockwise => MTLWinding::CounterClockwise,
                                FrontFace::Clockwise => MTLWinding::Clockwise,
                            });
                            encoder.setViewport(objc2_metal::MTLViewport {
                                originX: render_area[0] as f64,
                                originY: render_area[1] as f64,
                                width: render_area[2] as f64,
                                height: render_area[3] as f64,
                                znear: 0.0,
                                zfar: 1.0,
                            });
                            for binding in draw.bindings {
                                if binding.offset as u64 >= binding.allocation.allocation.size() {
                                    return Err(HalError::InvalidArgument);
                                }
                                unsafe {
                                    encoder.setVertexBuffer_offset_atIndex(
                                        Some(&binding.allocation.buffer),
                                        binding.offset,
                                        binding.index,
                                    );
                                    encoder.setFragmentBuffer_offset_atIndex(
                                        Some(&binding.allocation.buffer),
                                        binding.offset,
                                        binding.index,
                                    );
                                }
                            }
                            if let (Some(heap), Some(buffer)) =
                                (draw.texture_heap, argument_buffer.as_ref())
                            {
                                for texture in draw.textures {
                                    let resource = <ProtocolObject<dyn MTLTexture> as AsRef<
                                        ProtocolObject<dyn MTLResource>,
                                    >>::as_ref(
                                        &*texture.texture
                                    );
                                    encoder.useResource_usage_stages(
                                        resource,
                                        MTLResourceUsage::Read,
                                        MTLRenderStages::Fragment,
                                    );
                                }
                                unsafe {
                                    encoder.setFragmentBuffer_offset_atIndex(
                                        Some(buffer),
                                        0,
                                        heap.binding as usize,
                                    );
                                }
                            }
                            if let Some(bytes) =
                                core::ptr::NonNull::new(draw.push_constants.as_ptr() as *mut c_void)
                                && !draw.push_constants.is_empty()
                            {
                                unsafe {
                                    encoder.setVertexBytes_length_atIndex(
                                        bytes,
                                        draw.push_constants.len(),
                                        0,
                                    );
                                    encoder.setFragmentBytes_length_atIndex(
                                        bytes,
                                        draw.push_constants.len(),
                                        0,
                                    );
                                }
                            }
                            let primitive = match draw.state.topology {
                                PrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
                                PrimitiveTopology::PointList => MTLPrimitiveType::Point,
                                PrimitiveTopology::LineList => MTLPrimitiveType::Line,
                                PrimitiveTopology::LineStrip => MTLPrimitiveType::LineStrip,
                                PrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
                                PrimitiveTopology::TriangleFan => {
                                    return Err(HalError::Unsupported);
                                }
                            };
                            for command_index in 0..draw.draw_count {
                                unsafe {
                                    encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                                    primitive,
                                    MTLIndexType::UInt32,
                                    &draw.index.buffer,
                                    0,
                                    &draw.indirect.buffer,
                                    command_index as usize * 20,
                                )
                                };
                            }
                        }
                        NativeFrameAction::TextureReadback {
                            texture,
                            width,
                            height,
                        } => {
                            let (allocation, _, row_stride, size, _) = readbacks
                                .get(readback_index)
                                .ok_or(HalError::InvalidArgument)?;
                            let blit = command
                                .blitCommandEncoder()
                                .ok_or(HalError::NativeFailure)?;
                            unsafe {
                                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                                &texture.texture,
                                0,
                                0,
                                MTLOrigin { x: 0, y: 0, z: 0 },
                                MTLSize {
                                    width: *width as usize,
                                    height: *height as usize,
                                    depth: 1,
                                },
                                &allocation.buffer,
                                0,
                                *row_stride as usize,
                                *size as usize,
                            );
                                blit.endEncoding();
                            }
                            readback_index += 1;
                        }
                        NativeFrameAction::EndPass => {
                            let encoder = render_encoder.take().ok_or(HalError::InvalidArgument)?;
                            encoder.endEncoding();
                        }
                        NativeFrameAction::Present => {
                            if render_encoder.is_some() || presented {
                                return Err(HalError::InvalidArgument);
                            }
                            if capture_presented {
                                let (allocation, _, row_stride, size, _) = readbacks
                                    .get(readback_index)
                                    .ok_or(HalError::InvalidArgument)?;
                                let blit = command
                                    .blitCommandEncoder()
                                    .ok_or(HalError::NativeFailure)?;
                                unsafe {
                                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                                    drawable_texture.as_ref().ok_or(HalError::InvalidArgument)?,
                                    0,
                                    0,
                                    MTLOrigin { x: 0, y: 0, z: 0 },
                                    MTLSize {
                                        width: extent.0 as usize,
                                        height: extent.1 as usize,
                                        depth: 1,
                                    },
                                    &allocation.buffer,
                                    0,
                                    *row_stride as usize,
                                    *size as usize,
                                );
                                    blit.endEncoding();
                                }
                                readback_index += 1;
                            }
                            let drawable = drawable.as_ref().ok_or(HalError::InvalidArgument)?;
                            let drawable_ref = <ProtocolObject<dyn CAMetalDrawable> as AsRef<
                                ProtocolObject<dyn objc2_metal::MTLDrawable>,
                            >>::as_ref(drawable);
                            command.presentDrawable(drawable_ref);
                            presented = true;
                        }
                    }
                }
                if render_encoder.is_some()
                    || presented != presents
                    || readback_index != readbacks.len()
                {
                    return Err(HalError::InvalidArgument);
                }
                Ok(())
            })();
            if let Err(error) = encoded {
                for (allocation, _, _, _, _) in readbacks {
                    let _ = self.free(allocation);
                }
                return Err(error);
            }
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                for (allocation, _, _, _, _) in readbacks {
                    let _ = self.free(allocation);
                }
                return Err(HalError::NativeFailure);
            }
            let mut final_pixels = None;
            let mut remaining = readbacks.into_iter();
            while let Some((mut allocation, tight_row, row_stride, size, height)) = remaining.next()
            {
                let copied = (|| {
                    self.invalidate(&mut allocation, 0, size)
                        .map_err(map_allocation_hal)?;
                    let source = self.mapped_slice(&allocation).map_err(map_allocation_hal)?;
                    let packed_size = tight_row
                        .checked_mul(u64::from(height))
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or(HalError::InvalidArgument)?;
                    let mut packed = Vec::with_capacity(packed_size);
                    for row in 0..height as usize {
                        let start = row
                            .checked_mul(row_stride as usize)
                            .ok_or(HalError::InvalidArgument)?;
                        let end = start
                            .checked_add(tight_row as usize)
                            .ok_or(HalError::InvalidArgument)?;
                        packed.extend_from_slice(
                            source.get(start..end).ok_or(HalError::NativeFailure)?,
                        );
                    }
                    Ok(packed)
                })();
                let freed = self.free(allocation).map_err(map_allocation_hal);
                let mut packed = match (copied, freed) {
                    (Ok(packed), Ok(())) => packed,
                    (Err(error), _) | (_, Err(error)) => {
                        for (allocation, _, _, _, _) in remaining {
                            let _ = self.free(allocation);
                        }
                        return Err(error);
                    }
                };
                if capture_presented && remaining.len() == 0 {
                    for pixel in packed.chunks_exact_mut(4) {
                        pixel.swap(0, 2);
                    }
                    surface
                        .as_deref_mut()
                        .ok_or(HalError::InvalidArgument)?
                        .presented_rgba8 = packed.clone();
                }
                final_pixels = Some(packed);
            }
            Ok(final_pixels)
        }

        pub fn create_texture_rgba8(
            &mut self,
            mips: &[ImageMip<'_>],
            binding: u32,
            sampler_desc: TextureSamplerDesc,
        ) -> Result<(NativeTexture, Vec<CompletionToken>), AllocationError> {
            validate_rgba8_mips(mips).map_err(|_| AllocationError::ZeroSize)?;
            if binding >= TEXTURE_DESCRIPTOR_CAPACITY {
                return Err(AllocationError::ZeroSize);
            }
            let width = mips[0].width;
            let height = mips[0].height;
            // SAFETY: dimensions and mip count were validated against every bounded payload.
            let desc = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    MTLPixelFormat::RGBA8Unorm,
                    width as usize,
                    height as usize,
                    mips.len() > 1,
                )
            };
            unsafe { desc.setMipmapLevelCount(mips.len()) };
            desc.setUsage(MTLTextureUsage::ShaderRead);
            desc.setStorageMode(MTLStorageMode::Private);
            let allocation_desc =
                AllocationCreateDesc::texture(&self.device, "ez-gfx-texture", &desc);
            let allocation = self
                .allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .allocate(&allocation_desc)
                .map_err(map_allocator)?;
            let texture = match unsafe {
                allocation
                    .heap()
                    .newTextureWithDescriptor_offset(&desc, allocation.offset() as usize)
            } {
                Some(texture) => texture,
                None => {
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(&allocation)
                        .map_err(map_allocator)?;
                    return Err(AllocationError::OutOfMemory);
                }
            };
            let uploaded = (|| {
                let sampler_descriptor = MTLSamplerDescriptor::new();
                sampler_descriptor.setMinFilter(match sampler_desc.min_filter {
                    SamplerFilter::Nearest => MTLSamplerMinMagFilter::Nearest,
                    SamplerFilter::Linear => MTLSamplerMinMagFilter::Linear,
                });
                sampler_descriptor.setMagFilter(match sampler_desc.mag_filter {
                    SamplerFilter::Nearest => MTLSamplerMinMagFilter::Nearest,
                    SamplerFilter::Linear => MTLSamplerMinMagFilter::Linear,
                });
                let address = |mode| match mode {
                    SamplerAddressMode::Clamp => MTLSamplerAddressMode::ClampToEdge,
                    SamplerAddressMode::Repeat => MTLSamplerAddressMode::Repeat,
                };
                sampler_descriptor.setSAddressMode(address(sampler_desc.address_u));
                sampler_descriptor.setTAddressMode(address(sampler_desc.address_v));
                sampler_descriptor.setRAddressMode(address(sampler_desc.address_w));
                sampler_descriptor.setMaxAnisotropy(sampler_desc.max_anisotropy as usize);
                sampler_descriptor.setSupportArgumentBuffers(true);
                let sampler = self
                    .device
                    .newSamplerStateWithDescriptor(&sampler_descriptor)
                    .ok_or(AllocationError::NativeFailure)?;
                let mut completions = Vec::with_capacity(mips.len());
                for (level, mip) in mips.iter().enumerate() {
                    let size = mip.bytes.len() as u64;
                    let mut upload = self.allocate(
                        AllocationRequest::new(size, 4, MemoryClass::Upload, true, None)
                            .map_err(|_| AllocationError::ZeroSize)?,
                    )?;
                    let submitted = (|| {
                        self.mapped_slice_mut(&mut upload)?[..mip.bytes.len()]
                            .copy_from_slice(mip.bytes);
                        self.flush(&mut upload, 0, size)?;
                        let command = self
                            .queue
                            .commandBuffer()
                            .ok_or(AllocationError::NativeFailure)?;
                        let blit = command
                            .blitCommandEncoder()
                            .ok_or(AllocationError::NativeFailure)?;
                        unsafe {
                            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&upload.buffer, 0, mip.width as usize * 4, size as usize, MTLSize { width: mip.width as usize, height: mip.height as usize, depth: 1 }, &texture, 0, level, MTLOrigin { x: 0, y: 0, z: 0 });
                            blit.endEncoding();
                        }
                        command.commit();
                        let value = self.next_transfer_value;
                        self.next_transfer_value =
                            value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
                        command.waitUntilCompleted();
                        if command.status() != MTLCommandBufferStatus::Completed
                            || command.error().is_some()
                        {
                            return Err(AllocationError::NativeFailure);
                        }
                        self.completed_transfer_value = value;
                        CompletionToken::new(QueueKind::Transfer, value)
                            .map_err(|_| AllocationError::NativeFailure)
                    })();
                    let freed = self.free(upload);
                    let completion = match (submitted, freed) {
                        (Ok(completion), Ok(())) => completion,
                        (Err(error), _) | (_, Err(error)) => return Err(error),
                    };
                    completions.push(completion);
                }
                Ok((sampler, completions))
            })();
            match uploaded {
                Ok((sampler, completions)) => Ok((
                    NativeTexture {
                        texture,
                        allocation,
                        sampler,
                        binding,
                    },
                    completions,
                )),
                Err(error) => {
                    drop(texture);
                    self.allocator
                        .as_mut()
                        .ok_or(AllocationError::NativeFailure)?
                        .free(&allocation)
                        .map_err(map_allocator)?;
                    Err(error)
                }
            }
        }

        /// Copies a private RGBA8 texture into shared CPU-visible storage and returns packed rows.
        pub fn readback_texture_rgba8(
            &mut self,
            texture: &NativeTexture,
            width: u32,
            height: u32,
        ) -> Result<Vec<u8>, AllocationError> {
            let size = u64::from(width)
                .checked_mul(u64::from(height))
                .and_then(|value| value.checked_mul(4))
                .ok_or(AllocationError::ZeroSize)?;
            if width == 0 || height == 0 {
                return Err(AllocationError::ZeroSize);
            }
            let mut readback = self.allocate(
                AllocationRequest::new(size, 4, MemoryClass::Readback, true, None)
                    .map_err(|_| AllocationError::ZeroSize)?,
            )?;
            let result = (|| {
                let command = self
                    .queue
                    .commandBuffer()
                    .ok_or(AllocationError::NativeFailure)?;
                let blit = command
                    .blitCommandEncoder()
                    .ok_or(AllocationError::NativeFailure)?;
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(&texture.texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: width as usize, height: height as usize, depth: 1 }, &readback.buffer, 0, width as usize * 4, size as usize);
                    blit.endEncoding();
                }
                command.commit();
                command.waitUntilCompleted();
                if command.status() != MTLCommandBufferStatus::Completed
                    || command.error().is_some()
                {
                    return Err(AllocationError::NativeFailure);
                }
                self.invalidate(&mut readback, 0, size)?;
                Ok(self.mapped_slice(&readback)?[..size as usize].to_vec())
            })();
            match (result, self.free(readback)) {
                (Ok(pixels), Ok(())) => Ok(pixels),
                (Err(error), _) | (_, Err(error)) => Err(error),
            }
        }

        /// Acquires and presents one drawable from the borrowed CAMetalLayer; zero extent is minimized.
        pub fn acquire_present(
            &self,
            surface: &NativeSurface,
            width: u32,
            height: u32,
        ) -> Result<(), HalError> {
            if width == 0 || height == 0 {
                return Err(HalError::NotReady);
            }
            // SAFETY: the host promises that the opaque platform handle is a live CAMetalLayer.
            let layer = unsafe { &*(surface.layer as *const CAMetalLayer) };
            layer.setDevice(Some(&self.device));
            let drawable = layer.nextDrawable().ok_or(HalError::NotReady)?;
            let command = self.queue.commandBuffer().ok_or(HalError::NativeFailure)?;
            let drawable = <ProtocolObject<dyn CAMetalDrawable> as AsRef<
                ProtocolObject<dyn objc2_metal::MTLDrawable>,
            >>::as_ref(&*drawable);
            command.presentDrawable(drawable);
            command.commit();
            Ok(())
        }

        pub fn destroy_texture(&mut self, texture: NativeTexture) -> Result<(), AllocationError> {
            drop(texture.texture);
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(&texture.allocation)
                .map_err(map_allocator)
        }
    }

    impl MemoryAllocator for NativeContext {
        type Allocation = NativeAllocation;

        fn allocate(
            &mut self,
            request: AllocationRequest,
        ) -> Result<Self::Allocation, AllocationError> {
            let location = match request.memory_class {
                MemoryClass::Device | MemoryClass::Transient => MemoryLocation::GpuOnly,
                MemoryClass::Upload => MemoryLocation::CpuToGpu,
                MemoryClass::Readback => MemoryLocation::GpuToCpu,
            };
            let desc =
                AllocationCreateDesc::buffer(&self.device, "ez-gfx-buffer", request.size, location);
            if desc.alignment < request.alignment {
                return Err(AllocationError::InvalidAlignment);
            }
            let allocation = self
                .allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .allocate(&desc)
                .map_err(map_allocator)?;
            let heap = unsafe { allocation.heap() };
            let buffer = unsafe {
                heap.newBufferWithLength_options_offset(
                    allocation.size() as usize,
                    heap.resourceOptions(),
                    allocation.offset() as usize,
                )
            }
            .ok_or(AllocationError::OutOfMemory)?;
            let mapped_address = if request.mapped {
                let address = buffer.contents().as_ptr() as usize;
                if address == 0 {
                    return Err(AllocationError::NotHostVisible);
                }
                address
            } else {
                0
            };
            Ok(NativeAllocation {
                buffer,
                allocation,
                mapped_address,
            })
        }

        fn mapped_slice<'a>(
            &self,
            allocation: &'a Self::Allocation,
        ) -> Result<&'a [u8], AllocationError> {
            if allocation.mapped_address == 0 || allocation.allocation.size() > isize::MAX as u64 {
                return Err(AllocationError::NotHostVisible);
            }
            Ok(unsafe {
                core::slice::from_raw_parts(
                    allocation.mapped_address as *const u8,
                    allocation.allocation.size() as usize,
                )
            })
        }

        fn mapped_slice_mut<'a>(
            &mut self,
            allocation: &'a mut Self::Allocation,
        ) -> Result<&'a mut [u8], AllocationError> {
            if allocation.mapped_address == 0 || allocation.allocation.size() > isize::MAX as u64 {
                return Err(AllocationError::NotHostVisible);
            }
            Ok(unsafe {
                core::slice::from_raw_parts_mut(
                    allocation.mapped_address as *mut u8,
                    allocation.allocation.size() as usize,
                )
            })
        }

        fn flush(
            &mut self,
            allocation: &mut Self::Allocation,
            offset: u64,
            size: u64,
        ) -> Result<(), AllocationError> {
            validate_range(allocation.allocation.size(), offset, size)?;
            if allocation.mapped_address == 0 {
                return Err(AllocationError::NotHostVisible);
            }
            Ok(())
        }

        fn invalidate(
            &mut self,
            allocation: &mut Self::Allocation,
            offset: u64,
            size: u64,
        ) -> Result<(), AllocationError> {
            validate_range(allocation.allocation.size(), offset, size)?;
            if allocation.mapped_address == 0 {
                return Err(AllocationError::NotHostVisible);
            }
            Ok(())
        }

        fn free(&mut self, allocation: Self::Allocation) -> Result<(), AllocationError> {
            drop(allocation.buffer);
            self.allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .free(&allocation.allocation)
                .map_err(map_allocator)
        }

        fn retire(
            &mut self,
            allocation: Self::Allocation,
            completion: CompletionToken,
        ) -> Result<(), AllocationError> {
            self.retired.push(RetiredAllocation {
                allocation,
                completion,
            });
            Ok(())
        }

        fn reclaim(&mut self, queue: QueueKind, completed: u64) -> Result<usize, AllocationError> {
            let mut count = 0;
            let mut index = 0;
            while index < self.retired.len() {
                if self.retired[index].completion.queue == queue
                    && self.retired[index].completion.value <= completed
                {
                    let retired = self.retired.swap_remove(index);
                    self.free(retired.allocation)?;
                    count += 1;
                } else {
                    index += 1;
                }
            }
            Ok(count)
        }
    }

    impl BufferTransfer for NativeContext {
        fn copy_buffer(
            &mut self,
            source: &Self::Allocation,
            destination: &Self::Allocation,
            source_offset: u64,
            destination_offset: u64,
            size: u64,
        ) -> Result<CompletionToken, AllocationError> {
            validate_range(source.allocation.size(), source_offset, size)?;
            validate_range(destination.allocation.size(), destination_offset, size)?;
            let command = self
                .queue
                .commandBuffer()
                .ok_or(AllocationError::NativeFailure)?;
            let blit = command
                .blitCommandEncoder()
                .ok_or(AllocationError::NativeFailure)?;
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &source.buffer,
                    source_offset as usize,
                    &destination.buffer,
                    destination_offset as usize,
                    size as usize,
                );
                blit.endEncoding();
            }
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(AllocationError::NativeFailure);
            }
            let value = self.next_transfer_value;
            self.next_transfer_value =
                value.checked_add(1).ok_or(AllocationError::NativeFailure)?;
            self.completed_transfer_value = value;
            CompletionToken::new(QueueKind::Transfer, value)
                .map_err(|_| AllocationError::NativeFailure)
        }

        fn completed_transfer_value(&self) -> Result<u64, AllocationError> {
            Ok(self.completed_transfer_value)
        }
    }
    impl Drop for NativeContext {
        fn drop(&mut self) {
            let _ = self.wait_idle();
            while let Some(retired) = self.retired.pop() {
                let _ = self.free(retired.allocation);
            }
            drop(self.allocator.take());
        }
    }

    fn validate_range(length: u64, offset: u64, size: u64) -> Result<(), AllocationError> {
        if size == 0 || offset.checked_add(size).is_none_or(|end| end > length) {
            return Err(AllocationError::NativeFailure);
        }
        Ok(())
    }
    fn map_allocation_hal(error: AllocationError) -> HalError {
        match error {
            AllocationError::OutOfMemory => HalError::OutOfMemory,
            AllocationError::ZeroSize
            | AllocationError::InvalidAlignment
            | AllocationError::InvalidAliasClass => HalError::InvalidArgument,
            AllocationError::DeviceLost => HalError::DeviceLost,
            AllocationError::NotHostVisible | AllocationError::NativeFailure => {
                HalError::NativeFailure
            }
        }
    }

    fn map_allocator_hal(error: gpu_allocator::AllocationError) -> HalError {
        match error {
            gpu_allocator::AllocationError::OutOfMemory => HalError::OutOfMemory,
            _ => HalError::NativeFailure,
        }
    }
    fn map_allocator(error: gpu_allocator::AllocationError) -> AllocationError {
        match error {
            gpu_allocator::AllocationError::OutOfMemory => AllocationError::OutOfMemory,
            _ => AllocationError::NativeFailure,
        }
    }
}
