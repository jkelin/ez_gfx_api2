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
        AllocationError, AllocationRequest, BlendMode, BufferTransfer, CompletionToken, CullMode,
        DynamicPipelineState, FrontFace, HalError, ImageMip, MemoryAllocator, MemoryClass,
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
        MTLCreateSystemDefaultDevice, MTLCullMode, MTLDepthStencilDescriptor, MTLDevice,
        MTLFunction, MTLHeap, MTLIndexType, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat,
        MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
        MTLRenderPipelineDescriptor, MTLRenderStages, MTLResource, MTLResourceOptions,
        MTLResourceUsage, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
        MTLSamplerState, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
        MTLTextureUsage, MTLWinding,
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

    struct RetiredAllocation {
        allocation: NativeAllocation,
        completion: CompletionToken,
    }

    pub struct NativeSurface {
        layer: usize,
        presented_rgba8: Vec<u8>,
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

        /// Dispatches a validated compute grid and waits before transient PSO destruction.
        pub fn dispatch_compute(
            &self,
            pipeline: &NativePipeline,
            groups: [u32; 3],
            push_constants: &[u8],
            bindings: &[NativeBufferBinding<'_>],
        ) -> Result<(), HalError> {
            if groups.contains(&0) || push_constants.len() > 128 {
                return Err(HalError::InvalidArgument);
            }
            let command = self.queue.commandBuffer().ok_or(HalError::NativeFailure)?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or(HalError::NativeFailure)?;
            encoder.setComputePipelineState(&pipeline.state);
            for binding in bindings {
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
            if let Some(bytes) =
                core::ptr::NonNull::new(push_constants.as_ptr() as *mut core::ffi::c_void)
                && !push_constants.is_empty()
            {
                unsafe { encoder.setBytes_length_atIndex(bytes, push_constants.len(), 0) };
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups[0] as usize,
                    height: groups[1] as usize,
                    depth: groups[2] as usize,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                return Err(HalError::NativeFailure);
            }
            Ok(())
        }
        /// Creates an RGBA8 mip chain and publishes one completion value per uploaded level.
        /// Resolves and dispatches one artifact compute product without retaining transient pipeline state.
        pub fn dispatch_compute_shader(
            &self,
            shader: &NativeShader,
            product_index: usize,
            entry: &str,
            groups: [u32; 3],
            push_constants: &[u8],
            bindings: &[NativeBufferBinding<'_>],
        ) -> Result<(), HalError> {
            let pipeline = self.create_compute_pipeline(shader, product_index, entry)?;
            self.dispatch_compute(&pipeline, groups, push_constants, bindings)
        }

        /// Compiles an exact metallib graphics pair, executes indexed-indirect draws, and presents the drawable.
        #[allow(clippy::too_many_arguments)]
        pub fn draw_indexed_shader(
            &mut self,
            surface: &mut NativeSurface,
            shader: &NativeShader,
            graphics: &(usize, String, usize, String),
            depth_required: bool,
            texture_heap: Option<ShaderTextureHeapLayout>,
            state: DynamicPipelineState,
            index: &NativeAllocation,
            indirect: &NativeAllocation,
            draw_count: u32,
            push_constants: &[u8],
            extent: (u32, u32),
            bindings: &[NativeBufferBinding<'_>],
            textures: &[&NativeTexture],
            capture_presented: bool,
        ) -> Result<(), HalError> {
            if draw_count == 0
                || extent.0 == 0
                || extent.1 == 0
                || push_constants.len() > 128
                || !push_constants.len().is_multiple_of(4)
            {
                return Err(HalError::InvalidArgument);
            }
            if graphics.1.is_empty()
                || graphics.3.is_empty()
                || graphics.1.as_bytes().contains(&0)
                || graphics.3.as_bytes().contains(&0)
            {
                return Err(HalError::InvalidArgument);
            }
            let vertex_library = shader
                .libraries
                .get(graphics.0)
                .ok_or(HalError::InvalidArgument)?;
            let fragment_library = shader
                .libraries
                .get(graphics.2)
                .ok_or(HalError::InvalidArgument)?;
            let vertex = vertex_library
                .newFunctionWithName(&NSString::from_str(&graphics.1))
                .ok_or(HalError::InvalidArgument)?;
            let fragment = fragment_library
                .newFunctionWithName(&NSString::from_str(&graphics.3))
                .ok_or(HalError::InvalidArgument)?;
            let descriptor = MTLRenderPipelineDescriptor::new();
            descriptor.setVertexFunction(Some(&vertex));
            descriptor.setFragmentFunction(Some(&fragment));
            let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
            color.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            let depth_enabled = depth_required;
            if depth_enabled {
                descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
            }
            if state.blend == BlendMode::Alpha {
                color.setBlendingEnabled(true);
                color.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                color.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            }
            let pipeline = self
                .device
                .newRenderPipelineStateWithDescriptor_error(&descriptor)
                .map_err(|_| HalError::NativeFailure)?;
            let primitive = match state.topology {
                PrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
                PrimitiveTopology::PointList => MTLPrimitiveType::Point,
                PrimitiveTopology::LineList => MTLPrimitiveType::Line,
                PrimitiveTopology::LineStrip => MTLPrimitiveType::LineStrip,
                PrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
                PrimitiveTopology::TriangleFan => return Err(HalError::Unsupported),
            };
            // SAFETY: the host owns this live CAMetalLayer until surface destruction completes.
            let layer = unsafe { &*(surface.layer as *const CAMetalLayer) };
            layer.setDevice(Some(&self.device));
            layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            let drawable = layer.nextDrawable().ok_or(HalError::NotReady)?;
            let drawable_texture = drawable.texture();
            // Drawable size can diverge from the logical surface after a scale/resize.
            if drawable_texture.width() != extent.0 as usize
                || drawable_texture.height() != extent.1 as usize
            {
                return Err(HalError::NotReady);
            }
            let depth = if depth_enabled {
                let depth_descriptor = unsafe {
                    MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                        MTLPixelFormat::Depth32Float,
                        extent.0 as usize,
                        extent.1 as usize,
                        false,
                    )
                };
                depth_descriptor.setStorageMode(MTLStorageMode::Private);
                depth_descriptor.setUsage(MTLTextureUsage::RenderTarget);
                let texture = self
                    .device
                    .newTextureWithDescriptor(&depth_descriptor)
                    .ok_or(HalError::NativeFailure)?;
                let descriptor = MTLDepthStencilDescriptor::new();
                descriptor.setDepthCompareFunction(MTLCompareFunction::Less);
                descriptor.setDepthWriteEnabled(true);
                let state = self
                    .device
                    .newDepthStencilStateWithDescriptor(&descriptor)
                    .ok_or(HalError::NativeFailure)?;
                Some((texture, state))
            } else {
                None
            };
            let pass = MTLRenderPassDescriptor::renderPassDescriptor();
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(&drawable_texture));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setStoreAction(MTLStoreAction::Store);
            attachment.setClearColor(MTLClearColor {
                red: 0.1,
                green: 0.1,
                blue: 0.1,
                alpha: 1.0,
            });
            if let Some((depth_texture, _)) = depth.as_ref() {
                let depth_attachment = pass.depthAttachment();
                depth_attachment.setTexture(Some(depth_texture));
                depth_attachment.setLoadAction(MTLLoadAction::Clear);
                depth_attachment.setStoreAction(MTLStoreAction::DontCare);
                depth_attachment.setClearDepth(1.0);
            }
            let command = self.queue.commandBuffer().ok_or(HalError::NativeFailure)?;
            let encoder = command
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or(HalError::NativeFailure)?;
            encoder.setRenderPipelineState(&pipeline);
            if let Some((_, depth_state)) = depth.as_ref() {
                encoder.setDepthStencilState(Some(depth_state));
            }
            encoder.setCullMode(match state.cull {
                CullMode::None => MTLCullMode::None,
                CullMode::Front => MTLCullMode::Front,
                CullMode::Back => MTLCullMode::Back,
            });
            encoder.setFrontFacingWinding(match state.front_face {
                FrontFace::CounterClockwise => MTLWinding::CounterClockwise,
                FrontFace::Clockwise => MTLWinding::Clockwise,
            });
            for binding in bindings {
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
            let _argument_buffer = if let Some(heap) = texture_heap {
                let argument_encoder =
                    unsafe { fragment.newArgumentEncoderWithBufferIndex(heap.binding as usize) };
                let buffer = self
                    .device
                    .newBufferWithLength_options(
                        argument_encoder.encodedLength(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(HalError::NativeFailure)?;
                unsafe { argument_encoder.setArgumentBuffer_offset(Some(&buffer), 0) };
                for texture in textures {
                    if texture.binding >= heap.capacity {
                        return Err(HalError::InvalidArgument);
                    }
                    let texture_index = texture.binding as usize * heap.argument_stride as usize
                        + heap.texture_argument_offset as usize;
                    let sampler_index = texture.binding as usize * heap.argument_stride as usize
                        + heap.sampler_argument_offset as usize;
                    unsafe {
                        argument_encoder.setTexture_atIndex(Some(&texture.texture), texture_index);
                        argument_encoder
                            .setSamplerState_atIndex(Some(&texture.sampler), sampler_index);
                        let resource = <ProtocolObject<dyn MTLTexture> as AsRef<
                            ProtocolObject<dyn MTLResource>,
                        >>::as_ref(&*texture.texture);
                        encoder.useResource_usage_stages(
                            resource,
                            MTLResourceUsage::Read,
                            MTLRenderStages::Fragment,
                        );
                    }
                }
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(&buffer),
                        0,
                        heap.binding as usize,
                    )
                };
                Some(buffer)
            } else {
                None
            };
            if let Some(bytes) = core::ptr::NonNull::new(push_constants.as_ptr() as *mut c_void)
                && !push_constants.is_empty()
            {
                unsafe {
                    encoder.setVertexBytes_length_atIndex(bytes, push_constants.len(), 0);
                    encoder.setFragmentBytes_length_atIndex(bytes, push_constants.len(), 0);
                }
            }
            for command_index in 0..draw_count {
                unsafe {
                    encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(primitive, MTLIndexType::UInt32, &index.buffer, 0, &indirect.buffer, command_index as usize * 20)
                };
            }
            encoder.endEncoding();

            let mut capture = if capture_presented {
                // Metal texture-to-buffer copies require 256-byte rows; every product is checked.
                let tight_row = u64::from(extent.0)
                    .checked_mul(4)
                    .ok_or(HalError::InvalidArgument)?;
                let row_stride = tight_row
                    .checked_add(255)
                    .map(|value| value & !255)
                    .ok_or(HalError::InvalidArgument)?;
                let size = row_stride
                    .checked_mul(u64::from(extent.1))
                    .ok_or(HalError::InvalidArgument)?;
                let allocation = self
                    .allocate(
                        AllocationRequest::new(size, 256, MemoryClass::Readback, true, None)
                            .map_err(|_| HalError::InvalidArgument)?,
                    )
                    .map_err(|_| HalError::NativeFailure)?;
                Some((allocation, tight_row, row_stride, size))
            } else {
                None
            };
            if let Some((readback, _, row_stride, size)) = capture.as_ref() {
                let Some(blit) = command.blitCommandEncoder() else {
                    let (readback, _, _, _) = capture.take().expect("capture exists");
                    self.free(readback).map_err(|_| HalError::NativeFailure)?;
                    return Err(HalError::NativeFailure);
                };
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        &drawable_texture,
                        0,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                        MTLSize {
                            width: extent.0 as usize,
                            height: extent.1 as usize,
                            depth: 1,
                        },
                        &readback.buffer,
                        0,
                        *row_stride as usize,
                        *size as usize,
                    );
                    blit.endEncoding();
                }
            }

            let drawable_ref = <ProtocolObject<dyn CAMetalDrawable> as AsRef<
                ProtocolObject<dyn objc2_metal::MTLDrawable>,
            >>::as_ref(&*drawable);
            command.presentDrawable(drawable_ref);
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
                if let Some((readback, _, _, _)) = capture {
                    self.free(readback).map_err(|_| HalError::NativeFailure)?;
                }
                return Err(HalError::NativeFailure);
            }

            if let Some((mut readback, tight_row, row_stride, size)) = capture {
                if self.invalidate(&mut readback, 0, size).is_err() {
                    self.free(readback).map_err(|_| HalError::NativeFailure)?;
                    return Err(HalError::NativeFailure);
                }
                // Checked arithmetic and row slicing keep malformed extents from panicking.
                let packed_size = match tight_row
                    .checked_mul(u64::from(extent.1))
                    .and_then(|value| usize::try_from(value).ok())
                {
                    Some(value) => value,
                    None => {
                        self.free(readback).map_err(|_| HalError::NativeFailure)?;
                        return Err(HalError::InvalidArgument);
                    }
                };
                let source = match self.mapped_slice(&readback) {
                    Ok(source) => source,
                    Err(_) => {
                        self.free(readback).map_err(|_| HalError::NativeFailure)?;
                        return Err(HalError::NativeFailure);
                    }
                };
                let mut packed = Vec::with_capacity(packed_size);
                for row in 0..extent.1 as usize {
                    let start = match row.checked_mul(row_stride as usize) {
                        Some(value) => value,
                        None => {
                            self.free(readback).map_err(|_| HalError::NativeFailure)?;
                            return Err(HalError::InvalidArgument);
                        }
                    };
                    let end = match start.checked_add(tight_row as usize) {
                        Some(value) => value,
                        None => {
                            self.free(readback).map_err(|_| HalError::NativeFailure)?;
                            return Err(HalError::InvalidArgument);
                        }
                    };
                    let row = match source.get(start..end) {
                        Some(row) => row,
                        None => {
                            self.free(readback).map_err(|_| HalError::NativeFailure)?;
                            return Err(HalError::NativeFailure);
                        }
                    };
                    packed.extend_from_slice(row);
                }
                for pixel in packed.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
                self.free(readback).map_err(|_| HalError::NativeFailure)?;
                surface.presented_rgba8 = packed;
            }
            Ok(())
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
