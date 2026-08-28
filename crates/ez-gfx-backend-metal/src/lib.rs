#![deny(unsafe_op_in_unsafe_fn)]

use ez_gfx_core::Backend;

pub const BACKEND: Backend = Backend::Metal;
pub const SUPPORTED_ON_TARGET: bool = cfg!(target_vendor = "apple");

#[cfg(target_vendor = "apple")]
pub mod native {
    use core::ffi::c_void;

    use crate::BACKEND;
    use ez_gfx_core::capability::{
        AdapterCapabilities, AdapterClass, AdapterInfo, CompressionSupport, SemanticProfile,
    };
    use ez_gfx_hal::{
        AllocationError, AllocationRequest, BlendMode, BufferTransfer, CompletionToken, CullMode,
        DynamicPipelineState, FrontFace, HalError, ImageMip, MemoryAllocator, MemoryClass,
        PrimitiveTopology, QueueKind, validate_rgba8_mips,
    };
    use gpu_allocator::{
        MemoryLocation,
        metal::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    };
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLArgumentBuffersTier, MTLBlendFactor, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor,
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLCullMode, MTLDevice, MTLHeap,
        MTLIndexType, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
        MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLSize,
        MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLWinding,
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
        pub binding: u32,
    }

    pub struct NativePipeline {
        state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    struct RetiredAllocation {
        allocation: NativeAllocation,
        completion: CompletionToken,
    }

    pub struct NativeSurface {
        layer: usize,
    }
    impl NativeSurface {
        /// The CAMetalLayer pointer is borrowed; this crate never releases the host's layer.
        pub fn new(layer: *mut c_void) -> Result<Self, HalError> {
            if layer.is_null() {
                return Err(HalError::InvalidArgument);
            }
            Ok(Self {
                layer: layer as usize,
            })
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

    impl NativeContext {
        pub fn create_default() -> Result<Self, HalError> {
            let device = MTLCreateSystemDefaultDevice().ok_or(HalError::Unsupported)?;
            let queue = device.newCommandQueue().ok_or(HalError::NativeFailure)?;
            let tier_two = device.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2;
            let compression = if device.supportsBCTextureCompression() {
                CompressionSupport::BC
            } else {
                CompressionSupport::ASTC
            };
            let capabilities = AdapterCapabilities {
                bindless_sampled_textures: if tier_two { 4096 } else { 0 },
                bindless_storage_resources: if tier_two { 1024 } else { 0 },
                bindless_samplers: u32::try_from(device.maxArgumentBufferSamplerCount())
                    .unwrap_or(u32::MAX),
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
            {
                if !push_constants.is_empty() {
                    unsafe { encoder.setBytes_length_atIndex(bytes, push_constants.len(), 0) };
                }
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
        pub fn draw_indexed_shader(
            &self,
            surface: &NativeSurface,
            shader: &NativeShader,
            graphics: &(usize, String, usize, String),
            state: DynamicPipelineState,
            index: &NativeAllocation,
            indirect: &NativeAllocation,
            draw_count: u32,
            push_constants: &[u8],
            extent: (u32, u32),
            bindings: &[NativeBufferBinding<'_>],
        ) -> Result<(), HalError> {
            if draw_count == 0
                || extent.0 == 0
                || extent.1 == 0
                || push_constants.len() > 128
                || push_constants.len() % 4 != 0
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
            let layer = unsafe { &*(surface.layer as *const CAMetalLayer) };
            layer.setDevice(Some(&self.device));
            let drawable = layer.nextDrawable().ok_or(HalError::NotReady)?;
            let pass = MTLRenderPassDescriptor::renderPassDescriptor();
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(&drawable.texture()));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setStoreAction(MTLStoreAction::Store);
            attachment.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            });
            let command = self.queue.commandBuffer().ok_or(HalError::NativeFailure)?;
            let encoder = command
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or(HalError::NativeFailure)?;
            encoder.setRenderPipelineState(&pipeline);
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
            if let Some(bytes) = core::ptr::NonNull::new(push_constants.as_ptr() as *mut c_void) {
                if !push_constants.is_empty() {
                    unsafe {
                        encoder.setVertexBytes_length_atIndex(bytes, push_constants.len(), 0);
                        encoder.setFragmentBytes_length_atIndex(bytes, push_constants.len(), 0);
                    }
                }
            }
            for command_index in 0..draw_count {
                unsafe {
                    encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(primitive, MTLIndexType::UInt32, &index.buffer, 0, &indirect.buffer, command_index as usize * 20)
                };
            }
            encoder.endEncoding();
            let drawable_ref = <ProtocolObject<dyn CAMetalDrawable> as AsRef<
                ProtocolObject<dyn objc2_metal::MTLDrawable>,
            >>::as_ref(&*drawable);
            command.presentDrawable(drawable_ref);
            command.commit();
            command.waitUntilCompleted();
            Ok(())
        }

        pub fn create_texture_rgba8(
            &mut self,
            mips: &[ImageMip<'_>],
            binding: u32,
        ) -> Result<(NativeTexture, Vec<CompletionToken>), AllocationError> {
            validate_rgba8_mips(mips).map_err(|_| AllocationError::ZeroSize)?;
            if binding >= 4096 {
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
            let allocation_desc =
                AllocationCreateDesc::texture(&self.device, "ez-gfx-texture", &desc);
            let allocation = self
                .allocator
                .as_mut()
                .ok_or(AllocationError::NativeFailure)?
                .allocate(&allocation_desc)
                .map_err(map_allocator)?;
            let texture = unsafe {
                allocation
                    .heap()
                    .newTextureWithDescriptor_offset(&desc, allocation.offset() as usize)
            }
            .ok_or(AllocationError::OutOfMemory)?;
            let mut completions = Vec::with_capacity(mips.len());
            for (level, mip) in mips.iter().enumerate() {
                let size = mip.bytes.len() as u64;
                let mut upload = self.allocate(
                    AllocationRequest::new(size, 4, MemoryClass::Upload, true, None)
                        .map_err(|_| AllocationError::ZeroSize)?,
                )?;
                self.mapped_slice_mut(&mut upload)?[..mip.bytes.len()].copy_from_slice(mip.bytes);
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
                self.completed_transfer_value = value;
                completions.push(
                    CompletionToken::new(QueueKind::Transfer, value)
                        .map_err(|_| AllocationError::NativeFailure)?,
                );
                self.free(upload)?;
            }
            Ok((
                NativeTexture {
                    texture,
                    allocation,
                    binding,
                },
                completions,
            ))
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
            self.invalidate(&mut readback, 0, size)?;
            let pixels = self.mapped_slice(&readback)?[..size as usize].to_vec();
            self.free(readback)?;
            Ok(pixels)
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
